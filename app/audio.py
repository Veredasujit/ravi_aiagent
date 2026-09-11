"""
audio.py — small, dependency-light audio helpers.

Everything on the wire is signed 16-bit little-endian PCM, mono. Teler gives
us 8 kHz and wants 8 kHz back, and we configure Sarvam to emit 8 kHz too, so
in the happy path *no resampling happens at all* — that is where a lot of
voice agents quietly lose 20-40 ms and a bit of clarity.

The resampler here only runs for the OpenAI TTS fallback (24 kHz -> 8 kHz).

Note: Python 3.13 removed `audioop`, so mu-law and resampling are done with
numpy instead.
"""

from __future__ import annotations

import math

import numpy as np

BYTES_PER_SAMPLE = 2


# ─────────────────────────── framing ────────────────────────────────────────
def frame_bytes(sample_rate: int, ms: int) -> int:
    """Bytes in `ms` milliseconds of mono s16 audio at `sample_rate`."""
    return int(sample_rate * ms / 1000) * BYTES_PER_SAMPLE


def pcm_to_float(pcm: bytes) -> np.ndarray:
    """s16le bytes -> float32 array in [-1, 1]."""
    if not pcm:
        return np.zeros(0, dtype=np.float32)
    a = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    return a / 32768.0


def float_to_pcm(x: np.ndarray) -> bytes:
    """float array in [-1, 1] -> s16le bytes (clipped, not wrapped)."""
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype("<i2").tobytes()


def duration_ms(pcm: bytes, sample_rate: int) -> float:
    return len(pcm) / BYTES_PER_SAMPLE / sample_rate * 1000.0


def silence(sample_rate: int, ms: int) -> bytes:
    return b"\x00" * frame_bytes(sample_rate, ms)


# ─────────────────────────── level metering ─────────────────────────────────
def dbfs(pcm: bytes) -> float:
    """RMS level of a frame in dBFS. Returns -100.0 for digital silence."""
    x = pcm_to_float(pcm)
    if x.size == 0:
        return -100.0
    rms = float(np.sqrt(np.mean(np.square(x))))
    if rms <= 1e-7:
        return -100.0
    return 20.0 * math.log10(rms)


# ─────────────────────────── resampling ─────────────────────────────────────
def resample(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """
    Rate-convert mono s16 PCM.

    Downsampling runs a short FIR low-pass first so we don't fold high
    frequencies back into the voice band as aliasing hiss — that "tinny,
    slightly robotic" sound you get from naive decimation.
    """
    if src_rate == dst_rate or not pcm:
        return pcm

    x = pcm_to_float(pcm)

    if dst_rate < src_rate:
        ratio = dst_rate / src_rate
        # Windowed-sinc low-pass at 0.45 * dst_rate, 31 taps: cheap and enough
        # for telephony.
        taps = 31
        cutoff = 0.45 * ratio  # normalised to src Nyquist
        n = np.arange(taps) - (taps - 1) / 2.0
        h = np.sinc(2 * cutoff * n) * np.hamming(taps)
        h /= np.sum(h)
        x = np.convolve(x, h, mode="same")

    n_out = int(round(len(x) * dst_rate / src_rate))
    if n_out <= 0:
        return b""
    src_idx = np.linspace(0.0, len(x) - 1, num=n_out, dtype=np.float64)
    y = np.interp(src_idx, np.arange(len(x), dtype=np.float64), x)
    return float_to_pcm(y.astype(np.float32))


# ─────────────────────────── containers / codecs ────────────────────────────
def strip_wav_header(data: bytes) -> bytes:
    """
    Some TTS backends prepend a RIFF/WAVE header to the *first* chunk even when
    you ask for raw linear16. Feeding that header to the phone line produces an
    audible click or a burst of static, so we walk the chunks and return only
    the `data` payload.
    """
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return data
    pos = 12
    while pos + 8 <= len(data):
        cid = data[pos : pos + 4]
        size = int.from_bytes(data[pos + 4 : pos + 8], "little")
        pos += 8
        if cid == b"data":
            return data[pos : pos + size] if size else data[pos:]
        pos += size + (size & 1)
    return b""


_MULAW_BIAS = 0x84  # 132, applied in the 16-bit domain on decode
_MULAW_CLIP = 8159  # clip in the 14-bit domain on encode
_SEG_UEND = np.array(
    [0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF], dtype=np.int32
)


def pcm_to_mulaw(pcm: bytes) -> bytes:
    """
    s16le PCM -> G.711 mu-law, following the Sun reference implementation.

    Only needed if you switch Teler to a PCMU codec; the default L16 path never
    touches this.
    """
    x = np.frombuffer(pcm, dtype="<i2").astype(np.int32) >> 2  # 14-bit domain
    mask = np.where(x < 0, 0x7F, 0xFF).astype(np.int32)
    x = np.abs(x)
    np.clip(x, 0, _MULAW_CLIP, out=x)
    x = x + (_MULAW_BIAS >> 2)  # += 33

    seg = np.searchsorted(_SEG_UEND, x, side="left").astype(np.int32)
    np.clip(seg, 0, 7, out=seg)

    mantissa = (x >> (seg + 1)) & 0x0F
    uval = ((seg << 4) | mantissa) ^ mask
    return (uval & 0xFF).astype(np.uint8).tobytes()


def mulaw_to_pcm(mu: bytes) -> bytes:
    """G.711 mu-law -> s16le PCM."""
    u = np.frombuffer(mu, dtype=np.uint8).astype(np.int32)
    u = ~u & 0xFF
    sign = u & 0x80
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    magnitude = ((mantissa << 3) + _MULAW_BIAS) << exponent
    magnitude -= _MULAW_BIAS
    out = np.where(sign != 0, -magnitude, magnitude)
    return np.clip(out, -32768, 32767).astype("<i2").tobytes()


# ─────────────────────────── endianness ─────────────────────────────────────
def byteswap16(pcm: bytes) -> bytes:
    """Swap every 16-bit sample between big- and little-endian."""
    n = len(pcm) - (len(pcm) % 2)
    if n <= 0:
        return b""
    return np.frombuffer(pcm[:n], dtype="<i2").byteswap().tobytes()


def roughness(pcm: bytes) -> float:
    """
    Mean sample-to-sample change, normalised by amplitude.

    Speech at 8 kHz is smooth: consecutive samples are close together, so this
    lands around 0.1-0.4. Read a big-endian stream as little-endian and the
    high and low bytes trade places, turning every sample into near-noise and
    pushing this above 1.0. The gap between the two is roughly 8x, which is
    plenty to decide automatically instead of guessing.
    """
    a = np.frombuffer(pcm[: len(pcm) - (len(pcm) % 2)], dtype="<i2").astype(np.float64)
    if a.size < 2:
        return 0.0
    denom = float(np.mean(np.abs(a)))
    if denom < 1.0:  # silence tells us nothing
        return 0.0
    return float(np.mean(np.abs(np.diff(a))) / denom)


class EndianDetector:
    """
    Works out whether the carrier is sending big- or little-endian L16.

    RFC 2586 says L16 is network byte order (big-endian) and FreJun documents
    the stream as "L16/8000Hz", but carriers vary and getting this wrong makes
    the call sound like static, so we measure rather than assume.

    Feed it inbound audio; it returns audio normalised to little-endian for the
    rest of the pipeline. Once decided, the answer is fixed for the call.
    """

    def __init__(self, mode: str = "auto", min_bytes: int = 8000) -> None:
        self.mode = (mode or "auto").lower()
        self.min_bytes = min_bytes
        self._buf = bytearray()
        self.big_endian = self.mode == "big"
        self.decided = self.mode in ("big", "little")
        self.scores: tuple[float, float] = (0.0, 0.0)

    def feed(self, pcm: bytes) -> bytes:
        """Return `pcm` normalised to little-endian, deciding on the way."""
        if self.decided:
            return byteswap16(pcm) if self.big_endian else pcm

        # Only accumulate frames with real signal in them.
        if dbfs(pcm) > -45.0:
            self._buf.extend(pcm)

        if len(self._buf) >= self.min_bytes:
            self._decide()

        # Until we know, pass through unchanged.
        return pcm

    def _decide(self) -> None:
        raw = bytes(self._buf)
        as_little = roughness(raw)
        as_big = roughness(byteswap16(raw))
        self.scores = (round(as_little, 3), round(as_big, 3))
        # Lower roughness wins: that reading produced smooth, speech-like audio.
        self.big_endian = as_big < as_little
        self.decided = True

    def force(self, big_endian: bool) -> None:
        self.big_endian = big_endian
        self.decided = True
