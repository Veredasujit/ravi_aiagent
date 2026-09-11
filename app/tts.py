"""
tts.py — speech synthesis, streamed.

Primary: Sarvam `bulbul:v3` over WebSocket. We ask for `linear16` @ 8000 Hz,
which is exactly what Teler wants, so audio goes Sarvam -> phone line
untouched: no MP3 decode, no resample, no extra buffering.

Not every account/model combination accepts every codec, and a rejected config
shows up as an async 422 *after* the socket is already open — which looks like
"TTS is broken" when it is really "that one field was refused". So instead of
trusting the config, we PROVE it at connect time: send the config, synthesise
a one-word probe, and if the server refuses, step down a codec ladder until
something works. Every rung decodes to 8 kHz PCM with no new dependencies:

    linear16 @ 8000  -> raw PCM (strip RIFF header if present)
    mulaw    @ 8000  -> G.711 decode
    wav      @ 8000  -> strip RIFF header
    linear16 @ 16000 -> raw PCM, resampled down

Whatever we land on is logged once, so you always know what the call is using.

Fallback vendor: OpenAI TTS over HTTP (`response_format=pcm`, 24 kHz),
resampled to 8 kHz, used only if Sarvam is unusable entirely.

Barge-in: Sarvam's TTS socket has no server-side cancel message. The documented
approach is to stop playback locally and close the socket, so that is what we
do, then reconnect in the background.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import AsyncGenerator, Optional
from urllib.parse import urlencode

import aiohttp
import websockets

from . import config
from .audio import mulaw_to_pcm, resample, strip_wav_header
from .logging_setup import CallLog, error, info, log, warn

# Codecs we can decode to 8 kHz PCM without an audio library. mp3/opus/aac are
# deliberately absent — decoding them adds latency and a dependency for no
# benefit on an 8 kHz phone line.
CODEC_LADDER: list[tuple[str, int]] = [
    ("linear16", 8000),
    ("mulaw", 8000),
    ("wav", 8000),
    ("linear16", 16000),
]


class TTSAborted(Exception):
    """Raised inside a synthesis stream when the caller barged in."""


def _ladder_for(codec: str, rate: int) -> list[tuple[str, int]]:
    """Preferred combination first, then the rest of the ladder, deduplicated."""
    out = [(codec, rate)]
    for rung in CODEC_LADDER:
        if rung not in out:
            out.append(rung)
    return out


class SarvamTTS:
    """Persistent Sarvam WebSocket. One instance per call."""

    def __init__(self, call_log: CallLog) -> None:
        self.call_log = call_log
        self._ws = None
        self._lock = asyncio.Lock()
        self._connect_lock = asyncio.Lock()
        self._abort = asyncio.Event()
        self._closed = False
        self._first_chunk_seen = False
        self.connected = False

        self._ladder = _ladder_for(config.SARVAM_CODEC, config.SARVAM_SAMPLE_RATE)
        self._rung = 0
        self.validated = False

    # ── negotiated format ───────────────────────────────────────────────────
    @property
    def codec(self) -> str:
        return self._ladder[self._rung][0]

    @property
    def sample_rate(self) -> int:
        return self._ladder[self._rung][1]

    # ── connection ──────────────────────────────────────────────────────────
    def _url(self) -> str:
        """
        Exactly the URL shape the SDK builds, because that one is proven.

        Two details that cost us several rounds of debugging:

        1. `urlencode` escapes the colon, so the model arrives as
           `bulbul%3Av3`. A bare colon in a query string is legal per RFC 3986
           but is evidently not parsed the same way here.

        2. `send_completion_event` is NOT sent. The SDK strips None query
           params before encoding, so a plain `connect(model=...)` omits it —
           and that is the call we proved works against this account. The
           server defaults it to true anyway, and we no longer depend on the
           event to detect end-of-utterance (see the idle timeout below).
        """
        params = {"model": config.SARVAM_MODEL}
        if config.SARVAM_COMPLETION_EVENT:
            params["send_completion_event"] = "true"
        return f"{config.SARVAM_WS_URL}?{urlencode(params)}"

    def _config_payload(self) -> dict:
        """
        Byte-for-byte the same SHAPE the official SDK sends, because that shape
        is proven to work against this endpoint.

        Two things here are not obvious and both were causing 422s:

        1. `model` is null. The model goes in the URL query string only. The
           SDK's `configure()` never sets the field, and since `_send_model()`
           calls `.dict()` without `exclude_none`, it ships as an explicit
           null. Sending the actual model string here is rejected.

        2. Every optional field is present, including the null ones. The SDK
           never omits a key, so we don't either — omitting them appears to be
           what produced "Input parameters has to be a valid dictionary".

        The only values we deviate on are codec, sample rate and buffer size,
        which is the whole point of the exercise.
        """
        data: dict = {
            "model": None,  # goes in the query string, NOT in here
            "language_code": config.SARVAM_LANGUAGE,
            # Speaker names are case-sensitive and must be lowercase.
            "speaker": config.SARVAM_VOICE.lower(),
            "pitch": 0.0,
            "pace": float(config.SARVAM_SPEED),
            "loudness": 1.0,
            "temperature": None,
            "speech_sample_rate": int(self.sample_rate),
            "enable_preprocessing": False,
            "output_audio_codec": self.codec,
            "output_audio_bitrate": "128k",
            "dict_id": None,
            "min_buffer_size": int(config.SARVAM_MIN_BUFFER),
            "max_chunk_length": int(config.SARVAM_MAX_CHUNK_LEN),
        }
        return {"type": "config", "data": data}

    async def connect(self) -> bool:
        """Open the socket and send the config frame. Idempotent."""
        # The background reconnect task and the caller can both land here at
        # once; without this lock they each open a socket and we leak one per
        # turn (visible as two "socket ready" lines milliseconds apart).
        async with self._connect_lock:
            if self._closed:
                return False
            if self._ws is not None:
                return True
            return await self._do_connect()

    async def _do_connect(self) -> bool:
        if not config.SARVAM_API_KEY:
            error("tts", "SARVAM_API_KEY missing")
            return False

        t0 = time.monotonic()
        headers = {"api-subscription-key": config.SARVAM_API_KEY}
        try:
            try:
                self._ws = await asyncio.wait_for(
                    websockets.connect(
                        self._url(),
                        additional_headers=headers,
                        ping_interval=20,
                        ping_timeout=20,
                        close_timeout=2,
                        max_size=8 * 1024 * 1024,
                    ),
                    timeout=config.SARVAM_CONNECT_TIMEOUT_S,
                )
            except TypeError:
                self._ws = await asyncio.wait_for(
                    websockets.connect(
                        self._url(),
                        extra_headers=headers,
                        ping_interval=20,
                        ping_timeout=20,
                        close_timeout=2,
                        max_size=8 * 1024 * 1024,
                    ),
                    timeout=config.SARVAM_CONNECT_TIMEOUT_S,
                )
        except Exception as exc:
            error("tts", f"sarvam connect failed: {exc}")
            self.call_log.event("tts_connect_failed", vendor="sarvam", error=str(exc))
            self._ws = None
            self.connected = False
            return False

        payload = self._config_payload()
        if config.SARVAM_WIRE_LOG:
            log("tts", f"-> config {json.dumps(payload, ensure_ascii=False)}")
        try:
            await self._ws.send(json.dumps(payload))
        except Exception as exc:
            error("tts", f"sarvam config send failed: {exc}")
            self._ws = None
            return False

        # Give the server a beat to apply the config before any text arrives.
        await asyncio.sleep(config.SARVAM_CONFIG_SETTLE_S)

        ms = round((time.monotonic() - t0) * 1000, 1)
        self.connected = True
        self._first_chunk_seen = False
        info(
            "tts",
            f"sarvam socket ready in {ms} ms "
            f"({config.SARVAM_MODEL}/{config.SARVAM_VOICE}/"
            f"{self.codec}@{self.sample_rate})",
        )
        self.call_log.event(
            "tts_connected", vendor="sarvam", ms=ms,
            codec=self.codec, sample_rate=self.sample_rate,
        )
        return True

    # ── connect-time validation ─────────────────────────────────────────────
    async def prewarm(self) -> bool:
        """
        Open the socket AND prove the codec works, before the greeting.

        Walks the codec ladder until a probe synthesis returns audio. Returns
        False if every rung fails, so the engine can fall back to OpenAI TTS
        rather than leaving the caller in silence.
        """
        if self.validated:
            return True

        probe = "जी।"
        while self._rung < len(self._ladder):
            codec, rate = self._ladder[self._rung]
            if not await self.connect():
                return False
            try:
                got = 0
                async for pcm in self._stream_once(probe, probe=True):
                    got += len(pcm)
                if got > 0:
                    self.validated = True
                    info("tts", f"codec confirmed: {codec}@{rate} ({got} B probe)")
                    self.call_log.event(
                        "tts_codec_confirmed", codec=codec, sample_rate=rate,
                        probe_bytes=got,
                    )
                    return True
                warn("tts", f"{codec}@{rate} connected but returned no audio")
            except Exception as exc:
                warn("tts", f"{codec}@{rate} rejected: {exc}")
                self.call_log.event(
                    "tts_codec_rejected", codec=codec, sample_rate=rate,
                    error=str(exc),
                )

            await self._drop(reconnect=False)
            self._rung += 1
            if self._rung < len(self._ladder):
                nxt = self._ladder[self._rung]
                info("tts", f"stepping down to {nxt[0]}@{nxt[1]}")

        error(
            "tts",
            "every Sarvam codec was rejected — check that SARVAM_VOICE is a "
            "valid lowercase bulbul:v3 speaker and the key has TTS access",
        )
        self._rung = 0
        return False

    # ── decoding ────────────────────────────────────────────────────────────
    def _decode(self, raw: bytes) -> bytes:
        """Vendor chunk -> 8 kHz s16 PCM ready for the wire."""
        if self.codec == "mulaw":
            pcm = mulaw_to_pcm(raw)
        else:  # linear16 or wav
            pcm = raw
            if not self._first_chunk_seen:
                # linear16 sometimes arrives with a RIFF header on the very
                # first chunk; left in, it is an audible click.
                pcm = strip_wav_header(pcm)
        self._first_chunk_seen = True

        if self.sample_rate != config.TELER_SAMPLE_RATE:
            pcm = resample(pcm, self.sample_rate, config.TELER_SAMPLE_RATE)
        return pcm

    # ── synthesis ───────────────────────────────────────────────────────────
    async def stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Yield 8 kHz s16 PCM for `text`. Raises TTSAborted on barge-in."""
        text = (text or "").strip()
        if not text:
            return
        async with self._lock:
            async for pcm in self._stream_once(text):
                yield pcm

    async def _stream_once(
        self, text: str, probe: bool = False
    ) -> AsyncGenerator[bytes, None]:
        """One request/response cycle. Caller holds the lock (or is prewarm)."""
        self._abort.clear()
        if not await self.connect():
            raise RuntimeError("sarvam not connected")

        ws = self._ws
        assert ws is not None
        t_send = time.monotonic()

        text_msg = {"type": "text", "data": {"text": text}}
        if not probe:
            self.call_log.event(
                "tts_request", vendor="sarvam", chars=len(text), text=text,
                codec=self.codec,
            )
            log("tts", f"synthesising ({len(text)} chars): {text}")
        if config.SARVAM_WIRE_LOG:
            log("tts", f"-> {json.dumps(text_msg, ensure_ascii=False)}")

        try:
            await ws.send(json.dumps(text_msg))
            await ws.send(json.dumps({"type": "flush"}))
        except Exception as exc:
            error("tts", f"sarvam send failed: {exc}")
            await self._drop()
            raise

        first = True
        total = 0
        while True:
            if self._abort.is_set():
                log("tts", "aborted mid-stream (barge-in)")
                await self._drop()
                raise TTSAborted()

            # Before any audio arrives we allow a generous wait. Once audio is
            # flowing, a short gap means the utterance is finished — this is
            # what the Sarvam docs' own example does, and it means we no longer
            # need the completion event to know when to stop reading.
            timeout = (
                config.SARVAM_FIRST_CHUNK_TIMEOUT_S
                if total == 0
                else config.SARVAM_IDLE_END_S
            )
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            except asyncio.TimeoutError:
                if total > 0:
                    # We never saw the completion event, so we cannot know
                    # whether Sarvam has more to send. Anything still in flight
                    # would be read as the start of the NEXT reply, splicing
                    # sentences together. Throw the socket away instead — a
                    # reconnect costs ~600 ms in the background and is the only
                    # way to guarantee the next reply starts clean.
                    if not probe:
                        dur = total / 2 / config.TELER_SAMPLE_RATE
                        self.call_log.event(
                            "tts_done", vendor="sarvam", bytes=total,
                            audio_seconds=round(dur, 2), ended="idle",
                        )
                        warn(
                            "tts",
                            f"no completion event after {timeout}s — "
                            f"{total} bytes ({dur:.2f}s); dropping the socket so "
                            f"leftovers cannot contaminate the next reply",
                        )
                    await self._drop()
                    return
                warn("tts", "sarvam timed out before any audio")
                self.call_log.event("tts_timeout", vendor="sarvam")
                await self._drop()
                raise
            except websockets.exceptions.ConnectionClosed as exc:
                if self._abort.is_set():
                    raise TTSAborted()
                warn("tts", f"sarvam socket closed mid-stream: {exc.code}")
                await self._drop()
                raise
            except asyncio.CancelledError:
                await self._drop()
                raise

            try:
                msg = json.loads(raw)
            except (TypeError, ValueError):
                continue

            mtype = msg.get("type")

            if mtype == "audio":
                b64 = (msg.get("data") or {}).get("audio") or ""
                if not b64:
                    continue
                pcm = self._decode(base64.b64decode(b64))
                if not pcm:
                    continue
                if first and not probe:
                    ttfb = round((time.monotonic() - t_send) * 1000, 1)
                    info("tts", f"first audio chunk in {ttfb} ms")
                    self.call_log.event("tts_ttfb", vendor="sarvam", ms=ttfb)
                    first = False
                total += len(pcm)
                yield pcm

            elif mtype == "event":
                ev = (msg.get("data") or {}).get("event_type")
                if config.SARVAM_WIRE_LOG:
                    log("tts", f"<- event {ev}")
                if ev == "final":
                    # The authoritative end of synthesis. Everything for this
                    # request has arrived; the socket is clean for the next one.
                    if not probe:
                        dur = total / 2 / config.TELER_SAMPLE_RATE
                        self.call_log.event(
                            "tts_done", vendor="sarvam", bytes=total,
                            audio_seconds=round(dur, 2),
                        )
                        log("tts", f"done (final): {total} bytes ({dur:.2f}s)")
                    return

            elif mtype == "error" or msg.get("error"):
                detail = msg.get("data") or msg
                # Print exactly what we sent. This is the line that tells you
                # which field the server actually objected to.
                error("tts", f"sarvam error: {detail}")
                error(
                    "tts",
                    "  config sent: "
                    f"{json.dumps(self._config_payload(), ensure_ascii=False)}",
                )
                error(
                    "tts",
                    f"  text sent:   {json.dumps(text_msg, ensure_ascii=False)}",
                )
                self.call_log.event(
                    "tts_error", vendor="sarvam", detail=detail,
                    config_sent=self._config_payload(),
                )
                await self._drop()
                raise RuntimeError(f"sarvam {detail}")

    def abort(self) -> None:
        """Barge-in. Stops the current stream; socket is rebuilt afterwards."""
        self._abort.set()

    async def _drop(self, reconnect: bool = True) -> None:
        """Close the socket so the server stops generating, then reconnect."""
        ws, self._ws = self._ws, None
        self.connected = False
        self._first_chunk_seen = False
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass
        if reconnect and not self._closed:
            asyncio.create_task(self._reconnect_soon())

    async def _reconnect_soon(self) -> None:
        await asyncio.sleep(0.05)
        if not self._closed:
            await self.connect()

    async def close(self) -> None:
        self._closed = True
        self._abort.set()
        await self._drop(reconnect=False)


class OpenAITTS:
    """HTTP fallback. Streams 24 kHz PCM and downsamples to 8 kHz."""

    def __init__(self, call_log: CallLog) -> None:
        self.call_log = call_log
        self._session: Optional[aiohttp.ClientSession] = None
        self._abort = asyncio.Event()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30, sock_connect=5)
            )
        return self._session

    async def stream(self, text: str) -> AsyncGenerator[bytes, None]:
        text = (text or "").strip()
        if not text or not config.OPENAI_API_KEY:
            return
        self._abort.clear()
        session = await self._get_session()
        t0 = time.monotonic()
        payload = {
            "model": config.OPENAI_TTS_MODEL,
            "voice": config.OPENAI_TTS_VOICE,
            "input": text,
            "response_format": "pcm",
        }
        headers = {"Authorization": f"Bearer {config.OPENAI_API_KEY}"}
        info("tts", f"fallback to OpenAI TTS ({len(text)} chars)")
        self.call_log.event("tts_request", vendor="openai", chars=len(text))

        carry = b""
        first = True
        async with session.post(
            f"{config.OPENAI_BASE_URL}/audio/speech", json=payload, headers=headers
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                error("tts", f"openai tts {resp.status}: {body[:300]}")
                self.call_log.event("tts_error", vendor="openai", status=resp.status)
                return
            async for chunk in resp.content.iter_chunked(4096):
                if self._abort.is_set():
                    raise TTSAborted()
                buf = carry + chunk
                usable = len(buf) - (len(buf) % 2)  # keep s16 alignment
                carry = buf[usable:]
                buf = buf[:usable]
                if not buf:
                    continue
                pcm8 = resample(
                    buf, config.OPENAI_TTS_SAMPLE_RATE, config.TELER_SAMPLE_RATE
                )
                if first:
                    ttfb = round((time.monotonic() - t0) * 1000, 1)
                    self.call_log.event("tts_ttfb", vendor="openai", ms=ttfb)
                    info("tts", f"openai first chunk in {ttfb} ms")
                    first = False
                if pcm8:
                    yield pcm8

    def abort(self) -> None:
        self._abort.set()

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()


class TTSEngine:
    """
    Primary vendor (ElevenLabs or Sarvam) with OpenAI as the fallback.

    Which primary is used comes from TTS_PROVIDER. Both clients expose the same
    interface — prewarm/stream/abort/close, yielding 8 kHz PCM — so the rest of
    the pipeline neither knows nor cares which one is running.
    """

    def __init__(self, call_log: CallLog) -> None:
        self.call_log = call_log
        if config.TTS_PROVIDER == "sarvam":
            self.primary = SarvamTTS(call_log)
            self.vendor = "sarvam"
        else:
            from .tts_elevenlabs import ElevenLabsTTS

            self.primary = ElevenLabsTTS(call_log)
            self.vendor = "elevenlabs"
        self.openai = OpenAITTS(call_log) if config.TTS_FALLBACK_ENABLED else None
        self._failures = 0

    # Kept so existing code that reached for `.sarvam` still works.
    @property
    def sarvam(self):
        return self.primary

    async def prewarm(self) -> None:
        if await self.primary.prewarm():
            self.call_log.event(
                "tts_vendor", vendor=self.vendor,
                codec=self.primary.codec, sample_rate=self.primary.sample_rate,
            )
            return
        self._failures = 99  # skip the primary for this whole call
        warn(
            "tts",
            f"{self.vendor} unusable for this call — using OpenAI TTS throughout",
        )
        self.vendor = "openai"
        self.call_log.event(
            "tts_vendor", vendor="openai", reason="primary_failed"
        )

    async def stream(self, text: str) -> AsyncGenerator[bytes, None]:
        if self._failures < 3:
            try:
                async for pcm in self.primary.stream(text):
                    yield pcm
                self._failures = 0
                return
            except TTSAborted:
                raise
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._failures += 1
                warn(
                    "tts",
                    f"{self.vendor} failed ({self._failures}/3): {exc} — "
                    f"{'falling back to OpenAI' if self.openai else 'no fallback'}",
                )

        if self.openai is not None:
            self.vendor = "openai"
            async for pcm in self.openai.stream(text):
                yield pcm

    def abort(self) -> None:
        self.primary.abort()
        if self.openai:
            self.openai.abort()

    async def close(self) -> None:
        await self.primary.close()
        if self.openai:
            await self.openai.close()
