"""
vad.py — Silero VAD, wrapped so it answers the only two questions that matter
on a phone call:

    "did a *person* just start talking?"   -> SPEECH_START
    "have they finished?"                  -> SPEECH_END

Why not just threshold Silero's raw probability? Because on an Indian PSTN
line you get fan noise, TV in the background, a scooter horn, and line hiss,
and all of those can spike a single frame over 0.5. Two guards fix it:

  1. Hangover / minimum-duration counting: N *consecutive* speech frames before
     we declare onset, M consecutive silent frames before we declare offset.
  2. An adaptive noise-floor energy gate: a frame only counts as speech if it
     is meaningfully louder than the running background level. Steady hum
     raises the floor, so it stops triggering; a human voice jumps well above
     it and still gets through.

While the agent is speaking we raise both the probability threshold and the
minimum-duration requirement, because the far end's audio leaks back down the
line and we must not interrupt ourselves.

Silero v5 requires exactly 256 samples per call at 8 kHz. We hold a byte
buffer and only ever hand it perfectly sized frames.
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, Optional

import numpy as np

from . import config
from .audio import dbfs
from .logging_setup import error, info, log

FRAME_SAMPLES = config.VAD_FRAME_SAMPLES  # 256 @ 8 kHz
FRAME_BYTES = FRAME_SAMPLES * 2
FRAME_MS = FRAME_SAMPLES / config.VAD_SAMPLE_RATE * 1000.0  # 32.0 ms


class VadEvent(Enum):
    NONE = 0
    SPEECH_START = 1
    SPEECH_END = 2


_model = None
_model_failed = False


def _get_model():
    """Load Silero once per process. ONNX runtime keeps it off the GPU and fast."""
    global _model, _model_failed
    if _model is not None or _model_failed:
        return _model
    try:
        from silero_vad import load_silero_vad

        _model = load_silero_vad(onnx=True)
        info("vad", "silero VAD loaded (onnx)")
    except Exception as exc:
        _model_failed = True
        error(
            "vad",
            f"could not load silero VAD ({exc}); falling back to the energy gate "
            f"only. Run: pip install silero-vad onnxruntime",
        )
    return _model


class SileroVAD:
    """
    Feed it raw 8 kHz s16 PCM in any chunk size; it emits events.

    Usage:
        vad = SileroVAD()
        for ev, prob in vad.feed(pcm_bytes, bot_speaking=False):
            ...
    """

    def __init__(self, on_event: Optional[Callable[[VadEvent, float], None]] = None):
        self._buf = bytearray()
        self._model = _get_model() if config.VAD_ENABLED else None
        self._torch = None
        if self._model is not None:
            try:
                import torch

                self._torch = torch
            except Exception as exc:  # pragma: no cover
                error("vad", f"torch unavailable ({exc}); energy gate only")
                self._model = None

        self.speaking = False
        self._speech_frames = 0
        self._silence_frames = 0
        self._noise_floor_db = config.VAD_NOISE_FLOOR_INIT_DB
        self.on_event = on_event

        # Diagnostics you can read at the end of a call.
        self.frames_seen = 0
        self.frames_speech = 0
        self.frames_rejected_by_energy = 0
        self.last_prob = 0.0
        self.last_db = -100.0

    # ── internals ───────────────────────────────────────────────────────────
    def _probability(self, frame: bytes) -> float:
        if self._model is None or self._torch is None:
            return 0.0
        try:
            x = np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768.0
            t = self._torch.from_numpy(x.copy())
            with self._torch.no_grad():
                return float(self._model(t, config.VAD_SAMPLE_RATE).item())
        except Exception as exc:
            error("vad", f"inference failed: {exc}")
            return 0.0

    def _energy_ok(self, level_db: float) -> bool:
        """True if this frame is loud enough, relative to the running floor."""
        if not config.VAD_USE_ENERGY_GATE:
            return True
        return level_db >= self._noise_floor_db + config.VAD_ENERGY_MARGIN_DB

    def _update_noise_floor(self, level_db: float, is_speech: bool) -> None:
        # Only adapt on non-speech frames, otherwise a long sentence would drag
        # the floor up and deafen us.
        if is_speech:
            return
        a = config.VAD_NOISE_FLOOR_ALPHA
        if level_db <= -99.0:
            return
        self._noise_floor_db = (1 - a) * self._noise_floor_db + a * level_db

    def reset(self) -> None:
        """Called at turn boundaries so counters don't leak across turns."""
        self._speech_frames = 0
        self._silence_frames = 0
        self.speaking = False

    # ── public API ──────────────────────────────────────────────────────────
    def feed(self, pcm: bytes, bot_speaking: bool = False):
        """
        Push audio in, get `(VadEvent, probability)` tuples out.

        `bot_speaking` tightens the thresholds so our own voice, echoed back by
        the carrier, can't trigger a false barge-in.
        """
        self._buf.extend(pcm)

        threshold = (
            config.VAD_THRESHOLD_WHILE_SPEAKING
            if bot_speaking
            else config.VAD_THRESHOLD
        )
        min_speech_ms = (
            config.VAD_MIN_SPEECH_MS_WHILE_SPEAKING
            if bot_speaking
            else config.VAD_MIN_SPEECH_MS
        )
        need_speech = max(1, int(round(min_speech_ms / FRAME_MS)))
        need_silence = max(1, int(round(config.VAD_MIN_SILENCE_MS / FRAME_MS)))

        while len(self._buf) >= FRAME_BYTES:
            frame = bytes(self._buf[:FRAME_BYTES])
            del self._buf[:FRAME_BYTES]
            self.frames_seen += 1

            level_db = dbfs(frame)
            prob = self._probability(frame)
            self.last_prob = prob
            self.last_db = level_db

            loud_enough = self._energy_ok(level_db)
            is_speech = prob >= threshold and loud_enough
            if prob >= threshold and not loud_enough:
                self.frames_rejected_by_energy += 1

            self._update_noise_floor(level_db, prob >= threshold)
            if is_speech:
                self.frames_speech += 1

            if config.LOG_AUDIO_FRAMES:
                log(
                    "vad",
                    f"p={prob:.2f} db={level_db:.1f} floor={self._noise_floor_db:.1f} "
                    f"speech={is_speech} bot={bot_speaking}",
                )

            if is_speech:
                self._silence_frames = 0
                self._speech_frames += 1
                if not self.speaking and self._speech_frames >= need_speech:
                    self.speaking = True
                    if self.on_event:
                        self.on_event(VadEvent.SPEECH_START, prob)
                    yield VadEvent.SPEECH_START, prob
            else:
                self._speech_frames = 0
                self._silence_frames += 1
                if self.speaking and self._silence_frames >= need_silence:
                    self.speaking = False
                    if self.on_event:
                        self.on_event(VadEvent.SPEECH_END, prob)
                    yield VadEvent.SPEECH_END, prob

    def stats(self) -> dict:
        return {
            "frames_seen": self.frames_seen,
            "frames_speech": self.frames_speech,
            "frames_rejected_by_energy": self.frames_rejected_by_energy,
            "noise_floor_db": round(self._noise_floor_db, 1),
            "model_loaded": self._model is not None,
        }
