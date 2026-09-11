"""
playout.py — relay audio to Teler the way Teler expects it.

This module was originally a real-time pacer: one 20 ms frame every 20 ms on a
monotonic clock. That is the right design for a carrier that plays whatever you
hand it the instant it arrives. Teler is not that carrier — it runs its own
jitter buffer and synchronisation on the far side.

FreJun's own reference bridge makes this explicit. Its outbound handler does no
timing at all: it collects several TTS chunks into one buffer and relays that
buffer as a single `audio` message, then flushes the remainder when synthesis
finishes. No clock, no frame timer.

Pacing on top of Teler's buffer is actively harmful. Delivering at exactly real
time leaves zero slack, so any jitter on the path — and a free ngrok tunnel out
of India has plenty — starves the far-side buffer and the caller hears the
audio cut in and out. That is the "कटती हुई … बहुत buffering" the caller
described.

So: buffer and relay, matching the reference. What we keep from the old design
is a *model* of playback rather than a driver of it — we track how much audio
we have handed over and estimate when it will finish playing. That estimate is
what tells us when the farewell has been heard (so hangup doesn't clip it) and
how far into an utterance we are (for the barge-in grace window). Barge-in is
still instant: we drop our buffer and send Teler `clear`, which flushes what it
is holding.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Optional

from . import config
from .audio import byteswap16
from .logging_setup import CallLog, error, log

BYTES_PER_MS = config.TELER_SAMPLE_RATE * 2 / 1000.0

# The frame size we declare to Teler in the call flow. Every message we send is
# a whole multiple of this — see _flush().
FRAME = max(2, config.TELER_CHUNK_SIZE - (config.TELER_CHUNK_SIZE % 2))
MESSAGE_BYTES = FRAME * max(1, config.PLAYOUT_FRAMES_PER_MESSAGE)


class Playout:
    def __init__(self, ws, call_log: CallLog) -> None:
        self.ws = ws
        self.call_log = call_log

        self._buf = bytearray()
        self._buf_since = 0.0
        self._chunk_id = 0
        self._task: Optional[asyncio.Task] = None
        self._stopped = False
        self._generation = 0
        # Serialises every send. Without it a threshold flush and an explicit
        # end-of-clause flush can interleave and deliver audio out of order,
        # which sounds exactly like the stutter we are trying to remove.
        self._send_lock = asyncio.Lock()

        # Estimated monotonic time at which everything handed to Teler will
        # have finished playing. This is a model, not a schedule.
        self._play_end = 0.0
        self.utterance_started_at = 0.0
        self._tts_active = False

        self.drained = asyncio.Event()
        self.drained.set()

        # Carrier byte order, set once the detector decides.
        self.big_endian = config.TELER_ENDIAN == "big"

        self.bytes_sent = 0
        self.messages_sent = 0
        # True until the first flush of an utterance. We hold a larger cushion
        # at the start so the carrier is always playing from a queue, never
        # from whatever just arrived.
        self._priming = True

    # ── lifecycle ───────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="playout")

    async def stop(self) -> None:
        self._stopped = True
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    # ── state ───────────────────────────────────────────────────────────────
    @property
    def playing(self) -> bool:
        """True while audio is queued here or still playing at the carrier."""
        return bool(self._buf) or time.monotonic() < self._play_end

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def queued_ms(self) -> float:
        remaining = max(0.0, self._play_end - time.monotonic()) * 1000.0
        return remaining + len(self._buf) / BYTES_PER_MS

    def speaking_for_ms(self) -> float:
        if not self.playing:
            return 0.0
        return (time.monotonic() - self.utterance_started_at) * 1000.0

    # ── queueing ────────────────────────────────────────────────────────────
    def push(self, pcm: bytes, generation: int) -> None:
        """Queue audio. Ignored if a barge-in already invalidated this turn."""
        if self._stopped or not pcm or generation != self._generation:
            return
        if not self.playing:
            self.utterance_started_at = time.monotonic()
            self.drained.clear()
            self._priming = True
            self.call_log.event("playout_start")
        if not self._buf:
            self._buf_since = time.monotonic()
        self._buf.extend(pcm)

        # The 20 ms housekeeping loop picks this up once the threshold is hit.
        # Flushing from here with create_task would race the explicit
        # end-of-clause flush and reorder the audio.

    async def flush(self) -> None:
        """
        End of a TTS clause.

        If we are still priming AND more clauses are coming, this deliberately
        does nothing: flushing here would ship a 200 ms clause on its own and
        the carrier would run dry before the next one is synthesised. Keep
        building the cushion instead — the loop ships it once the cushion is
        full, and `tts_active(False)` releases whatever is left at the end of
        the turn.
        """
        if self._priming and self._tts_active:
            return
        await self._flush(final=not self._tts_active)

    def tts_active(self, active: bool) -> None:
        """
        Tell playout whether more audio is coming for this utterance.

        While the LLM is still writing and TTS is still producing, we keep the
        priming cushion. Once the turn is finished, the tail goes out even if
        it is smaller than the cushion — otherwise a short closing clause would
        sit in the buffer.
        """
        self._tts_active = active

    def clear(self, send_clear: bool = True) -> int:
        """
        Barge-in. Drops our buffer and tells Teler to flush what it holds.
        Returns the number of bytes discarded.
        """
        dropped = len(self._buf)
        self._buf.clear()
        self._generation += 1
        self._play_end = time.monotonic()
        self._priming = True
        self.drained.set()
        if send_clear:
            asyncio.create_task(self._send({"type": "clear"}))
        self.call_log.event("playout_clear", dropped_bytes=dropped)
        log("playout", f"cleared ({dropped} bytes dropped)")
        return dropped

    async def wait_drained(self, timeout: float = 30.0) -> None:
        """Wait until the queued audio should have finished playing."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self._buf and time.monotonic() >= self._play_end:
                self.drained.set()
                return
            await asyncio.sleep(0.05)
        log("playout", "wait_drained timed out")

    # ── sending ─────────────────────────────────────────────────────────────
    async def _flush(self, final: bool = False) -> None:
        """
        Send buffered audio as whole `chunk_size` frames.

        Teler slices what we send into the frame size we declared in the flow.
        A message that is not a whole number of frames leaves a partial frame at
        the end, and a partial frame is a click — one per message, which is
        exactly the light chopping that survived every other fix.

        So we send only complete frames and keep the remainder buffered for the
        next message. On the last flush of an utterance there is nothing to keep
        it company, so the tail is padded to a frame boundary with silence.
        """
        async with self._send_lock:
            if self._stopped or not self._buf:
                return

            usable = (len(self._buf) // FRAME) * FRAME
            if usable == 0 and not final:
                return  # not even one whole frame yet

            if final and usable < len(self._buf):
                # Pad the tail to a frame boundary rather than send a partial
                # one. A few ms of silence is inaudible; a torn frame is not.
                pad = FRAME - (len(self._buf) % FRAME)
                self._buf.extend(b"\x00" * pad)
                usable = len(self._buf)

            chunk = bytes(self._buf[:usable])
            del self._buf[:usable]
            self._priming = False

            wire = byteswap16(chunk) if self.big_endian else chunk
            self._chunk_id += 1
            self.bytes_sent += len(wire)
            self.messages_sent += 1

            # Advance the playback model. If the carrier has already caught up,
            # this chunk starts playing now; otherwise it queues behind the rest.
            now = time.monotonic()
            if self._play_end < now:
                self._play_end = now
            self._play_end += len(chunk) / 2 / config.TELER_SAMPLE_RATE

            await self._send(
                {
                    "type": "audio",
                    "audio_b64": base64.b64encode(wire).decode("ascii"),
                    "chunk_id": self._chunk_id,
                }
            )
            if config.LOG_AUDIO_FRAMES:
                log(
                    "playout",
                    f"relayed chunk {self._chunk_id}: {len(chunk)} B "
                    f"({len(chunk)/BYTES_PER_MS:.0f} ms)",
                )

    async def _loop(self) -> None:
        """
        Housekeeping only — this does not drive playback.

        It flushes a partial buffer that has been sitting too long (so a short
        trailing clause goes out promptly) and reports when playback finishes.
        """
        max_age = config.PLAYOUT_MAX_BUFFER_MS / 1000.0
        was_playing = False
        try:
            while not self._stopped:
                await asyncio.sleep(0.02)

                if self._buf:
                    # While priming, wait for a real cushion. Once the carrier
                    # has audio queued, smaller flushes keep latency down.
                    threshold = (
                        config.PLAYOUT_PRIME_BYTES
                        if self._priming
                        else max(MESSAGE_BYTES, config.PLAYOUT_FLUSH_BYTES)
                    )
                    # The age fallback stops a short final clause from waiting
                    # forever, but only once something is already playing —
                    # otherwise priming would be defeated by the timer.
                    aged = (time.monotonic() - self._buf_since) >= max_age
                    # `final` only when TTS is done and this is the tail —
                    # padding mid-utterance would inject silence into speech.
                    tail = aged and not self._tts_active
                    if len(self._buf) >= threshold or (aged and not self._priming):
                        await self._flush(final=tail)
                    elif tail and self._priming:
                        await self._flush(final=True)

                playing = self.playing
                if was_playing and not playing:
                    self.drained.set()
                    self.call_log.event(
                        "playout_drained",
                        bytes_sent=self.bytes_sent,
                        messages=self.messages_sent,
                        frame_bytes=FRAME,
                        aligned=self.bytes_sent % FRAME == 0,
                    )
                    log(
                        "playout",
                        f"drained ({self.bytes_sent} B in "
                        f"{self.messages_sent} messages)",
                    )
                was_playing = playing

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            error("playout", f"loop crashed: {exc}")

    async def _send(self, obj: dict) -> None:
        try:
            await self.ws.send_text(json.dumps(obj))
        except Exception as exc:
            log("playout", f"send failed (call probably ended): {exc}")
            self._stopped = True
