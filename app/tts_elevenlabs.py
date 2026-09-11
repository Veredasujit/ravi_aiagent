"""
tts_elevenlabs.py — ElevenLabs streaming TTS over WebSocket.

Endpoint:
    wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input
        ?model_id=eleven_flash_v2_5&output_format=ulaw_8000

Why `ulaw_8000`: ElevenLabs does not offer `pcm_8000` for text-to-speech — the
lowest PCM rate is 16 kHz. μ-law at 8 kHz is their telephony format (it is what
the Twilio integration uses), it is exactly the rate Teler wants, and we already
have a verified G.711 decoder in audio.py. So the audio path is one table lookup
with no resampling. `pcm_16000` is kept as a fallback rung and is resampled down.

Protocol (from the ElevenLabs realtime-tts guide):
  1. Open the socket, send an init message with a single space as `text` plus
     `voice_settings` and `generation_config`.
  2. For each utterance send `{"text": "...", "flush": true}`. The flush forces
     generation instead of waiting for the character buffer to fill — without it
     a short reply sits in their buffer and the caller hears nothing.
  3. Audio arrives as `{"audio": "<base64>", "isFinal": null}` and the sequence
     ends with `isFinal: true`.

Two things the docs warn about, both handled here:
  - Sending `{"text": ""}` CLOSES the connection. We never send an empty string;
    we flush instead and keep the socket for the whole call.
  - The socket closes after 20 s of inactivity. A keepalive sends a single space
    every 15 s.

Barge-in drops the socket rather than trying to cancel, and a reconnect runs in
the background — same approach as the Sarvam client, for the same reason: a
socket with unknown leftovers in it will splice audio into the next reply.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import AsyncGenerator, Optional
from urllib.parse import urlencode

import websockets

from . import config
from .audio import mulaw_to_pcm, resample
from .logging_setup import CallLog, error, info, log, warn

# Formats we can turn into 8 kHz PCM without adding a dependency.
# ElevenLabs has no pcm_8000 for TTS, so mu-law is the zero-resample option.
FORMAT_LADDER: list[str] = ["ulaw_8000", "pcm_16000", "pcm_24000"]

_RATE = {"ulaw_8000": 8000, "pcm_16000": 16000, "pcm_24000": 24000}


class TTSAborted(Exception):
    """Raised inside a synthesis stream when the caller barged in."""


def _ladder_for(preferred: str) -> list[str]:
    out = [preferred]
    for f in FORMAT_LADDER:
        if f not in out:
            out.append(f)
    return out


class ElevenLabsTTS:
    """Persistent ElevenLabs WebSocket. One instance per call."""

    def __init__(self, call_log: CallLog) -> None:
        self.call_log = call_log
        self._ws = None
        self._lock = asyncio.Lock()
        self._connect_lock = asyncio.Lock()
        self._abort = asyncio.Event()
        self._closed = False
        self._keepalive: Optional[asyncio.Task] = None
        self.connected = False

        self._ladder = _ladder_for(config.ELEVEN_OUTPUT_FORMAT)
        self._rung = 0
        self.validated = False

    # ── negotiated format ───────────────────────────────────────────────────
    @property
    def codec(self) -> str:
        return self._ladder[self._rung]

    @property
    def sample_rate(self) -> int:
        return _RATE.get(self.codec, 8000)

    # ── connection ──────────────────────────────────────────────────────────
    def _url(self) -> str:
        params = {
            "model_id": config.ELEVEN_MODEL,
            "output_format": self.codec,
            # Keeps the socket alive a little longer between utterances than the
            # 20 s default, so a thoughtful caller doesn't cost us a reconnect.
            "inactivity_timeout": str(config.ELEVEN_INACTIVITY_TIMEOUT_S),
        }
        if config.ELEVEN_LANGUAGE:
            params["language_code"] = config.ELEVEN_LANGUAGE
        return (
            f"{config.ELEVEN_WS_BASE}/{config.ELEVEN_VOICE_ID}/stream-input"
            f"?{urlencode(params)}"
        )

    def _init_message(self) -> dict:
        return {
            # A single space, per the docs. An empty string would close the
            # connection immediately.
            "text": " ",
            "voice_settings": {
                "stability": config.ELEVEN_STABILITY,
                "similarity_boost": config.ELEVEN_SIMILARITY,
                "speed": config.ELEVEN_SPEED,
                "use_speaker_boost": False,
            },
            "generation_config": {
                # Lower first threshold than the default [120,160,250,290] so a
                # short reply starts generating sooner. We flush anyway, but this
                # helps the long ones.
                "chunk_length_schedule": config.ELEVEN_CHUNK_SCHEDULE,
            },
        }

    async def connect(self) -> bool:
        async with self._connect_lock:
            if self._closed:
                return False
            if self._ws is not None:
                return True
            return await self._do_connect()

    async def _do_connect(self) -> bool:
        if not config.ELEVEN_API_KEY:
            error("tts", "ELEVENLABS_API_KEY missing")
            return False
        if not config.ELEVEN_VOICE_ID:
            error("tts", "ELEVEN_VOICE_ID missing")
            return False

        t0 = time.monotonic()
        headers = {"xi-api-key": config.ELEVEN_API_KEY}
        url = self._url()
        if config.ELEVEN_WIRE_LOG:
            log("tts", f"connecting: {url}")
        try:
            try:
                self._ws = await asyncio.wait_for(
                    websockets.connect(
                        url,
                        additional_headers=headers,
                        ping_interval=20,
                        ping_timeout=20,
                        close_timeout=2,
                        max_size=8 * 1024 * 1024,
                    ),
                    timeout=config.ELEVEN_CONNECT_TIMEOUT_S,
                )
            except TypeError:
                self._ws = await asyncio.wait_for(
                    websockets.connect(
                        url,
                        extra_headers=headers,
                        ping_interval=20,
                        ping_timeout=20,
                        close_timeout=2,
                        max_size=8 * 1024 * 1024,
                    ),
                    timeout=config.ELEVEN_CONNECT_TIMEOUT_S,
                )
        except Exception as exc:
            error("tts", f"elevenlabs connect failed: {exc}")
            self.call_log.event(
                "tts_connect_failed", vendor="elevenlabs", error=str(exc)
            )
            self._ws = None
            self.connected = False
            return False

        init = self._init_message()
        if config.ELEVEN_WIRE_LOG:
            log("tts", f"-> init {json.dumps(init, ensure_ascii=False)}")
        try:
            await self._ws.send(json.dumps(init))
        except Exception as exc:
            error("tts", f"elevenlabs init send failed: {exc}")
            self._ws = None
            return False

        ms = round((time.monotonic() - t0) * 1000, 1)
        self.connected = True
        info(
            "tts",
            f"elevenlabs ready in {ms} ms "
            f"({config.ELEVEN_MODEL}/{config.ELEVEN_VOICE_ID}/{self.codec})",
        )
        self.call_log.event(
            "tts_connected", vendor="elevenlabs", ms=ms,
            codec=self.codec, sample_rate=self.sample_rate,
        )

        if self._keepalive is None or self._keepalive.done():
            self._keepalive = asyncio.create_task(
                self._keepalive_loop(), name="eleven-keepalive"
            )
        return True

    async def _keepalive_loop(self) -> None:
        """A single space resets their inactivity timer without generating."""
        try:
            while not self._closed:
                await asyncio.sleep(config.ELEVEN_KEEPALIVE_S)
                if self._ws is None or self._lock.locked():
                    continue
                try:
                    await self._ws.send(json.dumps({"text": " "}))
                except Exception:
                    return
        except asyncio.CancelledError:
            pass

    # ── connect-time validation ─────────────────────────────────────────────
    async def prewarm(self) -> bool:
        """Open the socket and prove the format works before the greeting."""
        if self.validated:
            return True

        probe = "जी।"
        while self._rung < len(self._ladder):
            fmt = self._ladder[self._rung]
            if not await self.connect():
                return False
            try:
                got = 0
                async for pcm in self._stream_once(probe, probe=True):
                    got += len(pcm)
                if got > 0:
                    self.validated = True
                    info("tts", f"format confirmed: {fmt} ({got} B probe)")
                    self.call_log.event(
                        "tts_codec_confirmed", vendor="elevenlabs",
                        codec=fmt, sample_rate=self.sample_rate, probe_bytes=got,
                    )
                    return True
                warn("tts", f"{fmt} connected but returned no audio")
            except Exception as exc:
                warn("tts", f"{fmt} rejected: {exc}")
                self.call_log.event(
                    "tts_codec_rejected", vendor="elevenlabs", codec=fmt,
                    error=str(exc),
                )

            await self._drop(reconnect=False)
            self._rung += 1
            if self._rung < len(self._ladder):
                info("tts", f"stepping down to {self._ladder[self._rung]}")

        error(
            "tts",
            "every ElevenLabs output format was rejected — check "
            "ELEVEN_VOICE_ID is a real voice on this account and the key has "
            "text-to-speech permission",
        )
        self._rung = 0
        return False

    # ── decoding ────────────────────────────────────────────────────────────
    def _decode(self, raw: bytes) -> bytes:
        """Vendor chunk -> 8 kHz s16 PCM ready for the wire."""
        if self.codec == "ulaw_8000":
            return mulaw_to_pcm(raw)
        pcm = raw
        if self.sample_rate != config.TELER_SAMPLE_RATE:
            pcm = resample(pcm, self.sample_rate, config.TELER_SAMPLE_RATE)
        return pcm

    # ── synthesis ───────────────────────────────────────────────────────────
    async def stream(self, text: str) -> AsyncGenerator[bytes, None]:
        text = (text or "").strip()
        if not text:
            return
        async with self._lock:
            async for pcm in self._stream_once(text):
                yield pcm

    async def _stream_once(
        self, text: str, probe: bool = False
    ) -> AsyncGenerator[bytes, None]:
        self._abort.clear()
        if not await self.connect():
            raise RuntimeError("elevenlabs not connected")

        ws = self._ws
        assert ws is not None
        t_send = time.monotonic()

        # flush=true forces generation now instead of waiting for their
        # character buffer to fill. Without it a short reply never plays.
        msg = {"text": text + " ", "flush": True}
        if not probe:
            self.call_log.event(
                "tts_request", vendor="elevenlabs", chars=len(text), text=text,
                codec=self.codec,
            )
            log("tts", f"synthesising ({len(text)} chars): {text}")
        if config.ELEVEN_WIRE_LOG:
            log("tts", f"-> {json.dumps(msg, ensure_ascii=False)}")

        try:
            await ws.send(json.dumps(msg))
        except Exception as exc:
            error("tts", f"elevenlabs send failed: {exc}")
            await self._drop()
            raise

        first = True
        total = 0
        while True:
            if self._abort.is_set():
                log("tts", "aborted mid-stream (barge-in)")
                await self._drop()
                raise TTSAborted()

            timeout = (
                config.ELEVEN_FIRST_CHUNK_TIMEOUT_S
                if total == 0
                else config.ELEVEN_IDLE_END_S
            )
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            except asyncio.TimeoutError:
                if total > 0:
                    # We never saw isFinal, so more audio may still be coming.
                    # Reusing this socket would splice it onto the next reply.
                    if not probe:
                        dur = total / 2 / config.TELER_SAMPLE_RATE
                        self.call_log.event(
                            "tts_done", vendor="elevenlabs", bytes=total,
                            audio_seconds=round(dur, 2), ended="idle",
                        )
                        warn(
                            "tts",
                            f"no isFinal after {timeout}s — {total} bytes "
                            f"({dur:.2f}s); dropping the socket",
                        )
                    await self._drop()
                    return
                warn("tts", "elevenlabs timed out before any audio")
                self.call_log.event("tts_timeout", vendor="elevenlabs")
                await self._drop()
                raise
            except websockets.exceptions.ConnectionClosed as exc:
                if self._abort.is_set():
                    raise TTSAborted()
                if total > 0:
                    log("tts", f"socket closed after audio ({exc.code})")
                    await self._drop()
                    return
                warn("tts", f"elevenlabs socket closed: {exc.code} {exc.reason}")
                await self._drop()
                raise
            except asyncio.CancelledError:
                await self._drop()
                raise

            try:
                data = json.loads(raw)
            except (TypeError, ValueError):
                continue

            if data.get("error") or data.get("message") and not data.get("audio"):
                detail = data.get("error") or data.get("message")
                error("tts", f"elevenlabs error: {detail}")
                self.call_log.event(
                    "tts_error", vendor="elevenlabs", detail=str(detail)[:400]
                )
                await self._drop()
                raise RuntimeError(f"elevenlabs {detail}")

            audio_b64 = data.get("audio")
            if audio_b64:
                pcm = self._decode(base64.b64decode(audio_b64))
                if pcm:
                    if first and not probe:
                        ttfb = round((time.monotonic() - t_send) * 1000, 1)
                        info("tts", f"first audio chunk in {ttfb} ms")
                        self.call_log.event(
                            "tts_ttfb", vendor="elevenlabs", ms=ttfb
                        )
                        first = False
                    total += len(pcm)
                    yield pcm

            if data.get("isFinal"):
                if not probe:
                    dur = total / 2 / config.TELER_SAMPLE_RATE
                    self.call_log.event(
                        "tts_done", vendor="elevenlabs", bytes=total,
                        audio_seconds=round(dur, 2), ended="final",
                    )
                    log("tts", f"done (final): {total} bytes ({dur:.2f}s)")
                return

    def abort(self) -> None:
        self._abort.set()

    async def _drop(self, reconnect: bool = True) -> None:
        ws, self._ws = self._ws, None
        self.connected = False
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
        if self._keepalive and not self._keepalive.done():
            self._keepalive.cancel()
        await self._drop(reconnect=False)
