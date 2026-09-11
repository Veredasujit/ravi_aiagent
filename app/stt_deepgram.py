"""
stt_deepgram.py — Deepgram live transcription over a raw WebSocket.

We talk to `/v1/listen` directly rather than through the SDK so there is one
less layer between the phone and the model, and so every message is visible in
the logs.

Key settings for telephony:
  encoding=linear16, sample_rate=8000   -> matches Teler exactly, no resampling
  interim_results=true                  -> needed for UtteranceEnd, and lets us
                                           confirm a barge-in is real speech
  endpointing=300                       -> fast `speech_final` on a short pause
  utterance_end_ms=1000                 -> a noise-immune backstop for turn end
  vad_events=true                       -> SpeechStarted, cross-checked with Silero

Hindi: `nova-3` + `language=multi` handles Hinglish code-switching, which is
what people actually speak on these calls. Set DEEPGRAM_LANGUAGE=hi for pure
Devanagari Hindi if your callers never mix in English.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Awaitable, Callable, Optional
from urllib.parse import urlencode

import websockets

from . import config
from .logging_setup import CallLog, error, info, log, warn

TranscriptCb = Callable[[str, bool, bool], Awaitable[None]]
# (text, is_final, speech_final)
EventCb = Callable[[str], Awaitable[None]]


class DeepgramSTT:
    def __init__(
        self,
        call_log: CallLog,
        on_transcript: TranscriptCb,
        on_speech_started: Optional[EventCb] = None,
        on_utterance_end: Optional[EventCb] = None,
    ) -> None:
        self.call_log = call_log
        self.on_transcript = on_transcript
        self.on_speech_started = on_speech_started
        self.on_utterance_end = on_utterance_end

        self._ws = None
        self._recv_task: Optional[asyncio.Task] = None
        self._keepalive_task: Optional[asyncio.Task] = None
        self._closed = False
        self._audio_sent_bytes = 0
        self._connected_at = 0.0
        self.ready = asyncio.Event()

    # ── connection ──────────────────────────────────────────────────────────
    def _url(self) -> str:
        params = {
            "model": config.DEEPGRAM_MODEL,
            "language": config.DEEPGRAM_LANGUAGE,
            "encoding": "linear16",
            "sample_rate": str(config.TELER_SAMPLE_RATE),
            "channels": "1",
            "interim_results": "true",
            "punctuate": "true",
            "smart_format": "true" if config.DEEPGRAM_SMART_FORMAT else "false",
            "endpointing": str(config.DEEPGRAM_ENDPOINTING_MS),
            "utterance_end_ms": str(config.DEEPGRAM_UTTERANCE_END_MS),
            "vad_events": "true",
            "filler_words": "false",
            "no_delay": "true",
        }
        return f"{config.DEEPGRAM_URL}?{urlencode(params)}"

    async def start(self) -> None:
        if not config.DEEPGRAM_API_KEY:
            error("stt", "DEEPGRAM_API_KEY missing — transcription disabled")
            return
        url = self._url()
        log("stt", f"connecting: {url}")
        t0 = time.monotonic()
        try:
            self._ws = await websockets.connect(
                url,
                additional_headers={
                    "Authorization": f"Token {config.DEEPGRAM_API_KEY}"
                },
                ping_interval=5,
                ping_timeout=20,
                close_timeout=2,
                max_queue=64,
            )
        except TypeError:
            # websockets < 14 used extra_headers
            self._ws = await websockets.connect(
                url,
                extra_headers={"Authorization": f"Token {config.DEEPGRAM_API_KEY}"},
                ping_interval=5,
                ping_timeout=20,
                close_timeout=2,
                max_queue=64,
            )
        self._connected_at = time.monotonic()
        ms = round((self._connected_at - t0) * 1000, 1)
        info("stt", f"deepgram connected in {ms} ms ({config.DEEPGRAM_MODEL}/{config.DEEPGRAM_LANGUAGE})")
        self.call_log.event("stt_connected", ms=ms, model=config.DEEPGRAM_MODEL,
                            language=config.DEEPGRAM_LANGUAGE)
        self.ready.set()

        self._recv_task = asyncio.create_task(self._recv_loop(), name="dg-recv")
        self._keepalive_task = asyncio.create_task(
            self._keepalive_loop(), name="dg-keepalive"
        )

    # ── outbound audio ──────────────────────────────────────────────────────
    async def send_audio(self, pcm: bytes) -> None:
        if self._ws is None or self._closed or not pcm:
            return
        try:
            await self._ws.send(pcm)
            self._audio_sent_bytes += len(pcm)
        except Exception as exc:
            warn("stt", f"send failed: {exc}")

    async def finalize(self) -> None:
        """Ask Deepgram to flush whatever it is holding, right now."""
        await self._send_json({"type": "Finalize"})

    async def _send_json(self, obj: dict) -> None:
        if self._ws is None or self._closed:
            return
        try:
            await self._ws.send(json.dumps(obj))
        except Exception as exc:
            log("stt", f"control send failed: {exc}")

    async def _keepalive_loop(self) -> None:
        try:
            while not self._closed:
                await asyncio.sleep(config.DEEPGRAM_KEEPALIVE_S)
                await self._send_json({"type": "KeepAlive"})
        except asyncio.CancelledError:
            pass

    # ── inbound messages ────────────────────────────────────────────────────
    async def _recv_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                await self._handle(msg)
        except websockets.exceptions.ConnectionClosed as exc:
            log("stt", f"deepgram socket closed: {exc.code} {exc.reason}")
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            error("stt", f"recv loop crashed: {exc}")

    async def _handle(self, msg: dict) -> None:
        mtype = msg.get("type")

        if mtype == "Results":
            alt = (
                msg.get("channel", {})
                .get("alternatives", [{}])[0]
            )
            text = (alt.get("transcript") or "").strip()
            is_final = bool(msg.get("is_final"))
            speech_final = bool(msg.get("speech_final"))
            if not text:
                return
            conf = alt.get("confidence")
            self.call_log.event(
                "transcript",
                text=text,
                is_final=is_final,
                speech_final=speech_final,
                confidence=conf,
            )
            log(
                "stt",
                f"{'FINAL' if is_final else 'interim'}"
                f"{' (speech_final)' if speech_final else ''}: {text}",
                "INFO" if is_final else "DEBUG",
            )
            await self.on_transcript(text, is_final, speech_final)

        elif mtype == "SpeechStarted":
            log("stt", "SpeechStarted")
            self.call_log.event("stt_speech_started")
            if self.on_speech_started:
                await self.on_speech_started("SpeechStarted")

        elif mtype == "UtteranceEnd":
            log("stt", "UtteranceEnd")
            self.call_log.event("stt_utterance_end")
            if self.on_utterance_end:
                await self.on_utterance_end("UtteranceEnd")

        elif mtype == "Metadata":
            log("stt", f"metadata: request_id={msg.get('request_id')}")

        elif mtype == "Error" or msg.get("error"):
            error("stt", f"deepgram error: {msg}")
            self.call_log.event("stt_error", detail=msg)

    # ── teardown ────────────────────────────────────────────────────────────
    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for t in (self._keepalive_task, self._recv_task):
            if t and not t.done():
                t.cancel()
        if self._ws is not None:
            try:
                await self._ws.send(json.dumps({"type": "CloseStream"}))
            except Exception:
                pass
            try:
                await self._ws.close()
            except Exception:
                pass
        info(
            "stt",
            f"closed after {self._audio_sent_bytes} bytes "
            f"({self._audio_sent_bytes / 2 / config.TELER_SAMPLE_RATE:.1f}s of audio)",
        )
        self._ws = None
