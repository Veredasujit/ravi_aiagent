"""
session.py — one live phone call.

The pipeline, end to end:

    Teler WS ──► base64 decode ──► 8 kHz PCM ──┬──► Silero VAD  (is a human talking?)
                                               └──► Deepgram    (what did they say?)

    turn end detected ──► OpenAI ──► ONE Sarvam TTS request per reply
                                                    │
                        Teler WS ◄── buffered relay ┘

Two decisions carry most of the "does it feel like a real conversation" weight:

WHEN TO STOP TALKING (barge-in)
  Silero says speech started. That alone is not enough on a noisy line, so we
  also require Deepgram to return actual words within a short window. Silero
  gives us the sub-100 ms reaction; Deepgram gives us the confidence that it
  was a person and not a horn. During the first BARGE_IN_GRACE_MS of our own
  utterance we ignore everything, because that window is where line echo lives.

  Two things measured on a real call shaped this. Deepgram's first interim came
  1.93 s after Silero fired, so the ASR confirmation window has to be wider than
  that or every genuine interruption gets written off as noise. And the grace
  window must be measured from the start of the whole REPLY, not from whenever
  playout last restarted — otherwise every clause re-arms it and the caller can
  never get in.

ONE TTS REQUEST PER REPLY
  We used to synthesise clause by clause to shave first-word latency. On a real
  line that traded a smaller delay for three Sarvam round trips per reply, and
  every one of those round trips is an audible gap mid-sentence. FreJun's own
  bridge sends the whole reply in a single streamText() call. So do we now: the
  caller waits slightly longer to hear the first word, and then hears one
  continuous sentence instead of a stuttering one.

WHEN TO START TALKING (turn end)
  Whichever fires first: Deepgram's `speech_final`, its `UtteranceEnd`, or our
  own VAD silence timer. Any one of them alone has a failure mode; together
  they don't.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Optional

from . import config
from .audio import EndianDetector
from .llm import LLMClient, speakable
from .logging_setup import CallLog, error, info, log, new_call_id, set_call_id, warn
from .playout import Playout
from .prompt import GREETING, initial_messages
from .stt_deepgram import DeepgramSTT
from .tools import TOOLS_SCHEMA, ToolRunner
from .tts import TTSAborted, TTSEngine
from .vad import SileroVAD, VadEvent

# How long a VAD-armed barge-in waits for Deepgram to confirm real words.
# Measured: the first interim landed 1.93 s after Silero fired, so anything
# near the old 1.2 s discards genuine interruptions as background noise.
BARGE_ASR_WINDOW_S = 3.0

# A pause is not the end of a turn. People stop mid-sentence to think, and on a
# real call Ravi answered "first is," and "That the thing I face is" because our
# VAD silence timer fired during those pauses. Deepgram's `speech_final` and
# `UtteranceEnd` are the reliable end-of-turn signals; the VAD timer is only a
# backstop for when they never arrive. So the backstop waits this long after the
# LAST transcript activity before it will end a turn on its own.
VAD_TURN_END_QUIET_S = 1.6

# How long to wait for Teler's `start` frame before speaking anyway.
# Measured across calls, `start` arrives anywhere from 1.9 s to 12.5 s after the
# socket opens — and sometimes only AFTER we have already sent audio. So this is
# a short courtesy pause, not a real gate: waiting longer just creates dead air
# at the top of the call.
STREAM_READY_TIMEOUT_S = 1.0

# A final transcript that arrives *after* its turn already started stays in the
# pending buffer. If nothing consumes it, it must not resurface twenty seconds
# later glued to the next thing the caller says.
PENDING_TEXT_MAX_AGE_S = 10.0


class CallSession:
    def __init__(self, ws, call_id: Optional[str] = None) -> None:
        self.ws = ws
        self.call_id = call_id or new_call_id()
        set_call_id(self.call_id)

        self.call_log = CallLog(self.call_id)
        self.playout = Playout(ws, self.call_log)
        self.vad = SileroVAD()
        self.tts = TTSEngine(self.call_log)
        self.llm = LLMClient(self.call_log)
        self.tools = ToolRunner(self.call_log, self.call_id)
        self.stt = DeepgramSTT(
            self.call_log,
            on_transcript=self._on_transcript,
            on_speech_started=self._on_stt_speech_started,
            on_utterance_end=self._on_stt_utterance_end,
        )

        self.messages = initial_messages()

        # Carrier byte order. Decided from the first second of inbound audio
        # unless TELER_ENDIAN forces it.
        self.endian = EndianDetector(config.TELER_ENDIAN)
        self._endian_known = asyncio.Event()
        if self.endian.decided:
            self._endian_known.set()

        # turn state
        self._turn_task: Optional[asyncio.Task] = None
        self._turn_lock = asyncio.Lock()
        self._pending_user_text = ""
        self._pending_since = 0.0
        self._last_interim = ""
        self._turn_seq = 0

        # Set when Teler confirms the media stream is live. Audio pushed before
        # this is discarded by the carrier — it is the single biggest cause of
        # a greeting that sounds cut off at the start.
        self._stream_ready = asyncio.Event()

        # barge-in state
        self._vad_barge_at: float = 0.0
        self._barge_armed = False
        # When the current REPLY started coming out of the speaker. Playout's
        # own timer restarts on every queued chunk, which silently re-armed the
        # grace window mid-reply and made barge-in impossible.
        self._utterance_started_at: float = 0.0

        # housekeeping
        self._ended = asyncio.Event()
        self._last_user_audio_at = time.monotonic()
        self._last_activity_at = time.monotonic()
        self._reprompts = 0
        self._started_at = time.monotonic()
        self._watchdog: Optional[asyncio.Task] = None
        self._audio_frames_in = 0

    # ════════════════════════════════════════════════════════════════════════
    # lifecycle
    # ════════════════════════════════════════════════════════════════════════
    async def run(self) -> None:
        info("call", f"session start ({config.CLINIC_NAME})")
        self.call_log.event(
            "call_start",
            clinic=config.CLINIC_NAME,
            stt=f"{config.DEEPGRAM_MODEL}/{config.DEEPGRAM_LANGUAGE}",
            tts=f"{config.TTS_PROVIDER}",
            llm=config.OPENAI_MODEL,
            vad_enabled=config.VAD_ENABLED,
        )

        self.playout.start()

        # Start Deepgram in the background. The greeting needs TTS and nothing
        # else — nobody has spoken yet — and waiting on the STT socket was
        # adding ~1.8 s of dead air before the caller heard anything.
        stt_task = asyncio.create_task(self.stt.start(), name="stt-connect")
        try:
            await self.tts.prewarm()
        except Exception as exc:
            error("call", f"TTS prewarm failed: {exc}")
        # Wait for Teler to confirm the stream is live. Accepting the
        # WebSocket is NOT the same as the media path being open: on a measured
        # call the `start` frame arrived 1.9 s later, and everything we sent in
        # the meantime was dropped by the carrier.
        await self._await_stream_ready()

        await asyncio.sleep(min(config.GREETING_DELAY_MS, 150) / 1000.0)

        # Hold the greeting until we know the carrier's byte order, otherwise
        # the first thing the caller hears could go out byte-swapped. Capped so
        # a silent caller never blocks the greeting entirely.
        if not self._endian_known.is_set():
            try:
                await asyncio.wait_for(
                    self._endian_known.wait(),
                    timeout=config.ENDIAN_DETECT_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                warn("call", "endianness undecided (caller silent) — assuming "
                             f"{'big' if self.playout.big_endian else 'little'}-endian")
        await self._speak_text(GREETING, tag="greeting")

        if stt_task.done() and stt_task.exception():
            error("call", f"STT connect failed: {stt_task.exception()}")

        # Only now. Started any earlier, its silence timer fires while the
        # greeting is still being prepared and the caller's first words from
        # Ravi are "can you hear me?" instead of the greeting.
        self._last_activity_at = time.monotonic()
        self._watchdog = asyncio.create_task(self._watchdog_loop(), name="watchdog")

        try:
            await self._read_loop()
        finally:
            await self.shutdown()

    async def _await_stream_ready(self) -> None:
        """Block until Teler's `start` frame (or the first inbound audio)."""
        if self._stream_ready.is_set():
            return
        t0 = time.monotonic()
        try:
            await asyncio.wait_for(
                self._stream_ready.wait(), timeout=STREAM_READY_TIMEOUT_S
            )
            ms = round((time.monotonic() - t0) * 1000, 1)
            info("call", f"media stream live after {ms} ms — safe to speak")
            self.call_log.event("stream_ready_wait", ms=ms, timed_out=False)
        except asyncio.TimeoutError:
            warn(
                "call",
                f"no start frame after {STREAM_READY_TIMEOUT_S}s — speaking "
                f"anyway; the greeting may be clipped",
            )
            self.call_log.event(
                "stream_ready_wait",
                ms=STREAM_READY_TIMEOUT_S * 1000,
                timed_out=True,
            )
            self._stream_ready.set()

    async def shutdown(self) -> None:
        if self._ended.is_set():
            return
        self._ended.set()
        dur = round(time.monotonic() - self._started_at, 1)
        info("call", f"session end after {dur}s")
        self.call_log.event(
            "call_end",
            seconds=dur,
            turns=self._turn_seq,
            audio_frames_in=self._audio_frames_in,
            vad=self.vad.stats(),
            big_endian=self.endian.big_endian,
            booking=self.tools.booking,
            hangup_reason=self.tools.hangup_reason or None,
        )

        for t in (self._turn_task, self._watchdog):
            if t and not t.done():
                t.cancel()

        await asyncio.gather(
            self.playout.stop(),
            self.stt.close(),
            self.tts.close(),
            self.llm.close(),
            self.tools.close(),
            return_exceptions=True,
        )
        self.call_log.close()

    # ════════════════════════════════════════════════════════════════════════
    # inbound audio from Teler
    # ════════════════════════════════════════════════════════════════════════
    async def _read_loop(self) -> None:
        while not self._ended.is_set():
            try:
                raw = await self.ws.receive_text()
            except Exception as exc:
                log("call", f"websocket read ended: {exc}")
                return

            try:
                msg = json.loads(raw)
            except (TypeError, ValueError):
                log("call", f"non-JSON frame ignored ({len(raw)} bytes)")
                continue

            mtype = (msg.get("type") or msg.get("event") or "").lower()

            if mtype in ("audio", "media"):
                await self._on_audio(msg)
            elif mtype in ("start", "connected", "stream_started"):
                self._on_start(msg)
            elif mtype == "dtmf":
                digit = self._dig(msg)
                info("call", f"DTMF: {digit}")
                self.call_log.event("dtmf", digit=digit)
            elif mtype in ("stop", "stream_stopped", "hangup", "disconnected"):
                info("call", f"remote stop: {msg}")
                self.call_log.event("remote_stop", detail=msg)
                return
            else:
                log("call", f"unhandled frame type {mtype!r}: {str(msg)[:200]}")

    def _on_start(self, msg: dict) -> None:
        data = msg.get("data") or msg.get("start") or {}
        remote_id = (
            data.get("call_id")
            or data.get("stream_id")
            or msg.get("call_id")
            or msg.get("stream_id")
        )
        info("call", f"stream started (teler id={remote_id})")
        self.call_log.event("stream_start", teler_call_id=remote_id, detail=data)
        self._stream_ready.set()

    @staticmethod
    def _dig(msg: dict) -> str:
        data = msg.get("data") or {}
        return str(data.get("digit") or data.get("digits") or msg.get("digit") or "")

    @staticmethod
    def _extract_audio_b64(msg: dict) -> str:
        data = msg.get("data") or {}
        return (
            data.get("audio_b64")
            or data.get("audio")
            or data.get("payload")
            or msg.get("audio_b64")
            or ""
        )

    async def _on_audio(self, msg: dict) -> None:
        b64 = self._extract_audio_b64(msg)
        if not b64:
            return
        try:
            pcm = base64.b64decode(b64)
        except Exception as exc:
            warn("call", f"bad base64 audio frame: {exc}")
            return
        if not pcm:
            return

        self._audio_frames_in += 1
        self._last_user_audio_at = time.monotonic()

        # Inbound audio means the media path is open, whether or not we ever
        # saw a `start` frame.
        if not self._stream_ready.is_set():
            info("call", "inbound audio before start frame — stream is live")
            self._stream_ready.set()

        # Normalise the carrier's byte order before anything else touches the
        # audio. Get this wrong and Deepgram transcribes nothing and the VAD
        # sees noise, which is exactly what a byte-swapped stream looks like.
        was_decided = self.endian.decided
        pcm = self.endian.feed(pcm)
        if self.endian.decided and not was_decided:
            self.playout.big_endian = self.endian.big_endian
            order = "big" if self.endian.big_endian else "little"
            info("call", f"carrier byte order detected: {order}-endian "
                         f"(roughness little={self.endian.scores[0]} "
                         f"big={self.endian.scores[1]})")
            self.call_log.event(
                "endian_detected", big_endian=self.endian.big_endian,
                roughness_little=self.endian.scores[0],
                roughness_big=self.endian.scores[1],
            )
            self._endian_known.set()

        # Deepgram gets every byte, unconditionally — we never want a gap in
        # the transcript, even while the bot is talking.
        await self.stt.send_audio(pcm)

        if not config.VAD_ENABLED:
            return

        bot_speaking = self._bot_is_speaking()
        for event, prob in self.vad.feed(pcm, bot_speaking=bot_speaking):
            if event is VadEvent.SPEECH_START:
                self.call_log.event(
                    "vad_speech_start", prob=round(prob, 3),
                    bot_speaking=bot_speaking,
                )
                log("vad", f"SPEECH_START p={prob:.2f} bot_speaking={bot_speaking}")
                self._last_activity_at = time.monotonic()
                if bot_speaking:
                    await self._on_vad_barge_candidate()
            elif event is VadEvent.SPEECH_END:
                self.call_log.event("vad_speech_end", prob=round(prob, 3))
                log("vad", "SPEECH_END")
                self._barge_armed = False
                if not self._bot_is_speaking() and self._pending_user_text:
                    # Backstop only. If Deepgram is still producing transcripts
                    # the caller is mid-sentence, not finished — answering here
                    # is what makes Ravi talk over people.
                    quiet = time.monotonic() - self._last_transcript_at
                    if quiet >= VAD_TURN_END_QUIET_S:
                        await self._start_turn("vad_silence")
                    else:
                        log(
                            "turn",
                            f"vad_silence held: transcript activity {quiet:.1f}s "
                            f"ago, caller likely still speaking",
                        )

    def _bot_is_speaking(self) -> bool:
        return self.playout.playing or (
            self._turn_task is not None and not self._turn_task.done()
        )

    # ════════════════════════════════════════════════════════════════════════
    # barge-in
    # ════════════════════════════════════════════════════════════════════════
    async def _on_vad_barge_candidate(self) -> None:
        if not config.BARGE_IN_ENABLED:
            return

        speaking_ms = (
            (time.monotonic() - self._utterance_started_at) * 1000.0
            if self._utterance_started_at
            else self.playout.speaking_for_ms()
        )
        if speaking_ms < config.BARGE_IN_GRACE_MS:
            log("barge", f"ignored, only {speaking_ms:.0f} ms into our utterance")
            self.call_log.event("barge_ignored", reason="grace_window",
                                speaking_ms=round(speaking_ms))
            return

        if not config.BARGE_IN_REQUIRE_ASR:
            await self._do_barge_in("vad")
            return

        # Arm and wait for Deepgram to confirm there were actual words.
        self._barge_armed = True
        self._vad_barge_at = time.monotonic()
        log("barge", f"armed by VAD, waiting up to {BARGE_ASR_WINDOW_S}s for ASR")
        self.call_log.event("barge_armed")
        asyncio.create_task(self._barge_disarm_after(BARGE_ASR_WINDOW_S))

    async def _barge_disarm_after(self, seconds: float) -> None:
        await asyncio.sleep(seconds)
        if self._barge_armed:
            self._barge_armed = False
            log("barge", "disarmed — VAD fired but no words followed (noise)")
            self.call_log.event("barge_disarmed", reason="no_asr")

    async def _do_barge_in(self, source: str) -> None:
        if not self._bot_is_speaking():
            return
        self._barge_armed = False
        self._utterance_started_at = 0.0
        dropped = self.playout.clear()
        self.tts.abort()
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
        info("barge", f"INTERRUPTED by {source} ({dropped} bytes dropped)")
        self.call_log.event("barge_in", source=source, dropped_bytes=dropped)

        # Whatever we had already queued was never heard in full; tell the model
        # so it doesn't assume the caller heard the whole sentence.
        if self.messages and self.messages[-1].get("role") == "assistant":
            content = self.messages[-1].get("content") or ""
            if content:
                self.messages[-1]["content"] = content + " …(मरीज ने बीच में बात की)"

    # ════════════════════════════════════════════════════════════════════════
    # transcripts
    # ════════════════════════════════════════════════════════════════════════
    async def _on_transcript(
        self, text: str, is_final: bool, speech_final: bool
    ) -> None:
        self._last_activity_at = time.monotonic()
        self._last_transcript_at = time.monotonic()
        self._reprompts = 0

        if not is_final:
            self._last_interim = text
            if self._barge_armed and len(text.strip()) >= config.BARGE_IN_MIN_CHARS:
                await self._do_barge_in("vad+asr")
            return

        # Final transcript.
        if self._barge_armed and len(text.strip()) >= config.BARGE_IN_MIN_CHARS:
            await self._do_barge_in("vad+asr_final")

        if not self._pending_user_text:
            self._pending_since = time.monotonic()
        self._pending_user_text = (self._pending_user_text + " " + text).strip()

        if speech_final:
            await self._start_turn("speech_final")

    async def _on_stt_speech_started(self, _: str) -> None:
        self._last_activity_at = time.monotonic()

    async def _on_stt_utterance_end(self, _: str) -> None:
        if self._pending_user_text:
            await self._start_turn("utterance_end")

    # ════════════════════════════════════════════════════════════════════════
    # a turn
    # ════════════════════════════════════════════════════════════════════════
    async def _start_turn(self, trigger: str) -> None:
        async with self._turn_lock:
            text = self._pending_user_text.strip()
            if not text:
                return

            # A final that landed while an earlier turn was already running
            # stays buffered. If it was never consumed it is no longer what the
            # caller is talking about — dropping it beats prefixing it to the
            # next utterance, which is how "के लिए book कर सकते हो appointment?"
            # ended up glued to a sentence spoken twenty seconds later.
            age = time.monotonic() - self._pending_since
            if age > PENDING_TEXT_MAX_AGE_S:
                warn("turn", f"discarding {age:.0f}s-old pending text: {text!r}")
                self.call_log.event(
                    "stale_text_discarded", age_s=round(age, 1), text=text
                )
                self._pending_user_text = ""
                self._pending_since = 0.0
                return
            if self._turn_task and not self._turn_task.done():
                log("turn", f"turn already running, {trigger} ignored")
                return
            self._pending_user_text = ""
            self._pending_since = 0.0
            self._last_interim = ""
            self._turn_seq += 1
            seq = self._turn_seq

            info("turn", f"#{seq} trigger={trigger} user={text!r}")
            self.call_log.event("turn_start", seq=seq, trigger=trigger, user_text=text)
            self.call_log.mark(f"turn_{seq}")

            self.messages.append({"role": "user", "content": text})
            self._turn_task = asyncio.create_task(
                self._run_turn(seq), name=f"turn-{seq}"
            )

    async def _run_turn(self, seq: int, depth: int = 0) -> None:
        """One LLM round trip, streamed straight into TTS."""
        if depth > 3:
            warn("turn", "tool-call depth limit reached")
            return

        generation = self.playout.generation
        self.playout.tts_active(True)
        spoken_any = False
        assistant_text = ""
        tool_calls: list[dict] = []

        try:
            # Collect the whole reply first. Speaking it clause by clause shaved
            # a few hundred ms off the first word but cost a Sarvam round trip
            # per clause, and each of those is a gap the caller hears.
            async for kind, payload in self.llm.stream(self.messages, TOOLS_SCHEMA):
                if kind == "token":
                    assistant_text += payload
                elif kind == "tool_calls":
                    tool_calls = payload

            reply = assistant_text.strip()
            if reply:
                # ONE request, one continuous stream of audio.
                ok = await self._speak_chunk(
                    reply, generation, seq, latency_mark=f"turn_{seq}"
                )
                if not ok:
                    return
                spoken_any = True

        except asyncio.CancelledError:
            self.playout.tts_active(False)
            log("turn", f"#{seq} cancelled (barge-in)")
            if assistant_text.strip():
                self.messages.append(
                    {"role": "assistant", "content": assistant_text.strip()}
                )
            raise
        except Exception as exc:
            self.playout.tts_active(False)
            error("turn", f"#{seq} failed: {exc}")
            await self._speak_text(
                "माफ़ कीजिए, अभी कुछ तकनीकी दिक्कत आ गई। कृपया दोबारा बताइए।",
                tag="error",
            )
            return

        self.playout.tts_active(False)

        # Record what we said.
        if assistant_text.strip() or tool_calls:
            entry: dict = {"role": "assistant", "content": assistant_text.strip() or None}
            if tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"], ensure_ascii=False),
                        },
                    }
                    for tc in tool_calls
                ]
            self.messages.append(entry)

        if not tool_calls:
            if not spoken_any:
                warn("turn", f"#{seq} produced no speech")
            return

        # Run the tools, then let the model narrate the result.
        for tc in tool_calls:
            result = await self.tools.run(tc["name"], tc["arguments"])
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

        if self.tools.hangup_requested:
            await self._finish_and_hangup()
            return

        await self._run_turn(seq, depth + 1)

    # ════════════════════════════════════════════════════════════════════════
    # speaking
    # ════════════════════════════════════════════════════════════════════════
    async def _speak_chunk(
        self,
        chunk: str,
        generation: int,
        seq: int,
        latency_mark: Optional[str] = None,
    ) -> bool:
        """
        Synthesise a reply and queue it. Returns False if we were interrupted.
        """
        text = speakable(chunk)
        if not text:
            return True
        first = True
        try:
            async for pcm in self.tts.stream(text):
                if generation != self.playout.generation:
                    log("tts", "generation changed mid-reply — dropping audio")
                    return False
                if first:
                    # The reply starts here. The barge-in grace window is
                    # measured from this moment, once per reply.
                    self._utterance_started_at = time.monotonic()
                    if latency_mark:
                        self.call_log.latency(
                            "user_speech_to_first_audio", latency_mark, seq=seq
                        )
                    first = False
                self.playout.push(pcm, generation)
            # Synthesis is done: relay the tail instead of waiting for the
            # buffer to fill.
            if generation == self.playout.generation:
                await self.playout.flush()
            return True
        except TTSAborted:
            log("tts", "aborted by barge-in")
            return False
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error("tts", f"chunk failed: {exc}")
            return True  # keep the conversation alive; skip this clause

    async def _speak_text(self, text: str, tag: str = "say") -> None:
        """Speak a fixed line (greeting, re-prompt, error) outside the LLM loop."""
        generation = self.playout.generation
        self.call_log.event("say", tag=tag, text=text)
        info("say", f"[{tag}] {text}")
        try:
            first = True
            async for pcm in self.tts.stream(speakable(text)):
                if generation != self.playout.generation:
                    return
                if first:
                    self._utterance_started_at = time.monotonic()
                    first = False
                self.playout.push(pcm, generation)
            if generation == self.playout.generation:
                await self.playout.flush()
        except TTSAborted:
            pass
        except Exception as exc:
            error("say", f"failed: {exc}")

    async def _finish_and_hangup(self) -> None:
        info("call", f"hangup requested ({self.tools.hangup_reason})")
        self.call_log.event("hangup_requested", reason=self.tools.hangup_reason)
        # Let the farewell finish playing before we drop the line.
        await self.playout.wait_drained(timeout=25.0)
        await asyncio.sleep(config.GOODBYE_HANGUP_DELAY_S)
        await self._close_socket()

    async def _close_socket(self) -> None:
        try:
            await self.ws.close(code=1000)
        except Exception:
            pass
        self._ended.set()

    # ════════════════════════════════════════════════════════════════════════
    # watchdog: silence re-prompts and the hard call cap
    # ════════════════════════════════════════════════════════════════════════
    async def _watchdog_loop(self) -> None:
        try:
            while not self._ended.is_set():
                await asyncio.sleep(1.0)
                now = time.monotonic()

                if now - self._started_at > config.MAX_CALL_SECONDS:
                    warn("call", "max call duration reached")
                    self.call_log.event("max_duration_reached")
                    await self._speak_text(
                        "जी, समय की वजह से मुझे कॉल यहीं समाप्त करनी होगी। धन्यवाद।",
                        tag="timeout",
                    )
                    await self.playout.wait_drained(timeout=15.0)
                    await self._close_socket()
                    return

                if self._bot_is_speaking():
                    self._last_activity_at = now
                    continue

                idle = now - self._last_activity_at
                if idle < config.SILENCE_REPROMPT_S:
                    continue

                if self._reprompts >= config.MAX_SILENCE_REPROMPTS:
                    info("call", "no response after re-prompts — ending")
                    self.call_log.event("no_response_hangup")
                    await self._speak_text(
                        "जी, शायद आवाज़ नहीं आ रही। कृपया दोबारा कॉल कीजिए। धन्यवाद।",
                        tag="no_response",
                    )
                    await self.playout.wait_drained(timeout=15.0)
                    await self._close_socket()
                    return

                self._reprompts += 1
                self._last_activity_at = now
                self.call_log.event("silence_reprompt", n=self._reprompts)
                await self._speak_text(
                    "जी, क्या आप सुन पा रहे हैं?" if self._reprompts == 1
                    else "जी, कृपया बताइए, मैं आपकी कैसे मदद करूँ?",
                    tag="reprompt",
                )
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            error("call", f"watchdog crashed: {exc}")
