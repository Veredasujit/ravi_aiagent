"""
selftest.py — prove each leg of the pipeline works *before* you burn a phone call.

    python -m tools.selftest

Order matters here. TTS runs before VAD, because the only honest way to test a
speech detector is with real speech. Silero is trained on voice, so a synthetic
sine wave is correctly rejected — testing it with a tone tells you nothing.
Instead we take the Hindi line Sarvam just synthesised and feed that through the
VAD, which exercises the exact path a live call uses.

Checks:
  1. env            — which required keys are missing
  2. Sarvam TTS     — negotiates a codec, writes selftest_sarvam.wav @ 8 kHz
  3. OpenAI TTS     — fallback path, writes selftest_openai.wav
  4. Silero VAD     — must fire on the real speech above, and stay silent on silence
  5. Deepgram       — websocket opens and accepts audio
  6. OpenAI LLM     — streams a completion, reports time-to-first-token
  7. Teler          — credentials accepted (does NOT place a call)

Then listen:  afplay selftest_sarvam.wav
"""

from __future__ import annotations

import asyncio
import sys
import time
import wave

import numpy as np

from app import config
from app.audio import float_to_pcm
from app.logging_setup import CallLog, setup_logging

OK = "\033[92m  OK \033[0m"
BAD = "\033[91m FAIL\033[0m"
SKIP = "\033[93m SKIP\033[0m"

_results: list[tuple[str, bool]] = []
_speech_pcm: bytes = b""  # real synthesised speech, 8 kHz — used by the VAD check


def report(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok))
    print(f"[{OK if ok else BAD}] {name}" + (f" — {detail}" if detail else ""))


def skip(name: str, detail: str) -> None:
    print(f"[{SKIP}] {name} — {detail}")


def write_wav(path: str, pcm: bytes, rate: int) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)


def quiet(ms: int, rate: int = 8000) -> bytes:
    """Low-level room tone, not digital silence — closer to a real line."""
    n = int(rate * ms / 1000)
    rng = np.random.default_rng(7)
    return float_to_pcm((0.002 * rng.standard_normal(n)).astype(np.float32))


# ── 1. env ──────────────────────────────────────────────────────────────────
def check_env() -> None:
    missing = config.missing_required()
    report(
        "env",
        not missing,
        "all keys present" if not missing else f"missing: {', '.join(missing)}",
    )


# ── 2. Sarvam TTS ───────────────────────────────────────────────────────────
async def check_sarvam() -> None:
    global _speech_pcm
    if not config.SARVAM_API_KEY:
        skip("sarvam tts", "SARVAM_API_KEY not set")
        return
    from app.tts import SarvamTTS

    line = "नमस्ते जी, मैं रवि बोल रहा हूँ, कैपिटल हॉस्पिटल यमुनानगर से।"
    tts = SarvamTTS(CallLog("selftest"))
    try:
        # prewarm() walks the codec ladder and proves one of them actually works
        if not await tts.prewarm():
            report("sarvam tts", False,
                   "every codec rejected — rerun with SARVAM_WIRE_LOG=true to "
                   "see the exact payload Sarvam refused")
            return

        t0 = time.monotonic()
        pcm = b""
        ttfb = None
        async for chunk in tts.stream(line):
            if ttfb is None:
                ttfb = round((time.monotonic() - t0) * 1000)
            pcm += chunk
        if not pcm:
            report("sarvam tts", False, "negotiated a codec but produced no audio")
            return

        _speech_pcm = pcm
        write_wav("selftest_sarvam.wav", pcm, config.TELER_SAMPLE_RATE)
        secs = len(pcm) / 2 / config.TELER_SAMPLE_RATE
        report(
            "sarvam tts", True,
            f"codec={tts.codec}@{tts.sample_rate}, TTFB {ttfb} ms, {secs:.2f}s "
            f"-> selftest_sarvam.wav  (LISTEN TO THIS)",
        )
    except Exception as exc:
        report("sarvam tts", False, str(exc))
    finally:
        await tts.close()


# ── 3. OpenAI TTS fallback ──────────────────────────────────────────────────
async def check_openai_tts() -> None:
    global _speech_pcm
    if not config.TTS_FALLBACK_ENABLED:
        skip("openai tts", "TTS_FALLBACK_ENABLED=false")
        return
    if not config.OPENAI_API_KEY:
        skip("openai tts", "OPENAI_API_KEY not set")
        return
    from app.tts import OpenAITTS

    t = OpenAITTS(CallLog("selftest"))
    try:
        pcm = b""
        async for chunk in t.stream("नमस्ते, यह एक टेस्ट है। मैं रवि बोल रहा हूँ।"):
            pcm += chunk
        if pcm:
            write_wav("selftest_openai.wav", pcm, config.TELER_SAMPLE_RATE)
            if not _speech_pcm:  # VAD check needs *some* real speech
                _speech_pcm = pcm
        report(
            "openai tts", bool(pcm),
            f"{len(pcm)/2/config.TELER_SAMPLE_RATE:.2f}s -> selftest_openai.wav"
            if pcm else "no audio returned",
        )
    except Exception as exc:
        report("openai tts", False, str(exc))
    finally:
        await t.close()


# ── 4. Silero VAD, tested on real speech ────────────────────────────────────
def check_vad() -> None:
    if not config.VAD_ENABLED:
        skip("silero vad", "VAD_ENABLED=false")
        return
    from app.vad import SileroVAD, VadEvent

    v = SileroVAD()
    if v._model is None:
        report("silero vad", False,
               "model not loaded — pip install silero-vad onnxruntime torch")
        return

    if not _speech_pcm:
        skip("silero vad", "model loaded, but no TTS audio to test it against")
        return

    # Settle the noise floor on room tone, then play real speech through it.
    for _ in v.feed(quiet(600)):
        pass
    t0 = time.monotonic()
    events = [e.name for e, _ in v.feed(_speech_pcm)]
    # Then silence, which should close the utterance.
    events += [e.name for e, _ in v.feed(quiet(900))]
    ms = round((time.monotonic() - t0) * 1000)

    started = "SPEECH_START" in events
    ended = "SPEECH_END" in events
    stats = v.stats()
    detail = (
        f"events={events}, speech frames {stats['frames_speech']}/"
        f"{stats['frames_seen']}, floor {stats['noise_floor_db']} dBFS, "
        f"{ms} ms to process {len(_speech_pcm)/2/8000:.1f}s"
    )
    if started and ended:
        report("silero vad", True, detail)
    elif started:
        report("silero vad", False, "fired on speech but never closed — " + detail)
    else:
        report("silero vad", False,
               "did not fire on real speech; lower VAD_THRESHOLD — " + detail)

    # False-positive check: pure room tone must produce nothing at all.
    v2 = SileroVAD()
    noise_events = [e.name for e, _ in v2.feed(quiet(2000))]
    report("silero vad (noise rejection)", not noise_events,
           "2s of room tone produced no events" if not noise_events
           else f"room tone triggered {noise_events} — raise VAD_THRESHOLD")


# ── 5. Deepgram ─────────────────────────────────────────────────────────────
async def check_deepgram() -> None:
    if not config.DEEPGRAM_API_KEY:
        skip("deepgram", "DEEPGRAM_API_KEY not set")
        return
    from app.stt_deepgram import DeepgramSTT

    seen: list[str] = []

    async def on_tx(text: str, is_final: bool, speech_final: bool) -> None:
        seen.append(text)

    stt = DeepgramSTT(CallLog("selftest"), on_transcript=on_tx)
    try:
        t0 = time.monotonic()
        await stt.start()
        ok = stt.ready.is_set()
        ms = round((time.monotonic() - t0) * 1000)
        if ok and _speech_pcm:
            # Feed the real Hindi speech in call-sized frames.
            for i in range(0, len(_speech_pcm), 800):
                await stt.send_audio(_speech_pcm[i : i + 800])
                await asyncio.sleep(0.02)
            await stt.finalize()
            await asyncio.sleep(1.5)
        detail = f"connected in {ms} ms, {config.DEEPGRAM_MODEL}/{config.DEEPGRAM_LANGUAGE}"
        if seen:
            detail += f", transcribed: {seen[-1][:60]!r}"
        elif _speech_pcm:
            detail += ", but returned no transcript for the TTS sample"
        report("deepgram", ok, detail)
    except Exception as exc:
        report("deepgram", False, str(exc))
    finally:
        await stt.close()


# ── 6. OpenAI LLM ───────────────────────────────────────────────────────────
async def check_llm() -> None:
    if not config.OPENAI_API_KEY:
        skip("openai llm", "OPENAI_API_KEY not set")
        return
    from app.llm import LLMClient
    from app.tools import TOOLS_SCHEMA

    llm = LLMClient(CallLog("selftest"))
    try:
        msgs = [
            {"role": "system", "content": "एक छोटे हिंदी वाक्य में जवाब दें।"},
            {"role": "user", "content": "नमस्ते"},
        ]
        # Two runs: the first pays for TLS setup, the second is the number that
        # matters on a live call, where the session is already warm.
        timings = []
        text = ""
        for _ in range(2):
            t0 = time.monotonic()
            ttft = None
            async for kind, payload in llm.stream(msgs, TOOLS_SCHEMA):
                if kind == "token" and ttft is None:
                    ttft = round((time.monotonic() - t0) * 1000)
                if kind == "done":
                    text = payload
            timings.append(ttft)
        warm = timings[-1]
        note = ""
        if warm and warm > 1500:
            note = "  <-- high; check network latency to OpenAI"
        report("openai llm", bool(text),
               f"TTFT cold {timings[0]} ms / warm {warm} ms, "
               f"model={config.OPENAI_MODEL}, reply={text[:40]!r}{note}")
    except Exception as exc:
        report("openai llm", False, str(exc))
    finally:
        await llm.close()


# ── 7. Teler credentials ────────────────────────────────────────────────────
async def check_teler() -> None:
    if not config.TELER_API_KEY:
        skip("teler", "TELER_API_KEY not set")
        return
    import httpx

    # Deliberately empty body: we only want to know whether the key is accepted.
    # Any 4xx that is not 401/403 means auth passed and validation rejected us.
    url = f"{config.TELER_BASE_URL}/voice/calls/initiate"
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(url, json={}, headers={"X-Api-Key": config.TELER_API_KEY})
        if r.status_code in (401, 403):
            report("teler", False, f"credentials rejected ({r.status_code})")
        else:
            report("teler", True,
                   f"key accepted (validation response {r.status_code}) — no call placed")
    except Exception as exc:
        report("teler", False, str(exc))


async def main() -> int:
    setup_logging()
    print("\n── Ravi voice agent self-test ──────────────────────────────\n")
    host = config.PUBLIC_HOST or "(unset)"
    print(f"public host : {host}")
    if config.PUBLIC_HOST:
        print(f"ws url      : wss://{config.PUBLIC_HOST}/media-stream")
    if host.startswith("your-subdomain"):
        print("  WARNING: PUBLIC_HOST is still the placeholder from .env.example")
    print()

    check_env()
    await check_sarvam()
    await check_openai_tts()
    check_vad()
    await check_deepgram()
    await check_llm()
    await check_teler()

    failed = [n for n, ok in _results if not ok]
    print("\n────────────────────────────────────────────────────────────")
    if failed:
        print(f"{len(failed)} check(s) failed: {', '.join(failed)}")
        return 1
    print("All checks passed.  Now:  afplay selftest_sarvam.wav")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
