"""
config.py — single source of truth for every tunable in the agent.

Everything is read from the environment once, at import time, so that a
running call never pays for an os.getenv() lookup in the audio path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # python-dotenv is optional
    pass


def _b(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _i(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _s(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _list(name: str, default: str) -> list[str]:
    return [x.strip() for x in _s(name, default).split(",") if x.strip()]


# ─────────────────────────── logging ────────────────────────────────────────
# The master switch you asked for. Everything downstream honours it.
isLogging: bool = _b("IS_LOGGING", True)
IS_LOGGING = isLogging  # snake_case alias so both spellings work

LOG_LEVEL = _s("LOG_LEVEL", "DEBUG" if isLogging else "WARNING")
LOG_DIR = _s("LOG_DIR", "./logs")
LOG_AUDIO_FRAMES = _b("LOG_AUDIO_FRAMES", False)  # very chatty; off by default
LOG_JSONL = _b("LOG_JSONL", True)  # one machine-readable file per call


# ─────────────────────────── telephony (Teler / FreJun) ─────────────────────
TELER_API_KEY = _s("TELER_API_KEY")
TELER_BASE_URL = _s("TELER_BASE_URL", "https://api.frejun.ai/api/v1")
FREJUN_PHONE_NUMBER = _s("FREJUN_PHONE_NUMBER")
PUBLIC_HOST = _s("PUBLIC_HOST")  # e.g. agent.vedronix.com  (no scheme, no slash)
FREJUN_WEBHOOK_SECRET = _s("FREJUN_WEBHOOK_SECRET")
SKIP_SIGNATURE_VERIFICATION = _b("SKIP_SIGNATURE_VERIFICATION", False)

# Browser origins allowed to call this agent. The "Try a call" page on the
# website is a different origin, so without this the preflight fails and the
# page can never reach /call/outbound. Teler itself is server-to-server and
# sends no Origin header, so it is unaffected by this list.
CORS_ORIGINS = _list(
    "CORS_ORIGINS",
    "https://vedronix.com,https://www.vedronix.com,http://localhost:5173,http://localhost:3000",
)

# Guards on the public outbound endpoint. Every request spends real money, and
# the cooldown in the browser is cleared by a refresh, so these are the limits
# that actually hold.
OUTBOUND_NUMBER_COOLDOWN_S = _f("OUTBOUND_NUMBER_COOLDOWN_S", 90.0)
OUTBOUND_IP_WINDOW_S = _f("OUTBOUND_IP_WINDOW_S", 3600.0)
OUTBOUND_IP_MAX = _i("OUTBOUND_IP_MAX", 5)

# Teler streams L16 (signed 16-bit PCM) mono at 8 kHz, and expects the same back.
TELER_SAMPLE_RATE = _i("TELER_SAMPLE_RATE", 8000)
TELER_CHUNK_SIZE = _i("TELER_CHUNK_SIZE", 500)  # matches FreJun's reference bridge
TELER_RECORD = _b("TELER_RECORD", True)
# RFC 2586 defines L16 as network byte order (big-endian), but carriers vary and
# getting it wrong turns the call into static. "auto" measures the inbound
# stream and decides; "big" / "little" force it.
TELER_ENDIAN = _s("TELER_ENDIAN", "auto").lower()
# In auto mode, how long to wait for enough inbound audio to decide before
# speaking the greeting. Keeps the very first words from going out byte-swapped.
ENDIAN_DETECT_TIMEOUT_S = _f("ENDIAN_DETECT_TIMEOUT_S", 1.5)

# Declared in the call flow. FreJun's reference bridge sends the string "8k";
# omitting it entirely lets Teler pick a default that may not match our audio.
TELER_STREAM_SAMPLE_RATE = _s("TELER_STREAM_SAMPLE_RATE", "8k")

# Outbound audio is relayed in buffered chunks, not paced frame by frame —
# Teler runs its own jitter buffer and pacing on top of it starves that buffer.
# Every outbound message is an exact whole number of chunk_size frames — the
# size we declare in the call flow. A ragged tail on each message leaves Teler
# a partial frame to deal with, and a partial frame is a click. This is how
# many frames go in one message (5 x 400 B = 2000 B = 125 ms @ 8 kHz), matching
# the batch-of-5 in FreJun's reference bridge.
PLAYOUT_FRAMES_PER_MESSAGE = _i("PLAYOUT_FRAMES_PER_MESSAGE", 5)
PLAYOUT_FLUSH_BYTES = _i("PLAYOUT_FLUSH_BYTES", 4000)
# Larger cushion for the FIRST flush of an utterance. We synthesise clause by
# clause for latency, which means a short clause can finish playing before the
# next one is ready — an audible gap. Building this much of a head start first
# means the carrier is always playing from a queue (9600 B = 600 ms @ 8 kHz).
PLAYOUT_PRIME_BYTES = _i("PLAYOUT_PRIME_BYTES", 9600)
# ...or once a partial buffer has waited this long, so a short trailing clause
# still goes out promptly.
PLAYOUT_MAX_BUFFER_MS = _i("PLAYOUT_MAX_BUFFER_MS", 120)


# ─────────────────────────── Deepgram (STT) ─────────────────────────────────
DEEPGRAM_API_KEY = _s("DEEPGRAM_API_KEY")
DEEPGRAM_URL = _s("DEEPGRAM_URL", "wss://api.deepgram.com/v1/listen")
# nova-3 + language=multi handles Hinglish code-switching. Use "hi" for pure Hindi.
DEEPGRAM_MODEL = _s("DEEPGRAM_MODEL", "nova-3")
DEEPGRAM_LANGUAGE = _s("DEEPGRAM_LANGUAGE", "multi")
DEEPGRAM_ENDPOINTING_MS = _i("DEEPGRAM_ENDPOINTING_MS", 300)
DEEPGRAM_UTTERANCE_END_MS = _i("DEEPGRAM_UTTERANCE_END_MS", 1000)
DEEPGRAM_KEEPALIVE_S = _f("DEEPGRAM_KEEPALIVE_S", 5.0)
DEEPGRAM_SMART_FORMAT = _b("DEEPGRAM_SMART_FORMAT", True)


# ─────────────────────────── Sarvam (TTS) ───────────────────────────────────
SARVAM_API_KEY = _s("SARVAM_API_KEY")
SARVAM_WS_URL = _s("SARVAM_WS_URL", "wss://api.sarvam.ai/text-to-speech/ws")
SARVAM_MODEL = _s("SARVAM_MODEL", "bulbul:v3")
SARVAM_VOICE = _s("SARVAM_VOICE", "shubh")
SARVAM_LANGUAGE = _s("SARVAM_LANGUAGE", "hi-IN")
SARVAM_SPEED = _f("SARVAM_SPEED", 1.0)  # bulbul:v3 pace range is 0.5 – 2.0
# linear16 @ 8000 Hz drops straight onto the wire with zero resampling.
SARVAM_CODEC = _s("SARVAM_CODEC", "linear16")
SARVAM_SAMPLE_RATE = _i("SARVAM_SAMPLE_RATE", 8000)
SARVAM_MIN_BUFFER = _i("SARVAM_MIN_BUFFER", 50)  # 50 is the value Sarvam documents
SARVAM_MAX_CHUNK_LEN = _i("SARVAM_MAX_CHUNK_LEN", 150)
SARVAM_CONNECT_TIMEOUT_S = _f("SARVAM_CONNECT_TIMEOUT_S", 4.0)
SARVAM_FIRST_CHUNK_TIMEOUT_S = _f("SARVAM_FIRST_CHUNK_TIMEOUT_S", 6.0)
# Silence gap after audio has started that we treat as end-of-utterance. Keeps
# us independent of the completion event.
SARVAM_IDLE_END_S = _f("SARVAM_IDLE_END_S", 2.5)
# Ask for the completion event. ON: without it we can only guess that synthesis
# has finished by waiting for a silence gap, and any audio Sarvam sends after
# that guess stays in the socket and gets read as the beginning of the NEXT
# reply — the caller hears sentences spliced together and truncated. FreJun's
# own reference bridge sets this too.
SARVAM_COMPLETION_EVENT = _b("SARVAM_COMPLETION_EVENT", True)
# Log every JSON frame sent to / received from Sarvam. Turn this on the moment
# you see a 4xx from them — it prints the exact payload they rejected.
SARVAM_WIRE_LOG = _b("SARVAM_WIRE_LOG", False)
# Small pause after the config frame so the server has applied it before text
# arrives. Sending text in the same tick can race the config handler.
SARVAM_CONFIG_SETTLE_S = _f("SARVAM_CONFIG_SETTLE_S", 0.12)


# ─────────────────────────── ElevenLabs (TTS) ───────────────────────────────
# Which TTS vendor drives the call: "elevenlabs" or "sarvam".
TTS_PROVIDER = _s("TTS_PROVIDER", "elevenlabs").lower()

ELEVEN_API_KEY = _s("ELEVENLABS_API_KEY") or _s("ELEVEN_API_KEY")
ELEVEN_WS_BASE = _s("ELEVEN_WS_BASE", "wss://api.elevenlabs.io/v1/text-to-speech")
# Flash v2.5 is their low-latency model (~75 ms) and supports Hindi.
ELEVEN_MODEL = _s("ELEVEN_MODEL", "eleven_flash_v2_5")
# Pick a voice from your Voice Library and put its ID here. The default is the
# public "Alice" voice from their docs — replace it with a male Hindi voice.
ELEVEN_VOICE_ID = _s("ELEVEN_VOICE_ID", "Xb7hH8MSUJpSbSDYk0k2")
ELEVEN_LANGUAGE = _s("ELEVEN_LANGUAGE", "hi")  # ISO-639-1; blank to auto-detect
# ElevenLabs has no pcm_8000 for TTS. ulaw_8000 is their telephony format and is
# exactly Teler's rate, so it needs no resampling — only a G.711 table lookup.
ELEVEN_OUTPUT_FORMAT = _s("ELEVEN_OUTPUT_FORMAT", "ulaw_8000")
ELEVEN_STABILITY = _f("ELEVEN_STABILITY", 0.5)
ELEVEN_SIMILARITY = _f("ELEVEN_SIMILARITY", 0.8)
ELEVEN_SPEED = _f("ELEVEN_SPEED", 1.0)
ELEVEN_CONNECT_TIMEOUT_S = _f("ELEVEN_CONNECT_TIMEOUT_S", 5.0)
ELEVEN_FIRST_CHUNK_TIMEOUT_S = _f("ELEVEN_FIRST_CHUNK_TIMEOUT_S", 6.0)
ELEVEN_IDLE_END_S = _f("ELEVEN_IDLE_END_S", 2.5)
ELEVEN_KEEPALIVE_S = _f("ELEVEN_KEEPALIVE_S", 15.0)
ELEVEN_INACTIVITY_TIMEOUT_S = _i("ELEVEN_INACTIVITY_TIMEOUT_S", 60)
ELEVEN_WIRE_LOG = _b("ELEVEN_WIRE_LOG", False)


def _int_list(name: str, default: list[int]) -> list[int]:
    raw = _s(name)
    if not raw:
        return default
    try:
        return [int(x) for x in raw.replace(" ", "").split(",") if x]
    except ValueError:
        return default


# Characters buffered before they start generating. Lower first value than their
# default [120,160,250,290] so short replies start sooner.
ELEVEN_CHUNK_SCHEDULE = _int_list("ELEVEN_CHUNK_SCHEDULE", [50, 120, 160, 290])


# ─────────────────────────── OpenAI (LLM + fallback TTS) ────────────────────
OPENAI_API_KEY = _s("OPENAI_API_KEY")
OPENAI_BASE_URL = _s("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = _s("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = _f("OPENAI_TEMPERATURE", 0.3)
OPENAI_MAX_TOKENS = _i("OPENAI_MAX_TOKENS", 300)

TTS_FALLBACK_ENABLED = _b("TTS_FALLBACK_ENABLED", True)
OPENAI_TTS_MODEL = _s("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_TTS_VOICE = _s("OPENAI_TTS_VOICE", "nova")
OPENAI_TTS_SAMPLE_RATE = 24000  # OpenAI "pcm" output is always 24 kHz s16le mono


# ─────────────────────────── VAD (Silero) ───────────────────────────────────
# Silero v5 accepts exactly 256 samples at 8 kHz (= 32 ms, = 512 bytes).
VAD_ENABLED = _b("VAD_ENABLED", True)
VAD_FRAME_SAMPLES = 256
VAD_SAMPLE_RATE = 8000

# Probability above which a frame counts as speech.
VAD_THRESHOLD = _f("VAD_THRESHOLD", 0.55)
# While the bot is talking we get line echo + room noise, so we demand more.
VAD_THRESHOLD_WHILE_SPEAKING = _f("VAD_THRESHOLD_WHILE_SPEAKING", 0.72)

# How much *continuous* speech before we believe a human really started.
VAD_MIN_SPEECH_MS = _i("VAD_MIN_SPEECH_MS", 130)
VAD_MIN_SPEECH_MS_WHILE_SPEAKING = _i("VAD_MIN_SPEECH_MS_WHILE_SPEAKING", 260)
# Silence needed before we call the turn finished.
VAD_MIN_SILENCE_MS = _i("VAD_MIN_SILENCE_MS", 500)

# Energy gate: rejects steady background hum that Silero sometimes scores high.
VAD_USE_ENERGY_GATE = _b("VAD_USE_ENERGY_GATE", True)
VAD_ENERGY_MARGIN_DB = _f("VAD_ENERGY_MARGIN_DB", 8.0)  # dB above the noise floor
VAD_NOISE_FLOOR_INIT_DB = _f("VAD_NOISE_FLOOR_INIT_DB", -50.0)
VAD_NOISE_FLOOR_ALPHA = _f("VAD_NOISE_FLOOR_ALPHA", 0.02)  # slow adaptation

# Barge-in also needs the ASR to agree that words were said, not just noise.
BARGE_IN_ENABLED = _b("BARGE_IN_ENABLED", True)
BARGE_IN_REQUIRE_ASR = _b("BARGE_IN_REQUIRE_ASR", True)
BARGE_IN_MIN_CHARS = _i("BARGE_IN_MIN_CHARS", 2)
# Ignore anything in the first moments of our own utterance (tail echo).
BARGE_IN_GRACE_MS = _i("BARGE_IN_GRACE_MS", 350)


# ─────────────────────────── conversation ───────────────────────────────────
GREETING_DELAY_MS = _i("GREETING_DELAY_MS", 250)
MAX_CALL_SECONDS = _i("MAX_CALL_SECONDS", 480)
SILENCE_REPROMPT_S = _f("SILENCE_REPROMPT_S", 8.0)
MAX_SILENCE_REPROMPTS = _i("MAX_SILENCE_REPROMPTS", 2)
GOODBYE_HANGUP_DELAY_S = _f("GOODBYE_HANGUP_DELAY_S", 1.2)


# ─────────────────────────── clinic / booking API ───────────────────────────
CLINIC_ID = _s("CLINIC_ID", "clinic_001")
CLINIC_NAME = _s("CLINIC_NAME", "Capital Hospital, Yamunanagar")
CLINIC_API_BASE = _s("CLINIC_API_BASE", "https://api.vedronix.com/api/v1")
CLINIC_API_KEY = _s("CLINIC_001_API_KEY") or _s("APPOINTMENT_API_KEY")
CLINIC_API_SECRET = _s("CLINIC_001_API_SECRET") or _s("APPOINTMENT_API_SECRET")
CLINIC_API_TIMEOUT_S = _f("CLINIC_API_TIMEOUT_S", 6.0)
CLINIC_API_DRY_RUN = _b("CLINIC_API_DRY_RUN", False)

# WhatsApp
WHATSAPP_API_KEY = os.getenv("WHATSAPP_API_KEY", "")
WHATSAPP_API_URL = os.getenv(
    "WHATSAPP_API_URL",
    "https://whatsappev.vedronix.com/api/v1/messages/text",
)
WHATSAPP_ENABLED = os.getenv("WHATSAPP_ENABLED", "true").lower() == "true"
WHATSAPP_DEFAULT_COUNTRY_CODE = os.getenv("WHATSAPP_DEFAULT_COUNTRY_CODE", "91")

# Queue
QUEUE_MAX_SIZE = int(os.getenv("QUEUE_MAX_SIZE", "10000"))
QUEUE_WORKERS = int(os.getenv("QUEUE_WORKERS", "3"))
QUEUE_MAX_RETRIES = int(os.getenv("QUEUE_MAX_RETRIES", "3"))
QUEUE_RETRY_BACKOFF = float(os.getenv("QUEUE_RETRY_BACKOFF", "2.0"))
# config.py
DEBUG_ENDPOINTS_ENABLED = os.getenv(
    "DEBUG_ENDPOINTS_ENABLED", "true"
).lower() == "true"

@dataclass(frozen=True)
class Doctor:
    name: str = _s("DOCTOR_NAME", "Dr. Vikash Sharma")
    speciality: str = _s("DOCTOR_SPECIALITY", "Physiotherapist at Neuranta")
    address: str = _s(
        "CLINIC_ADDRESS",
        "Neuranta — Neuro & Pediatric Rehabilitation Centre in Gurgaon, Haryana",
    )
    morning: str = _s("OPD_MORNING", "सुबह नौ बजे से दोपहर दो बजे तक")
    evening: str = _s("OPD_EVENING", "शाम तीन बजे से शाम पाँच बजे तक")


DOCTOR = Doctor()


@dataclass(frozen=True)
class Missing:
    keys: list[str] = field(default_factory=list)


def missing_required() -> list[str]:
    """Names of env vars that must be set for the agent to actually work."""
    out = []
    for name, val in (
        ("OPENAI_API_KEY", OPENAI_API_KEY),
        ("DEEPGRAM_API_KEY", DEEPGRAM_API_KEY),
        (
            "ELEVENLABS_API_KEY" if TTS_PROVIDER == "elevenlabs" else "SARVAM_API_KEY",
            ELEVEN_API_KEY if TTS_PROVIDER == "elevenlabs" else SARVAM_API_KEY,
        ),
        ("TELER_API_KEY", TELER_API_KEY),
        ("PUBLIC_HOST", PUBLIC_HOST),
    ):
        if not val:
            out.append(name)
    return out