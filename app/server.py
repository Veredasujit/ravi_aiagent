"""
server.py — the HTTP/WebSocket surface Teler talks to.

POST /flow            Teler asks "what should this call do?" -> stream flow
POST /webhook         call lifecycle events (ringing, answered, completed)
WS   /media-stream    the actual audio, bidirectional
POST /call/outbound   you trigger an outbound call
GET  /health          readiness + config sanity
GET  /queue/stats     message queue observability

Inbound and outbound use the *same* flow and the same media-stream handler.
The only difference is who dialled: for outbound we call Teler's initiate
endpoint first, and Teler then fetches /flow exactly as it does for inbound.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any, Optional

import httpx
from fastapi import Body, FastAPI, Header, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import config
from .booking_state import BookingStatus, booking_registry
from .logging_setup import (
    error,
    info,
    log,
    new_call_id,
    set_call_id,
    setup_logging,
    warn,
)
from .messaging_service import messaging_service
from .session import CallSession

setup_logging()

app = FastAPI(title="Ravi — Capital Hospital voice agent", version="1.1.0")


# ═══════════════════════════════════════════════════════════════════════════
# startup / shutdown
# ═══════════════════════════════════════════════════════════════════════════
@app.on_event("startup")
async def _startup() -> None:
    info("boot", f"isLogging={config.isLogging} level={config.LOG_LEVEL}")
    info(
        "boot",
        f"stt={config.DEEPGRAM_MODEL}/{config.DEEPGRAM_LANGUAGE} "
        + (
            f"tts=elevenlabs:{config.ELEVEN_MODEL}/{config.ELEVEN_VOICE_ID}"
            f"@{config.ELEVEN_OUTPUT_FORMAT} "
            if config.TTS_PROVIDER == "elevenlabs"
            else f"tts=sarvam:{config.SARVAM_MODEL}/{config.SARVAM_VOICE}"
            f"@{config.SARVAM_SAMPLE_RATE} "
        )
        + f"llm={config.OPENAI_MODEL}",
    )
    missing = config.missing_required()
    if missing:
        warn("boot", f"MISSING ENV: {', '.join(missing)} — calls will fail")
    else:
        info("boot", "all required env vars present")

    if config.VAD_ENABLED:
        from .vad import _get_model

        _get_model()  # load Silero now so the first call isn't slowed by it

    # Start WhatsApp message queue workers
    await messaging_service.start()


@app.on_event("shutdown")
async def _shutdown() -> None:
    info("boot", "shutting down…")
    await messaging_service.stop()
    # Best-effort: flush stale booking records (>24h)
    removed = booking_registry.cleanup_older_than(24 * 3600)
    if removed:
        info("boot", f"cleaned {removed} stale booking records")


# ═══════════════════════════════════════════════════════════════════════════
# Teler call flow
# ═══════════════════════════════════════════════════════════════════════════
def _stream_flow() -> dict:
    return {
        "action": "stream",
        "ws_url": f"wss://{config.PUBLIC_HOST}/media-stream",
        "sample_rate": config.TELER_STREAM_SAMPLE_RATE,
        "chunk_size": config.TELER_CHUNK_SIZE,
        "record": config.TELER_RECORD,
    }


@app.post("/flow")
async def flow(payload: dict = Body(default={})) -> JSONResponse:
    info("flow", f"flow requested: {json.dumps(payload, ensure_ascii=False)[:300]}")
    if not config.PUBLIC_HOST:
        error("flow", "PUBLIC_HOST is not set — Teler cannot reach the websocket")
        raise HTTPException(500, "PUBLIC_HOST not configured")
    f = _stream_flow()
    log("flow", f"returning {f}")
    return JSONResponse(f)


# ═══════════════════════════════════════════════════════════════════════════
# Teler status webhook — the *reliable* "call completed" trigger
# ═══════════════════════════════════════════════════════════════════════════
def _verify_signature(body: bytes, signature: Optional[str]) -> bool:
    if config.SKIP_SIGNATURE_VERIFICATION:
        return True
    if not config.FREJUN_WEBHOOK_SECRET:
        warn("webhook", "no webhook secret configured; accepting unverified")
        return True
    if not signature:
        return False
    expected = hmac.new(
        config.FREJUN_WEBHOOK_SECRET.encode(), body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature.strip().lower())


def _extract_call_id(data: dict) -> Optional[str]:
    for key in ("call_id", "callId", "call_sid", "CallSid", "id", "uuid"):
        v = data.get(key)
        if v:
            return str(v)
    nested = data.get("data") or data.get("call") or {}
    if isinstance(nested, dict):
        for key in ("call_id", "callId", "call_sid", "id"):
            v = nested.get(key)
            if v:
                return str(v)
    return None


def _extract_status(data: dict) -> Optional[str]:
    for key in ("status", "event", "type", "call_status"):
        v = data.get(key)
        if v:
            return str(v).lower()
    nested = data.get("data") or data.get("call") or {}
    if isinstance(nested, dict):
        for key in ("status", "event", "type"):
            v = nested.get(key)
            if v:
                return str(v).lower()
    return None


def _extract_phone(data: dict) -> Optional[str]:
    for key in ("to_number", "to", "caller", "customer_number", "phone"):
        v = data.get(key)
        if v:
            return str(v)
    nested = data.get("data") or data.get("call") or {}
    if isinstance(nested, dict):
        for key in ("to_number", "to", "caller", "phone"):
            v = nested.get(key)
            if v:
                return str(v)
    return None


COMPLETED_EVENTS = {
    "completed",
    "call.completed",
    "call_completed",
    "hangup",
    "call.ended",
    "ended",
    "call_end",
}


@app.post("/webhook")
async def webhook(
    request: Request,
    x_teler_signature: Optional[str] = Header(default=None, alias="X-Teler-Signature"),
    x_signature: Optional[str] = Header(default=None, alias="X-Signature"),
) -> JSONResponse:
    body = await request.body()
    signature = x_teler_signature or x_signature
    if not _verify_signature(body, signature):
        warn("webhook", "signature verification failed")
        raise HTTPException(401, "invalid signature")

    try:
        data: Any = json.loads(body or b"{}")
    except ValueError:
        data = {"raw": body.decode("utf-8", "ignore")}

    info("webhook", json.dumps(data, ensure_ascii=False)[:600])

    if isinstance(data, dict):
        event = _extract_status(data)
        call_id = _extract_call_id(data)
        phone = _extract_phone(data)

        if event in COMPLETED_EVENTS:
            if not call_id:
                warn("webhook", "completed event with no call_id — cannot dispatch")
            else:
                # Make sure we have a phone number on record
                rec = booking_registry.get(call_id)
                if rec and phone and not rec.phone_number:
                    rec.phone_number = phone

                # Decide + enqueue the appropriate WhatsApp message
                asyncio_ok = await messaging_service.enqueue_for_completed_call(
                    call_id, phone_number=phone
                )
                if asyncio_ok:
                    info("webhook", f"queued WhatsApp for call {call_id}")

    return JSONResponse({"received": True})


# ═══════════════════════════════════════════════════════════════════════════
# media stream
# ═══════════════════════════════════════════════════════════════════════════
@app.websocket("/media-stream")
async def media_stream(ws: WebSocket) -> None:
    await ws.accept()
    call_id = new_call_id()
    set_call_id(call_id)
    info("ws", "media stream accepted")

    session = CallSession(ws, call_id=call_id)
    try:
        await session.run()
    except Exception as exc:
        error("ws", f"session crashed: {exc}")
        await session.shutdown()
    finally:
        try:
            await ws.close()
        except Exception:
            pass
        info("ws", "media stream closed")


# ═══════════════════════════════════════════════════════════════════════════
# outbound calling
# ═══════════════════════════════════════════════════════════════════════════
class OutboundRequest(BaseModel):
    to_number: str
    from_number: Optional[str] = None
    record: Optional[bool] = None


@app.post("/call/outbound")
async def call_outbound(req: OutboundRequest) -> JSONResponse:
    if not config.TELER_API_KEY:
        raise HTTPException(500, "TELER_API_KEY not configured")
    if not config.PUBLIC_HOST:
        raise HTTPException(500, "PUBLIC_HOST not configured")

    from_number = req.from_number or config.FREJUN_PHONE_NUMBER
    if not from_number:
        raise HTTPException(400, "no from_number and FREJUN_PHONE_NUMBER is unset")

    payload = {
        "from_number": from_number,
        "to_number": req.to_number,
        "flow_url": f"https://{config.PUBLIC_HOST}/flow",
        "status_callback_url": f"https://{config.PUBLIC_HOST}/webhook",
        "record": config.TELER_RECORD if req.record is None else req.record,
    }
    url = f"{config.TELER_BASE_URL}/voice/calls/initiate"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Api-Key": config.TELER_API_KEY,
    }

    info("outbound", f"dialling {req.to_number} from {from_number}")
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(url, json=payload, headers=headers)
    except Exception as exc:
        error("outbound", f"request failed: {exc}")
        raise HTTPException(502, f"teler unreachable: {exc}")

    log("outbound", f"teler {resp.status_code}: {resp.text[:400]}")
    if resp.status_code >= 400:
        raise HTTPException(resp.status_code, resp.text[:400])

    try:
        data = resp.json()
    except ValueError:
        data = {"raw": resp.text}
    info("outbound", f"call created: {json.dumps(data, ensure_ascii=False)[:300]}")
    return JSONResponse({"success": True, "teler": data})


# ═══════════════════════════════════════════════════════════════════════════
# health + observability
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/health")
async def health() -> JSONResponse:
    from .vad import _get_model

    return JSONResponse(
        {
            "ok": not config.missing_required(),
            "missing_env": config.missing_required(),
            "isLogging": config.isLogging,
            "public_host": config.PUBLIC_HOST or None,
            "ws_url": f"wss://{config.PUBLIC_HOST}/media-stream"
            if config.PUBLIC_HOST
            else None,
            "stt": {
                "model": config.DEEPGRAM_MODEL,
                "language": config.DEEPGRAM_LANGUAGE,
                "sample_rate": config.TELER_SAMPLE_RATE,
            },
            "tts": {
                "provider": config.TTS_PROVIDER,
                "model": config.ELEVEN_MODEL
                if config.TTS_PROVIDER == "elevenlabs"
                else config.SARVAM_MODEL,
                "voice": config.ELEVEN_VOICE_ID
                if config.TTS_PROVIDER == "elevenlabs"
                else config.SARVAM_VOICE,
                "codec": config.ELEVEN_OUTPUT_FORMAT
                if config.TTS_PROVIDER == "elevenlabs"
                else config.SARVAM_CODEC,
                "fallback": config.OPENAI_TTS_MODEL
                if config.TTS_FALLBACK_ENABLED
                else None,
            },
            "llm": config.OPENAI_MODEL,
            "vad": {
                "enabled": config.VAD_ENABLED,
                "model_loaded": _get_model() is not None,
                "threshold": config.VAD_THRESHOLD,
                "threshold_while_speaking": config.VAD_THRESHOLD_WHILE_SPEAKING,
                "barge_in": config.BARGE_IN_ENABLED,
                "barge_in_requires_asr": config.BARGE_IN_REQUIRE_ASR,
            },
            "whatsapp": {
                "enabled": config.WHATSAPP_ENABLED,
                "api_url": config.WHATSAPP_API_URL,
                "api_key_set": bool(config.WHATSAPP_API_KEY),
            },
            "queue": messaging_service.stats(),
        }
    )


@app.get("/queue/stats")
async def queue_stats() -> JSONResponse:
    return JSONResponse(messaging_service.stats())


@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse({"agent": "Ravi", "clinic": config.CLINIC_NAME})