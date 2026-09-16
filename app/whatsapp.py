"""
whatsapp.py — thin wrapper around the Vedronix WhatsApp text API.

IMPORTANT: Vedronix's upstream (Evolution API) REJECTS non-ASCII text in
the `text` field. All messages must be ASCII-only. We sanitize with
_ascii_safe() before sending.

Confirmed working payload (via curl):
    {
      "number": "919934601244",   ← 12 digits, country code, NO +
      "text":   "plain ASCII string",
      "delay":  1500
    }
"""

from __future__ import annotations

import json
from typing import Optional

import httpx

from . import config
from .booking_state import BookingRecord
from .logging_setup import error, info, log, warn
from .message_queue import MessageKind, OutboundMessage


# ---------------------------------------------------------------------------
# Phone normalisation — Vedronix wants: 12 digits, country code, NO +.
# ---------------------------------------------------------------------------
def _norm_phone(number: str) -> str:
    digits = "".join(ch for ch in number if ch.isdigit())
    if not digits:
        return number
    cc = config.WHATSAPP_DEFAULT_COUNTRY_CODE or "91"
    if len(digits) <= 10:
        digits = cc + digits
    return digits


# ---------------------------------------------------------------------------
# ASCII sanitizer — Vedronix rejects non-ASCII text
# ---------------------------------------------------------------------------
_HINDI_TO_EN = {
    # Time words
    "सुबह": "morning",
    "शाम": "evening",
    "दोपहर": "afternoon",
    "रात": "night",
    "बजे": "o'clock",
    "मिनट": "minutes",
    # Numbers
    "एक": "1",
    "दो": "2",
    "तीन": "3",
    "चार": "4",
    "पांच": "5",
    "पाँच": "5",
    "छह": "6",
    "सात": "7",
    "आठ": "8",
    "नौ": "9",
    "दस": "10",
    "ग्यारह": "11",
    "बारह": "12",
    # Prepositions
    "से": "to",
    "तक": "until",
    # Common booking words
    "डॉक्टर": "doctor",
    "अपॉइंटमेंट": "appointment",
    "बुकिंग": "booking",
    "मरीज": "patient",
    "नाम": "name",
    "समय": "time",
}


def _ascii_safe(text: str) -> str:
    """
    Vedronix upstream rejects any non-ASCII character in `text`.
    Translate common Devanagari words to English, then drop anything
    that's still non-ASCII. Guarantees the payload passes.
    """
    if not text:
        return text

    result = text
    for hindi, english in _HINDI_TO_EN.items():
        result = result.replace(hindi, english)

    # Drop any remaining non-ASCII (emoji, other scripts, smart quotes)
    result = result.encode("ascii", "ignore").decode("ascii")

    # Collapse double spaces left behind
    while "  " in result:
        result = result.replace("  ", " ")
    return result.strip()


# ---------------------------------------------------------------------------
# Templates — ASCII-only, no emoji, no non-English text
# ---------------------------------------------------------------------------
def render_booking_confirmed(rec: BookingRecord) -> str:
    parts = ["Thank you for calling Capital Hospital."]
    parts.append("Your appointment has been successfully booked.")
    if rec.booking_id:
        parts.append(f"Booking ID: {rec.booking_id}")
    if rec.patient_name:
        parts.append(f"Patient: {rec.patient_name}")
    if rec.doctor:
        parts.append(f"Doctor: {rec.doctor}")
    if rec.department:
        parts.append(f"Department: {rec.department}")
    if rec.appointment_time:
        parts.append(f"When: {rec.appointment_time}")
    parts.append("Reply here if you need to reschedule.")
    return "\n".join(parts)


def render_followup_no_booking(rec: BookingRecord) -> str:
    parts = ["Thank you for calling Capital Hospital."]
    parts.append(
        "If you would like to book an appointment or need any further help, "
        "please reply to this message. We will be happy to assist you."
    )
    return "\n".join(parts)


def render_generic_call_ended(rec: BookingRecord) -> str:
    return (
        "Thank you for calling Capital Hospital. "
        "If you need anything, just reply to this message."
    )


def build_message_for(rec: BookingRecord, kind: MessageKind) -> str:
    if kind == MessageKind.BOOKING_CONFIRMED:
        return render_booking_confirmed(rec)
    if kind == MessageKind.FOLLOWUP_NO_BOOKING:
        return render_followup_no_booking(rec)
    return render_generic_call_ended(rec)


# ---------------------------------------------------------------------------
# Low-level send
# ---------------------------------------------------------------------------
async def _post_text(number: str, text: str, delay: int = 1500) -> dict:
    if not config.WHATSAPP_ENABLED:
        warn("whatsapp", f"disabled — would have sent to {number}: {text[:60]}…")
        return {"skipped": True}

    if not config.WHATSAPP_API_KEY:
        raise RuntimeError("WHATSAPP_API_KEY not configured")

    # Vedronix upstream rejects non-ASCII. Sanitize EVERYTHING here.
    original_len = len(text)
    text = _ascii_safe(text)
    if len(text) != original_len:
        log(
            "whatsapp",
            f"ascii_safe: {original_len} -> {len(text)} chars "
            f"(removed {original_len - len(text)} non-ASCII)",
        )

    norm = _norm_phone(number)

    log(
        "whatsapp",
        f"DIAG input={number!r} normalised={norm!r} "
        f"text_len={len(text)} text_preview={text[:120]!r}",
    )

    headers = {
        "X-API-Key": config.WHATSAPP_API_KEY,
        "Content-Type": "application/json",
    }

    # Single confirmed working shape: flat text, 12-digit number, no +.
    payload = {"number": norm, "text": text, "delay": delay}

    try:
        payload_json = json.dumps(payload, ensure_ascii=False)
    except Exception:
        payload_json = str(payload)
    log("whatsapp", f"POST payload={payload_json[:400]}")

    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            config.WHATSAPP_API_URL, json=payload, headers=headers
        )

    body_preview = resp.text[:500]
    log(
        "whatsapp",
        f"RESP status={resp.status_code} body={body_preview}",
    )

    if resp.status_code in (401, 403):
        raise RuntimeError(
            f"whatsapp auth error ({resp.status_code}): {body_preview}"
        )
    if resp.status_code >= 400:
        raise RuntimeError(
            f"whatsapp http {resp.status_code}: {body_preview}"
        )

    try:
        data = resp.json()
    except ValueError:
        data = {"raw": resp.text}

    if isinstance(data, dict):
        ok = data.get("ok")
        status = str(data.get("status", "")).lower()
        if ok is False or status in ("error", "failed", "failure"):
            raise RuntimeError(f"whatsapp api error: {body_preview}")

    return data


# ---------------------------------------------------------------------------
# Adapter for MessageQueue
# ---------------------------------------------------------------------------
async def send_outbound_message(msg: OutboundMessage) -> dict:
    delay = int(msg.metadata.get("delay", 1500))
    info(
        "whatsapp",
        f"sending {msg.kind.value} → {msg.phone_number} "
        f"(call_id={msg.call_id})",
    )
    try:
        return await _post_text(msg.phone_number, msg.text, delay=delay)
    except Exception as exc:
        error("whatsapp", f"send failed to {msg.phone_number}: {exc}")
        raise