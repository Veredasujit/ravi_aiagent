"""
whatsapp.py — thin wrapper around the Vedronix WhatsApp text API.
"""

from __future__ import annotations

import httpx

from . import config
from .booking_state import BookingRecord
from .logging_setup import error, info, warn
from .message_queue import MessageKind, OutboundMessage


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------
def _norm_phone(number: str) -> str:
    """Strip non-digits; optionally prefix with default country code."""
    digits = "".join(ch for ch in number if ch.isdigit())
    if not digits:
        return number
    cc = config.WHATSAPP_DEFAULT_COUNTRY_CODE
    # If caller already passed a country code (>=11 digits for IN) keep it.
    if len(digits) <= 10 and cc:
        digits = cc + digits
    return digits


def render_booking_confirmed(rec: BookingRecord) -> str:
    lines = ["Thank you for calling Capital Hospital."]
    lines.append("Your appointment has been successfully booked. ✅")
    if rec.booking_id:
        lines.append(f"Booking ID: {rec.booking_id}")
    if rec.patient_name:
        lines.append(f"Patient: {rec.patient_name}")
    if rec.doctor:
        lines.append(f"Doctor: {rec.doctor}")
    if rec.department:
        lines.append(f"Department: {rec.department}")
    if rec.appointment_time:
        lines.append(f"When: {rec.appointment_time}")
    lines.append("We look forward to seeing you. Reply here if you need to reschedule.")
    return "\n".join(lines)


def render_followup_no_booking(rec: BookingRecord) -> str:
    lines = ["Thank you for calling Capital Hospital. 🙏"]
    lines.append(
        "If you would like to book an appointment or need any further help, "
        "please reply to this message. We'll be happy to assist you."
    )
    return "\n".join(lines)


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

    payload = {"number": _norm_phone(number), "text": text, "delay": delay}
    headers = {
        "X-API-Key": config.WHATSAPP_API_KEY,
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(config.WHATSAPP_API_URL, json=payload, headers=headers)

    if resp.status_code >= 400:
        raise RuntimeError(f"whatsapp {resp.status_code}: {resp.text[:300]}")

    try:
        return resp.json()
    except ValueError:
        return {"raw": resp.text}


# ---------------------------------------------------------------------------
# Adapter for MessageQueue
# ---------------------------------------------------------------------------
async def send_outbound_message(msg: OutboundMessage) -> dict:
    """
    Called by MessageQueue workers. Raises on failure so retry logic kicks in.
    """
    delay = int(msg.metadata.get("delay", 1500))
    info("whatsapp", f"sending {msg.kind.value} → {msg.phone_number}")
    try:
        result = await _post_text(msg.phone_number, msg.text, delay=delay)
        return result
    except Exception as exc:
        error("whatsapp", f"send failed to {msg.phone_number}: {exc}")
        raise