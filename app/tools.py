"""
tools.py — the functions the LLM can call.

`create_data`            books the appointment against the clinic API.
`end_call`               ends the call after the farewell has been spoken.
`set_caller_phone`       records the caller's phone number when stated verbally.
`mark_not_booked`        records that the caller declined to book.

All are fully logged: arguments in, HTTP status and body out, wall time.
If the clinic API is unreachable the tool returns a structured failure rather
than raising, so Ravi can apologise in Hindi instead of the call going silent.

BOOKING → WHATSAPP WIRING
─────────────────────────
The session owns the booking state (`booking_registry` keyed by call_id).
ToolRunner reaches back into the session via `self._session` (bound in
`CallSession.__init__`). When `create_data` succeeds, it calls
`session.on_booking_confirmed(...)` so the session records it.

The actual WhatsApp message is NOT sent here. It is enqueued by `/webhook`
when Teler reports the call as `completed` — the only reliable "call is
really over" signal.

ASCII-SAFE BOOKING DATA
───────────────────────
Vedronix's upstream (Evolution API) rejects non-ASCII characters in the
WhatsApp `text` field. So the booking confirmation stores an English
`symptom_en` alongside the original Hindi `symptom`, and the appointment
time is emitted as "9 AM to 2 PM" rather than "सुबह नौ बजे...".
"""

from __future__ import annotations

import re
import time
import uuid
from typing import Any, Optional

import aiohttp

from . import config
from .logging_setup import CallLog, error, info, log, warn

# ═══════════════════════════════════════════════════════════════════════════
# Tool schema exposed to the LLM
# ═══════════════════════════════════════════════════════════════════════════
TOOLS_SCHEMA: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "create_data",
            "description": (
                "Book the appointment. Call this ONLY after the patient has "
                "confirmed name, symptom, duration and preferred slot."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Patient's full name, as spoken.",
                    },
                    "symptom": {
                        "type": "string",
                        "description": "Main symptom or health issue.",
                    },
                    "days": {
                        "type": "string",
                        "description": (
                            "How long the patient has had the symptom, e.g. "
                            "'3 days', '2 weeks'."
                        ),
                    },
                    "preferred_time": {
                        "type": "string",
                        "enum": ["morning", "evening"],
                        "description": "Preferred OPD slot.",
                    },
                },
                "required": ["name", "symptom", "days", "preferred_time"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_caller_phone",
            "description": (
                "Record the caller's phone number when they state it verbally. "
                "Call this as soon as a 10-digit number is clearly heard. "
                "Do NOT call this for numbers the caller only mentions in "
                "passing (e.g. a relative's number)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "phone_number": {
                        "type": "string",
                        "description": (
                            "The 10-digit Indian mobile number, digits only. "
                            "Strip spaces, dashes and country code."
                        ),
                    }
                },
                "required": ["phone_number"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mark_not_booked",
            "description": (
                "Call this if the caller clearly will NOT book an appointment "
                "on this call (just asking for info, will visit in person, "
                "wrong number, etc.). Say the closing line in the same turn."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": (
                            "Short reason: 'only_enquiry', 'wrong_number', "
                            "'will_visit', 'not_interested', 'emergency_redirect'."
                        ),
                    }
                },
                "required": ["reason"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "end_call",
            "description": (
                "End the phone call. Say the farewell line in the same turn "
                "before calling this."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": (
                            "Short reason: 'booking_complete', 'caller_declined', "
                            "'emergency_redirect', 'no_response'."
                        ),
                    }
                },
                "required": ["reason"],
                "additionalProperties": False,
            },
        },
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# Hindi → English helpers (for WhatsApp-safe text)
# ═══════════════════════════════════════════════════════════════════════════
_HINDI_NUMBERS = {
    "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "पाँच": 5, "छह": 6,
    "छः": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10, "पंद्रह": 15, "बीस": 20,
    "एक्": 1,
}

# Common symptoms — extend this as you see more calls
_SYMPTOM_MAP = {
    "दिल में दर्द": "chest pain",
    "दिल मे दर्द": "chest pain",
    "सीने में दर्द": "chest pain",
    "कमर दर्द": "back pain",
    "कमर में दर्द": "back pain",
    "पीठ दर्द": "back pain",
    "सिर दर्द": "headache",
    "सर दर्द": "headache",
    "पेट दर्द": "stomach pain",
    "पेट में दर्द": "stomach pain",
    "बुखार": "fever",
    "खांसी": "cough",
    "खाँसी": "cough",
    "जुकाम": "cold",
    "जुक़ाम": "cold",
    "घुटने में दर्द": "knee pain",
    "जोड़ों में दर्द": "joint pain",
    "थकान": "fatigue",
    "चक्कर": "dizziness",
    "उल्टी": "vomiting",
    "दस्त": "diarrhea",
    "कब्ज": "constipation",
    "सांस": "breathing difficulty",
    "साँस": "breathing difficulty",
}


def _ascii_safe_patient_text(text: str) -> str:
    """
    Best-effort Hindi → English for WhatsApp-safe text.
    1. Try known symptom phrase mapping.
    2. Otherwise strip non-ASCII.
    Never returns empty — falls back to 'symptom'.
    """
    if not text:
        return "symptom"
    s = text.strip()

    # Direct phrase match
    for hindi, english in _SYMPTOM_MAP.items():
        if hindi in s:
            return english

    # Word-level fallback
    for hindi, english in _SYMPTOM_MAP.items():
        if any(part in s for part in hindi.split()):
            return english

    # Last resort — strip non-ASCII
    cleaned = s.encode("ascii", "ignore").decode("ascii").strip()
    return cleaned or "symptom"


def normalise_days(raw: str) -> tuple[str, Optional[int]]:
    """
    Turn '3 din' / 'तीन दिन' / 'do hafte' into (original, approx_days).
    The clinic API gets both, so nothing is lost if our guess is wrong.
    """
    s = (raw or "").strip()
    if not s:
        return s, None

    n: Optional[int] = None
    m = re.search(r"\d+", s)
    if m:
        n = int(m.group())
    else:
        for word, val in _HINDI_NUMBERS.items():
            if word in s:
                n = val
                break

    if n is None:
        return s, None

    low = s.lower()
    if "हफ" in s or "सप्ताह" in s or "week" in low:
        n *= 7
    elif "महीन" in s or "month" in low:
        n *= 30
    elif "साल" in s or "year" in low or "वर्ष" in s:
        n *= 365
    return s, n


def _english_duration(days_text: str, days_num: Optional[int]) -> str:
    """Produce an ASCII-only duration string for WhatsApp."""
    if days_num:
        if days_num == 1:
            return "1 day"
        if days_num < 7:
            return f"{days_num} days"
        if days_num < 30:
            weeks = max(1, days_num // 7)
            return f"{weeks} week{'s' if weeks > 1 else ''}"
        if days_num < 365:
            months = max(1, days_num // 30)
            return f"{months} month{'s' if months > 1 else ''}"
        years = max(1, days_num // 365)
        return f"{years} year{'s' if years > 1 else ''}"

    # Fall back to whatever the LLM gave, stripped of non-ASCII
    cleaned = (days_text or "").encode("ascii", "ignore").decode("ascii").strip()
    return cleaned or "unspecified"


# ═══════════════════════════════════════════════════════════════════════════
# ToolRunner
# ═══════════════════════════════════════════════════════════════════════════
class ToolRunner:
    """Executes tool calls for one call. Holds the HTTP session."""

    def __init__(self, call_log: CallLog, call_id: str) -> None:
        self.call_log = call_log
        self.call_id = call_id
        self._http: Optional[aiohttp.ClientSession] = None
        self.hangup_requested = False
        self.hangup_reason = ""
        self.booking: Optional[dict] = None

        # Bound by CallSession.__init__ via bind_session(). Optional — if
        # it's None, tools still work but won't trigger WhatsApp follow-ups.
        self._session = None

    # ── session binding ─────────────────────────────────────────────────
    def bind_session(self, session) -> None:
        """
        Called by CallSession.__init__ so the tool handlers can reach back
        into the session (booking confirmation, phone capture, etc.).
        """
        self._session = session
        info("tool", f"ToolRunner bound to session call_id={self.call_id}")

    # ── HTTP plumbing ──────────────────────────────────────────────────
    async def _get_http(self) -> aiohttp.ClientSession:
        if self._http is None or self._http.closed:
            self._http = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=config.CLINIC_API_TIMEOUT_S)
            )
        return self._http

    # ── dispatcher ─────────────────────────────────────────────────────
    async def run(self, name: str, args: dict[str, Any]) -> dict:
        t0 = time.monotonic()
        self.call_log.event("tool_call", tool=name, arguments=args)
        info("tool", f"{name}({args})")

        try:
            if name == "create_data":
                result = await self._create_data(args)
            elif name == "end_call":
                result = self._end_call(args)
            elif name == "set_caller_phone":
                result = self._set_caller_phone(args)
            elif name == "mark_not_booked":
                result = self._mark_not_booked(args)
            else:
                warn("tool", f"unknown tool: {name}")
                result = {"success": False, "error": f"unknown tool {name}"}
        except Exception as exc:
            error("tool", f"{name} raised: {exc}")
            result = {"success": False, "error": str(exc)}

        ms = round((time.monotonic() - t0) * 1000, 1)
        self.call_log.event("tool_result", tool=name, ms=ms, result=result)
        info("tool", f"{name} -> {result} ({ms} ms)")
        return result

    # ═══════════════════════════════════════════════════════════════════
    # create_data — book appointment
    # ═══════════════════════════════════════════════════════════════════
    async def _create_data(self, args: dict[str, Any]) -> dict:
        name = str(args.get("name") or "").strip()
        symptom = str(args.get("symptom") or "").strip()
        days_raw = str(args.get("days") or "").strip()
        slot = str(args.get("preferred_time") or "").strip().lower()

        missing = [
            k
            for k, v in (
                ("name", name),
                ("symptom", symptom),
                ("days", days_raw),
            )
            if not v
        ]
        if missing:
            return {
                "success": False,
                "error": "missing_fields",
                "missing_fields": missing,
                "say": "कुछ जानकारी छूट गई है, कृपया दोबारा बताइए।",
            }

        if slot not in {"morning", "evening"}:
            slot = "morning"

        days_text, days_num = normalise_days(days_raw)

        payload = {
            "clinic_id": config.CLINIC_ID,
            "clinic_name": config.CLINIC_NAME,
            "doctor_name": config.DOCTOR.name,
            "patient_name": name,
            "symptom": symptom,
            "duration_text": days_text,
            "duration_days": days_num,
            "preferred_time": slot,
            "source": "voice_agent",
            "call_id": self.call_id,
            "phone_number": self._caller_phone() or None,
            "status": "pending",
        }

        # ─── Dry-run / no-key path ─────────────────────────────────────
        if config.CLINIC_API_DRY_RUN or not config.CLINIC_API_KEY:
            reason = (
                "CLINIC_API_DRY_RUN is on"
                if config.CLINIC_API_DRY_RUN
                else "clinic API key not configured"
            )
            warn("tool", f"booking NOT persisted — {reason}")
            appointment_id = f"local_{int(time.time())}"
            self.booking = {**payload, "appointment_id": appointment_id}

            # Notify the session so the booking registry is populated and
            # WhatsApp will fire when the call ends.
            self._notify_booking_confirmed(
                name=name,
                symptom=symptom,
                days_text=days_text,
                days_num=days_num,
                slot=slot,
                appointment_id=appointment_id,
            )

            return {
                "success": True,
                "persisted": False,
                "note": reason,
                "appointment_id": appointment_id,
                "doctor": config.DOCTOR.name,
                "address": config.DOCTOR.address,
                "slot": slot,
                "slot_hindi": (
                    config.DOCTOR.morning if slot == "morning" else config.DOCTOR.evening
                ),
            }

        # ─── Real API call ────────────────────────────────────────────
        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": config.CLINIC_API_KEY,
        }
        if config.CLINIC_API_SECRET:
            headers["X-Api-Secret"] = config.CLINIC_API_SECRET

        url = f"{config.CLINIC_API_BASE}/appointments"
        try:
            session = await self._get_http()
            async with session.post(url, json=payload, headers=headers) as resp:
                body = await resp.text()
                log("tool", f"POST {url} -> {resp.status} {body[:400]}")

                if 200 <= resp.status < 300:
                    try:
                        data = await resp.json(content_type=None)
                    except Exception:
                        data = {}
                    appointment_id = (
                        (data or {}).get("id")
                        or (data or {}).get("appointment_id")
                        or f"apt_{int(time.time())}"
                    )
                    self.booking = {**payload, "appointment_id": appointment_id}

                    # Notify the session.
                    self._notify_booking_confirmed(
                        name=name,
                        symptom=symptom,
                        days_text=days_text,
                        days_num=days_num,
                        slot=slot,
                        appointment_id=appointment_id,
                    )

                    return {
                        "success": True,
                        "persisted": True,
                        "appointment_id": appointment_id,
                        "doctor": config.DOCTOR.name,
                        "address": config.DOCTOR.address,
                        "slot": slot,
                        "slot_hindi": (
                            config.DOCTOR.morning
                            if slot == "morning"
                            else config.DOCTOR.evening
                        ),
                    }

                error("tool", f"clinic API {resp.status}: {body[:300]}")
                return {
                    "success": False,
                    "error": f"clinic_api_{resp.status}",
                    "say": (
                        "अभी सिस्टम में दिक्कत आ रही है। "
                        "आपकी जानकारी नोट कर ली गई है, हमारी टीम आपको कॉल करेगी।"
                    ),
                }
        except Exception as exc:
            error("tool", f"clinic API call failed: {exc}")
            return {
                "success": False,
                "error": str(exc),
                "say": (
                    "अभी सिस्टम से कनेक्शन नहीं हो पा रहा। "
                    "आपकी जानकारी नोट कर ली गई है, हमारी टीम आपको कॉल करेगी।"
                ),
            }

    # ═══════════════════════════════════════════════════════════════════
    # set_caller_phone — LLM states the number verbally
    # ═══════════════════════════════════════════════════════════════════
    def _set_caller_phone(self, args: dict[str, Any]) -> dict:
        raw = str(args.get("phone_number") or "").strip()
        if not raw:
            return {"success": False, "error": "empty_phone"}

        digits = "".join(ch for ch in raw if ch.isdigit())
        # Strip leading 91 / 0 if the caller included country code.
        if len(digits) == 12 and digits.startswith("91"):
            digits = digits[2:]
        elif len(digits) == 11 and digits.startswith("0"):
            digits = digits[1:]

        if len(digits) != 10:
            warn("tool", f"phone_number not 10 digits after cleanup: {raw!r} -> {digits!r}")
            return {
                "success": False,
                "error": "invalid_phone",
                "digits": digits,
                "say": "क्या आप नंबर दोबारा बता सकते हैं? दस अंकों का।",
            }

        if self._session is not None:
            try:
                self._session.set_phone_number(digits)
            except Exception as exc:
                warn("tool", f"session.set_phone_number failed: {exc}")
        return {"success": True, "phone_number": digits}

    # ═══════════════════════════════════════════════════════════════════
    # mark_not_booked — caller declined
    # ═══════════════════════════════════════════════════════════════════
    def _mark_not_booked(self, args: dict[str, Any]) -> dict:
        reason = str(args.get("reason") or "unspecified")
        if self._session is not None:
            try:
                self._session.on_no_booking(reason=reason)
            except Exception as exc:
                warn("tool", f"session.on_no_booking failed: {exc}")
        else:
            self.booking = {"status": "not_booked", "reason": reason}
        return {"success": True, "status": "not_booked", "reason": reason}

    # ═══════════════════════════════════════════════════════════════════
    # end_call
    # ═══════════════════════════════════════════════════════════════════
    def _end_call(self, args: dict[str, Any]) -> dict:
        self.hangup_requested = True
        self.hangup_reason = str(args.get("reason") or "unspecified")

        # If the call ends and no booking was recorded, treat as not-booked.
        # This ensures the follow-up WhatsApp goes out even if the LLM
        # forgot to call mark_not_booked.
        if (
            self._session is not None
            and getattr(self._session, "booking_record", None) is None
        ):
            try:
                self._session.on_no_booking(reason=f"end_call:{self.hangup_reason}")
            except Exception as exc:
                warn("tool", f"session.on_no_booking on end_call failed: {exc}")

        return {"success": True, "message": "Call will end after this reply."}

    # ═══════════════════════════════════════════════════════════════════
    # helpers
    # ═══════════════════════════════════════════════════════════════════
    def _caller_phone(self) -> str:
        """Best-known caller number so far (empty string if unknown)."""
        if self._session is not None:
            return getattr(self._session, "phone_number", "") or ""
        return ""

    def _notify_booking_confirmed(
        self,
        *,
        name: str,
        symptom: str,
        days_text: str,
        days_num: Optional[int],
        slot: str,
        appointment_id: str,
    ) -> None:
        """
        Push the confirmed booking into the session's booking registry.
        All strings passed here must be ASCII-safe — the WhatsApp message
        will use them verbatim, and Vedronix rejects non-ASCII.

        We store the original Hindi alongside the English translation in
        `extra` so the call log still shows what the caller actually said.
        """
        if self._session is None:
            warn("tool", "no session bound — booking will not trigger WhatsApp")
            return

        # English appointment time
        appointment_time_en = (
            "9 AM to 2 PM" if slot == "morning" else "4 PM to 8 PM"
        )

        # English symptom
        symptom_en = _ascii_safe_patient_text(symptom)

        # English duration
        duration_en = _english_duration(days_text, days_num)

        # Full English summary for WhatsApp
        summary_en = (
            f"Patient: {name} | Symptom: {symptom_en} | "
            f"Duration: {duration_en} | Slot: {appointment_time_en}"
        )

        try:
            self._session.on_booking_confirmed(
                patient_name=name,
                doctor=config.DOCTOR.name,
                appointment_time=appointment_time_en,
                booking_id=appointment_id,
                extra={
                    # Original (for logs / clinic API)
                    "symptom_original": symptom,
                    "duration_original": days_text,
                    # English (for WhatsApp)
                    "symptom_en": symptom_en,
                    "duration_en": duration_en,
                    "summary_en": summary_en,
                    "preferred_slot": slot,
                },
            )
        except Exception as exc:
            error("tool", f"session.on_booking_confirmed failed: {exc}")

    # ═══════════════════════════════════════════════════════════════════
    # cleanup
    # ═══════════════════════════════════════════════════════════════════
    async def close(self) -> None:
        if self._http and not self._http.closed:
            await self._http.close()