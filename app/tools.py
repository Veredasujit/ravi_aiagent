"""
tools.py — the two functions the LLM can call.

`create_data`  books the appointment against the clinic API.
`end_call`     ends the call after the farewell has been spoken.

Both are fully logged: arguments in, HTTP status and body out, wall time.
If the clinic API is unreachable the tool returns a structured failure rather
than raising, so Ravi can apologise in Hindi instead of the call going silent.
"""

from __future__ import annotations

import re
import time
from typing import Any, Optional

import aiohttp

from . import config
from .logging_setup import CallLog, error, info, log, warn

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

_HINDI_NUMBERS = {
    "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "पाँच": 5, "छह": 6,
    "छः": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10, "पंद्रह": 15, "बीस": 20,
    "एक्": 1,
}


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


class ToolRunner:
    """Executes tool calls for one call. Holds the HTTP session."""

    def __init__(self, call_log: CallLog, call_id: str) -> None:
        self.call_log = call_log
        self.call_id = call_id
        self._session: Optional[aiohttp.ClientSession] = None
        self.hangup_requested = False
        self.hangup_reason = ""
        self.booking: Optional[dict] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=config.CLINIC_API_TIMEOUT_S)
            )
        return self._session

    async def run(self, name: str, args: dict[str, Any]) -> dict:
        t0 = time.monotonic()
        self.call_log.event("tool_call", tool=name, arguments=args)
        info("tool", f"{name}({args})")

        if name == "create_data":
            result = await self._create_data(args)
        elif name == "end_call":
            result = self._end_call(args)
        else:
            warn("tool", f"unknown tool: {name}")
            result = {"success": False, "error": f"unknown tool {name}"}

        ms = round((time.monotonic() - t0) * 1000, 1)
        self.call_log.event("tool_result", tool=name, ms=ms, result=result)
        info("tool", f"{name} -> {result} ({ms} ms)")
        return result

    # ── create_data ─────────────────────────────────────────────────────────
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
            "status": "pending",
        }

        if config.CLINIC_API_DRY_RUN or not config.CLINIC_API_KEY:
            reason = (
                "CLINIC_API_DRY_RUN is on"
                if config.CLINIC_API_DRY_RUN
                else "clinic API key not configured"
            )
            warn("tool", f"booking NOT persisted — {reason}")
            self.booking = payload
            return {
                "success": True,
                "persisted": False,
                "note": reason,
                "appointment_id": f"local_{int(time.time())}",
                "doctor": config.DOCTOR.name,
                "address": config.DOCTOR.address,
                "slot": slot,
                "slot_hindi": (
                    config.DOCTOR.morning if slot == "morning" else config.DOCTOR.evening
                ),
            }

        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": config.CLINIC_API_KEY,
        }
        if config.CLINIC_API_SECRET:
            headers["X-Api-Secret"] = config.CLINIC_API_SECRET

        url = f"{config.CLINIC_API_BASE}/appointments"
        try:
            session = await self._get_session()
            async with session.post(url, json=payload, headers=headers) as resp:
                body = await resp.text()
                log("tool", f"POST {url} -> {resp.status} {body[:400]}")
                if 200 <= resp.status < 300:
                    try:
                        data = await resp.json(content_type=None)
                    except Exception:
                        data = {}
                    self.booking = payload
                    return {
                        "success": True,
                        "persisted": True,
                        "appointment_id": (data or {}).get("id")
                        or (data or {}).get("appointment_id")
                        or f"apt_{int(time.time())}",
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

    # ── end_call ────────────────────────────────────────────────────────────
    def _end_call(self, args: dict[str, Any]) -> dict:
        self.hangup_requested = True
        self.hangup_reason = str(args.get("reason") or "unspecified")
        return {"success": True, "message": "Call will end after this reply."}

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
