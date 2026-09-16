"""
booking_state.py — tracks the outcome of each call so we know
which WhatsApp message (if any) to send when the call ends.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from threading import Lock
from typing import Any, Optional


class BookingStatus(str, Enum):
    NOT_STARTED = "not_started"
    COLLECTING = "collecting_details"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    NOT_BOOKED = "not_booked"


@dataclass
class BookingRecord:
    call_id: str
    phone_number: str
    status: BookingStatus = BookingStatus.NOT_STARTED
    patient_name: Optional[str] = None
    doctor: Optional[str] = None
    department: Optional[str] = None
    appointment_time: Optional[str] = None
    booking_id: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    whatsapp_sent: bool = False
    whatsapp_message_id: Optional[str] = None
    whatsapp_error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


class BookingRegistry:
    """
    Thread-safe in-memory registry of booking outcomes per call_id.
    Swap for Redis/DB in multi-process deployments.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._records: dict[str, BookingRecord] = {}

    def create(self, call_id: str, phone_number: str) -> BookingRecord:
        with self._lock:
            rec = BookingRecord(call_id=call_id, phone_number=phone_number)
            self._records[call_id] = rec
            return rec

    def get(self, call_id: str) -> Optional[BookingRecord]:
        with self._lock:
            return self._records.get(call_id)

    def mark_confirmed(
        self,
        call_id: str,
        *,
        patient_name: Optional[str] = None,
        doctor: Optional[str] = None,
        department: Optional[str] = None,
        appointment_time: Optional[str] = None,
        booking_id: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> Optional[BookingRecord]:
        with self._lock:
            rec = self._records.get(call_id)
            if not rec:
                return None
            rec.status = BookingStatus.CONFIRMED
            rec.patient_name = patient_name or rec.patient_name
            rec.doctor = doctor or rec.doctor
            rec.department = department or rec.department
            rec.appointment_time = appointment_time or rec.appointment_time
            rec.booking_id = booking_id or rec.booking_id
            if extra:
                rec.details.update(extra)
            rec.updated_at = time.time()
            return rec

    def mark_not_booked(self, call_id: str, reason: str = "") -> Optional[BookingRecord]:
        with self._lock:
            rec = self._records.get(call_id)
            if not rec:
                return None
            if rec.status == BookingStatus.CONFIRMED:
                return rec  # don't downgrade a confirmed booking
            rec.status = BookingStatus.NOT_BOOKED
            if reason:
                rec.details["reason"] = reason
            rec.updated_at = time.time()
            return rec

    def mark_cancelled(self, call_id: str) -> Optional[BookingRecord]:
        with self._lock:
            rec = self._records.get(call_id)
            if not rec:
                return None
            rec.status = BookingStatus.CANCELLED
            rec.updated_at = time.time()
            return rec

    def mark_whatsapp_sent(self, call_id: str, message_id: Optional[str]) -> None:
        with self._lock:
            rec = self._records.get(call_id)
            if rec:
                rec.whatsapp_sent = True
                rec.whatsapp_message_id = message_id
                rec.updated_at = time.time()

    def mark_whatsapp_failed(self, call_id: str, error: str) -> None:
        with self._lock:
            rec = self._records.get(call_id)
            if rec:
                rec.whatsapp_error = error
                rec.updated_at = time.time()

    def pop(self, call_id: str) -> Optional[BookingRecord]:
        with self._lock:
            return self._records.pop(call_id, None)

    def cleanup_older_than(self, seconds: float) -> int:
        cutoff = time.time() - seconds
        with self._lock:
            stale = [k for k, v in self._records.items() if v.updated_at < cutoff]
            for k in stale:
                self._records.pop(k, None)
        return len(stale)
    # booking_state.py — inside BookingRegistry class

def rekey(self, old_call_id: str, new_call_id: str) -> Optional[BookingRecord]:
    """
    Re-key a booking record when the session adopts Teler's call_id
    after the initial WebSocket `start` frame.
    """
    if not old_call_id or not new_call_id or old_call_id == new_call_id:
        return self._records.get(new_call_id)
    with self._lock:
        rec = self._records.pop(old_call_id, None)
        if rec is None:
            return self._records.get(new_call_id)
        # If a record already existed under the new ID, keep the one
        # with the more advanced status (confirmed > not_booked > ...).
        existing = self._records.get(new_call_id)
        if existing is not None:
            rank = {
                BookingStatus.NOT_STARTED: 0,
                BookingStatus.COLLECTING: 1,
                BookingStatus.NOT_BOOKED: 2,
                BookingStatus.CANCELLED: 2,
                BookingStatus.CONFIRMED: 3,
            }
            if rank.get(rec.status, 0) >= rank.get(existing.status, 0):
                rec.call_id = new_call_id
                self._records[new_call_id] = rec
            # else keep existing
            return self._records[new_call_id]
        rec.call_id = new_call_id
        self._records[new_call_id] = rec
        return rec


# module-level singleton
booking_registry = BookingRegistry()