# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Per-pool session table and handover flag of the listener broker (OMN-22379).

Rule A6 lives here as one atomic decision under one lock, so grants are
serialized per pool and the answer never depends on interleaving. Only a
CONFIRMED session counts as an acceptor; a provisional one counts for nothing,
and a session that holds a grant is already leaving and no longer counts either.
"""

from __future__ import annotations

import threading

from pydantic import BaseModel, ConfigDict, PrivateAttr

from omnibase_infra.enums.enum_runtime_listener_detach_verdict import (
    EnumRuntimeListenerDetachVerdict,
)
from omnibase_infra.models.model_runtime_listener_session import (
    ModelRuntimeListenerSession,
)


class ModelRuntimeListenerSessionTable(BaseModel):
    """Sessions by id, and the pools with a handover in progress."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _sessions: dict[str, ModelRuntimeListenerSession] = PrivateAttr(
        default_factory=dict
    )
    _handover: set[str] = PrivateAttr(default_factory=set)

    def open_session(self, session_id: str, pool: str, instance: str) -> str:
        """Record a new PROVISIONAL session; refuse an existing id with an empty state."""
        with self._lock:
            if session_id in self._sessions:
                return ""
            self._sessions[session_id] = ModelRuntimeListenerSession(
                pool=pool, instance=instance
            )
        return "provisional"

    def confirm_session(self, session_id: str) -> str:
        """Promote a session to CONFIRMED; return its state, or '' if unknown."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return ""
            session.confirmed = True
            return "confirmed"

    def end_session(self, session_id: str) -> None:
        """Forget a session, whether it left cleanly or its connection died."""
        with self._lock:
            self._sessions.pop(session_id, None)

    def mark_handover(self, pool: str, in_progress: bool) -> None:
        """Set or clear the agent's handover-in-progress flag for a pool."""
        with self._lock:
            if in_progress:
                self._handover.add(pool)
            else:
                self._handover.discard(pool)

    def handover_in_progress(self, pool: str) -> bool:
        with self._lock:
            return pool in self._handover

    def decide_detach(self, session_id: str) -> tuple[str, str]:
        """Answer one detach request: ``(verdict, error)``, exactly one non-empty.

        ``granted`` only when another CONFIRMED session of the pool is attached
        and holds no outstanding grant; the grant is recorded in the same step.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return "", "no_session"
            if not session.confirmed:
                return "", "not_confirmed"
            other_acceptor = any(
                other.pool == session.pool and other.confirmed and not other.granted
                for key, other in self._sessions.items()
                if key != session_id
            )
            # Every ask re-checks: a grant held since an earlier ask is withdrawn
            # when the acceptor it relied on has since ended.
            session.granted = other_acceptor
            if other_acceptor:
                return EnumRuntimeListenerDetachVerdict.GRANTED.value, ""
            if session.pool in self._handover:
                return EnumRuntimeListenerDetachVerdict.WAIT.value, ""
            return EnumRuntimeListenerDetachVerdict.ALONE.value, ""

    def confirmed_acceptors(self, pool: str) -> int:
        """Count the sessions of a pool that are confirmed and not leaving."""
        with self._lock:
            return sum(
                1
                for s in self._sessions.values()
                if s.pool == pool and s.confirmed and not s.granted
            )

    def session_count(self, pool: str) -> int:
        """Count every session of a pool, provisional ones included."""
        with self._lock:
            return sum(1 for s in self._sessions.values() if s.pool == pool)


__all__: list[str] = ["ModelRuntimeListenerSessionTable"]
