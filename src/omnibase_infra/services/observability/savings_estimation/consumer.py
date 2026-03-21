# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Async Kafka consumer for session-level savings estimation.

ServiceSavingsEstimator correlates events from multiple topics by session_id
in a bounded LRU buffer. When a session-outcome event arrives and a grace
window elapses, it finalizes the estimate using HandlerSavingsEstimator and
produces a ``savings-estimated.v1`` event.

Architecture:
    - Correlates events by ``session_id`` in bounded LRU buffer (max_sessions)
    - Finalizes on ``session-outcome.v1`` + grace_window_seconds
    - Produces ``savings-estimated.v1`` with deterministic source_event_id
    - In-memory finalized-session set (last N) as optimization
    - Correctness relies on downstream UNIQUE constraint, not in-memory set

Consumed topics:
    - onex.evt.omniintelligence.llm-call-completed.v1
    - onex.evt.omniclaude.session-outcome.v1
    - onex.evt.omniclaude.hook-context-injected.v1
    - onex.evt.omniclaude.validator-catch.v1

Produced:
    - onex.evt.omnibase-infra.savings-estimated.v1

Related Tickets:
    - OMN-5550: Create ServiceSavingsEstimator Kafka consumer
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from omnibase_infra.models.pricing.model_pricing_table import ModelPricingTable
from omnibase_infra.nodes.node_savings_estimation_compute.handlers.handler_savings_estimator import (
    HandlerSavingsEstimator,
)
from omnibase_infra.nodes.node_savings_estimation_compute.models import (
    ModelInjectionSignal,
    ModelLlmCallRecord,
    ModelSavingsBaselineConfig,
    ModelSavingsInput,
    ModelValidatorCatchSignal,
)
from omnibase_infra.services.observability.savings_estimation.config import (
    ConfigSavingsEstimation,
)

logger = logging.getLogger(__name__)


@dataclass
class SessionBuffer:
    """Accumulates signals for a single session."""

    session_id: str
    correlation_id: str = ""
    llm_calls: list[ModelLlmCallRecord] = field(default_factory=list)
    injection_signals: list[ModelInjectionSignal] = field(default_factory=list)
    validator_catches: list[ModelValidatorCatchSignal] = field(default_factory=list)
    treatment_group: str = "treatment"
    outcome_received: bool = False
    outcome_received_at: float = 0.0
    created_at: float = field(default_factory=lambda: time.monotonic())


class ServiceSavingsEstimator:
    """Session-event correlator producing savings-estimated.v1 events.

    Callers feed events via ``ingest_event()`` and collect produced
    events via ``finalize_ready_sessions()``. Kafka I/O is external.
    """

    def __init__(
        self,
        config: ConfigSavingsEstimation,
        pricing_table: ModelPricingTable,
    ) -> None:
        self._config = config
        self._handler = HandlerSavingsEstimator(pricing_table)
        self._sessions: OrderedDict[str, SessionBuffer] = OrderedDict()
        self._finalized: OrderedDict[str, bool] = OrderedDict()
        self._max_sessions = config.max_sessions
        self._max_finalized = config.finalized_session_cache_size
        self._grace_seconds = config.grace_window_seconds
        self._timeout_seconds = config.session_timeout_seconds
        self._schema_version = config.schema_version

    @property
    def active_session_count(self) -> int:
        return len(self._sessions)

    def is_finalized(self, session_id: str) -> bool:
        return session_id in self._finalized

    def ingest_event(self, topic: str, payload: dict[str, object]) -> None:
        """Ingest a consumed event into the correlation buffer."""
        session_id = str(payload.get("session_id", ""))
        if not session_id:
            return

        if self.is_finalized(session_id):
            return

        buf = self._get_or_create_session(session_id)

        if "llm-call-completed" in topic:
            self._ingest_llm_call(buf, payload)
        elif "session-outcome" in topic:
            self._ingest_session_outcome(buf, payload)
        elif "hook-context-injected" in topic:
            self._ingest_injection(buf, payload)
        elif "validator-catch" in topic:
            self._ingest_validator_catch(buf, payload)

    async def finalize_ready_sessions(self) -> list[dict[str, object]]:
        """Check all sessions and finalize those past the grace window or timed out.

        Returns a list of savings estimate dicts ready to be produced.
        """
        now = time.monotonic()
        ready_ids: list[str] = []

        for session_id, buf in self._sessions.items():
            if buf.outcome_received:
                elapsed = now - buf.outcome_received_at
                if elapsed >= self._grace_seconds:
                    ready_ids.append(session_id)
            elif now - buf.created_at > self._timeout_seconds:
                ready_ids.append(session_id)

        results: list[dict[str, object]] = []
        for session_id in ready_ids:
            buf = self._sessions.pop(session_id)
            estimate = await self._finalize_session(buf)
            if estimate is not None:
                results.append(estimate)
                self._mark_finalized(session_id)

        return results

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _get_or_create_session(self, session_id: str) -> SessionBuffer:
        if session_id in self._sessions:
            self._sessions.move_to_end(session_id)
            return self._sessions[session_id]

        if len(self._sessions) >= self._max_sessions:
            self._sessions.popitem(last=False)

        buf = SessionBuffer(session_id=session_id)
        self._sessions[session_id] = buf
        return buf

    def _ingest_llm_call(self, buf: SessionBuffer, payload: dict[str, object]) -> None:
        model_id = str(payload.get("model_id", ""))
        prompt_tokens = int(str(payload.get("prompt_tokens", 0)))
        completion_tokens = int(str(payload.get("completion_tokens", 0)))
        if model_id:
            buf.llm_calls.append(
                ModelLlmCallRecord(
                    model_id=model_id,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            )

    def _ingest_session_outcome(
        self, buf: SessionBuffer, payload: dict[str, object]
    ) -> None:
        buf.outcome_received = True
        buf.outcome_received_at = time.monotonic()
        if correlation_id := str(payload.get("correlation_id", "")):
            buf.correlation_id = correlation_id
        if treatment_group := str(payload.get("treatment_group", "")):
            buf.treatment_group = treatment_group

    def _ingest_injection(self, buf: SessionBuffer, payload: dict[str, object]) -> None:
        tokens_injected = int(str(payload.get("tokens_injected", 0)))
        patterns_count = int(str(payload.get("patterns_count", 0)))
        if tokens_injected > 0:
            buf.injection_signals.append(
                ModelInjectionSignal(
                    tokens_injected=tokens_injected,
                    patterns_count=patterns_count,
                )
            )

    def _ingest_validator_catch(
        self, buf: SessionBuffer, payload: dict[str, object]
    ) -> None:
        validator_type = str(payload.get("validator_type", ""))
        severity = str(payload.get("severity", ""))
        if validator_type and severity:
            buf.validator_catches.append(
                ModelValidatorCatchSignal(
                    validator_type=validator_type,
                    severity=severity,
                )
            )

    async def _finalize_session(self, buf: SessionBuffer) -> dict[str, object] | None:
        if (
            not buf.llm_calls
            and not buf.injection_signals
            and not buf.validator_catches
        ):
            return None

        savings_input = ModelSavingsInput(
            session_id=buf.session_id,
            correlation_id=buf.correlation_id or buf.session_id,
            llm_calls=buf.llm_calls,
            treatment_group=buf.treatment_group,
            injection_signals=buf.injection_signals,
            validator_catches=buf.validator_catches,
            baseline_config=ModelSavingsBaselineConfig(),
        )

        try:
            estimate = await self._handler.handle(savings_input)
            source_event_id = f"savings-{buf.session_id}-v{self._schema_version}"
            estimate["source_event_id"] = source_event_id
            return estimate
        except Exception:
            logger.exception(
                "Failed to finalize savings estimate for session %s",
                buf.session_id,
            )
            return None

    def _mark_finalized(self, session_id: str) -> None:
        if len(self._finalized) >= self._max_finalized:
            self._finalized.popitem(last=False)
        self._finalized[session_id] = True


__all__: list[str] = ["ServiceSavingsEstimator"]
