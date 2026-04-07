# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Async Kafka consumer for session-level savings estimation.

ServiceSavingsEstimator correlates events from multiple topics by session_id
in a bounded LRU buffer. When a session-outcome event arrives and a grace
window elapses, it finalizes the estimate using HandlerSavingsEstimation and
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
    - onex.evt.omniclaude.pattern-enforcement.v1

Produced:
    - onex.evt.omnibase-infra.savings-estimated.v1

Related Tickets:
    - OMN-5550: Create ServiceSavingsEstimator Kafka consumer
    - OMN-7494: Heuristic savings from validator catches
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import StrEnum
from uuid import UUID

from omnibase_infra.nodes.node_savings_estimation_compute.handlers.handler_savings_estimation import (
    HandlerSavingsEstimation,
)
from omnibase_infra.nodes.node_savings_estimation_compute.models import (
    EnumModelTier,
    EnumSavingsCategory,
    ModelEffectivenessEntry,
    ModelSavingsCategory,
    ModelSavingsEstimationInput,
)
from omnibase_infra.services.observability.savings_estimation.config import (
    ConfigSavingsEstimation,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Severity classification for validator catches
# ---------------------------------------------------------------------------


class EnumCatchSeverity(StrEnum):
    """Severity levels for validator catches."""

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


# Heuristic avoided-rework estimation per catch severity.
# These are rough order-of-magnitude USD estimates per catch, calibrated
# from observed rework patterns. Not derived from a fixed hourly rate.
_SEVERITY_SAVINGS_USD: dict[EnumCatchSeverity, float] = {
    EnumCatchSeverity.CRITICAL: 0.50,
    EnumCatchSeverity.MAJOR: 0.20,
    EnumCatchSeverity.MINOR: 0.05,
}

# Estimated tokens that would have been wasted in rework per severity.
_SEVERITY_TOKENS_SAVED: dict[EnumCatchSeverity, int] = {
    EnumCatchSeverity.CRITICAL: 2000,
    EnumCatchSeverity.MAJOR: 800,
    EnumCatchSeverity.MINOR: 200,
}

# Confidence in the heuristic estimate per severity.
_SEVERITY_CONFIDENCE: dict[EnumCatchSeverity, float] = {
    EnumCatchSeverity.CRITICAL: 0.7,
    EnumCatchSeverity.MAJOR: 0.6,
    EnumCatchSeverity.MINOR: 0.4,
}

# Counterfactual model: the highest-cost configured routing candidate.
# This is the model that _could_ have been selected for the request.
# Used to compute direct savings (actual cost vs counterfactual cost).
_COUNTERFACTUAL_MODEL_MAP: dict[str, str] = {
    # If actual model is sonnet, counterfactual is opus (more expensive)
    "claude-sonnet-4": "claude-opus-4-6",
    "claude-3-5-sonnet": "claude-opus-4-6",
    "claude-3.5-sonnet": "claude-opus-4-6",
    # If actual model is already opus, no cheaper alternative exists in same class
    # so counterfactual is self (zero direct savings from routing).
    "claude-opus-4-6": "claude-opus-4-6",
    "claude-3-opus": "claude-opus-4-6",
}


def _resolve_counterfactual(actual_model_id: str) -> str:
    """Resolve the counterfactual model for a given actual model.

    Returns the highest-cost configured model within the same capability class.
    Falls back to actual_model_id if no match is found.
    """
    lower = actual_model_id.lower()
    for key, value in _COUNTERFACTUAL_MODEL_MAP.items():
        if key in lower:
            return value
    return actual_model_id


# ---------------------------------------------------------------------------
# Signal dataclasses
# ---------------------------------------------------------------------------


@dataclass
class InjectionSignal:
    """Raw injection signal accumulated from hook-context-injected events."""

    tokens_injected: int = 0
    patterns_count: int = 0


@dataclass
class LlmCallSignal:
    """Raw LLM call signal accumulated from llm-call-completed events."""

    model_id: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class ValidatorCatchSignal:
    """Raw validator catch signal with severity for heuristic savings."""

    severity: EnumCatchSeverity = EnumCatchSeverity.MINOR
    validator_type: str = ""


@dataclass
class SessionBuffer:
    """Accumulates signals for a single session."""

    session_id: str
    correlation_id: str = ""
    llm_calls: list[LlmCallSignal] = field(default_factory=list)
    injection_signals: list[InjectionSignal] = field(default_factory=list)
    validator_catches: list[ValidatorCatchSignal] = field(default_factory=list)
    treatment_group: str = "treatment"
    outcome_received: bool = False
    outcome_received_at: float = 0.0
    created_at: float = field(default_factory=lambda: time.monotonic())

    @property
    def validator_catch_count(self) -> int:
        """Backwards-compatible count of validator catches."""
        return len(self.validator_catches)


def _classify_severity(raw: str) -> EnumCatchSeverity:
    """Classify a severity string into an enum value.

    Args:
        raw: Raw severity string (e.g. 'error', 'CRITICAL', 'warning').

    Returns:
        Classified severity. Defaults to MINOR for unrecognized values.
    """
    lower = raw.lower().strip()
    if lower in ("critical", "error", "fatal"):
        return EnumCatchSeverity.CRITICAL
    if lower in ("major", "warning", "warn"):
        return EnumCatchSeverity.MAJOR
    return EnumCatchSeverity.MINOR


def _compute_validator_catch_savings(
    catches: list[ValidatorCatchSignal],
) -> tuple[float, int, float]:
    """Compute heuristic avoided-rework savings from validator catches.

    Applies diminishing returns: each subsequent catch in the same session
    contributes less, using a logarithmic saturation curve. This prevents
    a session with 50 MINOR catches from claiming enormous savings.

    Formula: effective_multiplier = 1 / (1 + 0.3 * index)
    This gives: catch 1 = 1.0x, catch 2 = 0.77x, catch 3 = 0.63x, ...

    Args:
        catches: List of validator catch signals for a session.

    Returns:
        Tuple of (total_savings_usd, total_tokens_saved, avg_confidence).
    """
    if not catches:
        return 0.0, 0, 0.0

    total_usd = 0.0
    total_tokens = 0
    confidence_sum = 0.0

    # Sort by severity (critical first) so high-value catches get full weight
    sorted_catches = sorted(catches, key=lambda c: c.severity)

    for idx, catch in enumerate(sorted_catches):
        # Diminishing returns: each subsequent catch contributes less
        diminishing_factor = 1.0 / (1.0 + 0.3 * idx)

        base_usd = _SEVERITY_SAVINGS_USD.get(catch.severity, 0.05)
        base_tokens = _SEVERITY_TOKENS_SAVED.get(catch.severity, 200)
        confidence = _SEVERITY_CONFIDENCE.get(catch.severity, 0.4)

        total_usd += base_usd * diminishing_factor
        total_tokens += int(base_tokens * diminishing_factor)
        confidence_sum += confidence

    avg_confidence = confidence_sum / len(catches) if catches else 0.0
    return round(total_usd, 10), total_tokens, round(avg_confidence, 4)


def _model_tier_from_id(model_id: str) -> EnumModelTier:
    """Classify a model_id string into a pricing tier.

    Args:
        model_id: Model identifier (e.g. 'claude-opus-4-6', 'claude-sonnet-4').

    Returns:
        The corresponding model tier. Defaults to OPUS for unknown models.
    """
    lower = model_id.lower()
    if "sonnet" in lower:
        return EnumModelTier.SONNET
    return EnumModelTier.OPUS


def _build_effectiveness_entries(
    buf: SessionBuffer,
) -> tuple[ModelEffectivenessEntry, ...]:
    """Convert raw session signals into ModelEffectivenessEntry objects.

    Each injection signal produces one entry with tokens_saved from the
    injection and patterns_count for category classification. LLM calls
    are aggregated to determine the dominant model tier.

    Args:
        buf: Session buffer with accumulated raw signals.

    Returns:
        Tuple of effectiveness entries for the handler.
    """
    # Determine dominant model tier from LLM calls
    tier = EnumModelTier.OPUS
    if buf.llm_calls:
        tier = _model_tier_from_id(buf.llm_calls[0].model_id)

    entries: list[ModelEffectivenessEntry] = []

    for sig in buf.injection_signals:
        if sig.tokens_injected > 0:
            # Utilization score: heuristic based on patterns injected.
            # More patterns = higher utilization (capped at 1.0).
            utilization = (
                min(sig.patterns_count / 10.0, 1.0) if sig.patterns_count > 0 else 0.5
            )
            entries.append(
                ModelEffectivenessEntry(
                    utilization_score=round(utilization, 4),
                    patterns_count=sig.patterns_count,
                    tokens_saved=sig.tokens_injected,
                    model_tier=tier,
                    is_output_tokens=False,
                )
            )

    # If no injection signals but we have LLM calls, create a minimal entry
    # so the session still produces an estimate (zero savings, measured).
    if not entries and buf.llm_calls:
        total_tokens = sum(c.prompt_tokens + c.completion_tokens for c in buf.llm_calls)
        if total_tokens > 0:
            entries.append(
                ModelEffectivenessEntry(
                    utilization_score=0.0,
                    patterns_count=0,
                    tokens_saved=0,
                    model_tier=tier,
                    is_output_tokens=False,
                )
            )

    # If we have only validator catches and no other signals, create a
    # minimal entry so the session still produces an estimate.
    # Gate on outcome_received to avoid misclassifying timeout-only sessions
    # into the treatment cohort (treatment_group defaults to "treatment").
    if not entries and buf.validator_catches and buf.outcome_received:
        entries.append(
            ModelEffectivenessEntry(
                utilization_score=0.0,
                patterns_count=0,
                tokens_saved=0,
                model_tier=tier,
                is_output_tokens=False,
            )
        )

    return tuple(entries)


class ServiceSavingsEstimator:
    """Session-event correlator producing savings-estimated.v1 events.

    Callers feed events via ``ingest_event()`` and collect produced
    events via ``finalize_ready_sessions()``. Kafka I/O is external.
    """

    def __init__(
        self,
        config: ConfigSavingsEstimation,
    ) -> None:
        self._config = config
        self._handler = HandlerSavingsEstimation()
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
        elif "validator-catch" in topic or "pattern-enforcement" in topic:
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
                LlmCallSignal(
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
                InjectionSignal(
                    tokens_injected=tokens_injected,
                    patterns_count=patterns_count,
                )
            )

    def _ingest_validator_catch(
        self, buf: SessionBuffer, payload: dict[str, object]
    ) -> None:
        raw_severity = str(payload.get("severity", "minor"))
        validator_type = str(payload.get("validator_type", ""))
        severity = _classify_severity(raw_severity)
        buf.validator_catches.append(
            ValidatorCatchSignal(
                severity=severity,
                validator_type=validator_type,
            )
        )

    async def _finalize_session(self, buf: SessionBuffer) -> dict[str, object] | None:
        effectiveness_entries = _build_effectiveness_entries(buf)
        if not effectiveness_entries:
            return None

        # Compute total tokens from LLM calls for actual_cost calculation
        actual_total_tokens = sum(
            c.prompt_tokens + c.completion_tokens for c in buf.llm_calls
        )

        # Determine model ID from LLM calls
        actual_model_id = "claude-opus-4-6"
        if buf.llm_calls:
            actual_model_id = buf.llm_calls[0].model_id

        # Pass the session's correlation_id through to the estimate so it
        # appears in the Kafka payload.  The omnidash projection handler uses
        # correlation_id as the idempotency key (sourceEventId column).
        input_kwargs: dict[str, object] = {
            "session_id": buf.session_id,
            "effectiveness_entries": effectiveness_entries,
            "actual_total_tokens": actual_total_tokens,
            "actual_model_id": actual_model_id,
        }
        if buf.correlation_id:
            try:
                input_kwargs["correlation_id"] = UUID(buf.correlation_id)
            except ValueError:
                pass  # keep auto-generated UUID if not a valid UUID string

        savings_input = ModelSavingsEstimationInput(**input_kwargs)  # type: ignore[arg-type]

        try:
            estimate = await self._handler.handle(savings_input)
            source_event_id = f"savings-{buf.session_id}-v{self._schema_version}"
            result = estimate.model_dump(mode="json")
            result["source_event_id"] = source_event_id

            # --- Counterfactual model ---
            counterfactual = _resolve_counterfactual(actual_model_id)
            result["counterfactual_model_id"] = counterfactual

            # --- Heuristic savings from validator catches (Task 3) ---
            heuristic_usd, heuristic_tokens, heuristic_confidence = (
                _compute_validator_catch_savings(buf.validator_catches)
            )
            result["heuristic_savings_usd"] = heuristic_usd

            # Add validator_catch category to categories breakdown
            if heuristic_usd > 0:
                categories = list(result.get("categories", []))
                categories.append(
                    ModelSavingsCategory(
                        category=EnumSavingsCategory.VALIDATOR_CATCH,
                        savings_usd=heuristic_usd,
                        tokens_saved=heuristic_tokens,
                        confidence=heuristic_confidence,
                    ).model_dump(mode="json")
                )
                result["categories"] = categories

            # Update totals to include heuristic savings
            direct_usd = float(result.get("direct_savings_usd", 0.0))
            direct_tokens = int(result.get("direct_tokens_saved", 0))
            result["estimated_total_savings_usd"] = round(
                direct_usd + heuristic_usd, 10
            )
            result["estimated_total_tokens_saved"] = direct_tokens + heuristic_tokens

            # Update heuristic confidence average
            if heuristic_usd > 0:
                existing_conf = float(result.get("heuristic_confidence_avg", 0.0))
                # Weighted average: existing (from direct) + validator catch confidence
                result["heuristic_confidence_avg"] = round(
                    (existing_conf + heuristic_confidence) / 2.0
                    if existing_conf > 0
                    else heuristic_confidence,
                    4,
                )

            # Propagate treatment_group so omnidash can project it into the
            # savings_estimates table for A/B analysis.
            if buf.treatment_group:
                result["treatment_group"] = buf.treatment_group
            return result
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
