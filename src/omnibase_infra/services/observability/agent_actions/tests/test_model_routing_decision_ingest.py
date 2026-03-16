# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ModelRoutingDecisionIngest (OMN-3422).

Tests the permissive ingest model used at the Kafka boundary for
onex.evt.omniclaude.routing-decision.v1.

Covers:
    - Producer 1 (handler_routing_emitter.py) payload parsing
    - Producer 2 (route_via_events_wrapper.py) payload parsing
    - Field alias mapping (confidence, reasoning, session_id)
    - emitted_at -> created_at timestamp normalization
    - UTC timezone enforcement on created_at
    - Server-generated id and created_at defaults
    - confidence_score clamping and non-numeric coercion
    - Extra field tolerance (extra="ignore")
    - Input dict immutability
    - Strict ModelRoutingDecision still rejects unknown fields
    - Ingest model dump compatible with strict model
    - Writer field presence verification

Related Tickets:
    - OMN-3422: Fix ModelRoutingDecision Schema Drift
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timezone
from uuid import UUID

import pytest
from pydantic import ValidationError

from omnibase_infra.services.observability.agent_actions.models import (
    ModelRoutingDecision,
)
from omnibase_infra.services.observability.agent_actions.models.model_routing_decision_ingest import (
    ModelRoutingDecisionIngest,
)

# =============================================================================
# Fixtures: Producer Payloads
# =============================================================================

_PRODUCER_1_PAYLOAD: dict[str, object] = {
    "correlation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "session_id": "sess-abc-123",
    "selected_agent": "api-architect",
    "confidence": 0.92,
    "emitted_at": "2026-03-01T12:00:00Z",
    # Extra fields that must be ignored
    "confidence_breakdown": {"semantic": 0.80, "rule": 0.12},
    "routing_policy": "default",
    "routing_path": "semantic->rule->fallback",
    "prompt_preview": "Design a REST API for...",
    "prompt_length": 512,
}

_PRODUCER_2_PAYLOAD: dict[str, object] = {
    "correlation_id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
    "session_id": "sess-def-456",
    "selected_agent": "polymorphic-agent",
    "confidence": 0.75,
    "domain": "infrastructure",
    "reasoning": "Matched infrastructure domain pattern",
    "routing_method": "SEMANTIC",
    "latency_ms": 42,
    # Extra fields that must be ignored
    "event_attempted": True,
    "routing_policy": "v2",
    "routing_path": "semantic",
}


# =============================================================================
# Producer Payload Parsing
# =============================================================================


class TestProducerPayloadParsing:
    """Validate that both known producer payloads parse without errors."""

    @pytest.mark.unit
    def test_producer1_payload_parses_cleanly(self) -> None:
        """Full producer 1 dict parses without exception; all aliases mapped."""
        m = ModelRoutingDecisionIngest.model_validate(_PRODUCER_1_PAYLOAD)
        assert m.selected_agent == "api-architect"
        assert m.confidence_score == pytest.approx(0.92)
        assert m.claude_session_id == "sess-abc-123"
        assert m.routing_reason is None
        assert m.domain is None
        assert m.routing_method is None
        assert m.latency_ms is None
        assert isinstance(m.id, UUID)
        assert isinstance(m.correlation_id, UUID)

    @pytest.mark.unit
    def test_producer2_payload_parses_cleanly(self) -> None:
        """Full producer 2 dict parses without exception; all aliases mapped."""
        m = ModelRoutingDecisionIngest.model_validate(_PRODUCER_2_PAYLOAD)
        assert m.selected_agent == "polymorphic-agent"
        assert m.confidence_score == pytest.approx(0.75)
        assert m.claude_session_id == "sess-def-456"
        assert m.routing_reason == "Matched infrastructure domain pattern"
        assert m.domain == "infrastructure"
        assert m.routing_method == "SEMANTIC"
        assert m.latency_ms == 42


# =============================================================================
# Alias Mapping
# =============================================================================


class TestAliasMapping:
    """Verify field aliases map correctly from producer keys to internal names."""

    @pytest.mark.unit
    def test_confidence_alias_mapped(self) -> None:
        """confidence=0.9 in payload -> confidence_score=0.9 on model."""
        m = ModelRoutingDecisionIngest.model_validate(
            {
                "correlation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "selected_agent": "test-agent",
                "confidence": 0.9,
            }
        )
        assert m.confidence_score == pytest.approx(0.9)

    @pytest.mark.unit
    def test_reasoning_alias_mapped(self) -> None:
        """reasoning='low token' in payload -> routing_reason='low token' on model."""
        m = ModelRoutingDecisionIngest.model_validate(
            {
                "correlation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "selected_agent": "test-agent",
                "confidence": 0.5,
                "reasoning": "low token",
            }
        )
        assert m.routing_reason == "low token"

    @pytest.mark.unit
    def test_session_id_alias_mapped(self) -> None:
        """session_id='abc' in payload -> claude_session_id='abc' on model."""
        m = ModelRoutingDecisionIngest.model_validate(
            {
                "correlation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "selected_agent": "test-agent",
                "confidence": 0.5,
                "session_id": "abc",
            }
        )
        assert m.claude_session_id == "abc"


# =============================================================================
# Timestamp Normalization
# =============================================================================


class TestTimestampNormalization:
    """Verify emitted_at -> created_at mapping and UTC enforcement."""

    @pytest.mark.unit
    def test_emitted_at_mapped_to_created_at(self) -> None:
        """emitted_at='2026-03-01T12:00:00Z' -> created_at == datetime(2026,3,1,12,0,0,UTC)."""
        m = ModelRoutingDecisionIngest.model_validate(
            {
                "correlation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "selected_agent": "test-agent",
                "confidence": 0.5,
                "emitted_at": "2026-03-01T12:00:00Z",
            }
        )
        expected = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)
        assert m.created_at == expected

    @pytest.mark.unit
    def test_created_at_is_utc_aware(self) -> None:
        """emitted_at without tz info -> created_at.tzinfo == UTC."""
        m = ModelRoutingDecisionIngest.model_validate(
            {
                "correlation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "selected_agent": "test-agent",
                "confidence": 0.5,
                "emitted_at": "2026-03-01T12:00:00",  # naive ISO string
            }
        )
        assert m.created_at.tzinfo is not None
        # _ensure_utc validator converts naive datetimes to UTC
        assert m.created_at.tzinfo == UTC

    @pytest.mark.unit
    def test_created_at_defaults_to_now_when_absent(self) -> None:
        """No timestamp in payload -> created_at is within 5s of UTC now."""
        before = datetime.now(UTC)
        m = ModelRoutingDecisionIngest.model_validate(
            {
                "correlation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "selected_agent": "test-agent",
                "confidence": 0.5,
            }
        )
        after = datetime.now(UTC)
        assert before <= m.created_at <= after

    @pytest.mark.unit
    def test_created_at_preserves_timezone_aware_emitted_at(self) -> None:
        """created_at already present in payload -> not overridden by emitted_at."""
        explicit = "2026-02-01T08:00:00+00:00"
        m = ModelRoutingDecisionIngest.model_validate(
            {
                "correlation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "selected_agent": "test-agent",
                "confidence": 0.5,
                "created_at": explicit,
                "emitted_at": "2026-03-01T12:00:00Z",  # should not override
            }
        )
        assert m.created_at.year == 2026
        assert m.created_at.month == 2


# =============================================================================
# ID Generation
# =============================================================================


class TestIdGeneration:
    """Verify id is auto-generated when absent from producer payload."""

    @pytest.mark.unit
    def test_id_auto_generated(self) -> None:
        """No 'id' in input -> valid UUID generated."""
        m = ModelRoutingDecisionIngest.model_validate(
            {
                "correlation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "selected_agent": "test-agent",
                "confidence": 0.5,
            }
        )
        assert isinstance(m.id, UUID)
        assert m.id != UUID(int=0)

    @pytest.mark.unit
    def test_id_preserved_when_present(self) -> None:
        """Explicit 'id' in input -> preserved on model."""
        explicit_id = "c3d4e5f6-a7b8-9012-cdef-123456789012"
        m = ModelRoutingDecisionIngest.model_validate(
            {
                "id": explicit_id,
                "correlation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "selected_agent": "test-agent",
                "confidence": 0.5,
            }
        )
        assert str(m.id) == explicit_id


# =============================================================================
# Confidence Clamping
# =============================================================================


class TestConfidenceClamping:
    """Verify confidence_score is clamped to [0.0, 1.0]."""

    @pytest.mark.unit
    def test_confidence_clamped_above_1(self) -> None:
        """confidence=1.5 -> confidence_score=1.0."""
        m = ModelRoutingDecisionIngest.model_validate(
            {
                "correlation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "selected_agent": "test-agent",
                "confidence": 1.5,
            }
        )
        assert m.confidence_score == 1.0

    @pytest.mark.unit
    def test_confidence_clamped_below_0(self) -> None:
        """confidence=-0.1 -> confidence_score=0.0."""
        m = ModelRoutingDecisionIngest.model_validate(
            {
                "correlation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "selected_agent": "test-agent",
                "confidence": -0.1,
            }
        )
        assert m.confidence_score == 0.0

    @pytest.mark.unit
    def test_confidence_non_numeric_defaults_0(self) -> None:
        """confidence='bad' -> confidence_score=0.0."""
        m = ModelRoutingDecisionIngest.model_validate(
            {
                "correlation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "selected_agent": "test-agent",
                "confidence": "bad",
            }
        )
        assert m.confidence_score == 0.0

    @pytest.mark.unit
    def test_confidence_none_defaults_0(self) -> None:
        """confidence=None -> confidence_score=0.0."""
        m = ModelRoutingDecisionIngest.model_validate(
            {
                "correlation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "selected_agent": "test-agent",
                "confidence": None,
            }
        )
        assert m.confidence_score == 0.0

    @pytest.mark.unit
    def test_confidence_exactly_1_preserved(self) -> None:
        """confidence=1.0 -> confidence_score=1.0 (not clamped)."""
        m = ModelRoutingDecisionIngest.model_validate(
            {
                "correlation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "selected_agent": "test-agent",
                "confidence": 1.0,
            }
        )
        assert m.confidence_score == 1.0


# =============================================================================
# Extra Field Tolerance
# =============================================================================


class TestExtraFieldTolerance:
    """Verify extra producer fields are silently ignored."""

    @pytest.mark.unit
    def test_extra_fields_ignored(self) -> None:
        """confidence_breakdown, event_attempted, etc. do not raise."""
        m = ModelRoutingDecisionIngest.model_validate(
            {
                "correlation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "selected_agent": "test-agent",
                "confidence": 0.8,
                "confidence_breakdown": {"semantic": 0.6, "rule": 0.2},
                "event_attempted": True,
                "routing_policy": "default",
                "routing_path": "semantic->fallback",
                "prompt_preview": "some prompt",
                "prompt_length": 100,
            }
        )
        assert m.selected_agent == "test-agent"

    @pytest.mark.unit
    def test_completely_unknown_fields_ignored(self) -> None:
        """Arbitrary unknown fields are silently ignored."""
        m = ModelRoutingDecisionIngest.model_validate(
            {
                "correlation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "selected_agent": "test-agent",
                "confidence": 0.5,
                "totally_unknown_field_xyz": "should be ignored",
                "another_future_field": 42,
            }
        )
        assert m.selected_agent == "test-agent"


# =============================================================================
# Input Dict Immutability
# =============================================================================


class TestInputImmutability:
    """Verify the input dict is not mutated during model validation."""

    @pytest.mark.unit
    def test_dict_not_mutated(self) -> None:
        """Original input dict unchanged after validation (emitted_at case)."""
        original: dict[str, object] = {
            "correlation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "selected_agent": "test-agent",
            "confidence": 0.8,
            "emitted_at": "2026-03-01T10:00:00Z",
        }
        original_copy = dict(original)
        ModelRoutingDecisionIngest.model_validate(original)
        assert original == original_copy, "Input dict was mutated during validation"


# =============================================================================
# Strict Model Still Rejects Unknown Fields
# =============================================================================


class TestStrictModelPreserved:
    """Verify ModelRoutingDecision still raises ValidationError on unknown fields."""

    @pytest.mark.unit
    def test_strict_model_rejects_unknown_fields(self) -> None:
        """ModelRoutingDecision raises ValidationError when confidence (not confidence_score) given."""
        with pytest.raises(ValidationError):
            ModelRoutingDecision.model_validate(
                {
                    "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                    "correlation_id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
                    "selected_agent": "test-agent",
                    "confidence": 0.9,  # wrong name — strict model forbids this
                    "created_at": "2026-03-01T12:00:00Z",
                }
            )

    @pytest.mark.unit
    def test_strict_model_accepts_correct_field_names(self) -> None:
        """ModelRoutingDecision validates when using correct internal field names."""
        m = ModelRoutingDecision.model_validate(
            {
                "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "correlation_id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
                "selected_agent": "test-agent",
                "confidence_score": 0.9,
                "created_at": "2026-03-01T12:00:00Z",
            }
        )
        assert m.confidence_score == pytest.approx(0.9)


# =============================================================================
# Ingest -> Strict Model Compatibility
# =============================================================================


class TestIngestToStrictModelCompatibility:
    """Verify ingest model dump can construct the strict model."""

    @pytest.mark.unit
    def test_ingest_model_dump_constructs_strict_model(self) -> None:
        """ingest.model_dump() -> ModelRoutingDecision.model_validate(...) succeeds."""
        ingest = ModelRoutingDecisionIngest.model_validate(
            {
                "correlation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "selected_agent": "polymorphic-agent",
                "confidence": 0.85,
                "session_id": "sess-xyz",
                "emitted_at": "2026-03-01T12:00:00Z",
                "reasoning": "matched pattern",
                "domain": "api",
                "routing_method": "SEMANTIC",
                "latency_ms": 37,
                # Extra fields that are ignored by ingest and NOT passed to strict
                "confidence_breakdown": {},
                "event_attempted": False,
            }
        )
        dump = ingest.model_dump()
        strict = ModelRoutingDecision.model_validate(dump)
        assert strict.selected_agent == "polymorphic-agent"
        assert strict.confidence_score == pytest.approx(0.85)
        assert strict.claude_session_id == "sess-xyz"
        assert strict.routing_reason == "matched pattern"
        assert strict.routing_method == "SEMANTIC"
        assert strict.latency_ms == 37


# =============================================================================
# Writer Field Presence
# =============================================================================


class TestWriterFieldPresence:
    """Verify all fields the writer accesses exist on ModelRoutingDecisionIngest."""

    # These are the field names accessed in writer_postgres.py write_routing_decisions
    WRITER_FIELDS = [
        "id",
        "correlation_id",
        "selected_agent",
        "confidence_score",
        "created_at",
        "request_type",
        "alternatives",
        "routing_reason",
        "domain",
        "metadata",
        "project_path",
        "project_name",
        "claude_session_id",
        "routing_method",
        "latency_ms",
    ]

    @pytest.mark.unit
    def test_writer_fields_present_on_ingest_model(self) -> None:
        """All 15 fields the writer accesses exist on ModelRoutingDecisionIngest."""
        m = ModelRoutingDecisionIngest.model_validate(
            {
                "correlation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "selected_agent": "test-agent",
                "confidence": 0.5,
            }
        )
        for field in self.WRITER_FIELDS:
            assert hasattr(m, field), (
                f"Writer field '{field}' missing from ingest model"
            )

    @pytest.mark.unit
    def test_writer_field_count_is_15(self) -> None:
        """Exactly 15 fields are needed by the writer (invariant: matches SQL placeholder count)."""
        assert len(self.WRITER_FIELDS) == 15


# =============================================================================
# SQL Placeholder Count (writer invariant)
# =============================================================================


class TestWriterPlaceholderCount:
    """Verify the INSERT SQL has exactly 15 placeholders matching the param tuple."""

    @pytest.mark.unit
    def test_write_routing_decisions_placeholder_count(self) -> None:
        """The routing_decisions INSERT SQL uses exactly 15 $N placeholders."""
        # Read the SQL from the writer source to detect drift
        import inspect

        from omnibase_infra.services.observability.agent_actions.writer_postgres import (
            WriterAgentActionsPostgres,
        )

        source = inspect.getsource(WriterAgentActionsPostgres.write_routing_decisions)
        # Extract the INSERT ... VALUES section
        # Count $N placeholders in the VALUES clause
        placeholders = re.findall(r"\$(\d+)", source)
        if placeholders:
            max_placeholder = max(int(p) for p in placeholders)
            assert max_placeholder == 15, (
                f"Expected 15 SQL placeholders in write_routing_decisions, "
                f"found max=${max_placeholder}. Update test and SQL together."
            )
