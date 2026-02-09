# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for injection effectiveness query models.

Tests model validation, immutability, and default values.

Related Tickets:
    - OMN-2078: Golden path: injection metrics + ledger storage
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.services.observability.injection_effectiveness.models import (
    ModelInjectionEffectivenessQuery,
    ModelInjectionEffectivenessQueryResult,
    ModelInjectionEffectivenessRow,
    ModelLatencyBreakdownRow,
    ModelPatternHitRateRow,
)


class TestModelInjectionEffectivenessRow:
    """Tests for ModelInjectionEffectivenessRow."""

    def test_minimal_required_fields(self) -> None:
        """Can create with only required fields."""
        now = datetime.now(UTC)
        row = ModelInjectionEffectivenessRow(
            session_id=uuid4(),
            created_at=now,
            updated_at=now,
        )
        assert row.correlation_id is None
        assert row.cohort is None
        assert row.utilization_score is None

    def test_all_fields(self) -> None:
        """Can create with all fields populated."""
        now = datetime.now(UTC)
        sid = uuid4()
        row = ModelInjectionEffectivenessRow(
            session_id=sid,
            correlation_id=uuid4(),
            realm="production",
            runtime_id="runtime-1",
            routing_path="event",
            cohort="treatment",
            cohort_identity_type="session_id",
            total_injected_tokens=1500,
            patterns_injected=3,
            utilization_score=0.85,
            utilization_method="identifier_match",
            injected_identifiers_count=10,
            reused_identifiers_count=7,
            agent_match_score=0.9,
            expected_agent="coder",
            actual_agent="coder",
            user_visible_latency_ms=350,
            created_at=now,
            updated_at=now,
        )
        assert row.session_id == sid
        assert row.utilization_score == 0.85

    def test_frozen(self) -> None:
        """Row model is immutable."""
        now = datetime.now(UTC)
        row = ModelInjectionEffectivenessRow(
            session_id=uuid4(),
            created_at=now,
            updated_at=now,
        )
        with pytest.raises(ValidationError):
            row.cohort = "control"  # type: ignore[misc]

    def test_extra_forbid(self) -> None:
        """Rejects unknown fields."""
        now = datetime.now(UTC)
        with pytest.raises(ValidationError):
            ModelInjectionEffectivenessRow(
                session_id=uuid4(),
                created_at=now,
                updated_at=now,
                unknown_field="value",
            )


class TestModelLatencyBreakdownRow:
    """Tests for ModelLatencyBreakdownRow."""

    def test_required_fields(self) -> None:
        """Validates required fields are enforced."""
        now = datetime.now(UTC)
        row = ModelLatencyBreakdownRow(
            id=uuid4(),
            session_id=uuid4(),
            prompt_id=uuid4(),
            user_latency_ms=350,
            created_at=now,
        )
        assert row.cache_hit is False
        assert row.routing_latency_ms is None

    def test_frozen(self) -> None:
        """Row model is immutable."""
        now = datetime.now(UTC)
        row = ModelLatencyBreakdownRow(
            id=uuid4(),
            session_id=uuid4(),
            prompt_id=uuid4(),
            user_latency_ms=350,
            created_at=now,
        )
        with pytest.raises(ValidationError):
            row.user_latency_ms = 100  # type: ignore[misc]


class TestModelPatternHitRateRow:
    """Tests for ModelPatternHitRateRow."""

    def test_required_fields(self) -> None:
        """Validates required fields."""
        now = datetime.now(UTC)
        row = ModelPatternHitRateRow(
            id=uuid4(),
            pattern_id=uuid4(),
            utilization_method="identifier_match",
            utilization_score=0.72,
            created_at=now,
            updated_at=now,
        )
        assert row.confidence is None
        assert row.domain_id is None

    def test_confidence_nullable(self) -> None:
        """Confidence can be None (insufficient samples)."""
        now = datetime.now(UTC)
        row = ModelPatternHitRateRow(
            id=uuid4(),
            pattern_id=uuid4(),
            utilization_method="semantic",
            utilization_score=0.5,
            sample_count=10,
            confidence=None,
            created_at=now,
            updated_at=now,
        )
        assert row.confidence is None


class TestModelInjectionEffectivenessQuery:
    """Tests for ModelInjectionEffectivenessQuery."""

    def test_defaults(self) -> None:
        """All filters default to None, pagination has sensible defaults."""
        query = ModelInjectionEffectivenessQuery()
        assert query.session_id is None
        assert query.cohort is None
        assert query.limit == 100
        assert query.offset == 0

    def test_limit_validation(self) -> None:
        """Validates limit bounds."""
        with pytest.raises(ValidationError):
            ModelInjectionEffectivenessQuery(limit=0)

        with pytest.raises(ValidationError):
            ModelInjectionEffectivenessQuery(limit=10001)

    def test_cohort_literal(self) -> None:
        """Only accepts valid cohort values."""
        q = ModelInjectionEffectivenessQuery(cohort="control")
        assert q.cohort == "control"

        q = ModelInjectionEffectivenessQuery(cohort="treatment")
        assert q.cohort == "treatment"

        with pytest.raises(ValidationError):
            ModelInjectionEffectivenessQuery(cohort="invalid")

    def test_frozen(self) -> None:
        """Query model is immutable."""
        query = ModelInjectionEffectivenessQuery()
        with pytest.raises(ValidationError):
            query.limit = 50  # type: ignore[misc]


class TestModelInjectionEffectivenessQueryResult:
    """Tests for ModelInjectionEffectivenessQueryResult."""

    def test_empty_result(self) -> None:
        """Can create empty result."""
        query = ModelInjectionEffectivenessQuery()
        result = ModelInjectionEffectivenessQueryResult(
            rows=(),
            total_count=0,
            has_more=False,
            query=query,
        )
        assert result.rows == ()
        assert result.total_count == 0
        assert result.has_more is False

    def test_with_rows(self) -> None:
        """Can create result with rows."""
        now = datetime.now(UTC)
        row = ModelInjectionEffectivenessRow(
            session_id=uuid4(),
            created_at=now,
            updated_at=now,
        )
        query = ModelInjectionEffectivenessQuery()
        result = ModelInjectionEffectivenessQueryResult(
            rows=(row,),
            total_count=1,
            has_more=False,
            query=query,
        )
        assert len(result.rows) == 1
