# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for savings estimation models.

Tests cover model validation, constraints, and serialization.

Related Tickets:
    - OMN-6964: Token savings emitter
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.node_savings_estimation_compute.models import (
    EnumModelTier,
    EnumSavingsCategory,
    ModelEffectivenessEntry,
    ModelSavingsCategory,
    ModelSavingsEstimate,
    ModelSavingsEstimationInput,
)


class TestModelEffectivenessEntry:
    """Validation tests for effectiveness entry model."""

    def test_valid_entry(self) -> None:
        entry = ModelEffectivenessEntry(
            utilization_score=0.85,
            patterns_count=12,
            tokens_saved=1500,
        )
        assert entry.utilization_score == 0.85
        assert entry.model_tier == EnumModelTier.OPUS  # default

    def test_utilization_score_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ModelEffectivenessEntry(
                utilization_score=1.5,
                patterns_count=5,
                tokens_saved=100,
            )
        with pytest.raises(ValidationError):
            ModelEffectivenessEntry(
                utilization_score=-0.1,
                patterns_count=5,
                tokens_saved=100,
            )

    def test_tokens_saved_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            ModelEffectivenessEntry(
                utilization_score=0.5,
                patterns_count=5,
                tokens_saved=-100,
            )

    def test_frozen(self) -> None:
        entry = ModelEffectivenessEntry(
            utilization_score=0.5,
            patterns_count=5,
            tokens_saved=100,
        )
        with pytest.raises(ValidationError):
            entry.tokens_saved = 200  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            ModelEffectivenessEntry(
                utilization_score=0.5,
                patterns_count=5,
                tokens_saved=100,
                extra_field="not_allowed",  # type: ignore[call-arg]
            )


class TestModelSavingsEstimationInput:
    """Validation tests for estimation input model."""

    def test_valid_input(self) -> None:
        entries = (
            ModelEffectivenessEntry(
                utilization_score=0.85,
                patterns_count=12,
                tokens_saved=1500,
            ),
        )
        inp = ModelSavingsEstimationInput(
            session_id="test-session",
            effectiveness_entries=entries,
        )
        assert inp.session_id == "test-session"
        assert len(inp.effectiveness_entries) == 1

    def test_empty_session_id_rejected(self) -> None:
        entries = (
            ModelEffectivenessEntry(
                utilization_score=0.85,
                patterns_count=12,
                tokens_saved=1500,
            ),
        )
        with pytest.raises(ValidationError):
            ModelSavingsEstimationInput(
                session_id="",
                effectiveness_entries=entries,
            )

    def test_empty_entries_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelSavingsEstimationInput(
                session_id="test",
                effectiveness_entries=(),
            )


class TestModelSavingsCategory:
    """Validation tests for savings category model."""

    def test_valid_category(self) -> None:
        cat = ModelSavingsCategory(
            category=EnumSavingsCategory.FILE,
            savings_usd=0.036,
            tokens_saved=2400,
            confidence=0.85,
        )
        assert cat.category == "file"

    def test_negative_savings_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelSavingsCategory(
                category=EnumSavingsCategory.FILE,
                savings_usd=-1.0,
                tokens_saved=100,
                confidence=0.5,
            )


class TestModelSavingsEstimate:
    """Validation tests for output estimate model."""

    def test_valid_estimate(self) -> None:
        est = ModelSavingsEstimate(
            session_id="test-session",
            direct_savings_usd=0.036,
            direct_tokens_saved=2400,
            direct_confidence=0.85,
        )
        assert est.session_id == "test-session"
        assert est.estimation_method == "tiered_attribution_v1"

    def test_defaults(self) -> None:
        est = ModelSavingsEstimate(session_id="test")
        assert est.actual_total_tokens == 0
        assert est.actual_cost_usd == 0.0
        assert est.is_measured is False
        assert est.completeness_status == "complete"
        assert est.schema_version == "1.0"

    def test_to_kafka_payload(self) -> None:
        cat = ModelSavingsCategory(
            category=EnumSavingsCategory.FILE,
            savings_usd=0.036,
            tokens_saved=2400,
            confidence=0.85,
        )
        est = ModelSavingsEstimate(
            session_id="test",
            categories=(cat,),
        )
        payload = est.to_kafka_payload()
        assert isinstance(payload["categories"], list)
        assert payload["categories"][0]["category"] == "file"
