# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for ModelRoutingDecision.

Tests cover:
    - Frozen immutability
    - extra="forbid" enforcement
    - Required fields validation
    - Serialization roundtrip

Related:
    - OMN-7040: Node-based delegation pipeline
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.node_delegation_routing_reducer.models.model_routing_decision import (
    ModelRoutingDecision,
)

pytestmark = [pytest.mark.unit]


def _valid_decision(**overrides: object) -> ModelRoutingDecision:
    """Build a valid ModelRoutingDecision with sensible defaults."""
    defaults: dict[str, object] = {
        "correlation_id": uuid4(),
        "task_type": "test",
        "selected_model": "qwen3-coder-30b",
        "selected_backend_id": str(uuid4()),
        "endpoint_url": "http://192.168.86.201:8000",
        "cost_tier": "low",
        "max_context_tokens": 65536,
        "system_prompt": "You are a test generation assistant.",
        "rationale": "Test tasks route to qwen3-coder-30b for long context.",
    }
    defaults.update(overrides)
    return ModelRoutingDecision(**defaults)  # type: ignore[arg-type]


class TestModelRoutingDecisionFrozen:
    """Verify ConfigDict(frozen=True) enforcement."""

    def test_frozen_rejects_field_mutation(self) -> None:
        decision = _valid_decision()
        with pytest.raises(ValidationError):
            decision.selected_model = "other"  # type: ignore[misc]


class TestModelRoutingDecisionExtraForbid:
    """Verify extra='forbid' enforcement."""

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError, match="extra_forbidden"):
            _valid_decision(unknown_field="bad")


class TestModelRoutingDecisionRequiredFields:
    """Verify required field validation."""

    def test_missing_correlation_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            ModelRoutingDecision(
                task_type="test",
                selected_model="model",
                selected_backend_id=uuid4(),
                endpoint_url="http://localhost:8000",
                cost_tier="low",
                max_context_tokens=65536,
                system_prompt="prompt",
                rationale="reason",
            )  # type: ignore[call-arg]

    def test_missing_endpoint_url_raises(self) -> None:
        with pytest.raises(ValidationError):
            ModelRoutingDecision(
                correlation_id=uuid4(),
                task_type="test",
                selected_model="model",
                selected_backend_id=uuid4(),
                cost_tier="low",
                max_context_tokens=65536,
                system_prompt="prompt",
                rationale="reason",
            )  # type: ignore[call-arg]

    def test_missing_system_prompt_raises(self) -> None:
        with pytest.raises(ValidationError):
            ModelRoutingDecision(
                correlation_id=uuid4(),
                task_type="test",
                selected_model="model",
                selected_backend_id=uuid4(),
                endpoint_url="http://localhost:8000",
                cost_tier="low",
                max_context_tokens=65536,
                rationale="reason",
            )  # type: ignore[call-arg]


class TestModelRoutingDecisionSerialization:
    """Verify serialization roundtrip."""

    def test_roundtrip(self) -> None:
        decision = _valid_decision()
        data = decision.model_dump(mode="json")
        restored = ModelRoutingDecision.model_validate(data)
        assert restored == decision

    def test_from_attributes(self) -> None:
        decision = _valid_decision()
        reconstructed = ModelRoutingDecision.model_validate(
            decision, from_attributes=True
        )
        assert reconstructed == decision
