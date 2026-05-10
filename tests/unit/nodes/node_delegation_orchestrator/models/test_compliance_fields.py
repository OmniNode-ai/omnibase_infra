# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for tokens_to_compliance and compliance_attempts fields.

Tests cover both ModelDelegationResult and ModelTaskDelegatedEvent:
    - Defaults apply when fields are omitted
    - Explicit values are accepted
    - Serialization roundtrip preserves values
    - extra="forbid" rejects unknown fields

Related:
    - OMN-10790: Add compliance cost tracking fields
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.node_delegation_orchestrator.models.model_delegation_result import (
    ModelDelegationResult,
)
from omnibase_infra.nodes.node_delegation_orchestrator.models.model_task_delegated_event import (
    ModelTaskDelegatedEvent,
)

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_delegation_result(**overrides: object) -> ModelDelegationResult:
    """Build a valid ModelDelegationResult with sensible defaults."""
    defaults: dict[str, object] = {
        "correlation_id": uuid4(),
        "task_type": "test",
        "model_used": "Qwen3-Coder-30B-A3B",
        "endpoint_url": "http://192.168.86.201:8000",  # onex-allow-internal-ip
        "content": "def test_example():\n    assert True",
        "quality_passed": True,
        "quality_score": 0.95,
        "latency_ms": 1200,
        "fallback_to_claude": False,
    }
    defaults.update(overrides)
    return ModelDelegationResult(**defaults)  # type: ignore[arg-type]


def _valid_task_delegated_event(**overrides: object) -> ModelTaskDelegatedEvent:
    """Build a valid ModelTaskDelegatedEvent with sensible defaults."""
    defaults: dict[str, object] = {
        "timestamp": "2026-05-09T12:00:00Z",
        "correlation_id": uuid4(),
        "task_type": "code_generation",
        "delegated_to": "local-qwen3",
        "quality_gate_passed": True,
    }
    defaults.update(overrides)
    return ModelTaskDelegatedEvent(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ModelDelegationResult
# ---------------------------------------------------------------------------


class TestDelegationResultComplianceDefaults:
    """Verify default values when compliance fields are omitted."""

    def test_defaults_apply(self) -> None:
        result = _valid_delegation_result()
        assert result.tokens_to_compliance == 0
        assert result.compliance_attempts == 1

    def test_explicit_values(self) -> None:
        result = _valid_delegation_result(
            tokens_to_compliance=4500,
            compliance_attempts=3,
        )
        assert result.tokens_to_compliance == 4500
        assert result.compliance_attempts == 3

    def test_serialization_roundtrip(self) -> None:
        result = _valid_delegation_result(
            tokens_to_compliance=2000,
            compliance_attempts=2,
        )
        data = result.model_dump(mode="json")
        restored = ModelDelegationResult.model_validate(data)
        assert restored == result
        assert restored.tokens_to_compliance == 2000
        assert restored.compliance_attempts == 2

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError, match="extra_forbidden"):
            _valid_delegation_result(bogus_extra_field="nope")


# ---------------------------------------------------------------------------
# ModelTaskDelegatedEvent
# ---------------------------------------------------------------------------


class TestTaskDelegatedEventComplianceDefaults:
    """Verify default values when compliance fields are omitted."""

    def test_defaults_apply(self) -> None:
        event = _valid_task_delegated_event()
        assert event.tokens_to_compliance == 0
        assert event.compliance_attempts == 1

    def test_explicit_values(self) -> None:
        event = _valid_task_delegated_event(
            tokens_to_compliance=8000,
            compliance_attempts=5,
        )
        assert event.tokens_to_compliance == 8000
        assert event.compliance_attempts == 5

    def test_serialization_roundtrip(self) -> None:
        event = _valid_task_delegated_event(
            tokens_to_compliance=3500,
            compliance_attempts=4,
        )
        data = event.model_dump(mode="json")
        restored = ModelTaskDelegatedEvent.model_validate(data)
        assert restored == event
        assert restored.tokens_to_compliance == 3500
        assert restored.compliance_attempts == 4

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError, match="extra_forbidden"):
            _valid_task_delegated_event(bogus_extra_field="nope")
