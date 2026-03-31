# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for ModelDelegationResult.

Tests cover:
    - Frozen immutability
    - extra="forbid" enforcement
    - Required fields validation
    - Default values for token fields
    - Serialization roundtrip

Related:
    - OMN-7040: Node-based delegation pipeline
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.node_delegation_orchestrator.models.model_delegation_result import (
    ModelDelegationResult,
)

pytestmark = [pytest.mark.unit]


def _valid_result(**overrides: object) -> ModelDelegationResult:
    """Build a valid ModelDelegationResult with sensible defaults."""
    defaults: dict[str, object] = {
        "correlation_id": uuid4(),
        "task_type": "test",
        "model_used": "Qwen3-Coder-30B-A3B",
        "endpoint_url": "http://192.168.86.201:8000",
        "content": "def test_example():\n    assert True",
        "quality_passed": True,
        "quality_score": 0.95,
        "latency_ms": 1200,
        "fallback_to_claude": False,
    }
    defaults.update(overrides)
    return ModelDelegationResult(**defaults)  # type: ignore[arg-type]


class TestModelDelegationResultFrozen:
    """Verify ConfigDict(frozen=True) enforcement."""

    def test_frozen_rejects_field_mutation(self) -> None:
        result = _valid_result()
        with pytest.raises(ValidationError):
            result.content = "new content"  # type: ignore[misc]


class TestModelDelegationResultExtraForbid:
    """Verify extra='forbid' enforcement."""

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError, match="extra_forbidden"):
            _valid_result(unknown_field="bad")


class TestModelDelegationResultRequiredFields:
    """Verify required field validation."""

    def test_missing_correlation_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            ModelDelegationResult(
                task_type="test",
                model_used="model",
                endpoint_url="http://localhost:8000",
                content="output",
                quality_passed=True,
                quality_score=0.9,
                latency_ms=100,
                fallback_to_claude=False,
            )  # type: ignore[call-arg]

    def test_missing_content_raises(self) -> None:
        with pytest.raises(ValidationError):
            ModelDelegationResult(
                correlation_id=uuid4(),
                task_type="test",
                model_used="model",
                endpoint_url="http://localhost:8000",
                quality_passed=True,
                quality_score=0.9,
                latency_ms=100,
                fallback_to_claude=False,
            )  # type: ignore[call-arg]


class TestModelDelegationResultDefaults:
    """Verify default values for optional fields."""

    def test_token_defaults_zero(self) -> None:
        result = _valid_result()
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        assert result.total_tokens == 0

    def test_failure_reason_default_empty(self) -> None:
        result = _valid_result()
        assert result.failure_reason == ""

    def test_custom_token_values(self) -> None:
        result = _valid_result(
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300,
        )
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 200
        assert result.total_tokens == 300


class TestModelDelegationResultSerialization:
    """Verify serialization roundtrip."""

    def test_roundtrip(self) -> None:
        result = _valid_result(
            prompt_tokens=50,
            completion_tokens=150,
            total_tokens=200,
            failure_reason="quality gate failed",
        )
        data = result.model_dump(mode="json")
        restored = ModelDelegationResult.model_validate(data)
        assert restored == result

    def test_from_attributes(self) -> None:
        result = _valid_result()
        reconstructed = ModelDelegationResult.model_validate(
            result, from_attributes=True
        )
        assert reconstructed == result
