# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for ModelQualityGateInput and ModelQualityGateResult.

Tests cover:
    - Frozen immutability
    - extra="forbid" enforcement
    - Required fields validation
    - Default values
    - Serialization roundtrip

Related:
    - OMN-7040: Node-based delegation pipeline
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.node_delegation_quality_gate_reducer.models.model_quality_gate_input import (
    ModelQualityGateInput,
)
from omnibase_infra.nodes.node_delegation_quality_gate_reducer.models.model_quality_gate_result import (
    ModelQualityGateResult,
)

pytestmark = [pytest.mark.unit]


# =============================================================================
# ModelQualityGateInput
# =============================================================================


def _valid_input(**overrides: object) -> ModelQualityGateInput:
    """Build a valid ModelQualityGateInput with sensible defaults."""
    defaults: dict[str, object] = {
        "correlation_id": uuid4(),
        "task_type": "test",
        "llm_response_content": "def test_example():\n    assert True\n",
    }
    defaults.update(overrides)
    return ModelQualityGateInput(**defaults)  # type: ignore[arg-type]


class TestModelQualityGateInputFrozen:
    """Verify ConfigDict(frozen=True) enforcement."""

    def test_frozen_rejects_field_mutation(self) -> None:
        gate_input = _valid_input()
        with pytest.raises(ValidationError):
            gate_input.task_type = "document"  # type: ignore[misc]


class TestModelQualityGateInputExtraForbid:
    """Verify extra='forbid' enforcement."""

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError, match="extra_forbidden"):
            _valid_input(unknown_field="bad")


class TestModelQualityGateInputRequiredFields:
    """Verify required field validation."""

    def test_missing_correlation_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            ModelQualityGateInput(
                task_type="test",
                llm_response_content="output",
            )  # type: ignore[call-arg]

    def test_missing_task_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            ModelQualityGateInput(
                correlation_id=uuid4(),
                llm_response_content="output",
            )  # type: ignore[call-arg]

    def test_missing_llm_response_content_raises(self) -> None:
        with pytest.raises(ValidationError):
            ModelQualityGateInput(
                correlation_id=uuid4(),
                task_type="test",
            )  # type: ignore[call-arg]


class TestModelQualityGateInputDefaults:
    """Verify default values."""

    def test_expected_markers_default_empty(self) -> None:
        gate_input = _valid_input()
        assert gate_input.expected_markers == ()

    def test_min_response_length_default_60(self) -> None:
        gate_input = _valid_input()
        assert gate_input.min_response_length == 60

    def test_custom_expected_markers(self) -> None:
        gate_input = _valid_input(expected_markers=("def test_", "@pytest.mark"))
        assert gate_input.expected_markers == ("def test_", "@pytest.mark")

    def test_custom_min_response_length(self) -> None:
        gate_input = _valid_input(min_response_length=100)
        assert gate_input.min_response_length == 100


class TestModelQualityGateInputSerialization:
    """Verify serialization roundtrip."""

    def test_roundtrip(self) -> None:
        gate_input = _valid_input(
            expected_markers=("def test_",),
            min_response_length=80,
        )
        data = gate_input.model_dump(mode="json")
        restored = ModelQualityGateInput.model_validate(data)
        assert restored == gate_input

    def test_from_attributes(self) -> None:
        gate_input = _valid_input()
        reconstructed = ModelQualityGateInput.model_validate(
            gate_input, from_attributes=True
        )
        assert reconstructed == gate_input


# =============================================================================
# ModelQualityGateResult
# =============================================================================


def _valid_result(**overrides: object) -> ModelQualityGateResult:
    """Build a valid ModelQualityGateResult with sensible defaults."""
    defaults: dict[str, object] = {
        "correlation_id": uuid4(),
        "passed": True,
        "quality_score": 0.9,
    }
    defaults.update(overrides)
    return ModelQualityGateResult(**defaults)  # type: ignore[arg-type]


class TestModelQualityGateResultFrozen:
    """Verify ConfigDict(frozen=True) enforcement."""

    def test_frozen_rejects_field_mutation(self) -> None:
        result = _valid_result()
        with pytest.raises(ValidationError):
            result.passed = False  # type: ignore[misc]


class TestModelQualityGateResultExtraForbid:
    """Verify extra='forbid' enforcement."""

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError, match="extra_forbidden"):
            _valid_result(unknown_field="bad")


class TestModelQualityGateResultRequiredFields:
    """Verify required field validation."""

    def test_missing_correlation_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            ModelQualityGateResult(
                passed=True,
                quality_score=0.9,
            )  # type: ignore[call-arg]

    def test_missing_passed_raises(self) -> None:
        with pytest.raises(ValidationError):
            ModelQualityGateResult(
                correlation_id=uuid4(),
                quality_score=0.9,
            )  # type: ignore[call-arg]

    def test_missing_quality_score_raises(self) -> None:
        with pytest.raises(ValidationError):
            ModelQualityGateResult(
                correlation_id=uuid4(),
                passed=True,
            )  # type: ignore[call-arg]


class TestModelQualityGateResultDefaults:
    """Verify default values."""

    def test_failure_reasons_default_empty(self) -> None:
        result = _valid_result()
        assert result.failure_reasons == ()

    def test_fallback_recommended_default_false(self) -> None:
        result = _valid_result()
        assert result.fallback_recommended is False

    def test_custom_failure_reasons(self) -> None:
        result = _valid_result(
            passed=False,
            quality_score=0.2,
            failure_reasons=("REFUSAL: response contains refusal phrase",),
            fallback_recommended=True,
        )
        assert len(result.failure_reasons) == 1
        assert result.fallback_recommended is True


class TestModelQualityGateResultSerialization:
    """Verify serialization roundtrip."""

    def test_roundtrip(self) -> None:
        result = _valid_result(
            passed=False,
            quality_score=0.3,
            failure_reasons=("WEAK_OUTPUT: too short", "TASK_MISMATCH: no markers"),
            fallback_recommended=True,
        )
        data = result.model_dump(mode="json")
        restored = ModelQualityGateResult.model_validate(data)
        assert restored == result

    def test_from_attributes(self) -> None:
        result = _valid_result()
        reconstructed = ModelQualityGateResult.model_validate(
            result, from_attributes=True
        )
        assert reconstructed == result
