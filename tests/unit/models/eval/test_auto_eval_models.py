# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for auto-eval models and enums (OMN-6795).

Validates construction, serialization round-trip, validation constraints,
and enum coverage for ModelAutoEvalTask, ModelAutoEvalResult,
ModelAutoEvalBudgetCap, and EnumAutoEvalTaskType.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from omnibase_infra.enums.enum_auto_eval_task_type import EnumAutoEvalTaskType
from omnibase_infra.models.eval.model_auto_eval_budget_cap import (
    ModelAutoEvalBudgetCap,
)
from omnibase_infra.models.eval.model_auto_eval_result import ModelAutoEvalResult
from omnibase_infra.models.eval.model_auto_eval_task import ModelAutoEvalTask


class TestEnumAutoEvalTaskType:
    """Tests for EnumAutoEvalTaskType."""

    def test_all_members_exist(self) -> None:
        members = {m.value for m in EnumAutoEvalTaskType}
        assert members == {
            "code_generation",
            "embedding_quality",
            "routing_accuracy",
            "reasoning_depth",
        }

    def test_string_value(self) -> None:
        assert (
            str(EnumAutoEvalTaskType.CODE_GENERATION)
            == "EnumAutoEvalTaskType.CODE_GENERATION"
        )
        assert EnumAutoEvalTaskType.CODE_GENERATION.value == "code_generation"


class TestModelAutoEvalTask:
    """Tests for ModelAutoEvalTask."""

    def test_construction_minimal(self) -> None:
        task = ModelAutoEvalTask(
            task_type=EnumAutoEvalTaskType.CODE_GENERATION,
            prompt="Write hello world in Python",
            endpoint_url="http://localhost:8000",
            model_id="test-model",
        )
        assert task.task_type == EnumAutoEvalTaskType.CODE_GENERATION
        assert task.prompt == "Write hello world in Python"
        assert task.max_tokens == 1024
        assert task.metadata == {}

    def test_serialization_roundtrip(self) -> None:
        task = ModelAutoEvalTask(
            task_type=EnumAutoEvalTaskType.EMBEDDING_QUALITY,
            prompt="Embed this text",
            expected_output="vector",
            endpoint_url="http://localhost:8100",
            model_id="embed-model",
            max_tokens=512,
            metadata={"key": "value"},
        )
        data = task.model_dump(mode="json")
        restored = ModelAutoEvalTask.model_validate(data)
        assert restored == task

    def test_frozen(self) -> None:
        task = ModelAutoEvalTask(
            task_type=EnumAutoEvalTaskType.CODE_GENERATION,
            prompt="test",
            endpoint_url="http://localhost:8000",
            model_id="m",
        )
        with pytest.raises(Exception):
            task.prompt = "changed"  # type: ignore[misc]

    def test_extra_forbid(self) -> None:
        with pytest.raises(Exception):
            ModelAutoEvalTask(
                task_type=EnumAutoEvalTaskType.CODE_GENERATION,
                prompt="test",
                endpoint_url="http://localhost:8000",
                model_id="m",
                unknown_field="bad",  # type: ignore[call-arg]
            )

    def test_empty_prompt_rejected(self) -> None:
        with pytest.raises(Exception):
            ModelAutoEvalTask(
                task_type=EnumAutoEvalTaskType.CODE_GENERATION,
                prompt="",
                endpoint_url="http://localhost:8000",
                model_id="m",
            )

    def test_max_tokens_bounds(self) -> None:
        with pytest.raises(Exception):
            ModelAutoEvalTask(
                task_type=EnumAutoEvalTaskType.CODE_GENERATION,
                prompt="test",
                endpoint_url="http://localhost:8000",
                model_id="m",
                max_tokens=0,
            )


class TestModelAutoEvalResult:
    """Tests for ModelAutoEvalResult."""

    def test_construction(self) -> None:
        now = datetime.now(UTC)
        result = ModelAutoEvalResult(
            task_id=uuid4(),
            task_type=EnumAutoEvalTaskType.REASONING_DEPTH,
            score=0.85,
            raw_output="analysis complete",
            latency_ms=1200.5,
            tokens_used=500,
            cost_usd=0.0005,
            completed_at=now,
            model_id="reasoning-model",
            endpoint_url="http://localhost:8101",
        )
        assert result.score == 0.85
        assert result.tokens_used == 500

    def test_serialization_roundtrip(self) -> None:
        now = datetime.now(UTC)
        result = ModelAutoEvalResult(
            task_id=uuid4(),
            task_type=EnumAutoEvalTaskType.ROUTING_ACCURACY,
            score=1.0,
            latency_ms=50.0,
            tokens_used=100,
            completed_at=now,
            model_id="router",
            endpoint_url="http://localhost:8001",
        )
        data = result.model_dump(mode="json")
        restored = ModelAutoEvalResult.model_validate(data)
        assert restored.score == result.score
        assert restored.task_id == result.task_id

    def test_score_bounds(self) -> None:
        with pytest.raises(Exception):
            ModelAutoEvalResult(
                task_id=uuid4(),
                task_type=EnumAutoEvalTaskType.CODE_GENERATION,
                score=1.5,
                latency_ms=0.0,
                tokens_used=0,
                completed_at=datetime.now(UTC),
                model_id="m",
                endpoint_url="http://localhost:8000",
            )

    def test_error_result(self) -> None:
        result = ModelAutoEvalResult(
            task_id=uuid4(),
            task_type=EnumAutoEvalTaskType.CODE_GENERATION,
            score=0.0,
            latency_ms=0.0,
            tokens_used=0,
            error_message="Connection refused",
            completed_at=datetime.now(UTC),
            model_id="m",
            endpoint_url="http://localhost:8000",
        )
        assert result.error_message == "Connection refused"
        assert result.score == 0.0


class TestModelAutoEvalBudgetCap:
    """Tests for ModelAutoEvalBudgetCap."""

    def test_construction(self) -> None:
        cap = ModelAutoEvalBudgetCap(
            max_cost_usd=10.0,
            max_calls=100,
            time_window_hours=24.0,
        )
        assert cap.max_cost_usd == 10.0
        assert cap.max_calls == 100
        assert cap.time_window_hours == 24.0

    def test_serialization_roundtrip(self) -> None:
        cap = ModelAutoEvalBudgetCap(
            max_cost_usd=5.0,
            max_calls=50,
            time_window_hours=12.0,
        )
        data = cap.model_dump(mode="json")
        restored = ModelAutoEvalBudgetCap.model_validate(data)
        assert restored == cap

    def test_zero_cost_rejected(self) -> None:
        with pytest.raises(Exception):
            ModelAutoEvalBudgetCap(
                max_cost_usd=0.0,
                max_calls=10,
            )

    def test_zero_calls_rejected(self) -> None:
        with pytest.raises(Exception):
            ModelAutoEvalBudgetCap(
                max_cost_usd=1.0,
                max_calls=0,
            )

    def test_default_window(self) -> None:
        cap = ModelAutoEvalBudgetCap(max_cost_usd=1.0, max_calls=10)
        assert cap.time_window_hours == 24.0
