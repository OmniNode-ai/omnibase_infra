# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ServiceAutoEvalRunner (OMN-6796).

Tests budget enforcement, task execution with mocked LLM responses,
error handling, and scoring heuristics.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.enums.enum_auto_eval_task_type import EnumAutoEvalTaskType
from omnibase_infra.models.eval.model_auto_eval_budget_cap import (
    ModelAutoEvalBudgetCap,
)
from omnibase_infra.models.eval.model_auto_eval_task import ModelAutoEvalTask
from omnibase_infra.services.service_auto_eval_runner import (
    ModelBudgetState,
    ServiceAutoEvalRunner,
)


def _make_task(
    task_type: EnumAutoEvalTaskType = EnumAutoEvalTaskType.CODE_GENERATION,
    prompt: str = "Write hello world",
    expected_output: str | None = None,
) -> ModelAutoEvalTask:
    return ModelAutoEvalTask(
        task_type=task_type,
        prompt=prompt,
        expected_output=expected_output,
        endpoint_url="http://localhost:8000",
        model_id="test-model",
    )


def _make_cap(
    max_cost_usd: float = 10.0,
    max_calls: int = 100,
    time_window_hours: float = 24.0,
) -> ModelAutoEvalBudgetCap:
    return ModelAutoEvalBudgetCap(
        max_cost_usd=max_cost_usd,
        max_calls=max_calls,
        time_window_hours=time_window_hours,
    )


class TestBudgetEnforcement:
    """Tests for budget cap enforcement."""

    def test_initial_state(self) -> None:
        runner = ServiceAutoEvalRunner(budget_cap=_make_cap())
        assert runner.budget_state.total_calls == 0
        assert runner.budget_state.total_cost_usd == 0.0

    @pytest.mark.asyncio
    async def test_budget_call_limit(self) -> None:
        runner = ServiceAutoEvalRunner(budget_cap=_make_cap(max_calls=1))
        task = _make_task()

        # Mock _call_llm to avoid real HTTP
        runner._call_llm = AsyncMock(return_value=("hello world", 100))  # type: ignore[method-assign]

        result1 = await runner.run_task(task)
        assert result1.error_message == ""
        assert runner.budget_state.total_calls == 1

        result2 = await runner.run_task(task)
        assert "Budget exhausted" in result2.error_message
        assert result2.score == 0.0

    @pytest.mark.asyncio
    async def test_budget_cost_limit(self) -> None:
        runner = ServiceAutoEvalRunner(budget_cap=_make_cap(max_cost_usd=0.0001))
        task = _make_task()

        runner._call_llm = AsyncMock(return_value=("output", 500_000))  # type: ignore[method-assign]

        result1 = await runner.run_task(task)
        assert result1.error_message == ""

        result2 = await runner.run_task(task)
        assert "Budget exhausted" in result2.error_message

    def test_window_reset(self) -> None:
        runner = ServiceAutoEvalRunner(budget_cap=_make_cap(time_window_hours=1.0))
        runner._budget_state = ModelBudgetState(
            total_calls=99,
            total_cost_usd=9.99,
            window_start=datetime.now(UTC) - timedelta(hours=2),
        )
        runner._reset_window_if_expired()
        assert runner.budget_state.total_calls == 0
        assert runner.budget_state.total_cost_usd == 0.0


class TestTaskExecution:
    """Tests for task execution logic."""

    @pytest.mark.asyncio
    async def test_successful_execution(self) -> None:
        runner = ServiceAutoEvalRunner(budget_cap=_make_cap())
        task = _make_task(expected_output="hello world")

        runner._call_llm = AsyncMock(return_value=("hello world", 50))  # type: ignore[method-assign]

        result = await runner.run_task(task)
        assert result.score == 1.0
        assert result.tokens_used == 50
        assert result.error_message == ""
        assert result.task_id == task.task_id

    @pytest.mark.asyncio
    async def test_llm_failure(self) -> None:
        runner = ServiceAutoEvalRunner(budget_cap=_make_cap())
        task = _make_task()

        runner._call_llm = AsyncMock(side_effect=ConnectionError("refused"))  # type: ignore[method-assign]

        result = await runner.run_task(task)
        assert result.score == 0.0
        assert "refused" in result.error_message
        assert result.latency_ms >= 0.0

    @pytest.mark.asyncio
    async def test_batch_execution(self) -> None:
        runner = ServiceAutoEvalRunner(budget_cap=_make_cap())
        tasks = [_make_task() for _ in range(3)]

        runner._call_llm = AsyncMock(return_value=("output", 10))  # type: ignore[method-assign]

        results = await runner.run_tasks(tasks)
        assert len(results) == 3
        assert runner.budget_state.total_calls == 3


class TestScoring:
    """Tests for output scoring logic."""

    def test_empty_output_scores_zero(self) -> None:
        task = _make_task()
        assert ServiceAutoEvalRunner._score_output(task, "") == 0.0
        assert ServiceAutoEvalRunner._score_output(task, "   ") == 0.0

    def test_no_expected_output_scores_one(self) -> None:
        task = _make_task(expected_output=None)
        assert ServiceAutoEvalRunner._score_output(task, "anything") == 1.0

    def test_exact_match(self) -> None:
        task = _make_task(expected_output="hello world")
        assert ServiceAutoEvalRunner._score_output(task, "hello world") == 1.0

    def test_substring_match(self) -> None:
        task = _make_task(expected_output="hello")
        assert ServiceAutoEvalRunner._score_output(task, "say hello there") == 1.0

    def test_partial_word_overlap(self) -> None:
        task = _make_task(expected_output="hello world foo bar")
        score = ServiceAutoEvalRunner._score_output(task, "hello world baz qux")
        assert 0.0 < score < 1.0

    def test_cost_estimation(self) -> None:
        cost = ServiceAutoEvalRunner._estimate_cost(1000)
        assert cost == pytest.approx(0.001)

        cost_zero = ServiceAutoEvalRunner._estimate_cost(0)
        assert cost_zero == 0.0
