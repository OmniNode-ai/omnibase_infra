# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Autonomous LLM evaluation runner service.

Executes ``ModelAutoEvalTask`` instances against configured LLM endpoints,
enforces budget caps, and returns ``ModelAutoEvalResult`` per task.

The runner uses httpx to call OpenAI-compatible ``/v1/chat/completions``
endpoints. Budget tracking is in-memory per instance; a Valkey-backed
implementation is deferred (see OMN-6796 description).

Related:
    - OMN-6796: Build eval runner service
    - OMN-6795: Eval task models and enums
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime

import httpx
from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums.enum_auto_eval_task_type import EnumAutoEvalTaskType
from omnibase_infra.models.eval.model_auto_eval_budget_cap import (
    ModelAutoEvalBudgetCap,
)
from omnibase_infra.models.eval.model_auto_eval_result import ModelAutoEvalResult
from omnibase_infra.models.eval.model_auto_eval_task import ModelAutoEvalTask

logger = logging.getLogger(__name__)


class ModelBudgetState(BaseModel):
    """In-memory budget tracking state for a single window.

    Attributes:
        total_cost_usd: Accumulated cost in the current window.
        total_calls: Number of LLM calls in the current window.
        window_start: Start time of the current budget window.
    """

    model_config = ConfigDict(extra="forbid")

    total_cost_usd: float = Field(default=0.0, ge=0.0)
    total_calls: int = Field(default=0, ge=0)
    window_start: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ServiceAutoEvalRunner:
    """Executes eval tasks against LLM endpoints with budget enforcement.

    Args:
        budget_cap: Budget constraints for the eval window.
        timeout_seconds: HTTP timeout for LLM API calls.
    """

    def __init__(
        self,
        budget_cap: ModelAutoEvalBudgetCap,
        timeout_seconds: float = 60.0,
    ) -> None:
        self._budget_cap = budget_cap
        self._timeout = timeout_seconds
        self._budget_state = ModelBudgetState()

    @property
    def budget_state(self) -> ModelBudgetState:
        """Current budget tracking state (read-only view)."""
        return self._budget_state

    def _reset_window_if_expired(self) -> None:
        """Reset the budget window if the current one has expired."""
        now = datetime.now(UTC)
        elapsed_hours = (now - self._budget_state.window_start).total_seconds() / 3600.0
        if elapsed_hours >= self._budget_cap.time_window_hours:
            self._budget_state = ModelBudgetState(window_start=now)

    def _check_budget(self) -> str | None:
        """Check if budget allows another call.

        Returns:
            None if budget is available, or an error message string.
        """
        self._reset_window_if_expired()
        if self._budget_state.total_calls >= self._budget_cap.max_calls:
            return (
                f"Budget exhausted: {self._budget_state.total_calls} calls "
                f"reached max {self._budget_cap.max_calls}"
            )
        if self._budget_state.total_cost_usd >= self._budget_cap.max_cost_usd:
            return (
                f"Budget exhausted: ${self._budget_state.total_cost_usd:.4f} "
                f"reached max ${self._budget_cap.max_cost_usd:.4f}"
            )
        return None

    def _record_usage(self, cost_usd: float) -> None:
        """Record a completed call against the budget."""
        self._budget_state = ModelBudgetState(
            total_cost_usd=self._budget_state.total_cost_usd + cost_usd,
            total_calls=self._budget_state.total_calls + 1,
            window_start=self._budget_state.window_start,
        )

    async def run_task(self, task: ModelAutoEvalTask) -> ModelAutoEvalResult:
        """Execute a single eval task against the specified LLM endpoint.

        Args:
            task: The evaluation task to execute.

        Returns:
            ModelAutoEvalResult with score, latency, token usage, and cost.
        """
        budget_error = self._check_budget()
        if budget_error is not None:
            logger.warning("Eval task %s rejected: %s", task.task_id, budget_error)
            return ModelAutoEvalResult(
                task_id=task.task_id,
                task_type=task.task_type,
                score=0.0,
                latency_ms=0.0,
                tokens_used=0,
                cost_usd=0.0,
                error_message=budget_error,
                completed_at=datetime.now(UTC),
                model_id=task.model_id,
                endpoint_url=task.endpoint_url,
            )

        start_time = time.monotonic()
        try:
            raw_output, tokens_used = await self._call_llm(task)
            latency_ms = (time.monotonic() - start_time) * 1000.0
            score = self._score_output(task, raw_output)
            cost_usd = self._estimate_cost(tokens_used)
            self._record_usage(cost_usd)

            return ModelAutoEvalResult(
                task_id=task.task_id,
                task_type=task.task_type,
                score=score,
                raw_output=raw_output,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
                cost_usd=cost_usd,
                completed_at=datetime.now(UTC),
                model_id=task.model_id,
                endpoint_url=task.endpoint_url,
            )
        except Exception as exc:
            latency_ms = (time.monotonic() - start_time) * 1000.0
            logger.exception("Eval task %s failed: %s", task.task_id, exc)
            return ModelAutoEvalResult(
                task_id=task.task_id,
                task_type=task.task_type,
                score=0.0,
                latency_ms=latency_ms,
                tokens_used=0,
                cost_usd=0.0,
                error_message=str(exc),
                completed_at=datetime.now(UTC),
                model_id=task.model_id,
                endpoint_url=task.endpoint_url,
            )

    async def run_tasks(
        self, tasks: list[ModelAutoEvalTask]
    ) -> list[ModelAutoEvalResult]:
        """Execute a batch of eval tasks sequentially.

        Args:
            tasks: List of evaluation tasks to execute.

        Returns:
            List of results, one per input task.
        """
        results: list[ModelAutoEvalResult] = []
        for task in tasks:
            result = await self.run_task(task)
            results.append(result)
        return results

    async def _call_llm(self, task: ModelAutoEvalTask) -> tuple[str, int]:
        """Call the LLM endpoint and return (output_text, total_tokens).

        Uses the OpenAI-compatible /v1/chat/completions API format.
        """
        url = f"{task.endpoint_url.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": task.model_id,
            "messages": [{"role": "user", "content": task.prompt}],
            "max_tokens": task.max_tokens,
            "temperature": 0.0,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

        choices = data.get("choices", [])
        output_text = ""
        if choices:
            message = choices[0].get("message", {})
            output_text = message.get("content", "")

        usage = data.get("usage", {})
        total_tokens = usage.get("total_tokens", 0)

        return output_text, total_tokens

    @staticmethod
    def _score_output(task: ModelAutoEvalTask, output: str) -> float:
        """Score the LLM output against the task criteria.

        For tasks with expected_output, uses simple substring containment
        as a baseline scoring heuristic. Returns 1.0 for tasks without
        expected_output (presence-only check).
        """
        if not output.strip():
            return 0.0

        if task.expected_output is None:
            return 1.0

        if task.task_type == EnumAutoEvalTaskType.EMBEDDING_QUALITY:
            return 1.0 if output.strip() else 0.0

        expected_lower = task.expected_output.lower()
        output_lower = output.lower()
        if expected_lower in output_lower:
            return 1.0
        # Partial match: check word overlap
        expected_words = set(expected_lower.split())
        output_words = set(output_lower.split())
        if not expected_words:
            return 1.0
        overlap = len(expected_words & output_words)
        return overlap / len(expected_words)

    @staticmethod
    def _estimate_cost(tokens_used: int) -> float:
        """Estimate cost in USD based on token count.

        Uses a flat rate of $0.001 per 1000 tokens as a conservative
        default for local/cheap models. Real cost varies by provider.
        """
        return (tokens_used / 1000.0) * 0.001


__all__: list[str] = ["ServiceAutoEvalRunner", "ModelBudgetState"]
