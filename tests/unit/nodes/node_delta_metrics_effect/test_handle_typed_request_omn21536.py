# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed-request parity scenarios for the delta metrics upsert (OMN-21536)."""

from __future__ import annotations

from datetime import date
from typing import Literal
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from pydantic import BaseModel, ConfigDict

from omnibase_infra.enums import EnumPostgresErrorCode
from omnibase_infra.errors import InfraConnectionError, RepositoryExecutionError
from omnibase_infra.models.model_backend_result import ModelBackendResult
from omnibase_infra.nodes.node_delta_metrics_effect.handlers.handler_upsert_metrics import (
    SQL_UPSERT_METRICS,
    HandlerUpsertMetrics,
)

pytestmark = pytest.mark.unit

HANDLER_MODULE: str = (
    "omnibase_infra.nodes.node_delta_metrics_effect.handlers.handler_upsert_metrics"
)
PRESERVED_SYMBOLS: tuple[str, ...] = ("HandlerUpsertMetrics._execute_upsert",)


class ModelMetricsScenario(BaseModel):
    """One deterministic rollup contribution and database response."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    seed: int
    outcome: Literal["merged", "reverted", "closed"] = "merged"
    gate_decision: Literal["PASS", "WARN", "QUARANTINE"] = "PASS"
    is_fix_pr: bool = False
    gate_violation_count: int = 0
    failure: Literal["none", "timeout", "connection", "query"] = "none"

    @property
    def correlation_id(self) -> UUID:
        return UUID(int=self.seed)


SCENARIOS: tuple[ModelMetricsScenario, ...] = (
    ModelMetricsScenario(seed=1),
    ModelMetricsScenario(
        seed=2,
        outcome="reverted",
        gate_decision="QUARANTINE",
        is_fix_pr=True,
        gate_violation_count=3,
    ),
    ModelMetricsScenario(seed=3, outcome="closed", gate_decision="WARN"),
    ModelMetricsScenario(seed=4, failure="timeout"),
    ModelMetricsScenario(seed=5, failure="connection"),
    ModelMetricsScenario(seed=6, failure="query"),
)


async def run_scenario(scenario: ModelMetricsScenario) -> ModelBackendResult:
    """Execute the real upsert helper through a single typed request."""
    try:
        from omnibase_infra.nodes.node_delta_metrics_effect.models import (
            ModelPayloadUpsertMetrics,
        )
    except ImportError as exc:
        pytest.fail(f"the typed metrics request model is missing: {exc}")

    request = ModelPayloadUpsertMetrics(
        correlation_id=scenario.correlation_id,
        coding_model="test-model",
        subsystem="omnibase_infra",
        outcome=scenario.outcome,
        gate_decision=scenario.gate_decision,
        is_fix_pr=scenario.is_fix_pr,
        gate_violation_count=scenario.gate_violation_count,
        period_start=date(2026, 1, 1),
        period_end=date(2026, 1, 31),
    )
    conn = AsyncMock()
    if scenario.failure == "timeout":
        conn.execute.side_effect = TimeoutError("metrics timed out")
    elif scenario.failure == "connection":
        conn.execute.side_effect = InfraConnectionError("metrics unavailable")
    elif scenario.failure == "query":
        conn.execute.side_effect = RepositoryExecutionError("metrics query failed")
    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    handler = HandlerUpsertMetrics(pool)
    try:
        result = await handler.handle(request)
    except TypeError as exc:
        pytest.fail(f"handle() does not accept the single request model: {exc}")

    pool.acquire.assert_called_once_with()
    conn.execute.assert_awaited_once_with(
        SQL_UPSERT_METRICS,
        "test-model",
        "omnibase_infra",
        scenario.outcome,
        scenario.gate_decision,
        scenario.is_fix_pr,
        scenario.gate_violation_count,
        date(2026, 1, 1),
        date(2026, 1, 31),
    )
    assert result.correlation_id == scenario.correlation_id
    assert result.backend_id == "postgres"
    return result


class TestUpsertMetricsHandleTypedRequest:
    """The typed request preserves rollup values and error classification."""

    @pytest.mark.asyncio
    async def test_merged_contribution_succeeds(self) -> None:
        result = await run_scenario(SCENARIOS[0])
        assert result.success is True
        assert result.error_code is None

    @pytest.mark.asyncio
    async def test_quarantined_reverted_fix_preserves_counters(self) -> None:
        result = await run_scenario(SCENARIOS[1])
        assert result.success is True
        assert result.error is None

    @pytest.mark.asyncio
    async def test_closed_warn_with_zero_violations_succeeds(self) -> None:
        result = await run_scenario(SCENARIOS[2])
        assert result.success is True
        assert result.error_code is None

    @pytest.mark.asyncio
    async def test_timeout_returns_timeout_error(self) -> None:
        result = await run_scenario(SCENARIOS[3])
        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.TIMEOUT_ERROR

    @pytest.mark.asyncio
    async def test_connection_failure_returns_connection_error(self) -> None:
        result = await run_scenario(SCENARIOS[4])
        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.CONNECTION_ERROR

    @pytest.mark.asyncio
    async def test_query_failure_returns_upsert_error(self) -> None:
        result = await run_scenario(SCENARIOS[5])
        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.UPSERT_ERROR
