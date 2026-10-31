# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Parity tests for the single-payload decision-store handlers (OMN-21536).

Deterministic scenarios exercise the real SQL helpers with mocked connections.
The runners also serve as recorder inputs; the old two-argument handle() fails
each scenario explicitly instead of failing during collection.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from pydantic import BaseModel, ConfigDict

from omnibase_infra.enums import EnumPostgresErrorCode
from omnibase_infra.errors import InfraConnectionError, RepositoryExecutionError
from omnibase_infra.models.model_backend_result import ModelBackendResult
from omnibase_infra.nodes.node_decision_store_effect.handlers.handler_write_conflict import (
    SQL_INSERT_CONFLICT,
    HandlerWriteConflict,
)
from omnibase_infra.nodes.node_decision_store_effect.handlers.handler_write_decision import (
    SQL_ACTIVE_SAME_SCOPE,
    SQL_FETCH_OLD_STATUS,
    SQL_NOW,
    SQL_UPSERT_DECISION,
    HandlerWriteDecision,
)

pytestmark = pytest.mark.unit

HANDLER_MODULE: str = (
    "omnibase_infra.nodes.node_decision_store_effect.handlers.handler_write_conflict"
)
PRESERVED_SYMBOLS: tuple[str, ...] = ("HandlerWriteConflict._execute_insert",)


class ModelWriteConflictScenario(BaseModel):
    """A conflict pair, including ordering and database outcome branches."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    seed: int
    first_id: int = 10
    second_id: int = 20
    confidence: float = 0.9
    severity: Literal["HIGH", "MEDIUM", "LOW"] = "HIGH"
    outcome: Literal["insert", "skip", "timeout", "connection", "query"] = "insert"

    @property
    def correlation_id(self) -> UUID:
        return UUID(int=self.seed)


SCENARIOS: tuple[ModelWriteConflictScenario, ...] = (
    ModelWriteConflictScenario(seed=1),
    ModelWriteConflictScenario(
        seed=2, first_id=20, second_id=10, confidence=0.7, severity="MEDIUM"
    ),
    ModelWriteConflictScenario(seed=3, confidence=0.0, severity="LOW", outcome="skip"),
    ModelWriteConflictScenario(seed=4, outcome="timeout"),
    ModelWriteConflictScenario(seed=5, outcome="connection"),
    ModelWriteConflictScenario(seed=6, outcome="query"),
)


def _acquire_context(conn: AsyncMock) -> AsyncMock:
    """Wrap a connection in the async context manager returned by acquire()."""
    context = AsyncMock()
    context.__aenter__.return_value = conn
    context.__aexit__.return_value = None
    return context


async def run_scenario(scenario: ModelWriteConflictScenario) -> ModelBackendResult:
    """Run the actual conflict insert via one typed request argument."""
    try:
        from omnibase_infra.nodes.node_decision_store_effect.models import (
            ModelPayloadWriteConflict,
        )
    except ImportError as exc:
        pytest.fail(f"the typed conflict request model is missing: {exc}")

    request = ModelPayloadWriteConflict(
        correlation_id=scenario.correlation_id,
        decision_min_id=UUID(int=scenario.first_id),
        decision_max_id=UUID(int=scenario.second_id),
        structural_confidence=scenario.confidence,
        final_severity=scenario.severity,
    )
    conn = AsyncMock()
    conn.execute.return_value = (
        "INSERT 0 0" if scenario.outcome == "skip" else "INSERT 0 1"
    )
    if scenario.outcome == "timeout":
        conn.execute.side_effect = TimeoutError("deterministic insert timeout")
    elif scenario.outcome == "connection":
        conn.execute.side_effect = InfraConnectionError(
            "deterministic connection failure"
        )
    elif scenario.outcome == "query":
        conn.execute.side_effect = RepositoryExecutionError(
            "deterministic query failure"
        )
    pool = MagicMock()
    pool.acquire.return_value = _acquire_context(conn)
    handler = HandlerWriteConflict(pool)
    try:
        result = await handler.handle(request)
    except TypeError as exc:
        pytest.fail(f"handle() does not accept the single request model: {exc}")

    low, high = sorted((request.decision_min_id, request.decision_max_id))
    conn.execute.assert_awaited_once_with(
        SQL_INSERT_CONFLICT, low, high, scenario.confidence, scenario.severity
    )
    pool.acquire.assert_called_once_with()
    assert result.correlation_id == scenario.correlation_id
    assert result.backend_id == "postgres"
    return result


class TestWriteConflictHandleTypedRequest:
    """Single-payload calls preserve inserts, idempotency, and error codes."""

    @pytest.mark.asyncio
    async def test_insert_succeeds(self) -> None:
        result = await run_scenario(SCENARIOS[0])
        assert result.success is True
        assert result.error_code is None
        assert result.error is None

    @pytest.mark.asyncio
    async def test_reversed_pair_is_normalized(self) -> None:
        result = await run_scenario(SCENARIOS[1])
        assert result.success is True
        assert result.error_code is None

    @pytest.mark.asyncio
    async def test_duplicate_with_zero_confidence_succeeds(self) -> None:
        result = await run_scenario(SCENARIOS[2])
        assert result.success is True
        assert result.error_code is None

    @pytest.mark.asyncio
    async def test_timeout_is_classified(self) -> None:
        result = await run_scenario(SCENARIOS[3])
        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.TIMEOUT_ERROR

    @pytest.mark.asyncio
    async def test_connection_failure_is_classified(self) -> None:
        result = await run_scenario(SCENARIOS[4])
        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.CONNECTION_ERROR

    @pytest.mark.asyncio
    async def test_query_failure_uses_upsert_error(self) -> None:
        result = await run_scenario(SCENARIOS[5])
        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.UPSERT_ERROR


class ModelWriteDecisionScenario(BaseModel):
    """An insert or update through the two-stage decision writer."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    seed: int
    outcome: Literal["insert", "update", "stage2_failure", "timeout"]

    @property
    def correlation_id(self) -> UUID:
        return UUID(int=self.seed)


WRITE_DECISION_SCENARIOS: tuple[ModelWriteDecisionScenario, ...] = (
    ModelWriteDecisionScenario(seed=101, outcome="insert"),
    ModelWriteDecisionScenario(seed=102, outcome="update"),
    ModelWriteDecisionScenario(seed=103, outcome="stage2_failure"),
    ModelWriteDecisionScenario(seed=104, outcome="timeout"),
)


async def run_write_decision_scenario(
    scenario: ModelWriteDecisionScenario,
) -> ModelBackendResult:
    """Exercise decision writes without changing their transaction helpers."""
    try:
        from omnibase_infra.nodes.node_decision_store_effect.models import (
            ModelPayloadWriteDecision,
        )
    except ImportError as exc:
        pytest.fail(f"the typed decision request model is missing: {exc}")

    now = datetime(2026, 2, 25, 12, tzinfo=UTC)
    request = ModelPayloadWriteDecision(
        correlation_id=scenario.correlation_id,
        decision_id=UUID(int=scenario.seed + 1000),
        title="Deterministic decision",
        decision_type="DESIGN_PATTERN",
        scope_domain="INFRA",
        scope_services=["ZService", "aService"],
        scope_layer="architecture",
        rationale="Parity coverage",
        source="manual",
        created_at=now,
        created_by="parity@example.com",
    )
    conn1 = AsyncMock()
    conn1.fetchval.return_value = now.replace(tzinfo=None)
    conn1.fetchrow.side_effect = [
        {"status": "PROPOSED"} if scenario.outcome == "update" else None,
        {"was_insert": scenario.outcome != "update", "new_status": "ACTIVE"},
    ]
    transaction = AsyncMock()
    transaction.__aexit__.return_value = None
    conn1.transaction = MagicMock(return_value=transaction)
    conn2 = AsyncMock()
    conn2.fetch.return_value = []
    if scenario.outcome == "stage2_failure":
        conn2.fetch.side_effect = RuntimeError("deterministic stage 2 failure")
    elif scenario.outcome == "timeout":
        conn1.fetchval.side_effect = TimeoutError("deterministic stage 1 timeout")
    pool = MagicMock()
    pool.acquire.side_effect = [_acquire_context(conn1), _acquire_context(conn2)]
    handler = HandlerWriteDecision(pool)
    try:
        result = await handler.handle(request)
    except TypeError as exc:
        pytest.fail(f"handle() does not accept the single request model: {exc}")

    conn1.fetchval.assert_awaited_once_with(SQL_NOW)
    if scenario.outcome == "timeout":
        conn1.fetchrow.assert_not_awaited()
        conn2.fetch.assert_not_awaited()
        assert pool.acquire.call_count == 1
        assert transaction.__aexit__.await_args.args[0] is TimeoutError
    else:
        assert pool.acquire.call_count == 2
        assert conn1.fetchrow.await_args_list[0].args == (
            SQL_FETCH_OLD_STATUS,
            request.decision_id,
        )
        conn1.fetchrow.assert_awaited_with(
            SQL_UPSERT_DECISION,
            request.decision_id,
            request.correlation_id,
            request.title,
            request.decision_type,
            "ACTIVE",
            "infra",
            '["aservice", "zservice"]',
            request.scope_layer,
            request.rationale,
            "[]",
            "[]",
            request.source,
            None,
            "[]",
            None,
            now,
            request.created_by,
        )
        transaction.__aexit__.assert_awaited_once_with(None, None, None)
        conn2.fetch.assert_awaited_once_with(
            SQL_ACTIVE_SAME_SCOPE, "infra", "architecture", request.decision_id
        )
        conn2.fetchrow.assert_not_awaited()
    assert result.correlation_id == scenario.correlation_id
    assert result.backend_id == "postgres"
    return result


class TestWriteDecisionHandleTypedRequest:
    """Decision inserts, updates, and stage isolation retain their behavior."""

    @pytest.mark.asyncio
    async def test_insert_normalizes_scope_and_commits(self) -> None:
        result = await run_write_decision_scenario(WRITE_DECISION_SCENARIOS[0])
        assert result.success is True
        assert result.error_code is None

    @pytest.mark.asyncio
    async def test_update_existing_decision_commits(self) -> None:
        result = await run_write_decision_scenario(WRITE_DECISION_SCENARIOS[1])
        assert result.success is True
        assert result.error_code is None

    @pytest.mark.asyncio
    async def test_stage2_failure_preserves_stage1_success(self) -> None:
        result = await run_write_decision_scenario(WRITE_DECISION_SCENARIOS[2])
        assert result.success is True
        assert result.error is None
        assert result.error_code is None

    @pytest.mark.asyncio
    async def test_stage1_timeout_prevents_stage2(self) -> None:
        result = await run_write_decision_scenario(WRITE_DECISION_SCENARIOS[3])
        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.TIMEOUT_ERROR
