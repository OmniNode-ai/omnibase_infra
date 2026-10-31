# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed-request parity for bundle inserts and outcome updates (OMN-21536)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Literal
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from pydantic import BaseModel, ConfigDict

from omnibase_infra.enums import EnumPostgresErrorCode
from omnibase_infra.errors import InfraConnectionError
from omnibase_infra.models.model_backend_result import ModelBackendResult
from omnibase_infra.nodes.node_delta_bundle_effect.handlers.handler_update_outcome import (
    SQL_UPDATE_OUTCOME,
    HandlerUpdateOutcome,
)
from omnibase_infra.nodes.node_delta_bundle_effect.handlers.handler_write_bundle import (
    SQL_INSERT_BUNDLE,
    HandlerWriteBundle,
)

pytestmark = pytest.mark.unit

HANDLER_MODULE: str = (
    "omnibase_infra.nodes.node_delta_bundle_effect.handlers.handler_write_bundle"
)
PRESERVED_SYMBOLS: tuple[str, ...] = (
    "HandlerWriteBundle._execute_insert",
    "parse_stabilizes_label",
)


class ModelWriteScenario(BaseModel):
    """One insert with labels and a deterministic database response."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    seed: int
    labels: tuple[str, ...] = ()
    decision: Literal["PASS", "WARN", "QUARANTINE"] = "PASS"
    has_violation: bool = False
    db_result: Literal["INSERT 0 1", "INSERT 0 0"] = "INSERT 0 1"
    failure: Literal["none", "timeout", "connection"] = "none"
    expected_stabilizes_ref: str | None = None

    @property
    def correlation_id(self) -> UUID:
        return UUID(int=self.seed)


SCENARIOS: tuple[ModelWriteScenario, ...] = (
    ModelWriteScenario(seed=1, decision="QUARANTINE", has_violation=True),
    ModelWriteScenario(
        seed=2,
        labels=("bug", "stabilizes: owner/repo#42 "),
        expected_stabilizes_ref="owner/repo#42",
    ),
    ModelWriteScenario(seed=3, labels=("stabilizes:   ",), db_result="INSERT 0 0"),
    ModelWriteScenario(
        seed=4,
        labels=("stabilizes:", "stabilizes:owner/repo#7", "stabilizes:owner/repo#8"),
        decision="WARN",
        expected_stabilizes_ref="owner/repo#7",
    ),
    ModelWriteScenario(seed=5, failure="timeout"),
    ModelWriteScenario(seed=6, failure="connection"),
)


def _pool_for(conn: AsyncMock) -> MagicMock:
    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    return pool


async def run_scenario(scenario: ModelWriteScenario) -> ModelBackendResult:
    """Run the real insert and label parser with a single payload argument."""
    try:
        from omnibase_infra.nodes.node_delta_bundle_effect.models import (
            ModelPayloadWriteBundle,
        )
    except ImportError as exc:
        pytest.fail(f"the typed bundle request model is missing: {exc}")

    violations = (
        [{"rule_code": "RULE-1", "severity": "FAIL"}] if scenario.has_violation else []
    )
    request = ModelPayloadWriteBundle(
        correlation_id=scenario.correlation_id,
        bundle_id=UUID(int=100 + scenario.seed),
        pr_ref="owner/repo#99",
        head_sha="a" * 40,
        base_sha="b" * 40,
        gate_decision=scenario.decision,
        gate_violations=violations,
        labels=list(scenario.labels),
    )
    conn = AsyncMock()
    conn.execute.return_value = scenario.db_result
    if scenario.failure == "timeout":
        conn.execute.side_effect = TimeoutError("bundle timed out")
    elif scenario.failure == "connection":
        conn.execute.side_effect = InfraConnectionError("bundle unavailable")
    pool = _pool_for(conn)
    handler = HandlerWriteBundle(pool)
    try:
        result = await handler.handle(request)
    except TypeError as exc:
        pytest.fail(f"handle() does not accept the single request model: {exc}")

    pool.acquire.assert_called_once_with()
    conn.execute.assert_awaited_once_with(
        SQL_INSERT_BUNDLE,
        request.bundle_id,
        "owner/repo#99",
        "a" * 40,
        "b" * 40,
        None,
        None,
        scenario.decision,
        json.dumps(violations),
        scenario.expected_stabilizes_ref is not None,
        scenario.expected_stabilizes_ref,
    )
    assert result.correlation_id == scenario.correlation_id
    assert result.backend_id == "postgres"
    return result


class TestWriteBundleHandleTypedRequest:
    """Bundle SQL and label semantics survive the typed entry point."""

    @pytest.mark.asyncio
    async def test_quarantine_inserts_serialized_violations(self) -> None:
        result = await run_scenario(SCENARIOS[0])
        assert result.success is True
        assert result.error_code is None

    @pytest.mark.asyncio
    async def test_padded_stabilization_label_sets_fix_fields(self) -> None:
        result = await run_scenario(SCENARIOS[1])
        assert result.success is True
        assert result.error is None

    @pytest.mark.asyncio
    async def test_duplicate_insert_and_empty_label_succeed(self) -> None:
        result = await run_scenario(SCENARIOS[2])
        assert result.success is True
        assert result.error_code is None

    @pytest.mark.asyncio
    async def test_first_nonempty_stabilization_label_wins(self) -> None:
        result = await run_scenario(SCENARIOS[3])
        assert result.success is True
        assert result.error_code is None

    @pytest.mark.asyncio
    async def test_insert_timeout_returns_timeout_error(self) -> None:
        result = await run_scenario(SCENARIOS[4])
        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.TIMEOUT_ERROR

    @pytest.mark.asyncio
    async def test_insert_connection_failure_returns_connection_error(self) -> None:
        result = await run_scenario(SCENARIOS[5])
        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.CONNECTION_ERROR


class ModelUpdateScenario(BaseModel):
    """One outcome update, including already-completed bundles and failures."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    seed: int
    outcome: Literal["merged", "reverted", "closed"] = "merged"
    db_result: Literal["UPDATE 1", "UPDATE 0"] = "UPDATE 1"
    failure: Literal["none", "timeout", "connection"] = "none"

    @property
    def correlation_id(self) -> UUID:
        return UUID(int=self.seed)


UPDATE_SCENARIOS: tuple[ModelUpdateScenario, ...] = (
    ModelUpdateScenario(seed=11),
    ModelUpdateScenario(seed=12, outcome="reverted"),
    ModelUpdateScenario(seed=13, outcome="closed", db_result="UPDATE 0"),
    ModelUpdateScenario(seed=14, failure="timeout"),
    ModelUpdateScenario(seed=15, failure="connection"),
)


async def run_update_scenario(scenario: ModelUpdateScenario) -> ModelBackendResult:
    """Run an outcome update through the single typed request entry point."""
    try:
        from omnibase_infra.nodes.node_delta_bundle_effect.models import (
            ModelPayloadUpdateOutcome,
        )
    except ImportError as exc:
        pytest.fail(f"the typed outcome request model is missing: {exc}")

    merged_at = (
        datetime(2026, 1, 15, 12, tzinfo=UTC) if scenario.outcome == "merged" else None
    )
    request = ModelPayloadUpdateOutcome(
        correlation_id=scenario.correlation_id,
        pr_ref="owner/repo#99",
        head_sha="a" * 40,
        outcome=scenario.outcome,
        merged_at=merged_at,
    )
    conn = AsyncMock()
    conn.execute.return_value = scenario.db_result
    if scenario.failure == "timeout":
        conn.execute.side_effect = TimeoutError("outcome timed out")
    elif scenario.failure == "connection":
        conn.execute.side_effect = InfraConnectionError("outcome unavailable")
    pool = _pool_for(conn)
    handler = HandlerUpdateOutcome(pool)
    try:
        result = await handler.handle(request)
    except TypeError as exc:
        pytest.fail(f"handle() does not accept the single request model: {exc}")

    pool.acquire.assert_called_once_with()
    conn.execute.assert_awaited_once_with(
        SQL_UPDATE_OUTCOME,
        scenario.outcome,
        merged_at,
        "owner/repo#99",
        "a" * 40,
    )
    assert result.correlation_id == scenario.correlation_id
    assert result.backend_id == "postgres"
    return result


class TestUpdateOutcomeHandleTypedRequest:
    """Outcome values, timestamps and failures survive the typed entry point."""

    @pytest.mark.asyncio
    async def test_merged_outcome_passes_merge_timestamp(self) -> None:
        result = await run_update_scenario(UPDATE_SCENARIOS[0])
        assert result.success is True
        assert result.error_code is None

    @pytest.mark.asyncio
    async def test_reverted_outcome_has_no_merge_timestamp(self) -> None:
        result = await run_update_scenario(UPDATE_SCENARIOS[1])
        assert result.success is True
        assert result.error_code is None

    @pytest.mark.asyncio
    async def test_already_completed_closed_bundle_succeeds(self) -> None:
        result = await run_update_scenario(UPDATE_SCENARIOS[2])
        assert result.success is True
        assert result.error_code is None

    @pytest.mark.asyncio
    async def test_update_timeout_returns_timeout_error(self) -> None:
        result = await run_update_scenario(UPDATE_SCENARIOS[3])
        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.TIMEOUT_ERROR

    @pytest.mark.asyncio
    async def test_update_connection_failure_returns_connection_error(self) -> None:
        result = await run_update_scenario(UPDATE_SCENARIOS[4])
        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.CONNECTION_ERROR
