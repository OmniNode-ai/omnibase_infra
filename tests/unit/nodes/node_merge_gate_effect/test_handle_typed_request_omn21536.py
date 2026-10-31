# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed-request parity scenarios for merge gate persistence (OMN-21536)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from pydantic import BaseModel, ConfigDict

from omnibase_infra.enums import EnumPostgresErrorCode
from omnibase_infra.models.model_backend_result import ModelBackendResult
from omnibase_infra.nodes.node_merge_gate_effect.handlers.handler_upsert_merge_gate import (
    LINEAR_CREATE_ISSUE_MUTATION,
    SQL_UPSERT_MERGE_GATE,
    HandlerUpsertMergeGate,
)
from omnibase_infra.protocols.protocol_linear_graphql_transport import (
    ProtocolLinearGraphQLTransport,
)

pytestmark = pytest.mark.unit

HANDLER_MODULE: str = (
    "omnibase_infra.nodes.node_merge_gate_effect.handlers.handler_upsert_merge_gate"
)
PRESERVED_SYMBOLS: tuple[str, ...] = ("HandlerUpsertMergeGate._upsert_and_quarantine",)
GENERATED_CORRELATION_ID = UUID(int=999)


class ModelMergeGateScenario(BaseModel):
    """A deterministic decision, pool state and Linear response."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    seed: int
    decision: Literal["PASS", "WARN", "QUARANTINE"] = "PASS"
    was_insert: bool = True
    no_pool: bool = False
    linear_configured: bool = True
    linear_failure: bool = False
    missing_correlation_id: bool = False
    db_failure: Literal["none", "timeout"] = "none"

    @property
    def correlation_id(self) -> UUID:
        return UUID(int=self.seed)


SCENARIOS: tuple[ModelMergeGateScenario, ...] = (
    ModelMergeGateScenario(seed=1),
    ModelMergeGateScenario(seed=2, decision="WARN", was_insert=False),
    ModelMergeGateScenario(seed=3, decision="QUARANTINE"),
    ModelMergeGateScenario(seed=4, decision="QUARANTINE", was_insert=False),
    ModelMergeGateScenario(seed=5, no_pool=True),
    ModelMergeGateScenario(seed=6, db_failure="timeout"),
    ModelMergeGateScenario(seed=7, decision="QUARANTINE", linear_failure=True),
    ModelMergeGateScenario(seed=8, decision="QUARANTINE", linear_configured=False),
    ModelMergeGateScenario(seed=9, missing_correlation_id=True),
)


async def run_scenario(scenario: ModelMergeGateScenario) -> ModelBackendResult:
    """Exercise the real upsert and quarantine helpers using fake I/O."""
    try:
        from omnibase_infra.nodes.node_merge_gate_effect.models import (
            ModelMergeGateResult,
            ModelMergeGateViolation,
        )
    except ImportError as exc:
        pytest.fail(f"the typed merge gate request model is missing: {exc}")

    correlation_id = (
        None if scenario.missing_correlation_id else scenario.correlation_id
    )
    request = ModelMergeGateResult(
        gate_id=UUID(int=100 + scenario.seed),
        pr_ref="owner/repo#42",
        head_sha="a" * 40,
        base_sha="b" * 40,
        decision=scenario.decision,
        tier="tier-a",
        violations=[
            ModelMergeGateViolation(
                rule_code="RULE-1",
                severity="FAIL",
                message="Contract violated",
            )
        ]
        if scenario.decision == "QUARANTINE"
        else [],
        run_id=UUID(int=200 + scenario.seed),
        correlation_id=correlation_id,
        run_fingerprint="test-fingerprint",
        decided_at=datetime(2026, 1, 15, tzinfo=UTC),
    )
    conn = AsyncMock()
    conn.fetchrow.return_value = {"was_insert": scenario.was_insert}
    if scenario.db_failure == "timeout":
        conn.fetchrow.side_effect = TimeoutError("merge gate timed out")
    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    handler = HandlerUpsertMergeGate(
        None if scenario.no_pool else pool,
        linear_api_key="test-key" if scenario.linear_configured else None,
        linear_team_id="test-team" if scenario.linear_configured else None,
    )
    response = MagicMock()
    response.json.return_value = {
        "data": {
            "issueCreate": {
                "success": True,
                "issue": {
                    "identifier": "OMN-1",
                    "url": "https://example.com/OMN-1",
                },
            }
        }
    }
    transport = AsyncMock(spec=ProtocolLinearGraphQLTransport)
    transport.post_graphql.return_value = response
    if scenario.linear_failure:
        transport.post_graphql.side_effect = RuntimeError("Linear unavailable")
    ticket_scope = MagicMock()
    ticket_scope.__aenter__ = AsyncMock(return_value=transport)
    ticket_scope.__aexit__ = AsyncMock(return_value=None)
    with (
        patch(
            f"{HANDLER_MODULE}.uuid4",
            create=True,
            return_value=GENERATED_CORRELATION_ID,
        ),
        patch(
            f"{HANDLER_MODULE}.AdapterLinearGraphQLProjectTracker.graphql_transport",
            return_value=ticket_scope,
        ) as graphql_transport,
    ):
        try:
            result = await handler.handle(request)
        except TypeError as exc:
            pytest.fail(f"handle() does not accept the single request model: {exc}")

    effective_correlation_id = correlation_id or GENERATED_CORRELATION_ID
    if scenario.no_pool:
        pool.acquire.assert_not_called()
        conn.fetchrow.assert_not_awaited()
    else:
        pool.acquire.assert_called_once_with()
        conn.fetchrow.assert_awaited_once_with(
            SQL_UPSERT_MERGE_GATE,
            request.gate_id,
            "owner/repo#42",
            "a" * 40,
            "b" * 40,
            scenario.decision,
            "tier-a",
            json.dumps([v.model_dump(mode="json") for v in request.violations]),
            request.run_id,
            effective_correlation_id,
            "test-fingerprint",
            request.decided_at,
        )
        assert result.backend_id == "postgres"
    if (
        scenario.decision == "QUARANTINE"
        and scenario.was_insert
        and scenario.linear_configured
        and not scenario.no_pool
        and scenario.db_failure == "none"
    ):
        graphql_transport.assert_called_once_with(
            api_key="test-key", timeout_seconds=15.0
        )
        transport.post_graphql.assert_awaited_once()
        query, variables = transport.post_graphql.call_args.args
        assert query == LINEAR_CREATE_ISSUE_MUTATION
        assert variables["teamId"] == "test-team"
        assert variables["title"] == "QUARANTINE: owner/repo#42 (tier-a)"
        assert variables["priority"] == 1
        assert "RULE-1" in variables["description"]
        assert str(effective_correlation_id) in variables["description"]
    else:
        graphql_transport.assert_not_called()
        transport.post_graphql.assert_not_awaited()
    assert result.correlation_id == effective_correlation_id
    return result


class TestUpsertMergeGateHandleTypedRequest:
    """The payload alone drives persistence, quarantine and correlation IDs."""

    @pytest.mark.asyncio
    async def test_pass_insert_succeeds_without_linear_call(self) -> None:
        result = await run_scenario(SCENARIOS[0])
        assert result.success is True
        assert result.error_code is None

    @pytest.mark.asyncio
    async def test_warn_update_succeeds_without_linear_call(self) -> None:
        result = await run_scenario(SCENARIOS[1])
        assert result.success is True
        assert result.error_code is None

    @pytest.mark.asyncio
    async def test_quarantine_insert_creates_ticket(self) -> None:
        result = await run_scenario(SCENARIOS[2])
        assert result.success is True
        assert result.error_code is None

    @pytest.mark.asyncio
    async def test_quarantine_update_skips_duplicate_ticket(self) -> None:
        result = await run_scenario(SCENARIOS[3])
        assert result.success is True
        assert result.error_code is None

    @pytest.mark.asyncio
    async def test_missing_pool_returns_no_pool_error(self) -> None:
        result = await run_scenario(SCENARIOS[4])
        assert result.success is False
        assert result.error_code == "MERGE_GATE_NO_POOL"
        assert result.error == "db pool not configured"

    @pytest.mark.asyncio
    async def test_db_timeout_returns_timeout_error(self) -> None:
        result = await run_scenario(SCENARIOS[5])
        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.TIMEOUT_ERROR

    @pytest.mark.asyncio
    async def test_linear_failure_preserves_upsert_success(self) -> None:
        result = await run_scenario(SCENARIOS[6])
        assert result.success is True
        assert result.error_code is None

    @pytest.mark.asyncio
    async def test_unconfigured_linear_preserves_quarantine_success(self) -> None:
        result = await run_scenario(SCENARIOS[7])
        assert result.success is True
        assert result.error_code is None

    @pytest.mark.asyncio
    async def test_missing_correlation_id_is_generated(self) -> None:
        result = await run_scenario(SCENARIOS[8])
        assert result.success is True
        assert result.error_code is None
        assert result.correlation_id == GENERATED_CORRELATION_ID
