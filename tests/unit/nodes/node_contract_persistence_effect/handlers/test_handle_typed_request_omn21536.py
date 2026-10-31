# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Parity tests for the typed-request handle() of node_contract_persistence_effect.

The six PostgreSQL handlers took ``handle(payload, correlation_id)``; each now
takes the one payload model and reads the correlation id from it. These tests
drive ``HandlerPostgresContractUpsert`` (the node's representative handler)
through the new call shape. Against the merge-base handler the call raises
``TypeError``, which each test turns into its own assertion failure.

The scenarios double as the candidate inputs the adequacy receipt under
``scripts/ci/adequacy_receipts/`` was recorded over.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from pydantic import BaseModel, ConfigDict

from omnibase_infra.enums import EnumPostgresErrorCode
from omnibase_infra.models.model_backend_result import ModelBackendResult
from omnibase_infra.nodes.node_contract_persistence_effect.handlers.handler_postgres_contract_upsert import (
    HandlerPostgresContractUpsert,
)
from omnibase_infra.nodes.node_contract_registry_reducer.models import (
    ModelPayloadUpsertContract,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)


class ModelUpsertScenario(BaseModel):
    """One upsert request plus what the mocked pool answers."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    payload: ModelPayloadUpsertContract
    pool_answer: Literal["insert", "update", "no_row", "timeout"]


def _payload(
    seed: int, contract_yaml: dict[str, str] | str
) -> ModelPayloadUpsertContract:
    return ModelPayloadUpsertContract(
        correlation_id=UUID(int=seed),
        contract_id=f"node-{seed}:1.0.0",
        node_name=f"node-{seed}",
        version_major=1,
        version_minor=0,
        version_patch=0,
        contract_hash=f"hash-{seed}",
        contract_yaml=contract_yaml,
        is_active=True,
        registered_at=_NOW,
        last_seen_at=_NOW,
    )


SCENARIOS: tuple[ModelUpsertScenario, ...] = (
    ModelUpsertScenario(payload=_payload(1, {"name": "a"}), pool_answer="insert"),
    ModelUpsertScenario(payload=_payload(2, {"name": "b"}), pool_answer="update"),
    ModelUpsertScenario(payload=_payload(3, "name: c\n"), pool_answer="insert"),
    ModelUpsertScenario(payload=_payload(4, {"name": "d"}), pool_answer="no_row"),
    ModelUpsertScenario(payload=_payload(5, {"name": "e"}), pool_answer="timeout"),
)


def _handler_for(scenario: ModelUpsertScenario) -> HandlerPostgresContractUpsert:
    conn = AsyncMock()
    if scenario.pool_answer == "timeout":
        conn.fetchrow = AsyncMock(side_effect=TimeoutError("timed out"))
    elif scenario.pool_answer == "no_row":
        conn.fetchrow = AsyncMock(return_value=None)
    else:
        conn.fetchrow = AsyncMock(
            return_value={"was_insert": scenario.pool_answer == "insert"}
        )
    acquire = AsyncMock()
    acquire.__aenter__ = AsyncMock(return_value=conn)
    acquire.__aexit__ = AsyncMock(return_value=None)
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=acquire)
    return HandlerPostgresContractUpsert(pool)


async def run_scenario(scenario: ModelUpsertScenario) -> ModelBackendResult:
    """Call the handler the way the runtime does: one request argument."""
    handler = _handler_for(scenario)
    try:
        return await handler.handle(scenario.payload)
    except TypeError as exc:
        pytest.fail(f"handle() does not accept the single payload request: {exc}")


class TestUpsertHandleTypedRequest:
    """HandlerPostgresContractUpsert.handle takes one payload model."""

    @pytest.mark.asyncio
    async def test_insert_answers_with_the_payload_correlation_id(self) -> None:
        result = await run_scenario(SCENARIOS[0])
        assert result.success is True
        assert result.correlation_id == SCENARIOS[0].payload.correlation_id

    @pytest.mark.asyncio
    async def test_missing_row_fails_with_the_payload_correlation_id(self) -> None:
        result = await run_scenario(SCENARIOS[3])
        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.UPSERT_ERROR
        assert result.correlation_id == SCENARIOS[3].payload.correlation_id

    @pytest.mark.asyncio
    async def test_timeout_fails_with_the_payload_correlation_id(self) -> None:
        result = await run_scenario(SCENARIOS[4])
        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.TIMEOUT_ERROR
        assert result.correlation_id == SCENARIOS[4].payload.correlation_id
