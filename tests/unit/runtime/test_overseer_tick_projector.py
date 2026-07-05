# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the overseer-tick raw ledger projector (WS-L slice 1).

Proves that the shipped overseer_tick_projector.yaml contract, driven by the
generic ProjectorShell, materializes an onex.evt.omnimarket.overseer-tick.v1
event envelope into the append-only overseer_tick_ledger projection with the
correct column/value shape — with NO bespoke node or class (CLAUDE.md rule 7a).

This is the receiving end of the overseer-tick flat-file migration
(.onex_state/overseer-ticks.jsonl). The payload shape asserted here is the
verified on-disk snapshot shape produced by
omnimarket/.../node_overnight/handlers/overseer_tick.py::build_tick_snapshot
(11 keys, stable across all 1271 historical ticks).

Related:
    - OMN-13996 (this slice) / OMN-13989 (epic)
    - docs/plans/2026-07-05-ledger-learning-substrate-plan.md (§2.5, §6 Phase 2)
    - src/omnibase_infra/projectors/contracts/overseer_tick_projector.yaml
    - docker/migrations/forward/088_create_overseer_tick_ledger.sql
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import asyncpg
import pytest
import yaml
from pydantic import BaseModel

from omnibase_core.models.core.model_envelope_metadata import ModelEnvelopeMetadata
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_core.models.projectors import ModelProjectorContract
from omnibase_infra.runtime.projector_shell import ProjectorShell

# =============================================================================
# Constants
# =============================================================================

# The event-TYPE identifier (consumed_events / envelope metadata tag) uses an
# underscore to satisfy ModelProjectorContract's event-name pattern.
OVERSEER_TICK_EVENT_TYPE = "onex.evt.omnimarket.overseer_tick.v1"
# The Kafka TOPIC string (carried as payload.topic) is the hyphenated form,
# contract-declared in node_overnight/handlers/overseer_tick.py.
OVERSEER_TICK_TOPIC = "onex.evt.omnimarket.overseer-tick.v1"

# Contract lives alongside the reference registration_projector.yaml.
CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "omnibase_infra"
    / "projectors"
    / "contracts"
    / "overseer_tick_projector.yaml"
)

# Column order as declared in the contract — the projector inserts columns in
# this order, so the mock-pool param positions follow it.
EXPECTED_COLUMNS = [
    "envelope_id",
    "correlation_id",
    "topic",
    "session_id",
    "current_phase",
    "phase_progress",
    "phase_outcomes",
    "next_required_outcome",
    "approaching_halt_conditions",
    "accumulated_cost",
    "started_at",
    "emitted_at",
]


# =============================================================================
# Test payload model (mirrors build_tick_snapshot's on-disk shape)
# =============================================================================


class _OverseerTickPayload(BaseModel):
    """Representative overseer-tick snapshot payload (11 verified keys)."""

    topic: str
    contract_path: str
    session_id: str
    current_phase: str
    phase_progress: float
    phase_outcomes: dict[str, bool]
    next_required_outcome: str | None
    approaching_halt_conditions: list[dict[str, object]]
    accumulated_cost: float
    started_at: str
    emitted_at: str


def _make_tick_payload() -> _OverseerTickPayload:
    """Build a realistic tick payload matching the verified on-disk schema."""
    return _OverseerTickPayload(
        topic=OVERSEER_TICK_TOPIC,
        contract_path="contracts/overnight/test-contract.yaml",
        session_id="test-cli-overnight",
        current_phase="nightly_loop_controller",
        phase_progress=1.0,
        phase_outcomes={"pr_merged": False},
        next_required_outcome="pr_merged",
        approaching_halt_conditions=[
            {
                "condition_id": "cost_ceiling",
                "check_type": "cost_ceiling",
                "on_halt": "hard_halt",
                "proximity_pct": 12.5,
            }
        ],
        accumulated_cost=1.25,
        started_at="2026-04-22T00:25:20.102483+00:00",
        emitted_at="2026-04-22T00:25:20.102503+00:00",
    )


def _make_envelope(
    payload: _OverseerTickPayload,
    *,
    envelope_id: UUID | None = None,
    correlation_id: UUID | None = None,
    event_type: str = OVERSEER_TICK_EVENT_TYPE,
) -> ModelEventEnvelope[_OverseerTickPayload]:
    """Wrap a tick payload in a ModelEventEnvelope (the emit-side shape)."""
    return ModelEventEnvelope(
        payload=payload,
        envelope_id=envelope_id or uuid4(),
        envelope_timestamp=datetime.now(UTC),
        correlation_id=correlation_id or uuid4(),
        metadata=ModelEnvelopeMetadata(tags={"event_type": event_type}),
        onex_version=ModelSemVer(major=1, minor=0, patch=0),
        envelope_version=ModelSemVer(major=1, minor=0, patch=0),
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def overseer_tick_contract() -> ModelProjectorContract:
    """Load the SHIPPED contract YAML into a validated ModelProjectorContract.

    This proves the committed artifact parses and validates against the model,
    not just an in-memory hand-built contract.
    """
    raw = yaml.safe_load(CONTRACT_PATH.read_text())
    return ModelProjectorContract.model_validate(raw)


@pytest.fixture
def mock_pool() -> MagicMock:
    """Mocked asyncpg.Pool that captures the executed SQL and params."""
    pool = MagicMock(spec=asyncpg.Pool)
    conn = AsyncMock(spec_set=asyncpg.Connection)

    class _AcquireCtx:
        async def __aenter__(self) -> AsyncMock:
            return conn

        async def __aexit__(self, *_: object) -> None:
            return None

    pool.acquire.return_value = _AcquireCtx()
    conn.execute.return_value = "INSERT 0 1"
    conn.fetchrow.return_value = None
    conn.fetch.return_value = []
    pool._mock_conn = conn
    return pool


# =============================================================================
# Tests
# =============================================================================


def test_shipped_contract_loads_and_validates(
    overseer_tick_contract: ModelProjectorContract,
) -> None:
    """The committed overseer_tick_projector.yaml is a valid contract."""
    assert overseer_tick_contract.projector_id == "overseer-tick-ledger-projector"
    assert overseer_tick_contract.consumed_events == [OVERSEER_TICK_EVENT_TYPE]
    assert overseer_tick_contract.projection_schema.table == "overseer_tick_ledger"
    assert overseer_tick_contract.projection_schema.primary_key == "envelope_id"
    assert overseer_tick_contract.behavior.mode == "append"
    # Column set matches the migration + expected projection shape.
    contract_columns = [
        c.name for c in overseer_tick_contract.projection_schema.columns
    ]
    assert contract_columns == EXPECTED_COLUMNS


@pytest.mark.asyncio
async def test_tick_projects_into_ledger_with_correct_shape(
    overseer_tick_contract: ModelProjectorContract,
    mock_pool: MagicMock,
) -> None:
    """A real-shaped overseer-tick envelope lands in overseer_tick_ledger.

    Proves: append mode -> plain INSERT into the ledger table with every
    projected column, envelope-level identity resolved from the envelope and
    payload fields resolved from payload.*.
    """
    payload = _make_tick_payload()
    envelope_id = uuid4()
    correlation_id = uuid4()
    envelope = _make_envelope(
        payload, envelope_id=envelope_id, correlation_id=correlation_id
    )

    projector = ProjectorShell(contract=overseer_tick_contract, pool=mock_pool)
    result = await projector.project(envelope, correlation_id)

    assert result.success is True
    assert result.skipped is False
    assert result.rows_affected == 1

    # Inspect the executed statement + params.
    call_args = mock_pool._mock_conn.execute.call_args
    assert call_args is not None
    sql = call_args[0][0]
    params = list(call_args[0][1:])

    # Append mode must be a plain INSERT into the ledger table — never an
    # ON CONFLICT DO UPDATE (that would mutate append-only history).
    assert 'INSERT INTO "overseer_tick_ledger"' in sql
    assert "ON CONFLICT" not in sql
    assert "DO UPDATE" not in sql

    # Every projected column is present, in contract order.
    for col in EXPECTED_COLUMNS:
        assert f'"{col}"' in sql, f"column {col} missing from INSERT"

    # Param positions follow contract column order. Assert the load-bearing
    # envelope->column and payload->column mappings.
    by_col = dict(zip(EXPECTED_COLUMNS, params, strict=True))
    assert by_col["envelope_id"] == envelope_id
    assert by_col["correlation_id"] == correlation_id
    assert by_col["topic"] == OVERSEER_TICK_TOPIC
    assert by_col["session_id"] == "test-cli-overnight"
    assert by_col["current_phase"] == "nightly_loop_controller"
    assert by_col["phase_progress"] == 1.0
    assert by_col["phase_outcomes"] == {"pr_merged": False}
    assert by_col["next_required_outcome"] == "pr_merged"
    assert by_col["approaching_halt_conditions"][0]["condition_id"] == "cost_ceiling"
    assert by_col["accumulated_cost"] == 1.25


@pytest.mark.asyncio
async def test_null_next_required_outcome_projects(
    overseer_tick_contract: ModelProjectorContract,
    mock_pool: MagicMock,
) -> None:
    """A tick with all outcomes satisfied (next_required_outcome=None) projects.

    Guards the nullable column: None must survive extraction, not be dropped.
    """
    payload = _make_tick_payload()
    payload = payload.model_copy(
        update={"next_required_outcome": None, "phase_outcomes": {"pr_merged": True}}
    )
    envelope = _make_envelope(payload)

    projector = ProjectorShell(contract=overseer_tick_contract, pool=mock_pool)
    result = await projector.project(envelope, uuid4())

    assert result.success is True
    assert result.rows_affected == 1
    params = list(mock_pool._mock_conn.execute.call_args[0][1:])
    by_col = dict(zip(EXPECTED_COLUMNS, params, strict=True))
    assert by_col["next_required_outcome"] is None


@pytest.mark.asyncio
async def test_non_consumed_event_is_skipped(
    overseer_tick_contract: ModelProjectorContract,
    mock_pool: MagicMock,
) -> None:
    """An event whose type is not in consumed_events performs no write."""
    payload = _make_tick_payload()
    envelope = _make_envelope(payload, event_type="onex.evt.omnimarket.other.v1")

    projector = ProjectorShell(contract=overseer_tick_contract, pool=mock_pool)
    result = await projector.project(envelope, uuid4())

    assert result.skipped is True
    mock_pool._mock_conn.execute.assert_not_called()
