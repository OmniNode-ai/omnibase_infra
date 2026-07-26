# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the verification-receipt raw ledger projector (WS-L slice 2).

Proves that the shipped verification_receipt_projector.yaml contract, driven by
the generic ProjectorShell, materializes an
onex.evt.omniclaude.verification_receipt.v1 event envelope into the append-only
verification_receipt_ledger projection with the correct column/value shape — with
NO bespoke node or class (CLAUDE.md rule 7a).

This is the receiving end of the verification-receipts flat-file migration
(.onex_state/verification-receipts/*.yaml) — the doctrine §1 fix. The payload
shapes asserted here mirror the verified on-disk reality: the standard ticket
receipt shape and the legacy baseline snapshot shape (which omits ticket_id and
phases), both captured losslessly via receipt_body.

Related:
    - OMN-13997 (this slice) / OMN-13989 (epic)
    - docs/plans/2026-07-05-ledger-learning-substrate-plan.md (§2.5, §6 Phase 1)
    - src/omnibase_infra/projectors/contracts/verification_receipt_projector.yaml
    - docker/migrations/forward/089_create_verification_receipt_ledger.sql
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
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

VERIFICATION_RECEIPT_EVENT_TYPE = "onex.evt.omniclaude.verification_receipt.v1"

CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "omnibase_infra"
    / "projectors"
    / "contracts"
    / "verification_receipt_projector.yaml"
)

# Column order as declared in the contract (drives mock-pool param positions).
EXPECTED_COLUMNS = [
    "envelope_id",
    "correlation_id",
    "ticket_id",
    "sweep_timestamp",
    "overall_status",
    "overseer_verdict",
    "idempotent",
    "phases",
    "receipt_body",
]


# =============================================================================
# Test payload model (mirrors the on-disk receipt shape + full-fidelity body)
# =============================================================================


class _VerificationReceiptPayload(BaseModel):
    """Representative verification-receipt payload.

    Core fields are lifted for querying; receipt_body carries the full raw
    receipt dict so heterogeneous shapes are captured losslessly.
    """

    ticket_id: str | None = None
    sweep_timestamp: str
    overall_status: str
    overseer_verdict: str | None = None
    idempotent: bool | None = None
    phases: dict[str, Any] | None = None
    receipt_body: dict[str, Any]


def _standard_receipt_payload() -> _VerificationReceiptPayload:
    """A standard ticket receipt (mirrors .onex_state/.../OMN-8606.yaml)."""
    phases = {
        "dashboard": {"status": "skip", "endpoints_checked": 0, "results": []},
        "database": {"status": "skip", "tables_checked": 0, "results": []},
        "dod_evidence": {
            "status": "pass",
            "items_checked": 2,
            "results": [
                {
                    "type": "integration_test",
                    "pr": "OmniNode-ai/omnidash#598",
                    "status": "pass",
                    "evidence": "All 20 CI checks green",
                }
            ],
        },
    }
    body: dict[str, Any] = {
        "ticket_id": "OMN-8606",
        "sweep_timestamp": "2026-04-14T22:50:00Z",
        "overall_status": "pass",
        "phases": phases,
        "overseer_verdict": "PASS",
        "idempotent": True,
    }
    return _VerificationReceiptPayload(
        ticket_id="OMN-8606",
        sweep_timestamp="2026-04-14T22:50:00Z",
        overall_status="pass",
        overseer_verdict="PASS",
        idempotent=True,
        phases=phases,
        receipt_body=body,
    )


def _baseline_snapshot_payload() -> _VerificationReceiptPayload:
    """Legacy baseline snapshot — omits ticket_id + phases (heterogeneous shape)."""
    body: dict[str, Any] = {
        "type": "baseline",
        "sweep_timestamp": "2026-04-15T05:00:00Z",
        "overall_status": "pass",
        "recent_completions": 12,
        "dispatch_health": "ok",
        "overseer_checks_passed": 17,
    }
    return _VerificationReceiptPayload(
        ticket_id=None,
        sweep_timestamp="2026-04-15T05:00:00Z",
        overall_status="pass",
        overseer_verdict=None,
        idempotent=None,
        phases=None,
        receipt_body=body,
    )


def _make_envelope(
    payload: _VerificationReceiptPayload,
    *,
    envelope_id: UUID | None = None,
    correlation_id: UUID | None = None,
    event_type: str = VERIFICATION_RECEIPT_EVENT_TYPE,
) -> ModelEventEnvelope[_VerificationReceiptPayload]:
    """Wrap a receipt payload in a ModelEventEnvelope (the emit-side shape)."""
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
def verification_receipt_contract() -> ModelProjectorContract:
    """Load the SHIPPED contract YAML into a validated ModelProjectorContract."""
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
    verification_receipt_contract: ModelProjectorContract,
) -> None:
    """The committed verification_receipt_projector.yaml is a valid contract."""
    c = verification_receipt_contract
    assert c.projector_id == "verification-receipt-ledger-projector"
    assert c.consumed_events == [VERIFICATION_RECEIPT_EVENT_TYPE]
    assert c.projection_schema.table == "verification_receipt_ledger"
    assert c.projection_schema.primary_key == "envelope_id"
    assert c.behavior.mode == "append"
    assert [col.name for col in c.projection_schema.columns] == EXPECTED_COLUMNS


@pytest.mark.asyncio
async def test_standard_receipt_projects_with_correct_shape(
    verification_receipt_contract: ModelProjectorContract,
    mock_pool: MagicMock,
) -> None:
    """A real-shaped ticket receipt lands in verification_receipt_ledger.

    Proves append mode -> plain INSERT with every projected column, envelope
    identity from the envelope, receipt fields from payload.*, and the full raw
    body captured in receipt_body.
    """
    payload = _standard_receipt_payload()
    envelope_id = uuid4()
    correlation_id = uuid4()
    envelope = _make_envelope(
        payload, envelope_id=envelope_id, correlation_id=correlation_id
    )

    projector = ProjectorShell(contract=verification_receipt_contract, pool=mock_pool)
    result = await projector.project(envelope, correlation_id)

    assert result.success is True
    assert result.skipped is False
    assert result.rows_affected == 1

    call_args = mock_pool._mock_conn.execute.call_args
    assert call_args is not None
    sql = call_args[0][0]
    params = list(call_args[0][1:])

    # Append mode -> plain INSERT, never ON CONFLICT DO UPDATE.
    assert 'INSERT INTO "verification_receipt_ledger"' in sql
    assert "ON CONFLICT" not in sql
    assert "DO UPDATE" not in sql
    for col in EXPECTED_COLUMNS:
        assert f'"{col}"' in sql, f"column {col} missing from INSERT"

    by_col = dict(zip(EXPECTED_COLUMNS, params, strict=True))
    assert by_col["envelope_id"] == envelope_id
    assert by_col["correlation_id"] == correlation_id
    assert by_col["ticket_id"] == "OMN-8606"
    assert by_col["sweep_timestamp"] == datetime.fromisoformat(
        payload.sweep_timestamp.replace("Z", "+00:00")
    )
    assert by_col["overall_status"] == "pass"
    assert by_col["overseer_verdict"] == "PASS"
    assert by_col["idempotent"] is True
    assert by_col["phases"]["dod_evidence"]["status"] == "pass"
    # Full-fidelity body carries the complete raw payload.
    assert by_col["receipt_body"]["ticket_id"] == "OMN-8606"
    assert by_col["receipt_body"]["phases"]["dashboard"]["status"] == "skip"


@pytest.mark.asyncio
async def test_baseline_snapshot_nullable_fields_survive(
    verification_receipt_contract: ModelProjectorContract,
    mock_pool: MagicMock,
) -> None:
    """The heterogeneous baseline snapshot (no ticket_id/phases) still projects.

    Guards lossless capture: NULL core fields survive extraction (not dropped),
    and the distinct baseline fields are retained in receipt_body.
    """
    payload = _baseline_snapshot_payload()
    envelope = _make_envelope(payload)

    projector = ProjectorShell(contract=verification_receipt_contract, pool=mock_pool)
    result = await projector.project(envelope, uuid4())

    assert result.success is True
    assert result.rows_affected == 1
    params = list(mock_pool._mock_conn.execute.call_args[0][1:])
    by_col = dict(zip(EXPECTED_COLUMNS, params, strict=True))
    assert by_col["ticket_id"] is None
    assert by_col["overseer_verdict"] is None
    assert by_col["idempotent"] is None
    assert by_col["phases"] is None
    # No shape is dropped — the baseline-only fields live in receipt_body.
    assert by_col["receipt_body"]["type"] == "baseline"
    assert by_col["receipt_body"]["dispatch_health"] == "ok"


@pytest.mark.asyncio
async def test_non_consumed_event_is_skipped(
    verification_receipt_contract: ModelProjectorContract,
    mock_pool: MagicMock,
) -> None:
    """An event whose type is not in consumed_events performs no write."""
    payload = _standard_receipt_payload()
    envelope = _make_envelope(payload, event_type="onex.evt.omniclaude.other.v1")

    projector = ProjectorShell(contract=verification_receipt_contract, pool=mock_pool)
    result = await projector.project(envelope, uuid4())

    assert result.skipped is True
    mock_pool._mock_conn.execute.assert_not_called()
