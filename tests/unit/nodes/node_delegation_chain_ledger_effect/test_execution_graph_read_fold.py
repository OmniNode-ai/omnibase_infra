# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Graph read folding selects explicit offset bounds without stability claims."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from omnibase_core.models.execution_graph_replay import (
    EnumExecutionGraphCursorMode,
    ModelExecutionGraphRequest,
    ModelExecutionGraphSourceCursor,
    ModelExecutionGraphTopologyVersion,
)
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.execution_graph_read_fold import (
    ExecutionGraphReadFold,
)
from omnibase_infra.runtime.db.execution_graph_read_adapters import (
    DelegationOwnerProof,
    ExecutionGraphLedgerRecord,
)
from omnibase_infra.runtime.execution_graph_ownership import (
    ExecutionGraphOwnershipAdmission,
)
from omnibase_infra.runtime.execution_graph_read_authority import (
    TrustedGatewaySignerScope,
    VerifiedExecutionGraphReadAuthority,
)
from omnibase_infra.runtime.execution_graph_topology_registry import (
    PackagedExecutionGraphTopologyContract,
    PinnedExecutionGraphTopology,
)

HEAD_TOPIC = "onex.cmd.omnimarket.delegate-skill.v1"
CHILD_TOPIC = "onex.cmd.omnibase-infra.delegation-request.v1"
READ_AT = datetime(2026, 9, 26, tzinfo=UTC)


def _topology() -> PinnedExecutionGraphTopology:
    return PackagedExecutionGraphTopologyContract().resolve(
        ModelExecutionGraphTopologyVersion(
            contract_version=ModelSemVer(major=1, minor=3, patch=0),
            topology_sha256="0505ab0b163492380739a15646c0442a3ecfb54efb0acb53bc23d232640fbfd3",
        )
    )


def _authority(
    request: ModelExecutionGraphRequest, tenant_id: UUID
) -> VerifiedExecutionGraphReadAuthority:
    # Pure fold fixtures bypass minting; signed-ingress tests prove the real mint.
    authority = object.__new__(VerifiedExecutionGraphReadAuthority)
    object.__setattr__(authority, "tenant_id", tenant_id)
    object.__setattr__(authority, "correlation_id", request.correlation_id)
    object.__setattr__(authority, "request", request)
    object.__setattr__(
        authority,
        "signer_scope",
        TrustedGatewaySignerScope("gateway", "test", "bus"),
    )
    object.__setattr__(authority, "payload_hash", "fixture")
    return authority


def _row(
    correlation_id: UUID,
    tenant_id: UUID,
    *,
    topic: str,
    envelope_id: UUID,
    parent_id: UUID | None,
    offset: int,
) -> ExecutionGraphLedgerRecord:
    return ExecutionGraphLedgerRecord(
        ledger_entry_id=uuid4(),
        topic=topic,
        partition=0,
        kafka_offset=offset,
        event_key=None,
        event_value=json.dumps(
            {"tenant_id": str(tenant_id), "prompt": "SECRET_NOT_FOR_GRAPH"}
        ).encode(),
        onex_headers=json.dumps(
            {"parent_message_id": str(parent_id)} if parent_id else {}
        ),
        envelope_id=envelope_id,
        correlation_id=correlation_id,
        event_type=None,
        source=None,
        event_timestamp=None,
        ledger_written_at=READ_AT,
    )


def _admission(
    request: ModelExecutionGraphRequest, tenant_id: UUID
) -> ExecutionGraphOwnershipAdmission:
    owner = object.__new__(DelegationOwnerProof)
    object.__setattr__(owner, "tenant_id", tenant_id)
    object.__setattr__(owner, "correlation_id", request.correlation_id)
    head_id = uuid4()
    rows = (
        _row(
            request.correlation_id,
            tenant_id,
            topic=HEAD_TOPIC,
            envelope_id=head_id,
            parent_id=None,
            offset=12,
        ),
        _row(
            request.correlation_id,
            tenant_id,
            topic=CHILD_TOPIC,
            envelope_id=uuid4(),
            parent_id=head_id,
            offset=7,
        ),
    )
    return ExecutionGraphOwnershipAdmission(
        owner=owner,
        head_envelope_id=head_id,
        owned_rows=rows,
        owned_envelope_ids=tuple(row.envelope_id for row in rows if row.envelope_id),
        withheld_envelope_ids=(),
        withheld_count=0,
    )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_latest_folds_owned_rows_without_leaking_raw_body() -> None:
    request = ModelExecutionGraphRequest(
        correlation_id=uuid4(), cursor_mode=EnumExecutionGraphCursorMode.LATEST
    )
    tenant_id = uuid4()
    terminal = await ExecutionGraphReadFold(
        workflow_type="delegation-execution-graph-read",
        read_clock=lambda: READ_AT,
    )(
        request,
        _authority(request, tenant_id),
        _topology(),
        _admission(request, tenant_id),
    )

    assert terminal.status == "completed"
    assert terminal.result is not None
    assert len(terminal.result.replay.nodes) == 2
    assert terminal.result.replay.source_cursors == (
        ModelExecutionGraphSourceCursor(
            topic=CHILD_TOPIC, partition=0, max_kafka_offset=7
        ),
        ModelExecutionGraphSourceCursor(
            topic=HEAD_TOPIC, partition=0, max_kafka_offset=12
        ),
    )
    assert "SECRET_NOT_FOR_GRAPH" not in terminal.model_dump_json()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_bounded_request_folds_only_selected_partition_offsets() -> None:
    request = ModelExecutionGraphRequest(
        correlation_id=uuid4(),
        cursor_mode=EnumExecutionGraphCursorMode.BOUNDED,
        source_cursors=(
            ModelExecutionGraphSourceCursor(
                topic=HEAD_TOPIC, partition=0, max_kafka_offset=12
            ),
            ModelExecutionGraphSourceCursor(
                topic=CHILD_TOPIC, partition=0, max_kafka_offset=7
            ),
        ),
    )
    tenant_id = uuid4()
    terminal = await ExecutionGraphReadFold(
        workflow_type="delegation-execution-graph-read",
        read_clock=lambda: READ_AT,
    )(
        request,
        _authority(request, tenant_id),
        _topology(),
        _admission(request, tenant_id),
    )

    assert terminal.status == "completed"
    assert terminal.result is not None
    assert len(terminal.result.replay.nodes) == 2
    assert terminal.result.replay.source_cursors == tuple(
        sorted(request.source_cursors or (), key=lambda cursor: cursor.topic)
    )
    assert terminal.refusal is None


@pytest.mark.asyncio
@pytest.mark.unit
async def test_bounded_request_excludes_rows_above_selected_offset() -> None:
    request = ModelExecutionGraphRequest(
        correlation_id=uuid4(),
        cursor_mode=EnumExecutionGraphCursorMode.BOUNDED,
        source_cursors=(
            ModelExecutionGraphSourceCursor(
                topic=HEAD_TOPIC, partition=0, max_kafka_offset=12
            ),
            ModelExecutionGraphSourceCursor(
                topic=CHILD_TOPIC, partition=0, max_kafka_offset=6
            ),
        ),
    )
    tenant_id = uuid4()
    terminal = await ExecutionGraphReadFold(
        workflow_type="delegation-execution-graph-read",
        read_clock=lambda: READ_AT,
    )(
        request,
        _authority(request, tenant_id),
        _topology(),
        _admission(request, tenant_id),
    )

    assert terminal.status == "completed"
    assert terminal.result is not None
    assert len(terminal.result.replay.nodes) == 1
    assert terminal.result.replay.nodes[0].topic == HEAD_TOPIC
    assert terminal.result.replay.edges == ()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_full_read_withheld_count_is_visible_without_exposing_rows() -> None:
    request = ModelExecutionGraphRequest(
        correlation_id=uuid4(), cursor_mode=EnumExecutionGraphCursorMode.LATEST
    )
    tenant_id = uuid4()
    admission = replace(
        _admission(request, tenant_id),
        withheld_envelope_ids=(uuid4(), uuid4()),
        withheld_count=2,
    )
    terminal = await ExecutionGraphReadFold(
        workflow_type="delegation-execution-graph-read",
        read_clock=lambda: READ_AT,
    )(request, _authority(request, tenant_id), _topology(), admission)

    assert terminal.status == "completed"
    assert terminal.result is not None
    assert terminal.result.replay.withheld_count == 2
    assert all(
        str(envelope_id) not in terminal.model_dump_json()
        for envelope_id in admission.withheld_envelope_ids
    )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_latest_refuses_admitted_reroute_without_partial_success() -> None:
    request = ModelExecutionGraphRequest(
        correlation_id=uuid4(), cursor_mode=EnumExecutionGraphCursorMode.LATEST
    )
    tenant_id = uuid4()
    original = _admission(request, tenant_id)
    reroute_id = uuid4()
    reroute = _row(
        request.correlation_id,
        tenant_id,
        topic="onex.evt.omnibase-infra.quality-gate-result.v1",
        envelope_id=reroute_id,
        parent_id=original.owned_rows[1].envelope_id,
        offset=40,
    )
    admission = ExecutionGraphOwnershipAdmission(
        owner=original.owner,
        head_envelope_id=original.head_envelope_id,
        owned_rows=(*original.owned_rows, reroute),
        owned_envelope_ids=(*original.owned_envelope_ids, reroute_id),
        withheld_envelope_ids=(),
        withheld_count=0,
    )
    terminal = await ExecutionGraphReadFold(
        workflow_type="delegation-execution-graph-read",
        read_clock=lambda: READ_AT,
    )(request, _authority(request, tenant_id), _topology(), admission)

    assert terminal.status == "failed"
    assert terminal.result is None
    assert terminal.refusal is not None
    assert terminal.refusal.code == "invalid_graph_evidence"
