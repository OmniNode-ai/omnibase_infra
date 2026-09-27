# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The graph command seam cannot read or publish without signed admission."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import UTC, datetime
from typing import cast
from uuid import UUID, uuid4

import pytest

from omnibase_core.crypto.crypto_ed25519_signer import generate_keypair
from omnibase_core.models.envelope.model_message_envelope import ModelMessageEnvelope
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.execution_graph_replay.model_execution_graph_request import (
    ModelExecutionGraphRequest,
)
from omnibase_core.models.execution_graph_replay.model_execution_graph_stored_chain_annotation import (
    ModelExecutionGraphStoredChainAnnotation,
)
from omnibase_core.models.execution_graph_replay.model_execution_graph_terminal_refusal import (
    ModelExecutionGraphTerminalRefusal,
)
from omnibase_core.models.execution_graph_replay.model_execution_graph_terminal_result import (
    ModelExecutionGraphTerminalResult,
)
from omnibase_infra.runtime.db.execution_graph_read_adapters import (
    DelegationOwnerProof,
    ExecutionGraphCurrentEvidence,
    ExecutionGraphCurrentEvidenceReader,
    ExecutionGraphLedgerRecord,
)
from omnibase_infra.runtime.db.protocol_execution_graph_stored_chain_reader import (
    ProtocolExecutionGraphStoredChainReader,
)
from omnibase_infra.runtime.dispatch_envelope_context import (
    bind_execution_graph_read_authority,
)
from omnibase_infra.runtime.execution_graph_ownership import (
    ExecutionGraphOwnershipAdmission,
)
from omnibase_infra.runtime.execution_graph_read_authority import (
    TrustedExecutionGraphGatewayPolicy,
    TrustedGatewaySignerScope,
    VerifiedExecutionGraphReadAuthority,
    verify_signed_execution_graph_read_authority,
)
from omnibase_infra.runtime.execution_graph_read_command_handler import (
    ExecutionGraphReadCommandError,
    ExecutionGraphReadCommandExecutor,
)
from omnibase_infra.runtime.execution_graph_topology_registry import (
    PackagedExecutionGraphTopologyContract,
    PinnedExecutionGraphTopology,
)
from tests.helpers.projection_tenant_authority import InMemoryKeyProvider

_SCOPE = TrustedGatewaySignerScope(
    runtime_id="trusted-api-gateway", realm="test", bus_id="graph-read"
)
_POLICY = TrustedExecutionGraphGatewayPolicy(scopes=frozenset({_SCOPE}))
_WORKFLOW_TYPE = "delegation_execution_graph_read"


def _signed_authority() -> VerifiedExecutionGraphReadAuthority:
    tenant_id = uuid4()
    correlation_id = uuid4()
    inner = ModelEventEnvelope[dict[str, object]](
        tenant_id=str(tenant_id),
        correlation_id=correlation_id,
        event_type="omnibase-infra.delegation-execution-graph-requested",
        payload={
            "correlation_id": str(correlation_id),
            "cursor_mode": "latest",
            "source_cursors": None,
        },
    ).model_dump(mode="json")
    keys = generate_keypair()
    envelope = ModelMessageEnvelope[dict[str, object]].create_signed(
        realm=_SCOPE.realm,
        runtime_id=_SCOPE.runtime_id,
        bus_id=_SCOPE.bus_id,
        trace_id=correlation_id,
        tenant_id=str(tenant_id),
        payload=inner,
        private_key=keys.private_key_bytes,
    )
    return verify_signed_execution_graph_read_authority(
        envelope,
        InMemoryKeyProvider({_SCOPE.runtime_id: keys.public_key_bytes}),
        _POLICY,
    )


def _topology() -> PinnedExecutionGraphTopology:
    version = PackagedExecutionGraphTopologyContract().resolve
    # The packaged registry is the only production mint; avoid test-built topology.
    from omnibase_core.models.execution_graph_replay.model_execution_graph_topology_version import (
        ModelExecutionGraphTopologyVersion,
    )
    from omnibase_core.models.primitives.model_semver import ModelSemVer

    return version(
        ModelExecutionGraphTopologyVersion(
            contract_version=ModelSemVer(major=1, minor=3, patch=0),
            topology_sha256="0505ab0b163492380739a15646c0442a3ecfb54efb0acb53bc23d232640fbfd3",
        )
    )


def _failed_terminal(
    authority: VerifiedExecutionGraphReadAuthority,
    *,
    tenant_id: UUID | None = None,
) -> ModelExecutionGraphTerminalResult:
    return ModelExecutionGraphTerminalResult(
        tenant_id=tenant_id or authority.tenant_id,
        correlation_id=authority.correlation_id,
        workflow_type=_WORKFLOW_TYPE,
        status="failed",
        refusal=ModelExecutionGraphTerminalRefusal(code="refused", message="test"),
    )


def _evidence(
    authority: VerifiedExecutionGraphReadAuthority,
) -> ExecutionGraphCurrentEvidence:
    owner = object.__new__(DelegationOwnerProof)
    object.__setattr__(owner, "tenant_id", authority.tenant_id)
    object.__setattr__(owner, "correlation_id", authority.correlation_id)
    return ExecutionGraphCurrentEvidence(
        owner=owner,
        ledger_rows=(
            ExecutionGraphLedgerRecord(
                ledger_entry_id=uuid4(),
                topic="onex.cmd.omnimarket.delegate-skill.v1",
                partition=0,
                kafka_offset=1,
                event_key=None,
                event_value=json.dumps(
                    {"tenant_id": str(authority.tenant_id)}
                ).encode(),
                onex_headers="{}",
                envelope_id=uuid4(),
                correlation_id=authority.correlation_id,
                event_type=None,
                source=None,
                event_timestamp=None,
                ledger_written_at=datetime.now(UTC),
            ),
        ),
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unsigned_command_never_reads_folds_or_publishes() -> None:
    calls: list[str] = []
    authority = _signed_authority()

    async def read(*_args: object) -> object:
        calls.append("read")
        return _evidence(authority)

    async def fold(*_args: object) -> ModelExecutionGraphTerminalResult:
        calls.append("fold")
        return _failed_terminal(authority)

    async def publish(*_args: object) -> None:
        calls.append("publish")

    handler = ExecutionGraphReadCommandExecutor(
        evidence_reader=cast("ExecutionGraphCurrentEvidenceReader", _Reader(read)),
        stored_chain_reader=cast(
            "ProtocolExecutionGraphStoredChainReader", _StoredChainReader(calls)
        ),
        topology=_topology(),
        workflow_type=_WORKFLOW_TYPE,
        fold=fold,
        publish_terminal=publish,
    )

    with pytest.raises(ExecutionGraphReadCommandError, match="verified signed"):
        await handler.handle(authority.request)
    assert calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_signed_command_requires_matching_request_before_read() -> None:
    calls: list[str] = []
    authority = _signed_authority()
    mismatched = authority.request.model_copy(update={"correlation_id": uuid4()})

    async def read(*_args: object) -> object:
        calls.append("read")
        return object()

    handler = ExecutionGraphReadCommandExecutor(
        evidence_reader=cast("ExecutionGraphCurrentEvidenceReader", _Reader(read)),
        stored_chain_reader=cast(
            "ProtocolExecutionGraphStoredChainReader", _StoredChainReader(calls)
        ),
        topology=_topology(),
        workflow_type=_WORKFLOW_TYPE,
        fold=cast("Callable[..., Awaitable[ModelExecutionGraphTerminalResult]]", None),
        publish_terminal=cast("Callable[..., Awaitable[None]]", None),
    )

    with bind_execution_graph_read_authority(authority):
        with pytest.raises(ExecutionGraphReadCommandError, match="conflicts"):
            await handler.handle(mismatched)
    assert calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_signed_owner_first_admission_precedes_fold_and_published_terminal_is_bound() -> (
    None
):
    calls: list[str] = []
    authority = _signed_authority()
    evidence = _evidence(authority)
    stored = ModelExecutionGraphStoredChainAnnotation(
        node_id=evidence.ledger_rows[0].envelope_id,
        hop_index=0,
        replay_green=False,
        verifier_verdict="fail",
    )

    async def read(
        received_authority: VerifiedExecutionGraphReadAuthority,
        received_read_set: object,
    ) -> object:
        assert received_authority is authority
        assert received_read_set == _topology().read_set
        calls.append("owner-first-read")
        return evidence

    async def fold(
        request: ModelExecutionGraphRequest,
        received_authority: VerifiedExecutionGraphReadAuthority,
        topology: PinnedExecutionGraphTopology,
        received_admission: ExecutionGraphOwnershipAdmission,
        stored_chain: tuple[ModelExecutionGraphStoredChainAnnotation, ...],
    ) -> ModelExecutionGraphTerminalResult:
        assert request == authority.request
        assert received_authority is authority
        assert topology == _topology()
        assert received_admission.owner == evidence.owner
        assert stored_chain == (stored,)
        assert calls == ["owner-first-read", "stored-chain-read"]
        calls.append("fold")
        return _failed_terminal(authority)

    async def publish(
        received_authority: VerifiedExecutionGraphReadAuthority,
        terminal: ModelExecutionGraphTerminalResult,
    ) -> None:
        assert received_authority is authority
        assert terminal == _failed_terminal(authority)
        assert calls == ["owner-first-read", "stored-chain-read", "fold"]
        calls.append("publish")

    handler = ExecutionGraphReadCommandExecutor(
        evidence_reader=cast("ExecutionGraphCurrentEvidenceReader", _Reader(read)),
        stored_chain_reader=cast(
            "ProtocolExecutionGraphStoredChainReader",
            _StoredChainReader(calls, (stored,)),
        ),
        topology=_topology(),
        workflow_type=_WORKFLOW_TYPE,
        fold=fold,
        publish_terminal=publish,
    )

    with bind_execution_graph_read_authority(authority):
        result = await handler.handle(authority.request)
    assert result == _failed_terminal(authority)
    assert calls == ["owner-first-read", "stored-chain-read", "fold", "publish"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_second_head_refuses_before_stored_chain_read_or_publish() -> None:
    authority = _signed_authority()
    current = _evidence(authority)
    second_head = replace(
        current.ledger_rows[0],
        ledger_entry_id=uuid4(),
        envelope_id=uuid4(),
        kafka_offset=999,
    )
    calls: list[str] = []

    async def read(*_args: object) -> ExecutionGraphCurrentEvidence:
        calls.append("owner-first-read")
        return replace(current, ledger_rows=(*current.ledger_rows, second_head))

    async def fold(*_args: object) -> ModelExecutionGraphTerminalResult:
        calls.append("fold")
        return _failed_terminal(authority)

    async def publish(*_args: object) -> None:
        calls.append("publish")

    handler = ExecutionGraphReadCommandExecutor(
        evidence_reader=cast("ExecutionGraphCurrentEvidenceReader", _Reader(read)),
        stored_chain_reader=cast(
            "ProtocolExecutionGraphStoredChainReader", _StoredChainReader(calls)
        ),
        topology=_topology(),
        workflow_type=_WORKFLOW_TYPE,
        fold=fold,
        publish_terminal=publish,
    )
    with bind_execution_graph_read_authority(authority):
        with pytest.raises(ValueError, match="exactly one"):
            await handler.handle(authority.request)
    assert calls == ["owner-first-read"]


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_kind", ["foreign", "duplicate"])
async def test_faulty_stored_chain_reader_cannot_inject_unowned_or_duplicate_nodes(
    invalid_kind: str,
) -> None:
    authority = _signed_authority()
    current = _evidence(authority)
    head = current.ledger_rows[0].envelope_id
    assert head is not None
    admitted = ModelExecutionGraphStoredChainAnnotation(
        node_id=head, hop_index=0, replay_green=True, verifier_verdict="pass"
    )
    stray = ModelExecutionGraphStoredChainAnnotation(
        node_id=uuid4(), hop_index=1, replay_green=True, verifier_verdict="pass"
    )
    returned = (admitted, stray) if invalid_kind == "foreign" else (admitted,) * 2
    calls: list[str] = []

    async def read(*_args: object) -> ExecutionGraphCurrentEvidence:
        calls.append("owner-first-read")
        return current

    async def fold(*_args: object) -> ModelExecutionGraphTerminalResult:
        calls.append("fold")
        return _failed_terminal(authority)

    async def publish(*_args: object) -> None:
        calls.append("publish")

    handler = ExecutionGraphReadCommandExecutor(
        evidence_reader=cast("ExecutionGraphCurrentEvidenceReader", _Reader(read)),
        stored_chain_reader=cast(
            "ProtocolExecutionGraphStoredChainReader",
            _StoredChainReader(calls, returned),
        ),
        topology=_topology(),
        workflow_type=_WORKFLOW_TYPE,
        fold=fold,
        publish_terminal=publish,
    )
    with bind_execution_graph_read_authority(authority):
        with pytest.raises(ExecutionGraphReadCommandError, match="stored chain"):
            await handler.handle(authority.request)
    assert calls == ["owner-first-read", "stored-chain-read"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_terminal_with_foreign_tenant_is_not_published() -> None:
    authority = _signed_authority()
    calls: list[str] = []

    async def read(*_args: object) -> object:
        return _evidence(authority)

    async def fold(*_args: object) -> ModelExecutionGraphTerminalResult:
        return _failed_terminal(authority, tenant_id=uuid4())

    async def publish(*_args: object) -> None:
        calls.append("publish")

    handler = ExecutionGraphReadCommandExecutor(
        evidence_reader=cast("ExecutionGraphCurrentEvidenceReader", _Reader(read)),
        stored_chain_reader=cast(
            "ProtocolExecutionGraphStoredChainReader", _StoredChainReader(calls)
        ),
        topology=_topology(),
        workflow_type=_WORKFLOW_TYPE,
        fold=fold,
        publish_terminal=publish,
    )

    with bind_execution_graph_read_authority(authority):
        with pytest.raises(ExecutionGraphReadCommandError, match="conflicts"):
            await handler.handle(authority.request)
    assert calls == ["stored-chain-read"]


class _Reader:
    def __init__(self, read: Callable[..., Awaitable[object]]) -> None:
        self._read = read

    async def read_authorized_current(self, *args: object) -> object:
        return await self._read(*args)


class _StoredChainReader:
    def __init__(
        self,
        calls: list[str],
        annotations: tuple[ModelExecutionGraphStoredChainAnnotation, ...] = (),
    ) -> None:
        self._calls = calls
        self._annotations = annotations

    async def read_current(
        self, *_args: object
    ) -> tuple[ModelExecutionGraphStoredChainAnnotation, ...]:
        self._calls.append("stored-chain-read")
        return self._annotations
