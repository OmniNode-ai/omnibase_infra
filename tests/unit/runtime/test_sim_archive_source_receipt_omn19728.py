# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A caller's archive/manifest cannot authorize sim outbox enqueue."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from omnibase_core.crypto.crypto_ed25519_signer import generate_keypair
from omnibase_core.models.envelope.model_message_envelope import ModelMessageEnvelope
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.execution_graph_replay.model_execution_graph_topology_version import (
    ModelExecutionGraphTopologyVersion,
)
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.runtime.db.execution_graph_read_adapters import (
    _OWNER_PROOF_MINT,
    DelegationOwnerProof,
    ExecutionGraphCurrentEvidenceReader,
    ExecutionGraphLedgerRecord,
)
from omnibase_infra.runtime.execution_graph_ownership import (
    ExecutionGraphOwnershipRefusalError,
)
from omnibase_infra.runtime.execution_graph_read_authority import (
    TrustedExecutionGraphGatewayPolicy,
    TrustedGatewaySignerScope,
    verify_signed_execution_graph_read_authority,
)
from omnibase_infra.runtime.execution_graph_topology_registry import (
    PackagedExecutionGraphTopologyContract,
)
from omnibase_infra.runtime.sim_archive_source_receipt import (
    KafkaSimArchiveSourceReader,
    RawSimArchiveRecord,
    SimArchiveSourceReceiptVerifier,
    VerifiedSimArchiveRehydrationPlan,
)

pytestmark = pytest.mark.unit
_TOPOLOGY_SHA = "0505ab0b163492380739a15646c0442a3ecfb54efb0acb53bc23d232640fbfd3"


def _authority():
    tenant, correlation = uuid4(), uuid4()
    scope = TrustedGatewaySignerScope(
        runtime_id="trusted-api-gateway", realm="test", bus_id="graph-read"
    )
    inner = ModelEventEnvelope[dict[str, object]](
        tenant_id=str(tenant),
        correlation_id=correlation,
        metadata={"tags": {"workflow_id": str(uuid4())}},
        event_type="omnibase-infra.delegation-execution-graph-requested",
        payload={
            "correlation_id": str(correlation),
            "cursor_mode": "latest",
            "source_cursors": None,
        },
    ).model_dump(mode="json")
    keys = generate_keypair()
    envelope = ModelMessageEnvelope[dict[str, object]].create_signed(
        realm=scope.realm,
        runtime_id=scope.runtime_id,
        bus_id=scope.bus_id,
        trace_id=correlation,
        tenant_id=str(tenant),
        payload=inner,
        private_key=keys.private_key_bytes,
    )

    from tests.helpers.projection_tenant_authority import InMemoryKeyProvider

    return verify_signed_execution_graph_read_authority(
        envelope,
        InMemoryKeyProvider({scope.runtime_id: keys.public_key_bytes}),
        TrustedExecutionGraphGatewayPolicy(scopes=frozenset({scope})),
    )


def _topology():
    return PackagedExecutionGraphTopologyContract().resolve(
        ModelExecutionGraphTopologyVersion(
            contract_version=ModelSemVer(major=1, minor=3, patch=0),
            topology_sha256=_TOPOLOGY_SHA,
        )
    )


def _kafka_config():
    from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

    return ModelKafkaEventBusConfig(
        bootstrap_servers="broker:9092",
        environment="test",
    )


class _OwnerReader:
    def __init__(self, authority, calls: list[str]) -> None:
        self.authority = authority
        self.calls = calls

    async def require_owner(self, authority):
        self.calls.append("owner")
        assert authority is self.authority
        return DelegationOwnerProof(
            correlation_id=authority.correlation_id,
            tenant_id=authority.tenant_id,
            _mint=_OWNER_PROOF_MINT,
        )


class _LedgerReader:
    def __init__(self, rows, calls: list[str]) -> None:
        self.rows = rows
        self.calls = calls

    async def read_full_current(self, authority, owner, read_set):
        self.calls.append("full_current")
        assert owner.correlation_id == authority.correlation_id
        assert read_set.head_topic == self.rows[0].topic
        return self.rows


class _SourceReader:
    def __init__(self, records, calls: list[str]) -> None:
        self.records = {record.source_key: record for record in records}
        self.calls = calls

    async def read_exact(self, topic: str, partition: int, offset: int):
        self.calls.append("fresh_source")
        return self.records.get((topic, partition, offset))


class _KafkaRecord:
    def __init__(
        self,
        *,
        topic: str,
        partition: int,
        offset: int,
        key: bytes | None,
        value: bytes,
        headers: tuple[tuple[str, bytes | None], ...],
        timestamp: int,
    ) -> None:
        self.topic = topic
        self.partition = partition
        self.offset = offset
        self.key = key
        self.value = value
        self.headers = headers
        self.timestamp = timestamp


class _ExactConsumer:
    def __init__(self, record: object) -> None:
        self.record = record
        self.assigned: list[object] = []
        self.seek_calls: list[tuple[object, int]] = []
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def assign(self, partitions: list[object]) -> None:
        self.assigned = partitions

    def seek(self, partition: object, offset: int) -> None:
        self.seek_calls.append((partition, offset))

    async def getone(self) -> object:
        return self.record


def _fixture():
    authority = _authority()
    topology = _topology()
    topics = tuple(hop.topic for hop in topology.declared_chain)
    ids = tuple(uuid4() for _ in topics)
    parents: tuple[UUID | None, ...] = (None, ids[0], ids[1], ids[2], ids[0])
    rows = []
    records = []
    for index, (topic, envelope_id, parent) in enumerate(
        zip(topics, ids, parents, strict=True)
    ):
        body = {
            "correlation_id": str(authority.correlation_id),
            "envelope_id": str(envelope_id),
            "parent_envelope_id": str(parent) if parent else None,
            "tenant_id": str(authority.tenant_id) if index in (0, 1, 4) else None,
        }
        value = json.dumps(body, sort_keys=True).encode()
        parent_headers = (
            (("parent_message_id", str(parent).encode()),) if parent else ()
        )
        headers = (
            ("correlation_id", str(authority.correlation_id).encode()),
            ("message_id", str(envelope_id).encode()),
            *parent_headers,
            ("timestamp", b"2026-09-27T00:00:00+00:00"),
            ("source", b"test"),
            ("event_type", b"test"),
        )
        records.append(
            RawSimArchiveRecord(
                topic=topic,
                partition=0,
                offset=index + 10,
                key=str(authority.correlation_id).encode(),
                value=value,
                headers=headers,
                timestamp_ms=1_800_000_000_000 + index,
            )
        )
        rows.append(
            ExecutionGraphLedgerRecord(
                ledger_entry_id=uuid4(),
                topic=topic,
                partition=0,
                kafka_offset=index + 10,
                event_key=records[-1].key,
                event_value=value,
                onex_headers=json.dumps(
                    {"parent_message_id": str(parent)} if parent else {}
                ),
                envelope_id=envelope_id,
                correlation_id=authority.correlation_id,
                event_type="test",
                source="test",
                event_timestamp=datetime.now(UTC),
                ledger_written_at=datetime.now(UTC),
            )
        )
    calls: list[str] = []
    source = _SourceReader(records, calls)
    current_reader = ExecutionGraphCurrentEvidenceReader(
        _OwnerReader(authority, calls), _LedgerReader(tuple(rows), calls)
    )
    adapter = SimArchiveSourceReceiptVerifier(
        current_reader=current_reader, source_reader=source, topology=topology
    )
    return authority, adapter, tuple(records), tuple(rows), source, calls


@pytest.mark.asyncio
async def test_receipt_verifies_full_owner_first_tree_and_exact_raw_bytes() -> None:
    authority, adapter, records, _rows, _source, calls = _fixture()
    plans = await adapter.verify(authority, records)
    assert len(plans) == 5
    assert all(type(plan) is VerifiedSimArchiveRehydrationPlan for plan in plans)
    assert [plan.source_key for plan in plans] == [
        record.source_key for record in records
    ]
    assert calls[:2] == ["owner", "full_current"]
    assert calls.count("fresh_source") == 5
    with pytest.raises(TypeError, match="verified source receipt"):
        VerifiedSimArchiveRehydrationPlan(records[0], _mint=object())


@pytest.mark.asyncio
async def test_receipt_rejects_archive_header_or_timestamp_mismatch() -> None:
    authority, adapter, records, _rows, _source, _calls = _fixture()
    changed = (
        replace(records[0], headers=records[0].headers + (("extra", b"x"),)),
        *records[1:],
    )
    with pytest.raises(ValueError, match="fresh source receipt"):
        await adapter.verify(authority, changed)
    changed_time = (replace(records[0], timestamp_ms=1), *records[1:])
    with pytest.raises(ValueError, match="fresh source receipt"):
        await adapter.verify(authority, changed_time)


@pytest.mark.asyncio
async def test_receipt_rejects_source_or_ledger_mismatch() -> None:
    authority, adapter, records, _rows, source, _calls = _fixture()
    source.records[records[0].source_key] = replace(records[0], value=b"different")
    with pytest.raises(ValueError, match="fresh source receipt"):
        await adapter.verify(authority, records)


@pytest.mark.asyncio
async def test_missing_original_header_refuses_even_when_archive_matches_source() -> (
    None
):
    authority, adapter, records, _rows, source, _calls = _fixture()
    head = replace(records[0], headers=())
    source.records[head.source_key] = head
    with pytest.raises(ValueError, match="required original source identity"):
        await adapter.verify(authority, (head, *records[1:]))


@pytest.mark.asyncio
async def test_second_head_outside_selection_refuses_before_source_read() -> None:
    authority, _adapter, records, rows, source, calls = _fixture()
    second = replace(
        rows[0], ledger_entry_id=uuid4(), kafka_offset=99, envelope_id=uuid4()
    )
    current_reader = ExecutionGraphCurrentEvidenceReader(
        _OwnerReader(authority, calls), _LedgerReader((*rows, second), calls)
    )
    adapter = SimArchiveSourceReceiptVerifier(
        current_reader=current_reader, source_reader=source, topology=_topology()
    )
    with pytest.raises(ExecutionGraphOwnershipRefusalError):
        await adapter.verify(authority, records)
    assert calls == ["owner", "full_current"]


@pytest.mark.asyncio
async def test_caller_constructed_manifest_cannot_replace_signed_authority() -> None:
    _authority_proof, adapter, records, _rows, _source, calls = _fixture()
    with pytest.raises(TypeError, match="signed gateway authority"):
        await adapter.verify(object(), records)  # type: ignore[arg-type]
    assert calls == []


@pytest.mark.asyncio
async def test_kafka_source_reader_assigns_and_seeks_without_a_consumer_group() -> None:
    record = _KafkaRecord(
        topic="onex.cmd.omnimarket.delegate-skill.v1",
        partition=2,
        offset=11,
        key=b"key",
        value=b"value",
        headers=(("header", b"value"), ("repeated", None)),
        timestamp=1_800_000_000_123,
    )
    consumer = _ExactConsumer(record)
    reader = KafkaSimArchiveSourceReader(
        _kafka_config(),
        source_topic_namespace="",
        consumer_factory=lambda: consumer,
        timeout_seconds=1,
    )

    observed = await reader.read_exact(record.topic, record.partition, record.offset)

    assert observed == RawSimArchiveRecord(
        topic=record.topic,
        partition=record.partition,
        offset=record.offset,
        key=record.key,
        value=record.value,
        headers=record.headers,
        timestamp_ms=record.timestamp,
    )
    assert consumer.started is True
    assert consumer.stopped is True
    assert len(consumer.assigned) == 1
    assert len(consumer.seek_calls) == 1


@pytest.mark.asyncio
async def test_kafka_source_reader_preserves_aiokafka_list_header_order() -> None:
    record = _KafkaRecord(
        topic="onex.cmd.omnimarket.delegate-skill.v1",
        partition=2,
        offset=11,
        key=b"key",
        value=b"value",
        headers=(("first", b"one"), ("first", b"two")),
        timestamp=1,
    )
    record.headers = list(record.headers)
    consumer = _ExactConsumer(record)
    reader = KafkaSimArchiveSourceReader(
        _kafka_config(),
        source_topic_namespace="",
        consumer_factory=lambda: consumer,
        timeout_seconds=1,
    )

    observed = await reader.read_exact(record.topic, record.partition, record.offset)

    assert observed is not None
    assert observed.headers == (("first", b"one"), ("first", b"two"))


@pytest.mark.asyncio
async def test_kafka_source_reader_refuses_a_wrong_coordinate_without_rewriting_it() -> (
    None
):
    consumer = _ExactConsumer(
        _KafkaRecord(
            topic="onex.cmd.omnimarket.delegate-skill.v1",
            partition=2,
            offset=12,
            key=b"key",
            value=b"value",
            headers=(),
            timestamp=1,
        )
    )
    reader = KafkaSimArchiveSourceReader(
        _kafka_config(),
        source_topic_namespace="",
        consumer_factory=lambda: consumer,
        timeout_seconds=1,
    )

    assert (
        await reader.read_exact("onex.cmd.omnimarket.delegate-skill.v1", 2, 11) is None
    )
    assert consumer.stopped is True


@pytest.mark.asyncio
async def test_source_namespace_is_explicit_not_ambient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KAFKA_TOPIC_NAMESPACE", "wrong-sim")
    canonical = "onex.cmd.omnimarket.delegate-skill.v1"
    consumer = _ExactConsumer(
        _KafkaRecord(
            topic=f"source-lane.{canonical}",
            partition=2,
            offset=11,
            key=None,
            value=b"source",
            headers=(),
            timestamp=1,
        )
    )
    reader = KafkaSimArchiveSourceReader(
        _kafka_config(),
        source_topic_namespace="source-lane",
        consumer_factory=lambda: consumer,
    )

    observed = await reader.read_exact(canonical, 2, 11)

    assert observed is not None
    assert observed.topic == canonical
    assert consumer.assigned[0].topic == f"source-lane.{canonical}"
