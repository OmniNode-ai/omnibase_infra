# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit capture flow with fake HTTP/Kafka and real test-only wire signatures."""

import asyncio
import base64
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from aiokafka.structs import TopicPartition

from omnibase_core.crypto import generate_keypair
from omnibase_core.models.envelope.model_message_envelope import ModelMessageEnvelope
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.runtime.sim_archive_source_receipt import RawSimArchiveRecord
from scripts.runtime_build import capture_sim_preflight_graph_read as capture_tool

pytestmark = pytest.mark.unit


@dataclass
class FakeConsumer:
    records: list[RawSimArchiveRecord]
    events: list[str] = field(default_factory=list)
    ends: dict[TopicPartition, int] = field(default_factory=dict)
    assigned: list[TopicPartition] = field(default_factory=list)
    seeks: list[tuple[TopicPartition, int]] = field(default_factory=list)
    stopped: bool = False
    hang: bool = False

    async def start(self) -> None:
        self.events.append("start")

    async def stop(self) -> None:
        self.stopped = True

    def partitions_for_topic(self, topic: str) -> set[int]:
        return {0, 1}

    async def end_offsets(
        self, partitions: list[TopicPartition]
    ) -> dict[TopicPartition, int]:
        self.events.append("snapshot")
        return self.ends

    def assign(self, partitions: list[TopicPartition]) -> None:
        self.assigned = partitions

    def seek(self, partition: TopicPartition, offset: int) -> None:
        self.seeks.append((partition, offset))

    async def getone(self) -> RawSimArchiveRecord:
        if self.hang:
            await asyncio.sleep(60)
        return self.records.pop(0)


def inputs(
    tmp_path: Path,
) -> tuple[
    capture_tool.ModelGraphReadCaptureConfig,
    capture_tool.ModelGraphReadAck,
    FakeConsumer,
]:
    tmp_path.chmod(0o700)
    tenant, correlation, workflow = uuid4(), uuid4(), uuid4()
    key = generate_keypair()
    keymap = tmp_path / "keymap.json"
    keymap.write_text(
        json.dumps(
            {
                "keys": {
                    "gateway-test": base64.urlsafe_b64encode(
                        key.public_key_bytes
                    ).decode()
                }
            }
        )
    )
    token = tmp_path / "token.jwt"
    token.write_text("unit-token.unit-body.unit-signature")
    metadata = tmp_path / "owner.metadata"
    metadata.write_text(
        f"correlation_id={correlation}\ntenant_id={tenant}\nsource_db=omnidash_analytics\nsource_table=public.delegation_events\nrow_sha256={'a' * 64}\nschema_sha256={'b' * 64}\n"
    )
    for path in (keymap, token, metadata):
        path.chmod(0o600)
    inner = ModelEventEnvelope[dict[str, object]](
        correlation_id=correlation,
        tenant_id=str(tenant),
        event_type="omnibase-infra.delegation-execution-graph-requested",
        metadata={
            "tags": {
                "workflow_id": str(workflow),
                "workflow_type": "delegation-execution-graph-read",
                "contract_id": "node_execution_graph_read_effect:1.0.0",
            }
        },
        payload={
            "correlation_id": str(correlation),
            "cursor_mode": "latest",
            "source_cursors": None,
        },
    )
    signed = ModelMessageEnvelope[dict[str, object]].create_signed(
        realm="test",
        runtime_id="gateway-test",
        bus_id="test-bus",
        trace_id=correlation,
        tenant_id=str(tenant),
        payload=inner.model_dump(mode="json"),
        private_key=key.private_key_bytes,
    )
    raw = RawSimArchiveRecord(
        capture_tool._TOPIC, 1, 9, b"key", signed.model_dump_json().encode(), (), 1
    )
    ack = capture_tool.ModelGraphReadAck(
        workflow_id=workflow,
        envelope_id=inner.envelope_id,
        correlation_id=correlation,
        workflow_type="delegation-execution-graph-read",
        status="published",
        accepted_at=datetime.now(UTC),
    )
    config = capture_tool.ModelGraphReadCaptureConfig.model_validate(
        {
            "token_path": token,
            "owner_metadata_path": metadata,
            "signed_command_output": tmp_path / "signed-wire.json",
            "gateway": {
                "command_topic": capture_tool._TOPIC,
                "runtime_id": "gateway-test",
                "realm": "test",
                "bus_id": "test-bus",
                "public_key_path": keymap,
            },
            "kafka": {"bootstrap_servers": "127.0.0.1:65092"},
            "topic_namespace": "",
        }
    )
    consumer = FakeConsumer(
        [raw],
        ends={
            TopicPartition(capture_tool._TOPIC, 0): 4,
            TopicPartition(capture_tool._TOPIC, 1): 9,
        },
    )
    return config, ack, consumer


@pytest.mark.asyncio
async def test_snapshots_before_submit_then_saves_exact_verified_wire(
    tmp_path: Path,
) -> None:
    config, ack, consumer = inputs(tmp_path)
    wire = consumer.records[0].value

    async def submit(
        token: str, body: dict[str, object], timeout: float
    ) -> capture_tool.ModelGraphReadAck:
        assert token == "unit-token.unit-body.unit-signature"
        assert consumer.events == ["start", "snapshot"]
        assert len(consumer.seeks) == 2
        assert body == {
            "workflow_type": "delegation-execution-graph-read",
            "correlation_id": str(ack.correlation_id),
            "payload": {"cursor_mode": "latest"},
        }
        consumer.events.append("post")
        return ack

    result = await capture_tool.capture(
        config, consumer_factory=lambda: consumer, submit=submit
    )
    assert result["status"] == "captured_authorized_wire"
    assert result["partition"] == 1 and result["offset"] == 9
    assert config.signed_command_output.read_bytes() == wire
    assert config.signed_command_output.stat().st_mode & 0o777 == 0o600
    assert consumer.stopped


@pytest.mark.parametrize("field", ["workflow_id", "correlation_id", "envelope_id"])
@pytest.mark.asyncio
async def test_real_ack_mismatch_never_writes_output(
    tmp_path: Path, field: str
) -> None:
    config, ack, consumer = inputs(tmp_path)
    wrong = ack.model_copy(update={field: uuid4()})
    consumer.hang = field == "workflow_id"
    config = config.model_copy(update={"timeout_seconds": 0.02})
    with pytest.raises((ValueError, TimeoutError)):
        await capture_tool.capture(
            config,
            consumer_factory=lambda: consumer,
            submit=AsyncMock(return_value=wrong),
        )
    assert not config.signed_command_output.exists()
    assert consumer.stopped


@pytest.mark.asyncio
async def test_tampered_tenant_signature_is_not_authority(tmp_path: Path) -> None:
    config, ack, consumer = inputs(tmp_path)
    record = consumer.records[0]
    data = json.loads(record.value)
    data["tenant_id"] = str(uuid4())
    consumer.records[0] = RawSimArchiveRecord(
        record.topic,
        record.partition,
        record.offset,
        record.key,
        json.dumps(data).encode(),
        (),
        1,
    )
    with pytest.raises(PermissionError):
        await capture_tool.capture(
            config,
            consumer_factory=lambda: consumer,
            submit=AsyncMock(return_value=ack),
        )
    assert not config.signed_command_output.exists()
    assert consumer.stopped


@pytest.mark.parametrize(
    "problem", ["public_token", "duplicate_metadata", "existing_output", "missing_ends"]
)
@pytest.mark.asyncio
async def test_preflight_failures_do_not_submit(tmp_path: Path, problem: str) -> None:
    config, _, consumer = inputs(tmp_path)
    if problem == "public_token":
        config.token_path.chmod(0o644)
    elif problem == "duplicate_metadata":
        config.owner_metadata_path.write_text(
            config.owner_metadata_path.read_text() + "tenant_id=duplicate\n"
        )
    elif problem == "existing_output":
        config.signed_command_output.write_bytes(b"existing-private-data")
    else:
        consumer.ends = {}
    submit = AsyncMock()
    with pytest.raises(ValueError):
        await capture_tool.capture(
            config, consumer_factory=lambda: consumer, submit=submit
        )
    submit.assert_not_awaited()
    if problem == "existing_output":
        assert config.signed_command_output.read_bytes() == b"existing-private-data"
    else:
        assert not config.signed_command_output.exists()


@pytest.mark.asyncio
async def test_deadline_stops_consumer_and_never_writes(tmp_path: Path) -> None:
    config, ack, consumer = inputs(tmp_path)
    consumer.hang = True
    config = config.model_copy(update={"timeout_seconds": 0.02})
    with pytest.raises(TimeoutError):
        await capture_tool.capture(
            config,
            consumer_factory=lambda: consumer,
            submit=AsyncMock(return_value=ack),
        )
    assert consumer.stopped
    assert not config.signed_command_output.exists()


def test_production_consumer_never_joins_group_or_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _, _ = inputs(tmp_path)
    factory = MagicMock()
    monkeypatch.setattr(capture_tool, "AIOKafkaConsumer", factory)
    capture_tool._KafkaCaptureConsumer(config.kafka)
    assert factory.call_args.kwargs["group_id"] is None
    assert factory.call_args.kwargs["enable_auto_commit"] is False
    assert factory.call_args.args == (capture_tool._TOPIC,)


@pytest.mark.asyncio
async def test_cleanup_cancellation_preserves_original_capture_refusal(
    tmp_path: Path,
) -> None:
    config, _, consumer = inputs(tmp_path)
    consumer.ends = {}
    consumer.stop = AsyncMock(side_effect=asyncio.CancelledError())
    with pytest.raises(ValueError, match="end snapshot is incomplete"):
        await capture_tool.capture(
            config, consumer_factory=lambda: consumer, submit=AsyncMock()
        )
    assert not config.signed_command_output.exists()


@pytest.mark.asyncio
async def test_failure_diagnostic_identifies_phase_without_private_values(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config, _, consumer = inputs(tmp_path)
    private_message = "private-token-and-response-example"
    submit = AsyncMock(side_effect=ValueError(private_message))
    with pytest.raises(ValueError, match=private_message):
        await capture_tool.capture(
            config, consumer_factory=lambda: consumer, submit=submit
        )
    assert "phase=submit" in caplog.text
    assert "error_kind=refusal" in caplog.text
    assert private_message not in caplog.text
    assert consumer.stopped


@pytest.mark.parametrize(
    "broker",
    [
        "localhost:65092",
        "127.0.0.1:19092",
        "192.168.86.201:29092",  # cloud-bus-ok OMN-19726 # kafka-fallback-ok: rejected fixture
    ],
)
def test_config_refuses_other_brokers(tmp_path: Path, broker: str) -> None:
    config, _, _ = inputs(tmp_path)
    data = config.model_dump(exclude_computed_fields=True)
    data["kafka"]["bootstrap_servers"] = broker
    with pytest.raises(ValueError, match="exact local"):
        capture_tool.ModelGraphReadCaptureConfig.model_validate(data)


@pytest.mark.asyncio
async def test_explicit_namespace_is_used_without_ambient_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, ack, consumer = inputs(tmp_path)
    monkeypatch.setenv("KAFKA_TOPIC_NAMESPACE", "wrong-ambient")
    config = config.model_copy(update={"topic_namespace": "sim-preflight"})
    record = consumer.records[0]
    topic = "sim-preflight." + capture_tool._TOPIC
    consumer.records[0] = RawSimArchiveRecord(
        topic, record.partition, record.offset, record.key, record.value, (), 1
    )
    consumer.ends = {TopicPartition(topic, 0): 4, TopicPartition(topic, 1): 9}
    await capture_tool.capture(
        config, consumer_factory=lambda: consumer, submit=AsyncMock(return_value=ack)
    )
    assert {partition.topic for partition in consumer.assigned} == {topic}


@pytest.mark.parametrize("problem", ["owner_tenant", "signer_scope"])
@pytest.mark.asyncio
async def test_valid_signature_still_requires_owner_and_policy(
    tmp_path: Path, problem: str
) -> None:
    config, ack, consumer = inputs(tmp_path)
    if problem == "owner_tenant":
        lines = config.owner_metadata_path.read_text().splitlines()
        config.owner_metadata_path.write_text(
            "\n".join(
                "tenant_id=" + str(uuid4()) if line.startswith("tenant_id=") else line
                for line in lines
            )
            + "\n"
        )
    else:
        config = config.model_copy(
            update={"gateway": config.gateway.model_copy(update={"realm": "wrong"})}
        )
    with pytest.raises((ValueError, PermissionError)):
        await capture_tool.capture(
            config,
            consumer_factory=lambda: consumer,
            submit=AsyncMock(return_value=ack),
        )
    assert not config.signed_command_output.exists()
    assert consumer.stopped


@pytest.mark.parametrize("status", [202, 302, 401, 403, 500])
@pytest.mark.asyncio
async def test_http_submit_fixed_target_no_redirect_or_ambient_proxy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    _, ack, _ = inputs(tmp_path)
    client = AsyncMock()
    response = MagicMock(status_code=status)
    response.json.return_value = ack.model_dump(mode="json")
    client.post.return_value = response
    context = AsyncMock()
    context.__aenter__.return_value = client
    factory = MagicMock(return_value=context)
    monkeypatch.setattr(capture_tool.httpx, "AsyncClient", factory)
    body = {"workflow_type": capture_tool._WORKFLOW}
    if status == 202:
        assert await capture_tool._submit("private-token", body, 2) == ack
    else:
        with pytest.raises(ValueError, match="did not publish"):
            await capture_tool._submit("private-token", body, 2)
    factory.assert_called_once_with(trust_env=False, follow_redirects=False, timeout=2)
    client.post.assert_awaited_once_with(
        "http://localhost:8090/v1/workflows",
        headers={"Authorization": "Bearer private-token"},
        json=body,
    )


def test_cli_refusal_does_not_print_private_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    private = tmp_path / "invalid.json"
    private.write_text('{"secret": "never-print-this-token"}')
    private.chmod(0o600)
    monkeypatch.setattr("sys.argv", ["capture", "--config", str(private)])
    assert capture_tool.main() == 65
    assert json.loads(capsys.readouterr().out) == {"status": "refused"}
