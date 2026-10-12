# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20867: the gateway forwarder process runs mirror-only.

A mirror-only deployment consumes one source lane and republishes the topic set
its overlay selects onto the mirror lanes, with no cloud leg at all. The node
model has accepted that shape since OMN-18691; this pins that the PROCESS does:
the loader resolves it from the shipped contract plus a deployment overlay, and
``run_gateway_forwarder`` builds only the lane legs.

Which lanes, which hosts and which credentials a deployment uses come from the
overlay. Everything the shipped contract contributes here is product shape: the
closed set of nameable lanes and the named topic sets.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
import yaml

from omnibase_core.models.runtime.model_transport_message import ModelTransportMessage
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.runtime import gateway_forwarder

pytestmark = pytest.mark.unit

_CONTRACT_PATH = (
    Path(gateway_forwarder.__file__).resolve().parents[1]
    / "nodes"
    / "node_bus_forwarder_effect"
    / "contract.yaml"
)
_TENANT_SLUG = "lane-mirror-capacity-test"
_SOURCE_LANE = "lane-src"
_MIRROR_LANES = ("lane-a", "lane-b")


def _lane_mirror_block() -> dict[str, Any]:
    contract = yaml.safe_load(_CONTRACT_PATH.read_text(encoding="utf-8"))
    block = contract["config"]["gateway_forwarder"]["lane_mirror"]
    assert isinstance(block, dict)
    return block


def _topic_set(name: str) -> tuple[str, ...]:
    return tuple(_lane_mirror_block()["topic_sets"][name])


def _bus(address: str, *, environment: str) -> dict[str, Any]:
    return {
        "bootstrap_servers": address,
        "environment": environment,
        "enable_auto_commit": False,
        "auto_offset_reset": "earliest",
    }


def _mirror_only_overlay(
    tmp_path: Path,
    *,
    topic_sets: Sequence[str],
    mirror_lanes: Sequence[str] = _MIRROR_LANES,
    declared_lanes: Sequence[str] | None = (_SOURCE_LANE, *_MIRROR_LANES),
) -> Path:
    """The deployment YAML a private overlay would supply: no cloud keys at all."""
    binding: dict[str, Any] = {
        "source_lane": _SOURCE_LANE,
        "mirror_lanes": list(mirror_lanes),
        "topic_sets": list(topic_sets),
    }
    if declared_lanes is not None:
        binding["declared_lanes"] = list(declared_lanes)
    overlay = {
        "forwarder": {
            "tenant_identity": {
                "tenant_id": str(UUID("79afa726-3852-464f-b7a4-d4b8b9c75ee7")),
                "tenant_slug": _TENANT_SLUG,
                "principal_id": "t-79afa7263852464fb7a4d4b8b9c75ee7",
            },
            "local_transport_flavor": "containerized",
            "dedupe_store_path": str(tmp_path / "mirror.sqlite3"),
            "lane_mirror_set": "node_bus_forwarder_effect",
            "lane_mirror_binding": binding,
        },
        "lane_mirror_source_bus": _bus("source.invalid:9092", environment="src"),
        "lane_mirror_buses": {
            lane: _bus(f"{lane}.invalid:9092", environment=f"dst-{lane}")
            for lane in mirror_lanes
        },
    }
    path = tmp_path / "mirror-only.yaml"
    path.write_text(yaml.safe_dump(overlay), encoding="utf-8")
    return path


def _load(
    path: Path, tmp_path: Path
) -> gateway_forwarder.ModelGatewayForwarderRuntimeConfig:
    return gateway_forwarder.load_gateway_forwarder_runtime_config(
        path,
        broker_ref_map_path=None,
    )


def test_contract_names_the_two_landing_topic_sets() -> None:
    assert (
        _topic_set("lab_capacity")
        == (
            "onex.evt.omnimarket.lab-host-capacity-advertised.v1",  # onex-topic-allow: OMN-20867
        )
    )
    assert _topic_set("pr_state") == (
        "onex.evt.omnimarket.pr-state-observed.v1",  # onex-topic-allow: OMN-20867
    )


def test_contract_names_the_ci_run_failed_topic_set() -> None:
    """Red triage on a mirror lane needs CI-failed events, and only those."""
    assert _topic_set("ci_run_failed") == (
        "onex.evt.omnimarket.ci-run-failed.v1",  # onex-topic-allow: OMN-20867
    )


def test_loader_selects_pr_state_and_ci_run_failed_together(tmp_path: Path) -> None:
    config = _load(
        _mirror_only_overlay(
            tmp_path, topic_sets=["pr_state", "ci_run_failed"], mirror_lanes=["lane-b"]
        ),
        tmp_path,
    )

    lane_mirror = config.forwarder.lane_mirror
    assert lane_mirror is not None
    assert set(lane_mirror.topics) == set(_topic_set("pr_state")) | set(
        _topic_set("ci_run_failed")
    )
    assert set(lane_mirror.topics).isdisjoint(_topic_set("lab_capacity"))


def test_loader_resolves_a_mirror_only_overlay_without_any_cloud_input(
    tmp_path: Path,
) -> None:
    config = _load(
        _mirror_only_overlay(tmp_path, topic_sets=["lab_capacity"]), tmp_path
    )

    assert config.forwarder.cloud_bus is None
    assert config.forwarder.mirror_topics is None
    assert config.forwarder.canary is None
    assert config.local_bus is None
    assert config.cloud_bus is None
    lane_mirror = config.forwarder.lane_mirror
    assert lane_mirror is not None
    assert lane_mirror.source_lane == _SOURCE_LANE
    assert lane_mirror.mirror_lanes == _MIRROR_LANES
    assert lane_mirror.declared_lanes == (_SOURCE_LANE, *_MIRROR_LANES)
    assert lane_mirror.topics == _topic_set("lab_capacity")


def test_loader_selects_only_the_named_topic_set(tmp_path: Path) -> None:
    config = _load(
        _mirror_only_overlay(
            tmp_path, topic_sets=["pr_state"], mirror_lanes=["lane-b"]
        ),
        tmp_path,
    )

    lane_mirror = config.forwarder.lane_mirror
    assert lane_mirror is not None
    assert lane_mirror.topics == _topic_set("pr_state")
    assert set(lane_mirror.topics).isdisjoint(_topic_set("lab_capacity"))


def test_a_lane_the_overlay_does_not_declare_is_refused(tmp_path: Path) -> None:
    path = _mirror_only_overlay(
        tmp_path,
        topic_sets=["lab_capacity"],
        declared_lanes=(_SOURCE_LANE, _MIRROR_LANES[0]),
    )

    with pytest.raises(ValueError, match="declared_lanes"):
        _load(path, tmp_path)


def test_without_overlay_lanes_the_contracts_default_set_applies(
    tmp_path: Path,
) -> None:
    path = _mirror_only_overlay(
        tmp_path, topic_sets=["lab_capacity"], declared_lanes=None
    )

    with pytest.raises(ValueError, match="declared_lanes"):
        _load(path, tmp_path)


@pytest.mark.parametrize("bad", [[], "lane-a", [1], ["lane-a", 2]])
def test_malformed_overlay_declared_lanes_are_refused(
    tmp_path: Path, bad: object
) -> None:
    path = _mirror_only_overlay(tmp_path, topic_sets=["lab_capacity"])
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["forwarder"]["lane_mirror_binding"]["declared_lanes"] = bad
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="declared_lanes must be a non-empty list"):
        _load(path, tmp_path)


def test_loader_refuses_a_config_with_neither_a_cloud_leg_nor_a_mirror_leg(
    tmp_path: Path,
) -> None:
    path = _mirror_only_overlay(tmp_path, topic_sets=["lab_capacity"])
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("lane_mirror_set", "lane_mirror_binding"):
        raw["forwarder"].pop(key)
    raw.pop("lane_mirror_source_bus")
    raw.pop("lane_mirror_buses")
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="at least one leg"):
        _load(path, tmp_path)


def test_the_process_parser_takes_a_mirror_only_config_without_a_broker_ref_map(
    tmp_path: Path,
) -> None:
    args = gateway_forwarder._build_parser().parse_args(
        ["--config", str(tmp_path / "mirror-only.yaml")]
    )

    assert args.broker_ref_map is None


def test_a_cloud_leg_config_refuses_to_load_without_a_broker_ref_map(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cloud.yaml"
    path.write_text(
        yaml.safe_dump({"forwarder": {"cloud_bus": {"cloud_broker_ref": "x"}}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="needs --broker-ref-map"):
        _load(path, tmp_path)


def test_loader_refuses_cloud_selectors_on_a_deployment_with_no_cloud_bus(
    tmp_path: Path,
) -> None:
    path = _mirror_only_overlay(tmp_path, topic_sets=["lab_capacity"])
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["forwarder"]["mirror_topic_set"] = "node_bus_forwarder_effect"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="mirror_topic_set"):
        _load(path, tmp_path)


class _RecordingTransport:
    """Stands in for ``KafkaTransport``; the registry is class-level per test."""

    instances: list[_RecordingTransport] = []
    pending: dict[str, list[ModelTransportMessage]] = {}
    source_drained: asyncio.Event

    def __init__(
        self,
        *,
        config: ModelKafkaEventBusConfig,
        group: str,
        topics: Sequence[str],
        auto_offset_reset: str,
        refused_topic_retry_seconds: float | None = None,
        consume_only: bool = False,
    ) -> None:
        self.config = config
        self.group = group
        self.topics = tuple(topics)
        self.consume_only = consume_only
        self.started = False
        self.closed = False
        self.sent: list[tuple[str, bytes | None, bytes]] = []
        self.committed: list[ModelTransportMessage] = []
        self.nacked: list[ModelTransportMessage] = []
        self.offset = 0
        _RecordingTransport.instances.append(self)

    async def start(self) -> None:
        self.started = True

    async def close(self) -> None:
        self.closed = True

    async def poll(
        self, *, max_messages: int, timeout_ms: int
    ) -> Sequence[ModelTransportMessage]:
        batch = _RecordingTransport.pending.pop(self.group, [])
        if not batch:
            _RecordingTransport.source_drained.set()
            await asyncio.sleep(0.01)
        return batch

    async def commit(self, message: ModelTransportMessage) -> None:
        self.committed.append(message)

    async def nack(self, message: ModelTransportMessage) -> None:
        self.nacked.append(message)

    async def send_with_coordinate(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: Mapping[str, bytes],
    ) -> tuple[str, int, int]:
        self.sent.append((topic, key, value))
        self.offset += 1
        return (topic, 0, self.offset)


class _MemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.started = False
        self.closed = False
        self.seen: set[UUID] = set()

    async def start(self) -> None:
        self.started = True

    async def close(self) -> None:
        self.closed = True

    async def check_and_record(self, *args: object, **kwargs: object) -> bool:
        return True

    async def is_processed(self, message_id: UUID, domain: str | None = None) -> bool:
        return message_id in self.seen

    async def mark_processed(
        self, message_id: UUID, *args: object, **kwargs: Any
    ) -> None:
        self.seen.add(message_id)

    async def cleanup_expired(self, ttl_seconds: int) -> int:
        return 0


def _record(topic: str, *, offset: int) -> ModelTransportMessage:
    message_id = str(uuid4()).encode("utf-8")
    return ModelTransportMessage(
        topic=topic,
        partition=0,
        offset=offset,
        key=None,
        value=b'{"payload": {}}',
        headers={
            "message_id": message_id,
            "content_type": b"application/json",
        },
        ack_token=f"{topic}:0:{offset}",
    )


@pytest.mark.asyncio
async def test_mirror_only_process_republishes_the_selected_set_and_nothing_else(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    capacity_topic = _topic_set("lab_capacity")[0]
    outside_topic = _topic_set("pr_state")[0]
    config = _load(
        _mirror_only_overlay(tmp_path, topic_sets=["lab_capacity"]), tmp_path
    )
    source_group = f"tenant-{_TENANT_SLUG}-gateway-lane-mirror-source"
    inside = _record(capacity_topic, offset=0)
    outside = _record(outside_topic, offset=1)

    _RecordingTransport.instances = []
    _RecordingTransport.pending = {source_group: [inside, outside]}
    _RecordingTransport.source_drained = asyncio.Event()
    monkeypatch.setattr(gateway_forwarder, "KafkaTransport", _RecordingTransport)
    monkeypatch.setattr(gateway_forwarder, "StoreIdempotencySqlite", _MemoryStore)

    async def _no_secret(_reference: str) -> str | None:
        raise AssertionError("a mirror-only process resolves no secret reference")

    shutdown_event = asyncio.Event()
    ready_path = tmp_path / "ready"
    task = asyncio.create_task(
        gateway_forwarder.run_gateway_forwarder(
            config,
            shutdown_event=shutdown_event,
            resolve_secret=_no_secret,
            ready_path=ready_path,
        )
    )
    try:
        await asyncio.wait_for(_RecordingTransport.source_drained.wait(), timeout=10)
        assert ready_path.exists()
    finally:
        shutdown_event.set()
        await asyncio.wait_for(task, timeout=10)

    by_group = {t.group: t for t in _RecordingTransport.instances}
    # Source plus one producer per mirror lane. No local, no cloud transport.
    assert set(by_group) == {
        source_group,
        *(
            f"tenant-{_TENANT_SLUG}-gateway-lane-mirror-{lane}"
            for lane in _MIRROR_LANES
        ),
    }
    source = by_group[source_group]
    assert source.topics == (capacity_topic,)
    assert source.consume_only
    for lane in _MIRROR_LANES:
        producer = by_group[f"tenant-{_TENANT_SLUG}-gateway-lane-mirror-{lane}"]
        assert not producer.consume_only
        assert [topic for topic, _key, _value in producer.sent] == [capacity_topic]
    # The record outside the selected set is committed past, never republished.
    assert source.committed == [inside, outside]
    assert source.nacked == []
    assert all(t.started and t.closed for t in _RecordingTransport.instances)
    assert not ready_path.exists()


@pytest.mark.asyncio
async def test_mirror_only_process_fails_loudly_when_the_mirror_loop_dies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _load(
        _mirror_only_overlay(tmp_path, topic_sets=["lab_capacity"]), tmp_path
    )

    class _BrokenSource(_RecordingTransport):
        async def poll(
            self, *, max_messages: int, timeout_ms: int
        ) -> Sequence[ModelTransportMessage]:
            raise RuntimeError("source broker unreachable")

    _RecordingTransport.instances = []
    _RecordingTransport.pending = {}
    _RecordingTransport.source_drained = asyncio.Event()
    monkeypatch.setattr(gateway_forwarder, "KafkaTransport", _BrokenSource)
    monkeypatch.setattr(gateway_forwarder, "StoreIdempotencySqlite", _MemoryStore)

    async def _no_secret(_reference: str) -> str | None:
        return None

    shutdown_event = asyncio.Event()
    try:
        with pytest.raises(RuntimeError, match="source broker unreachable"):
            await asyncio.wait_for(
                gateway_forwarder.run_gateway_forwarder(
                    config,
                    shutdown_event=shutdown_event,
                    resolve_secret=_no_secret,
                ),
                timeout=10,
            )
    finally:
        shutdown_event.set()
        with contextlib.suppress(Exception):
            await asyncio.sleep(0)
    assert all(t.closed for t in _RecordingTransport.instances)
