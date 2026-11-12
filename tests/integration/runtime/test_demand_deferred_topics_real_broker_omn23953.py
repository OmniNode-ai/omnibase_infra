# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-23953: a demand-deferred contract's topics are created when it starts.

Real Redpanda broker (testcontainers); the provisioner, the bus and the demand
source are the production classes. The seam under test is the one between the
boot provisioning pass (OMN-23951) and the demand split (OMN-18087): the broker
is read, not the code, for which topics exist.

Golden chain: at boot the in-use contract's and the publish-only contract's
publish topics are created and the deferred contract's are not; a first record
on the deferred contract's subscribe topic starts it and its publish topic
appears. Error chain: a replication-policy refusal on the publish-only topic
still ends the boot, and the deferred contract's publish topic stays absent.
"""

from __future__ import annotations

import asyncio
import socket
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterator
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import pytest
from aiokafka import AIOKafkaProducer
from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from aiokafka.errors import KafkaError
from testcontainers.core.container import DockerContainer
from testcontainers.core.docker_client import DockerClient

from omnibase_infra.errors import TopicReplicationPolicyError
from omnibase_infra.event_bus.enum_contract_attach_exclusion_reason import (
    EnumContractAttachExclusionReason,
)
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.model_contract_attach_exclusion import (
    ModelContractAttachExclusion,
)
from omnibase_infra.event_bus.model_topic_readiness_config import (
    ModelTopicReadinessConfig,
)
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.event_bus.service_topic_manager import TopicProvisioner
from omnibase_infra.handlers.handler_kafka_topic_demand_source import (
    HandlerKafkaTopicDemandSource,
)
from omnibase_infra.models.projection.model_snapshot_topic_config import (
    ModelSnapshotTopicConfig,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    subscribe_wired_contract_topics,
    wire_from_manifest,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from omnibase_infra.topics.model_topic_spec import ModelTopicSpec
from omnibase_infra.topics.topic_namespace import apply_topic_namespace

pytestmark = [pytest.mark.integration, pytest.mark.kafka]

BROKER_IMAGE = "redpandadata/redpanda:v24.2.7"
BROKER_READY_SECONDS = 90.0
DEMAND_START_SECONDS = 60.0
BUS_ENVIRONMENT = "local"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


async def _wait_for_broker(bootstrap: str) -> None:
    deadline = time.monotonic() + BROKER_READY_SECONDS
    last: Exception | None = None
    while time.monotonic() < deadline:
        admin = AIOKafkaAdminClient(bootstrap_servers=bootstrap)
        try:
            await admin.start()
            await admin.describe_cluster()
            return
        except (KafkaError, OSError) as exc:
            last = exc
            await asyncio.sleep(1.0)
        finally:
            await admin.close()
    raise AssertionError(f"broker at {bootstrap} not ready: {last!r}")


@pytest.fixture(scope="module")
def broker_bootstrap() -> Iterator[str]:
    """A throwaway Redpanda broker on a free port, advertised at that port."""
    port = _free_port()
    host = DockerClient().host()
    container = (
        DockerContainer(BROKER_IMAGE)
        .with_bind_ports(port, port)
        .with_command(
            "redpanda start "
            f"--kafka-addr PLAINTEXT://0.0.0.0:{port} "
            f"--advertise-kafka-addr PLAINTEXT://{host}:{port} "
            "--mode dev-container --smp 1 --memory 1G --reserve-memory 0M "
            "--reactor-backend=epoll --default-log-level=warn"
        )
    )
    with container:
        bootstrap = f"{host}:{port}"
        asyncio.run(_wait_for_broker(bootstrap))
        yield bootstrap


@pytest.fixture
async def bus(broker_bootstrap: str) -> AsyncGenerator[EventBusKafka, None]:
    started = EventBusKafka(
        config=ModelKafkaEventBusConfig(
            bootstrap_servers=broker_bootstrap,
            environment=BUS_ENVIRONMENT,
            timeout_seconds=30,
            max_retry_attempts=0,
        )
    )
    await started.start()
    yield started
    await started.close()


class _Names:
    """Topic names of one scenario; the suffix keeps scenarios apart on one broker."""

    def __init__(self) -> None:
        suffix = uuid.uuid4().hex[:8]
        self.in_use_subscribe = f"onex.evt.platform.hotload-rb-in-use-{suffix}.v1"
        self.in_use_publish = f"onex.evt.platform.hotload-rb-in-use-out-{suffix}.v1"
        self.deferred_subscribe = f"onex.cmd.platform.hotload-rb-deferred-{suffix}.v1"
        self.deferred_publish = f"onex.evt.platform.hotload-rb-deferred-out-{suffix}.v1"
        self.publish_only = f"onex.evt.omnimarket.hotload-rb-floor-{suffix}.v1"


async def _broker_topics(bootstrap: str) -> set[str]:
    admin = AIOKafkaAdminClient(bootstrap_servers=bootstrap)
    await admin.start()
    try:
        return set(await admin.list_topics())
    finally:
        await admin.close()


async def _create_topic(bootstrap: str, logical: str) -> None:
    admin = AIOKafkaAdminClient(bootstrap_servers=bootstrap)
    await admin.start()
    try:
        await admin.create_topics(
            [
                NewTopic(
                    name=apply_topic_namespace(logical),
                    num_partitions=1,
                    replication_factor=1,
                )
            ]
        )
    finally:
        await admin.close()


async def _write_record(bootstrap: str, logical: str) -> None:
    producer = AIOKafkaProducer(bootstrap_servers=bootstrap)
    await producer.start()
    try:
        await producer.send_and_wait(
            apply_topic_namespace(logical), key=b"k", value=b"{}"
        )
    finally:
        await producer.stop()


class _FakeHandler:
    async def handle(self, envelope: object) -> None:
        return None


def _contract(
    name: str, subscribe: tuple[str, ...], publish: tuple[str, ...]
) -> ModelDiscoveredContract:
    routing = (
        ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(name="FakeHandler", module="fake.module"),
                    event_model=None,
                    operation=None,
                ),
            ),
        )
        if subscribe
        else None
    )
    return ModelDiscoveredContract(
        name=name,
        node_type="ORCHESTRATOR_GENERIC" if subscribe else "EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path(f"/fake/{name}/contract.yaml"),
        entry_point_name=name,
        package_name="test-package",
        event_bus=ModelEventBusWiring(
            subscribe_topics=subscribe, publish_topics=publish
        ),
        handler_routing=routing,
    )


class _RefusingProvisioner(TopicProvisioner):
    """The production provisioner with one topic refused by the replication policy."""

    def __init__(
        self, refused: str, bootstrap_servers: str, contracts_root: Path
    ) -> None:
        super().__init__(
            bootstrap_servers=bootstrap_servers, contracts_root=contracts_root
        )
        self._refused = refused

    async def ensure_topic_exists(
        self,
        topic_name: str,
        config: ModelSnapshotTopicConfig | None = None,
        correlation_id: UUID | None = None,
        *,
        spec: ModelTopicSpec | None = None,
    ) -> bool:
        if topic_name == self._refused:
            raise TopicReplicationPolicyError("replication factor 1 refused")
        return await super().ensure_topic_exists(
            topic_name, config, correlation_id, spec=spec
        )


async def _boot(
    names: _Names,
    bus: EventBusKafka,
    provisioner: TopicProvisioner,
    exclusions: list[ModelContractAttachExclusion],
    stops: list[Callable[[], Awaitable[None]]],
) -> None:
    manifest = ModelAutoWiringManifest(
        contracts=(
            _contract(
                "node_in_use", (names.in_use_subscribe,), (names.in_use_publish,)
            ),
            _contract(
                "node_deferred",
                (names.deferred_subscribe,),
                (names.deferred_publish,),
            ),
            _contract("node_publish_only", (), (names.publish_only,)),
        )
    )
    engine = MessageDispatchEngine()
    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=_FakeHandler,
    ):
        report = await wire_from_manifest(
            manifest,
            engine,
            event_bus=bus,
            environment=BUS_ENVIRONMENT,
            subscribe_immediately=False,
        )
        await subscribe_wired_contract_topics(
            manifest=manifest,
            report=report,
            dispatch_engine=engine,
            event_bus=bus,
            environment=BUS_ENVIRONMENT,
            provisioner=provisioner,
            readiness_config=ModelTopicReadinessConfig(
                max_concurrent_contract_attach=1
            ),
            exclusions_out=exclusions,
            demand_source=HandlerKafkaTopicDemandSource(bus.config),
            demand_watch_stops_out=stops,
        )


@pytest.fixture(autouse=True)
def _lane_declares_on_demand(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The lane's overlay row declares on_demand (OMN-21318); a control removes it."""
    root = tmp_path / "overlays"
    (root / "runtime_contract_hotload").mkdir(parents=True)
    (root / "runtime_contract_hotload" / "overlay.yaml").write_text(
        "schema_version: 1\nlanes:\n  dogfood:\n    enabled: true\n"
        "    mode: on_demand\n    allowed_environments: [local]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ONEX_SKILL_OVERLAY_ROOTS", str(root))
    monkeypatch.setenv("ONEX_RUNTIME_LANE", "dogfood")


async def _prepare(bootstrap: str, names: _Names) -> None:
    """In-use: its subscribe topic holds a record. Deferred: its topic is empty."""
    await _create_topic(bootstrap, names.in_use_subscribe)
    await _write_record(bootstrap, names.in_use_subscribe)
    await _create_topic(bootstrap, names.deferred_subscribe)


async def test_deferred_contract_topics_appear_when_it_starts_not_at_boot(
    broker_bootstrap: str, bus: EventBusKafka, tmp_path: Path
) -> None:
    names = _Names()
    await _prepare(broker_bootstrap, names)
    provisioner = TopicProvisioner(
        bootstrap_servers=broker_bootstrap, contracts_root=tmp_path
    )
    exclusions: list[ModelContractAttachExclusion] = []
    stops: list[Callable[[], Awaitable[None]]] = []

    try:
        await _boot(names, bus, provisioner, exclusions, stops)

        at_boot = await _broker_topics(broker_bootstrap)
        assert [
            e.contract_name
            for e in exclusions
            if e.reason is EnumContractAttachExclusionReason.DEMAND_DEFERRED
        ] == ["node_deferred"]
        assert apply_topic_namespace(names.in_use_publish) in at_boot
        assert apply_topic_namespace(names.publish_only) in at_boot
        assert apply_topic_namespace(names.deferred_publish) not in at_boot

        await _write_record(broker_bootstrap, names.deferred_subscribe)
        deadline = time.monotonic() + DEMAND_START_SECONDS
        while time.monotonic() < deadline:
            if apply_topic_namespace(names.deferred_publish) in await _broker_topics(
                broker_bootstrap
            ):
                break
            await asyncio.sleep(1.0)
        else:
            raise AssertionError("the demand start never created the publish topic")
    finally:
        for stop in stops:
            await stop()


async def test_policy_refusal_on_publish_only_topic_still_ends_boot(
    broker_bootstrap: str, bus: EventBusKafka, tmp_path: Path
) -> None:
    names = _Names()
    await _prepare(broker_bootstrap, names)
    provisioner = _RefusingProvisioner(names.publish_only, broker_bootstrap, tmp_path)
    stops: list[Callable[[], Awaitable[None]]] = []

    try:
        with pytest.raises(TopicReplicationPolicyError):
            await _boot(names, bus, provisioner, [], stops)
        assert apply_topic_namespace(names.deferred_publish) not in (
            await _broker_topics(broker_bootstrap)
        )
    finally:
        for stop in stops:
            await stop()


async def test_a_lane_with_no_overlay_row_creates_the_deferred_topic_at_boot(
    broker_bootstrap: str,
    bus: EventBusKafka,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failing control (OMN-21318): the lane's row, not the code, withholds the topic."""
    monkeypatch.delenv("ONEX_RUNTIME_LANE")
    names = _Names()
    await _prepare(broker_bootstrap, names)
    provisioner = TopicProvisioner(
        bootstrap_servers=broker_bootstrap, contracts_root=tmp_path
    )
    stops: list[Callable[[], Awaitable[None]]] = []

    try:
        await _boot(names, bus, provisioner, [], stops)
        at_boot = await _broker_topics(broker_bootstrap)
        assert apply_topic_namespace(names.deferred_publish) in at_boot
        assert apply_topic_namespace(names.in_use_publish) in at_boot
        assert apply_topic_namespace(names.publish_only) in at_boot
    finally:
        for stop in stops:
            await stop()
