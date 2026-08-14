# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path
from uuid import UUID

import pytest
import yaml

from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayCanaryConfig,
    ModelGatewayCloudBusConfig,
    ModelGatewayForwarderConfig,
    ModelGatewayForwarderRuntimeConfig,
    ModelGatewayMirrorTopics,
    ModelGatewayTenantIdentity,
)
from omnibase_infra.runtime import gateway_forwarder


def _runtime_config() -> ModelGatewayForwarderRuntimeConfig:
    return ModelGatewayForwarderRuntimeConfig(
        forwarder=ModelGatewayForwarderConfig(
            tenant_identity=ModelGatewayTenantIdentity(
                tenant_id=UUID("79afa726-3852-464f-b7a4-d4b8b9c75ee7"),
                tenant_slug="beta-gateway-canary-79afa7263852",
                principal_id="t-79afa7263852464fb7a4d4b8b9c75ee7",
            ),
            cloud_bus=ModelGatewayCloudBusConfig(
                broker_provider_id=UUID("22222222-2222-2222-2222-222222222222"),
                cloud_broker_ref="gateway.cloud.kafka.broker",
                cloud_auth_ref="gateway.cloud.kafka.msk_iam",
                acl_provisioner_ref="gateway.cloud.kafka.authorization",
                msk_region_ref="gateway.cloud.kafka.msk_region",
                sasl_mechanism="AWS_MSK_IAM",
            ),
            local_transport_flavor="containerized",
            dedupe_store_path=Path.cwd() / "gateway-test.sqlite3",
            mirror_topics=ModelGatewayMirrorTopics(
                inbound=("onex.cmd.omnibase-infra.delegation-request.v1",),
                outbound=(
                    "onex.evt.omnibase-infra.delegation-completed.v1",
                    "onex.evt.omnibase-infra.gateway-heartbeat.v1",
                ),
            ),
            canary=ModelGatewayCanaryConfig(
                topic="onex.evt.omnibase-infra.gateway-canary.v1",
                cadence_seconds=30,
                produce_deadline_seconds=8,
                readback_deadline_seconds=12,
            ),
        ),
        local_bus=ModelKafkaEventBusConfig(
            bootstrap_servers="redpanda:9092",
            environment="gateway-local",
            enable_auto_commit=False,
            auto_offset_reset="earliest",
        ),
        cloud_bus=ModelKafkaEventBusConfig(
            bootstrap_servers="b-1.example.kafka.amazonaws.com:9098",
            environment="gateway-cloud",
            security_protocol="SASL_SSL",
            sasl_mechanism="AWS_MSK_IAM",
            msk_region="us-east-1",
            enable_auto_commit=False,
            auto_offset_reset="earliest",
        ),
    )


def test_runtime_config_rejects_one_broker_for_both_legs() -> None:
    config = _runtime_config()
    with pytest.raises(ValueError, match="must be distinct"):
        ModelGatewayForwarderRuntimeConfig(
            forwarder=config.forwarder,
            local_bus=config.local_bus,
            cloud_bus=config.local_bus,
        )


def test_runtime_config_requires_outbound_heartbeat() -> None:
    config = _runtime_config()
    forwarder = config.forwarder.model_copy(
        update={
            "mirror_topics": ModelGatewayMirrorTopics(
                inbound=config.forwarder.mirror_topics.inbound,
                outbound=("onex.evt.omnibase-infra.delegation-completed.v1",),
            )
        }
    )
    with pytest.raises(ValueError, match="requires an outbound heartbeat"):
        ModelGatewayForwarderRuntimeConfig(
            forwarder=forwarder,
            local_bus=config.local_bus,
            cloud_bus=config.cloud_bus,
        )


def test_runtime_config_rejects_source_auto_commit() -> None:
    config = _runtime_config()
    auto_commit_local = config.local_bus.model_copy(update={"enable_auto_commit": True})

    with pytest.raises(ValueError, match="enable_auto_commit=false"):
        ModelGatewayForwarderRuntimeConfig(
            forwarder=config.forwarder,
            local_bus=auto_commit_local,
            cloud_bus=config.cloud_bus,
        )


def test_runtime_config_rejects_latest_auto_offset_reset_on_local_leg() -> None:
    """OMN-15781: a `latest` local leg drops any backlog produced during an outage.

    ``enable_auto_commit=false`` means offsets commit only after durable
    destination delivery -- but that guarantee is moot if the consumer starts
    reading from ``latest`` on rejoin, since the backlog produced while the
    consumer group was unjoined is never in the read window at all.
    """
    config = _runtime_config()
    latest_local = config.local_bus.model_copy(update={"auto_offset_reset": "latest"})

    with pytest.raises(ValueError, match="auto_offset_reset=earliest"):
        ModelGatewayForwarderRuntimeConfig(
            forwarder=config.forwarder,
            local_bus=latest_local,
            cloud_bus=config.cloud_bus,
        )


def test_runtime_config_rejects_latest_auto_offset_reset_on_cloud_leg() -> None:
    """OMN-15781: same hazard on the cloud leg -- both legs are pull-based."""
    config = _runtime_config()
    latest_cloud = config.cloud_bus.model_copy(update={"auto_offset_reset": "latest"})

    with pytest.raises(ValueError, match="auto_offset_reset=earliest"):
        ModelGatewayForwarderRuntimeConfig(
            forwarder=config.forwarder,
            local_bus=config.local_bus,
            cloud_bus=latest_cloud,
        )


def test_runtime_config_accepts_earliest_auto_offset_reset() -> None:
    """Sanity: the correct value is not itself rejected."""
    config = _runtime_config()
    assert config.local_bus.auto_offset_reset == "earliest"
    assert config.cloud_bus.auto_offset_reset == "earliest"


def test_runtime_config_loader_round_trips_yaml(tmp_path: Path) -> None:
    config = _runtime_config()
    path = tmp_path / "gateway.yaml"
    contract_path = tmp_path / "contract.yaml"
    dumped = config.model_dump(mode="json")
    dumped["local_bus"].pop("acks_aiokafka")
    dumped["cloud_bus"].pop("acks_aiokafka")
    mirror_topics = dumped["forwarder"].pop("mirror_topics")
    canary = dumped["forwarder"].pop("canary")
    dumped["forwarder"]["mirror_topic_set"] = "node_bus_forwarder_effect"
    dumped["forwarder"]["canary_topic_set"] = "node_bus_forwarder_effect"
    path.write_text(
        yaml.safe_dump(dumped),
        encoding="utf-8",
    )
    contract_path.write_text(
        yaml.safe_dump(
            {
                "contract_name": "node_bus_forwarder_effect",
                "config": {
                    "gateway_forwarder": {
                        "mirror_topics": mirror_topics,
                        "canary": canary,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = gateway_forwarder.load_gateway_forwarder_runtime_config(
        path,
        contract_path=contract_path,
    )

    assert loaded == config


def test_runtime_config_rejects_inline_mirror_topic_literals(tmp_path: Path) -> None:
    path = tmp_path / "gateway.yaml"
    dumped = _runtime_config().model_dump(mode="json")
    dumped["local_bus"].pop("acks_aiokafka")
    dumped["cloud_bus"].pop("acks_aiokafka")
    path.write_text(yaml.safe_dump(dumped), encoding="utf-8")

    with pytest.raises(ValueError, match="must name mirror_topic_set"):
        gateway_forwarder.load_gateway_forwarder_runtime_config(path)


def test_runtime_config_rejects_inline_canary_literals(tmp_path: Path) -> None:
    config = _runtime_config()
    path = tmp_path / "gateway.yaml"
    contract_path = tmp_path / "contract.yaml"
    dumped = config.model_dump(mode="json")
    dumped["local_bus"].pop("acks_aiokafka")
    dumped["cloud_bus"].pop("acks_aiokafka")
    mirror_topics = dumped["forwarder"].pop("mirror_topics")
    dumped["forwarder"]["mirror_topic_set"] = "node_bus_forwarder_effect"
    # canary is left inline (not popped) -- the redeclare that must be rejected.
    path.write_text(yaml.safe_dump(dumped), encoding="utf-8")
    contract_path.write_text(
        yaml.safe_dump(
            {
                "contract_name": "node_bus_forwarder_effect",
                "config": {"gateway_forwarder": {"mirror_topics": mirror_topics}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must name canary_topic_set"):
        gateway_forwarder.load_gateway_forwarder_runtime_config(
            path, contract_path=contract_path
        )


def test_staging_canary_resolves_topics_from_node_contract() -> None:
    repo_root = Path(__file__).parents[3]

    loaded = gateway_forwarder.load_gateway_forwarder_runtime_config(
        repo_root / "docker/gateway/beta-gateway-canary.yaml"
    )

    assert len(loaded.forwarder.mirror_topics.inbound) == 3
    assert len(loaded.forwarder.mirror_topics.outbound) == 6
    assert loaded.local_bus.bootstrap_servers == "redpanda:9092"
    # OMN-15781: deployed config must resolve to "earliest" on both legs, not
    # the model default -- a "latest" resolved leg here silently drops any
    # backlog produced while the consumer group was unjoined (crash/rejoin
    # window), and this is the exact deployed config the production process
    # loads. The runtime-config validator also fails closed on "latest" (see
    # test_runtime_config_rejects_latest_auto_offset_reset_on_*_leg above),
    # so a regression here would fail at load time, not just this assertion.
    assert loaded.local_bus.auto_offset_reset == "earliest"
    assert loaded.cloud_bus.auto_offset_reset == "earliest"
    assert loaded.forwarder.canary.topic == "onex.evt.omnibase-infra.gateway-canary.v1"
    assert loaded.forwarder.canary.cadence_seconds == 30


@pytest.mark.asyncio
async def test_process_starts_both_legs_and_cleans_readiness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    instances: list[_FakeTransport] = []
    stores: list[_FakeStore] = []

    class _Factory(_FakeTransport):
        def __init__(
            self,
            *,
            config: ModelKafkaEventBusConfig,
            group: str,
            topics: Sequence[str],
            auto_offset_reset: str,
        ) -> None:
            super().__init__(config=config, group=group, topics=topics)
            instances.append(self)

    class _StoreFactory(_FakeStore):
        def __init__(self, path: Path) -> None:
            super().__init__(path)
            stores.append(self)

    monkeypatch.setattr(gateway_forwarder, "KafkaTransport", _Factory)
    monkeypatch.setattr(gateway_forwarder, "StoreIdempotencySqlite", _StoreFactory)
    shutdown_event = asyncio.Event()
    shutdown_event.set()
    ready_path = tmp_path / "ready"

    await gateway_forwarder.run_gateway_forwarder(
        _runtime_config(),
        shutdown_event=shutdown_event,
        ready_path=ready_path,
    )

    assert len(instances) == 2
    assert all(instance.started for instance in instances)
    assert all(instance.closed for instance in instances)
    assert len(stores) == 1
    assert stores[0].started is True
    assert stores[0].closed is True
    assert not ready_path.exists()


@pytest.mark.asyncio
async def test_heartbeat_loop_emits_immediately_and_stops() -> None:
    shutdown_event = asyncio.Event()

    class _Forwarder:
        calls = 0

        async def publish_heartbeat(self) -> None:
            self.calls += 1
            shutdown_event.set()

    forwarder = _Forwarder()
    await gateway_forwarder._run_heartbeat_loop(
        forwarder,  # type: ignore[arg-type]
        _runtime_config(),
        shutdown_event,
    )

    assert forwarder.calls == 1


class _FakeSupervisedDelivery:
    """Fakes ``NodeGatewayDelivery`` for ``_supervise_gateway_delivery`` tests.

    ``wait()`` raises each queued exception in order, then blocks forever
    (an unset ``asyncio.Event``) to represent a healthy running loop.
    """

    def __init__(self, failures: Sequence[Exception]) -> None:
        self._failures = list(failures)
        self._never = asyncio.Event()
        self.start_calls = 0
        self.stop_calls = 0

    async def wait(self) -> None:
        if self._failures:
            raise self._failures.pop(0)
        await self._never.wait()

    async def start(self) -> None:
        self.start_calls += 1

    async def stop(self) -> None:
        self.stop_calls += 1


class _FakeStatusForwarder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, str]] = []

    async def publish_status(
        self,
        status: str,
        *,
        consecutive_failures: int = 0,
        detail: str = "",
    ) -> None:
        self.calls.append((status, consecutive_failures, detail))


async def _never_ending_heartbeat(shutdown_event: asyncio.Event) -> None:
    await shutdown_event.wait()


@pytest.mark.asyncio
async def test_supervise_gateway_delivery_retries_without_raising() -> None:
    """OMN-15742: a cloud-leg failure retries with backoff, never propagates."""
    shutdown_event = asyncio.Event()
    delivery = _FakeSupervisedDelivery([ValueError("boom-1"), ValueError("boom-2")])
    forwarder = _FakeStatusForwarder()
    heartbeat_task = asyncio.create_task(_never_ending_heartbeat(shutdown_event))
    config = _runtime_config().forwarder.model_copy(
        update={
            "reconnect_backoff_initial_seconds": 0.01,
            "reconnect_backoff_max_seconds": 0.01,
            "reconnect_backoff_jitter_seconds": 0.0,
            "degraded_after_seconds": 1000,
            "heartbeat_interval_seconds": 1,
        }
    )

    async def _stop_after_recovery() -> None:
        # Two failed attempts + one full recovery-confirm window
        # (heartbeat_interval_seconds) is enough time for the loop to have
        # restarted twice and reset its failure window; extra margin for
        # scheduling jitter on a loaded CI host.
        await asyncio.sleep(1.8)
        shutdown_event.set()

    stopper = asyncio.create_task(_stop_after_recovery())
    try:
        await gateway_forwarder._supervise_gateway_delivery(
            forwarder=forwarder,  # type: ignore[arg-type]
            delivery=delivery,  # type: ignore[arg-type]
            heartbeat_task=heartbeat_task,
            shutdown_event=shutdown_event,
            config=config,
        )
    finally:
        heartbeat_task.cancel()
        await asyncio.gather(heartbeat_task, return_exceptions=True)
        await stopper

    assert delivery.start_calls == 2
    assert delivery.stop_calls == 2
    assert forwarder.calls == []  # degraded_after_seconds never crossed


@pytest.mark.asyncio
async def test_supervise_gateway_delivery_emits_degraded_then_recovers() -> None:
    """OMN-15742: sustained failure crosses the contract degradation window."""
    shutdown_event = asyncio.Event()
    delivery = _FakeSupervisedDelivery(
        [ValueError("boom-1"), ValueError("boom-2"), ValueError("boom-3")]
    )
    forwarder = _FakeStatusForwarder()
    heartbeat_task = asyncio.create_task(_never_ending_heartbeat(shutdown_event))
    config = _runtime_config().forwarder.model_copy(
        update={
            "reconnect_backoff_initial_seconds": 0.5,
            "reconnect_backoff_max_seconds": 0.5,
            "reconnect_backoff_jitter_seconds": 0.0,
            "degraded_after_seconds": 1,
            "heartbeat_interval_seconds": 1,
        }
    )

    async def _stop_when_recovered() -> None:
        # 3 failures spaced ~0.5s apart cross the 1s degraded window;
        # +1s recovery-confirm window after the loop stabilizes; extra
        # margin for scheduling jitter on a loaded CI host.
        await asyncio.sleep(3.5)
        shutdown_event.set()

    stopper = asyncio.create_task(_stop_when_recovered())
    try:
        await gateway_forwarder._supervise_gateway_delivery(
            forwarder=forwarder,  # type: ignore[arg-type]
            delivery=delivery,  # type: ignore[arg-type]
            heartbeat_task=heartbeat_task,
            shutdown_event=shutdown_event,
            config=config,
        )
    finally:
        heartbeat_task.cancel()
        await asyncio.gather(heartbeat_task, return_exceptions=True)
        await stopper

    assert delivery.start_calls == 3
    statuses = [call[0] for call in forwarder.calls]
    assert statuses == ["degraded", "active"]
    degraded_call = forwarder.calls[0]
    assert degraded_call[1] == 3  # consecutive_failures at emission
    assert "boom-3" in degraded_call[2]


class _FakeTransport:
    def __init__(
        self,
        *,
        config: ModelKafkaEventBusConfig,
        group: str,
        topics: Sequence[str],
    ) -> None:
        self.config = config
        self.group = group
        self.topics = tuple(topics)
        self.started = False
        self.closed = False

    async def start(self) -> None:
        self.started = True

    async def close(self) -> None:
        self.closed = True

    async def poll(self, *, max_messages: int, timeout_ms: int) -> Sequence[object]:
        await asyncio.sleep(0)
        return []

    async def commit(self, message: object) -> None:
        raise AssertionError("no source message should be committed")

    async def nack(self, message: object) -> None:
        raise AssertionError("no source message should be nacked")

    async def send(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: Mapping[str, bytes],
    ) -> None:
        raise AssertionError("no message should be published in lifecycle test")


class _FakeStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.started = False
        self.closed = False

    async def start(self) -> None:
        self.started = True

    async def close(self) -> None:
        self.closed = True

    async def cleanup_expired(self, ttl_seconds: int) -> int:
        return 0

    async def is_processed(self, message_id: UUID, domain: str | None = None) -> bool:
        return False

    async def mark_processed(self, *args: object, **kwargs: object) -> None:
        pass

    async def check_and_record(self, *args: object, **kwargs: object) -> bool:
        return True
