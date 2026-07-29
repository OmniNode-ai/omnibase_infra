# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from uuid import UUID

import pytest
import yaml

from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
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
            mirror_topics=ModelGatewayMirrorTopics(
                inbound=("onex.cmd.omnibase-infra.delegation-request.v1",),
                outbound=(
                    "onex.evt.omnibase-infra.delegation-completed.v1",
                    "onex.evt.omnibase-infra.gateway-heartbeat.v1",
                ),
            ),
        ),
        local_bus=ModelKafkaEventBusConfig(
            bootstrap_servers="redpanda:9092",
            environment="gateway-local",
        ),
        cloud_bus=ModelKafkaEventBusConfig(
            bootstrap_servers="b-1.example.kafka.amazonaws.com:9098",
            environment="gateway-cloud",
            security_protocol="SASL_SSL",
            sasl_mechanism="AWS_MSK_IAM",
            msk_region="us-east-1",
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


def test_runtime_config_loader_round_trips_yaml(tmp_path: Path) -> None:
    config = _runtime_config()
    path = tmp_path / "gateway.yaml"
    contract_path = tmp_path / "contract.yaml"
    dumped = config.model_dump(mode="json")
    dumped["local_bus"].pop("acks_aiokafka")
    dumped["cloud_bus"].pop("acks_aiokafka")
    mirror_topics = dumped["forwarder"].pop("mirror_topics")
    dumped["forwarder"]["mirror_topic_set"] = "node_bus_forwarder_effect"
    path.write_text(
        yaml.safe_dump(dumped),
        encoding="utf-8",
    )
    contract_path.write_text(
        yaml.safe_dump(
            {
                "contract_name": "node_bus_forwarder_effect",
                "config": {"gateway_forwarder": {"mirror_topics": mirror_topics}},
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


def test_staging_canary_resolves_topics_from_node_contract() -> None:
    repo_root = Path(__file__).parents[3]

    loaded = gateway_forwarder.load_gateway_forwarder_runtime_config(
        repo_root / "docker/gateway/beta-gateway-canary.yaml"
    )

    assert len(loaded.forwarder.mirror_topics.inbound) == 3
    assert len(loaded.forwarder.mirror_topics.outbound) == 6
    assert loaded.local_bus.bootstrap_servers == "redpanda:9092"


@pytest.mark.asyncio
async def test_process_starts_both_legs_and_cleans_readiness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    instances: list[_FakeBus] = []

    class _Factory(_FakeBus):
        def __init__(self, *, config: ModelKafkaEventBusConfig) -> None:
            super().__init__(config=config)
            instances.append(self)

    monkeypatch.setattr(gateway_forwarder, "EventBusKafka", _Factory)
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


class _FakeBus:
    def __init__(self, *, config: ModelKafkaEventBusConfig) -> None:
        self.config = config
        self.started = False
        self.closed = False
        self.subscriptions: dict[str, Callable[[object], Awaitable[None]]] = {}

    async def start(self) -> None:
        self.started = True

    async def close(self) -> None:
        self.closed = True

    async def subscribe(
        self,
        topic: str,
        node_identity: object | None = None,
        on_message: Callable[[object], Awaitable[None]] | None = None,
        *,
        group_id: str | None = None,
        **kwargs: object,
    ) -> Callable[[], Awaitable[None]]:
        assert on_message is not None
        assert group_id is not None
        self.subscriptions[topic] = on_message

        async def _unsubscribe() -> None:
            self.subscriptions.pop(topic, None)

        return _unsubscribe

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: object | None = None,
    ) -> None:
        raise AssertionError("no message should be published in lifecycle test")
