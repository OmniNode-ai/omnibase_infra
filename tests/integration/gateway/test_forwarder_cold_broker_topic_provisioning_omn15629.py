# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-15629: the shipped forwarder starts against a cold local broker.

The dev-lane local broker runs with topic auto-creation off, so on a cold or
rebuilt broker the contract-declared local topics do not exist and the first
consumer ``start()`` raised ``UnknownTopicOrPartitionError``: the unit
crash-looped until an operator ran ``rpk topic create`` by hand. These tests
resolve the shipped deployment, ``docker/gateway/beta-gateway-canary.yaml``,
through the real runtime config loader and run the real
``run_gateway_forwarder`` against a stateful stand-in for that broker: it
holds a topic set, admits ``create_topics``, and refuses a consumer start on a
topic it does not hold. Only aiokafka is replaced.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml
from aiokafka.admin import NewTopic
from aiokafka.errors import UnknownTopicOrPartitionError

from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayForwarderRuntimeConfig,
)
from omnibase_infra.runtime import gateway_forwarder

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONTRACT = (
    _REPO_ROOT / "src/omnibase_infra/nodes/node_bus_forwarder_effect/contract.yaml"
)
HEARTBEAT = "onex.evt.omnibase-infra.gateway-heartbeat.v1"


class _ColdBroker:
    """A local broker with auto-creation off that starts with no topics."""

    def __init__(self, bootstrap_servers: str) -> None:
        self.bootstrap_servers = bootstrap_servers
        self.topics: set[str] = set()


class _Admin:
    broker: _ColdBroker

    def __init__(self, *, bootstrap_servers: str, **_auth: object) -> None:
        assert bootstrap_servers == self.broker.bootstrap_servers

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def create_topics(self, new_topics: Sequence[object]) -> None:
        for new_topic in new_topics:
            assert isinstance(new_topic, NewTopic)
            self.broker.topics.add(new_topic.name)


class _Transport:
    """Refuses to start on a local-broker topic the broker does not hold."""

    broker: _ColdBroker

    def __init__(
        self,
        *,
        config: ModelKafkaEventBusConfig,
        group: str,
        topics: Sequence[str],
        **_kwargs: object,
    ) -> None:
        self.config = config
        self.topics = tuple(topics)

    async def start(self) -> None:
        if self.config.bootstrap_servers != self.broker.bootstrap_servers:
            return
        for topic in self.topics:
            if topic not in self.broker.topics:
                raise UnknownTopicOrPartitionError(topic)

    async def close(self) -> None:
        return None

    async def poll(self, *, max_messages: int, timeout_ms: int) -> Sequence[object]:
        await asyncio.sleep(0)
        return []

    async def commit(self, message: object) -> None:
        raise AssertionError("no source message should be committed")


class _Store:
    def __init__(self, path: Path) -> None:
        self.path = path

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def cleanup_expired(self, ttl_seconds: int) -> int:
        return 0

    async def is_processed(self, message_id: object, domain: str | None = None) -> bool:
        return False

    async def mark_processed(self, *args: object, **kwargs: object) -> None:
        return None

    async def check_and_record(self, *args: object, **kwargs: object) -> bool:
        return True


def _shipped_config(tmp_path: Path) -> ModelGatewayForwarderRuntimeConfig:
    credential_map = tmp_path / "lane-credentials.yaml"
    credential_map.write_text(
        "lane.dev.kafka.scram:\n"
        "  username: fixture-principal\n"
        "  password: fixture-not-a-real-secret\n",
        encoding="utf-8",
    )
    return gateway_forwarder.load_gateway_forwarder_runtime_config(
        _REPO_ROOT / "docker/gateway/beta-gateway-canary.yaml",
        broker_ref_map_path=_REPO_ROOT
        / "tests/fixtures/gateway/beta-gateway-canary-broker-ref-map.yaml",
        lane_credential_map_path=credential_map,
    )


def _wire(
    monkeypatch: pytest.MonkeyPatch,
    config: ModelGatewayForwarderRuntimeConfig,
    *,
    provision: bool,
) -> _ColdBroker:
    assert config.local_bus is not None
    broker = _ColdBroker(config.local_bus.bootstrap_servers)
    admin = type("_BoundAdmin", (_Admin,), {"broker": broker})
    real_ensure = gateway_forwarder.ensure_topics_exist

    async def _ensure(bus: ModelKafkaEventBusConfig, topics: Sequence[str]) -> None:
        if provision:
            await real_ensure(bus, topics, admin_factory=admin)

    monkeypatch.setattr(gateway_forwarder, "ensure_topics_exist", _ensure)
    monkeypatch.setattr(
        gateway_forwarder,
        "KafkaTransport",
        type("_BoundTransport", (_Transport,), {"broker": broker}),
    )
    monkeypatch.setattr(gateway_forwarder, "StoreIdempotencySqlite", _Store)
    return broker


async def _run(config: ModelGatewayForwarderRuntimeConfig) -> None:
    shutdown_event = asyncio.Event()
    shutdown_event.set()

    async def _no_secret(_reference: str) -> str | None:
        return None

    await gateway_forwarder.run_gateway_forwarder(
        config,
        shutdown_event=shutdown_event,
        resolve_secret=_no_secret,
    )


@pytest.mark.asyncio
async def test_cold_local_broker_crash_loops_without_provisioning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Control: with provisioning off, the cold broker reproduces the outage."""
    config = _shipped_config(tmp_path)
    _wire(monkeypatch, config, provision=False)

    with pytest.raises(UnknownTopicOrPartitionError):
        await _run(config)


@pytest.mark.asyncio
async def test_shipped_forwarder_starts_on_a_cold_local_broker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _shipped_config(tmp_path)
    broker = _wire(monkeypatch, config, provision=True)

    await _run(config)

    contract = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
    mirror = contract["config"]["gateway_forwarder"]["mirror_topics"]
    assert {*mirror["inbound"], *mirror["outbound"]} <= broker.topics
    assert HEARTBEAT in broker.topics
