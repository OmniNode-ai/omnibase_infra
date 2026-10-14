# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-15629: the forwarder provisions its contract-declared local topics.

A lane broker with ``auto_create_topics_enabled=false`` never creates the
topics ``node_bus_forwarder_effect/contract.yaml`` declares, and the forwarder
(``ONEX_TOPIC_ENFORCEMENT_MODE=reject``) never created them either: the first
consumer ``start()`` died on ``UnknownTopicOrPartitionError`` and the unit
crash-looped on a cold broker until an operator ran ``rpk topic create``.

These tests read the topic list from the shipped contract, never from a literal
here, so a topic the contract gains is covered without editing this file.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml
from aiokafka.admin import NewTopic
from aiokafka.errors import (
    KafkaConnectionError,
    TopicAlreadyExistsError,
    TopicAuthorizationFailedError,
)

from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.runtime import gateway_forwarder

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).parents[3]
_CONTRACT = (
    _REPO_ROOT / "src/omnibase_infra/nodes/node_bus_forwarder_effect/contract.yaml"
)


class _Events:
    """Ordered record of admin creates and transport starts."""

    log: list[str] = []
    created: list[str] = []


class _FakeAdmin:
    def __init__(self, *, bootstrap_servers: str, **_auth: object) -> None:
        self.bootstrap_servers = bootstrap_servers

    async def start(self) -> None:
        _Events.log.append("admin.start")

    async def close(self) -> None:
        _Events.log.append("admin.close")

    async def create_topics(self, new_topics: Sequence[object]) -> None:
        for new_topic in new_topics:
            assert isinstance(new_topic, NewTopic)
            name = new_topic.name
            _Events.created.append(name)
            _Events.log.append(f"create:{name}")


class _ExistingAdmin(_FakeAdmin):
    async def create_topics(self, new_topics: Sequence[object]) -> None:
        raise TopicAlreadyExistsError


class _RefusingAdmin(_FakeAdmin):
    async def create_topics(self, new_topics: Sequence[object]) -> None:
        raise TopicAuthorizationFailedError("broker refused CreateTopics")


class _UnreachableAdmin(_FakeAdmin):
    async def start(self) -> None:
        raise KafkaConnectionError("broker unreachable")


class _FakeTransport:
    def __init__(
        self, *, config: object, group: str, topics: Sequence[str], **_kw: object
    ):
        self.group = group
        self.topics = tuple(topics)

    async def start(self) -> None:
        _Events.log.append(f"transport.start:{self.group}")

    async def close(self) -> None:
        pass

    async def poll(self, *, max_messages: int, timeout_ms: int) -> Sequence[object]:
        await asyncio.sleep(0)
        return []

    async def commit(self, message: object) -> None:
        raise AssertionError("no source message should be committed")


class _FakeStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def cleanup_expired(self, ttl_seconds: int) -> int:
        return 0

    async def is_processed(self, message_id: object, domain: str | None = None) -> bool:
        return False

    async def mark_processed(self, *args: object, **kwargs: object) -> None:
        pass

    async def check_and_record(self, *args: object, **kwargs: object) -> bool:
        return True


@pytest.fixture(autouse=True)
def _reset_events() -> None:
    _Events.log = []
    _Events.created = []


def _bus() -> ModelKafkaEventBusConfig:
    return ModelKafkaEventBusConfig(
        bootstrap_servers="redpanda:9092",
        environment="gateway-local",
        enable_auto_commit=False,
        auto_offset_reset="earliest",
    )


def _contract_topics() -> set[str]:
    """Inbound and outbound mirror topics, read straight from the contract."""
    contract = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
    mirror = contract["config"]["gateway_forwarder"]["mirror_topics"]
    return {*mirror["inbound"], *mirror["outbound"]}


@pytest.mark.asyncio
async def test_ensure_topics_exist_creates_every_requested_topic() -> None:
    await gateway_forwarder.ensure_topics_exist(
        _bus(), ("topic.a", "topic.b"), admin_factory=_FakeAdmin
    )

    assert _Events.created == ["topic.a", "topic.b"]
    assert _Events.log[0] == "admin.start"
    assert _Events.log[-1] == "admin.close"


@pytest.mark.asyncio
async def test_ensure_topics_exist_with_no_topics_opens_no_connection() -> None:
    await gateway_forwarder.ensure_topics_exist(_bus(), (), admin_factory=_FakeAdmin)

    assert _Events.log == []


@pytest.mark.asyncio
async def test_ensure_topics_exist_is_idempotent_on_an_existing_topic() -> None:
    await gateway_forwarder.ensure_topics_exist(
        _bus(), ("topic.a",), admin_factory=_ExistingAdmin
    )

    assert _Events.log[-1] == "admin.close"


@pytest.mark.asyncio
async def test_ensure_topics_exist_never_raises_when_the_broker_refuses(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        await gateway_forwarder.ensure_topics_exist(
            _bus(), ("topic.a", "topic.b"), admin_factory=_RefusingAdmin
        )

    refused = [
        r.getMessage() for r in caplog.records if "failed to ensure" in r.getMessage()
    ]
    assert len(refused) == 2
    assert all("TopicAuthorizationFailedError" in message for message in refused)
    assert all("refused CreateTopics" not in message for message in refused)


@pytest.mark.asyncio
async def test_ensure_topics_exist_never_raises_when_the_admin_is_unreachable() -> None:
    await gateway_forwarder.ensure_topics_exist(
        _bus(), ("topic.a",), admin_factory=_UnreachableAdmin
    )

    assert _Events.created == []


@pytest.mark.asyncio
async def test_process_provisions_contract_topics_before_any_consumer_starts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The shipped contract's topics exist before the first consumer starts."""
    credential_map = tmp_path / "lane-credentials.yaml"
    credential_map.write_text(
        "lane.dev.kafka.scram:\n"
        "  username: fixture-principal\n"
        "  password: fixture-not-a-real-secret\n",
        encoding="utf-8",
    )
    config = gateway_forwarder.load_gateway_forwarder_runtime_config(
        _REPO_ROOT / "docker/gateway/beta-gateway-canary.yaml",
        broker_ref_map_path=_REPO_ROOT
        / "tests/fixtures/gateway/beta-gateway-canary-broker-ref-map.yaml",
        lane_credential_map_path=credential_map,
    )
    real_ensure = gateway_forwarder.ensure_topics_exist

    async def _ensure(bus: ModelKafkaEventBusConfig, topics: Sequence[str]) -> None:
        await real_ensure(bus, topics, admin_factory=_FakeAdmin)

    monkeypatch.setattr(gateway_forwarder, "ensure_topics_exist", _ensure)
    monkeypatch.setattr(gateway_forwarder, "KafkaTransport", _FakeTransport)
    monkeypatch.setattr(gateway_forwarder, "StoreIdempotencySqlite", _FakeStore)
    shutdown_event = asyncio.Event()
    shutdown_event.set()

    async def _no_secret(_reference: str) -> str | None:
        return None

    await gateway_forwarder.run_gateway_forwarder(
        config,
        shutdown_event=shutdown_event,
        resolve_secret=_no_secret,
    )

    assert set(_Events.created) >= _contract_topics()
    assert "onex.evt.omnibase-infra.gateway-heartbeat.v1" in _Events.created
    first_start = next(
        i for i, entry in enumerate(_Events.log) if entry.startswith("transport.start")
    )
    last_create = max(
        i for i, entry in enumerate(_Events.log) if entry.startswith("create:")
    )
    assert last_create < first_start
