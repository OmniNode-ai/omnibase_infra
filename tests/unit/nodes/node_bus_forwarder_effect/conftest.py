# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Fixtures for the OMN-17034 lane-mirror leg tests.

The fakes here implement only the pull/publish surface ``NodeLaneMirror``
actually depends on, so the tests prove the service's ordering contract
(publish to every mirror, then durably mark, then commit the source) without a
broker.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

import pytest
import yaml

from omnibase_core.models.runtime.model_transport_message import ModelTransportMessage

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SOURCE_BOOTSTRAP = "omnibase-infra-stability-test-redpanda:9092"
_DEV_BOOTSTRAP = "omnibase-infra-redpanda:9092"


@dataclass
class _SentRecord:
    topic: str
    key: bytes | None
    value: bytes
    headers: object | None


class _FakeMirrorProducer:
    """Publish surface of one mirror lane; can be made to fail once."""

    def __init__(self) -> None:
        self.sent: list[_SentRecord] = []
        self.fail_next = False

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: object | None = None,
    ) -> None:
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("mirror leg refused the publish")
        self.sent.append(
            _SentRecord(topic=topic, key=key, value=value, headers=headers)
        )


class _FakeSourceConsumer:
    """Pull surface of the source lane with explicit commit / nack tracking."""

    def __init__(self) -> None:
        self._pending: list[ModelTransportMessage] = []
        self.committed: list[object] = []
        self.nacked: list[object] = []

    def offer(self, message: ModelTransportMessage) -> None:
        self._pending.append(message)

    async def poll(
        self, *, max_messages: int, timeout_ms: int
    ) -> Sequence[ModelTransportMessage]:
        batch = self._pending[:max_messages]
        self._pending = self._pending[max_messages:]
        return batch

    async def commit(self, message: object) -> None:
        self.committed.append(message)

    async def nack(self, message: object) -> None:
        self.nacked.append(message)


class _FakeIdempotencyStore:
    """Only the two ProtocolIdempotencyStore methods the mirror leg calls."""

    def __init__(self) -> None:
        self._seen: set[tuple[UUID, str | None]] = set()

    async def is_processed(self, message_id: UUID, domain: str | None = None) -> bool:
        return (message_id, domain) in self._seen

    async def mark_processed(
        self,
        message_id: UUID,
        domain: str | None = None,
        correlation_id: UUID | None = None,
        processed_at: object = None,
    ) -> None:
        self._seen.add((message_id, domain))


@dataclass
class _LaneMirrorHarness:
    source: _FakeSourceConsumer
    mirrors: dict[str, _FakeMirrorProducer]
    kwargs: dict[str, Any] = field(default_factory=dict)

    def record(
        self,
        *,
        envelope_id: str,
        topic: str = "onex.evt.omniclaude.tool-executed.v1",
        offset: int = 0,
    ) -> ModelTransportMessage:
        stable_id = uuid5(NAMESPACE_URL, f"omn17034/{envelope_id}")
        now = datetime.now(UTC).isoformat()
        envelope = {
            "envelope_id": str(stable_id),
            "envelope_timestamp": now,
            "correlation_id": str(stable_id),
            "event_type": topic,
            "payload": {"probe": "omn17034"},
        }
        return ModelTransportMessage(
            topic=topic,
            partition=0,
            offset=offset,
            key=None,
            value=json.dumps(envelope).encode("utf-8"),
            headers={},
            ack_token=f"{topic}:0:{offset}",
        )


@pytest.fixture
def lane_mirror_harness() -> _LaneMirrorHarness:
    from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
        ModelGatewayLaneMirrorConfig,
    )

    source = _FakeSourceConsumer()
    mirrors = {"dev": _FakeMirrorProducer()}
    harness = _LaneMirrorHarness(source=source, mirrors=mirrors)
    harness.kwargs = {
        "config": ModelGatewayLaneMirrorConfig(
            source_lane="stability-test",
            mirror_lanes=("dev",),
            topics=(
                "onex.evt.omniclaude.session-started.v1",
                "onex.evt.omniclaude.session-ended.v1",
                "onex.evt.omniclaude.tool-executed.v1",
                "onex.evt.omniclaude.prompt-submitted.v1",
            ),
        ),
        "source_consumer": source,
        "mirror_producers": dict(mirrors),
        "idempotency_store": _FakeIdempotencyStore(),
    }
    return harness


@pytest.fixture
def lane_mirror_runtime_raw() -> dict[str, Any]:
    """Raw runtime-config mapping already materialized from the contract."""
    from omnibase_infra.runtime.gateway_forwarder import (
        _materialize_contract_canary_config,
        _materialize_contract_lane_mirror,
        _materialize_contract_mirror_topics,
    )

    resolved_path = _REPO_ROOT / "docker" / "gateway" / "beta-gateway-canary.yaml"
    contract_path = (
        _REPO_ROOT
        / "src"
        / "omnibase_infra"
        / "nodes"
        / "node_bus_forwarder_effect"
        / "contract.yaml"
    )
    raw: dict[str, Any] = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    _materialize_contract_mirror_topics(raw, contract_path)
    _materialize_contract_canary_config(raw, contract_path)
    _materialize_contract_lane_mirror(raw, contract_path)
    raw["cloud_bus"]["bootstrap_servers"] = "cloud-broker.example:9094"
    raw.setdefault("lane_mirror_source_bus", {})["bootstrap_servers"] = (
        _SOURCE_BOOTSTRAP
    )
    raw.setdefault("lane_mirror_buses", {}).setdefault("dev", {})[
        "bootstrap_servers"
    ] = _DEV_BOOTSTRAP
    return raw
