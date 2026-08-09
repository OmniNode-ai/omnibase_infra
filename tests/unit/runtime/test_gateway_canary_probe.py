# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the OMN-15741 path-verifying gateway canary probe.

Covers the exact acceptance criteria: a real produce+readback per leg, a
distinguishable local-vs-cloud failure report, non-zero exit when either leg
is dead even though "the process" (here: the fake transport) is reachable,
and the cadence cache that keeps the healthcheck from spamming the brokers.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayCanaryConfig,
    ModelGatewayCloudBusConfig,
    ModelGatewayForwarderConfig,
    ModelGatewayForwarderRuntimeConfig,
    ModelGatewayMirrorTopics,
    ModelGatewayTenantIdentity,
)
from omnibase_infra.runtime import gateway_canary_probe

CANARY_TOPIC = "onex.evt.omnibase-infra.gateway-canary.v1"


def _canary(**overrides: object) -> ModelGatewayCanaryConfig:
    values: dict[str, object] = {
        "topic": CANARY_TOPIC,
        "cadence_seconds": 30,
        "produce_deadline_seconds": 1,
        "readback_deadline_seconds": 1,
    }
    values.update(overrides)
    return ModelGatewayCanaryConfig(**values)  # type: ignore[arg-type]


def _bus_config() -> ModelKafkaEventBusConfig:
    return ModelKafkaEventBusConfig(
        bootstrap_servers="redpanda:9092",
        environment="gateway-test",
        enable_auto_commit=False,
    )


class _FakeHealthyTransport:
    """Round-trips whatever was sent back out of poll(), like a real broker."""

    def __init__(
        self,
        *,
        config: ModelKafkaEventBusConfig,
        group: str,
        topics: Sequence[str],
        auto_offset_reset: str,
    ) -> None:
        self.config = config
        self.group = group
        self.topics = tuple(topics)
        self.started = False
        self.closed = False
        self._inbox: list[object] = []

    async def start(self) -> None:
        self.started = True

    async def close(self) -> None:
        self.closed = True

    async def send(
        self, topic: str, key: bytes | None, value: bytes, headers: Mapping[str, bytes]
    ) -> None:
        self._inbox.append(
            _Message(topic=topic, partition=0, offset=0, key=key, value=value)
        )

    async def poll(self, *, max_messages: int, timeout_ms: int) -> Sequence[object]:
        await asyncio.sleep(0)
        batch, self._inbox = self._inbox[:max_messages], self._inbox[max_messages:]
        return batch


class _Message:
    def __init__(
        self,
        *,
        topic: str,
        partition: int,
        offset: int,
        key: bytes | None,
        value: bytes,
    ) -> None:
        self.topic = topic
        self.partition = partition
        self.offset = offset
        self.key = key
        self.value = value


class _FakeDeadTransport(_FakeHealthyTransport):
    """Connects fine but never returns the produced record (cloud leg dead)."""

    async def poll(self, *, max_messages: int, timeout_ms: int) -> Sequence[object]:
        await asyncio.sleep(timeout_ms / 1000)
        return []


class _FakeUnreachableTransport(_FakeHealthyTransport):
    """Fails to even connect."""

    async def start(self) -> None:
        raise ConnectionError("no route to broker")


@pytest.mark.asyncio
async def test_check_canary_leg_passes_on_real_round_trip() -> None:
    result = await gateway_canary_probe.check_canary_leg(
        leg="local",
        bus_config=_bus_config(),
        topic=CANARY_TOPIC,
        canary=_canary(),
        transport_factory=_FakeHealthyTransport,
    )
    assert result.passed is True
    assert "local" in result.detail


@pytest.mark.asyncio
async def test_check_canary_leg_fails_on_readback_timeout() -> None:
    result = await gateway_canary_probe.check_canary_leg(
        leg="cloud",
        bus_config=_bus_config(),
        topic=CANARY_TOPIC,
        canary=_canary(readback_deadline_seconds=0.1),
        transport_factory=_FakeDeadTransport,
    )
    assert result.passed is False
    assert "cloud" in result.detail
    assert "timed out" in result.detail


@pytest.mark.asyncio
async def test_check_canary_leg_fails_on_connect_error() -> None:
    result = await gateway_canary_probe.check_canary_leg(
        leg="cloud",
        bus_config=_bus_config(),
        topic=CANARY_TOPIC,
        canary=_canary(),
        transport_factory=_FakeUnreachableTransport,
    )
    assert result.passed is False
    assert "connect failed" in result.detail


@pytest.mark.asyncio
async def test_run_canary_check_distinguishes_local_healthy_cloud_dead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact 2026-08-04 failure mode: local leg fine, cloud leg dead."""
    config = ModelGatewayForwarderRuntimeConfig(
        forwarder=ModelGatewayForwarderConfig(
            tenant_identity=ModelGatewayTenantIdentity(
                tenant_id=UUID("11111111-1111-1111-1111-111111111111"),
                tenant_slug="acme",
                principal_id="t-33333333333333333333333333333333",
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
                outbound=("onex.evt.omnibase-infra.gateway-heartbeat.v1",),
            ),
            canary=_canary(readback_deadline_seconds=0.1),
        ),
        local_bus=_bus_config(),
        cloud_bus=ModelKafkaEventBusConfig(
            bootstrap_servers="b-1.example.kafka.amazonaws.com:9098",
            environment="gateway-cloud",
            security_protocol="SASL_SSL",
            sasl_mechanism="AWS_MSK_IAM",
            msk_region="us-east-1",
            enable_auto_commit=False,
        ),
    )

    real_check_canary_leg = gateway_canary_probe.check_canary_leg

    async def _fake_check(*, leg, bus_config, topic, canary, transport_factory=None):  # type: ignore[no-untyped-def]
        healthy = leg == "local"
        factory = _FakeHealthyTransport if healthy else _FakeDeadTransport
        return await real_check_canary_leg(
            leg=leg,
            bus_config=bus_config,
            topic=topic,
            canary=canary,
            transport_factory=factory,
        )

    monkeypatch.setattr(gateway_canary_probe, "check_canary_leg", _fake_check)

    local_result, cloud_result = await gateway_canary_probe.run_canary_check(config)

    assert local_result.leg == "local"
    assert local_result.passed is True
    assert cloud_result.leg == "cloud"
    assert cloud_result.passed is False
    assert "timed out" in cloud_result.detail


@pytest.mark.asyncio
async def test_probe_reports_overall_fail_when_either_leg_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ModelGatewayForwarderRuntimeConfig(
        forwarder=ModelGatewayForwarderConfig(
            tenant_identity=ModelGatewayTenantIdentity(
                tenant_id=UUID("11111111-1111-1111-1111-111111111111"),
                tenant_slug="acme",
                principal_id="t-33333333333333333333333333333333",
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
                outbound=("onex.evt.omnibase-infra.gateway-heartbeat.v1",),
            ),
            canary=_canary(readback_deadline_seconds=0.1),
        ),
        local_bus=_bus_config(),
        cloud_bus=ModelKafkaEventBusConfig(
            bootstrap_servers="b-1.example.kafka.amazonaws.com:9098",
            environment="gateway-cloud",
            security_protocol="SASL_SSL",
            sasl_mechanism="AWS_MSK_IAM",
            msk_region="us-east-1",
            enable_auto_commit=False,
        ),
    )

    real_check_canary_leg = gateway_canary_probe.check_canary_leg

    async def _fake_check(*, leg, bus_config, topic, canary, transport_factory=None):  # type: ignore[no-untyped-def]
        factory = _FakeHealthyTransport if leg == "local" else _FakeUnreachableTransport
        return await real_check_canary_leg(
            leg=leg,
            bus_config=bus_config,
            topic=topic,
            canary=canary,
            transport_factory=factory,
        )

    monkeypatch.setattr(gateway_canary_probe, "check_canary_leg", _fake_check)

    state_path = tmp_path / "state.json"
    passed, report = await gateway_canary_probe.probe(config, state_path=state_path)

    assert passed is False
    assert "PASS" in report and "local" in report
    assert "FAIL" in report and "cloud" in report

    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["passed"] is False


@pytest.mark.asyncio
async def test_probe_serves_cached_result_within_cadence(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {"checked_at": time.time(), "passed": True, "report": "PASS: cached"}
        ),
        encoding="utf-8",
    )

    calls = {"count": 0}

    async def _never_called(*args: object, **kwargs: object) -> None:
        calls["count"] += 1

    # If the cache is honored, run_canary_check must never be invoked.
    original = gateway_canary_probe.run_canary_check
    gateway_canary_probe.run_canary_check = _never_called  # type: ignore[assignment]
    try:
        config = ModelGatewayForwarderRuntimeConfig(
            forwarder=ModelGatewayForwarderConfig(
                tenant_identity=ModelGatewayTenantIdentity(
                    tenant_id=UUID("11111111-1111-1111-1111-111111111111"),
                    tenant_slug="acme",
                    principal_id="t-33333333333333333333333333333333",
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
                    outbound=("onex.evt.omnibase-infra.gateway-heartbeat.v1",),
                ),
                canary=_canary(cadence_seconds=300),
            ),
            local_bus=_bus_config(),
            cloud_bus=ModelKafkaEventBusConfig(
                bootstrap_servers="b-1.example.kafka.amazonaws.com:9098",
                environment="gateway-cloud",
                security_protocol="SASL_SSL",
                sasl_mechanism="AWS_MSK_IAM",
                msk_region="us-east-1",
                enable_auto_commit=False,
            ),
        )
        passed, report = await gateway_canary_probe.probe(config, state_path=state_path)
    finally:
        gateway_canary_probe.run_canary_check = original  # type: ignore[assignment]

    assert calls["count"] == 0
    assert passed is True
    assert report == "PASS: cached"


def test_canary_config_rejects_empty_topic() -> None:
    with pytest.raises(ValidationError):
        ModelGatewayCanaryConfig(
            topic="",
            cadence_seconds=30,
            produce_deadline_seconds=8,
            readback_deadline_seconds=12,
        )
