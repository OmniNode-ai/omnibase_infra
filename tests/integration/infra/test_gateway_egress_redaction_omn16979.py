# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for the OMN-16979 outbound redaction boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

import pytest
import yaml

from omnibase_core.models.core.model_envelope_metadata import ModelEnvelopeMetadata
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayCanaryConfig,
    ModelGatewayCloudBusConfig,
    ModelGatewayEgressRedaction,
    ModelGatewayForwarderConfig,
    ModelGatewayMirrorTopics,
    ModelGatewayTenantIdentity,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_forwarder import (
    ServiceGatewayForwarder,
)

pytestmark = pytest.mark.integration

TENANT_ID = UUID("11111111-1111-1111-1111-111111111111")
BROKER_PROVIDER_ID = UUID("22222222-2222-2222-2222-222222222222")
PRINCIPAL_ID = "t-33333333333333333333333333333333"

TOOL_EXECUTED = "onex.evt.omniclaude.tool-executed.v1"
PROMPT_SUBMITTED = "onex.evt.omniclaude.prompt-submitted.v1"
SESSION_STARTED = "onex.evt.omniclaude.session-started.v1"
SESSION_ENDED = "onex.evt.omniclaude.session-ended.v1"
SKILL_STARTED = "onex.evt.omniclaude.skill-started.v1"
SKILL_COMPLETED = "onex.evt.omniclaude.skill-completed.v1"
TOOL_OUTPUT_CAPTURED = "onex.evt.omnimarket.tool-output-captured.v1"
CAPTURE_TOPICS = (
    SESSION_STARTED,
    SESSION_ENDED,
    PROMPT_SUBMITTED,
    TOOL_EXECUTED,
    SKILL_STARTED,
    SKILL_COMPLETED,
    TOOL_OUTPUT_CAPTURED,
)
UNGOVERNED_OUTBOUND = "onex.evt.omnibase-infra.inference-response.v1"

STATE_REDACTED = "redacted"
STATE_RESTRICTED = "restricted"
STATE_SECRET_DETECTED = "secret_detected"
STATE_RAW = "raw"
ADMITTED_STATES = (STATE_REDACTED, STATE_RESTRICTED, STATE_SECRET_DETECTED)

CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_bus_forwarder_effect"
    / "contract.yaml"
)


@dataclass(frozen=True)
class Message:
    topic: str
    key: bytes | None
    value: bytes
    headers: object | None = None


class MockGatewayBus:
    def __init__(self) -> None:
        self.published: list[Message] = []

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: object | None = None,
    ) -> None:
        self.published.append(Message(topic, key, value, headers))

    def message(
        self,
        topic: str,
        envelope: ModelEventEnvelope[dict[str, object]],
    ) -> Message:
        return Message(
            topic=topic,
            key=b"key-1",
            value=envelope.model_dump_json().encode("utf-8"),
        )


def _contract_forwarder_block() -> dict[str, object]:
    loaded = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    config = loaded["config"]
    assert isinstance(config, dict)
    forwarder = config["gateway_forwarder"]
    assert isinstance(forwarder, dict)
    return forwarder


def _config(dedupe_store_path: Path) -> ModelGatewayForwarderConfig:
    return ModelGatewayForwarderConfig(
        tenant_identity=ModelGatewayTenantIdentity(
            tenant_id=TENANT_ID,
            tenant_slug="acme",
            principal_id=PRINCIPAL_ID,
        ),
        cloud_bus=ModelGatewayCloudBusConfig(
            broker_provider_id=BROKER_PROVIDER_ID,
            cloud_broker_ref="gateway.cloud.kafka.broker",
            cloud_auth_ref="gateway.cloud.kafka.oauth",
            acl_provisioner_ref="gateway.cloud.kafka.authorization",
            client_id_ref="gateway.cloud.kafka.oauth.client_id",
            client_secret_api_key_ref="gateway.cloud.kafka.oauth.client_secret",
        ),
        local_transport_flavor="containerized",
        dedupe_store_path=dedupe_store_path,
        mirror_topics=ModelGatewayMirrorTopics(
            inbound=("onex.cmd.omnibase-infra.delegation-inference-request.v1",),
            outbound=(UNGOVERNED_OUTBOUND, *CAPTURE_TOPICS),
        ),
        canary=ModelGatewayCanaryConfig(
            topic="onex.evt.omnibase-infra.gateway-canary.v1",
            cadence_seconds=30,
            produce_deadline_seconds=8,
            readback_deadline_seconds=12,
        ),
        egress_redaction=ModelGatewayEgressRedaction(
            state_field="redaction_state",
            admitted_states=ADMITTED_STATES,
            governed_topics=CAPTURE_TOPICS,
        ),
    )


def _envelope(payload: dict[str, object]) -> ModelEventEnvelope[dict[str, object]]:
    return ModelEventEnvelope[dict[str, object]](
        envelope_id=uuid4(),
        correlation_id=uuid4(),
        event_type="OmniclaudeCaptureEvent",
        payload=payload,
        metadata=ModelEnvelopeMetadata(
            tags={
                "source_tenant_id": str(TENANT_ID),
                "source_tenant_principal_id": PRINCIPAL_ID,
            }
        ),
    )


def test_contract_declares_every_capture_topic_under_redaction_gate() -> None:
    forwarder = _contract_forwarder_block()
    outbound = forwarder["mirror_topics"]["outbound"]  # type: ignore[index]
    redaction = forwarder["egress_redaction"]  # type: ignore[index]
    governed = redaction["governed_topics"]  # type: ignore[index]

    assert set(CAPTURE_TOPICS) <= set(outbound)
    assert set(governed) == set(CAPTURE_TOPICS)


@pytest.mark.asyncio
@pytest.mark.parametrize("topic", CAPTURE_TOPICS)
@pytest.mark.parametrize("redaction_state", ADMITTED_STATES)
async def test_capture_topics_cross_only_with_admitted_redaction_state(
    topic: str,
    redaction_state: str,
    tmp_path: Path,
) -> None:
    local_bus = MockGatewayBus()
    cloud_bus = MockGatewayBus()
    service = ServiceGatewayForwarder(
        config=_config(tmp_path / "gateway-egress-integration.sqlite3"),
        local_bus=local_bus,
        cloud_bus=cloud_bus,
    )

    await service.forward_outbound_message(
        local_bus.message(topic, _envelope({"redaction_state": redaction_state}))
    )
    await service.forward_outbound_message(local_bus.message(topic, _envelope({})))

    assert [message.topic for message in cloud_bus.published] == [
        f"tenant-acme.{topic}"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("topic", CAPTURE_TOPICS)
async def test_capture_topics_drop_raw_state_on_the_integration_path(
    topic: str,
    tmp_path: Path,
) -> None:
    local_bus = MockGatewayBus()
    cloud_bus = MockGatewayBus()
    service = ServiceGatewayForwarder(
        config=_config(tmp_path / "gateway-egress-integration.sqlite3"),
        local_bus=local_bus,
        cloud_bus=cloud_bus,
    )

    await service.forward_outbound_message(
        local_bus.message(topic, _envelope({"redaction_state": STATE_RAW}))
    )

    assert cloud_bus.published == []
