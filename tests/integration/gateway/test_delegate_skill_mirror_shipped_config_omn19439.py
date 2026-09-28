# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19439: the SHIPPED forwarder deployment mirrors delegate-skill terminals, metadata only.

The unit tests build a forwarder config by hand. This test resolves the one the
.201 gateway actually runs: ``docker/gateway/beta-gateway-canary.yaml`` through
``load_gateway_forwarder_runtime_config``, which reads ``mirror_topics`` and
``egress_metadata_scrub`` out of the real node contract. It then drives the
real ``ServiceGatewayForwarder`` over both outbound paths (single record and
the HTTPS batch) and checks what would cross.

It mirrors the lab proof run on .201 on 2026-09-24: the branch code on the
deployed forwarder image, fed the latest 10 completed and 10 failed terminals
off the dev broker, crossed 20 of 20 on each path with no prompt, response,
error or attempt text in the output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest

from omnibase_core.models.core.model_envelope_metadata import ModelEnvelopeMetadata
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayForwarderRuntimeConfig,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_forwarder import (
    ServiceGatewayForwarder,
    content_addressed_event_id,
)
from omnibase_infra.runtime import gateway_forwarder

pytestmark = pytest.mark.integration

_EVENT_ID_TAG = "event_id"

_REPO_ROOT = Path(__file__).resolve().parents[3]
COMPLETED = "onex.evt.omnimarket.delegate-skill-completed.v1"
FAILED = "onex.evt.omnimarket.delegate-skill-failed.v1"
PROMPT = "summarise the confidential design doc for project nightjar"
ANSWER = "nightjar ships the new billing engine in october"


@dataclass(frozen=True)
class _Message:
    topic: str
    key: bytes | None
    value: bytes
    headers: object | None = None


class _Bus:
    def __init__(self) -> None:
        self.published: list[_Message] = []

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: object | None = None,
    ) -> None:
        self.published.append(_Message(topic, key, value, headers))


class _BatchBus(_Bus):
    async def publish_batch(
        self, records: list[tuple[str, bytes | None, bytes, object | None]]
    ) -> None:
        for topic, key, value, headers in records:
            self.published.append(_Message(topic, key, value, headers))


def _shipped_config(
    tmp_path: Path,
) -> ModelGatewayForwarderRuntimeConfig:
    # The shipped config's dev-lane legs authenticate and the loader fails
    # closed without a credential map. Obviously-fake values: this test is
    # about topic and scrub resolution, not the credential.
    credential_map = tmp_path / "lane-credentials.yaml"
    credential_map.write_text(
        "lane.dev.kafka.scram:\n"
        "  username: fixture-principal\n"
        "  password: fixture-not-a-credential\n",
        encoding="utf-8",
    )
    return gateway_forwarder.load_gateway_forwarder_runtime_config(
        _REPO_ROOT / "docker/gateway/beta-gateway-canary.yaml",
        broker_ref_map_path=_REPO_ROOT
        / "tests/fixtures/gateway/beta-gateway-canary-broker-ref-map.yaml",
        lane_credential_map_path=credential_map,
    )


def _terminal_payload(status: str) -> dict[str, object]:
    # Keys as read off the .201 dev broker. ``tenant_id`` is null on every live
    # terminal read on 2026-09-24, which is why none is refused by the
    # attached-tenant check.
    return {
        "status": status,
        "correlation_id": str(uuid4()),
        "task_type": "research",
        "tenant_id": None,
        "provider": "local",
        "model_name": "qwen3.8-27b",
        "model_cloud_baseline": "claude-opus-4-6",
        "pricing_manifest_version": 3,
        "prompt_text": PROMPT,
        "response": ANSWER,
        "quality_gate_passed": status == "completed",
        "quality_score": 1.0,
        "required_quality_bar": 0.7,
        "score_vs_required_bar": "above",
        "terminal_failure_cause": None if status == "completed" else "provider_error",
        "failure_reason": f"refused before the handler ran: {PROMPT}",
        "error_message": f"model echoed: {ANSWER}",
        "quality_gates_failed": [f"semantic_adequacy: response echoed {ANSWER}"],
        "metrics": {"input_tokens": 12, "output_tokens": 9, "latency_ms": 310},
        "attempts_count": 1,
        "attempts": [{"reasoning_preamble": ANSWER, "acceptance_detail": PROMPT}],
        "escalation_count": 0,
        "queue_wait_ms": 4,
        "execution_duration_ms": 310,
        "secret_ref": "llm.glm.api_key",
    }


def _message(topic: str, status: str) -> _Message:
    envelope = ModelEventEnvelope[dict[str, object]](
        envelope_id=uuid4(),
        correlation_id=uuid4(),
        event_type="DelegateSkillTerminal",
        payload=_terminal_payload(status),
        metadata=ModelEnvelopeMetadata(tags={}),
    )
    return _Message(
        topic=topic, key=b"k", value=envelope.model_dump_json().encode("utf-8")
    )


@pytest.mark.asyncio
async def test_shipped_deployment_mirrors_both_terminals_metadata_only(
    tmp_path: Path,
) -> None:
    config = _shipped_config(tmp_path)
    forwarder_config = config.forwarder
    scrub = forwarder_config.egress_metadata_scrub
    assert scrub is not None
    retained = set(scrub.retained_payload_fields)
    slug = forwarder_config.tenant_identity.tenant_slug

    messages = [_message(COMPLETED, "completed"), _message(FAILED, "failed")]

    single = _Bus()
    await ServiceGatewayForwarder(
        config=forwarder_config, local_bus=_Bus(), cloud_bus=single
    ).forward_outbound_message(messages[0])
    await ServiceGatewayForwarder(
        config=forwarder_config, local_bus=_Bus(), cloud_bus=single
    ).forward_outbound_message(messages[1])

    batch = _BatchBus()
    await ServiceGatewayForwarder(
        config=forwarder_config, local_bus=_Bus(), cloud_bus=batch
    ).forward_outbound_messages(list(messages))

    for published in (single.published, batch.published):
        assert [m.topic for m in published] == [
            f"tenant-{slug}.{COMPLETED}",
            f"tenant-{slug}.{FAILED}",
        ]
        for message in published:
            text = message.value.decode("utf-8")
            assert PROMPT not in text
            assert ANSWER not in text
            assert "llm.glm.api_key" not in text
            payload = json.loads(text)["payload"]
            assert set(payload) <= retained, set(payload) - retained
            assert "status" in payload
            assert "metrics" in payload

    # Both outbound paths hash the same scrubbed form, so a record that crosses
    # once on each leg is deduplicated downstream under one event_id, and the
    # id is the hash of what crossed, never of the unscrubbed payload.
    def _event_ids(published: list[_Message]) -> list[str]:
        return [
            json.loads(m.value)["metadata"]["tags"][_EVENT_ID_TAG] for m in published
        ]

    assert _event_ids(single.published) == _event_ids(batch.published)
    for message, source in zip(single.published, messages, strict=True):
        crossed = ModelEventEnvelope[dict[str, object]].model_validate_json(
            message.value
        )
        assert _event_ids([message])[0] == content_addressed_event_id(
            crossed, source.topic
        )
