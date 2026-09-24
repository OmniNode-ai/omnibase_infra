# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19439: the forwarder mirrors delegate-skill terminals, metadata only.

MD-42 read "not probed": after the forwarder restarted on the .201 dev lane at
17:23:43Z on 2026-09-24, a delegate-skill terminal landed at 17:26:54Z and the
outbound grep for it returned 0, because ``mirror_topics.outbound`` held no
``onex.evt.omnimarket.delegate-skill-*`` topic.

A delegate-skill terminal carries the delegated prompt (``prompt_text``), the
model's answer (``response``) and model text again inside every attempt record.
None of that may cross. So the two terminals cross behind a contract-declared
metadata ALLOWLIST applied at the boundary, before the content-addressed
``event_id`` is computed: anything not named is dropped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from omnibase_core.models.core.model_envelope_metadata import ModelEnvelopeMetadata
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayEgressMetadataScrub,
    ModelGatewayForwarderConfig,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.models.model_gateway_forwarder_config import (
    requires_metadata_scrub,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_forwarder import (
    ServiceGatewayForwarder,
    content_addressed_event_id,
)
from tests.unit.nodes.node_bus_forwarder_effect.test_egress_redaction_omn16979 import (
    PRINCIPAL_ID,
    STATE_REDACTED,
    TENANT_ID,
    TOOL_EXECUTED,
    UNGOVERNED_OUTBOUND,
    _governed_config,
)
from tests.unit.nodes.node_bus_forwarder_effect.test_egress_redaction_omn16979 import (
    _config as _base_config,
)

_CONTRACT_PATH = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_bus_forwarder_effect"
    / "contract.yaml"
)

COMPLETED = "onex.evt.omnimarket.delegate-skill-completed.v1"
FAILED = "onex.evt.omnimarket.delegate-skill-failed.v1"
TERMINALS = (COMPLETED, FAILED)

PROMPT = "summarise the confidential design doc for project nightjar"
ANSWER = "nightjar ships the new billing engine in october"

RETAINED = (
    "status",
    "correlation_id",
    "task_type",
    "tenant_id",
    "provider",
    "model_name",
    "model_cloud_baseline",
    "pricing_manifest_version",
    "quality_gate_passed",
    "quality_score",
    "required_quality_bar",
    "score_vs_required_bar",
    "terminal_failure_cause",
    "preamble_chars",
    "metrics",
    "escalation_count",
    "attempts_count",
    "queue_wait_ms",
    "execution_duration_ms",
)


def _terminal_payload() -> dict[str, object]:
    """The live delegate-skill terminal shape (read off the .201 dev broker,
    keys only), with text in every field that can carry it."""
    return {
        "status": "completed",
        "correlation_id": str(uuid4()),
        "task_type": "research",
        "tenant_id": None,
        "provider": "local",
        "model_name": "qwen3.8-27b",
        "model_cloud_baseline": "claude-opus-4-6",
        "pricing_manifest_version": 3,
        "prompt_text": PROMPT,
        "response": ANSWER,
        "quality_gate_passed": True,
        "quality_score": 1.0,
        "required_quality_bar": 0.7,
        "score_vs_required_bar": "above",
        "terminal_failure_cause": None,
        "response_contract_evidence": {"excerpt": ANSWER},
        "budget_evidence": {"note": PROMPT},
        "budget_refusal": None,
        "output_refusal": None,
        "preamble_chars": 0,
        "quality_gates_failed": [f"semantic_adequacy: response echoed {ANSWER}"],
        "metrics": {"input_tokens": 12, "output_tokens": 9, "latency_ms": 310},
        "credential_refusal": None,
        "credential_withheld": None,
        "error_message": f"model echoed: {ANSWER}",
        "escalation_count": 0,
        "attempts_count": 1,
        "attempts": [{"reasoning_preamble": ANSWER, "acceptance_detail": PROMPT}],
        "queue_wait_ms": 4,
        "execution_duration_ms": 310,
        "secret_ref": "llm.glm.api_key",
    }


def _scrub() -> ModelGatewayEgressMetadataScrub:
    return ModelGatewayEgressMetadataScrub(
        scrubbed_topics=TERMINALS, retained_payload_fields=RETAINED
    )


def _config(
    *,
    outbound: tuple[str, ...],
    scrub: ModelGatewayEgressMetadataScrub | None,
) -> ModelGatewayForwarderConfig:
    base = _base_config(outbound=(UNGOVERNED_OUTBOUND,), egress_redaction=None)
    raw = base.model_dump()
    raw["mirror_topics"] = {
        "inbound": list(base.mirror_topics.inbound),
        "outbound": list(outbound),
    }
    raw["egress_metadata_scrub"] = scrub.model_dump() if scrub is not None else None
    return ModelGatewayForwarderConfig.model_validate(raw)


def _envelope(payload: dict[str, object]) -> ModelEventEnvelope[dict[str, object]]:
    return ModelEventEnvelope[dict[str, object]](
        envelope_id=uuid4(),
        correlation_id=uuid4(),
        event_type="DelegateSkillCompleted",
        payload=payload,
        metadata=ModelEnvelopeMetadata(
            tags={
                "source_tenant_id": str(TENANT_ID),
                "source_tenant_principal_id": PRINCIPAL_ID,
            }
        ),
    )


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


def _message(topic: str, payload: dict[str, object]) -> _Message:
    return _Message(
        topic=topic,
        key=b"k",
        value=_envelope(payload).model_dump_json().encode("utf-8"),
    )


def _published_payload(message: _Message) -> dict[str, object]:
    decoded = json.loads(message.value)
    payload = decoded["payload"]
    assert isinstance(payload, dict)
    return payload


# ---------------------------------------------------------------------------
# AC1 -- the contract declares both terminals behind the scrub
# ---------------------------------------------------------------------------


def _forwarder_block() -> dict[str, object]:
    loaded = yaml.safe_load(_CONTRACT_PATH.read_text(encoding="utf-8"))
    forwarder = loaded["config"]["gateway_forwarder"]
    assert isinstance(forwarder, dict)
    return forwarder


@pytest.mark.unit
@pytest.mark.parametrize("topic", TERMINALS)
def test_contract_mirrors_each_terminal_outbound(topic: str) -> None:
    outbound = _forwarder_block()["mirror_topics"]["outbound"]  # type: ignore[index]
    assert topic in outbound


@pytest.mark.unit
def test_contract_scrub_governs_both_terminals_and_retains_no_text() -> None:
    block = _forwarder_block()["egress_metadata_scrub"]
    scrub = ModelGatewayEgressMetadataScrub.model_validate(block)
    assert set(scrub.scrubbed_topics) == set(TERMINALS)
    for text_field in (
        "prompt_text",
        "response",
        "attempts",
        "error_message",
        "quality_gates_failed",
        "response_contract_evidence",
        "budget_evidence",
        "failed_acceptance_criteria",
    ):
        assert text_field not in scrub.retained_payload_fields


# ---------------------------------------------------------------------------
# The scrub itself
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_scrub_drops_prompt_and_response_text_and_keeps_metadata() -> None:
    scrubbed = _scrub().scrub(_terminal_payload())
    blob = json.dumps(scrubbed)
    assert PROMPT not in blob
    assert ANSWER not in blob
    assert "llm.glm.api_key" not in blob
    assert set(scrubbed) == set(RETAINED)
    assert scrubbed["model_name"] == "qwen3.8-27b"
    assert scrubbed["quality_gate_passed"] is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "field",
    [
        "prompt_text",
        "response",
        "attempts",
        "secret_ref",
        "quality_gates_failed",
        "response_contract_evidence",
    ],
)
def test_model_refuses_to_retain_a_text_field(field: str) -> None:
    with pytest.raises(ValueError, match="never contain"):
        ModelGatewayEgressMetadataScrub(
            scrubbed_topics=TERMINALS, retained_payload_fields=(*RETAINED, field)
        )


@pytest.mark.unit
def test_rule_names_every_delegate_skill_terminal_and_nothing_else() -> None:
    assert requires_metadata_scrub(COMPLETED)
    assert requires_metadata_scrub(FAILED)
    assert requires_metadata_scrub("onex.evt.omnimarket.delegate-skill-timeout.v1")
    assert not requires_metadata_scrub(UNGOVERNED_OUTBOUND)
    assert not requires_metadata_scrub("onex.evt.omnimarket.tool-output-captured.v1")


@pytest.mark.unit
def test_config_refuses_a_terminal_mirrored_without_the_scrub() -> None:
    with pytest.raises(ValueError, match="unscrubbed"):
        _config(outbound=(UNGOVERNED_OUTBOUND, COMPLETED), scrub=None)


@pytest.mark.unit
def test_config_refuses_a_scrubbed_topic_nobody_mirrors() -> None:
    with pytest.raises(ValueError, match="outbound"):
        _config(outbound=(UNGOVERNED_OUTBOUND, COMPLETED), scrub=_scrub())


# ---------------------------------------------------------------------------
# Runtime behaviour at the boundary, both publish paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("topic", TERMINALS)
async def test_a_terminal_crosses_without_its_prompt_or_response(topic: str) -> None:
    local_bus, cloud_bus = _Bus(), _Bus()
    service = ServiceGatewayForwarder(
        config=_config(outbound=(UNGOVERNED_OUTBOUND, *TERMINALS), scrub=_scrub()),
        local_bus=local_bus,
        cloud_bus=cloud_bus,
    )
    await service.forward_outbound_message(_message(topic, _terminal_payload()))
    assert len(cloud_bus.published) == 1
    sent = cloud_bus.published[0]
    assert sent.topic == f"tenant-acme.{topic}"
    assert PROMPT.encode() not in sent.value
    assert ANSWER.encode() not in sent.value
    payload = _published_payload(sent)
    assert set(payload) == set(RETAINED)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_event_id_is_computed_over_the_scrubbed_payload() -> None:
    local_bus, cloud_bus = _Bus(), _Bus()
    service = ServiceGatewayForwarder(
        config=_config(outbound=(UNGOVERNED_OUTBOUND, *TERMINALS), scrub=_scrub()),
        local_bus=local_bus,
        cloud_bus=cloud_bus,
    )
    payload = _terminal_payload()
    await service.forward_outbound_message(_message(COMPLETED, payload))
    decoded = json.loads(cloud_bus.published[0].value)
    expected = content_addressed_event_id(_envelope(_scrub().scrub(payload)), COMPLETED)
    assert decoded["metadata"]["tags"]["event_id"] == expected


@pytest.mark.unit
@pytest.mark.asyncio
async def test_batch_path_scrubs_too() -> None:
    local_bus, cloud_bus = _Bus(), _BatchBus()
    service = ServiceGatewayForwarder(
        config=_config(outbound=(UNGOVERNED_OUTBOUND, *TERMINALS), scrub=_scrub()),
        local_bus=local_bus,
        cloud_bus=cloud_bus,
    )
    await service.forward_outbound_messages(
        [
            _message(COMPLETED, _terminal_payload()),
            _message(FAILED, {**_terminal_payload(), "status": "failed"}),
        ]
    )
    assert len(cloud_bus.published) == 2
    for sent in cloud_bus.published:
        assert PROMPT.encode() not in sent.value
        assert ANSWER.encode() not in sent.value


@pytest.mark.unit
@pytest.mark.asyncio
async def test_an_unscrubbed_topic_is_unchanged() -> None:
    """Positive control: the scrub never touches a topic it does not govern."""
    local_bus, cloud_bus = _Bus(), _Bus()
    service = ServiceGatewayForwarder(
        config=_config(outbound=(UNGOVERNED_OUTBOUND, *TERMINALS), scrub=_scrub()),
        local_bus=local_bus,
        cloud_bus=cloud_bus,
    )
    payload: dict[str, object] = {"content": ANSWER, "model_used": "m"}
    await service.forward_outbound_message(_message(UNGOVERNED_OUTBOUND, payload))
    assert _published_payload(cloud_bus.published[0]) == payload


@pytest.mark.unit
@pytest.mark.asyncio
async def test_batch_path_applies_the_redaction_admission() -> None:
    """The batch (HTTPS) path used to skip ``egress_admits``: an unstamped
    governed capture record crossed whenever a poll returned it in a batch.
    Same batch, one stamped and one unstamped record: only the stamped one
    crosses."""
    local_bus, cloud_bus = _Bus(), _BatchBus()
    service = ServiceGatewayForwarder(
        config=_governed_config(), local_bus=local_bus, cloud_bus=cloud_bus
    )
    await service.forward_outbound_messages(
        [
            _message(TOOL_EXECUTED, {"tool_name": "Bash"}),
            _message(
                TOOL_EXECUTED,
                {"tool_name": "Bash", "redaction_state": STATE_REDACTED},
            ),
        ]
    )
    assert len(cloud_bus.published) == 1
    assert _published_payload(cloud_bus.published[0])["redaction_state"] == (
        STATE_REDACTED
    )
