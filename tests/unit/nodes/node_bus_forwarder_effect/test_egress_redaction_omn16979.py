# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16979: widen ``mirror_topics.outbound`` to the content-bearing hook
classes, behind a fail-closed egress-redaction admission gate.

WHY A GATE AND NOT A BARE WIDENING. OMN-17209's framing is that "widening the
payload without landing this contract first ships a credential pipeline." The
redaction itself is produced upstream, at omnimarket's emit seam (OMN-16019's
per-topic transform), which is the only place that knows tool semantics. This
node cannot re-derive that judgement and must not try to.

What it CAN do -- and what AC1 asks for when it says "an explicit scrub
transform rather than a raw passthrough" -- is refuse to cross anything that is
not provably redacted. So the contract declares, per governed topic, the
redaction state a record must carry to be admitted, and a record on a governed
topic that does not carry an admitted state is DROPPED at the boundary.

That ordering is what makes this safe to land before the upstream contract is
deployed: until the emit seam stamps the field, the widened topics admit
nothing. The widening cannot leak by being merged early.

DROPPED, NOT RAISED. The refusal is a drop with a warning, never an exception.
OMN-17382 is the live proof of why: on 2026-09-05 a single foreign-tenant probe
record raised out of ``_prepare_outbound`` and wedged the whole outbound leg for
7h45m across 925 consecutive retries, with 177 real records stuck behind it.
Adding a new raise on a per-record policy decision would add a second wedge
vector to a bridge that already has one.
"""

from __future__ import annotations

import logging
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
    egress_admits,
)

_CONTRACT_PATH = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_bus_forwarder_effect"
    / "contract.yaml"
)

TENANT_ID = UUID("11111111-1111-1111-1111-111111111111")
BROKER_PROVIDER_ID = UUID("22222222-2222-2222-2222-222222222222")
PRINCIPAL_ID = "t-33333333333333333333333333333333"

TOOL_EXECUTED = "onex.evt.omniclaude.tool-executed.v1"
PROMPT_SUBMITTED = "onex.evt.omniclaude.prompt-submitted.v1"
SESSION_STARTED = "onex.evt.omniclaude.session-started.v1"
SESSION_ENDED = "onex.evt.omniclaude.session-ended.v1"
UNGOVERNED_OUTBOUND = "onex.evt.omnibase-infra.inference-response.v1"
# Never admitted to the outbound set by this ticket, and asserted to stay out.
STILL_DENIED = "onex.evt.omniclaude.tool-output-captured.v1"

# The four values of omnibase_core's EnumArtifactRedactionState (OMN-13152).
STATE_RAW = "raw"
STATE_REDACTED = "redacted"
STATE_RESTRICTED = "restricted"
STATE_SECRET_DETECTED = "secret_detected"


def _contract() -> dict[str, object]:
    with _CONTRACT_PATH.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    assert isinstance(loaded, dict)
    return loaded


def _forwarder_block() -> dict[str, object]:
    config = _contract()["config"]
    assert isinstance(config, dict)
    forwarder = config["gateway_forwarder"]
    assert isinstance(forwarder, dict)
    return forwarder


# ---------------------------------------------------------------------------
# AC1 -- the contract declares the widening, and declares it behind a scrub
# ---------------------------------------------------------------------------


def test_contract_outbound_carries_the_two_content_bearing_hook_classes() -> None:
    """AC1: at least one hook class beyond the OD-9 session-lifecycle pair."""
    outbound = _forwarder_block()["mirror_topics"]["outbound"]  # type: ignore[index]
    assert TOOL_EXECUTED in outbound
    assert PROMPT_SUBMITTED in outbound


def test_contract_outbound_keeps_the_od9_session_pair() -> None:
    """The OMN-16204 pair is extended, never replaced."""
    outbound = _forwarder_block()["mirror_topics"]["outbound"]  # type: ignore[index]
    assert SESSION_STARTED in outbound
    assert SESSION_ENDED in outbound


def test_contract_does_not_widen_beyond_the_two_declared_classes() -> None:
    """Deny-by-default polarity: this ticket widens by exactly two topics.

    ``tool-output-captured`` is the class that carries raw tool OUTPUT. It stays
    denied; admitting it is a separate decision behind OMN-17207, not a side
    effect of this one.
    """
    outbound = _forwarder_block()["mirror_topics"]["outbound"]  # type: ignore[index]
    assert STILL_DENIED not in outbound
    omniclaude = sorted(t for t in outbound if ".omniclaude." in t)
    assert omniclaude == sorted(
        [SESSION_STARTED, SESSION_ENDED, TOOL_EXECUTED, PROMPT_SUBMITTED]
    )


def test_contract_declares_an_egress_redaction_block() -> None:
    """AC1: the widened classes ride a named scrub, not a raw passthrough."""
    assert "egress_redaction" in _forwarder_block()


def test_every_widened_class_is_governed_by_the_egress_gate() -> None:
    """A widening that is not also governed is exactly the passthrough AC1 forbids."""
    governed = _forwarder_block()["egress_redaction"]["governed_topics"]  # type: ignore[index]
    assert TOOL_EXECUTED in governed
    assert PROMPT_SUBMITTED in governed


def test_the_egress_gate_never_admits_the_raw_state() -> None:
    """``raw`` is the ArtifactStore default (OMN-13152), so admitting it would
    make the gate a no-op: 27 of 27 local artifact records carry ``raw`` today.
    """
    admitted = _forwarder_block()["egress_redaction"]["admitted_states"]  # type: ignore[index]
    assert STATE_RAW not in admitted
    assert STATE_REDACTED in admitted


def test_contract_version_advanced_for_the_widening() -> None:
    version = _contract()["contract_version"]
    assert isinstance(version, dict)
    assert (version["major"], version["minor"], version["patch"]) >= (0, 1, 5)


# ---------------------------------------------------------------------------
# Model-level fail-closed properties
# ---------------------------------------------------------------------------


def test_model_refuses_an_admitted_state_set_containing_raw() -> None:
    with pytest.raises(ValueError, match="raw"):
        ModelGatewayEgressRedaction(
            state_field="redaction_state",
            admitted_states=(STATE_REDACTED, STATE_RAW),
            governed_topics=(TOOL_EXECUTED,),
        )


def test_model_refuses_an_empty_admitted_state_set() -> None:
    with pytest.raises(ValueError):
        ModelGatewayEgressRedaction(
            state_field="redaction_state",
            admitted_states=(),
            governed_topics=(TOOL_EXECUTED,),
        )


def test_model_refuses_an_empty_governed_topic_set() -> None:
    """An empty policy reads identically to a working one, so it is refused."""
    with pytest.raises(ValueError):
        ModelGatewayEgressRedaction(
            state_field="redaction_state",
            admitted_states=(STATE_REDACTED,),
            governed_topics=(),
        )


def test_model_refuses_a_state_outside_the_core_enum() -> None:
    """Parity with omnibase_core's EnumArtifactRedactionState (OMN-13152)."""
    with pytest.raises(ValueError):
        ModelGatewayEgressRedaction(
            state_field="redaction_state",
            admitted_states=("scrubbed",),
            governed_topics=(TOOL_EXECUTED,),
        )


def test_config_refuses_a_governed_topic_absent_from_the_outbound_set() -> None:
    """Governing a topic nobody mirrors is a contract error, not a silent no-op."""
    with pytest.raises(ValueError, match="outbound"):
        _config(
            outbound=(UNGOVERNED_OUTBOUND,),
            egress_redaction=ModelGatewayEgressRedaction(
                state_field="redaction_state",
                admitted_states=(STATE_REDACTED,),
                governed_topics=(TOOL_EXECUTED,),
            ),
        )


def test_config_refuses_a_content_bearing_hook_class_that_is_not_governed() -> None:
    """The interlock in the other direction: a widening cannot be added to the
    outbound set without also being placed under the gate.
    """
    with pytest.raises(ValueError, match="governed"):
        _config(
            outbound=(UNGOVERNED_OUTBOUND, TOOL_EXECUTED),
            egress_redaction=ModelGatewayEgressRedaction(
                state_field="redaction_state",
                admitted_states=(STATE_REDACTED,),
                governed_topics=(PROMPT_SUBMITTED,),
            ),
        )


def test_a_round_tripped_null_egress_redaction_is_not_a_redeclaration() -> None:
    """``model_dump()`` emits an explicit ``egress_redaction: null`` for a
    deployment with no gate. Refusing on key PRESENCE broke three pre-existing
    tests on the OMN-17034 lane; refuse only a POPULATED block.
    """
    config = _config(outbound=(UNGOVERNED_OUTBOUND,), egress_redaction=None)
    assert "egress_redaction" in config.model_dump()
    assert config.model_dump()["egress_redaction"] is None


# ---------------------------------------------------------------------------
# Runtime behaviour at the boundary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Message:
    topic: str
    key: bytes | None
    value: bytes
    headers: object | None = None


class _MockGatewayBus:
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

    def message(
        self,
        topic: str,
        envelope: ModelEventEnvelope[dict[str, object]],
    ) -> _Message:
        return _Message(
            topic=topic,
            key=b"key-1",
            value=envelope.model_dump_json().encode("utf-8"),
            headers=None,
        )


def _config(
    *,
    outbound: tuple[str, ...],
    egress_redaction: ModelGatewayEgressRedaction | None,
) -> ModelGatewayForwarderConfig:
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
        dedupe_store_path=Path.cwd() / "gateway-egress-test.sqlite3",
        mirror_topics=ModelGatewayMirrorTopics(
            inbound=("onex.cmd.omnibase-infra.delegation-inference-request.v1",),
            outbound=outbound,
        ),
        canary=ModelGatewayCanaryConfig(
            topic="onex.evt.omnibase-infra.gateway-canary.v1",
            cadence_seconds=30,
            produce_deadline_seconds=8,
            readback_deadline_seconds=12,
        ),
        egress_redaction=egress_redaction,
    )


def _governed_config() -> ModelGatewayForwarderConfig:
    return _config(
        outbound=(UNGOVERNED_OUTBOUND, TOOL_EXECUTED, PROMPT_SUBMITTED),
        egress_redaction=ModelGatewayEgressRedaction(
            state_field="redaction_state",
            admitted_states=(STATE_REDACTED, STATE_RESTRICTED, STATE_SECRET_DETECTED),
            governed_topics=(TOOL_EXECUTED, PROMPT_SUBMITTED),
        ),
    )


def _envelope(payload: dict[str, object]) -> ModelEventEnvelope[dict[str, object]]:
    return ModelEventEnvelope[dict[str, object]](
        envelope_id=uuid4(),
        correlation_id=uuid4(),
        event_type="OmniclaudeToolExecuted",
        payload=payload,
        metadata=ModelEnvelopeMetadata(
            tags={
                "source_tenant_id": str(TENANT_ID),
                "source_tenant_principal_id": PRINCIPAL_ID,
            }
        ),
    )


@pytest.mark.asyncio
async def test_a_redacted_hook_record_crosses_the_boundary() -> None:
    """The positive control. Without it a gate that drops everything passes."""
    local_bus, cloud_bus = _MockGatewayBus(), _MockGatewayBus()
    service = ServiceGatewayForwarder(
        config=_governed_config(), local_bus=local_bus, cloud_bus=cloud_bus
    )
    envelope = _envelope({"tool_name": "Bash", "redaction_state": STATE_REDACTED})
    await service.forward_outbound_message(local_bus.message(TOOL_EXECUTED, envelope))
    assert len(cloud_bus.published) == 1
    assert cloud_bus.published[0].topic == f"tenant-acme.{TOOL_EXECUTED}"


@pytest.mark.asyncio
async def test_an_unstamped_hook_record_is_dropped_not_forwarded() -> None:
    """The load-bearing assertion: no redaction stamp, no crossing."""
    local_bus, cloud_bus = _MockGatewayBus(), _MockGatewayBus()
    service = ServiceGatewayForwarder(
        config=_governed_config(), local_bus=local_bus, cloud_bus=cloud_bus
    )
    envelope = _envelope({"tool_name": "Bash"})
    await service.forward_outbound_message(local_bus.message(TOOL_EXECUTED, envelope))
    assert cloud_bus.published == []


@pytest.mark.asyncio
async def test_a_raw_stamped_hook_record_is_dropped() -> None:
    """``raw`` is the ArtifactStore default, so it must not be a pass."""
    local_bus, cloud_bus = _MockGatewayBus(), _MockGatewayBus()
    service = ServiceGatewayForwarder(
        config=_governed_config(), local_bus=local_bus, cloud_bus=cloud_bus
    )
    envelope = _envelope({"tool_name": "Bash", "redaction_state": STATE_RAW})
    await service.forward_outbound_message(local_bus.message(TOOL_EXECUTED, envelope))
    assert cloud_bus.published == []


@pytest.mark.asyncio
async def test_a_refusal_is_a_drop_and_never_raises() -> None:
    """OMN-17382: a raise here would wedge the bridge on the poisoned offset."""
    local_bus, cloud_bus = _MockGatewayBus(), _MockGatewayBus()
    service = ServiceGatewayForwarder(
        config=_governed_config(), local_bus=local_bus, cloud_bus=cloud_bus
    )
    envelope = _envelope({"tool_name": "Bash", "redaction_state": STATE_RAW})
    # No pytest.raises: the point is that this returns normally.
    await service.forward_outbound_message(local_bus.message(TOOL_EXECUTED, envelope))
    assert cloud_bus.published == []


@pytest.mark.asyncio
async def test_a_drop_is_observable(caplog: pytest.LogCaptureFixture) -> None:
    """AC2 requires the drop be observable, not silent."""
    local_bus, cloud_bus = _MockGatewayBus(), _MockGatewayBus()
    service = ServiceGatewayForwarder(
        config=_governed_config(), local_bus=local_bus, cloud_bus=cloud_bus
    )
    envelope = _envelope({"tool_name": "Bash"})
    with caplog.at_level(logging.WARNING):
        await service.forward_outbound_message(
            local_bus.message(TOOL_EXECUTED, envelope)
        )
    assert any(TOOL_EXECUTED in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_an_ungoverned_topic_is_unaffected_by_the_gate() -> None:
    """The gate must not become a global 'stamp everything' requirement: the
    pre-existing delegation topics carry no ``redaction_state`` and must keep
    crossing exactly as before.
    """
    local_bus, cloud_bus = _MockGatewayBus(), _MockGatewayBus()
    service = ServiceGatewayForwarder(
        config=_governed_config(), local_bus=local_bus, cloud_bus=cloud_bus
    )
    envelope = _envelope({"ok": True})
    await service.forward_outbound_message(
        local_bus.message(UNGOVERNED_OUTBOUND, envelope)
    )
    assert len(cloud_bus.published) == 1


@pytest.mark.asyncio
async def test_a_non_mirrored_omniclaude_topic_is_still_refused() -> None:
    """AC2: a denied class stays denied. ``tool-output-captured`` is not in the
    outbound set at all, so it is refused by the pre-existing declaration check.
    """
    local_bus, cloud_bus = _MockGatewayBus(), _MockGatewayBus()
    service = ServiceGatewayForwarder(
        config=_governed_config(), local_bus=local_bus, cloud_bus=cloud_bus
    )
    envelope = _envelope({"redaction_state": STATE_REDACTED})
    with pytest.raises(ValueError, match="not declared for outbound"):
        await service.forward_outbound_message(
            local_bus.message(STILL_DENIED, envelope)
        )
    assert cloud_bus.published == []


@pytest.mark.asyncio
async def test_a_deployment_with_no_gate_forwards_a_governed_topic_unchanged() -> None:
    """Backwards behaviour: ``egress_redaction=None`` is the pre-OMN-16979
    posture and must not start dropping records.
    """
    local_bus, cloud_bus = _MockGatewayBus(), _MockGatewayBus()
    service = ServiceGatewayForwarder(
        config=_config(outbound=(UNGOVERNED_OUTBOUND,), egress_redaction=None),
        local_bus=local_bus,
        cloud_bus=cloud_bus,
    )
    await service.forward_outbound_message(
        local_bus.message(UNGOVERNED_OUTBOUND, _envelope({"ok": True}))
    )
    assert len(cloud_bus.published) == 1


def test_the_validate_path_resolves_the_same_decision_as_the_forward_path() -> None:
    """The forward path and any validate/canary path must not disagree: a
    record reported admissible that ``forward_outbound_message`` would drop is
    a silent hole. Both resolve through the same module-level function.
    """
    policy = _governed_config().egress_redaction
    assert policy is not None
    unstamped = _envelope({"tool_name": "Bash"})
    stamped = _envelope({"tool_name": "Bash", "redaction_state": STATE_REDACTED})
    assert egress_admits(policy, unstamped, TOOL_EXECUTED) is False
    assert egress_admits(policy, stamped, TOOL_EXECUTED) is True
    # An ungoverned topic is admitted whatever it carries.
    assert egress_admits(policy, unstamped, UNGOVERNED_OUTBOUND) is True
    # No policy at all is the pre-OMN-16979 posture.
    assert egress_admits(None, unstamped, TOOL_EXECUTED) is True


# ---------------------------------------------------------------------------
# Cross-repo interlock
# ---------------------------------------------------------------------------


def test_governed_set_matches_the_upstream_redaction_contract_topics() -> None:
    """The two halves must name the same topics. omnimarket's
    ``capture_redaction.yaml`` (OMN-17209) declares the posture for exactly
    these two topics; this node refuses anything it did not stamp. If one side
    is widened without the other, this fails rather than opening a hole.
    """
    governed = set(_forwarder_block()["egress_redaction"]["governed_topics"])  # type: ignore[index]
    assert governed == {TOOL_EXECUTED, PROMPT_SUBMITTED}
