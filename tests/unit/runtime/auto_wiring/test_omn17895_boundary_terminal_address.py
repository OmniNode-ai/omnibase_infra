# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17895 — the last-resort terminal address is only legitimate for a COMMAND.

WHAT WAS MEASURED (dev lane ``omnibase-infra``, 2026-09-04T13:45Z, read-only).
``onex.evt.omnimarket.swarm-fanout-completed.v1`` held 11,340 records and **not
one** of them was a ``ModelSwarmFanoutResult``. Every record was a
``ModelBoundaryFailureTerminal`` describing a failure on a DIFFERENT topic —
``origin_topic=onex.evt.omnimarket.delegation-escalation-triggered.v1`` — 12 of
them published directly by the auto-wiring boundary
(``source_tool="auto-wiring-boundary"``) per two-hourly cycle and the other
11,328 re-injected by ``node_dlq_replay_effect``.

HOW THE BOUNDARY GOT THAT ADDRESS. ``node_swarm_fanout_orchestrator`` subscribes
``onex.evt.omnimarket.delegation-escalation-triggered.v1`` but registers a
dispatcher only for ``operation: fanout``, so the record fails with
``failure_class=no_dispatcher``. Its ``contract.yaml`` declares no failure
terminal distinct from its success terminal, so
``_resolve_boundary_terminal_answer_topic`` took the OMN-17432 last-resort
branch and answered on the contract's own SUCCESS topic.

WHY THAT ADDRESS IS WRONG FOR AN EVENT. OMN-17432's justification is that the
success terminal is "by construction, where the caller is already listening" —
which is a statement about a CALLER, and a caller exists only for a COMMAND. The
publisher of an ``onex.evt.*`` notification holds no correlation open and awaits
no answer; it has already moved on. Answering there does not reach anyone, and
it lands a ``ModelBoundaryFailureTerminal`` on a topic whose declared payload
model is something else entirely — here two live consumers
(``node_swarm_subtask_state_reducer``, ``node_swarm_dispatch_orchestrator``) do
hold a dispatcher for ``event_type=omnimarket.swarm-fanout-completed`` and
``model_validate`` against ``ModelSwarmFanoutResult``, so each rejected it and
re-DLQ'd it. That is the same "a terminal is an answer, not a request; nobody is
holding a correlation open behind it" reasoning the sibling guard
``_is_boundary_failure_terminal_record`` already applies (OMN-17432), applied one
step earlier.

WHAT IS DELIBERATELY UNCHANGED. OMN-17432's own motivating route is a COMMAND
(``onex.cmd.omnibase-infra.gateway-detach-request.v1`` — every subscribe topic on
``node_gateway_attach_effect`` is ``onex.cmd.*``), so the 503 it fixed stays
fixed; that is asserted below rather than assumed. A contract that DECLARES a
distinct failure terminal keeps answering there whatever the consumed category
is (OMN-16812) — a declared failure terminal is contract-sanctioned to carry
failures, an inferred success terminal is not.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import UUID, uuid4

import pytest
import yaml
from pydantic import BaseModel, ConfigDict

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _resolve_boundary_terminal_answer_topic,
    wire_from_manifest,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

_THIS_MODULE = "tests.unit.runtime.auto_wiring.test_omn17895_boundary_terminal_address"

# The exact live pair, quoted from the wire read in the ticket.
_LIVE_CONSUMED_EVENT = "onex.evt.omnimarket.delegation-escalation-triggered.v1"  # onex-topic-allow: quotes the measured live record
_LIVE_SUCCESS_TERMINAL = "onex.evt.omnimarket.swarm-fanout-completed.v1"  # onex-topic-allow: quotes the measured live record

_GATEWAY_CONTRACT_PATH = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_gateway_attach_effect"
    / "contract.yaml"
)


# ---------------------------------------------------------------------------
# Band 1 — the resolver, on the exact live topic pair
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_an_event_subscription_never_resolves_to_the_contracts_success_terminal() -> (
    None
):
    """RED before the fix: this is the address 11,340 dev records were sent to."""
    assert (
        _resolve_boundary_terminal_answer_topic(
            failure_terminal_topics=(),
            terminal_answer_topic=_LIVE_SUCCESS_TERMINAL,
            consumed_topic=_LIVE_CONSUMED_EVENT,
        )
        is None
    ), (
        "the boundary answered a failed EVENT consumption on the contract's own "
        "success terminal — no caller awaits there, and the topic's declared "
        "payload model is not ModelBoundaryFailureTerminal"
    )


@pytest.mark.unit
def test_an_intent_subscription_is_refused_on_the_same_grounds() -> None:
    """``onex.intent.*`` is a notification too — no correlation is held open."""
    assert (
        _resolve_boundary_terminal_answer_topic(
            failure_terminal_topics=(),
            terminal_answer_topic="onex.evt.omnimarket.intent-completed.v1",  # onex-topic-allow: synthetic, test-only
            consumed_topic="onex.intent.omnimarket.plan-requested.v1",  # onex-topic-allow: synthetic, test-only
        )
        is None
    )


@pytest.mark.unit
def test_a_command_subscription_still_gets_the_last_resort_answer() -> None:
    """OMN-17432 unchanged where it was filed: a command has a waiting caller."""
    assert (
        _resolve_boundary_terminal_answer_topic(
            failure_terminal_topics=(),
            terminal_answer_topic="onex.evt.omnibase-infra.gateway-session.v1",  # onex-topic-allow: the real gateway answer topic
            consumed_topic="onex.cmd.omnibase-infra.gateway-detach-request.v1",  # onex-topic-allow: the real gateway command topic
        )
        == "onex.evt.omnibase-infra.gateway-session.v1"
    )


@pytest.mark.unit
def test_a_declared_failure_terminal_still_answers_an_event_failure() -> None:
    """OMN-16812 unchanged: a DECLARED failure terminal is contract-sanctioned.

    The category restriction bites only the INFERRED address. A contract that
    names a topic for its failures has said, in the contract, that failures
    belong there.
    """
    assert (
        _resolve_boundary_terminal_answer_topic(
            failure_terminal_topics=(
                "onex.evt.omnimarket.swarm-fanout-failed.v1",
            ),  # onex-topic-allow: synthetic, test-only
            terminal_answer_topic=_LIVE_SUCCESS_TERMINAL,
            consumed_topic=_LIVE_CONSUMED_EVENT,
        )
        == "onex.evt.omnimarket.swarm-fanout-failed.v1"
    )


@pytest.mark.unit
def test_the_gateway_contract_this_rule_must_not_regress_is_all_commands() -> None:
    """The premise of the exemption, read off the REAL contract, not asserted.

    If a future edit gives ``node_gateway_attach_effect`` an ``onex.evt.*``
    subscription, this test fails and the exemption above must be re-argued
    rather than silently narrowing OMN-17432's fix.
    """
    raw = yaml.safe_load(_GATEWAY_CONTRACT_PATH.read_text(encoding="utf-8"))
    subscribe = [str(t) for t in raw["event_bus"]["subscribe_topics"]]
    assert subscribe
    assert all(t.startswith("onex.cmd.") for t in subscribe), subscribe


# ---------------------------------------------------------------------------
# Band 2 — the same claim at the real boundary seam
# ---------------------------------------------------------------------------


class SwarmFanoutSubscriptionError(Exception):
    """Stands in for whatever makes the record undispatchable on the live lane.

    Live it is ``failure_class=no_dispatcher``; the address defect is identical
    for any boundary failure, and a raise reaches ``_route_swallowed_exception``
    by the same path (``_raise_if_no_dispatcher_drop`` raises
    ``HandlerDispatchFailureError`` into it).
    """


class ModelMirrorEscalationEvent(BaseModel):
    """Mirror of the escalation event shape measured on the wire."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_type: str
    model_id: str
    escalation_reason: str


class ModelMirrorFanoutResult(BaseModel):
    """Mirror of the SUCCESS model the terminal topic actually declares."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dispatches: int
    wall_latency_ms: int


class HandlerMirrorFanoutRaises:
    def handle(self, event: ModelMirrorEscalationEvent) -> ModelMirrorFanoutResult:
        raise SwarmFanoutSubscriptionError(f"no dispatcher for {event.task_type}")


def _fanout_shaped_contract(contract_path: Path) -> ModelDiscoveredContract:
    """A contract with the exact declaration shape that produced the storm.

    One ``.evt.`` subscription, one publish topic which is also the declared
    ``terminal_event``, and no failure terminal distinct from it.
    """
    return ModelDiscoveredContract(
        name="node_omn17895_fanout_shaped",
        node_type="ORCHESTRATOR_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=contract_path,
        entry_point_name="node_omn17895_fanout_shaped",
        package_name="omnibase_infra",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(_LIVE_CONSUMED_EVENT,),
            publish_topics=(_LIVE_SUCCESS_TERMINAL,),
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="operation_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name="HandlerMirrorFanoutRaises", module=_THIS_MODULE
                    ),
                    event_model=ModelHandlerRef(
                        name="ModelMirrorEscalationEvent", module=_THIS_MODULE
                    ),
                    operation="fanout",
                ),
            ),
        ),
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_terminal_lands_on_the_success_topic_for_a_failed_event(
    tmp_path: Path,
) -> None:
    """The wire claim: nothing with a failure-terminal payload on that topic.

    RED before the fix — a ``ModelBoundaryFailureTerminal`` arrives here and
    ``ModelMirrorFanoutResult.model_validate`` on it raises exactly the live
    ``dispatches: Field required; wall_latency_ms: Field required``.
    """
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(
        yaml.safe_dump(
            {
                "event_bus": {
                    "subscribe_topics": [_LIVE_CONSUMED_EVENT],
                    "publish_topics": [_LIVE_SUCCESS_TERMINAL],
                },
                "terminal_event": _LIVE_SUCCESS_TERMINAL,
                "terminal_events": [_LIVE_SUCCESS_TERMINAL],
            }
        ),
        encoding="utf-8",
    )

    correlation_id = uuid4()
    seen: list[ModelEventEnvelope[object]] = []
    arrived = asyncio.Event()

    async def _collect(message: ModelEventMessage) -> None:
        envelope = ModelEventEnvelope[object].model_validate_json(message.value)
        if envelope.correlation_id == correlation_id:
            seen.append(envelope)
            arrived.set()

    bus = EventBusInmemory(environment="test", group="omn-17895-fanout")
    await bus.start()
    try:
        await bus.subscribe(
            _LIVE_SUCCESS_TERMINAL, group_id="omn17895-downstream", on_message=_collect
        )

        engine = MessageDispatchEngine()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
                lambda *a, **k: HandlerMirrorFanoutRaises,
            )
            await wire_from_manifest(
                ModelAutoWiringManifest(
                    contracts=(_fanout_shaped_contract(contract_path),)
                ),
                engine,
                event_bus=bus,
                environment="local",
            )
        engine.freeze()

        event = ModelEventEnvelope[object](
            payload={
                "task_type": "test",
                "model_id": "qwen3.8",
                "escalation_reason": "provider HTTP 404 Not Found",
            },
            correlation_id=correlation_id,
            event_type="omnimarket.delegation-escalation-triggered",
        )
        await bus.publish(
            _LIVE_CONSUMED_EVENT, None, event.model_dump_json().encode("utf-8"), None
        )
        try:
            await asyncio.wait_for(arrived.wait(), timeout=5)
        except TimeoutError:
            pass
    finally:
        await bus.close()

    assert seen == [], (
        "a boundary failure terminal was published onto the contract's own "
        "success terminal for a failed EVENT consumption — this is the record "
        "shape that filled onex.evt.omnimarket.swarm-fanout-completed.v1 with "
        "11,340 undispatchable records: "
        f"{[e.model_dump(mode='json').get('payload_type') for e in seen]}"
    )


@pytest.mark.unit
def test_the_success_model_rejects_a_failure_terminal_payload() -> None:
    """Why landing there is not merely untidy: the consumer cannot parse it.

    Reproduces the live validation error verbatim, so the cost of the wrong
    address is stated in the test rather than in a comment.
    """
    from pydantic import ValidationError

    from omnibase_infra.runtime.boundary_failure_terminal import (
        ModelBoundaryFailureTerminal,
    )

    terminal = ModelBoundaryFailureTerminal(
        correlation_id=UUID(int=1),
        failure_class="HandlerDispatchFailureError",
        retryable=True,
        failure_reason="no dispatcher for omnimarket.delegation-escalation-triggered",
        origin_topic=_LIVE_CONSUMED_EVENT,
    )
    with pytest.raises(ValidationError) as excinfo:
        ModelMirrorFanoutResult.model_validate(terminal.model_dump(mode="json"))
    rendered = str(excinfo.value)
    assert "dispatches" in rendered
    assert "wall_latency_ms" in rendered
