# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16812 — a boundary that ACKs a failure must terminalize it.

OMN-16798 closed "consumes and commits with NO effect" for the RECORD: a routing
handler that raises is DLQ'd, and an offset is never committed over a record
that landed nowhere durable. On the ``.201`` dev lane that guard fired correctly
and in milliseconds::

    error_type=HandlerDispatchFailureError
    error=... ProtocolConfigurationError: [ONEX_CORE_041_INVALID_CONFIGURATION]
          No tier has a configured endpoint for task_type='agent_delegation' ...
    metric_name=boundary_swallow_prevented dlq_routed=true dlq_enabled=True

...and the CALLER then waited 120.0 s, twice, on two revisions, to be told::

    {"ok": false, "error": {"code": "dispatch_timeout", "retryable": true}}

The record was safe and the request was abandoned in silence. The runtime knew
the class before the caller finished its first second of waiting and threw that
knowledge away; ``retryable: true`` then instructed the caller to keep doing it.

WHY THESE TESTS ARE AT THE REAL SEAM. They drive ``wire_from_manifest`` with a
real ``MessageDispatchEngine`` and a real ``EventBusInmemory``, publish the
verbatim wire record off the live topic, and assert on what reaches the bus —
the DLQ record AND the terminal, on one raise. The same reason the OMN-16798
suite exists: an isolation test that calls the handler directly passes through
this entire outage, because the defect is not in the handler.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast
from unittest.mock import patch
from uuid import UUID

import pytest
from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.event_bus.topic_constants import get_dlq_topic_for_original
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _BOUNDARY_DLQ_ENV,
    BoundaryDlqNotPersistedError,
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
from omnibase_infra.runtime.boundary_failure_terminal import (
    ModelBoundaryFailureTerminal,
    classify_boundary_failure,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

_THIS_MODULE = "tests.unit.runtime.auto_wiring.test_omn16812_boundary_failure_terminal"

# The live topics and correlation from the OMN-16812 reproduction.
_SUBSCRIBE_TOPIC = "onex.cmd.omnibase-infra.delegation-routing-request.v1"  # onex-topic-allow: verbatim from the live incident trace
_DECISION_TOPIC = "onex.evt.omnibase-infra.routing-decision.v1"  # onex-topic-allow: verbatim from the live incident trace
_FAILURE_TOPIC = "onex.evt.omnibase-infra.delegation-failed.v1"  # onex-topic-allow: verbatim from the live incident trace
_CORRELATION = "7a300827-1000-4000-8000-000000000012"
_LIVE_ONEX_CODE = "ONEX_CORE_041_INVALID_CONFIGURATION"

_PATCH_IMPORT_HANDLER = (
    "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class"
)


# ---------------------------------------------------------------------------
# Field-for-field mirrors of the omnimarket / omnibase_core production shapes
# ---------------------------------------------------------------------------


class ModelMirrorDelegationRequest(BaseModel):
    """Mirror of ``ModelDelegationRequest`` — the inner domain payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt: str
    task_type: str
    correlation_id: UUID
    max_tokens: int = 2048


class ModelMirrorRoutingIntent(BaseModel):
    """Mirror of ``omnibase_core.models.delegation.wire.ModelRoutingIntent``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    intent: str = Field(default="routing_reducer")
    payload: ModelMirrorDelegationRequest
    min_tier_name: str | None = None
    excluded_backend_refs: tuple[str, ...] = Field(default_factory=tuple)


class ModelMirrorRoutingDecision(BaseModel):
    """Mirror of the ``ModelRoutingDecision`` the reducer publishes when it works."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    selected_model: str
    correlation_id: UUID


def _raise_live_routing_failure() -> None:
    """Raise the exact exception the .201 lane raised, with its real code."""
    context = ModelInfraErrorContext.with_correlation(
        transport_type=EnumInfraTransportType.HTTP,
        operation="resolve_routing_endpoint",
    )
    raise ProtocolConfigurationError(
        "No tier has a configured endpoint for task_type='agent_delegation'",
        context=context,
    )


class HandlerMirrorRoutingIntentRaises:
    """Mirror of ``HandlerRoutingIntent`` on the day it raised.

    The engine's catch-all swallows this into a FAILED ``ModelDispatchResult``,
    which is why the boundary sees ``HandlerDispatchFailureError`` and not the
    ``ProtocolConfigurationError`` itself — the whole reason the attribution has
    to read the flattened message.
    """

    def handle(self, intent: ModelMirrorRoutingIntent) -> ModelMirrorRoutingDecision:
        _raise_live_routing_failure()
        raise AssertionError("unreachable")  # pragma: no cover - defensive


# ---------------------------------------------------------------------------
# Contract construction — declares BOTH terminals, as the fix requires
# ---------------------------------------------------------------------------

# ``terminal_events`` is the declaration ``_declared_failure_terminal_topics``
# reads (through the same single reader the Pattern B broker's subscription set
# is built from). Written to a real file so the resolution under test is the one
# production performs.
_CONTRACT_YAML = f"""
name: "node_delegation_routing_reducer_mirror"
node_type: "REDUCER_GENERIC"
terminal_events:
  success: "{_DECISION_TOPIC}"
  failure: "{_FAILURE_TOPIC}"
event_bus:
  subscribe_topics:
    - "{_SUBSCRIBE_TOPIC}"
  publish_topics:
    - "{_DECISION_TOPIC}"
    - "{_FAILURE_TOPIC}"
published_events:
  - event_type: "MirrorRoutingDecision"
    topic: "{_DECISION_TOPIC}"
    description: "Routing decision emitted by the mirrored reducer."
"""


def _contract(contract_path: Path) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name="node_delegation_routing_reducer_mirror",
        node_type="REDUCER_GENERIC",
        contract_version=ModelContractVersion(major=0, minor=3, patch=0),
        contract_path=contract_path,
        entry_point_name="node_delegation_routing_reducer_mirror",
        package_name="omnimarket",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(_SUBSCRIBE_TOPIC,),
            publish_topics=(_DECISION_TOPIC, _FAILURE_TOPIC),
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="operation_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name="HandlerMirrorRoutingIntentRaises", module=_THIS_MODULE
                    ),
                    event_model=ModelHandlerRef(
                        name="ModelMirrorRoutingIntent", module=_THIS_MODULE
                    ),
                    operation="delegation_routing",
                ),
            ),
        ),
    )


def _routing_intent_wire() -> dict[str, object]:
    """The ModelRoutingIntent as published on the live lane."""
    return {
        "intent": "routing_reducer",
        "payload": {
            "prompt": "Reply with the single word: alive.",
            "task_type": "agent_delegation",
            "correlation_id": _CORRELATION,
            "max_tokens": 32,
        },
        "min_tier_name": None,
        "excluded_backend_refs": [],
    }


@pytest.fixture
def contract_path(tmp_path: Path) -> Path:
    path = tmp_path / "contract.yaml"
    path.write_text(_CONTRACT_YAML, encoding="utf-8")
    return path


class _DlqRecordingInmemoryBus(EventBusInmemory):
    """In-memory bus that honors the boundary's duck-typed DLQ contract.

    ``EventBusInmemory`` has no ``_publish_raw_to_dlq``; ``EventBusKafka`` does,
    and that method is the whole reason the boundary can preserve a record at
    all. Implementing it here as a real publish onto the topic's DLQ address
    puts BOTH effects of one raise — the DLQ record and the terminal — on the
    same observable bus, which is exactly the claim under test.

    ``dlq_persisted`` mirrors the Kafka method's return contract: ``False``
    means "did not durably persist" and makes the boundary withhold the offset.
    """

    def __init__(self, *, dlq_persisted: bool = True, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._dlq_persisted = dlq_persisted

    async def _publish_raw_to_dlq(
        self,
        *,
        original_topic: str,
        raw_msg: object,
        error: Exception,
        correlation_id: UUID,
        failure_type: str,
        consumer_group: str,
        dlq_topic: str,
    ) -> bool:
        if not self._dlq_persisted:
            return False
        record = ModelEventEnvelope[object](
            payload={
                "original_topic": original_topic,
                "failure_type": failure_type,
                "consumer_group": consumer_group,
                "error_type": type(error).__name__,
            },
            correlation_id=correlation_id,
            event_type="omnibase-infra.dlq",
        )
        await self.publish(
            dlq_topic, None, record.model_dump_json().encode("utf-8"), None
        )
        return True


async def _drive_one_failing_record(
    bus: EventBusInmemory,
    contract_path: Path,
    *,
    collect_topics: tuple[str, ...],
) -> dict[str, list[ModelEventEnvelope[object]]]:
    """Wire the contract for real, publish one record, return what each topic saw."""
    correlation_id = UUID(_CORRELATION)
    seen: dict[str, list[ModelEventEnvelope[object]]] = {t: [] for t in collect_topics}
    arrived: asyncio.Event = asyncio.Event()

    def _collector(topic: str):  # type: ignore[no-untyped-def]
        async def collect(message: ModelEventMessage) -> None:
            envelope = ModelEventEnvelope[object].model_validate_json(message.value)
            if envelope.correlation_id == correlation_id:
                seen[topic].append(envelope)
                arrived.set()

        return collect

    for topic in collect_topics:
        await bus.subscribe(
            topic, group_id=f"omn16812-{topic}", on_message=_collector(topic)
        )

    engine = MessageDispatchEngine()
    with patch(_PATCH_IMPORT_HANDLER, return_value=HandlerMirrorRoutingIntentRaises):
        await wire_from_manifest(
            ModelAutoWiringManifest(contracts=(_contract(contract_path),)),
            engine,
            event_bus=bus,
            environment="local",
        )
    engine.freeze()

    command = ModelEventEnvelope[object](
        payload=_routing_intent_wire(),
        correlation_id=correlation_id,
        event_type="omnibase-infra.delegation-routing-request",
    )
    await bus.publish(
        _SUBSCRIBE_TOPIC, None, command.model_dump_json().encode("utf-8"), None
    )
    # The boundary's own bounded retry runs before it gives up; wait on the
    # first arrival rather than a fixed sleep, then let the sibling publish land.
    try:
        await asyncio.wait_for(arrived.wait(), timeout=10)
    except TimeoutError:  # pragma: no cover - surfaced by the assertions below
        pass
    for _ in range(20):
        if all(seen[t] for t in collect_topics):
            break
        await asyncio.sleep(0.05)
    return seen


# ---------------------------------------------------------------------------
# AC1 + AC4 — RED before the fix: the raise produced a DLQ record and nothing else
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reducer_raise_emits_both_a_dlq_record_and_a_failure_terminal(
    contract_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One raise, two effects: the record is parked AND the caller is answered.

    Pre-fix this test fails on the terminal alone — the DLQ half has passed
    since OMN-16798, which is precisely why the outage survived that ticket.
    """
    monkeypatch.setenv(_BOUNDARY_DLQ_ENV, "1")
    dlq_topic = get_dlq_topic_for_original(_SUBSCRIBE_TOPIC)

    bus = _DlqRecordingInmemoryBus(environment="test", group="omn-16812-both")
    await bus.start()
    try:
        seen = await _drive_one_failing_record(
            bus, contract_path, collect_topics=(dlq_topic, _FAILURE_TOPIC)
        )
    finally:
        await bus.close()

    assert seen[dlq_topic], "OMN-16798's DLQ guard must keep working (regression)"
    assert seen[_FAILURE_TOPIC], (
        "no terminal on the contract's declared failure terminal — this is the "
        "120 s dispatch_timeout the caller saw on the .201 dev lane"
    )

    terminal = ModelBoundaryFailureTerminal.model_validate(
        seen[_FAILURE_TOPIC][0].payload
    )
    # AC1: correlation-exact, one terminal for one failed record.
    assert terminal.correlation_id == UUID(_CORRELATION)
    assert seen[_FAILURE_TOPIC][0].correlation_id == UUID(_CORRELATION)
    assert len(seen[_FAILURE_TOPIC]) == 1
    # AC2: the ORIGINATING class, not the boundary wrapper and not "timeout".
    assert terminal.failure_class == "ProtocolConfigurationError"
    assert terminal.failure_code == _LIVE_ONEX_CODE
    # AC3: a missing routing endpoint is not fixed by trying again.
    assert terminal.retryable is False
    assert terminal.status == "failed"
    assert terminal.origin_topic == _SUBSCRIBE_TOPIC


@pytest.mark.unit
@pytest.mark.asyncio
async def test_terminal_is_emitted_even_with_the_dlq_flag_off(
    contract_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whether the RECORD was preserved is orthogonal to whether the CALLER is told.

    ``ONEX_BOUNDARY_DLQ_ENABLED`` staged the record-preservation rollout. The
    offset advances on this path too, so the request is just as abandoned — and
    a caller left on a 120 s timeout is not a staged rollout of anything.
    """
    monkeypatch.delenv(_BOUNDARY_DLQ_ENV, raising=False)

    bus = _DlqRecordingInmemoryBus(environment="test", group="omn-16812-flagoff")
    await bus.start()
    try:
        seen = await _drive_one_failing_record(
            bus, contract_path, collect_topics=(_FAILURE_TOPIC,)
        )
    finally:
        await bus.close()

    assert seen[_FAILURE_TOPIC], "flag-off still ACKs the record; it must still answer"
    terminal = ModelBoundaryFailureTerminal.model_validate(
        seen[_FAILURE_TOPIC][0].payload
    )
    assert terminal.retryable is False
    assert terminal.failure_code == _LIVE_ONEX_CODE


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_terminal_when_the_offset_is_withheld(
    contract_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record that will be REDELIVERED must not be terminalized (negative control).

    The OMN-14936/OMN-16798 invariant: a DLQ write that did not durably persist
    raises so the offset stays put. The request is still live at that point, so
    a terminal published here would be a lie about a request that is about to be
    retried — and the retry would then emit a SECOND terminal for the same
    correlation.
    """
    monkeypatch.setenv(_BOUNDARY_DLQ_ENV, "1")

    bus = _DlqRecordingInmemoryBus(
        environment="test", group="omn-16812-withheld", dlq_persisted=False
    )
    await bus.start()
    try:
        seen = await _drive_one_failing_record(
            bus, contract_path, collect_topics=(_FAILURE_TOPIC,)
        )
    finally:
        await bus.close()

    assert not seen[_FAILURE_TOPIC], (
        "terminalized a record whose offset was withheld — the request is still "
        "live and will be redelivered"
    )


# ---------------------------------------------------------------------------
# AC3 at the CALLER's seam — the readers the ingress answer is built from
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_broker_reads_the_terminal_as_an_immediate_attributed_failure() -> None:
    """The caller-visible outcome: typed failure, attributed cause, retryable false.

    These three readers are the entire path from a terminal on the bus to the
    ``ModelLocalRuntimeIngressResponse.error`` the caller receives. Asserting on
    them is asserting on the answer, without booting a broker.
    """
    from omnibase_infra.runtime.contract_terminal_events import (
        resolve_terminal_retryable,
        resolve_terminal_verdict,
    )
    from omnibase_infra.runtime.runtime_local_ingress import (
        ModelRuntimeLocalIngressRoute,
    )
    from omnibase_infra.runtime.service_pattern_b_broker import (
        _status_for_terminal_topic,
        _terminal_error_message,
    )

    terminal = ModelBoundaryFailureTerminal(
        correlation_id=UUID(_CORRELATION),
        failure_class="ProtocolConfigurationError",
        failure_code=_LIVE_ONEX_CODE,
        retryable=False,
        failure_reason=(
            "HandlerDispatchFailureError: ProtocolConfigurationError: "
            f"[{_LIVE_ONEX_CODE}] No tier has a configured endpoint"
        ),
        origin_topic=_SUBSCRIBE_TOPIC,
    )
    # The broker decodes the envelope and hands the INNER body to its readers.
    body = terminal.model_dump(mode="json")

    assert resolve_terminal_verdict(body) is False
    assert resolve_terminal_retryable(body) is False

    route = ModelRuntimeLocalIngressRoute(
        contract_name="node_delegation_routing_reducer_mirror",
        node_name="node_delegation_routing_reducer_mirror",
        contract_path="contract.yaml",  # unread by the readers under test
        package_name="omnimarket",
        event_type="omnibase-infra.delegation-routing-request",
        command_topic=_SUBSCRIBE_TOPIC,
        terminal_event=_DECISION_TOPIC,
        terminal_events=(_DECISION_TOPIC, _FAILURE_TOPIC),
    )
    assert _status_for_terminal_topic(route, _FAILURE_TOPIC, body) == "failed"
    # AC2: the message the caller reads names the class, not "timed out".
    error_message = _terminal_error_message(body)
    assert error_message is not None
    assert _LIVE_ONEX_CODE in error_message


@pytest.mark.unit
def test_a_terminal_that_states_nothing_leaves_the_historical_derivation_alone() -> (
    None
):
    """``resolve_terminal_retryable`` is a read, never a guess (fail-closed)."""
    from omnibase_infra.runtime.contract_terminal_events import (
        resolve_terminal_retryable,
    )

    assert resolve_terminal_retryable({"status": "failed"}) is None
    assert resolve_terminal_retryable({"payload": {"status": "failed"}}) is None
    assert resolve_terminal_retryable(None) is None
    # An enveloped record: the flag lives in the body, not the envelope.
    assert resolve_terminal_retryable({"payload": {"retryable": False}}) is False
    assert resolve_terminal_retryable({"retryable": True}) is True


# ---------------------------------------------------------------------------
# The attribution itself
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_attribution_reads_through_the_engine_flattened_message() -> None:
    """The live shape: the real cause survives only as text inside the wrapper.

    ``MessageDispatchEngine.dispatch()`` catches the handler's exception and
    records it; the boundary is handed ``HandlerDispatchFailureError`` with no
    ``__cause__`` to walk. An attribution that reads only exception TYPES calls
    this ``HandlerDispatchFailureError``/retryable — which is the defect.
    """
    from omnibase_infra.runtime.auto_wiring.handler_wiring import (
        HandlerDispatchFailureError,
    )

    exc = HandlerDispatchFailureError(
        f"dispatch to topic={_SUBSCRIBE_TOPIC} returned status=handler_error with "
        "no terminal output (dispatcher_id=routing): Dispatcher 'routing' failed: "
        f"ProtocolConfigurationError: [{_LIVE_ONEX_CODE}] No tier has a configured "
        "endpoint for task_type='agent_delegation'",
        failure_code="ONEX_CORE_030_HANDLER_EXECUTION_ERROR",
    )

    terminal = classify_boundary_failure(
        exc,
        topic=_SUBSCRIBE_TOPIC,
        correlation_id=UUID(_CORRELATION),
        failure_reason=str(exc),
        failure_code=exc.failure_code,
    )

    assert terminal.failure_class == "ProtocolConfigurationError"
    # The SPECIFIC code wins over the engine's generic HANDLER_EXECUTION_ERROR.
    assert terminal.failure_code == _LIVE_ONEX_CODE
    assert terminal.retryable is False


@pytest.mark.unit
def test_attribution_walks_a_real_exception_chain() -> None:
    """The directly-raised shape, where the cause IS on the chain."""
    try:
        try:
            _raise_live_routing_failure()
        except ProtocolConfigurationError as inner:
            raise RuntimeError("boundary re-raise") from inner
    except RuntimeError as exc:
        terminal = classify_boundary_failure(
            exc,
            topic=_SUBSCRIBE_TOPIC,
            correlation_id=UUID(_CORRELATION),
            failure_reason="boundary re-raise",
        )

    assert terminal.failure_class == "RuntimeError"
    assert terminal.failure_code == _LIVE_ONEX_CODE
    assert terminal.retryable is False


@pytest.mark.unit
def test_a_transient_failure_stays_retryable() -> None:
    """Contrast case — the classifier must not answer False to everything.

    A blanket ``retryable: false`` would be as wrong as the blanket ``true``
    this ticket is about: it would tell a caller to give up on a failure that a
    retry would have cleared.
    """
    terminal = classify_boundary_failure(
        TimeoutError("upstream did not answer in time"),
        topic=_SUBSCRIBE_TOPIC,
        correlation_id=UUID(_CORRELATION),
        failure_reason="TimeoutError: upstream did not answer in time",
    )

    assert terminal.retryable is True
    assert terminal.failure_class == "TimeoutError"
    assert terminal.failure_code is None


@pytest.mark.unit
def test_ordinary_prose_in_a_message_is_not_mistaken_for_a_failure_class() -> None:
    """The token scan is anchored on the Error/Exception suffix for a reason.

    A "any CapWord" scan would mint a failure class out of the sentence and, if
    the sentence happened to contain a non-retryable name, flip ``retryable``
    on prose alone.
    """
    terminal = classify_boundary_failure(
        RuntimeError("Redpanda broker Valkey Postgres unreachable during Dispatch"),
        topic=_SUBSCRIBE_TOPIC,
        correlation_id=UUID(_CORRELATION),
        failure_reason="unreachable",
    )

    assert terminal.failure_class == "RuntimeError"
    assert terminal.retryable is True


@pytest.mark.unit
def test_dlq_not_persisted_error_is_importable_for_the_withheld_control() -> None:
    """Guards the negative control above against a silent rename."""
    assert issubclass(BoundaryDlqNotPersistedError, Exception)


@pytest.mark.unit
def test_declared_failure_terminal_topics_resolves_the_mirror_contract(
    contract_path: Path,
) -> None:
    """The address the boundary answers at comes from the CONTRACT, not a constant."""
    from omnibase_infra.runtime.auto_wiring.handler_wiring import (
        _declared_failure_terminal_topics,
    )

    resolved = _declared_failure_terminal_topics(
        _contract(contract_path), success_topic=_DECISION_TOPIC
    )
    assert resolved == (_FAILURE_TOPIC,)
    # An undeclared topic is never a valid answer address: publishing there
    # would break the contract's own publish allowlist.
    assert cast("tuple[str, ...]", resolved) != ()
