# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17372 — the two producer-side residuals of the keyless-customer refusal.

AC2 landed and is proven live on ``onex-dev``: a tenant with a valid API key and
no registered provider key gets a delegation terminal carrying
``failure_class=CustomerKeyRefusedError``,
``failure_code=ONEX_MARKET_CUSTOMER_PROVIDER_KEY_ABSENT`` and a remediation. The
STRUCTURED half of that answer is correct. Two things the same live response
proved wrong are fixed here, both at the producer::

    "terminal_failure_class": "CustomerKeyRefusedError",
    "terminal_failure_code": "ONEX_MARKET_CUSTOMER_PROVIDER_KEY_ABSENT",
    "terminal_failure_reason": "HandlerDispatchFailureError: dispatch to
        topic=onex.cmd.omnibase-infra.delegation-routing-request.v1 returned
        status=handler_error ... CustomerKeyRefusedError:
        [ONEX_MARKET_CUSTOMER_PROVIDER_KEY_ABSENT]
        delegation.customer_provider_key.absent: delegation refused for tenant
        'operator-ledger-probe' (task_type='summarization', surface=cloud): no
        provider key is registered f... [truncated]"

**R1 — the refusal was published ``retryable=true``.**
``classify_boundary_failure`` derives retryability from
:class:`EnumNonRetryableErrorCategory`, and that enum did not name
``CustomerKeyRefusedError``. A caller obeying ``retryable`` therefore retries a
condition no retry can fix: the tenant has no provider key, and only the tenant
registering one changes that. The fix goes through the enum the classifier
already keys on — not a special case bolted next to it — so DLQ replay, the
event bus and the boundary keep having ONE answer to "is this worth retrying".

**R2 — the customer-facing reason was an internal dispatch trace, cut mid-word.**
The string above is ``HandlerDispatchFailureError``'s own message: it names the
Kafka topic and the dispatcher id (internal topology, on a customer response
body), spends its 500-character budget on that trace, and then truncates the
one sentence the customer needed — the refusal's remediation is gone. The cause
is only ever text by the time the boundary sees it: the engine's catch-all
flattened the real exception into ``error_message`` long before. So the fix
unwraps that text at the attribution seam, where the FULL wrapper message is
still in hand, and publishes the refusal's own sentence as the reason. The
wrapper is not discarded — it stays on the boundary's own log line, which is
where a topic name and a dispatcher id belong.

The gateway (``omninode_infra#1187``) reads the producer's ``failure_reason``
verbatim, so it needs no edit for either half.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import pytest
from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumNonRetryableErrorCategory
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    HandlerDispatchFailureError,
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
from omnibase_infra.utils.util_error_sanitization import sanitize_error_message

_THIS_MODULE = "tests.unit.runtime.auto_wiring.test_omn17372_refusal_producer_residuals"

# Verbatim from the live 2026-09-06 proof (workflow c98636f9-1a41-4fce-9fb6-ebbdee4e71d9).
_SUBSCRIBE_TOPIC = "onex.cmd.omnibase-infra.delegation-routing-request.v1"  # onex-topic-allow: verbatim from the live OMN-17372 AC2 trace
_DECISION_TOPIC = "onex.evt.omnibase-infra.routing-decision.v1"  # onex-topic-allow: verbatim from the live OMN-17372 AC2 trace
_FAILURE_TOPIC = "onex.evt.omnibase-infra.delegation-failed.v1"  # onex-topic-allow: verbatim from the live OMN-17372 AC2 trace
_CORRELATION = "8c53ee9e-6f1f-4c13-b54a-8249e8024cff"
_DISPATCHER_ID = (
    "dispatcher.auto.node_delegation_routing_reducer.HandlerRoutingIntent."
    "delegation_routing_a2c2bde5"
)

# ``omnimarket.routing.customer_key_terminus``'s pinned customer contract.
_REFUSAL_CLASS = "CustomerKeyRefusedError"
_REFUSAL_ONEX_CODE = "ONEX_MARKET_CUSTOMER_PROVIDER_KEY_ABSENT"

# ``ModelCustomerKeyRefusal.boundary_message`` as the live tenant produced it.
# THIS is what a customer must read. It is a complete, actionable sentence that
# names no topic, no dispatcher and no internal class.
_REFUSAL_BOUNDARY_MESSAGE = (
    f"[{_REFUSAL_ONEX_CODE}] delegation.customer_provider_key.absent: "
    "delegation refused for tenant 'operator-ledger-probe' "
    "(task_type='summarization', surface=cloud): no provider key is registered "
    "for this tenant. Register a provider key for this tenant and retry."
)

_PATCH_IMPORT_HANDLER = (
    "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class"
)


class CustomerKeyRefusedError(Exception):
    """Field-for-field mirror of ``omnimarket.routing.customer_key_terminus``'s.

    ``omnibase_infra`` may not import ``omnimarket`` (layering: infra is BELOW
    the node packages), so the class the live refusal raises is mirrored here.
    Both halves that matter to the boundary are mirrored exactly: the class
    NAME, which is what ``EnumNonRetryableErrorCategory`` keys on, and the
    ``error_code`` attribute plus the ``[CODE] …`` message lead, which is what
    ``_first_onex_code`` reads.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.error_code = _REFUSAL_ONEX_CODE


def _engine_flattened_refusal() -> HandlerDispatchFailureError:
    """The exception the CONSUME BOUNDARY actually holds, built the live way.

    ``MessageDispatchEngine`` catches the refusal, records
    ``sanitize_error_message(exc)`` into ``error_message``, and returns a FAILED
    result; ``_raise_if_silent_dispatch_failure`` then wraps that string. The
    refusal object never reaches the boundary, so there is no ``__cause__`` to
    walk — reproduced here rather than asserted about.
    """
    engine_error_message = (
        f"Dispatcher {_DISPATCHER_ID!r} failed: "
        f"{sanitize_error_message(CustomerKeyRefusedError(_REFUSAL_BOUNDARY_MESSAGE))}"
    )
    return HandlerDispatchFailureError(
        f"dispatch to topic={_SUBSCRIBE_TOPIC} returned status=handler_error "
        f"with no terminal output (dispatcher_id={_DISPATCHER_ID}): "
        f"{engine_error_message}",
        failure_code="ONEX_CORE_030_HANDLER_EXECUTION_ERROR",
    )


def _classify_the_live_refusal() -> ModelBoundaryFailureTerminal:
    """Classify the flattened refusal exactly as the boundary classifies it.

    ``failure_reason`` is ``sanitize_error_message(exc)`` because that is the
    literal argument ``_route_swallowed_exception`` passes — the already
    truncated wrapper string, which is the whole of R2.
    """
    exc = _engine_flattened_refusal()
    return classify_boundary_failure(
        exc,
        topic=_SUBSCRIBE_TOPIC,
        correlation_id=UUID(_CORRELATION),
        failure_reason=sanitize_error_message(exc),
        failure_code=exc.failure_code,
    )


# ---------------------------------------------------------------------------
# R1 — a missing customer key is not retryable
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_enum_the_classifier_keys_on_names_the_refusal() -> None:
    """The fix belongs IN the enum, not beside it.

    ``classify_boundary_failure``, DLQ replay's ``NON_RETRYABLE_ERRORS`` and the
    event bus all resolve retry eligibility through this one classmethod. A
    special case in the classifier would give the runtime two answers to one
    question and leave a replayed DLQ record retrying the same refusal.
    """
    assert EnumNonRetryableErrorCategory.is_non_retryable(_REFUSAL_CLASS) is True
    assert _REFUSAL_CLASS in EnumNonRetryableErrorCategory.get_all_values()


@pytest.mark.unit
def test_the_enum_still_says_retryable_for_a_transient_class() -> None:
    """Positive control for the assertion above — the enum is not now all-True."""
    assert (
        EnumNonRetryableErrorCategory.is_non_retryable("InfraConnectionError") is False
    )
    assert EnumNonRetryableErrorCategory.is_non_retryable("TimeoutError") is False


@pytest.mark.unit
def test_the_keyless_refusal_is_published_not_retryable() -> None:
    """R1 at the seam that publishes it.

    RED before the fix: ``retryable is True``, because the only class names
    observable from the flattened wrapper are ``HandlerDispatchFailureError``
    (a boundary wrapper, deliberately never in the enum) and
    ``CustomerKeyRefusedError`` (which the enum did not name).
    """
    terminal = _classify_the_live_refusal()

    assert terminal.failure_class == _REFUSAL_CLASS
    assert terminal.failure_code == _REFUSAL_ONEX_CODE
    assert terminal.retryable is False, (
        "the customer has no provider key; retrying cannot register one"
    )


# ---------------------------------------------------------------------------
# R2 — the customer reads the refusal, not the dispatch trace
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_customer_facing_reason_is_the_refusals_own_sentence() -> None:
    """R2. The reason must be the refusal's message, whole, and nothing else.

    RED before the fix: the reason is the ``HandlerDispatchFailureError``
    wrapper, cut at 500 characters part-way through "registered f", so the
    remediation sentence the customer needed never arrives.
    """
    terminal = _classify_the_live_refusal()

    assert terminal.failure_reason == _REFUSAL_BOUNDARY_MESSAGE


@pytest.mark.unit
def test_the_customer_facing_reason_leaks_no_internal_topology() -> None:
    """The wrapper named a Kafka topic and a dispatcher id on a customer body."""
    reason = _classify_the_live_refusal().failure_reason

    assert "HandlerDispatchFailureError" not in reason
    assert _SUBSCRIBE_TOPIC not in reason
    assert _DISPATCHER_ID not in reason
    assert "dispatcher_id" not in reason
    assert "[truncated]" not in reason
    # The half a customer acts on, which truncation had removed entirely.
    assert reason.endswith("Register a provider key for this tenant and retry.")


@pytest.mark.unit
def test_the_wrapper_is_kept_on_the_boundarys_log_line() -> None:
    """Unwrapping is not discarding: the dispatch trace stays where it belongs.

    ``_route_swallowed_exception`` logs the full sanitized wrapper — topic,
    dispatcher id and all — at ``ERROR`` against the same correlation, before it
    terminalizes. That log line is the operator's copy; the terminal is the
    customer's. This asserts the operator's copy is not the thing being removed.
    """
    exc = _engine_flattened_refusal()
    wrapper = sanitize_error_message(exc)

    assert wrapper.startswith("HandlerDispatchFailureError: ")
    assert _DISPATCHER_ID in wrapper
    # ...and the terminal built from that same wrapper carries none of it.
    assert _DISPATCHER_ID not in _classify_the_live_refusal().failure_reason


@pytest.mark.unit
def test_an_unwrapped_failure_keeps_its_reason_verbatim() -> None:
    """Negative control — the unwrap fires ONLY where a wrapper hid the cause.

    When the attributed class IS the exception the boundary caught, there is no
    wrapper to strip and no judgement to make; the reason the boundary passed is
    published unchanged. Keeping this narrow is deliberate: every other failure
    shape in the runtime reaches the caller byte-identical to before.
    """
    terminal = classify_boundary_failure(
        TimeoutError("upstream did not answer in time"),
        topic=_SUBSCRIBE_TOPIC,
        correlation_id=UUID(_CORRELATION),
        failure_reason="TimeoutError: upstream did not answer in time",
    )

    assert terminal.failure_reason == "TimeoutError: upstream did not answer in time"
    assert terminal.retryable is True


@pytest.mark.unit
def test_a_wrapper_with_no_recoverable_cause_keeps_its_reason_verbatim() -> None:
    """Second negative control — no cause found means no substitution."""
    exc = HandlerDispatchFailureError(
        f"dispatch to topic={_SUBSCRIBE_TOPIC} returned status=handler_error "
        "with no terminal output (dispatcher_id=routing): handler/coercion failure"
    )
    terminal = classify_boundary_failure(
        exc,
        topic=_SUBSCRIBE_TOPIC,
        correlation_id=UUID(_CORRELATION),
        failure_reason=sanitize_error_message(exc),
    )

    assert terminal.failure_class == "HandlerDispatchFailureError"
    assert terminal.failure_reason == sanitize_error_message(exc)


@pytest.mark.unit
def test_the_unwrapped_reason_is_bounded() -> None:
    """A cause is not a licence to publish an unbounded string to a customer.

    The wrapper's own budget was spent on the trace; the unwrapped reason gets
    the same 500-character bound, spent on the cause instead.
    """
    long_tail = "x" * 4000
    exc = HandlerDispatchFailureError(
        f"dispatch to topic={_SUBSCRIBE_TOPIC} returned status=handler_error "
        f"with no terminal output (dispatcher_id=routing): "
        f"Dispatcher 'routing' failed: {_REFUSAL_CLASS}: "
        f"[{_REFUSAL_ONEX_CODE}] {long_tail}"
    )
    terminal = classify_boundary_failure(
        exc,
        topic=_SUBSCRIBE_TOPIC,
        correlation_id=UUID(_CORRELATION),
        failure_reason=sanitize_error_message(exc),
    )

    assert len(terminal.failure_reason) <= 500 + len("... [truncated]")
    assert terminal.failure_reason.endswith("... [truncated]")
    assert _SUBSCRIBE_TOPIC not in terminal.failure_reason


@pytest.mark.unit
def test_unwrapping_never_defeats_the_sanitizer() -> None:
    """The unwrap reads the RAW wrapper, so it re-sanitizes what it extracts.

    ``sanitize_error_message`` collapses a whole message on any of its sensitive
    substrings. Extracting a cause out of the raw exception would sail straight
    past that collapse if the extracted half were published unchecked — so the
    candidate is put back through the same sanitizer, and a redacted candidate
    falls back to the reason the boundary already sanitized.
    """
    exc = HandlerDispatchFailureError(
        f"dispatch to topic={_SUBSCRIBE_TOPIC} returned status=handler_error "
        "with no terminal output (dispatcher_id=routing): Dispatcher 'routing' "
        "failed: ProtocolConfigurationError: [ONEX_CORE_041_INVALID_CONFIGURATION] "
        "endpoint rejected password=hunter2"
    )
    terminal = classify_boundary_failure(
        exc,
        topic=_SUBSCRIBE_TOPIC,
        correlation_id=UUID(_CORRELATION),
        failure_reason=sanitize_error_message(exc),
    )

    assert "hunter2" not in terminal.failure_reason
    assert terminal.failure_reason == sanitize_error_message(exc)
    # Positive control: the same shape WITHOUT the sensitive token does unwrap.
    clean = HandlerDispatchFailureError(
        f"dispatch to topic={_SUBSCRIBE_TOPIC} returned status=handler_error "
        "with no terminal output (dispatcher_id=routing): Dispatcher 'routing' "
        "failed: ProtocolConfigurationError: [ONEX_CORE_041_INVALID_CONFIGURATION] "
        "endpoint rejected the request"
    )
    clean_terminal = classify_boundary_failure(
        clean,
        topic=_SUBSCRIBE_TOPIC,
        correlation_id=UUID(_CORRELATION),
        failure_reason=sanitize_error_message(clean),
    )
    assert clean_terminal.failure_reason == (
        "[ONEX_CORE_041_INVALID_CONFIGURATION] endpoint rejected the request"
    )


# ---------------------------------------------------------------------------
# Both residuals at the real seam — wire, publish, read what reached the bus
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


class ModelMirrorRoutingDecision(BaseModel):
    """Mirror of the decision the reducer publishes when it does not refuse."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    selected_model: str
    correlation_id: UUID


class HandlerMirrorRoutingIntentRefuses:
    """Mirror of ``HandlerRoutingIntent`` on the keyless-customer path.

    ``refuse_keyless_customer_on_cloud`` raises BEFORE the platform ladder is
    loaded. The engine's catch-all flattens it, which is the whole reason both
    residuals exist.
    """

    def handle(self, intent: ModelMirrorRoutingIntent) -> ModelMirrorRoutingDecision:
        raise CustomerKeyRefusedError(_REFUSAL_BOUNDARY_MESSAGE)


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
                        name="HandlerMirrorRoutingIntentRefuses", module=_THIS_MODULE
                    ),
                    event_model=ModelHandlerRef(
                        name="ModelMirrorRoutingIntent", module=_THIS_MODULE
                    ),
                    operation="delegation_routing",
                ),
            ),
        ),
    )


@pytest.fixture
def contract_path(tmp_path: Path) -> Path:
    path = tmp_path / "contract.yaml"
    path.write_text(_CONTRACT_YAML, encoding="utf-8")
    return path


async def _terminal_from_one_refused_record(
    contract_path: Path,
) -> ModelBoundaryFailureTerminal:
    """Wire the contract for real, publish one refused record, read the terminal."""
    correlation_id = UUID(_CORRELATION)
    seen: list[ModelEventEnvelope[object]] = []
    arrived = asyncio.Event()

    async def _collect(message: ModelEventMessage) -> None:
        envelope = ModelEventEnvelope[object].model_validate_json(message.value)
        if envelope.correlation_id == correlation_id:
            seen.append(envelope)
            arrived.set()

    bus = EventBusInmemory(environment="test", group="omn-17372-residuals")
    await bus.start()
    try:
        await bus.subscribe(
            _FAILURE_TOPIC, group_id="omn17372-failure", on_message=_collect
        )
        engine = MessageDispatchEngine()
        with patch(
            _PATCH_IMPORT_HANDLER, return_value=HandlerMirrorRoutingIntentRefuses
        ):
            await wire_from_manifest(
                ModelAutoWiringManifest(contracts=(_contract(contract_path),)),
                engine,
                event_bus=bus,
                environment="local",
            )
        engine.freeze()

        command = ModelEventEnvelope[object](
            payload={
                "intent": "routing_reducer",
                "payload": {
                    "prompt": "OMN-17372 AC2 live proof",
                    "task_type": "summarization",
                    "correlation_id": _CORRELATION,
                    "max_tokens": 32,
                },
            },
            correlation_id=correlation_id,
            event_type="omnibase-infra.delegation-routing-request",
        )
        await bus.publish(
            _SUBSCRIBE_TOPIC, None, command.model_dump_json().encode("utf-8"), None
        )
        try:
            await asyncio.wait_for(arrived.wait(), timeout=10)
        except TimeoutError:  # pragma: no cover - surfaced by the assertion below
            pass
    finally:
        await bus.close()

    assert seen, "no terminal on the contract's declared failure terminal"
    return ModelBoundaryFailureTerminal.model_validate(seen[0].payload)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_published_refusal_terminal_answers_both_residuals(
    contract_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """End to end on a real bus: what a keyless customer's refusal puts on the wire.

    This is the seam the gateway consumer reads, so these four fields ARE the
    customer's answer. An isolation test on the handler passes through both
    residuals — neither defect is in the handler.
    """
    with caplog.at_level(logging.ERROR):
        terminal = await _terminal_from_one_refused_record(contract_path)

    assert terminal.failure_class == _REFUSAL_CLASS
    assert terminal.failure_code == _REFUSAL_ONEX_CODE
    # R1
    assert terminal.retryable is False
    # R2
    assert terminal.failure_reason == _REFUSAL_BOUNDARY_MESSAGE
    assert _SUBSCRIBE_TOPIC not in terminal.failure_reason

    # ...and the operator's copy of the dispatch trace is still emitted.
    boundary_logs = "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.ERROR
    )
    assert "HandlerDispatchFailureError" in boundary_logs
    assert _SUBSCRIBE_TOPIC in boundary_logs
