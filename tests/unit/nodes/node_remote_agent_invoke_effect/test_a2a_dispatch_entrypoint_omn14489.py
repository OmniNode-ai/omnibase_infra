# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-14489 — HandlerA2ATask is REACHABLE through the real auto-wiring dispatch path.

This is the RED-against-EXISTS-but-WRONG proof for the missing-entrypoint defect.

Before this ticket, ``HandlerA2ATask`` was contract-declared, wired, ingress-valid and
CI-green while exposing only ``submit()`` / ``watch()``. Auto-wiring's
``_make_dispatch_callback`` looks for ``handle_async``, then ``handle``; finding neither
it binds ``_missing_handle``, which raises::

    ModelOnexError: Auto-wired handler HandlerA2ATask does not expose a callable
                    handle() or handle_async() dispatch entrypoint.

...on the FIRST dispatch. The A2A remote-agent branch passed ingress and then died.

These tests drive the REAL production dispatch callback over the REAL handler class (no
fake handler, no patched entrypoint), so they fail against the entrypoint-less handler
and pass only once a genuine def-B ``handle`` exists. Repointing the contract's models
(the rest of OMN-14489) does NOT make them pass — only a real entrypoint does.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from omnibase_core.enums.enum_agent_protocol import EnumAgentProtocol
from omnibase_core.enums.enum_agent_task_lifecycle_type import (
    EnumAgentTaskLifecycleType,
)
from omnibase_core.enums.enum_invocation_kind import EnumInvocationKind
from omnibase_core.models.delegation.model_a2a_task_request import ModelA2ATaskRequest
from omnibase_core.models.delegation.model_a2a_task_response import ModelA2ATaskResponse
from omnibase_core.models.delegation.model_agent_task_lifecycle_event import (
    ModelAgentTaskLifecycleEvent,
)
from omnibase_core.models.delegation.model_invocation_command import (
    ModelInvocationCommand,
)
from omnibase_core.models.delegation.model_target_agent import ModelTargetAgent
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.nodes.node_remote_agent_invoke_effect.handlers.handler_a2a_task import (
    HandlerA2ATask,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import _make_dispatch_callback
from omnibase_infra.runtime.auto_wiring.models import ModelHandlerRef

_TARGET_REF = "peer-agent"
_REMOTE_HANDLE = "remote-task-abc123"

# The contract-declared event_model, exactly as node_remote_agent_invoke_effect
# declares it — this is what auto-wiring passes to _make_dispatch_callback.
_EVENT_MODEL = ModelHandlerRef(
    name="ModelInvocationCommand",
    module="omnibase_core.models.delegation.model_invocation_command",
)


class _StubTransport:
    """Transport boundary stub — keeps the test off the network, not off the seam."""

    async def submit(
        self,
        *,
        target: ModelTargetAgent,
        request_model: ModelA2ATaskRequest,
    ) -> ModelA2ATaskResponse:
        del target, request_model
        return ModelA2ATaskResponse(
            remote_task_handle=_REMOTE_HANDLE,
            status=EnumAgentTaskLifecycleType.SUBMITTED,
        )

    async def get_task(self, **kwargs: object) -> object:  # pragma: no cover
        raise NotImplementedError


def _handler() -> HandlerA2ATask:
    return HandlerA2ATask(
        target_registry={
            _TARGET_REF: ModelTargetAgent(
                target_ref=_TARGET_REF,
                base_url="http://peer.invalid",
                protocol=EnumAgentProtocol.A2A,
            )
        },
        transport=_StubTransport(),
    )


def _command() -> ModelInvocationCommand:
    """The payload the delegation orchestrator really publishes to
    ``onex.cmd.omnibase-infra.remote-agent-invoke.v1``."""
    return ModelInvocationCommand(
        task_id=uuid4(),
        correlation_id=uuid4(),
        invocation_kind=EnumInvocationKind.AGENT,
        agent_protocol=EnumAgentProtocol.A2A,
        target_ref=_TARGET_REF,
    )


@pytest.mark.unit
def test_handler_exposes_a_dispatch_entrypoint() -> None:
    """The bare invariant: auto-wiring can only bind handle/handle_async.

    RED against the pre-OMN-14489 handler, which had only submit()/watch().
    """
    assert callable(getattr(HandlerA2ATask, "handle", None)) or callable(
        getattr(HandlerA2ATask, "handle_async", None)
    ), (
        "HandlerA2ATask exposes neither handle() nor handle_async(); auto-wiring "
        "binds _missing_handle and every dispatch raises ModelOnexError."
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_real_dispatch_callback_invokes_handler_and_emits_lifecycle_event() -> (
    None
):
    """LOAD-BEARING: a real producer payload dispatched through the REAL auto-wiring
    callback reaches the handler and yields a lifecycle event on the output topic.

    Against the entrypoint-less handler this raises ModelOnexError (_missing_handle)
    rather than returning a result — that raise IS the bug, caught here.
    """
    callback = _make_dispatch_callback(_handler(), _EVENT_MODEL)
    command = _command()
    envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
        payload=command,
        correlation_id=command.correlation_id,
        event_type="ModelInvocationCommand",
    )

    result = await callback(envelope)

    assert result is not None, "Dispatch produced no result — the handler never ran."
    assert len(result.output_events) == 1, (
        "Expected exactly one lifecycle event on "
        "onex.evt.omnibase-infra.agent-task-lifecycle.v1 (the node's 0-OUT starvation). "
        f"Got {result.output_events!r}"
    )
    event = result.output_events[0]
    assert isinstance(event, ModelAgentTaskLifecycleEvent)
    assert event.lifecycle_type is EnumAgentTaskLifecycleType.SUBMITTED
    assert event.remote_task_handle == _REMOTE_HANDLE
    assert event.task_id == command.task_id
    assert event.correlation_id == command.correlation_id


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handler_receives_typed_command_not_raw_envelope() -> None:
    """Seam check: a def-B ``handle`` must receive a validated
    ModelInvocationCommand, NOT a raw envelope — on BOTH dispatch arms.

    The dispatch callback has two arms. The typed arm (declared ``event_model``)
    validates the payload against the declared model. Since OMN-14716 the untyped
    arm (``operation_match``, no ``event_model`` — the shape the .201
    finding-aggregator canary used) ALSO delivers a typed def-B payload: it
    signature-introspects the ``handle(request: ModelX)`` annotation and coerces
    the payload into ``ModelX`` at the adapter boundary, mirroring runtime_local
    (``_coercion_target_model_type``, OMN-8724). Before OMN-14716 the untyped arm
    handed the handler the RAW ModelEventEnvelope, which a def-B handler broke on
    at first attribute access (``'dict' object has no attribute ...``) — the exact
    incident. ``event_model`` remains load-bearing, but for TYPE-SCOPING
    (``payload_type_matcher`` / multi-handler routing, OMN-12416), no longer as
    the only thing that types a def-B handler's input.
    """
    command = _command()

    async def _dispatch_with(event_model: ModelHandlerRef | None) -> object:
        seen: list[object] = []
        handler = _handler()
        original = handler.handle

        async def _spy(request: ModelInvocationCommand) -> ModelAgentTaskLifecycleEvent:
            seen.append(request)
            return await original(request)

        # Bind the spy on the instance; _make_dispatch_callback resolves handle from it.
        handler.handle = _spy  # type: ignore[method-assign]
        callback = _make_dispatch_callback(handler, event_model)
        envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
            payload=command,
            correlation_id=command.correlation_id,
            event_type="ModelInvocationCommand",
        )
        try:
            await callback(envelope)
        except Exception:  # noqa: BLE001 — WHAT the handler received is the assertion,
            # not whether the downstream transport stub succeeded.
            pass
        assert len(seen) == 1, "Handler was not invoked exactly once."
        return seen[0]

    # Typed arm — the contract as OMN-14489 leaves it (event_model declared).
    typed = await _dispatch_with(_EVENT_MODEL)
    assert isinstance(typed, ModelInvocationCommand), (
        "Handler received a raw envelope instead of a validated ModelInvocationCommand: "
        f"{type(typed).__name__}. The typed def-B dispatch arm must validate the payload "
        "against the declared event_model."
    )
    assert typed.target_ref == _TARGET_REF

    # Untyped arm — operation_match / no event_model (the finding-aggregator shape).
    # OMN-14716: this arm now ALSO delivers a validated ModelInvocationCommand via
    # def-B signature-introspection coercion, so a def-B handler no longer receives a
    # raw envelope and crashes. (Pre-OMN-14716 this handed back a raw
    # ModelEventEnvelope — the incident this ticket fixes.)
    untyped = await _dispatch_with(None)
    assert isinstance(untyped, ModelInvocationCommand), (
        "Expected the no-event_model (operation_match) arm to coerce the payload into "
        f"the def-B handler's declared ModelInvocationCommand (got {type(untyped).__name__}). "
        "OMN-14716 brings this arm to parity with runtime_local; a raw envelope here is "
        "the 'dict object has no attribute config' regression."
    )
    assert untyped.target_ref == _TARGET_REF


@pytest.mark.unit
def test_clock_is_timezone_aware() -> None:
    """Guard the occurred_at the entrypoint stamps (frozen models reject naive dt)."""
    event = ModelAgentTaskLifecycleEvent(
        task_id=uuid4(),
        correlation_id=uuid4(),
        lifecycle_type=EnumAgentTaskLifecycleType.SUBMITTED,
        occurred_at=datetime.now(UTC),
    )
    assert event.occurred_at.tzinfo is not None
