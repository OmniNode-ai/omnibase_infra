# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17397 — a command no dispatcher accepts dies with a TYPED cause.

C16's R-DELEG-26 probe submits a delegation whose ``task_type`` no consumer
accepts. The gateway takes it (``task_type`` is a bare string there), and the
delegate-skill consumer's closed ``Literal`` set refuses it at type-scoping.
Live on the ``.201`` dev lane, 2026-09-23T03:43:34Z, correlation
``1b94ea62-…``, the boundary did terminalize it — and said this::

    metric_name=boundary_failure_terminalized
      terminal_topic=onex.evt.omnimarket.delegate-skill-failed.v1
      failure_class=ValueError failure_code=None retryable=True

The engine had resolved the code — ``ENVELOPE_VALIDATION_FAILED`` for a
payload a registered dispatcher refused, ``ITEM_NOT_REGISTERED`` for a true
wiring gap — and ``_raise_if_no_dispatcher_drop`` dropped it on the floor
when it built the exception, so the caller got a class and no code. And it
told the caller to RETRY a record that will be refused identically on every
delivery.

These tests drive the real seam (``wire_from_manifest``, a real
``MessageDispatchEngine``, a real in-memory bus) with a mirror of the
delegate-skill request's closed task-class set, then pin the two
``_raise_if_no_dispatcher_drop`` shapes directly.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from unittest.mock import patch
from uuid import UUID

import pytest
from pydantic import BaseModel, ConfigDict

from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    HandlerDispatchFailureError,
    _raise_if_no_dispatcher_drop,
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

_THIS_MODULE = "tests.integration.runtime.test_omn17397_unaccepted_task_class_terminal"
_SUBSCRIBE_TOPIC = "onex.cmd.omnimarket.delegate-skill.v1"  # onex-topic-allow: verbatim from the live R-DELEG-26 trace
_SUCCESS_TOPIC = "onex.evt.omnimarket.delegate-skill-completed.v1"  # onex-topic-allow: verbatim from the live R-DELEG-26 trace
_FAILURE_TOPIC = "onex.evt.omnimarket.delegate-skill-failed.v1"  # onex-topic-allow: verbatim from the live R-DELEG-26 trace
_CORRELATION = "1b94ea62-3d08-40ff-b6ac-e809077041df"
_DYING_TASK_TYPE = "omn19181_not_a_task_class"

_PATCH_IMPORT_HANDLER = (
    "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class"
)


class ModelMirrorDelegateSkillRequest(BaseModel):
    """Mirror of omnimarket's ``ModelDelegateSkillRequest``: a CLOSED task class set."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt: str
    task_type: Literal["test", "document", "research"]
    source: str
    max_tokens: int = 2048


class ModelMirrorDelegateSkillResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ok: bool


class HandlerMirrorDelegateSkill:
    """Never reached: type-scoping refuses the payload before dispatch."""

    def handle(
        self, request: ModelMirrorDelegateSkillRequest
    ) -> ModelMirrorDelegateSkillResponse:  # pragma: no cover - unreachable by design
        raise AssertionError("an unaccepted task class must never reach the handler")


_CONTRACT_YAML = f"""
name: "node_delegate_skill_orchestrator_mirror"
node_type: "ORCHESTRATOR_GENERIC"
terminal_events:
  success: "{_SUCCESS_TOPIC}"
  failure: "{_FAILURE_TOPIC}"
event_bus:
  subscribe_topics:
    - "{_SUBSCRIBE_TOPIC}"
  publish_topics:
    - "{_SUCCESS_TOPIC}"
    - "{_FAILURE_TOPIC}"
published_events:
  - event_type: "DelegateSkillCompleted"
    topic: "{_SUCCESS_TOPIC}"
    description: "Delegation completed with typed result."
"""


def _contract(contract_path: Path) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name="node_delegate_skill_orchestrator_mirror",
        node_type="ORCHESTRATOR_GENERIC",
        contract_version=ModelContractVersion(major=0, minor=1, patch=0),
        contract_path=contract_path,
        entry_point_name="node_delegate_skill_orchestrator_mirror",
        package_name="omnimarket",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(_SUBSCRIBE_TOPIC,),
            publish_topics=(_SUCCESS_TOPIC, _FAILURE_TOPIC),
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="operation_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name="HandlerMirrorDelegateSkill", module=_THIS_MODULE
                    ),
                    event_model=ModelHandlerRef(
                        name="ModelMirrorDelegateSkillRequest", module=_THIS_MODULE
                    ),
                    operation="delegate_skill",
                ),
            ),
        ),
    )


@pytest.fixture
def contract_path(tmp_path: Path) -> Path:
    path = tmp_path / "contract.yaml"
    path.write_text(_CONTRACT_YAML, encoding="utf-8")
    return path


async def _drive_unaccepted_task_class(
    contract_path: Path,
) -> list[ModelEventEnvelope[object]]:
    correlation_id = UUID(_CORRELATION)
    seen: list[ModelEventEnvelope[object]] = []
    arrived = asyncio.Event()

    async def collect(message: ModelEventMessage) -> None:
        envelope = ModelEventEnvelope[object].model_validate_json(message.value)
        if envelope.correlation_id == correlation_id:
            seen.append(envelope)
            arrived.set()

    bus = EventBusInmemory(environment="test", group="omn-17397")
    await bus.start()
    try:
        await bus.subscribe(_FAILURE_TOPIC, group_id="omn17397", on_message=collect)
        engine = MessageDispatchEngine()
        with patch(_PATCH_IMPORT_HANDLER, return_value=HandlerMirrorDelegateSkill):
            await wire_from_manifest(
                ModelAutoWiringManifest(contracts=(_contract(contract_path),)),
                engine,
                event_bus=bus,
                environment="local",
            )
        engine.freeze()
        command = ModelEventEnvelope[object](
            payload={
                "prompt": "Reply with the single word: alive.",
                "task_type": _DYING_TASK_TYPE,
                "source": "external-client",
                "max_tokens": 32,
            },
            correlation_id=correlation_id,
            event_type="omnimarket.delegate-skill",
        )
        await bus.publish(
            _SUBSCRIBE_TOPIC, None, command.model_dump_json().encode("utf-8"), None
        )
        try:
            await asyncio.wait_for(arrived.wait(), timeout=10)
        except TimeoutError:  # pragma: no cover - surfaced by the assertions
            pass
    finally:
        await bus.close()
    return seen


@pytest.mark.integration
@pytest.mark.asyncio
async def test_unaccepted_task_class_terminal_names_its_cause_and_code(
    contract_path: Path,
) -> None:
    """The live R-DELEG-26 shape: typed class, canonical code, not retryable."""
    seen = await _drive_unaccepted_task_class(contract_path)

    assert len(seen) == 1, "exactly one terminal answers the refused command"
    terminal = ModelBoundaryFailureTerminal.model_validate(seen[0].payload)
    assert terminal.correlation_id == UUID(_CORRELATION)
    assert terminal.failure_class == "ValueError"
    # RED before the fix: None. The engine resolved this code and the boundary
    # dropped it while building the exception.
    assert terminal.failure_code == EnumCoreErrorCode.ENVELOPE_VALIDATION_FAILED.value
    # RED before the fix: True. The same record is refused on every delivery.
    assert terminal.retryable is False
    # The customer-facing reason names the refused field, not a dispatch trace.
    assert terminal.failure_reason.startswith("task_type:")
    assert _SUBSCRIBE_TOPIC not in terminal.failure_reason


def _no_dispatcher_result(
    *, failure_class: str, error_code: EnumCoreErrorCode
) -> ModelDispatchResult:
    now = datetime.now(UTC)
    return ModelDispatchResult(
        status=EnumDispatchStatus.NO_DISPATCHER,
        topic=_SUBSCRIBE_TOPIC,
        started_at=now,
        completed_at=now,
        correlation_id=UUID(_CORRELATION),
        error_message="refused",
        error_code=error_code,
        error_details={"failure_class": failure_class},
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("failure_class", "error_code"),
    [
        ("publisher_malformed", EnumCoreErrorCode.ENVELOPE_VALIDATION_FAILED),
        ("no_dispatcher", EnumCoreErrorCode.ITEM_NOT_REGISTERED),
    ],
)
def test_no_dispatcher_drop_carries_the_engine_code_and_is_not_retryable(
    failure_class: str, error_code: EnumCoreErrorCode
) -> None:
    with pytest.raises(HandlerDispatchFailureError) as excinfo:
        _raise_if_no_dispatcher_drop(
            _no_dispatcher_result(failure_class=failure_class, error_code=error_code),
            _SUBSCRIBE_TOPIC,
        )
    assert excinfo.value.failure_code == error_code.value
    terminal = classify_boundary_failure(
        excinfo.value,
        topic=_SUBSCRIBE_TOPIC,
        correlation_id=UUID(_CORRELATION),
        failure_reason="refused",
        # Called exactly as the boundary calls it (handler_wiring terminal emit).
        failure_code=excinfo.value.failure_code,
    )
    assert terminal.failure_code == error_code.value
    assert terminal.retryable is False


@pytest.mark.integration
def test_a_failed_dispatch_keeps_its_retryable_derivation() -> None:
    """Positive control: only the no-dispatcher shape is pinned non-retryable.

    A handler crash with no non-retryable class on its chain stays retryable,
    exactly as OMN-16812 derived it.
    """
    exc = HandlerDispatchFailureError(
        "dispatch returned status=handler_error: TimeoutError: upstream slow",
        failure_code=EnumCoreErrorCode.HANDLER_EXECUTION_ERROR.value,
    )
    terminal = classify_boundary_failure(
        exc,
        topic=_SUBSCRIBE_TOPIC,
        correlation_id=UUID(_CORRELATION),
        failure_reason="upstream slow",
    )
    assert terminal.retryable is True
