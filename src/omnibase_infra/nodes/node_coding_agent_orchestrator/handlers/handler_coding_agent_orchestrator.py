# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Coding-agent ORCHESTRATOR handler (OMN-13247, plan §5.1 / §5.3).

Sequences the workflow validate -> invoke -> capture by dispatching commands OVER
THE BUS. It NEVER constructs sibling handlers in-process, never runs an in-process
FSM loop, and never does I/O. ``handle(envelope)`` returns
``ModelHandlerOutput.for_orchestrator`` carrying the next bus command(s) as event
envelopes the runtime publishes; the orchestrator reacts to the resulting
workspace / invoke events.

Flow (event-driven, no in-process loop):

  1. consume invoke-requested -> emit the workspace-validate command (-> COMPUTE).
  2. consume workspace-validated:
       - rejected -> emit the FSM workspace-rejected advance (terminal REJECTED;
         NO subprocess ever runs).
       - ok -> emit the invoke command (-> the agent EFFECT).
  3. consume invoke-completed -> emit the FSM invoke-completed advance, then the
     capture is driven by the FSM's next intent; consume invoke-failed -> emit
     the FSM invoke-failed advance (the REDUCER owns retry/breaker).

Topic strings are NEVER hardcoded — they are resolved from the contract's
event_bus.publish_topics by suffix, the single source of truth.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from uuid import UUID, uuid4

from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.coding_agent.model_coding_agent_invoke_command import (
    ModelCodingAgentInvokeCommand,
)
from omnibase_infra.models.coding_agent.model_workspace_validate_command import (
    ModelWorkspaceValidateCommand,
)
from omnibase_infra.models.coding_agent.model_workspace_validate_result import (
    ModelWorkspaceValidateResult,
)
from omnibase_infra.nodes.node_coding_agent_orchestrator.contract_topics import (
    contract_allowed_workspace_roots,
    contract_publish_topics,
)

HANDLER_ID = "coding-agent-orchestrator"

_CONTRACT = Path(__file__).resolve().parent.parent / "contract.yaml"
_PUBLISH = contract_publish_topics(_CONTRACT)


def _topic_with_suffix(suffix: str) -> str:
    """Resolve exactly one contract publish topic ending with ``suffix``."""
    matches = [t for t in _PUBLISH if t.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(
            f"Contract {_CONTRACT} must declare exactly one event_bus.publish_topics "
            f"topic ending in {suffix!r}; found {matches}"
        )
    return matches[0]


TOPIC_WORKSPACE_VALIDATE = _topic_with_suffix("coding-agent-workspace-validate.v1")
TOPIC_INVOKE = _topic_with_suffix("coding-agent-effect-invoke.v1")
TOPIC_FSM_ADVANCE = _topic_with_suffix("coding-agent-fsm-advance.v1")


class HandlerCodingAgentOrchestrator:
    """Canonical orchestrator: dispatch coding-agent workflow commands over the bus."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self, envelope: ModelEventEnvelope[object]
    ) -> ModelHandlerOutput[None]:
        """Route one workflow lifecycle event to the next bus command(s)."""
        event_type = envelope.event_type or ""
        correlation_id = envelope.correlation_id or uuid4()

        if event_type.endswith("workspace-validated.v1"):
            events = self._on_workspace_validated(envelope, correlation_id)
        else:
            # Default entrypoint: the invoke-requested command.
            events = self._on_invoke_requested(envelope, correlation_id)

        return ModelHandlerOutput.for_orchestrator(
            input_envelope_id=envelope.envelope_id,
            correlation_id=correlation_id,
            handler_id=HANDLER_ID,
            events=tuple(events),
        )

    def _on_invoke_requested(
        self, envelope: ModelEventEnvelope[object], correlation_id: UUID
    ) -> list[ModelEventEnvelope[object]]:
        """Emit the workspace-validate command for an invoke-requested command."""
        command = _coerce_invoke(envelope.payload, correlation_id)
        validate_command = ModelWorkspaceValidateCommand(
            correlation_id=command.correlation_id,
            workspace_path=command.workspace_path,
            allowed_roots=_resolve_allowed_roots(),
            sandbox=command.sandbox,
            prompt=command.prompt,
        )
        return [
            ModelEventEnvelope(
                payload=validate_command,
                correlation_id=command.correlation_id,
                event_type=TOPIC_WORKSPACE_VALIDATE,
            )
        ]

    def _on_workspace_validated(
        self, envelope: ModelEventEnvelope[object], correlation_id: UUID
    ) -> list[ModelEventEnvelope[object]]:
        """React to the validation verdict: invoke if valid, else reject."""
        verdict, command = _coerce_validated(envelope.payload, correlation_id)

        if not verdict.valid:
            # No subprocess ever runs: advance the FSM straight to REJECTED.
            advance = {
                "event": {
                    "correlation_id": str(command.correlation_id),
                    "kind": "workspace_rejected",
                    "error_message": verdict.rejection_reason or "workspace rejected",
                }
            }
            return [
                ModelEventEnvelope(
                    payload=advance,
                    correlation_id=command.correlation_id,
                    event_type=TOPIC_FSM_ADVANCE,
                )
            ]

        return [
            ModelEventEnvelope(
                payload=command,
                correlation_id=command.correlation_id,
                event_type=TOPIC_INVOKE,
            )
        ]


def _resolve_allowed_roots() -> tuple[str, ...]:
    """Resolve the allowed workspace roots from the contract (contract-declared).

    The orchestrator contract's ``descriptor.allowed_workspace_roots`` is the
    single source of truth (overridable by an operator overlay contract, same as
    every other contract field). Fails closed: if the contract declares no roots,
    this raises so the COMPUTE node never receives an empty allowlist by accident
    and no workspace is silently permitted. No hardcoded ``/Users`` or
    ``/Volumes`` path is introduced — the policy lives entirely in the contract.
    """
    roots = contract_allowed_workspace_roots(_CONTRACT)
    if not roots:
        raise ValueError(
            f"contract {_CONTRACT} must declare a non-empty "
            "descriptor.allowed_workspace_roots; the coding-agent orchestrator "
            "fails closed rather than permit an unscoped workspace"
        )
    return roots


def _coerce_invoke(
    payload: object, correlation_id: UUID
) -> ModelCodingAgentInvokeCommand:
    if isinstance(payload, ModelCodingAgentInvokeCommand):
        return payload
    if isinstance(payload, Mapping):
        data = dict(payload)
        data.setdefault("correlation_id", str(correlation_id))
        return ModelCodingAgentInvokeCommand.model_validate(data)
    if hasattr(payload, "model_dump"):
        return ModelCodingAgentInvokeCommand.model_validate(
            payload.model_dump(mode="json")
        )
    raise TypeError(
        "invoke-requested payload must be ModelCodingAgentInvokeCommand or a "
        f"mapping; got {type(payload).__name__}"
    )


def _coerce_validated(
    payload: object, correlation_id: UUID
) -> tuple[ModelWorkspaceValidateResult, ModelCodingAgentInvokeCommand]:
    """Coerce the workspace-validated payload into (verdict, original command)."""
    mapping = _as_mapping(payload)
    if mapping is None:
        raise TypeError(
            "workspace-validated payload must be a mapping/model carrying 'verdict' "
            f"and 'command'; got {type(payload).__name__}"
        )
    if "verdict" not in mapping or "command" not in mapping:
        raise ValueError(
            "workspace-validated payload must carry both 'verdict' and 'command'"
        )
    verdict = ModelWorkspaceValidateResult.model_validate(_as_dict(mapping["verdict"]))
    command = ModelCodingAgentInvokeCommand.model_validate(_as_dict(mapping["command"]))
    return verdict, command


def _as_mapping(candidate: object) -> Mapping[str, object] | None:
    if isinstance(candidate, Mapping):
        return candidate
    model_dump = getattr(candidate, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        if isinstance(dumped, Mapping):
            return dumped
    return None


def _as_dict(candidate: object) -> dict[str, object]:
    mapping = _as_mapping(candidate)
    if mapping is None:
        raise TypeError(f"expected a mapping/model; got {type(candidate).__name__}")
    return dict(mapping)


__all__: list[str] = [
    "HANDLER_ID",
    "TOPIC_FSM_ADVANCE",
    "TOPIC_INVOKE",
    "TOPIC_WORKSPACE_VALIDATE",
    "HandlerCodingAgentOrchestrator",
]
