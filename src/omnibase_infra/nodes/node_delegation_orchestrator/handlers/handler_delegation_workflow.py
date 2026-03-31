# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Delegation orchestrator handler with correlation_id-keyed FSM.

Coordinates the full delegation workflow:
1. Receive ModelDelegationRequest -> state RECEIVED
2. Invoke routing reducer -> state ROUTED
3. Invoke LLM inference effect -> state INFERENCE_COMPLETED
4. Invoke quality gate reducer -> state GATE_EVALUATED
5. Emit delegation-completed or delegation-failed -> COMPLETED | FAILED

The FSM is replay-safe: duplicate events for the same correlation_id
are rejected if the workflow is already in or past that state.

Related:
    - OMN-7040: Node-based delegation pipeline
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from uuid import UUID

from omnibase_infra.nodes.node_delegation_orchestrator.enums import (
    EnumDelegationState,
)
from omnibase_infra.nodes.node_delegation_orchestrator.models.model_delegation_event import (
    ModelDelegationEvent,
)
from omnibase_infra.nodes.node_delegation_orchestrator.models.model_delegation_request import (
    ModelDelegationRequest,
)
from omnibase_infra.nodes.node_delegation_orchestrator.models.model_delegation_result import (
    ModelDelegationResult,
)
from omnibase_infra.nodes.node_delegation_orchestrator.models.model_inference_intent import (
    ModelInferenceIntent,
)
from omnibase_infra.nodes.node_delegation_orchestrator.models.model_inference_response_data import (
    ModelInferenceResponseData,
)
from omnibase_infra.nodes.node_delegation_orchestrator.models.model_quality_gate_intent import (
    ModelQualityGateIntent,
)
from omnibase_infra.nodes.node_delegation_orchestrator.models.model_routing_intent import (
    ModelRoutingIntent,
)
from omnibase_infra.nodes.node_delegation_quality_gate_reducer.models.model_quality_gate_input import (
    ModelQualityGateInput,
)
from omnibase_infra.nodes.node_delegation_quality_gate_reducer.models.model_quality_gate_result import (
    ModelQualityGateResult,
)
from omnibase_infra.nodes.node_delegation_routing_reducer.models.model_routing_decision import (
    ModelRoutingDecision,
)

# Valid state transitions: from_state -> set of valid to_states
_VALID_TRANSITIONS: dict[EnumDelegationState, frozenset[EnumDelegationState]] = {
    EnumDelegationState.RECEIVED: frozenset({EnumDelegationState.ROUTED}),
    EnumDelegationState.ROUTED: frozenset({EnumDelegationState.INFERENCE_COMPLETED}),
    EnumDelegationState.INFERENCE_COMPLETED: frozenset(
        {EnumDelegationState.GATE_EVALUATED}
    ),
    EnumDelegationState.GATE_EVALUATED: frozenset(
        {EnumDelegationState.COMPLETED, EnumDelegationState.FAILED}
    ),
    EnumDelegationState.COMPLETED: frozenset(),
    EnumDelegationState.FAILED: frozenset(),
}


@dataclass
class DelegationWorkflowState:
    """Mutable workflow state for a single delegation correlation_id."""

    correlation_id: UUID
    state: EnumDelegationState = EnumDelegationState.RECEIVED
    request: ModelDelegationRequest | None = None
    routing_decision: ModelRoutingDecision | None = None
    inference_content: str | None = None
    inference_model_used: str | None = None
    inference_latency_ms: int = 0
    inference_prompt_tokens: int = 0
    inference_completion_tokens: int = 0
    inference_total_tokens: int = 0
    gate_result: ModelQualityGateResult | None = None
    started_at_ns: int = field(default_factory=time.monotonic_ns)


class HandlerDelegationWorkflow:
    """Delegation orchestrator with correlation_id-keyed FSM state machine.

    Each delegation request creates a workflow keyed by its correlation_id.
    Events are matched to workflows by correlation_id and processed through
    the FSM. Duplicate or out-of-order events are handled safely.
    """

    def __init__(self) -> None:
        self._workflows: dict[UUID, DelegationWorkflowState] = {}

    @property
    def workflows(self) -> dict[UUID, DelegationWorkflowState]:
        """Expose workflows for testing/observability."""
        return self._workflows

    def _transition(
        self,
        workflow: DelegationWorkflowState,
        target: EnumDelegationState,
    ) -> None:
        """Transition workflow to target state, enforcing FSM validity."""
        valid = _VALID_TRANSITIONS.get(workflow.state, frozenset())
        if target not in valid:
            msg = (
                f"Invalid state transition: {workflow.state} -> {target} "
                f"for correlation_id={workflow.correlation_id}"
            )
            raise InvalidStateTransitionError(msg)
        workflow.state = target

    def handle_delegation_request(
        self,
        request: ModelDelegationRequest,
    ) -> list[ModelRoutingIntent]:
        """Handle incoming delegation request. Returns intents to emit.

        Creates a new workflow for this correlation_id or rejects duplicates.
        Emits an intent to the routing reducer.
        """
        cid = request.correlation_id

        if cid in self._workflows:
            return []

        workflow = DelegationWorkflowState(
            correlation_id=cid,
            request=request,
        )
        self._workflows[cid] = workflow

        return [ModelRoutingIntent(payload=request)]

    def handle_routing_decision(
        self,
        decision: ModelRoutingDecision,
    ) -> list[ModelInferenceIntent]:
        """Handle routing decision from the routing reducer.

        Transitions RECEIVED -> ROUTED, then emits intent to LLM inference.
        """
        cid = decision.correlation_id
        workflow = self._workflows.get(cid)
        if workflow is None:
            return []

        if workflow.state != EnumDelegationState.RECEIVED:
            return []

        self._transition(workflow, EnumDelegationState.ROUTED)
        workflow.routing_decision = decision

        assert workflow.request is not None
        return [
            ModelInferenceIntent(
                base_url=decision.endpoint_url,
                model=decision.selected_model,
                system_prompt=decision.system_prompt,
                prompt=workflow.request.prompt,
                max_tokens=workflow.request.max_tokens,
                correlation_id=cid,
            )
        ]

    def handle_inference_response(
        self,
        response: ModelInferenceResponseData,
    ) -> list[ModelQualityGateIntent]:
        """Handle LLM inference response.

        Transitions ROUTED -> INFERENCE_COMPLETED, then emits intent to
        the quality gate reducer.
        """
        workflow = self._workflows.get(response.correlation_id)
        if workflow is None:
            return []

        if workflow.state != EnumDelegationState.ROUTED:
            return []

        self._transition(workflow, EnumDelegationState.INFERENCE_COMPLETED)
        workflow.inference_content = response.content
        workflow.inference_model_used = response.model_used
        workflow.inference_latency_ms = response.latency_ms
        workflow.inference_prompt_tokens = response.prompt_tokens
        workflow.inference_completion_tokens = response.completion_tokens
        workflow.inference_total_tokens = response.total_tokens

        assert workflow.request is not None
        gate_input = ModelQualityGateInput(
            correlation_id=response.correlation_id,
            task_type=workflow.request.task_type,
            llm_response_content=response.content,
        )

        return [ModelQualityGateIntent(payload=gate_input)]

    def handle_gate_result(
        self,
        result: ModelQualityGateResult,
    ) -> list[ModelDelegationEvent]:
        """Handle quality gate result.

        Transitions INFERENCE_COMPLETED -> GATE_EVALUATED, then evaluates
        pass/fail to transition to COMPLETED or FAILED. Returns the
        delegation result event to emit.
        """
        cid = result.correlation_id
        workflow = self._workflows.get(cid)
        if workflow is None:
            return []

        if workflow.state != EnumDelegationState.INFERENCE_COMPLETED:
            return []

        self._transition(workflow, EnumDelegationState.GATE_EVALUATED)
        workflow.gate_result = result

        assert workflow.request is not None
        assert workflow.routing_decision is not None
        assert workflow.inference_content is not None
        assert workflow.inference_model_used is not None

        elapsed_ms = (time.monotonic_ns() - workflow.started_at_ns) // 1_000_000

        delegation_result = ModelDelegationResult(
            correlation_id=cid,
            task_type=workflow.request.task_type,
            model_used=workflow.inference_model_used,
            endpoint_url=workflow.routing_decision.endpoint_url,
            content=workflow.inference_content,
            quality_passed=result.passed,
            quality_score=result.quality_score,
            latency_ms=elapsed_ms,
            prompt_tokens=workflow.inference_prompt_tokens,
            completion_tokens=workflow.inference_completion_tokens,
            total_tokens=workflow.inference_total_tokens,
            fallback_to_claude=result.fallback_recommended,
            failure_reason="; ".join(result.failure_reasons)
            if not result.passed
            else "",
        )

        if result.passed:
            self._transition(workflow, EnumDelegationState.COMPLETED)
            return [
                ModelDelegationEvent(
                    topic="onex.evt.omnibase-infra.delegation-completed.v1",
                    payload=delegation_result,
                )
            ]
        else:
            self._transition(workflow, EnumDelegationState.FAILED)
            return [
                ModelDelegationEvent(
                    topic="onex.evt.omnibase-infra.delegation-failed.v1",
                    payload=delegation_result,
                )
            ]


class InvalidStateTransitionError(Exception):
    """Raised when an FSM state transition is invalid."""


__all__: list[str] = [
    "DelegationWorkflowState",
    "HandlerDelegationWorkflow",
    "InvalidStateTransitionError",
]
