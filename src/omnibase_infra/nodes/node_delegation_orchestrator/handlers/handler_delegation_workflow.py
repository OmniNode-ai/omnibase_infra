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
from datetime import UTC, datetime
from uuid import UUID

from omnibase_infra.event_bus.topic_constants import (
    TOPIC_DELEGATION_COMPLETED,
    TOPIC_DELEGATION_FAILED,
)
from omnibase_infra.nodes.node_delegation_orchestrator.enums import (
    EnumDelegationState,
)
from omnibase_infra.nodes.node_delegation_orchestrator.models.model_baseline_intent import (
    ModelBaselineIntent,
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
from omnibase_infra.nodes.node_delegation_orchestrator.models.model_task_delegated_event import (
    ModelTaskDelegatedEvent,
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

GateResultEvent = ModelDelegationEvent | ModelTaskDelegatedEvent | ModelBaselineIntent

# Temperature by task type (Task 10, OMN-7040)
_TASK_TEMPERATURE: dict[str, float] = {
    "test": 0.3,
    "document": 0.5,
    "research": 0.7,
}

# Approximate Claude pricing for savings estimation (Task 11, OMN-7040)
# Claude Sonnet 3.5: ~$3/M input, ~$15/M output tokens
_CLAUDE_INPUT_PRICE_PER_TOKEN: float = 3.0 / 1_000_000
_CLAUDE_OUTPUT_PRICE_PER_TOKEN: float = 15.0 / 1_000_000

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
    inference_llm_call_id: str = ""
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
        temperature = _TASK_TEMPERATURE.get(workflow.request.task_type, 0.3)
        return [
            ModelInferenceIntent(
                base_url=decision.endpoint_url,
                model=decision.selected_model,
                system_prompt=decision.system_prompt,
                prompt=workflow.request.prompt,
                max_tokens=workflow.request.max_tokens,
                temperature=temperature,
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
        workflow.inference_llm_call_id = response.llm_call_id

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
    ) -> list[GateResultEvent]:
        """Handle quality gate result.

        Transitions INFERENCE_COMPLETED -> GATE_EVALUATED, then evaluates
        pass/fail to transition to COMPLETED or FAILED. Returns:
        1. The delegation result event (completed or failed)
        2. A backward-compatible task-delegated.v1 event for omnidash (Task 12)
        3. A baseline comparison intent for savings computation (Task 11, pass only)
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

        # Estimate Claude cost for savings comparison (Task 11)
        estimated_claude_cost = (
            workflow.inference_prompt_tokens * _CLAUDE_INPUT_PRICE_PER_TOKEN
            + workflow.inference_completion_tokens * _CLAUDE_OUTPUT_PRICE_PER_TOKEN
        )

        # Backward-compatible task-delegated.v1 event for omnidash (Task 12)
        compat_event = ModelTaskDelegatedEvent(
            topic="onex.evt.omniclaude.task-delegated.v1",
            timestamp=datetime.now(UTC).isoformat(),
            correlation_id=cid,
            session_id=None,
            task_type=workflow.request.task_type,
            delegated_to=workflow.inference_model_used,
            model_name=workflow.routing_decision.selected_model,
            quality_gate_passed=result.passed,
            quality_gates_failed=list(result.failure_reasons),
            cost_usd=0.0,
            cost_savings_usd=round(estimated_claude_cost, 6),
            delegation_latency_ms=elapsed_ms,
            llm_call_id=workflow.inference_llm_call_id,
        )

        events: list[GateResultEvent] = []

        if result.passed:
            self._transition(workflow, EnumDelegationState.COMPLETED)
            events.append(
                ModelDelegationEvent(
                    topic=TOPIC_DELEGATION_COMPLETED,
                    payload=delegation_result,
                )
            )
            # Baseline comparison for savings pipeline (Task 11)
            events.append(
                ModelBaselineIntent(
                    correlation_id=cid,
                    task_type=workflow.request.task_type,
                    baseline_cost_usd=estimated_claude_cost,
                    candidate_cost_usd=0.0,
                    prompt_tokens=workflow.inference_prompt_tokens,
                    completion_tokens=workflow.inference_completion_tokens,
                    total_tokens=workflow.inference_total_tokens,
                )
            )
        else:
            self._transition(workflow, EnumDelegationState.FAILED)
            events.append(
                ModelDelegationEvent(
                    topic=TOPIC_DELEGATION_FAILED,
                    payload=delegation_result,
                )
            )

        # Always emit backward-compatible event for omnidash (Task 12)
        events.append(compat_event)

        return events


class InvalidStateTransitionError(Exception):
    """Raised when an FSM state transition is invalid."""


__all__: list[str] = [
    "DelegationWorkflowState",
    "HandlerDelegationWorkflow",
    "InvalidStateTransitionError",
]
