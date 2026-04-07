# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that orchestrates the 7-phase autonomous build loop.

The orchestrator REACTS to reducer-approved state. It never independently
decides phase transitions — those are the sole authority of the reducer.

Flow per cycle:
    1. Receive start command -> feed IDLE event to reducer
    2. Reducer emits intent -> orchestrator invokes corresponding effect/compute
    3. Effect/compute result -> orchestrator feeds event back to reducer
    4. Repeat until reducer reaches COMPLETE or FAILED

Related:
    - OMN-7319: node_autonomous_loop_orchestrator
    - OMN-5113: Autonomous Build Loop epic
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import httpx

if TYPE_CHECKING:
    from omnibase_infra.protocols.protocol_event_bus_like import ProtocolEventBusLike

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.enums.enum_build_loop_intent_type import EnumBuildLoopIntentType
from omnibase_infra.enums.enum_build_loop_phase import EnumBuildLoopPhase
from omnibase_infra.enums.enum_buildability import EnumBuildability
from omnibase_infra.nodes.node_autonomous_loop_orchestrator.models.model_loop_cycle_summary import (
    ModelLoopCycleSummary,
)
from omnibase_infra.nodes.node_autonomous_loop_orchestrator.models.model_loop_orchestrator_result import (
    ModelLoopOrchestratorResult,
)
from omnibase_infra.nodes.node_autonomous_loop_orchestrator.models.model_loop_start_command import (
    ModelLoopStartCommand,
)
from omnibase_infra.nodes.node_build_dispatch_effect.handlers.handler_build_dispatch import (
    HandlerBuildDispatch,
)
from omnibase_infra.nodes.node_build_dispatch_effect.models.model_build_target import (
    ModelBuildTarget,
)
from omnibase_infra.nodes.node_closeout_effect.handlers.handler_closeout import (
    HandlerCloseout,
)
from omnibase_infra.nodes.node_loop_state_reducer.handlers.handler_loop_state import (
    HandlerLoopState,
)
from omnibase_infra.nodes.node_loop_state_reducer.models.model_build_loop_event import (
    ModelBuildLoopEvent,
)
from omnibase_infra.nodes.node_loop_state_reducer.models.model_build_loop_intent import (
    ModelBuildLoopIntent,
)
from omnibase_infra.nodes.node_loop_state_reducer.models.model_build_loop_state import (
    ModelBuildLoopState,
)
from omnibase_infra.nodes.node_rsd_fill_compute.handlers.handler_rsd_fill import (
    HandlerRsdFill,
)
from omnibase_infra.nodes.node_rsd_fill_compute.models.model_scored_ticket import (
    ModelScoredTicket,
)
from omnibase_infra.nodes.node_ticket_classify_compute.handlers.handler_ticket_classify import (
    HandlerTicketClassify,
)
from omnibase_infra.nodes.node_ticket_classify_compute.models.model_ticket_classify_output import (
    ModelTicketClassifyOutput,
)
from omnibase_infra.nodes.node_ticket_classify_compute.models.model_ticket_for_classification import (
    ModelTicketForClassification,
)
from omnibase_infra.nodes.node_verify_effect.handlers.handler_verify import (
    HandlerVerify,
)
from omnibase_infra.runtime.emit_daemon.topics import TOPIC_DEPLOY_REBUILD_REQUESTED
from omnibase_infra.utils.util_friction_emitter import emit_build_loop_friction

logger = logging.getLogger(__name__)


class HandlerLoopOrchestrator:
    """Orchestrates the autonomous build loop cycle.

    Reacts to reducer state by invoking the appropriate effect/compute node
    for each phase, then feeding the result back to the reducer.
    """

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def __init__(
        self,
        event_bus: ProtocolEventBusLike | None = None,
        linear_api_key: str | None = None,
    ) -> None:
        self._reducer = HandlerLoopState()
        self._closeout = HandlerCloseout()
        self._verify = HandlerVerify()
        self._rsd_fill = HandlerRsdFill()
        self._classify = HandlerTicketClassify()
        self._dispatch = HandlerBuildDispatch()
        self._event_bus = event_bus
        self._linear_api_key = linear_api_key
        # Inter-phase state: carry results between phases
        self._last_fill_result: tuple[ModelScoredTicket, ...] = ()
        self._last_classify_result: tuple[ModelBuildTarget, ...] = ()
        self._last_closeout_prs_merged: int = 0

    async def handle(
        self,
        command: ModelLoopStartCommand,
    ) -> ModelLoopOrchestratorResult:
        """Run the autonomous build loop for up to max_cycles.

        Args:
            command: Start command with configuration.

        Returns:
            ModelLoopOrchestratorResult with per-cycle summaries.
        """
        logger.info(
            "[BUILD-LOOP] === HANDLER ENTRY === handle() called "
            "(correlation_id=%s, max_cycles=%d, dry_run=%s, skip_closeout=%s, "
            "requested_at=%s, linear_api_key_set=%s)",
            command.correlation_id,
            command.max_cycles,
            command.dry_run,
            command.skip_closeout,
            command.requested_at,
            bool(self._linear_api_key),
        )

        summaries: list[ModelLoopCycleSummary] = []
        total_dispatched = 0
        cycles_completed = 0
        cycles_failed = 0

        for cycle_idx in range(command.max_cycles):
            logger.info(
                "[BUILD-LOOP] Starting cycle %d/%d (correlation_id=%s)",
                cycle_idx + 1,
                command.max_cycles,
                command.correlation_id,
            )
            summary = await self._run_cycle(
                correlation_id=command.correlation_id,
                skip_closeout=command.skip_closeout,
                dry_run=command.dry_run,
                max_consecutive_failures=3,
            )
            summaries.append(summary)
            logger.info(
                "[BUILD-LOOP] Cycle %d/%d finished: final_phase=%s, "
                "tickets_filled=%d, tickets_classified=%d, tickets_dispatched=%d, "
                "error=%s (correlation_id=%s)",
                cycle_idx + 1,
                command.max_cycles,
                summary.final_phase.value,
                summary.tickets_filled,
                summary.tickets_classified,
                summary.tickets_dispatched,
                summary.error_message,
                command.correlation_id,
            )

            if summary.final_phase == EnumBuildLoopPhase.COMPLETE:
                cycles_completed += 1
                total_dispatched += summary.tickets_dispatched
            else:
                cycles_failed += 1
                emit_build_loop_friction(
                    phase=summary.final_phase.value,
                    correlation_id=command.correlation_id,
                    severity="critical",
                    description=f"Build loop cycle {cycle_idx + 1} failed in phase {summary.final_phase.value}",
                    error_message=summary.error_message,
                )
                break

        logger.info(
            "[BUILD-LOOP] === HANDLER EXIT === Autonomous loop finished: "
            "%d completed, %d failed, %d total dispatched (correlation_id=%s)",
            cycles_completed,
            cycles_failed,
            total_dispatched,
            command.correlation_id,
        )

        return ModelLoopOrchestratorResult(
            correlation_id=command.correlation_id,
            cycles_completed=cycles_completed,
            cycles_failed=cycles_failed,
            cycle_summaries=tuple(summaries),
            total_tickets_dispatched=total_dispatched,
        )

    async def _run_cycle(
        self,
        correlation_id: UUID,
        skip_closeout: bool,
        dry_run: bool,
        max_consecutive_failures: int,
    ) -> ModelLoopCycleSummary:
        """Run a single build loop cycle through the FSM."""
        logger.info(
            "[BUILD-LOOP] _run_cycle entry (correlation_id=%s, "
            "skip_closeout=%s, dry_run=%s)",
            correlation_id,
            skip_closeout,
            dry_run,
        )
        cycle_start = datetime.now(tz=UTC)

        # Initialize state
        state = ModelBuildLoopState(
            correlation_id=correlation_id,
            phase=EnumBuildLoopPhase.IDLE,
            skip_closeout=skip_closeout,
            dry_run=dry_run,
            max_consecutive_failures=max_consecutive_failures,
        )

        # Kick off: send start event to reducer
        start_event = ModelBuildLoopEvent(
            correlation_id=correlation_id,
            source_phase=EnumBuildLoopPhase.IDLE,
            success=True,
            timestamp=datetime.now(tz=UTC),
        )
        state, intents = self._reducer.delta(state, start_event)

        # Process intents until terminal state
        while state.phase not in (
            EnumBuildLoopPhase.COMPLETE,
            EnumBuildLoopPhase.FAILED,
        ):
            if not intents:
                logger.error(
                    "No intents emitted but not in terminal state: %s",
                    state.phase.value,
                )
                break

            for intent in intents:
                event = await self._execute_intent(intent, state)
                state, intents = self._reducer.delta(state, event)

        return ModelLoopCycleSummary(
            correlation_id=correlation_id,
            cycle_number=state.cycle_number,
            final_phase=state.phase,
            started_at=cycle_start,
            completed_at=datetime.now(tz=UTC),
            tickets_filled=state.tickets_filled,
            tickets_classified=state.tickets_classified,
            tickets_dispatched=state.tickets_dispatched,
            error_message=state.error_message,
        )

    async def _execute_intent(
        self,
        intent: ModelBuildLoopIntent,
        state: ModelBuildLoopState,
    ) -> ModelBuildLoopEvent:
        """Execute a single intent by invoking the corresponding node handler.

        Returns a ModelBuildLoopEvent with the result.
        """
        now = datetime.now(tz=UTC)
        correlation_id = state.correlation_id
        logger.info(
            "[BUILD-LOOP] _execute_intent: intent_type=%s, current_phase=%s "
            "(correlation_id=%s)",
            intent.intent_type.value
            if hasattr(intent.intent_type, "value")
            else intent.intent_type,
            state.phase.value,
            correlation_id,
        )

        try:
            if intent.intent_type == EnumBuildLoopIntentType.START_CLOSEOUT:
                logger.info(
                    "[BUILD-LOOP] Phase CLOSING_OUT: invoking HandlerCloseout "
                    "(correlation_id=%s, dry_run=%s)",
                    correlation_id,
                    state.dry_run,
                )
                closeout_result = await self._closeout.handle(
                    correlation_id=correlation_id,
                    dry_run=state.dry_run,
                )
                self._last_closeout_prs_merged = closeout_result.prs_merged
                return ModelBuildLoopEvent(
                    correlation_id=correlation_id,
                    source_phase=EnumBuildLoopPhase.CLOSING_OUT,
                    success=True,
                    timestamp=now,
                )

            elif intent.intent_type == EnumBuildLoopIntentType.START_DEPLOY:
                logger.info(
                    "[BUILD-LOOP] Phase DEPLOYING: prs_merged=%d "
                    "(correlation_id=%s, dry_run=%s)",
                    self._last_closeout_prs_merged,
                    correlation_id,
                    state.dry_run,
                )
                if (
                    self._last_closeout_prs_merged > 0
                    and self._event_bus is not None
                    and not state.dry_run
                ):
                    payload = {
                        "correlation_id": str(uuid4()),
                        "requested_by": "build-loop",
                        "scope": "runtime",
                        "services": [],
                        "git_ref": "origin/main",
                    }
                    await self._event_bus.publish(
                        topic=TOPIC_DEPLOY_REBUILD_REQUESTED,
                        key=None,
                        value=json.dumps(payload).encode(),
                    )
                    logger.info(
                        "[BUILD-LOOP] Published ModelRebuildRequested to %s",
                        TOPIC_DEPLOY_REBUILD_REQUESTED,
                    )
                else:
                    logger.info(
                        "[BUILD-LOOP] Deploy skipped: prs_merged=%d, "
                        "has_event_bus=%s, dry_run=%s",
                        self._last_closeout_prs_merged,
                        self._event_bus is not None,
                        state.dry_run,
                    )
                return ModelBuildLoopEvent(
                    correlation_id=correlation_id,
                    source_phase=EnumBuildLoopPhase.DEPLOYING,
                    success=True,
                    timestamp=now,
                )

            elif intent.intent_type == EnumBuildLoopIntentType.START_VERIFY:
                logger.info(
                    "[BUILD-LOOP] Phase VERIFYING: invoking HandlerVerify "
                    "(correlation_id=%s, dry_run=%s)",
                    correlation_id,
                    state.dry_run,
                )
                result = await self._verify.handle(
                    correlation_id=correlation_id,
                    dry_run=state.dry_run,
                )
                if not result.all_critical_passed:
                    emit_build_loop_friction(
                        phase="VERIFYING",
                        correlation_id=correlation_id,
                        severity="high",
                        description="Critical verification checks failed",
                        timestamp=now,
                    )
                return ModelBuildLoopEvent(
                    correlation_id=correlation_id,
                    source_phase=EnumBuildLoopPhase.VERIFYING,
                    success=result.all_critical_passed,
                    timestamp=now,
                    error_message=None
                    if result.all_critical_passed
                    else "Critical verification checks failed",
                )

            elif intent.intent_type == EnumBuildLoopIntentType.START_FILL:
                logger.info(
                    "[BUILD-LOOP] Phase FILLING: fetching tickets from Linear "
                    "(correlation_id=%s, api_key_set=%s)",
                    correlation_id,
                    bool(self._linear_api_key),
                )
                scored = await _fetch_scored_tickets_from_linear(
                    api_key=self._linear_api_key,
                )
                fill_result = await self._rsd_fill.handle(
                    correlation_id=correlation_id,
                    scored_tickets=scored,
                    max_tickets=5,
                )
                self._last_fill_result = fill_result.selected_tickets
                logger.info(
                    "[BUILD-LOOP] Phase FILLING complete: scored=%d, selected=%d "
                    "(correlation_id=%s)",
                    len(scored),
                    fill_result.total_selected,
                    correlation_id,
                )
                return ModelBuildLoopEvent(
                    correlation_id=correlation_id,
                    source_phase=EnumBuildLoopPhase.FILLING,
                    success=True,
                    timestamp=now,
                    tickets_filled=fill_result.total_selected,
                )

            elif intent.intent_type == EnumBuildLoopIntentType.START_CLASSIFY:
                logger.info(
                    "[BUILD-LOOP] Phase CLASSIFYING: %d tickets to classify "
                    "(correlation_id=%s)",
                    len(self._last_fill_result),
                    correlation_id,
                )
                tickets_for_classify = _scored_to_classification(
                    self._last_fill_result,
                )
                classify_result = await self._classify.handle(
                    correlation_id=correlation_id,
                    tickets=tickets_for_classify,
                )
                self._last_classify_result = _extract_build_targets(
                    classify_result,
                )
                return ModelBuildLoopEvent(
                    correlation_id=correlation_id,
                    source_phase=EnumBuildLoopPhase.CLASSIFYING,
                    success=True,
                    timestamp=now,
                    tickets_classified=len(classify_result.classifications),
                )

            elif intent.intent_type == EnumBuildLoopIntentType.START_BUILD:
                logger.info(
                    "[BUILD-LOOP] Phase BUILDING: %d targets to dispatch "
                    "(correlation_id=%s, dry_run=%s, has_event_bus=%s)",
                    len(self._last_classify_result),
                    correlation_id,
                    state.dry_run,
                    self._event_bus is not None,
                )
                targets = self._last_classify_result
                dispatch_result = await self._dispatch.handle(
                    correlation_id=correlation_id,
                    targets=targets,
                    dry_run=state.dry_run,
                )

                # Orchestrator publishes delegation payloads via the
                # injected event bus (handlers must not have direct
                # event bus access).
                if self._event_bus is not None:
                    for dp in dispatch_result.delegation_payloads:
                        await self._event_bus.publish(
                            topic=dp.topic,
                            key=None,
                            value=json.dumps(dp.payload, default=str).encode(),
                        )

                return ModelBuildLoopEvent(
                    correlation_id=correlation_id,
                    source_phase=EnumBuildLoopPhase.BUILDING,
                    success=True,
                    timestamp=now,
                    tickets_dispatched=dispatch_result.total_dispatched,
                )

            elif intent.intent_type == EnumBuildLoopIntentType.CYCLE_COMPLETE:
                logger.info(
                    "[BUILD-LOOP] Phase COMPLETE: cycle finished (correlation_id=%s)",
                    correlation_id,
                )
                return ModelBuildLoopEvent(
                    correlation_id=correlation_id,
                    source_phase=EnumBuildLoopPhase.COMPLETE,
                    success=True,
                    timestamp=now,
                )

            else:
                logger.error("Unknown intent type: %s", intent.intent_type)
                emit_build_loop_friction(
                    phase=state.phase.value,
                    correlation_id=correlation_id,
                    severity="high",
                    description=f"Unknown intent type: {intent.intent_type}",
                    timestamp=now,
                )
                return ModelBuildLoopEvent(
                    correlation_id=correlation_id,
                    source_phase=state.phase,
                    success=False,
                    timestamp=now,
                    error_message=f"Unknown intent type: {intent.intent_type}",
                )

        except Exception as exc:
            logger.exception(
                "[BUILD-LOOP] Intent execution FAILED: intent_type=%s, "
                "phase=%s, error=%s (correlation_id=%s)",
                intent.intent_type,
                state.phase.value,
                exc,
                correlation_id,
            )
            emit_build_loop_friction(
                phase=state.phase.value,
                correlation_id=correlation_id,
                severity="critical",
                description=f"Intent execution failed: {intent.intent_type}",
                error_message=str(exc),
                timestamp=now,
            )
            return ModelBuildLoopEvent(
                correlation_id=correlation_id,
                source_phase=state.phase,
                success=False,
                timestamp=now,
                error_message=str(exc),
            )


_LINEAR_API_URL = "https://api.linear.app/graphql"

# Linear priority values: 0=No priority, 1=Urgent, 2=High, 3=Medium, 4=Low
_PRIORITY_TO_RSD_SCORE: dict[int, float] = {
    1: 4.0,
    2: 3.0,
    3: 2.0,
    4: 1.0,
}


async def _fetch_scored_tickets_from_linear(
    api_key: str | None = None,
) -> tuple[ModelScoredTicket, ...]:
    """Fetch backlog/unstarted tickets from Linear and score by priority.

    Args:
        api_key: Linear API key. Caller is responsible for sourcing this
                 from environment or config — handlers must not read env vars.
    """
    if not api_key:
        logger.warning("LINEAR_API_KEY not set — returning empty ticket list")
        return ()

    query = """
    query {
      issues(
        filter: {
          state: { type: { in: ["backlog", "unstarted"] } }
          project: { name: { eq: "Active Sprint" } }
        }
        first: 20
        orderBy: updatedAt
      ) {
        nodes {
          id
          identifier
          title
          priority
          description
          state { name type }
          labels { nodes { name } }
        }
      }
    }
    """

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                _LINEAR_API_URL,
                json={"query": query},
                headers={
                    "Authorization": api_key,
                    "Content-Type": "application/json",
                },
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning(
            "Linear API request failed: %s — returning empty ticket list", exc
        )
        return ()

    nodes = (data.get("data") or {}).get("issues", {}).get("nodes", [])
    scored: list[ModelScoredTicket] = []
    for node in nodes:
        priority = node.get("priority", 0) or 0
        rsd_score = _PRIORITY_TO_RSD_SCORE.get(priority, 0.5)
        labels = tuple(
            label["name"] for label in node.get("labels", {}).get("nodes", [])
        )
        scored.append(
            ModelScoredTicket(
                ticket_id=node.get("identifier", node.get("id", "")),
                title=node.get("title", ""),
                rsd_score=rsd_score,
                priority=priority,
                labels=labels,
                description=node.get("description", "") or "",
                state=node.get("state", {}).get("name", ""),
            )
        )

    logger.info("Fetched %d tickets from Linear", len(scored))
    return tuple(scored)


def _scored_to_classification(
    scored_tickets: tuple[ModelScoredTicket, ...],
) -> tuple[ModelTicketForClassification, ...]:
    """Convert scored tickets from RSD fill into classification input format."""
    return tuple(
        ModelTicketForClassification(
            ticket_id=t.ticket_id,
            title=t.title,
            description=t.description,
            labels=t.labels,
            state=t.state,
            priority=t.priority,
        )
        for t in scored_tickets
    )


def _extract_build_targets(
    classify_result: ModelTicketClassifyOutput,
) -> tuple[ModelBuildTarget, ...]:
    """Filter classification results to AUTO_BUILDABLE and convert to build targets."""

    return tuple(
        ModelBuildTarget(
            ticket_id=c.ticket_id,
            title=c.title,
            buildability=c.buildability,
        )
        for c in classify_result.classifications
        if c.buildability == EnumBuildability.AUTO_BUILDABLE
    )
