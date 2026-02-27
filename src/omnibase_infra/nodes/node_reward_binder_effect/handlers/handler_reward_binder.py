# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Handler that emits structured reward events to Kafka.

Emits two event types in strict order after ScoringReducer produces a
ModelEvaluationResult:

  1. ``ModelRewardAssignedEvent``     -> ``onex.evt.omnimemory.reward-assigned.v1``
  2. ``ModelPolicyStateUpdatedEvent`` -> ``onex.evt.omnimemory.policy-state-updated.v1``

Note: ``ModelRunEvaluatedEvent`` (``onex.evt.omnimemory.run-evaluated.v1``) was removed
in OMN-2929. The canonical run-evaluated event is produced by omniintelligence
``node_evidence_collection_effect`` to ``onex.evt.omniintelligence.run-evaluated.v1``.
The orphan ``onex.evt.omnimemory.run-evaluated.v1`` topic had zero consumers.

Design constraints:
  - No scoring logic -- only emits what ScoringReducer produced.
  - Kafka publish failures are never swallowed silently; they propagate to the caller.
  - Events are emitted in the order listed above.
  - ``objective_fingerprint`` is SHA-256 of ``ModelObjectiveSpec.model_dump_json()``
    (deterministic serialisation).
  - ``evidence_refs`` in ``ModelRewardAssignedEvent`` trace back to specific
    ``ModelEvidenceItem.item_id`` values from the input ``ModelEvidenceBundle``.
  - Emits canonical omnibase_core.ModelRewardAssignedEvent (OMN-2928):
    bridges run-level score vector fields with consumer's policy signal fields.
  - ``reward_delta`` is computed as mean(score_vector) * 2 - 1, normalised to [-1, +1].
  - ``idempotency_key`` is SHA-256 of (event_id.hex, policy_id.hex, run_id.hex).
  - Envelope must include ``policy_id`` (UUID) and ``policy_type`` (EnumPolicyType).

Ticket: OMN-2927, OMN-2928
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from omnibase_core.enums.enum_policy_type import EnumPolicyType
from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_core.models.objective.model_reward_assigned_event import (
    ModelRewardAssignedEvent,
)
from omnibase_core.models.objective.model_score_vector import ModelScoreVector
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.errors import ModelInfraErrorContext, RuntimeHostError
from omnibase_infra.nodes.node_reward_binder_effect.models.model_evaluation_result import (
    ModelEvaluationResult,
)
from omnibase_infra.nodes.node_reward_binder_effect.models.model_objective_spec import (
    ModelObjectiveSpec,
)
from omnibase_infra.nodes.node_reward_binder_effect.models.model_policy_state_updated_event import (
    ModelPolicyStateUpdatedEvent,
)
from omnibase_infra.nodes.node_reward_binder_effect.models.model_reward_binder_output import (
    ModelRewardBinderOutput,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from omnibase_core.models.container.model_onex_container import ModelONEXContainer

logger = logging.getLogger(__name__)

# ==============================================================================
# Topic name constants
# ==============================================================================
_TOPIC_REWARD_ASSIGNED = "onex.evt.omnimemory.reward-assigned.v1"
_TOPIC_POLICY_STATE_UPDATED = "onex.evt.omnimemory.policy-state-updated.v1"


def _compute_objective_fingerprint(spec: ModelObjectiveSpec) -> str:
    """Compute a tamper-evident SHA-256 fingerprint of the ModelObjectiveSpec.

    Uses ``ModelObjectiveSpec.model_dump_json()`` for deterministic serialisation.

    Args:
        spec: The ModelObjectiveSpec to fingerprint.

    Returns:
        64-character lowercase hex digest.
    """
    serialized = spec.model_dump_json()
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _build_evidence_refs(result: ModelEvaluationResult) -> tuple[UUID, ...]:
    """Extract all ModelEvidenceItem IDs from the ModelEvidenceBundle.

    Args:
        result: The ModelEvaluationResult containing the evidence bundle.

    Returns:
        Tuple of ModelEvidenceItem.item_id values.
    """
    return tuple(item.item_id for item in result.evidence_bundle.items)


def _compute_reward_delta(sv: ModelScoreVector) -> float:
    """Derive a signed reward delta [-1.0, +1.0] from a canonical score vector.

    Formula: mean(all 6 score fields) * 2 - 1
    This maps [0.0, 1.0] score range to [-1.0, +1.0] reward delta range:
      - 1.0 mean score -> +1.0 delta (maximum improvement)
      - 0.5 mean score ->  0.0 delta (neutral)
      - 0.0 mean score -> -1.0 delta (maximum degradation)

    Result is clamped to [-1.0, +1.0] to guard against floating-point drift.

    Args:
        sv: Canonical score vector from ScoringReducer.

    Returns:
        Signed reward delta in [-1.0, +1.0].
    """
    mean_score = (
        sv.correctness
        + sv.safety
        + sv.cost
        + sv.latency
        + sv.maintainability
        + sv.human_time
    ) / 6.0
    delta = mean_score * 2.0 - 1.0
    return max(-1.0, min(1.0, delta))


def _compute_idempotency_key(event_id: UUID, policy_id: UUID, run_id: UUID) -> str:
    """Compute deterministic idempotency key from (event_id, policy_id, run_id).

    Uses SHA-256 of the concatenated UUID hex values (no hyphens).
    Consumers use this to detect and discard duplicate reward events.

    Args:
        event_id: Unique event identifier.
        policy_id: Policy entity receiving the reward.
        run_id: Evaluation run ID.

    Returns:
        64-character lowercase hex digest.
    """
    combined = f"{event_id.hex}{policy_id.hex}{run_id.hex}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


class HandlerRewardBinder:
    """Emits structured reward events to Kafka after an evaluation run.

    Accepts a publisher callable injected at construction time, enabling
    easy mocking in unit tests without touching Kafka infrastructure.

    The publisher signature matches ``PublisherTopicScoped.publish``:
    ``async (event_type, payload, topic, correlation_id) -> bool``.

    Attributes:
        _container: ONEX DI container.
        _publisher: Async callable for publishing to Kafka topics.
        _initialized: Whether the handler has been initialised.
    """

    def __init__(
        self,
        container: ModelONEXContainer,
        publisher: Callable[..., Awaitable[bool]] | None = None,
    ) -> None:
        """Initialise the reward binder handler.

        Args:
            container: ONEX dependency injection container.
            publisher: Async callable matching PublisherTopicScoped.publish.
                If None, a RuntimeHostError is raised on execute().
        """
        self._container = container
        self._publisher = publisher
        self._initialized: bool = False

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def initialize(self, config: dict[str, object]) -> None:
        """Initialise the handler."""
        self._initialized = True
        logger.info("HandlerRewardBinder initialized")

    async def shutdown(self) -> None:
        """Shut down the handler."""
        self._initialized = False
        logger.info("HandlerRewardBinder shutdown")

    async def execute(self, envelope: dict[str, object]) -> ModelHandlerOutput:
        """Emit reward events to Kafka.

        Envelope keys:
            evaluation_result (ModelEvaluationResult): Result from ScoringReducer.
            objective_spec (ModelObjectiveSpec): Spec used for this run.
            correlation_id (UUID): Tracing correlation ID.
            policy_id (UUID): Policy entity receiving the reward.
            policy_type (EnumPolicyType): Policy type for the entity.

        Events are emitted in order:
          1. ModelRewardAssignedEvent (canonical score vector fields inline)
          2. ModelPolicyStateUpdatedEvent

        Raises:
            RuntimeHostError: If the publisher is not configured or required
                envelope fields are missing.
            Exception: Any exception from the publisher is propagated
                (never swallowed silently per contract constraint).

        Returns:
            ModelHandlerOutput wrapping a ModelRewardBinderOutput.
        """
        correlation_id_raw = envelope.get("correlation_id")
        context = ModelInfraErrorContext.with_correlation(
            correlation_id=(
                correlation_id_raw if isinstance(correlation_id_raw, UUID) else None
            ),
            transport_type=EnumInfraTransportType.KAFKA,
            operation="emit_reward_events",
            target_name="reward_binder_kafka",
        )
        corr_id = context.correlation_id
        if corr_id is None:
            raise RuntimeHostError(
                "correlation_id must not be None",
                context=context,
            )

        if self._publisher is None:
            raise RuntimeHostError(
                "HandlerRewardBinder requires a publisher to be configured. "
                "Use RegistryInfraRewardBinderEffect.create_with_publisher().",
                context=context,
            )

        # Extract and validate inputs
        raw_result = envelope.get("evaluation_result")
        if raw_result is None:
            raise RuntimeHostError(
                "emit_reward_events requires 'evaluation_result' in the envelope",
                context=context,
            )
        if not isinstance(raw_result, ModelEvaluationResult):
            raise RuntimeHostError(
                f"Expected ModelEvaluationResult, got {type(raw_result).__name__}",
                context=context,
            )
        result: ModelEvaluationResult = raw_result

        raw_spec = envelope.get("objective_spec")
        if raw_spec is None:
            raise RuntimeHostError(
                "emit_reward_events requires 'objective_spec' in the envelope",
                context=context,
            )
        if not isinstance(raw_spec, ModelObjectiveSpec):
            raise RuntimeHostError(
                f"Expected ModelObjectiveSpec, got {type(raw_spec).__name__}",
                context=context,
            )
        spec: ModelObjectiveSpec = raw_spec

        raw_policy_id = envelope.get("policy_id")
        if raw_policy_id is None:
            raise RuntimeHostError(
                "emit_reward_events requires 'policy_id' in the envelope",
                context=context,
            )
        if not isinstance(raw_policy_id, UUID):
            raise RuntimeHostError(
                f"Expected policy_id as UUID, got {type(raw_policy_id).__name__}",
                context=context,
            )
        policy_id: UUID = raw_policy_id

        raw_policy_type = envelope.get("policy_type")
        if raw_policy_type is None:
            raise RuntimeHostError(
                "emit_reward_events requires 'policy_type' in the envelope",
                context=context,
            )
        if not isinstance(raw_policy_type, EnumPolicyType):
            raise RuntimeHostError(
                f"Expected policy_type as EnumPolicyType, got {type(raw_policy_type).__name__}",
                context=context,
            )
        policy_type: EnumPolicyType = raw_policy_type

        # Compute fingerprint, evidence refs, and canonical reward delta
        fingerprint = _compute_objective_fingerprint(spec)
        evidence_refs = _build_evidence_refs(result)
        sv = result.score_vector
        reward_delta = _compute_reward_delta(sv)
        occurred_at_utc = datetime.now(UTC).isoformat()

        # Event 1: ModelRewardAssignedEvent (canonical score vector fields inline)
        reward_event_id = uuid4()
        reward_event = ModelRewardAssignedEvent(
            event_id=reward_event_id,
            run_id=result.run_id,
            correctness=sv.correctness,
            safety=sv.safety,
            cost=sv.cost,
            latency=sv.latency,
            maintainability=sv.maintainability,
            human_time=sv.human_time,
            evidence_refs=evidence_refs,
            policy_id=policy_id,
            policy_type=policy_type,
            reward_delta=reward_delta,
            objective_id=result.objective_id,
            idempotency_key=_compute_idempotency_key(
                event_id=reward_event_id,
                policy_id=policy_id,
                run_id=result.run_id,
            ),
            occurred_at_utc=occurred_at_utc,
        )
        await self._publish(
            event_type="reward.assigned",
            topic=_TOPIC_REWARD_ASSIGNED,
            payload=json.loads(reward_event.model_dump_json()),
            correlation_id=corr_id,
        )
        logger.info(
            "Emitted canonical ModelRewardAssignedEvent run_id=%s "
            "policy_id=%s policy_type=%s reward_delta=%.4f",
            result.run_id,
            policy_id,
            policy_type.value,
            reward_delta,
        )

        # Event 2: ModelPolicyStateUpdatedEvent
        policy_event = ModelPolicyStateUpdatedEvent(
            run_id=result.run_id,
            old_state=result.policy_state_before,
            new_state=result.policy_state_after,
        )
        await self._publish(
            event_type="policy.state.updated",
            topic=_TOPIC_POLICY_STATE_UPDATED,
            payload=json.loads(policy_event.model_dump_json()),
            correlation_id=corr_id,
        )
        logger.info(
            "Emitted ModelPolicyStateUpdatedEvent run_id=%s",
            result.run_id,
        )

        # Build output
        topics: tuple[str, ...] = (
            _TOPIC_REWARD_ASSIGNED,
            _TOPIC_POLICY_STATE_UPDATED,
        )
        output = ModelRewardBinderOutput(
            success=True,
            correlation_id=corr_id,
            run_id=result.run_id,
            objective_fingerprint=fingerprint,
            reward_assigned_event_ids=(reward_event.event_id,),
            policy_state_updated_event_id=policy_event.event_id,
            topics_published=topics,
        )
        return ModelHandlerOutput.for_compute(
            input_envelope_id=uuid4(),
            correlation_id=corr_id,
            handler_id="handler-reward-binder",
            result=output,
        )

    async def _publish(
        self,
        *,
        event_type: str,
        topic: str,
        payload: object,
        correlation_id: UUID,
    ) -> None:
        """Publish to Kafka via the injected publisher.

        Errors are never swallowed -- any exception from the publisher
        propagates to the caller per contract constraint.

        Args:
            event_type: Logical event type identifier for logging.
            topic: Kafka topic name.
            payload: JSON-serialisable payload dict.
            correlation_id: Tracing correlation ID.
        """
        assert self._publisher is not None  # checked at top of execute()
        await self._publisher(
            event_type=event_type,
            payload=payload,
            topic=topic,
            correlation_id=correlation_id,
        )


__all__: list[str] = [
    "HandlerRewardBinder",
    "_compute_idempotency_key",
    "_compute_objective_fingerprint",
    "_compute_reward_delta",
]
