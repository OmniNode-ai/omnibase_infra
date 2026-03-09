# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Handler that serializes an artifact update plan to YAML and emits it as an event.

Receives a ModelUpdatePlan, serializes it to YAML, and returns a
ModelYamlEmitResult containing the YAML payload and target Kafka topic.

Emission target:
    onex.evt.artifact.update-plan-emitted.v1

This handler performs NO filesystem writes. It returns a pure data payload
that the orchestrator layer routes to the event bus.

Tracking:
    - OMN-3944: Task 7 — Reconciliation ORCHESTRATOR Node
    - OMN-3925: Artifact Reconciliation + Update Planning MVP
"""

from __future__ import annotations

import logging
from uuid import UUID

import yaml

from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
)
from omnibase_infra.enums.generated import EnumArtifactTopic
from omnibase_infra.nodes.node_artifact_reconciliation_orchestrator.models.model_yaml_emit_result import (
    ModelYamlEmitResult,
)
from omnibase_infra.nodes.node_update_plan_reducer.models.model_update_plan import (
    ModelUpdatePlan,
)

logger = logging.getLogger(__name__)

_EMIT_TOPIC: EnumArtifactTopic = EnumArtifactTopic.EVT_UPDATE_PLAN_EMITTED_V1


class HandlerPlanToYaml:
    """Serialize an artifact update plan to YAML for event emission.

    Pure transformation handler: takes a ModelUpdatePlan and returns a
    ModelYamlEmitResult. No I/O is performed; the orchestrator routes the
    output to the event bus (onex.evt.artifact.update-plan-emitted.v1).

    Attributes:
        handler_id: Unique handler identifier.
        handler_type: Architectural role (INFRA_HANDLER).
        handler_category: Behavioral classification (COMPUTE — pure transformation).
    """

    @property
    def handler_id(self) -> str:
        """Unique handler identifier."""
        return "handler-plan-to-yaml"

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role: infrastructure handler."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification: pure compute transformation."""
        return EnumHandlerTypeCategory.COMPUTE

    def serialize_plan(
        self,
        plan: ModelUpdatePlan,
        correlation_id: UUID | None = None,
    ) -> ModelYamlEmitResult:
        """Serialize a ModelUpdatePlan to YAML and return the event payload.

        Converts the plan to a Python dict via Pydantic's model_dump(), then
        serializes to YAML. The result is intended for emission on the
        ``onex.evt.artifact.update-plan-emitted.v1`` Kafka topic.

        Args:
            plan: The update plan to serialize.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            ModelYamlEmitResult with yaml_payload (YAML string), plan_id
            (UUID), and target Kafka topic.
        """
        plan_dict = plan.model_dump(mode="json")
        yaml_payload = yaml.dump(
            plan_dict,
            default_flow_style=False,
            sort_keys=True,
            allow_unicode=True,
        )

        logger.info(
            "Serialized plan to YAML: plan_id=%s bytes=%d",
            plan.plan_id,
            len(yaml_payload),
            extra={"plan_id": str(plan.plan_id)},
        )

        return ModelYamlEmitResult(
            yaml_payload=yaml_payload,
            plan_id=plan.plan_id,
            topic=_EMIT_TOPIC,
        )


__all__: list[str] = ["HandlerPlanToYaml"]
