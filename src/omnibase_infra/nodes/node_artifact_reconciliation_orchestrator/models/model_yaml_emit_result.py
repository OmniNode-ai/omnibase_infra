# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Result model for YAML plan emission operations.

Tracking:
    - OMN-3944: Task 7 — Reconciliation ORCHESTRATOR Node
    - OMN-3925: Artifact Reconciliation + Update Planning MVP
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict


class ModelYamlEmitResult(BaseModel):
    """Result of serializing an update plan to YAML for event emission.

    Returned by HandlerPlanToYaml after serializing a ModelUpdatePlan to
    YAML. The yaml_payload is intended for emission on the
    onex.evt.artifact.update-plan-emitted.v1 Kafka topic.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    yaml_payload: str
    """YAML serialization of the update plan."""

    plan_id: UUID
    """UUID of the serialized plan."""

    topic: str
    """Target Kafka topic for emission."""


__all__: list[str] = ["ModelYamlEmitResult"]
