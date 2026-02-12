# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Node Validation Orchestrator -- validation pipeline coordinator.

This package provides the NodeValidationOrchestrator, a declarative
orchestrator that coordinates the validation pipeline: receives pattern
candidates, builds validation plans, dispatches to executor and adjudicator,
and publishes results.

Available Exports:
    - NodeValidationOrchestrator: The declarative orchestrator node
    - ModelPatternCandidate: Input model for pattern candidates
    - ModelPlannedCheck: Individual check within a validation plan
    - ModelValidationPlan: Output model for validation plans
    - RegistryInfraValidationOrchestrator: DI registry

Tracking:
    - OMN-2147: Validation Skeleton Orchestrator + Executor
"""

from omnibase_infra.nodes.node_validation_orchestrator.models import (
    ModelPatternCandidate,
    ModelPlannedCheck,
    ModelValidationPlan,
)
from omnibase_infra.nodes.node_validation_orchestrator.node import (
    NodeValidationOrchestrator,
)
from omnibase_infra.nodes.node_validation_orchestrator.registry import (
    RegistryInfraValidationOrchestrator,
)

__all__: list[str] = [
    "ModelPatternCandidate",
    "ModelPlannedCheck",
    "ModelValidationPlan",
    "NodeValidationOrchestrator",
    "RegistryInfraValidationOrchestrator",
]
