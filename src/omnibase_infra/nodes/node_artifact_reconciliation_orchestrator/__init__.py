# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Node: artifact reconciliation orchestrator.

Coordinates the artifact reconciliation pipeline: receives update plans,
posts PR comments, and emits YAML plan events.

Available Exports:
    - NodeArtifactReconciliationOrchestrator: The declarative orchestrator node
    - HandlerPlanToPRComment: Posts update plans as GitHub PR comments
    - HandlerPlanToYaml: Serializes update plans to YAML for event emission
    - RegistryInfraArtifactReconciliationOrchestrator: DI registry

Tracking:
    - OMN-3944: Task 7 — Reconciliation ORCHESTRATOR Node
    - OMN-3925: Artifact Reconciliation + Update Planning MVP
"""

from omnibase_infra.nodes.node_artifact_reconciliation_orchestrator.handlers import (
    HandlerPlanToPRComment,
    HandlerPlanToYaml,
)
from omnibase_infra.nodes.node_artifact_reconciliation_orchestrator.models import (
    ModelPRCommentResult,
    ModelYamlEmitResult,
)
from omnibase_infra.nodes.node_artifact_reconciliation_orchestrator.node import (
    NodeArtifactReconciliationOrchestrator,
)
from omnibase_infra.nodes.node_artifact_reconciliation_orchestrator.registry import (
    RegistryInfraArtifactReconciliationOrchestrator,
)

__all__: list[str] = [
    "HandlerPlanToPRComment",
    "HandlerPlanToYaml",
    "ModelPRCommentResult",
    "ModelYamlEmitResult",
    "NodeArtifactReconciliationOrchestrator",
    "RegistryInfraArtifactReconciliationOrchestrator",
]
