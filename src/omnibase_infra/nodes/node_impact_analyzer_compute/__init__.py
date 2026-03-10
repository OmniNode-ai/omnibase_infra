# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Node Impact Analyzer Compute -- deterministic artifact impact scoring.

This package provides NodeImpactAnalyzerCompute, a pure compute node that
matches change triggers against the artifact registry using fnmatch patterns
and applies the OMN-3925 scoring table to produce ModelImpactAnalysisResult.

Capabilities:
    - artifact.analyze_impact: Score changed files against registry triggers
      to determine artifact impact strength and required actions.

Available Exports:
    - NodeImpactAnalyzerCompute: The declarative compute node
    - HandlerImpactAnalysis: Handler implementing the scoring logic
    - ModelImpactedArtifact: Per-artifact impact model
    - ModelImpactAnalysisResult: Aggregated result model
    - RegistryInfraImpactAnalyzer: DI registry

Tracking:
    - OMN-3935: Task 3 - Impact Analyzer COMPUTE Node
    - OMN-3925: Artifact Reconciliation + Update Planning MVP
"""

from omnibase_infra.nodes.node_impact_analyzer_compute.handlers import (
    HandlerImpactAnalysis,
)
from omnibase_infra.nodes.node_impact_analyzer_compute.models.model_impact_analysis_result import (
    ModelImpactAnalysisResult,
)
from omnibase_infra.nodes.node_impact_analyzer_compute.models.model_impacted_artifact import (
    ModelImpactedArtifact,
)
from omnibase_infra.nodes.node_impact_analyzer_compute.node import (
    NodeImpactAnalyzerCompute,
)
from omnibase_infra.nodes.node_impact_analyzer_compute.registry import (
    RegistryInfraImpactAnalyzer,
)

__all__: list[str] = [
    # Node
    "NodeImpactAnalyzerCompute",
    # Handlers
    "HandlerImpactAnalysis",
    # Models
    "ModelImpactedArtifact",
    "ModelImpactAnalysisResult",
    # Registry
    "RegistryInfraImpactAnalyzer",
]
