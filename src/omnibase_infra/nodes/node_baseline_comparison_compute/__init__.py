# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Node Baseline Comparison Compute -- A/B delta computation.

This package provides the NodeBaselineComparisonCompute, a compute node
that takes paired baseline and candidate run results and computes cost
and outcome deltas to produce a ModelAttributionRecord.

Capabilities:
    - baseline.compare: Compare baseline and candidate run metrics to
      compute cost/outcome deltas and ROI determination.

Available Exports:
    - NodeBaselineComparisonCompute: The declarative compute node
    - ModelBaselineComparisonInput: Input model for paired run results
    - HandlerBaselineComparison: Handler for delta computation
    - RegistryInfraBaselineComparison: DI registry

Tracking:
    - OMN-2155: Baselines + ROI -- A/B Run Infrastructure
"""

from omnibase_infra.nodes.node_baseline_comparison_compute.handlers import (
    HandlerBaselineComparison,
)
from omnibase_infra.nodes.node_baseline_comparison_compute.models import (
    ModelBaselineComparisonInput,
)
from omnibase_infra.nodes.node_baseline_comparison_compute.node import (
    NodeBaselineComparisonCompute,
)
from omnibase_infra.nodes.node_baseline_comparison_compute.registry import (
    RegistryInfraBaselineComparison,
)

__all__: list[str] = [
    # Node
    "NodeBaselineComparisonCompute",
    # Handlers
    "HandlerBaselineComparison",
    # Models
    "ModelBaselineComparisonInput",
    # Registry
    "RegistryInfraBaselineComparison",
]
