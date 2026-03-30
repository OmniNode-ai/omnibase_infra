# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Node Savings Estimation Compute -- token savings calculation.

This package provides the NodeSavingsEstimationCompute, a compute node
that takes injection effectiveness data and computes token and cost
savings using tiered model pricing.

Capabilities:
    - savings.estimate: Compute savings from effectiveness data using
      tiered model pricing (Opus/Sonnet input/output rates).

Available Exports:
    - NodeSavingsEstimationCompute: The declarative compute node
    - ModelSavingsEstimationInput: Input model for effectiveness data
    - ModelSavingsEstimate: Output model for computed savings
    - HandlerSavingsEstimation: Handler for savings computation
    - RegistryInfraSavingsEstimation: DI registry

Tracking:
    - OMN-6964: Token savings emitter
"""

from omnibase_infra.nodes.node_savings_estimation_compute.handlers import (
    HandlerSavingsEstimation,
)
from omnibase_infra.nodes.node_savings_estimation_compute.models import (
    ModelSavingsEstimate,
    ModelSavingsEstimationInput,
)
from omnibase_infra.nodes.node_savings_estimation_compute.node import (
    NodeSavingsEstimationCompute,
)
from omnibase_infra.nodes.node_savings_estimation_compute.registry import (
    RegistryInfraSavingsEstimation,
)

__all__: list[str] = [
    # Node
    "NodeSavingsEstimationCompute",
    # Handlers
    "HandlerSavingsEstimation",
    # Models
    "ModelSavingsEstimationInput",
    "ModelSavingsEstimate",
    # Registry
    "RegistryInfraSavingsEstimation",
]
