# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Node Savings Estimation Compute -- tiered attribution.

This package provides the NodeSavingsEstimationCompute, a compute node
that estimates how much token spend the ONEX platform saved in a session
by comparing actual costs against a counterfactual reference model.

Capabilities:
    - savings.estimate: Compute tiered token savings attribution

Available Exports:
    - NodeSavingsEstimationCompute: The declarative compute node
    - ModelSavingsInput: Input model for session signals
    - HandlerSavingsEstimator: Handler for savings computation
    - RegistryInfraSavingsEstimation: DI registry

Tracking:
    - OMN-5547: Create HandlerSavingsEstimator compute handler
"""

from omnibase_infra.nodes.node_savings_estimation_compute.handlers import (
    HandlerSavingsEstimator,
)
from omnibase_infra.nodes.node_savings_estimation_compute.models import (
    ModelSavingsInput,
)
from omnibase_infra.nodes.node_savings_estimation_compute.node import (
    NodeSavingsEstimationCompute,
)
from omnibase_infra.nodes.node_savings_estimation_compute.registry import (
    RegistryInfraSavingsEstimation,
)

__all__: list[str] = [
    "HandlerSavingsEstimator",
    "ModelSavingsInput",
    "NodeSavingsEstimationCompute",
    "RegistryInfraSavingsEstimation",
]
