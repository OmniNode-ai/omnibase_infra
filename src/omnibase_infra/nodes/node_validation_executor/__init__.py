# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Node Validation Executor -- EFFECT node for running validation checks.

This package provides the NodeValidationExecutor, an effect node that
receives a validation plan and executes checks (typecheck, lint, unit
tests). Each check produces a ModelCheckResult, and results are
aggregated into a ModelExecutorResult.

Available Exports:
    - NodeValidationExecutor: The declarative effect node
    - ModelCheckResult: Individual check result
    - ModelExecutorResult: Aggregated executor result
    - HandlerRunChecks: Handler that executes planned checks
    - RegistryInfraValidationExecutor: DI registry

Tracking:
    - OMN-2147: Validation Executor Effect Node
"""

from omnibase_infra.nodes.node_validation_executor.handlers import HandlerRunChecks
from omnibase_infra.nodes.node_validation_executor.models import (
    ModelCheckResult,
    ModelExecutorResult,
)
from omnibase_infra.nodes.node_validation_executor.node import NodeValidationExecutor
from omnibase_infra.nodes.node_validation_executor.registry import (
    RegistryInfraValidationExecutor,
)

__all__: list[str] = [
    # Node
    "NodeValidationExecutor",
    # Handlers
    "HandlerRunChecks",
    # Models
    "ModelCheckResult",
    "ModelExecutorResult",
    # Registry
    "RegistryInfraValidationExecutor",
]
