# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Node Pattern Lifecycle Effect -- tier management for validated patterns.

This package provides the NodePatternLifecycleEffect, an effect node that
applies validation verdicts to promote, demote, or suppress patterns
according to the lifecycle tier progression.

Capabilities:
    - lifecycle.update: Apply validation verdict to pattern lifecycle tier

Tier Progression:
    OBSERVED -> SUGGESTED -> SHADOW_APPLY -> PROMOTED -> DEFAULT

Auto-Rollback:
    - 2 consecutive FAIL verdicts: demote one tier
    - 3 consecutive FAIL verdicts: suppress the pattern

Available Exports:
    - NodePatternLifecycleEffect: The declarative effect node
    - ModelLifecycleState: Lifecycle state for a single pattern
    - ModelLifecycleResult: Result of a lifecycle tier update
    - HandlerLifecycleUpdate: Apply verdict and compute tier transition
    - RegistryInfraPatternLifecycle: DI registry

Tracking:
    - OMN-2147: Validation Skeleton -- Orchestrator + Executor
"""

from omnibase_infra.nodes.node_pattern_lifecycle_effect.handlers import (
    HandlerLifecycleUpdate,
)
from omnibase_infra.nodes.node_pattern_lifecycle_effect.models import (
    ModelLifecycleResult,
    ModelLifecycleState,
)
from omnibase_infra.nodes.node_pattern_lifecycle_effect.node import (
    NodePatternLifecycleEffect,
)
from omnibase_infra.nodes.node_pattern_lifecycle_effect.registry import (
    RegistryInfraPatternLifecycle,
)

__all__: list[str] = [
    # Node
    "NodePatternLifecycleEffect",
    # Handlers
    "HandlerLifecycleUpdate",
    # Models
    "ModelLifecycleResult",
    "ModelLifecycleState",
    # Registry
    "RegistryInfraPatternLifecycle",
]
