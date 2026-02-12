# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Node Validation Adjudicator — pure FSM reducer for verdict production.

This package provides the NodeValidationAdjudicator, a pure FSM reducer
that aggregates check results and produces a PASS/FAIL/QUARANTINE verdict:
collecting -> adjudicating -> verdict_emitted -> collecting.

Available Exports:
    - NodeValidationAdjudicator: The declarative reducer node
    - ModelAdjudicatorState: Immutable FSM state model
    - ModelVerdict: Validation verdict output model
    - RegistryInfraValidationAdjudicator: DI registry

Tracking:
    - OMN-2147: Validation Skeleton -- Orchestrator + Executor
"""

from omnibase_infra.nodes.node_validation_adjudicator.models import (
    ModelAdjudicatorState,
    ModelVerdict,
)
from omnibase_infra.nodes.node_validation_adjudicator.node import (
    NodeValidationAdjudicator,
)
from omnibase_infra.nodes.node_validation_adjudicator.registry import (
    RegistryInfraValidationAdjudicator,
)

__all__: list[str] = [
    "ModelAdjudicatorState",
    "ModelVerdict",
    "NodeValidationAdjudicator",
    "RegistryInfraValidationAdjudicator",
]
