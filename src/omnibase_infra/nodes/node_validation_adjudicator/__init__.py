# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Node Validation Adjudicator — pure FSM reducer for verdict production.

This package provides the NodeValidationAdjudicator, a pure FSM reducer
that aggregates check results and produces a PASS/FAIL/QUARANTINE verdict:
collecting -> adjudicating -> verdict_emitted -> collecting.

The FSM state model (``ModelAdjudicatorState``) and verdict model
(``ModelVerdict``) live in ``omnibase_infra.models.validation`` beside the
events the pipeline's other nodes exchange; the dispatch handler is
``omnibase_infra.handlers.handler_validation_adjudicate``.

Available Exports:
    - NodeValidationAdjudicator: The declarative reducer node
    - RegistryInfraValidationAdjudicator: DI registry

Tracking:
    - OMN-2147: Validation Skeleton -- Orchestrator + Executor
    - OMN-21056: Validation pipeline end to end on the lab
"""

from omnibase_infra.nodes.node_validation_adjudicator.node import (
    NodeValidationAdjudicator,
)
from omnibase_infra.nodes.node_validation_adjudicator.registry import (
    RegistryInfraValidationAdjudicator,
)

__all__: list[str] = [
    "NodeValidationAdjudicator",
    "RegistryInfraValidationAdjudicator",
]
