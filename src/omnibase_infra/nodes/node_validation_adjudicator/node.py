# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Node Validation Adjudicator — pure FSM reducer for verdict production.

This reducer follows the ONEX declarative pattern:
    - DECLARATIVE reducer driven by contract.yaml
    - Zero custom routing logic — all behavior from FSM state_machine
    - Lightweight shell that delegates to NodeReducer base class
    - Pattern: "Contract-driven, FSM state transitions"

FSM Pattern:
    1. Collecting — accumulating check results from executor
    2. begin_adjudication -> adjudicating (apply scoring policy)
    3. emit_verdict -> verdict_emitted (produce PASS/FAIL/QUARANTINE)
    4. reset -> collecting (ready for next validation run)

Design Decisions:
    - 100% Contract-Driven: All FSM logic in YAML, not Python
    - Zero Custom Methods: Base class handles everything
    - Declarative Execution: State transitions defined in state_machine
    - Pure Function Pattern: (state, event) -> (new_state, intents)

Tracking:
    - OMN-2147: Validation Skeleton — Orchestrator + Executor
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_reducer import NodeReducer

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.nodes.node_validation_adjudicator.models.model_adjudicator_state import (
        ModelAdjudicatorState,
    )


class NodeValidationAdjudicator(
    NodeReducer["ModelAdjudicatorState", "ModelAdjudicatorState"]
):
    """Validation adjudicator reducer — FSM state transitions driven by contract.yaml.

    This reducer aggregates check results from the validation executor,
    applies scoring policy, and produces a PASS/FAIL/QUARANTINE verdict:
    collecting -> adjudicating -> verdict_emitted -> collecting

    All state transition logic, intent emission, and validation are driven
    entirely by the contract.yaml FSM configuration.

    Attributes:
        container: ONEX dependency injection container.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the reducer.

        Args:
            container: ONEX dependency injection container.
        """
        super().__init__(container)


__all__: list[str] = ["NodeValidationAdjudicator"]
