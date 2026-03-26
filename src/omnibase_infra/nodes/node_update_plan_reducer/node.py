# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Node Update Plan Reducer — pure FSM for update plan lifecycle.

This reducer follows the ONEX declarative pattern:
    - DECLARATIVE reducer driven by contract.yaml
    - Zero custom routing logic — all behavior from FSM state_machine
    - Lightweight shell that delegates to NodeReducer base class
    - Pattern: "Contract-driven, FSM state transitions"

FSM Pattern:
    1. Idle — waiting for impact analysis result
    2. create_plan -> created (plan created, tasks assigned)
    3. post_comment -> comment_posted (PR comment posted)
    4. emit_yaml -> yaml_emitted (YAML plan artifact emitted)
    5. close -> closed (plan fully processed)
    6. waive -> waived (plan explicitly waived, e.g. merge_policy == "none")

Design Decisions:
    - 100% Contract-Driven: All FSM logic in YAML, not Python
    - Zero Custom Methods: Base class handles everything
    - Declarative Execution: State transitions defined in state_machine
    - Pure Function Pattern: (state, event) -> (new_state, intents)

Tracking:
    - OMN-3943: Task 6 — Update Plan REDUCER Node
    - OMN-3925: Artifact Reconciliation + Update Planning MVP
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_reducer import NodeReducer

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.nodes.node_update_plan_reducer.models.model_update_plan_state import (
        ModelUpdatePlanState,
    )


class NodeUpdatePlanReducer(
    NodeReducer["ModelUpdatePlanState", "ModelUpdatePlanState"]
):
    """Update plan reducer — FSM state transitions driven by contract.yaml.

    This reducer manages the lifecycle of an artifact update plan:
    idle -> created -> comment_posted -> yaml_emitted -> closed
                                                      \\-> waived

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


__all__: list[str] = ["NodeUpdatePlanReducer"]
