# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Node Session Lifecycle Reducer — pure FSM for session lifecycle.

This reducer follows the ONEX declarative pattern:
    - DECLARATIVE reducer driven by contract.yaml
    - Zero custom routing logic — all behavior from FSM state_machine
    - Lightweight shell that delegates to NodeReducer base class
    - Pattern: "Contract-driven, FSM state transitions"

FSM Pattern:
    1. Idle — waiting for pipeline start
    2. create_run -> run_created (emits session.index.write + run.write intents)
    3. activate_run -> run_active
    4. end_run -> run_ended (emits session.index.update intent)
    5. reset -> idle (ready for next run)

Concurrency Model:
    Each pipeline instance has its own FSM. Multiple FSMs can be active
    simultaneously, each tracking a different run_id. The session index
    (session.json) tracks all active runs via append-only recent_run_ids.

Design Decisions:
    - 100% Contract-Driven: All FSM logic in YAML, not Python
    - Zero Custom Methods: Base class handles everything
    - Declarative Execution: State transitions defined in state_machine
    - Pure Function Pattern: (state, event) -> (new_state, intents)

Tracking:
    - OMN-2117: Canonical State Nodes
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_reducer import NodeReducer

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.nodes.node_session_lifecycle_reducer.models.model_session_lifecycle_state import (
        ModelSessionLifecycleState,
    )


class NodeSessionLifecycleReducer(
    NodeReducer["ModelSessionLifecycleState", "ModelSessionLifecycleState"]
):
    """Session lifecycle reducer — FSM state transitions driven by contract.yaml.

    This reducer manages the lifecycle of individual pipeline runs:
    idle -> run_created -> run_active -> run_ended -> idle

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


__all__: list[str] = ["NodeSessionLifecycleReducer"]
