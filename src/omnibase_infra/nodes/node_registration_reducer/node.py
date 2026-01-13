# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Registration Reducer - FSM-driven declarative reducer.

This reducer follows the ONEX declarative pattern:
    - DECLARATIVE reducer driven by contract.yaml
    - Zero custom routing logic - all behavior from FSM state_machine
    - Lightweight shell that delegates to NodeReducer base class
    - Used for ONEX-compliant runtime execution via RuntimeHostProcess
    - Pattern: "Contract-driven, FSM state transitions"

Extends NodeReducer from omnibase_core for FSM-driven state management.
All state transition logic is 100% driven by contract.yaml, not Python code.

FSM Pattern:
    1. Receive introspection event (trigger: introspection_received)
    2. FSM transitions idle -> pending (emits registration intents)
    3. Receive confirmation events (triggers: consul_confirmed, postgres_confirmed)
    4. FSM transitions pending -> partial -> complete
    5. On errors, FSM transitions to failed
    6. Reset events allow retry (failed/complete -> idle)

All state transitions, intent emission, and validation are handled by the
NodeReducer base class using contract.yaml FSM configuration.

Design Decisions:
    - 100% Contract-Driven: All FSM logic in YAML, not Python
    - Zero Custom Methods: Base class handles everything
    - Declarative Execution: State transitions defined in state_machine
    - Pure Function Pattern: (state, event) -> (new_state, intents)

Related Modules:
    - contract.yaml: FSM state machine and intent emission configuration
    - models/: Validation result and state models
    - registry/: Dependency injection registry

Testing Notes:
    - Unit tests: tests/unit/nodes/node_registration_reducer/
    - Integration tests: REQUIRED before production (OMN-1263)
        WARNING: This reducer MUST NOT be deployed to production without
        the following integration tests implemented and passing:
        * Intent emission through RuntimeHostProcess
        * End-to-end registration workflow with Consul/PostgreSQL mocks
        * FSM state persistence via RegistrationProjector
    - Test coverage matrix: docs/decisions/adr-any-type-pydantic-workaround.md

Tracking:
    - OMN-1104: Declarative reducer refactoring
    - OMN-1263: Integration test coverage (https://linear.app/omninode/issue/OMN-1263)
        Pre-production blocker - prioritize before deployment
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_reducer import NodeReducer

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer

    # Import needed for string annotation in class definition (NodeReducer["ModelRegistrationState", ...])
    from omnibase_infra.nodes.node_registration_reducer.models.model_registration_state import (
        ModelRegistrationState,
    )


class NodeRegistrationReducer(
    NodeReducer["ModelRegistrationState", "ModelRegistrationState"]
):
    """Registration reducer - FSM state transitions driven by contract.yaml.

    This reducer processes node registration workflows by:
    1. Receiving introspection events (via process() method)
    2. Executing FSM transitions defined in contract.yaml
    3. Emitting registration intents for Consul and PostgreSQL
    4. Processing confirmation events to complete registration

    All state transition logic, intent emission, and validation are driven
    entirely by the contract.yaml FSM configuration.

    Example YAML Contract (state_machine section):
        ```yaml
        state_machine:
          state_machine_name: registration_fsm
          initial_state: idle
          states:
            - state_name: idle
              description: "Waiting for introspection event"
            - state_name: pending
              description: "Registration initiated"
              entry_actions:
                - emit_consul_intent
                - emit_postgres_intent
            - state_name: partial
              description: "One backend confirmed"
            - state_name: complete
              description: "Registration successful"
            - state_name: failed
              description: "Registration failed"
          transitions:
            - from_state: idle
              to_state: pending
              trigger: introspection_received
              conditions:
                - expression: "node_id is_present"
                  required: true
        ```

    Usage:
        ```python
        from omnibase_core.models.container import ModelONEXContainer
        from omnibase_core.models.reducer import ModelReducerInput
        from omnibase_core.enums import EnumReductionType

        # Create and initialize
        container = ModelONEXContainer()
        reducer = NodeRegistrationReducer(container)

        # Process input via FSM
        input_data = ModelReducerInput(
            data=introspection_event,
            reduction_type=EnumReductionType.MERGE,
            metadata=ModelReducerMetadata(trigger="introspection_received"),
        )
        result = await reducer.process(input_data)
        ```
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the reducer.

        Args:
            container: ONEX dependency injection container
        """
        super().__init__(container)


__all__ = ["NodeRegistrationReducer"]
