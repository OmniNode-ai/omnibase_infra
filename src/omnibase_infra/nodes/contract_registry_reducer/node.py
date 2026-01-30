# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Contract Registry Reducer - FSM-driven declarative reducer.

This reducer follows the ONEX declarative pattern:
    - DECLARATIVE reducer driven by contract.yaml
    - Zero custom routing logic - all behavior from FSM state_machine
    - Lightweight shell that delegates to NodeReducer base class
    - Used for ONEX-compliant runtime execution via RuntimeHostProcess
    - Pattern: "Contract-driven, FSM state transitions"

Extends NodeReducer from omnibase_core for FSM-driven state management.
All state transition logic is 100% driven by contract.yaml, not Python code.

Purpose:
    This reducer projects contract registration events to PostgreSQL:
    1. Receives contract registration events from Kafka
    2. Validates and deduplicates events using Kafka offset tracking
    3. Emits PostgreSQL upsert intents for the contract_registry table
    4. Processes heartbeat events to update last_seen timestamps
    5. Handles deregistration events to mark contracts as inactive

FSM Pattern:
    1. Receive contract event (trigger: contract_registered, heartbeat, deregistered)
    2. Check idempotency via Kafka offset (skip duplicates)
    3. Emit PostgreSQL intent for persistence
    4. Update state with processed event tracking

Design Decisions:
    - 100% Contract-Driven: All FSM logic in YAML, not Python
    - Zero Custom Methods: Base class handles everything
    - Declarative Execution: State transitions defined in state_machine
    - Pure Function Pattern: (state, event) -> (new_state, intents)

Related Modules:
    - contract.yaml: FSM state machine and intent emission configuration
    - models/: State model for idempotency and statistics tracking
    - OMN-1653: Contract registry reducer implementation

Testing Notes:
    - Unit tests: tests/unit/nodes/contract_registry_reducer/
    - Integration tests: Required before production
        * Intent emission through RuntimeHostProcess
        * End-to-end contract workflow with PostgreSQL mocks
        * Kafka offset-based idempotency verification
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_reducer import NodeReducer

if TYPE_CHECKING:
    # Import needed for string annotation in class definition
    from omnibase_infra.nodes.contract_registry_reducer.models.model_contract_registry_state import (
        ModelContractRegistryState,
    )


class NodeContractRegistryReducer(
    NodeReducer["ModelContractRegistryState", "ModelContractRegistryState"]
):
    """Contract registry reducer - FSM state transitions driven by contract.yaml.

    This is a purely declarative reducer. All behavior is defined in contract.yaml.
    No custom Python logic is required - the base NodeReducer class handles all
    FSM-driven state transitions via the contract configuration.

    This reducer processes contract registration workflows by:
    1. Receiving contract events (registration, heartbeat, deregistration)
    2. Executing FSM transitions defined in contract.yaml
    3. Emitting PostgreSQL upsert intents for the contract_registry table
    4. Tracking processed events for idempotency via Kafka offsets

    All state transition logic, intent emission, and validation are driven
    entirely by the contract.yaml FSM configuration.

    Example YAML Contract (state_machine section):
        ```yaml
        state_machine:
          state_machine_name: contract_registry_fsm
          initial_state: idle
          states:
            - state_name: idle
              description: "Waiting for contract events"
            - state_name: processing
              description: "Processing contract event"
              entry_actions:
                - emit_postgres_upsert_intent
          transitions:
            - from_state: idle
              to_state: processing
              trigger: contract_received
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
        reducer = NodeContractRegistryReducer(container)

        # Process input via FSM
        input_data = ModelReducerInput(
            data=contract_event,
            reduction_type=EnumReductionType.MERGE,
            metadata=ModelReducerMetadata(trigger="contract_registered"),
        )
        result = await reducer.process(input_data)
        ```
    """


__all__ = ["NodeContractRegistryReducer"]
