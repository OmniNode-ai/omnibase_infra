# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Registration Orchestrator - Declarative workflow coordinator.

This orchestrator uses the declarative pattern where workflow behavior
is 100% driven by contract.yaml, not Python code.

Workflow Pattern:
    1. Receive introspection event (consumed_events in contract)
    2. Call reducer to compute intents (workflow_definition.execution_graph)
    3. Execute intents via effect (workflow_definition.execution_graph)
    4. Publish result events (published_events in contract)

All workflow logic, retry policies, and result aggregation are handled
by the NodeOrchestrator base class using contract.yaml configuration.

Design Decisions:
    - 100% Contract-Driven: All workflow logic in YAML, not Python
    - Zero Custom Methods: Base class handles everything
    - Declarative Execution: Workflow steps defined in execution_graph
    - Retry at Base Class: NodeOrchestrator owns retry policy

Thread Safety:
    This orchestrator is NOT thread-safe. Each instance should handle one
    workflow at a time. For concurrent workflows, create multiple instances.

FUTURE Features (not yet implemented):
    - FUTURE(OMN-973): time_injection - Contract declares time injection from
      RuntimeTick events, but the orchestrator does not yet parse this from
      contract.yaml or wire it into workflow step execution. The infrastructure
      (DispatchContextEnforcer) provides time injection at dispatch time, but
      explicit contract-driven configuration is pending.

    - FUTURE(OMN-930): projection_reader - Contract declares a projection reader
      dependency (ProtocolProjectionReader), but:
      1. The protocol does not yet exist in omnibase_spi.protocols
      2. The orchestrator does not resolve/inject this dependency
      3. The "read_projection" workflow step has no implementation
      The concrete implementation (ProjectionReaderRegistration) exists and
      is ready for use once the SPI protocol and DI wiring are complete.

Related Modules:
    - contract.yaml: Workflow definition and execution graph
    - models/: Input, output, and configuration models (kept for compatibility)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_orchestrator import NodeOrchestrator

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeRegistrationOrchestrator(NodeOrchestrator):
    """Registration orchestrator - workflow driven by contract.yaml.

    This orchestrator coordinates node registration by:
    1. Receiving introspection events (consumed_events in contract)
    2. Calling reducer to compute intents (workflow_definition.execution_graph)
    3. Executing intents via effect (workflow_definition.execution_graph)
    4. Publishing result events (published_events in contract)

    All workflow logic, retry policies, and result aggregation are handled
    by the NodeOrchestrator base class using contract.yaml configuration.

    Example YAML Contract:
        ```yaml
        workflow_coordination:
          workflow_definition:
            workflow_metadata:
              workflow_name: node_registration
              workflow_version: {major: 1, minor: 0, patch: 0}
              execution_mode: sequential
              description: "Node registration workflow"

            execution_graph:
              nodes:
                - node_id: "compute_intents"
                  node_type: reducer
                  description: "Compute registration intents"
                - node_id: "execute_consul"
                  node_type: effect
                  description: "Register with Consul"
                - node_id: "execute_postgres"
                  node_type: effect
                  description: "Register in PostgreSQL"

            coordination_rules:
              parallel_execution_allowed: false
              failure_recovery_strategy: retry
              max_retries: 3
              timeout_ms: 30000
        ```

    Usage:
        ```python
        from omnibase_core.models.container import ModelONEXContainer

        # Create and initialize
        container = ModelONEXContainer()
        orchestrator = NodeRegistrationOrchestrator(container)

        # Workflow definition must be set (from contract or manually)
        orchestrator.workflow_definition = ModelWorkflowDefinition(...)

        # Process input
        result = await orchestrator.process(input_data)
        ```
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the orchestrator.

        Args:
            container: ONEX dependency injection container
        """
        super().__init__(container)


__all__ = ["NodeRegistrationOrchestrator"]
