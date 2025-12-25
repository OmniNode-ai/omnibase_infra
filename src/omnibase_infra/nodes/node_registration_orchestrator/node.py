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

Timeout Coordination (OMN-932):
    The orchestrator also handles RuntimeTick events for timeout detection.
    When a RuntimeTick is received:
    1. The timeout_coordinator queries for overdue entities
    2. Emits timeout events (NodeRegistrationAckTimedOut, NodeLivenessExpired)
    3. Updates projection markers to prevent duplicate emissions

    To wire timeout coordination:
    ```python
    from omnibase_infra.nodes.node_registration_orchestrator import (
        NodeRegistrationOrchestrator,
        TimeoutCoordinator,
    )
    from omnibase_infra.services import TimeoutScanner, TimeoutEmitter

    # Wire dependencies
    timeout_query = TimeoutScanner(projection_reader)
    timeout_emission = TimeoutEmitter(
        timeout_query=timeout_query,
        event_bus=event_bus,
        projector=projector,
    )
    timeout_coordinator = TimeoutCoordinator(timeout_query, timeout_emission)

    # Create orchestrator with timeout coordinator
    orchestrator = NodeRegistrationOrchestrator(container)
    orchestrator.set_timeout_coordinator(timeout_coordinator)

    # Handle RuntimeTick
    result = await orchestrator.handle_runtime_tick(tick)
    ```

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
    - timeout_coordinator.py: RuntimeTick timeout coordinator
    - models/: Input, output, and configuration models (kept for compatibility)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_orchestrator import NodeOrchestrator

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer

    from omnibase_infra.nodes.node_registration_orchestrator.timeout_coordinator import (
        ModelTimeoutCoordinationResult,
        TimeoutCoordinator,
    )
    from omnibase_infra.runtime.models.model_runtime_tick import ModelRuntimeTick


class NodeRegistrationOrchestrator(NodeOrchestrator):
    """Registration orchestrator - workflow driven by contract.yaml.

    This orchestrator coordinates node registration by:
    1. Receiving introspection events (consumed_events in contract)
    2. Calling reducer to compute intents (workflow_definition.execution_graph)
    3. Executing intents via effect (workflow_definition.execution_graph)
    4. Publishing result events (published_events in contract)

    Additionally, handles RuntimeTick events for timeout coordination:
    1. Queries for overdue entities (ack timeouts, liveness expirations)
    2. Emits timeout events
    3. Updates projection markers

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

    Timeout Coordination Usage:
        ```python
        # Wire timeout coordinator
        orchestrator.set_timeout_coordinator(timeout_coordinator)

        # Handle RuntimeTick events
        result = await orchestrator.handle_runtime_tick(tick)
        ```
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the orchestrator.

        Args:
            container: ONEX dependency injection container
        """
        super().__init__(container)
        self._timeout_coordinator: TimeoutCoordinator | None = None

    def set_timeout_coordinator(self, coordinator: TimeoutCoordinator) -> None:
        """Set the timeout coordinator for RuntimeTick coordination.

        The timeout coordinator is used to coordinate RuntimeTick events for
        detecting and emitting timeout events.

        Args:
            coordinator: Configured TimeoutCoordinator instance.

        Example:
            >>> timeout_coordinator = TimeoutCoordinator(timeout_query, timeout_emission)
            >>> orchestrator.set_timeout_coordinator(timeout_coordinator)
        """
        self._timeout_coordinator = coordinator

    @property
    def has_timeout_coordinator(self) -> bool:
        """Check if timeout coordinator is configured."""
        return self._timeout_coordinator is not None

    async def handle_runtime_tick(
        self,
        tick: ModelRuntimeTick,
        domain: str = "registration",
    ) -> ModelTimeoutCoordinationResult:
        """Handle a RuntimeTick event for timeout coordination.

        Delegates to the configured timeout coordinator to coordinate timeouts.
        Uses tick.now for all time-based decisions (never system clock).

        Args:
            tick: The RuntimeTick event with injected 'now'.
            domain: Domain namespace for queries (default: "registration").

        Returns:
            ModelTimeoutCoordinationResult with coordination details.

        Raises:
            RuntimeError: If no timeout coordinator is configured.
            InfraConnectionError: If database/Kafka connection fails.
            InfraTimeoutError: If operations time out.
            InfraUnavailableError: If circuit breaker is open.

        Example:
            >>> result = await orchestrator.handle_runtime_tick(tick)
            >>> print(f"Emitted {result.total_emitted} timeout events")
        """
        if self._timeout_coordinator is None:
            raise RuntimeError(
                "Timeout coordinator not configured. "
                "Call set_timeout_coordinator() before handling RuntimeTick events."
            )

        return await self._timeout_coordinator.coordinate(tick, domain=domain)


__all__ = ["NodeRegistrationOrchestrator"]
