# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Registration Orchestrator - Declarative workflow coordinator.

This orchestrator follows the ONEX declarative pattern:
    - DECLARATIVE orchestrator driven by contract.yaml
    - Zero custom routing logic - all behavior from workflow_definition
    - Lightweight shell that delegates to TimeoutCoordinator and HeartbeatHandler
    - Used for ONEX-compliant runtime execution via RuntimeHostProcess
    - Pattern: "Contract-driven, handlers wired externally"

Extends NodeOrchestrator from omnibase_core for workflow-driven coordination.
All workflow logic is 100% driven by contract.yaml, not Python code.

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

Heartbeat Handling (OMN-1006):
    The orchestrator handles node heartbeat events for liveness tracking.
    When a heartbeat is received:
    1. The heartbeat_handler updates last_heartbeat_at in the projection
    2. Extends the liveness_deadline based on the configured liveness window
    3. Returns a result with the updated timestamps

    To wire heartbeat handling:
    ```python
    from omnibase_infra.nodes.node_registration_orchestrator.handlers import HandlerNodeHeartbeat

    # Wire heartbeat handler with projection dependencies
    heartbeat_handler = HandlerNodeHeartbeat(
        projection_reader=projection_reader,
        projector=projector,
        liveness_window_seconds=90.0,
    )

    # Create orchestrator with heartbeat handler
    orchestrator = NodeRegistrationOrchestrator(container)
    orchestrator.set_heartbeat_handler(heartbeat_handler)

    # Handle heartbeat events
    result = await orchestrator.handle_heartbeat(heartbeat_event)
    ```

Design Decisions:
    - 100% Contract-Driven: All workflow logic in YAML, not Python
    - Zero Custom Methods: Base class handles everything
    - Declarative Execution: Workflow steps defined in execution_graph
    - Retry at Base Class: NodeOrchestrator owns retry policy

Coroutine Safety:
    This orchestrator is NOT coroutine-safe. Each instance should handle one
    workflow at a time. For concurrent workflows, create multiple instances.

Implemented Features:
    - OMN-973 (Time Injection): The DispatchContextEnforcer provides time injection
      at dispatch time. Orchestrators receive injected `now` timestamps from
      RuntimeTick events, enabling deterministic timeout evaluation in workflow
      steps. See: omnibase_infra/runtime/dispatch_context_enforcer.py

    - OMN-930 (Projection Reader): ProtocolProjectionReader is defined in
      omnibase_spi.protocols (merged in omnibase_spi#44). The orchestrator can
      resolve and inject projection readers as dependencies, enabling the
      "read_projection" workflow step to query current registration state.
      See: omnibase_spi/protocols/protocol_projection_reader.py

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

    from omnibase_infra.models.registration import ModelNodeHeartbeatEvent
    from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
        HandlerNodeHeartbeat,
        ModelHeartbeatHandlerResult,
    )
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
        self._heartbeat_handler: HandlerNodeHeartbeat | None = None

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

    def set_heartbeat_handler(self, handler: HandlerNodeHeartbeat) -> None:
        """Set the heartbeat handler for processing node heartbeat events.

        The heartbeat handler is used to update last_heartbeat_at and extend
        liveness_deadline when heartbeat events are received from active nodes.

        Args:
            handler: Configured HandlerNodeHeartbeat instance.

        Example:
            >>> heartbeat_handler = HandlerNodeHeartbeat(
            ...     projection_reader=reader,
            ...     projector=projector,
            ... )
            >>> orchestrator.set_heartbeat_handler(heartbeat_handler)
        """
        self._heartbeat_handler = handler

    @property
    def has_heartbeat_handler(self) -> bool:
        """Check if heartbeat handler is configured."""
        return self._heartbeat_handler is not None

    async def handle_heartbeat(
        self,
        event: ModelNodeHeartbeatEvent,
        domain: str = "registration",
    ) -> ModelHeartbeatHandlerResult:
        """Handle a node heartbeat event for liveness tracking.

        Delegates to the configured heartbeat handler to update the registration
        projection with the heartbeat timestamp and extended liveness deadline.

        Args:
            event: The heartbeat event from an active node.
            domain: Domain namespace for projection lookup (default: "registration").

        Returns:
            ModelHeartbeatHandlerResult with processing outcome including:
            - success: Whether the heartbeat was processed successfully
            - last_heartbeat_at: Updated heartbeat timestamp
            - liveness_deadline: Extended liveness deadline
            - node_not_found: True if no projection exists for this node

        Raises:
            RuntimeError: If no heartbeat handler is configured.
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If database operation times out.
            RuntimeHostError: For other infrastructure errors.

        Example:
            >>> result = await orchestrator.handle_heartbeat(heartbeat_event)
            >>> if result.success:
            ...     print(f"Heartbeat processed, deadline: {result.liveness_deadline}")
        """
        if self._heartbeat_handler is None:
            raise RuntimeError(
                "Heartbeat handler not configured. "
                "Call set_heartbeat_handler() before handling heartbeat events."
            )

        return await self._heartbeat_handler.handle(event, domain=domain)


__all__ = ["NodeRegistrationOrchestrator"]
