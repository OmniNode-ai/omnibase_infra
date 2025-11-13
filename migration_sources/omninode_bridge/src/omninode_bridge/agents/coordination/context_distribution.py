"""
Agent context distribution system for parallel coordination.

This module provides agent-specific context packaging and distribution
with <200ms performance target per agent.
"""

import logging
import sys
import time
from typing import Any, Optional
from uuid import uuid4

from omninode_bridge.agents.coordination.context_models import (
    AgentAssignment,
    AgentContext,
    ContextDistributionMetrics,
    ContextUpdateRequest,
    CoordinationMetadata,
    CoordinationProtocols,
    ResourceAllocation,
    SharedIntelligence,
)
from omninode_bridge.agents.coordination.thread_safe_state import ThreadSafeState
from omninode_bridge.agents.metrics.collector import MetricsCollector

logger = logging.getLogger(__name__)


class ContextDistributor:
    """
    Agent context distribution system.

    Distributes agent-specific context packages with coordination metadata,
    shared intelligence, and resource allocation for parallel workflows.

    Performance Target: <200ms per agent distribution
    Throughput: 50+ concurrent agents
    Thread Safety: Uses ThreadSafeState for safe context storage

    Features:
    - Agent-specific context packaging
    - Coordination metadata injection
    - Shared intelligence distribution
    - Resource allocation per agent
    - Context versioning and updates
    - Thread-safe context storage

    Example:
        ```python
        from omninode_bridge.agents.coordination import ThreadSafeState
        from omninode_bridge.agents.metrics import MetricsCollector

        state = ThreadSafeState()
        metrics = MetricsCollector()
        await metrics.start()

        distributor = ContextDistributor(
            state=state,
            metrics_collector=metrics
        )

        # Distribute context to agents
        contexts = await distributor.distribute_agent_context(
            coordination_state={
                "coordination_id": "coord-123",
                "session_id": "session-456"
            },
            agent_assignments={
                "model_gen": {
                    "objective": "Generate Pydantic models",
                    "tasks": ["parse_contract", "generate_models"],
                    "input_data": {"contract_path": "./contract.yaml"}
                },
                "validator_gen": {
                    "objective": "Generate validators",
                    "tasks": ["generate_validators"],
                    "dependencies": ["model_gen"]
                }
            }
        )

        # Retrieve agent context
        context = distributor.get_agent_context("coord-123", "model_gen")
        ```
    """

    def __init__(
        self,
        state: ThreadSafeState,
        metrics_collector: Optional[MetricsCollector] = None,
        default_resource_allocation: Optional[ResourceAllocation] = None,
        default_coordination_protocols: Optional[CoordinationProtocols] = None,
    ):
        """
        Initialize context distributor.

        Args:
            state: ThreadSafeState instance for context storage
            metrics_collector: Optional MetricsCollector for distribution tracking
            default_resource_allocation: Default resource limits for agents
            default_coordination_protocols: Default coordination protocols
        """
        self.state = state
        self.metrics = metrics_collector

        # Default configurations
        self.default_resource_allocation = (
            default_resource_allocation or ResourceAllocation()
        )
        self.default_coordination_protocols = (
            default_coordination_protocols or CoordinationProtocols()
        )

        # Initialize state keys
        self._initialize_state()

        logger.info("ContextDistributor initialized")

    async def distribute_agent_context(
        self,
        coordination_state: dict[str, Any],
        agent_assignments: dict[str, dict[str, Any]],
        shared_intelligence: Optional[SharedIntelligence] = None,
        resource_allocations: Optional[dict[str, ResourceAllocation]] = None,
        coordination_protocols: Optional[dict[str, CoordinationProtocols]] = None,
    ) -> dict[str, AgentContext]:
        """
        Distribute specialized context to each parallel agent.

        Performance Target: <200ms per agent

        Args:
            coordination_state: Coordination workflow state (must contain
                coordination_id and session_id)
            agent_assignments: Agent ID to assignment mapping
            shared_intelligence: Optional shared intelligence (type registry, patterns, etc.)
            resource_allocations: Optional per-agent resource allocations
            coordination_protocols: Optional per-agent coordination protocols

        Returns:
            Dictionary mapping agent_id to AgentContext

        Raises:
            ValueError: If coordination_state is missing required fields
            RuntimeError: If distribution fails

        Example:
            ```python
            contexts = await distributor.distribute_agent_context(
                coordination_state={
                    "coordination_id": "coord-123",
                    "session_id": "session-456"
                },
                agent_assignments={
                    "model_gen": {
                        "objective": "Generate models",
                        "tasks": ["parse_contract", "generate_models"]
                    }
                }
            )
            ```
        """
        start_time = time.time()

        # 1. Validate coordination state
        coordination_id = coordination_state.get("coordination_id")
        session_id = coordination_state.get("session_id")

        if not coordination_id or not session_id:
            raise ValueError(
                "coordination_state must contain 'coordination_id' and 'session_id'"
            )

        # 2. Prepare shared intelligence (default if not provided)
        shared_intel = shared_intelligence or SharedIntelligence()

        # 3. Create agent contexts
        agent_contexts: dict[str, AgentContext] = {}

        for agent_id, assignment_data in agent_assignments.items():
            try:
                # Create context for this agent
                context = await self._create_agent_context(
                    coordination_id=coordination_id,
                    session_id=session_id,
                    agent_id=agent_id,
                    assignment_data=assignment_data,
                    shared_intelligence=shared_intel,
                    resource_allocation=(
                        resource_allocations.get(agent_id)
                        if resource_allocations
                        else None
                    ),
                    coordination_protocol=(
                        coordination_protocols.get(agent_id)
                        if coordination_protocols
                        else None
                    ),
                )

                agent_contexts[agent_id] = context

                # Store in ThreadSafeState
                self._store_agent_context(coordination_id, agent_id, context)

                # Record metrics
                if self.metrics:
                    context_size = sys.getsizeof(context.model_dump_json())
                    await self.metrics.record_gauge(
                        metric_name="context_size_bytes",
                        value=float(context_size),
                        unit="bytes",
                        tags={"agent_id": agent_id, "coordination_id": coordination_id},
                        correlation_id=coordination_id,
                    )

            except Exception as e:
                logger.error(f"Failed to create context for agent '{agent_id}': {e}")
                raise RuntimeError(
                    f"Context distribution failed for agent '{agent_id}': {e}"
                ) from e

        # 4. Store distributed contexts
        self._store_coordination_contexts(coordination_id, agent_contexts)

        # 5. Record distribution metrics
        elapsed_ms = (time.time() - start_time) * 1000

        if self.metrics:
            await self.metrics.record_timing(
                metric_name="context_distribution_time_ms",
                duration_ms=elapsed_ms,
                tags={
                    "coordination_id": coordination_id,
                    "agent_count": str(len(agent_contexts)),
                },
                correlation_id=coordination_id,
            )

            # Record per-agent distribution time
            avg_per_agent_ms = elapsed_ms / len(agent_contexts)
            await self.metrics.record_timing(
                metric_name="context_distribution_per_agent_ms",
                duration_ms=avg_per_agent_ms,
                tags={"coordination_id": coordination_id},
                correlation_id=coordination_id,
            )

        # Log performance
        logger.info(
            f"Distributed context to {len(agent_contexts)} agents in {elapsed_ms:.2f}ms "
            f"({elapsed_ms / len(agent_contexts):.2f}ms per agent)"
        )

        # Check performance target
        if elapsed_ms / len(agent_contexts) > 200:
            logger.warning(
                f"Context distribution exceeded 200ms target: "
                f"{elapsed_ms / len(agent_contexts):.2f}ms per agent"
            )

        return agent_contexts

    def get_agent_context(
        self, coordination_id: str, agent_id: str
    ) -> Optional[AgentContext]:
        """
        Retrieve context for specific agent.

        Performance: <5ms (ThreadSafeState get operation)

        Args:
            coordination_id: Coordination workflow ID
            agent_id: Agent identifier

        Returns:
            AgentContext if found, None otherwise

        Example:
            ```python
            context = distributor.get_agent_context("coord-123", "model_gen")
            if context:
                print(f"Agent role: {context.coordination_metadata.agent_role}")
            ```
        """
        contexts_key = f"coordination_contexts_{coordination_id}"
        all_contexts = self.state.get(contexts_key, {})

        if agent_id not in all_contexts:
            return None

        context_data = all_contexts[agent_id]
        return AgentContext(**context_data)

    def update_shared_intelligence(
        self, update_request: ContextUpdateRequest
    ) -> dict[str, bool]:
        """
        Update shared intelligence across agents.

        This updates the shared intelligence portion of agent contexts
        and optionally increments the context version.

        Args:
            update_request: Context update request

        Returns:
            Dictionary mapping agent_id to success status

        Example:
            ```python
            from omninode_bridge.agents.coordination.context_models import ContextUpdateRequest

            results = distributor.update_shared_intelligence(
                ContextUpdateRequest(
                    coordination_id="coord-123",
                    update_type="type_registry",
                    update_data={"CustomType": "class CustomType(BaseModel): ..."},
                    target_agents=None,  # Update all agents
                    increment_version=True
                )
            )
            ```
        """
        coordination_id = update_request.coordination_id
        contexts_key = f"coordination_contexts_{coordination_id}"

        all_contexts = self.state.get(contexts_key, {})

        if not all_contexts:
            logger.warning(
                f"No contexts found for coordination_id '{coordination_id}'"
            )
            return {}

        # Determine target agents
        target_agents = (
            update_request.target_agents
            if update_request.target_agents
            else list(all_contexts.keys())
        )

        results: dict[str, bool] = {}

        for agent_id in target_agents:
            if agent_id not in all_contexts:
                logger.warning(
                    f"Agent '{agent_id}' not found in coordination '{coordination_id}'"
                )
                results[agent_id] = False
                continue

            try:
                # Get existing context
                context_data = all_contexts[agent_id]
                context = AgentContext(**context_data)

                # Update shared intelligence
                shared_intel_dict = context.shared_intelligence.model_dump()

                if update_request.update_type == "type_registry":
                    shared_intel_dict["type_registry"].update(
                        update_request.update_data
                    )
                elif update_request.update_type == "pattern_library":
                    shared_intel_dict["pattern_library"].update(
                        update_request.update_data
                    )
                elif update_request.update_type == "validation_rules":
                    shared_intel_dict["validation_rules"].update(
                        update_request.update_data
                    )
                elif update_request.update_type == "naming_conventions":
                    shared_intel_dict["naming_conventions"].update(
                        update_request.update_data
                    )
                elif update_request.update_type == "dependency_graph":
                    shared_intel_dict["dependency_graph"].update(
                        update_request.update_data
                    )
                else:
                    logger.warning(f"Unknown update_type: {update_request.update_type}")
                    results[agent_id] = False
                    continue

                # Create updated context
                new_version = (
                    context.context_version + 1
                    if update_request.increment_version
                    else context.context_version
                )

                updated_context_dict = context.model_dump()
                updated_context_dict["shared_intelligence"] = shared_intel_dict
                updated_context_dict["context_version"] = new_version

                updated_context = AgentContext(**updated_context_dict)

                # Store updated context
                all_contexts[agent_id] = updated_context.model_dump()
                results[agent_id] = True

                logger.debug(
                    f"Updated shared intelligence for agent '{agent_id}' "
                    f"(version {new_version})"
                )

            except Exception as e:
                logger.error(
                    f"Failed to update context for agent '{agent_id}': {e}",
                    exc_info=True,
                )
                results[agent_id] = False

        # Save updated contexts
        self.state.set(contexts_key, all_contexts, changed_by="context_distributor")

        logger.info(
            f"Updated shared intelligence for {sum(results.values())}/{len(results)} agents"
        )

        return results

    def list_coordination_contexts(self, coordination_id: str) -> list[str]:
        """
        List all agent IDs with contexts for a coordination workflow.

        Args:
            coordination_id: Coordination workflow ID

        Returns:
            List of agent IDs

        Example:
            ```python
            agent_ids = distributor.list_coordination_contexts("coord-123")
            print(f"Agents: {agent_ids}")
            ```
        """
        contexts_key = f"coordination_contexts_{coordination_id}"
        all_contexts = self.state.get(contexts_key, {})
        return list(all_contexts.keys())

    def clear_coordination_contexts(self, coordination_id: str) -> bool:
        """
        Clear all contexts for a coordination workflow.

        Use this after workflow completion to free memory.

        Args:
            coordination_id: Coordination workflow ID

        Returns:
            True if contexts were cleared, False if no contexts found

        Example:
            ```python
            # After workflow completion
            cleared = distributor.clear_coordination_contexts("coord-123")
            ```
        """
        contexts_key = f"coordination_contexts_{coordination_id}"

        if not self.state.has(contexts_key):
            return False

        self.state.delete(contexts_key, changed_by="context_distributor")
        logger.info(f"Cleared contexts for coordination '{coordination_id}'")
        return True

    # Private methods

    async def _create_agent_context(
        self,
        coordination_id: str,
        session_id: str,
        agent_id: str,
        assignment_data: dict[str, Any],
        shared_intelligence: SharedIntelligence,
        resource_allocation: Optional[ResourceAllocation],
        coordination_protocol: Optional[CoordinationProtocols],
    ) -> AgentContext:
        """
        Create agent-specific context package.

        Args:
            coordination_id: Coordination workflow ID
            session_id: Coordination session ID
            agent_id: Agent identifier
            assignment_data: Agent assignment data
            shared_intelligence: Shared intelligence
            resource_allocation: Optional resource allocation
            coordination_protocol: Optional coordination protocol

        Returns:
            AgentContext
        """
        # Create coordination metadata
        coordination_metadata = CoordinationMetadata(
            session_id=session_id,
            coordination_id=coordination_id,
            agent_id=agent_id,
            agent_role=assignment_data.get("agent_role", agent_id),
        )

        # Create agent assignment
        agent_assignment = AgentAssignment(
            objective=assignment_data.get("objective", ""),
            tasks=assignment_data.get("tasks", []),
            input_data=assignment_data.get("input_data", {}),
            dependencies=assignment_data.get("dependencies", []),
            output_requirements=assignment_data.get("output_requirements", {}),
            success_criteria=assignment_data.get("success_criteria", {}),
        )

        # Use provided or default resource allocation
        resource_alloc = resource_allocation or self.default_resource_allocation

        # Use provided or default coordination protocols
        coord_protocol = coordination_protocol or self.default_coordination_protocols

        # Create agent context
        context = AgentContext(
            coordination_metadata=coordination_metadata,
            shared_intelligence=shared_intelligence,
            agent_assignment=agent_assignment,
            coordination_protocols=coord_protocol,
            resource_allocation=resource_alloc,
            context_version=1,
        )

        return context

    def _store_agent_context(
        self, coordination_id: str, agent_id: str, context: AgentContext
    ) -> None:
        """
        Store agent context in ThreadSafeState.

        Args:
            coordination_id: Coordination workflow ID
            agent_id: Agent identifier
            context: AgentContext to store
        """
        # Store individual context for quick access
        context_key = f"agent_context_{coordination_id}_{agent_id}"
        self.state.set(
            context_key, context.model_dump(), changed_by="context_distributor"
        )

    def _store_coordination_contexts(
        self, coordination_id: str, agent_contexts: dict[str, AgentContext]
    ) -> None:
        """
        Store all agent contexts for a coordination workflow.

        Args:
            coordination_id: Coordination workflow ID
            agent_contexts: Agent ID to AgentContext mapping
        """
        contexts_key = f"coordination_contexts_{coordination_id}"

        # Convert contexts to dict
        contexts_dict = {
            agent_id: context.model_dump()
            for agent_id, context in agent_contexts.items()
        }

        self.state.set(contexts_key, contexts_dict, changed_by="context_distributor")

    def _initialize_state(self) -> None:
        """Initialize state keys if not present."""
        # No global state needed - contexts are stored per coordination_id
        pass
