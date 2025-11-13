"""
Coordination orchestrator for multi-agent workflows.

This module provides a unified coordination system that integrates all 4 coordination
components for complete agent coordination workflows.

Components Integrated:
1. SignalCoordinator (Pattern 3) - Agent-to-agent communication
2. SmartRoutingOrchestrator (Pattern 4) - Intelligent routing
3. ContextDistributor (Pattern 9) - Context distribution
4. DependencyResolver (Pattern 10) - Dependency resolution

Performance Targets:
- Full workflow coordination: <2s for typical code generation workflow
- Component integration overhead: <100ms
- Support 50+ concurrent agents

Example:
    ```python
    from omninode_bridge.agents.coordination import (
        CoordinationOrchestrator,
        ThreadSafeState,
    )
    from omninode_bridge.agents.metrics import MetricsCollector
    from omninode_bridge.agents.registry import AgentRegistry

    # Initialize orchestrator
    state = ThreadSafeState()
    metrics = MetricsCollector()
    await metrics.start()

    registry = AgentRegistry(state=state, metrics_collector=metrics)
    orchestrator = CoordinationOrchestrator(
        state=state,
        metrics_collector=metrics,
        agent_registry=registry
    )

    # Coordinate workflow
    result = await orchestrator.coordinate_workflow(
        workflow_id="codegen-session-1",
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
        },
        shared_intelligence={
            "patterns": ["singleton", "factory"],
            "conventions": {"naming": "snake_case"}
        }
    )

    print(f"Workflow completed in {result['duration_ms']}ms")
    print(f"Contexts distributed: {result['contexts_distributed']}")
    print(f"Dependencies resolved: {result['dependencies_resolved']}")
    print(f"Signals sent: {result['signals_sent']}")
    ```
"""

import logging
import time
from typing import Any, Optional
from uuid import uuid4

from omninode_bridge.agents.metrics.collector import MetricsCollector
from omninode_bridge.agents.registry.matcher import CapabilityMatchEngine
from omninode_bridge.agents.registry.registry import AgentRegistry
from omninode_bridge.agents.type_defs import (
    CoordinationMetricsDict,
    CoordinationResultDict,
    DependencyResolutionDict,
    RoutingResultDict,
)

from .context_distribution import ContextDistributor
from .context_models import AgentContext, SharedIntelligence
from .dependency_models import Dependency, DependencyType
from .dependency_resolution import DependencyResolver
from .routing import SmartRoutingOrchestrator
from .routing_models import RoutingContext, RoutingStrategy
from .signals import SignalCoordinator
from .thread_safe_state import ThreadSafeState

logger = logging.getLogger(__name__)


class CoordinationOrchestrator:
    """
    Unified coordination orchestrator for multi-agent workflows.

    Integrates all 4 coordination components:
    - SignalCoordinator: Agent-to-agent communication
    - SmartRoutingOrchestrator: Intelligent task routing
    - ContextDistributor: Context distribution to agents
    - DependencyResolver: Dependency resolution

    Performance:
    - Full workflow coordination: <2s (target)
    - Component integration overhead: <100ms
    - Support 50+ concurrent agents

    Features:
    - Unified workflow coordination API
    - Automatic dependency resolution
    - Intelligent routing and context distribution
    - Real-time signal coordination
    - Comprehensive metrics collection
    - Thread-safe state management

    Example:
        ```python
        orchestrator = CoordinationOrchestrator(
            state=state,
            metrics_collector=metrics,
            agent_registry=registry
        )

        result = await orchestrator.coordinate_workflow(
            workflow_id="session-1",
            agent_assignments={"agent-1": {...}, "agent-2": {...}}
        )
        ```
    """

    def __init__(
        self,
        state: ThreadSafeState,
        metrics_collector: MetricsCollector,
        agent_registry: Optional[AgentRegistry] = None,
        enable_routing: bool = True,
        enable_dependency_resolution: bool = True,
    ):
        """
        Initialize coordination orchestrator.

        Args:
            state: ThreadSafeState instance for centralized storage
            metrics_collector: MetricsCollector for performance tracking
            agent_registry: Optional AgentRegistry for routing decisions
            enable_routing: Enable smart routing orchestration
            enable_dependency_resolution: Enable dependency resolution
        """
        self.state = state
        self.metrics = metrics_collector
        self.agent_registry = agent_registry

        # Initialize all 4 coordination components
        self.signal_coordinator = SignalCoordinator(
            state=state,
            metrics_collector=metrics_collector,
        )

        self.context_distributor = ContextDistributor(
            state=state,
            metrics_collector=metrics_collector,
        )

        self.dependency_resolver = DependencyResolver(
            signal_coordinator=self.signal_coordinator,
            metrics_collector=metrics_collector,
            state=state,
        )

        # Smart routing orchestrator (requires agent registry)
        self.routing_orchestrator = None
        if enable_routing and agent_registry:
            match_engine = CapabilityMatchEngine(
                agent_registry=agent_registry,
                metrics_collector=metrics_collector,
            )
            self.routing_orchestrator = SmartRoutingOrchestrator(
                state=state,
                match_engine=match_engine,
                metrics_collector=metrics_collector,
            )

        self._enable_routing = enable_routing
        self._enable_dependency_resolution = enable_dependency_resolution

        logger.info(
            f"CoordinationOrchestrator initialized: "
            f"routing={enable_routing}, "
            f"dependency_resolution={enable_dependency_resolution}"
        )

    async def coordinate_workflow(
        self,
        workflow_id: str,
        agent_assignments: dict[str, dict[str, Any]],
        shared_intelligence: Optional[SharedIntelligence | dict[str, Any]] = None,
        enable_signals: bool = True,
    ) -> CoordinationResultDict:
        """
        Coordinate complete multi-agent workflow.

        This is the main orchestration method that:
        1. Distributes contexts to all agents
        2. Resolves dependencies between agents
        3. Routes tasks intelligently (if enabled)
        4. Signals coordination events

        Performance Target: <2s for typical code generation workflow

        Args:
            workflow_id: Unique workflow identifier
            agent_assignments: Agent-specific assignments and tasks
            shared_intelligence: Optional shared intelligence data
            enable_signals: Enable signal coordination (default: True)

        Returns:
            Coordination result dictionary with:
            - workflow_id: Workflow identifier
            - coordination_id: Generated coordination ID
            - duration_ms: Total coordination time
            - contexts_distributed: Number of contexts distributed
            - dependencies_resolved: Number of dependencies resolved
            - signals_sent: Number of signals sent
            - routing_decisions: Number of routing decisions made
            - agent_contexts: Distributed agent contexts

        Example:
            ```python
            result = await orchestrator.coordinate_workflow(
                workflow_id="codegen-session-1",
                agent_assignments={
                    "model_gen": {
                        "objective": "Generate models",
                        "tasks": ["parse", "generate"],
                        "input_data": {"contract": "./contract.yaml"}
                    },
                    "validator_gen": {
                        "objective": "Generate validators",
                        "tasks": ["validate"],
                        "dependencies": ["model_gen"]
                    }
                },
                shared_intelligence={
                    "patterns": ["singleton"],
                    "conventions": {"naming": "snake_case"}
                }
            )
            ```
        """
        start_time = time.time()
        coordination_id = f"{workflow_id}-{uuid4().hex[:8]}"

        logger.info(
            f"Starting workflow coordination: {coordination_id} "
            f"with {len(agent_assignments)} agents"
        )

        # Initialize result tracking
        result = {
            "workflow_id": workflow_id,
            "coordination_id": coordination_id,
            "contexts_distributed": 0,
            "dependencies_resolved": 0,
            "signals_sent": 0,
            "routing_decisions": 0,
            "agent_contexts": {},
        }

        # Step 1: Signal workflow start
        if enable_signals:
            await self.signal_coordinator.signal_coordination_event(
                coordination_id=coordination_id,
                event_type="workflow_started",
                event_data={
                    "workflow_id": workflow_id,
                    "agent_count": len(agent_assignments),
                    "timestamp": time.time(),
                },
            )
            result["signals_sent"] += 1

        # Step 2: Distribute contexts to agents
        try:
            contexts = await self.context_distributor.distribute_agent_context(
                coordination_state={
                    "coordination_id": coordination_id,
                    "workflow_id": workflow_id,
                },
                agent_assignments=agent_assignments,
                shared_intelligence=(
                    shared_intelligence
                    if isinstance(shared_intelligence, SharedIntelligence)
                    else SharedIntelligence(**(shared_intelligence or {}))
                ),
            )
            result["contexts_distributed"] = len(contexts)
            result["agent_contexts"] = {
                agent_id: ctx.model_dump() for agent_id, ctx in contexts.items()
            }

            logger.debug(
                f"Distributed contexts to {len(contexts)} agents "
                f"for {coordination_id}"
            )

        except Exception as e:
            logger.error(
                f"Context distribution failed for {coordination_id}: {e}",
                exc_info=True,
            )
            result["error"] = f"Context distribution failed: {e!s}"
            return result

        # Step 3: Resolve dependencies (if enabled)
        if self._enable_dependency_resolution:
            dependency_results = await self._resolve_workflow_dependencies(
                coordination_id=coordination_id,
                agent_assignments=agent_assignments,
            )
            result["dependencies_resolved"] = dependency_results["resolved_count"]
            result["dependency_details"] = dependency_results["details"]

            logger.debug(
                f"Resolved {dependency_results['resolved_count']} dependencies "
                f"for {coordination_id}"
            )

        # Step 4: Route tasks intelligently (if enabled)
        if self._enable_routing and self.routing_orchestrator:
            routing_results = await self._route_workflow_tasks(
                coordination_id=coordination_id,
                agent_assignments=agent_assignments,
            )
            result["routing_decisions"] = routing_results["decision_count"]
            result["routing_details"] = routing_results["details"]

            logger.debug(
                f"Made {routing_results['decision_count']} routing decisions "
                f"for {coordination_id}"
            )

        # Step 5: Signal workflow ready
        if enable_signals:
            await self.signal_coordinator.signal_coordination_event(
                coordination_id=coordination_id,
                event_type="workflow_ready",
                event_data={
                    "workflow_id": workflow_id,
                    "contexts_distributed": result["contexts_distributed"],
                    "dependencies_resolved": result["dependencies_resolved"],
                    "routing_decisions": result["routing_decisions"],
                    "timestamp": time.time(),
                },
            )
            result["signals_sent"] += 1

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        result["duration_ms"] = duration_ms

        # Record metrics
        await self.metrics.record_timing(
            metric_name="workflow_coordination_time_ms",
            duration_ms=duration_ms,
            tags={
                "workflow_id": workflow_id,
                "agent_count": len(agent_assignments),
            },
            correlation_id=coordination_id,
        )

        logger.info(
            f"Workflow coordination completed: {coordination_id} "
            f"in {duration_ms:.2f}ms"
        )

        return result

    async def signal_agent_completion(
        self,
        coordination_id: str,
        agent_id: str,
        result_summary: dict[str, Any],
    ) -> bool:
        """
        Signal that an agent has completed its work.

        Args:
            coordination_id: Coordination session ID
            agent_id: Agent identifier
            result_summary: Summary of agent's work and results

        Returns:
            True if signal sent successfully, False otherwise

        Example:
            ```python
            await orchestrator.signal_agent_completion(
                coordination_id="coord-123",
                agent_id="model-gen",
                result_summary={
                    "models_generated": 5,
                    "quality_score": 0.95,
                    "execution_time_ms": 1234.5
                }
            )
            ```
        """
        return await self.signal_coordinator.signal_coordination_event(
            coordination_id=coordination_id,
            event_type="agent_completed",
            event_data={
                "agent_id": agent_id,
                **result_summary,
            },
            sender_agent_id=agent_id,
        )

    async def get_agent_context(
        self,
        coordination_id: str,
        agent_id: str,
    ) -> Optional[AgentContext]:
        """
        Retrieve agent context for coordination session.

        Args:
            coordination_id: Coordination session ID
            agent_id: Agent identifier

        Returns:
            AgentContext if found, None otherwise

        Example:
            ```python
            context = await orchestrator.get_agent_context(
                coordination_id="coord-123",
                agent_id="model-gen"
            )
            ```
        """
        return self.context_distributor.get_agent_context(
            coordination_id=coordination_id,
            agent_id=agent_id,
        )

    async def check_dependency_status(
        self,
        coordination_id: str,
        dependency_id: str,
    ) -> bool:
        """
        Check if dependency has been resolved.

        Args:
            coordination_id: Coordination session ID
            dependency_id: Dependency identifier

        Returns:
            True if dependency is resolved, False otherwise

        Example:
            ```python
            is_resolved = await orchestrator.check_dependency_status(
                coordination_id="coord-123",
                dependency_id="model_gen_complete"
            )
            ```
        """
        return self.dependency_resolver.is_dependency_resolved(
            coordination_id=coordination_id,
            dependency_id=dependency_id,
        )

    def get_coordination_metrics(
        self,
        coordination_id: str,
    ) -> CoordinationMetricsDict:
        """
        Get comprehensive coordination metrics.

        Args:
            coordination_id: Coordination session ID

        Returns:
            Dictionary with metrics from all components:
            - signal_metrics: Signal coordination metrics
            - context_metrics: Context distribution metrics
            - routing_metrics: Routing decision metrics

        Example:
            ```python
            metrics = orchestrator.get_coordination_metrics("coord-123")
            print(f"Signals sent: {metrics['signal_metrics'].total_signals_sent}")
            ```
        """
        return {
            "signal_metrics": self.signal_coordinator.get_signal_metrics(
                coordination_id
            ).model_dump(),
            "context_metrics": self.context_distributor.get_distribution_metrics(
                coordination_id
            ).model_dump(),
            "routing_metrics": (
                self.routing_orchestrator.get_routing_history(
                    coordination_id=coordination_id,
                    limit=10,
                ).model_dump()
                if self.routing_orchestrator
                else {}
            ),
        }

    # Private helper methods

    async def _resolve_workflow_dependencies(
        self,
        coordination_id: str,
        agent_assignments: dict[str, dict[str, Any]],
    ) -> DependencyResolutionDict:
        """
        Resolve all dependencies in workflow.

        Args:
            coordination_id: Coordination session ID
            agent_assignments: Agent assignments with dependencies

        Returns:
            Dictionary with resolution results
        """
        resolved_count = 0
        details = []

        for agent_id, assignment in agent_assignments.items():
            dependencies = assignment.get("dependencies", [])

            for dep_target in dependencies:
                # Create agent completion dependency
                dependency = Dependency(
                    dependency_id=f"{agent_id}_waits_for_{dep_target}",
                    dependency_type=DependencyType.AGENT_COMPLETION,
                    target=dep_target,
                    timeout=120,  # 2 minutes default
                )

                try:
                    result = await self.dependency_resolver.resolve_dependency(
                        coordination_id=coordination_id,
                        dependency=dependency,
                    )

                    resolved_count += 1
                    details.append(
                        {
                            "dependency_id": dependency.dependency_id,
                            "status": result.status.value,
                            "duration_ms": result.duration_ms,
                        }
                    )

                except Exception as e:
                    logger.warning(
                        f"Dependency resolution failed for {dependency.dependency_id}: {e}"
                    )
                    details.append(
                        {
                            "dependency_id": dependency.dependency_id,
                            "status": "failed",
                            "error": str(e),
                        }
                    )

        return {
            "resolved_count": resolved_count,
            "details": details,
        }

    async def _route_workflow_tasks(
        self,
        coordination_id: str,
        agent_assignments: dict[str, dict[str, Any]],
    ) -> RoutingResultDict:
        """
        Route tasks for all agents in workflow.

        Args:
            coordination_id: Coordination session ID
            agent_assignments: Agent assignments with tasks

        Returns:
            Dictionary with routing results
        """
        if not self.routing_orchestrator:
            return {"decision_count": 0, "details": []}

        decision_count = 0
        details = []

        for agent_id, assignment in agent_assignments.items():
            tasks = assignment.get("tasks", [])

            for task_name in tasks:
                try:
                    # Create routing context
                    context = RoutingContext(
                        workflow_id=coordination_id,
                        current_step=task_name,
                        agent_id=agent_id,
                        metadata=assignment.get("metadata", {}),
                    )

                    # Make routing decision
                    result = await self.routing_orchestrator.route(
                        state=self.state.get_all(),
                        context=context,
                        strategy=RoutingStrategy.CONDITIONAL,
                    )

                    decision_count += 1
                    details.append(
                        {
                            "agent_id": agent_id,
                            "task": task_name,
                            "decision": result.decision.value,
                            "confidence": result.confidence_score,
                        }
                    )

                except Exception as e:
                    logger.warning(f"Routing failed for {agent_id}/{task_name}: {e}")
                    details.append(
                        {
                            "agent_id": agent_id,
                            "task": task_name,
                            "error": str(e),
                        }
                    )

        return {
            "decision_count": decision_count,
            "details": details,
        }
