"""
Dependency resolution system for multi-agent coordination.

This module provides production-ready dependency resolution with:
- Three dependency types: agent_completion, resource_availability, quality_gate
- Timeout-based waiting with configurable timeouts
- Async non-blocking dependency checks
- Integration with SignalCoordinator for event signaling
- Performance target: <2s total resolution time

Performance targets (validated from omniagent):
- Dependency resolution: <2s total
- Support 100+ dependencies per coordination session
- Timeout-based waiting with configurable timeouts
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Optional

from omninode_bridge.agents.metrics.collector import MetricsCollector

from .dependency_models import (
    AgentCompletionConfig,
    Dependency,
    DependencyResolutionResult,
    DependencyStatus,
    DependencyType,
    QualityGateConfig,
    ResourceAvailabilityConfig,
)
from .exceptions import DependencyResolutionError, DependencyTimeoutError
from .signals import SignalCoordinator
from .thread_safe_state import ThreadSafeState

logger = logging.getLogger(__name__)


class DependencyResolver:
    """
    Dependency resolver for multi-agent coordination.

    Features:
    - Three dependency types: agent_completion, resource_availability, quality_gate
    - Timeout-based waiting (configurable per dependency)
    - Async non-blocking checks with exponential backoff
    - Signal coordination for dependency resolution events
    - Metrics collection for performance tracking

    Performance Targets:
    - Dependency resolution: <2s total
    - Support 100+ dependencies per coordination session
    - Non-blocking async checks

    Example:
        ```python
        resolver = DependencyResolver(
            signal_coordinator=signal_coordinator,
            metrics_collector=metrics_collector,
            state=shared_state
        )

        # Create dependency
        dependency = Dependency(
            dependency_id="model_gen_complete",
            dependency_type=DependencyType.AGENT_COMPLETION,
            target="agent-model-generator",
            timeout=120
        )

        # Resolve dependency
        result = await resolver.resolve_dependency(
            coordination_id="coord-123",
            dependency=dependency
        )

        print(f"Resolved in {result.duration_ms}ms")
        ```
    """

    def __init__(
        self,
        signal_coordinator: SignalCoordinator,
        metrics_collector: MetricsCollector,
        state: Optional[ThreadSafeState] = None,
        max_concurrent_resolutions: int = 10,
    ) -> None:
        """
        Initialize dependency resolver.

        Args:
            signal_coordinator: Signal coordinator for event signaling
            metrics_collector: Metrics collector for performance tracking
            state: Optional shared state for resource availability checks
            max_concurrent_resolutions: Maximum concurrent dependency resolutions
        """
        self.signal_coordinator = signal_coordinator
        self.metrics = metrics_collector
        self.state = state or ThreadSafeState[dict[str, Any]]()

        # Track resolved dependencies per coordination session
        self.resolved_dependencies: dict[str, dict[str, bool]] = {}

        # Track pending dependencies per coordination session
        self.pending_dependencies: dict[str, list[Dependency]] = {}

        # Semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent_resolutions)

        # Resource availability cache (for resource_availability checks)
        self._resource_cache: dict[str, bool] = {}

        # Quality gate results cache (for quality_gate checks)
        self._quality_gate_cache: dict[str, float] = {}

        logger.info(
            f"[DependencyResolver] Initialized with max_concurrent={max_concurrent_resolutions}"
        )

    async def resolve_agent_dependencies(
        self,
        coordination_id: str,
        agent_context: dict[str, Any],
    ) -> bool:
        """
        Resolve all dependencies for an agent.

        This is the main entry point for dependency resolution. It resolves all
        dependencies specified in the agent context sequentially (dependencies
        must be resolved in order).

        Args:
            coordination_id: Coordination session ID
            agent_context: Agent context containing dependency specifications

        Returns:
            True if all dependencies resolved successfully, False otherwise

        Raises:
            DependencyResolutionError: If dependency resolution fails
            DependencyTimeoutError: If dependency resolution times out

        Performance Target: <2s total for all dependencies
        """
        start_time = asyncio.get_event_loop().time()

        try:
            # Extract dependencies from agent context
            dependencies_spec = agent_context.get("dependencies", [])

            if not dependencies_spec:
                # No dependencies to resolve
                logger.debug(
                    f"[DependencyResolver] No dependencies for coordination_id={coordination_id}"
                )
                return True

            # Parse dependencies
            dependencies = self._parse_dependencies(dependencies_spec)

            logger.info(
                f"[DependencyResolver] Resolving {len(dependencies)} dependencies "
                f"for coordination_id={coordination_id}"
            )

            # Store pending dependencies
            if coordination_id not in self.pending_dependencies:
                self.pending_dependencies[coordination_id] = []
            self.pending_dependencies[coordination_id].extend(dependencies)

            # Resolve each dependency sequentially
            results = []
            for dependency in dependencies:
                try:
                    result = await self.resolve_dependency(coordination_id, dependency)
                    results.append(result)

                    if not result.success:
                        logger.error(
                            f"[DependencyResolver] Dependency '{dependency.dependency_id}' "
                            f"failed to resolve: {result.error_message}"
                        )
                        return False
                except DependencyTimeoutError as e:
                    logger.error(
                        f"[DependencyResolver] Dependency '{dependency.dependency_id}' "
                        f"timed out: {e}"
                    )
                    return False

            # Signal successful dependency resolution
            await self.signal_coordinator.signal_event(
                coordination_id=coordination_id,
                event_type="dependency_resolved",
                event_data={
                    "resolver_agent": agent_context.get("agent_id", "unknown"),
                    "dependency_count": len(dependencies),
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

            # Track metrics
            duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            await self.metrics.record_timing(
                metric_name="dependency_resolution_time_ms",
                duration_ms=duration_ms,
                tags={
                    "coordination_id": coordination_id,
                    "dependency_count": str(len(dependencies)),
                },
                correlation_id=coordination_id,
            )

            logger.info(
                f"[DependencyResolver] All dependencies resolved successfully "
                f"for coordination_id={coordination_id} in {duration_ms:.2f}ms"
            )

            return True

        except Exception as e:
            duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000

            await self.metrics.record_timing(
                metric_name="dependency_resolution_time_ms",
                duration_ms=duration_ms,
                tags={
                    "coordination_id": coordination_id,
                    "success": "false",
                },
                correlation_id=coordination_id,
            )

            logger.error(
                f"[DependencyResolver] Dependency resolution failed for "
                f"coordination_id={coordination_id}: {e}"
            )
            raise DependencyResolutionError(
                coordination_id=coordination_id,
                dependency_id="all",
                error_message=str(e),
            ) from e

    async def resolve_dependency(
        self,
        coordination_id: str,
        dependency: Dependency,
    ) -> DependencyResolutionResult:
        """
        Resolve a single dependency.

        Args:
            coordination_id: Coordination session ID
            dependency: Dependency to resolve

        Returns:
            DependencyResolutionResult with resolution status and metrics

        Raises:
            DependencyTimeoutError: If dependency resolution times out
        """
        async with self._semaphore:
            start_time = asyncio.get_event_loop().time()
            dependency.status = DependencyStatus.IN_PROGRESS

            logger.debug(
                f"[DependencyResolver] Resolving dependency '{dependency.dependency_id}' "
                f"of type '{dependency.dependency_type.value}' for coordination_id={coordination_id}"
            )

            try:
                # Resolve based on dependency type
                if dependency.dependency_type == DependencyType.AGENT_COMPLETION:
                    success = await self._wait_for_agent_completion(
                        coordination_id, dependency
                    )
                elif dependency.dependency_type == DependencyType.RESOURCE_AVAILABILITY:
                    success = await self._check_resource_availability(
                        coordination_id, dependency
                    )
                elif dependency.dependency_type == DependencyType.QUALITY_GATE:
                    success = await self._wait_for_quality_gate(
                        coordination_id, dependency
                    )
                else:
                    raise ValueError(
                        f"Unknown dependency type: {dependency.dependency_type}"
                    )

                if success:
                    dependency.mark_resolved()
                    self._mark_dependency_resolved(
                        coordination_id, dependency.dependency_id
                    )
                else:
                    dependency.mark_failed("Resolution check failed")

                duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000

                result = DependencyResolutionResult(
                    coordination_id=coordination_id,
                    dependency_id=dependency.dependency_id,
                    success=success,
                    status=dependency.status,
                    duration_ms=duration_ms,
                    attempts=dependency.retry_count + 1,
                )

                logger.debug(
                    f"[DependencyResolver] Dependency '{dependency.dependency_id}' "
                    f"resolved with success={success} in {duration_ms:.2f}ms"
                )

                return result

            except asyncio.TimeoutError as e:
                dependency.mark_timeout()
                duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000

                result = DependencyResolutionResult(
                    coordination_id=coordination_id,
                    dependency_id=dependency.dependency_id,
                    success=False,
                    status=dependency.status,
                    duration_ms=duration_ms,
                    attempts=dependency.retry_count + 1,
                    error_message=f"Timeout after {dependency.timeout}s",
                )

                logger.error(
                    f"[DependencyResolver] Dependency '{dependency.dependency_id}' "
                    f"timed out after {dependency.timeout}s"
                )

                raise DependencyTimeoutError(
                    coordination_id=coordination_id,
                    dependency_id=dependency.dependency_id,
                    timeout_seconds=dependency.timeout,
                ) from e

    async def _wait_for_agent_completion(
        self,
        coordination_id: str,
        dependency: Dependency,
    ) -> bool:
        """
        Wait for specific agent to complete.

        Uses SignalCoordinator to check for agent completion signals.

        Args:
            coordination_id: Coordination session ID
            dependency: Agent completion dependency

        Returns:
            True if agent completed within timeout, False otherwise

        Raises:
            asyncio.TimeoutError: If timeout exceeded
        """
        # Extract agent_id from dependency metadata or target
        config = AgentCompletionConfig(
            agent_id=dependency.metadata.get("agent_id", dependency.target),
            completion_event=dependency.metadata.get("completion_event", "completion"),
            require_success=dependency.metadata.get("require_success", True),
        )

        logger.debug(
            f"[DependencyResolver] Waiting for agent '{config.agent_id}' completion "
            f"with timeout={dependency.timeout}s"
        )

        start_time = asyncio.get_event_loop().time()
        check_interval = 0.1  # Check every 100ms

        while asyncio.get_event_loop().time() - start_time < dependency.timeout:
            # Check if agent has completed via SignalCoordinator
            completed = await self.signal_coordinator.check_agent_completion(
                coordination_id=coordination_id,
                agent_id=config.agent_id,
                completion_event=config.completion_event,
            )

            if completed:
                logger.debug(
                    f"[DependencyResolver] Agent '{config.agent_id}' has completed"
                )
                return True

            # Wait before next check
            await asyncio.sleep(check_interval)

        # Timeout occurred
        logger.warning(
            f"[DependencyResolver] Timeout waiting for agent '{config.agent_id}' "
            f"after {dependency.timeout}s"
        )
        raise asyncio.TimeoutError(
            f"Timeout waiting for agent '{config.agent_id}' completion"
        )

    async def _check_resource_availability(
        self,
        coordination_id: str,
        dependency: Dependency,
    ) -> bool:
        """
        Check if resource is available.

        Checks resource availability in ThreadSafeState or via resource registry.

        Args:
            coordination_id: Coordination session ID
            dependency: Resource availability dependency

        Returns:
            True if resource is available, False otherwise

        Raises:
            asyncio.TimeoutError: If timeout exceeded
        """
        # Extract resource config from dependency metadata
        config = ResourceAvailabilityConfig(
            resource_id=dependency.metadata.get("resource_id", dependency.target),
            resource_type=dependency.metadata.get("resource_type", "unknown"),
            check_interval_ms=dependency.metadata.get("check_interval_ms", 100),
            availability_threshold=dependency.metadata.get(
                "availability_threshold", 1.0
            ),
        )

        logger.debug(
            f"[DependencyResolver] Checking resource '{config.resource_id}' availability "
            f"of type '{config.resource_type}' with timeout={dependency.timeout}s"
        )

        start_time = asyncio.get_event_loop().time()
        check_interval = config.check_interval_ms / 1000.0

        while asyncio.get_event_loop().time() - start_time < dependency.timeout:
            # Check resource availability in state
            resource_key = f"resource_{config.resource_id}_available"
            is_available = self.state.get(resource_key, default=False)

            # Also check cache
            if config.resource_id in self._resource_cache:
                is_available = self._resource_cache[config.resource_id]

            if is_available:
                logger.debug(
                    f"[DependencyResolver] Resource '{config.resource_id}' is available"
                )
                return True

            # Wait before next check
            await asyncio.sleep(check_interval)

        # Timeout occurred
        logger.warning(
            f"[DependencyResolver] Timeout waiting for resource '{config.resource_id}' "
            f"after {dependency.timeout}s"
        )
        raise asyncio.TimeoutError(
            f"Timeout waiting for resource '{config.resource_id}' availability"
        )

    async def _wait_for_quality_gate(
        self,
        coordination_id: str,
        dependency: Dependency,
    ) -> bool:
        """
        Wait for quality gate to pass.

        Checks quality gate status in ThreadSafeState or via quality gate registry.

        Args:
            coordination_id: Coordination session ID
            dependency: Quality gate dependency

        Returns:
            True if quality gate passes, False otherwise

        Raises:
            asyncio.TimeoutError: If timeout exceeded
        """
        # Extract quality gate config from dependency metadata
        config = QualityGateConfig(
            gate_id=dependency.metadata.get("gate_id", dependency.target),
            gate_type=dependency.metadata.get("gate_type", "unknown"),
            threshold=dependency.metadata.get("threshold", 0.8),
            check_interval_ms=dependency.metadata.get("check_interval_ms", 500),
        )

        logger.debug(
            f"[DependencyResolver] Waiting for quality gate '{config.gate_id}' "
            f"of type '{config.gate_type}' with threshold={config.threshold}, "
            f"timeout={dependency.timeout}s"
        )

        start_time = asyncio.get_event_loop().time()
        check_interval = config.check_interval_ms / 1000.0

        while asyncio.get_event_loop().time() - start_time < dependency.timeout:
            # Check quality gate status in state
            gate_key = f"quality_gate_{config.gate_id}_score"
            gate_score = self.state.get(gate_key, default=0.0)

            # Also check cache
            if config.gate_id in self._quality_gate_cache:
                gate_score = self._quality_gate_cache[config.gate_id]

            if gate_score >= config.threshold:
                logger.debug(
                    f"[DependencyResolver] Quality gate '{config.gate_id}' passed "
                    f"with score={gate_score:.2f} (threshold={config.threshold})"
                )
                return True

            # Wait before next check
            await asyncio.sleep(check_interval)

        # Timeout occurred
        logger.warning(
            f"[DependencyResolver] Timeout waiting for quality gate '{config.gate_id}' "
            f"after {dependency.timeout}s"
        )
        raise asyncio.TimeoutError(
            f"Timeout waiting for quality gate '{config.gate_id}' to pass"
        )

    def get_dependency_status(
        self,
        coordination_id: str,
        dependency_id: str,
    ) -> dict[str, Any]:
        """
        Get status of specific dependency.

        Args:
            coordination_id: Coordination session ID
            dependency_id: Dependency identifier

        Returns:
            Dictionary with dependency status information
        """
        # Check if dependency is resolved
        is_resolved = (
            coordination_id in self.resolved_dependencies
            and dependency_id in self.resolved_dependencies[coordination_id]
        )

        # Find dependency in pending list
        dependency = None
        if coordination_id in self.pending_dependencies:
            for dep in self.pending_dependencies[coordination_id]:
                if dep.dependency_id == dependency_id:
                    dependency = dep
                    break

        if dependency:
            return {
                "dependency_id": dependency_id,
                "resolved": is_resolved,
                "status": dependency.status.value,
                "dependency_type": dependency.dependency_type.value,
                "target": dependency.target,
                "resolved_at": (
                    dependency.resolved_at.isoformat()
                    if dependency.resolved_at
                    else None
                ),
                "error_message": dependency.error_message,
            }

        return {
            "dependency_id": dependency_id,
            "resolved": is_resolved,
            "status": "unknown",
            "error_message": "Dependency not found",
        }

    def _mark_dependency_resolved(
        self,
        coordination_id: str,
        dependency_id: str,
    ) -> None:
        """Mark dependency as resolved in internal tracking."""
        if coordination_id not in self.resolved_dependencies:
            self.resolved_dependencies[coordination_id] = {}

        self.resolved_dependencies[coordination_id][dependency_id] = True

    def _parse_dependencies(
        self,
        dependencies_spec: list[dict[str, Any]],
    ) -> list[Dependency]:
        """
        Parse dependency specifications from agent context.

        Args:
            dependencies_spec: List of dependency specifications

        Returns:
            List of Dependency objects
        """
        dependencies = []

        for spec in dependencies_spec:
            dependency = Dependency(
                dependency_id=spec.get(
                    "id", spec.get("dependency_id", f"dep_{len(dependencies)}")
                ),
                dependency_type=DependencyType(
                    spec.get("type", spec.get("dependency_type"))
                ),
                target=spec.get("target"),
                timeout=spec.get("timeout", 120),
                max_retries=spec.get("max_retries", 3),
                metadata=spec.get("metadata", {}),
            )
            dependencies.append(dependency)

        return dependencies

    # Helper methods for testing and resource/quality gate updates

    async def mark_resource_available(
        self,
        resource_id: str,
        available: bool = True,
    ) -> None:
        """
        Mark a resource as available/unavailable (for testing or external updates).

        Args:
            resource_id: Resource identifier
            available: Availability status
        """
        self._resource_cache[resource_id] = available
        resource_key = f"resource_{resource_id}_available"
        self.state.set(resource_key, available, changed_by="dependency_resolver")

        logger.debug(
            f"[DependencyResolver] Resource '{resource_id}' marked as "
            f"{'available' if available else 'unavailable'}"
        )

    async def update_quality_gate_score(
        self,
        gate_id: str,
        score: float,
    ) -> None:
        """
        Update quality gate score (for testing or external updates).

        Args:
            gate_id: Quality gate identifier
            score: Quality gate score (0.0-1.0)
        """
        self._quality_gate_cache[gate_id] = score
        gate_key = f"quality_gate_{gate_id}_score"
        self.state.set(gate_key, score, changed_by="dependency_resolver")

        logger.debug(
            f"[DependencyResolver] Quality gate '{gate_id}' score updated to {score:.2f}"
        )

    def clear_coordination_dependencies(
        self,
        coordination_id: str,
    ) -> None:
        """
        Clear dependencies for coordination session (cleanup after completion).

        Args:
            coordination_id: Coordination session ID
        """
        if coordination_id in self.resolved_dependencies:
            del self.resolved_dependencies[coordination_id]

        if coordination_id in self.pending_dependencies:
            del self.pending_dependencies[coordination_id]

        logger.debug(
            f"[DependencyResolver] Cleared dependencies for coordination_id={coordination_id}"
        )

    def get_pending_dependencies_count(
        self,
        coordination_id: str,
    ) -> int:
        """
        Get count of pending dependencies for coordination session.

        Args:
            coordination_id: Coordination session ID

        Returns:
            Number of pending dependencies
        """
        if coordination_id not in self.pending_dependencies:
            return 0

        return len(
            [
                dep
                for dep in self.pending_dependencies[coordination_id]
                if dep.status != DependencyStatus.RESOLVED
            ]
        )

    def __repr__(self) -> str:
        """String representation of DependencyResolver."""
        return (
            f"DependencyResolver(coordination_sessions={len(self.resolved_dependencies)}, "
            f"state_version={self.state.get_version()})"
        )
