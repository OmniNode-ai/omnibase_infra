"""
Smart Routing Orchestration for multi-agent coordination.

This module implements intelligent routing strategies for code generation workflows
with support for conditional routing, parallel execution, state analysis, and
priority-based routing.

Performance Targets:
- Routing decision: <5ms with caching
- Throughput: 100+ routing decisions per second
- History tracking: Automatic with size limiting
"""

import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Optional

from ..metrics.collector import MetricsCollector
from ..metrics.decorators import timed
from ..registry.matcher import CapabilityMatchEngine
from ..registry.models import AgentInfo, Task
from ..registry.registry import AgentRegistry
from .routing_models import (
    ConditionalRule,
    ParallelizationHint,
    PriorityRoutingConfig,
    RoutingContext,
    RoutingDecision,
    RoutingHistoryRecord,
    RoutingResult,
    RoutingStrategy,
    StateComplexityMetrics,
)
from .thread_safe_state import ThreadSafeState

logger = logging.getLogger(__name__)


class BaseRouter(ABC):
    """
    Base class for routing strategies.

    All routers must implement the evaluate() method to return a routing decision.
    """

    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        """
        Initialize base router.

        Args:
            metrics_collector: Optional metrics collector for performance tracking
        """
        self.metrics = metrics_collector
        self.strategy: RoutingStrategy

    @abstractmethod
    def evaluate(
        self, state: dict[str, Any], context: RoutingContext
    ) -> RoutingResult:
        """
        Evaluate routing decision based on state and context.

        Args:
            state: Current state snapshot
            context: Routing context

        Returns:
            RoutingResult with decision and reasoning
        """
        pass


class ConditionalRouter(BaseRouter):
    """
    Conditional routing based on state conditions.

    Routes based on user-defined rules that evaluate state conditions.

    Example:
        ```python
        rules = [
            ConditionalRule(
                rule_id="error_handling",
                name="Retry on error",
                condition_key="error_count",
                condition_operator=">",
                condition_value=0,
                decision=RoutingDecision.RETRY,
                priority=90
            )
        ]
        router = ConditionalRouter(rules=rules)
        result = router.evaluate(state, context)
        ```
    """

    def __init__(
        self,
        rules: list[ConditionalRule],
        metrics_collector: Optional[MetricsCollector] = None,
    ):
        """
        Initialize conditional router.

        Args:
            rules: List of conditional rules (evaluated by priority)
            metrics_collector: Optional metrics collector
        """
        super().__init__(metrics_collector)
        self.strategy = RoutingStrategy.CONDITIONAL
        # Sort rules by priority (highest first)
        self.rules = sorted(rules, key=lambda r: r.priority, reverse=True)

    def evaluate(
        self, state: dict[str, Any], context: RoutingContext
    ) -> RoutingResult:
        """
        Evaluate conditional rules and return first matching decision.

        Args:
            state: Current state snapshot
            context: Routing context

        Returns:
            RoutingResult with decision based on first matching rule
        """
        # Evaluate rules in priority order
        for rule in self.rules:
            if self._evaluate_condition(state, rule):
                return RoutingResult(
                    decision=rule.decision,
                    confidence=1.0,  # Rule-based = high confidence
                    reasoning=f"Conditional rule '{rule.name}' matched: "
                    f"{rule.condition_key} {rule.condition_operator} {rule.condition_value}",
                    strategy=self.strategy,
                    next_task=rule.next_task,
                    metadata={
                        "rule_id": rule.rule_id,
                        "rule_name": rule.name,
                        "rule_priority": rule.priority,
                    },
                )

        # No rules matched - continue by default
        return RoutingResult(
            decision=RoutingDecision.CONTINUE,
            confidence=0.5,
            reasoning="No conditional rules matched, continuing with default behavior",
            strategy=self.strategy,
            metadata={"rules_evaluated": len(self.rules)},
        )

    def _evaluate_condition(
        self, state: dict[str, Any], rule: ConditionalRule
    ) -> bool:
        """
        Evaluate a single conditional rule.

        Args:
            state: Current state
            rule: Rule to evaluate

        Returns:
            True if condition is met, False otherwise
        """
        # Get value from state
        value = state.get(rule.condition_key)

        if value is None:
            return False

        # Evaluate based on operator
        try:
            if rule.condition_operator == "==":
                return value == rule.condition_value
            elif rule.condition_operator == "!=":
                return value != rule.condition_value
            elif rule.condition_operator == ">":
                return value > rule.condition_value
            elif rule.condition_operator == "<":
                return value < rule.condition_value
            elif rule.condition_operator == ">=":
                return value >= rule.condition_value
            elif rule.condition_operator == "<=":
                return value <= rule.condition_value
            elif rule.condition_operator == "in":
                return value in rule.condition_value
            elif rule.condition_operator == "not_in":
                return value not in rule.condition_value
            elif rule.condition_operator == "contains":
                return rule.condition_value in value
            elif rule.condition_operator == "not_contains":
                return rule.condition_value not in value
            else:
                logger.warning(f"Unknown operator: {rule.condition_operator}")
                return False
        except (TypeError, KeyError, AttributeError) as e:
            logger.warning(
                f"Error evaluating condition for rule '{rule.rule_id}': {e}"
            )
            return False


class ParallelRouter(BaseRouter):
    """
    Identifies tasks that can be executed in parallel.

    Analyzes task dependencies and state to determine parallelizable operations.

    Example:
        ```python
        router = ParallelRouter(
            parallelization_hints=[
                ParallelizationHint(
                    task_group=["generate_model", "generate_validator"],
                    dependencies=["parse_contract"]
                )
            ]
        )
        result = router.evaluate(state, context)
        ```
    """

    def __init__(
        self,
        parallelization_hints: Optional[list[ParallelizationHint]] = None,
        metrics_collector: Optional[MetricsCollector] = None,
    ):
        """
        Initialize parallel router.

        Args:
            parallelization_hints: Optional hints for parallel execution
            metrics_collector: Optional metrics collector
        """
        super().__init__(metrics_collector)
        self.strategy = RoutingStrategy.PARALLEL
        self.hints = parallelization_hints or []

    def evaluate(
        self, state: dict[str, Any], context: RoutingContext
    ) -> RoutingResult:
        """
        Identify parallel execution opportunities.

        Args:
            state: Current state snapshot
            context: Routing context

        Returns:
            RoutingResult indicating if parallel execution is possible
        """
        # Check if current task matches any parallelization hints
        for hint in self.hints:
            if context.current_task in hint.task_group:
                # Check if dependencies are satisfied
                completed_tasks = state.get("completed_tasks", [])
                dependencies_met = all(
                    dep in completed_tasks for dep in hint.dependencies
                )

                if dependencies_met:
                    # Get other tasks in group that aren't completed yet
                    parallel_tasks = [
                        task
                        for task in hint.task_group
                        if task not in completed_tasks and task != context.current_task
                    ]

                    if parallel_tasks:
                        return RoutingResult(
                            decision=RoutingDecision.PARALLEL,
                            confidence=0.9,
                            reasoning=f"Task '{context.current_task}' can be parallelized with {parallel_tasks}",
                            strategy=self.strategy,
                            metadata={
                                "parallel_tasks": parallel_tasks,
                                "dependencies": hint.dependencies,
                                "estimated_duration_ms": hint.estimated_duration_ms,
                            },
                        )

        # No parallelization opportunities found
        return RoutingResult(
            decision=RoutingDecision.CONTINUE,
            confidence=0.7,
            reasoning="No parallel execution opportunities identified",
            strategy=self.strategy,
            metadata={"hints_checked": len(self.hints)},
        )


class StateAnalysisRouter(BaseRouter):
    """
    Analyzes state complexity to inform routing decisions.

    Makes routing decisions based on state size, complexity, and data quality.

    Example:
        ```python
        router = StateAnalysisRouter(
            max_complexity_score=0.8,
            error_handling_decision=RoutingDecision.RETRY
        )
        result = router.evaluate(state, context)
        ```
    """

    def __init__(
        self,
        max_complexity_score: float = 0.8,
        error_handling_decision: RoutingDecision = RoutingDecision.RETRY,
        metrics_collector: Optional[MetricsCollector] = None,
    ):
        """
        Initialize state analysis router.

        Args:
            max_complexity_score: Maximum complexity before branching (0.0-1.0)
            error_handling_decision: Decision to make when errors detected
            metrics_collector: Optional metrics collector
        """
        super().__init__(metrics_collector)
        self.strategy = RoutingStrategy.STATE_ANALYSIS
        self.max_complexity = max_complexity_score
        self.error_decision = error_handling_decision

    def evaluate(
        self, state: dict[str, Any], context: RoutingContext
    ) -> RoutingResult:
        """
        Analyze state complexity and make routing decision.

        Args:
            state: Current state snapshot
            context: Routing context

        Returns:
            RoutingResult based on state complexity analysis
        """
        # Calculate state complexity metrics
        metrics = self._calculate_complexity(state)

        # Check for errors first
        if metrics.has_errors:
            return RoutingResult(
                decision=self.error_decision,
                confidence=0.95,
                reasoning=f"State contains errors (key_count={metrics.key_count}, "
                f"complexity={metrics.complexity_score:.2f})",
                strategy=self.strategy,
                metadata={
                    "complexity_metrics": metrics.model_dump(),
                    "retry_count": context.retry_count,
                },
            )

        # Check for incomplete data
        if metrics.has_incomplete_data:
            return RoutingResult(
                decision=RoutingDecision.SKIP,
                confidence=0.8,
                reasoning="State has incomplete data, skipping task",
                strategy=self.strategy,
                metadata={"complexity_metrics": metrics.model_dump()},
            )

        # Check complexity threshold
        if metrics.complexity_score > self.max_complexity:
            return RoutingResult(
                decision=RoutingDecision.BRANCH,
                confidence=0.85,
                reasoning=f"State complexity ({metrics.complexity_score:.2f}) "
                f"exceeds threshold ({self.max_complexity:.2f}), branching to handle complexity",
                strategy=self.strategy,
                metadata={"complexity_metrics": metrics.model_dump()},
            )

        # Normal complexity - continue
        return RoutingResult(
            decision=RoutingDecision.CONTINUE,
            confidence=0.9,
            reasoning=f"State complexity ({metrics.complexity_score:.2f}) within normal range",
            strategy=self.strategy,
            metadata={"complexity_metrics": metrics.model_dump()},
        )

    def _calculate_complexity(self, state: dict[str, Any]) -> StateComplexityMetrics:
        """
        Calculate state complexity metrics.

        Args:
            state: State to analyze

        Returns:
            StateComplexityMetrics with quantitative metrics
        """
        key_count = len(state)
        nested_depth = self._get_max_depth(state)
        total_size = self._estimate_size(state)

        # Calculate complexity score (0.0-1.0)
        # Based on: key_count (40%), nested_depth (30%), total_size (30%)
        complexity_score = min(
            1.0,
            (key_count / 100.0) * 0.4
            + (nested_depth / 10.0) * 0.3
            + (total_size / 1000000.0) * 0.3,  # 1MB baseline
        )

        # Check for error indicators
        has_errors = any(
            key in state and state[key]
            for key in ["error", "errors", "error_count", "failed"]
        )

        # Check for incomplete data
        has_incomplete = any(
            key in state and not state[key]
            for key in ["required_data", "mandatory_fields"]
        )

        return StateComplexityMetrics(
            key_count=key_count,
            nested_depth=nested_depth,
            total_size_bytes=total_size,
            complexity_score=complexity_score,
            has_errors=has_errors,
            has_incomplete_data=has_incomplete,
        )

    def _get_max_depth(self, obj: Any, current_depth: int = 0) -> int:
        """Recursively calculate maximum nesting depth."""
        if not isinstance(obj, dict):
            return current_depth

        if not obj:
            return current_depth

        return max(
            self._get_max_depth(value, current_depth + 1) for value in obj.values()
        )

    def _estimate_size(self, obj: Any) -> int:
        """Estimate size of object in bytes (rough approximation)."""
        try:
            import sys

            return sys.getsizeof(obj)
        except Exception:
            # Fallback: rough estimate
            if isinstance(obj, dict):
                return sum(self._estimate_size(v) for v in obj.values())
            elif isinstance(obj, (list, tuple)):
                return sum(self._estimate_size(item) for item in obj)
            elif isinstance(obj, str):
                return len(obj)
            else:
                return 8  # Rough estimate for other types


class PriorityRouter(BaseRouter):
    """
    Priority-based routing for task scheduling.

    Routes tasks based on priority levels with configurable thresholds.

    Example:
        ```python
        config = PriorityRoutingConfig(
            high_priority_threshold=80,
            low_priority_threshold=20
        )
        router = PriorityRouter(config=config)
        result = router.evaluate(state, context)
        ```
    """

    def __init__(
        self,
        config: Optional[PriorityRoutingConfig] = None,
        metrics_collector: Optional[MetricsCollector] = None,
    ):
        """
        Initialize priority router.

        Args:
            config: Optional priority routing configuration
            metrics_collector: Optional metrics collector
        """
        super().__init__(metrics_collector)
        self.strategy = RoutingStrategy.PRIORITY
        self.config = config or PriorityRoutingConfig()

    def evaluate(
        self, state: dict[str, Any], context: RoutingContext
    ) -> RoutingResult:
        """
        Route based on task priority.

        Args:
            state: Current state snapshot
            context: Routing context

        Returns:
            RoutingResult based on priority level
        """
        # Get priority from context or state
        priority = context.custom_data.get("priority") or state.get("task_priority", 50)

        # Determine decision based on priority
        if priority >= self.config.high_priority_threshold:
            decision = self.config.high_priority_decision
            confidence = 0.95
            reasoning = f"High-priority task (priority={priority})"
        elif priority <= self.config.low_priority_threshold:
            decision = self.config.low_priority_decision
            confidence = 0.85
            reasoning = f"Low-priority task (priority={priority})"
        else:
            decision = self.config.default_decision
            confidence = 0.8
            reasoning = f"Medium-priority task (priority={priority})"

        return RoutingResult(
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            strategy=self.strategy,
            metadata={
                "priority": priority,
                "high_threshold": self.config.high_priority_threshold,
                "low_threshold": self.config.low_priority_threshold,
            },
        )


class SmartRoutingOrchestrator:
    """
    Orchestrates multiple routing strategies with priority-based consolidation.

    Collects decisions from all routers and consolidates them based on confidence
    and strategy priority.

    Performance Targets:
    - Routing decision: <5ms with caching
    - Throughput: 100+ routing decisions per second
    - History tracking: Automatic with configurable size

    Example:
        ```python
        orchestrator = SmartRoutingOrchestrator(
            metrics_collector=metrics_collector,
            max_history_size=1000
        )

        # Add routers
        orchestrator.add_router(conditional_router)
        orchestrator.add_router(parallel_router)
        orchestrator.add_router(state_analysis_router)
        orchestrator.add_router(priority_router)

        # Make routing decision
        result = orchestrator.route(
            state=state_snapshot,
            current_task="generate_model",
            execution_time=45.2
        )
        ```
    """

    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        state: Optional[ThreadSafeState] = None,
        max_history_size: int = 1000,
    ):
        """
        Initialize routing orchestrator.

        Args:
            metrics_collector: Optional metrics collector for performance tracking
            state: Optional ThreadSafeState for state access
            max_history_size: Maximum routing history records to keep
        """
        self.metrics = metrics_collector
        self.state = state
        self.routers: dict[RoutingStrategy, BaseRouter] = {}
        self.routing_history: deque[RoutingHistoryRecord] = deque(
            maxlen=max_history_size
        )

        # Strategy priority (higher = more important)
        self.strategy_priority = {
            RoutingStrategy.CONDITIONAL: 100,  # Highest - explicit rules
            RoutingStrategy.STATE_ANALYSIS: 80,  # High - data-driven
            RoutingStrategy.PARALLEL: 60,  # Medium - optimization
            RoutingStrategy.PRIORITY: 40,  # Lower - general scheduling
        }

    def add_router(self, router: BaseRouter) -> None:
        """
        Add a routing strategy.

        Args:
            router: Router to add
        """
        self.routers[router.strategy] = router
        logger.info(f"Added router: {router.strategy.value}")

    def remove_router(self, strategy: RoutingStrategy) -> None:
        """
        Remove a routing strategy.

        Args:
            strategy: Strategy to remove
        """
        if strategy in self.routers:
            del self.routers[strategy]
            logger.info(f"Removed router: {strategy.value}")

    @timed("routing_decision_time_ms")
    def route(
        self,
        state: dict[str, Any],
        current_task: str,
        execution_time: float = 0.0,
        retry_count: int = 0,
        custom_data: Optional[dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Make routing decision using all registered routers.

        Performance Target: <5ms per routing decision

        Args:
            state: Current state snapshot
            current_task: Task being evaluated
            execution_time: Time spent executing current task (ms)
            retry_count: Number of retries for current task
            custom_data: Additional context data
            correlation_id: Optional correlation ID for tracing

        Returns:
            Dictionary with routing decision and metadata
        """
        start_time = time.perf_counter()

        # Create routing context
        context = RoutingContext(
            current_task=current_task,
            state_summary=self._summarize_state(state),
            execution_time=execution_time,
            retry_count=retry_count,
            custom_data=custom_data or {},
            correlation_id=correlation_id,
        )

        # Collect decisions from all routers
        results: list[tuple[RoutingResult, int]] = []

        for strategy, router in self.routers.items():
            try:
                result = router.evaluate(state, context)
                priority = self.strategy_priority.get(strategy, 50)
                results.append((result, priority))
            except Exception as e:
                logger.error(f"Error in router {strategy.value}: {e}")
                continue

        # Consolidate decisions (priority-based with confidence weighting)
        if not results:
            # No routers available - default to CONTINUE
            final_result = RoutingResult(
                decision=RoutingDecision.CONTINUE,
                confidence=0.5,
                reasoning="No routers available, using default decision",
                strategy=RoutingStrategy.PRIORITY,
            )
        else:
            final_result = self._consolidate_decisions(results)

        # Calculate routing time
        routing_time_ms = (time.perf_counter() - start_time) * 1000

        # Record in history
        history_record = RoutingHistoryRecord(
            routing_id=context.routing_id,
            context=context,
            result=final_result,
            routing_time_ms=routing_time_ms,
        )
        self.routing_history.append(history_record)

        # Record metrics (only if event loop is running)
        if self.metrics:
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                asyncio.create_task(
                    self.metrics.record_timing(
                        "routing_decision_time_ms",
                        routing_time_ms,
                        tags={
                            "decision": final_result.decision.value,
                            "strategy": final_result.strategy.value,
                        },
                        correlation_id=correlation_id,
                    )
                )
                asyncio.create_task(
                    self.metrics.record_counter(
                        "routing_decisions_count",
                        count=1,
                        tags={"decision": final_result.decision.value},
                        correlation_id=correlation_id,
                    )
                )
            except RuntimeError:
                # No running event loop - skip metrics recording
                # This is expected in synchronous test contexts
                pass

        # Return comprehensive result
        return {
            "routing_id": str(context.routing_id),
            "decision": final_result.decision.value,
            "confidence": final_result.confidence,
            "reasoning": final_result.reasoning,
            "strategy": final_result.strategy.value,
            "next_task": final_result.next_task,
            "metadata": final_result.metadata,
            "routing_time_ms": routing_time_ms,
            "routers_evaluated": len(results),
        }

    def _consolidate_decisions(
        self, results: list[tuple[RoutingResult, int]]
    ) -> RoutingResult:
        """
        Consolidate multiple routing decisions using priority and confidence.

        Args:
            results: List of (RoutingResult, priority) tuples

        Returns:
            Consolidated RoutingResult
        """
        # Calculate weighted scores: (confidence * priority)
        weighted_results = [
            (result, result.confidence * priority) for result, priority in results
        ]

        # Sort by weighted score (highest first)
        weighted_results.sort(key=lambda x: x[1], reverse=True)

        # Return highest weighted decision
        best_result, best_score = weighted_results[0]

        return best_result

    def _summarize_state(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Create a summary of state for routing context.

        Args:
            state: Full state

        Returns:
            Summarized state (reduced size for performance)
        """
        return {
            "key_count": len(state),
            "has_errors": any(
                key in state for key in ["error", "errors", "error_count"]
            ),
            "completed_tasks": state.get("completed_tasks", []),
            "task_priority": state.get("task_priority", 50),
        }

    def get_history(
        self, task: Optional[str] = None, limit: Optional[int] = None
    ) -> list[RoutingHistoryRecord]:
        """
        Get routing history.

        Args:
            task: Filter by task name (None = all tasks)
            limit: Maximum number of records to return (None = all)

        Returns:
            List of routing history records (most recent first)
        """
        history = list(self.routing_history)

        # Filter by task if specified
        if task is not None:
            history = [
                record for record in history if record.context.current_task == task
            ]

        # Reverse to get most recent first
        history.reverse()

        # Limit if specified
        if limit is not None:
            history = history[:limit]

        return history

    def get_stats(self) -> dict[str, Any]:
        """
        Get routing statistics.

        Returns:
            Dictionary with routing statistics
        """
        if not self.routing_history:
            return {
                "total_routings": 0,
                "avg_routing_time_ms": 0.0,
                "min_routing_time_ms": 0.0,
                "max_routing_time_ms": 0.0,
                "decisions": {},
                "strategies": {},
            }

        routing_times = [record.routing_time_ms for record in self.routing_history]
        decisions = {}
        strategies = {}

        for record in self.routing_history:
            # Count decisions
            decision_key = record.result.decision.value
            decisions[decision_key] = decisions.get(decision_key, 0) + 1

            # Count strategies
            strategy_key = record.result.strategy.value
            strategies[strategy_key] = strategies.get(strategy_key, 0) + 1

        return {
            "total_routings": len(self.routing_history),
            "avg_routing_time_ms": sum(routing_times) / len(routing_times),
            "min_routing_time_ms": min(routing_times),
            "max_routing_time_ms": max(routing_times),
            "decisions": decisions,
            "strategies": strategies,
        }

    def clear_history(self) -> None:
        """Clear routing history."""
        self.routing_history.clear()
        logger.info("Routing history cleared")
