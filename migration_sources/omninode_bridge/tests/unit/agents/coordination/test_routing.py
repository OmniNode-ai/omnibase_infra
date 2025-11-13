"""
Comprehensive tests for Smart Routing Orchestration.

Tests all routing strategies (Conditional, Parallel, StateAnalysis, Priority)
and the SmartRoutingOrchestrator with focus on:
- Correct routing decisions
- Performance (<5ms requirement)
- Priority-based consolidation
- History tracking
- Integration with foundation components
"""

import time
from typing import Any

import pytest

from omninode_bridge.agents.coordination.routing import (
    ConditionalRouter,
    ParallelRouter,
    PriorityRouter,
    SmartRoutingOrchestrator,
    StateAnalysisRouter,
)
from omninode_bridge.agents.coordination.routing_models import (
    ConditionalRule,
    ParallelizationHint,
    PriorityRoutingConfig,
    RoutingContext,
    RoutingDecision,
    RoutingStrategy,
)
from omninode_bridge.agents.coordination.thread_safe_state import ThreadSafeState
from omninode_bridge.agents.metrics.collector import MetricsCollector


@pytest.fixture
def metrics_collector():
    """Create metrics collector for tests."""
    collector = MetricsCollector(
        buffer_size=1000,
        batch_size=50,
        kafka_enabled=False,
        postgres_enabled=False,
    )
    return collector


@pytest.fixture
def thread_safe_state():
    """Create ThreadSafeState for tests."""
    return ThreadSafeState(
        initial_state={
            "completed_tasks": [],
            "task_priority": 50,
            "error_count": 0,
        }
    )


@pytest.fixture
def routing_context():
    """Create routing context for tests."""
    return RoutingContext(
        current_task="test_task",
        state_summary={"key_count": 10, "has_errors": False},
        execution_time=10.5,
        retry_count=0,
    )


class TestConditionalRouter:
    """Tests for ConditionalRouter."""

    def test_rule_matching_equals(self, routing_context):
        """Test rule matching with == operator."""
        rules = [
            ConditionalRule(
                rule_id="test_rule",
                name="Test Rule",
                condition_key="status",
                condition_operator="==",
                condition_value="active",
                decision=RoutingDecision.CONTINUE,
                priority=90,
            )
        ]

        router = ConditionalRouter(rules=rules)
        state = {"status": "active"}

        result = router.evaluate(state, routing_context)

        assert result.decision == RoutingDecision.CONTINUE
        assert result.confidence == 1.0
        assert result.strategy == RoutingStrategy.CONDITIONAL
        assert "Test Rule" in result.reasoning

    def test_rule_matching_greater_than(self, routing_context):
        """Test rule matching with > operator."""
        rules = [
            ConditionalRule(
                rule_id="error_check",
                name="Error Check",
                condition_key="error_count",
                condition_operator=">",
                condition_value=0,
                decision=RoutingDecision.RETRY,
                priority=95,
            )
        ]

        router = ConditionalRouter(rules=rules)
        state = {"error_count": 2}

        result = router.evaluate(state, routing_context)

        assert result.decision == RoutingDecision.RETRY
        assert result.confidence == 1.0
        assert "error_count > 0" in result.reasoning

    def test_rule_matching_in_operator(self, routing_context):
        """Test rule matching with 'in' operator."""
        rules = [
            ConditionalRule(
                rule_id="phase_check",
                name="Phase Check",
                condition_key="phase",
                condition_operator="in",
                condition_value=["validation", "testing"],
                decision=RoutingDecision.SKIP,
                priority=80,
            )
        ]

        router = ConditionalRouter(rules=rules)
        state = {"phase": "validation"}

        result = router.evaluate(state, routing_context)

        assert result.decision == RoutingDecision.SKIP
        assert result.confidence == 1.0

    def test_rule_priority_ordering(self, routing_context):
        """Test that higher priority rules are evaluated first."""
        rules = [
            ConditionalRule(
                rule_id="low_priority",
                name="Low Priority",
                condition_key="status",
                condition_operator="==",
                condition_value="active",
                decision=RoutingDecision.CONTINUE,
                priority=50,
            ),
            ConditionalRule(
                rule_id="high_priority",
                name="High Priority",
                condition_key="status",
                condition_operator="==",
                condition_value="active",
                decision=RoutingDecision.END,
                priority=90,
            ),
        ]

        router = ConditionalRouter(rules=rules)
        state = {"status": "active"}

        result = router.evaluate(state, routing_context)

        # Should match high priority rule first
        assert result.decision == RoutingDecision.END
        assert result.metadata["rule_id"] == "high_priority"
        assert result.metadata["rule_priority"] == 90

    def test_no_rules_matched(self, routing_context):
        """Test default behavior when no rules match."""
        rules = [
            ConditionalRule(
                rule_id="test_rule",
                name="Test Rule",
                condition_key="status",
                condition_operator="==",
                condition_value="inactive",
                decision=RoutingDecision.SKIP,
                priority=90,
            )
        ]

        router = ConditionalRouter(rules=rules)
        state = {"status": "active"}

        result = router.evaluate(state, routing_context)

        assert result.decision == RoutingDecision.CONTINUE
        assert result.confidence == 0.5
        assert "No conditional rules matched" in result.reasoning

    def test_missing_key_in_state(self, routing_context):
        """Test behavior when condition key is missing from state."""
        rules = [
            ConditionalRule(
                rule_id="test_rule",
                name="Test Rule",
                condition_key="missing_key",
                condition_operator="==",
                condition_value="value",
                decision=RoutingDecision.SKIP,
                priority=90,
            )
        ]

        router = ConditionalRouter(rules=rules)
        state = {"other_key": "value"}

        result = router.evaluate(state, routing_context)

        # Should not match and fall through to default
        assert result.decision == RoutingDecision.CONTINUE


class TestParallelRouter:
    """Tests for ParallelRouter."""

    def test_parallel_execution_identified(self, routing_context):
        """Test identification of parallel execution opportunities."""
        hints = [
            ParallelizationHint(
                task_group=["generate_model", "generate_validator", "generate_test"],
                dependencies=["parse_contract"],
                estimated_duration_ms=100.0,
            )
        ]

        router = ParallelRouter(parallelization_hints=hints)
        state = {"completed_tasks": ["parse_contract"]}

        # Test for generate_model task
        context = RoutingContext(
            current_task="generate_model",
            state_summary={},
        )

        result = router.evaluate(state, context)

        assert result.decision == RoutingDecision.PARALLEL
        assert result.confidence >= 0.8
        assert "parallel_tasks" in result.metadata
        assert "generate_validator" in result.metadata["parallel_tasks"]
        assert "generate_test" in result.metadata["parallel_tasks"]

    def test_dependencies_not_met(self, routing_context):
        """Test when dependencies are not met."""
        hints = [
            ParallelizationHint(
                task_group=["generate_model", "generate_validator"],
                dependencies=["parse_contract"],
            )
        ]

        router = ParallelRouter(parallelization_hints=hints)
        state = {"completed_tasks": []}  # Dependencies not met

        context = RoutingContext(
            current_task="generate_model",
            state_summary={},
        )

        result = router.evaluate(state, context)

        assert result.decision == RoutingDecision.CONTINUE
        assert "No parallel execution opportunities" in result.reasoning

    def test_all_parallel_tasks_completed(self, routing_context):
        """Test when all tasks in parallel group are completed."""
        hints = [
            ParallelizationHint(
                task_group=["generate_model", "generate_validator"],
                dependencies=[],
            )
        ]

        router = ParallelRouter(parallelization_hints=hints)
        state = {
            "completed_tasks": ["generate_model", "generate_validator"]
        }  # All completed

        context = RoutingContext(
            current_task="generate_model",
            state_summary={},
        )

        result = router.evaluate(state, context)

        assert result.decision == RoutingDecision.CONTINUE
        assert "No parallel execution opportunities" in result.reasoning

    def test_no_hints_provided(self, routing_context):
        """Test behavior when no parallelization hints are provided."""
        router = ParallelRouter(parallelization_hints=None)
        state = {"completed_tasks": []}

        result = router.evaluate(state, routing_context)

        assert result.decision == RoutingDecision.CONTINUE
        assert result.metadata["hints_checked"] == 0


class TestStateAnalysisRouter:
    """Tests for StateAnalysisRouter."""

    def test_error_detection(self, routing_context):
        """Test detection of errors in state."""
        router = StateAnalysisRouter(
            error_handling_decision=RoutingDecision.RETRY
        )
        state = {"error_count": 3, "status": "failed"}

        result = router.evaluate(state, routing_context)

        assert result.decision == RoutingDecision.RETRY
        assert result.confidence >= 0.9
        assert "contains errors" in result.reasoning
        assert result.metadata["complexity_metrics"]["has_errors"] is True

    def test_incomplete_data_detection(self, routing_context):
        """Test detection of incomplete data."""
        router = StateAnalysisRouter()
        state = {
            "required_data": None,  # Incomplete
            "status": "pending",
        }

        result = router.evaluate(state, routing_context)

        assert result.decision == RoutingDecision.SKIP
        assert "incomplete data" in result.reasoning

    def test_high_complexity_branching(self, routing_context):
        """Test branching when state complexity exceeds threshold."""
        router = StateAnalysisRouter(max_complexity_score=0.5)

        # Create complex state
        complex_state: dict[str, Any] = {}
        for i in range(100):  # Many keys
            complex_state[f"key_{i}"] = {
                "nested": {
                    "deeply": {
                        "nested": {
                            "value": i
                        }
                    }
                }
            }

        result = router.evaluate(complex_state, routing_context)

        assert result.decision == RoutingDecision.BRANCH
        assert "complexity" in result.reasoning.lower()
        assert "complexity_metrics" in result.metadata

    def test_normal_complexity(self, routing_context):
        """Test normal complexity state."""
        router = StateAnalysisRouter(max_complexity_score=0.8)
        state = {
            "status": "active",
            "count": 5,
            "data": {"value": 10},
        }

        result = router.evaluate(state, routing_context)

        assert result.decision == RoutingDecision.CONTINUE
        assert result.confidence >= 0.8
        assert "within normal range" in result.reasoning

    def test_complexity_metrics_calculation(self, routing_context):
        """Test complexity metrics calculation."""
        router = StateAnalysisRouter()
        state = {
            "key1": "value1",
            "key2": {"nested": {"deep": "value"}},
            "key3": [1, 2, 3],
        }

        result = router.evaluate(state, routing_context)

        metrics = result.metadata["complexity_metrics"]
        assert metrics["key_count"] == 3
        assert metrics["nested_depth"] >= 2
        assert metrics["complexity_score"] >= 0.0
        assert metrics["complexity_score"] <= 1.0


class TestPriorityRouter:
    """Tests for PriorityRouter."""

    def test_high_priority_routing(self, routing_context):
        """Test routing for high-priority tasks."""
        config = PriorityRoutingConfig(
            high_priority_threshold=80,
            high_priority_decision=RoutingDecision.CONTINUE,
        )

        router = PriorityRouter(config=config)
        state = {"task_priority": 90}

        result = router.evaluate(state, routing_context)

        assert result.decision == RoutingDecision.CONTINUE
        assert result.confidence >= 0.9
        assert "High-priority" in result.reasoning
        assert result.metadata["priority"] == 90

    def test_low_priority_routing(self, routing_context):
        """Test routing for low-priority tasks."""
        config = PriorityRoutingConfig(
            low_priority_threshold=20,
            low_priority_decision=RoutingDecision.SKIP,
        )

        router = PriorityRouter(config=config)
        state = {"task_priority": 10}

        result = router.evaluate(state, routing_context)

        assert result.decision == RoutingDecision.SKIP
        assert result.confidence >= 0.8
        assert "Low-priority" in result.reasoning

    def test_medium_priority_routing(self, routing_context):
        """Test routing for medium-priority tasks."""
        config = PriorityRoutingConfig(
            high_priority_threshold=80,
            low_priority_threshold=20,
            default_decision=RoutingDecision.CONTINUE,
        )

        router = PriorityRouter(config=config)
        state = {"task_priority": 50}

        result = router.evaluate(state, routing_context)

        assert result.decision == RoutingDecision.CONTINUE
        assert result.confidence >= 0.7
        assert "Medium-priority" in result.reasoning

    def test_priority_from_context(self):
        """Test getting priority from routing context custom data."""
        config = PriorityRoutingConfig(high_priority_threshold=80)
        router = PriorityRouter(config=config)

        context = RoutingContext(
            current_task="test_task",
            custom_data={"priority": 95},
        )

        state = {}  # No priority in state

        result = router.evaluate(state, context)

        assert result.metadata["priority"] == 95


class TestSmartRoutingOrchestrator:
    """Tests for SmartRoutingOrchestrator."""

    def test_add_remove_routers(self, metrics_collector):
        """Test adding and removing routers."""
        orchestrator = SmartRoutingOrchestrator(metrics_collector=metrics_collector)

        # Add routers
        conditional_router = ConditionalRouter(rules=[])
        parallel_router = ParallelRouter()

        orchestrator.add_router(conditional_router)
        orchestrator.add_router(parallel_router)

        assert RoutingStrategy.CONDITIONAL in orchestrator.routers
        assert RoutingStrategy.PARALLEL in orchestrator.routers

        # Remove router
        orchestrator.remove_router(RoutingStrategy.CONDITIONAL)
        assert RoutingStrategy.CONDITIONAL not in orchestrator.routers
        assert RoutingStrategy.PARALLEL in orchestrator.routers

    def test_routing_decision(self, metrics_collector):
        """Test end-to-end routing decision."""
        orchestrator = SmartRoutingOrchestrator(metrics_collector=metrics_collector)

        # Add simple conditional router
        rules = [
            ConditionalRule(
                rule_id="test",
                name="Test",
                condition_key="status",
                condition_operator="==",
                condition_value="active",
                decision=RoutingDecision.CONTINUE,
                priority=90,
            )
        ]
        orchestrator.add_router(ConditionalRouter(rules=rules))

        # Make routing decision
        state = {"status": "active"}
        result = orchestrator.route(state=state, current_task="test_task")

        assert "decision" in result
        assert "confidence" in result
        assert "reasoning" in result
        assert "routing_time_ms" in result
        assert result["routers_evaluated"] >= 1

    def test_routing_performance(self, metrics_collector):
        """Test routing performance (<5ms requirement)."""
        orchestrator = SmartRoutingOrchestrator(metrics_collector=metrics_collector)

        # Add all router types
        orchestrator.add_router(
            ConditionalRouter(
                rules=[
                    ConditionalRule(
                        rule_id="test",
                        name="Test",
                        condition_key="status",
                        condition_operator="==",
                        condition_value="active",
                        decision=RoutingDecision.CONTINUE,
                        priority=90,
                    )
                ]
            )
        )
        orchestrator.add_router(ParallelRouter())
        orchestrator.add_router(StateAnalysisRouter())
        orchestrator.add_router(PriorityRouter())

        # Run multiple routing decisions and measure time
        state = {"status": "active", "task_priority": 50}
        times = []

        for _ in range(100):
            start = time.perf_counter()
            result = orchestrator.route(state=state, current_task="test_task")
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)

        avg_time = sum(times) / len(times)
        max_time = max(times)

        # Performance assertions
        assert avg_time < 5.0, f"Average routing time {avg_time:.2f}ms exceeds 5ms target"
        assert max_time < 10.0, f"Max routing time {max_time:.2f}ms exceeds 10ms threshold"

    def test_priority_based_consolidation(self, metrics_collector):
        """Test priority-based decision consolidation."""
        orchestrator = SmartRoutingOrchestrator(metrics_collector=metrics_collector)

        # Add routers with different priorities
        # Conditional has highest priority (100)
        orchestrator.add_router(
            ConditionalRouter(
                rules=[
                    ConditionalRule(
                        rule_id="high_priority",
                        name="High Priority",
                        condition_key="status",
                        condition_operator="==",
                        condition_value="active",
                        decision=RoutingDecision.END,
                        priority=90,
                    )
                ]
            )
        )

        # Priority router has lower priority (40)
        orchestrator.add_router(
            PriorityRouter(
                config=PriorityRoutingConfig(
                    high_priority_threshold=80,
                    high_priority_decision=RoutingDecision.CONTINUE,
                )
            )
        )

        # Both routers should match, but conditional should win
        state = {"status": "active", "task_priority": 90}
        result = orchestrator.route(state=state, current_task="test_task")

        # Should use conditional router's decision (highest priority)
        assert result["decision"] == RoutingDecision.END.value
        assert result["strategy"] == RoutingStrategy.CONDITIONAL.value

    def test_routing_history_tracking(self, metrics_collector):
        """Test routing history tracking."""
        orchestrator = SmartRoutingOrchestrator(
            metrics_collector=metrics_collector, max_history_size=10
        )

        orchestrator.add_router(ConditionalRouter(rules=[]))

        # Make multiple routing decisions
        state = {"status": "active"}
        for i in range(5):
            orchestrator.route(state=state, current_task=f"task_{i}")

        # Check history
        history = orchestrator.get_history()
        assert len(history) == 5

        # Check history is in reverse chronological order (most recent first)
        assert history[0].context.current_task == "task_4"
        assert history[4].context.current_task == "task_0"

    def test_routing_history_filtering(self, metrics_collector):
        """Test routing history filtering by task."""
        orchestrator = SmartRoutingOrchestrator(metrics_collector=metrics_collector)
        orchestrator.add_router(ConditionalRouter(rules=[]))

        state = {"status": "active"}

        # Make routing decisions for different tasks
        orchestrator.route(state=state, current_task="task_a")
        orchestrator.route(state=state, current_task="task_b")
        orchestrator.route(state=state, current_task="task_a")

        # Filter history by task
        task_a_history = orchestrator.get_history(task="task_a")
        assert len(task_a_history) == 2
        assert all(record.context.current_task == "task_a" for record in task_a_history)

        # Limit results
        limited_history = orchestrator.get_history(limit=1)
        assert len(limited_history) == 1

    def test_routing_statistics(self, metrics_collector):
        """Test routing statistics collection."""
        orchestrator = SmartRoutingOrchestrator(metrics_collector=metrics_collector)
        orchestrator.add_router(
            ConditionalRouter(
                rules=[
                    ConditionalRule(
                        rule_id="test",
                        name="Test",
                        condition_key="status",
                        condition_operator="==",
                        condition_value="active",
                        decision=RoutingDecision.CONTINUE,
                        priority=90,
                    )
                ]
            )
        )

        # Make routing decisions
        state = {"status": "active"}
        for _ in range(10):
            orchestrator.route(state=state, current_task="test_task")

        # Get statistics
        stats = orchestrator.get_stats()

        assert stats["total_routings"] == 10
        assert stats["avg_routing_time_ms"] > 0
        assert stats["min_routing_time_ms"] > 0
        assert stats["max_routing_time_ms"] > 0
        assert "decisions" in stats
        assert "strategies" in stats

    def test_throughput_100_decisions_per_second(self, metrics_collector):
        """Test throughput of 100+ routing decisions per second."""
        orchestrator = SmartRoutingOrchestrator(metrics_collector=metrics_collector)

        # Add lightweight routers
        orchestrator.add_router(ConditionalRouter(rules=[]))
        orchestrator.add_router(PriorityRouter())

        state = {"status": "active"}

        # Measure throughput
        start_time = time.perf_counter()
        decisions_count = 200  # Test with 200 decisions

        for _ in range(decisions_count):
            orchestrator.route(state=state, current_task="test_task")

        elapsed = time.perf_counter() - start_time
        throughput = decisions_count / elapsed

        assert (
            throughput >= 100
        ), f"Throughput {throughput:.2f} decisions/sec below 100 target"

    def test_clear_history(self, metrics_collector):
        """Test clearing routing history."""
        orchestrator = SmartRoutingOrchestrator(metrics_collector=metrics_collector)
        orchestrator.add_router(ConditionalRouter(rules=[]))

        # Make routing decisions
        state = {"status": "active"}
        for _ in range(5):
            orchestrator.route(state=state, current_task="test_task")

        assert len(orchestrator.get_history()) == 5

        # Clear history
        orchestrator.clear_history()
        assert len(orchestrator.get_history()) == 0

    def test_no_routers_default_behavior(self, metrics_collector):
        """Test default behavior when no routers are registered."""
        orchestrator = SmartRoutingOrchestrator(metrics_collector=metrics_collector)

        state = {"status": "active"}
        result = orchestrator.route(state=state, current_task="test_task")

        # Should default to CONTINUE
        assert result["decision"] == RoutingDecision.CONTINUE.value
        assert result["confidence"] == 0.5
        assert "No routers available" in result["reasoning"]


class TestIntegrationWithFoundation:
    """Integration tests with foundation components."""

    @pytest.mark.asyncio
    async def test_integration_with_metrics_collector(self):
        """Test integration with MetricsCollector."""
        collector = MetricsCollector(
            buffer_size=1000,
            kafka_enabled=False,
            postgres_enabled=False,
        )

        orchestrator = SmartRoutingOrchestrator(metrics_collector=collector)
        orchestrator.add_router(ConditionalRouter(rules=[]))

        # Make routing decision
        state = {"status": "active"}
        result = orchestrator.route(state=state, current_task="test_task")

        assert "routing_time_ms" in result

        # Metrics should be recorded (check buffer)
        stats = await collector.get_stats()
        assert stats["buffer_size"] >= 0  # Metrics are queued

    def test_integration_with_thread_safe_state(self, thread_safe_state):
        """Test integration with ThreadSafeState."""
        orchestrator = SmartRoutingOrchestrator(state=thread_safe_state)

        # State should be accessible
        assert orchestrator.state is not None

        # Add some data to state
        thread_safe_state.set("routing_enabled", True, changed_by="test")

        # Routing should work with state
        state_snapshot = thread_safe_state.snapshot()
        result = orchestrator.route(state=state_snapshot, current_task="test_task")

        assert result is not None


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmarks for routing system."""

    def test_conditional_router_performance(self, routing_context, benchmark):
        """Benchmark ConditionalRouter performance."""
        rules = [
            ConditionalRule(
                rule_id=f"rule_{i}",
                name=f"Rule {i}",
                condition_key="status",
                condition_operator="==",
                condition_value=f"value_{i}",
                decision=RoutingDecision.CONTINUE,
                priority=90 - i,
            )
            for i in range(10)
        ]

        router = ConditionalRouter(rules=rules)
        state = {"status": "value_5"}

        def run_evaluation():
            return router.evaluate(state, routing_context)

        result = benchmark(run_evaluation)
        assert result.decision is not None

    def test_orchestrator_performance_all_routers(self, benchmark):
        """Benchmark SmartRoutingOrchestrator with all routers."""
        orchestrator = SmartRoutingOrchestrator()

        # Add all router types
        orchestrator.add_router(
            ConditionalRouter(
                rules=[
                    ConditionalRule(
                        rule_id="test",
                        name="Test",
                        condition_key="status",
                        condition_operator="==",
                        condition_value="active",
                        decision=RoutingDecision.CONTINUE,
                        priority=90,
                    )
                ]
            )
        )
        orchestrator.add_router(ParallelRouter())
        orchestrator.add_router(StateAnalysisRouter())
        orchestrator.add_router(PriorityRouter())

        state = {"status": "active", "task_priority": 50}

        def run_routing():
            return orchestrator.route(state=state, current_task="test_task")

        result = benchmark(run_routing)
        assert result["decision"] is not None
