"""
Comprehensive tests for Error Recovery Orchestration (Pattern 7).

Tests all recovery strategies, error pattern matching, and performance targets.

Performance targets:
- Error analysis: <100ms
- Recovery decision: <50ms
- Total recovery overhead: <500ms
- Success rate: 80%+ for recoverable errors
"""

import asyncio
import time

import pytest

from omninode_bridge.agents.coordination.signals import SignalCoordinator
from omninode_bridge.agents.coordination.thread_safe_state import ThreadSafeState
from omninode_bridge.agents.metrics.collector import MetricsCollector
from omninode_bridge.agents.workflows.error_recovery import ErrorRecoveryOrchestrator
from omninode_bridge.agents.workflows.recovery_models import (
    ErrorPattern,
    ErrorType,
    RecoveryContext,
    RecoveryResult,
    RecoveryStatistics,
    RecoveryStrategy,
)
from omninode_bridge.agents.workflows.recovery_strategies import (
    AlternativePathStrategy,
    ErrorCorrectionStrategy,
    EscalationStrategy,
    GracefulDegradationStrategy,
    RetryStrategy,
)

# Test fixtures


@pytest.fixture
def metrics_collector():
    """Create MetricsCollector for testing."""
    return MetricsCollector(buffer_size=100, batch_size=10, flush_interval_ms=1000)


@pytest.fixture
def signal_coordinator(metrics_collector):
    """Create SignalCoordinator for testing."""
    state = ThreadSafeState()
    return SignalCoordinator(state=state, metrics_collector=metrics_collector)


@pytest.fixture
def error_orchestrator(metrics_collector, signal_coordinator):
    """Create ErrorRecoveryOrchestrator for testing."""
    return ErrorRecoveryOrchestrator(
        metrics_collector=metrics_collector,
        signal_coordinator=signal_coordinator,
        max_retries=3,
        base_delay=0.1,  # Fast for testing
    )


@pytest.fixture
def recovery_context():
    """Create RecoveryContext for testing."""
    return RecoveryContext(
        workflow_id="test-workflow-1",
        node_name="test_generator",
        step_count=5,
        state={"contract": {"name": "TestContract"}, "template_name": "default"},
        exception=SyntaxError("invalid syntax"),
        error_message="SyntaxError: invalid syntax at line 45",
        correlation_id="test-correlation-1",
    )


# Recovery Models Tests


class TestRecoveryModels:
    """Test recovery data models."""

    def test_error_pattern_validation(self):
        """Test ErrorPattern validation."""
        # Valid pattern
        pattern = ErrorPattern(
            pattern_id="test_pattern",
            error_type=ErrorType.SYNTAX,
            regex_pattern=r"SyntaxError.*",
            recovery_strategy=RecoveryStrategy.ERROR_CORRECTION,
            max_retries=3,
            priority=7,
        )
        assert pattern.pattern_id == "test_pattern"
        assert pattern.enabled is True

        # Invalid pattern - empty pattern_id
        with pytest.raises(ValueError, match="pattern_id cannot be empty"):
            ErrorPattern(
                pattern_id="",
                error_type=ErrorType.SYNTAX,
                regex_pattern=r"SyntaxError",
                recovery_strategy=RecoveryStrategy.RETRY,
            )

        # Invalid pattern - invalid regex
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            ErrorPattern(
                pattern_id="bad_regex",
                error_type=ErrorType.SYNTAX,
                regex_pattern=r"[invalid(regex",
                recovery_strategy=RecoveryStrategy.RETRY,
            )

        # Invalid pattern - invalid priority
        with pytest.raises(ValueError, match="priority must be between"):
            ErrorPattern(
                pattern_id="bad_priority",
                error_type=ErrorType.SYNTAX,
                regex_pattern=r"Error",
                recovery_strategy=RecoveryStrategy.RETRY,
                priority=11,
            )

    def test_error_pattern_matching(self):
        """Test error pattern matching."""
        pattern = ErrorPattern(
            pattern_id="syntax_async",
            error_type=ErrorType.SYNTAX,
            regex_pattern=r"SyntaxError.*\bdef\s+execute",
            recovery_strategy=RecoveryStrategy.ERROR_CORRECTION,
        )

        # Should match
        assert pattern.matches("SyntaxError at line 10: def execute_effect")
        assert pattern.matches("SyntaxError: invalid syntax - def execute")

        # Should not match
        assert not pattern.matches("ImportError: module not found")
        assert not pattern.matches("def execute_something")

        # Disabled pattern should not match
        pattern.enabled = False
        assert not pattern.matches("SyntaxError: def execute_effect")

    def test_error_pattern_extract_groups(self):
        """Test error pattern group extraction."""
        pattern = ErrorPattern(
            pattern_id="import_error",
            error_type=ErrorType.IMPORT,
            regex_pattern=r"ImportError.*No module named '(?P<module>\w+)'",
            recovery_strategy=RecoveryStrategy.ALTERNATIVE_PATH,
        )

        groups = pattern.extract_groups("ImportError: No module named 'typing'")
        assert groups == {"module": "typing"}

        # No match
        groups = pattern.extract_groups("SyntaxError: invalid syntax")
        assert groups == {}

    def test_recovery_context_validation(self):
        """Test RecoveryContext validation."""
        # Valid context
        context = RecoveryContext(
            workflow_id="workflow-1",
            node_name="generator",
            step_count=3,
            state={"key": "value"},
        )
        assert context.workflow_id == "workflow-1"

        # Invalid context - empty workflow_id
        with pytest.raises(ValueError, match="workflow_id cannot be empty"):
            RecoveryContext(
                workflow_id="",
                node_name="generator",
                step_count=3,
                state={},
            )

        # Invalid context - negative step_count
        with pytest.raises(ValueError, match="step_count must be >= 0"):
            RecoveryContext(
                workflow_id="workflow-1",
                node_name="generator",
                step_count=-1,
                state={},
            )

    def test_recovery_context_error_extraction(self):
        """Test error message extraction from exception."""
        exception = ValueError("Test error message")

        context = RecoveryContext(
            workflow_id="workflow-1",
            node_name="generator",
            step_count=1,
            state={},
            exception=exception,
        )

        # Error message should be extracted
        assert context.error_message == "Test error message"

    def test_recovery_result_validation(self):
        """Test RecoveryResult validation."""
        # Valid result
        result = RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.RETRY,
            retry_count=2,
            error_fixed=True,
            fallback_used=False,
            duration_ms=150.5,
        )
        assert result.success is True

        # Invalid result - negative retry_count
        with pytest.raises(ValueError, match="retry_count must be >= 0"):
            RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.RETRY,
                retry_count=-1,
                error_fixed=False,
                fallback_used=False,
            )

    def test_recovery_statistics(self):
        """Test RecoveryStatistics aggregation."""
        stats = RecoveryStatistics()

        # Initial state
        assert stats.total_recoveries == 0
        assert stats.success_rate == 0.0

        # Update with successful result
        result1 = RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.RETRY,
            retry_count=2,
            error_fixed=True,
            fallback_used=False,
            duration_ms=200.0,
            pattern_matched="pattern_1",
        )
        stats.update_from_result(result1)

        assert stats.total_recoveries == 1
        assert stats.successful_recoveries == 1
        assert stats.success_rate == 1.0
        assert stats.average_duration_ms == 200.0
        assert stats.strategies_used[RecoveryStrategy.RETRY] == 1

        # Update with failed result
        result2 = RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategy.ERROR_CORRECTION,
            retry_count=1,
            error_fixed=False,
            fallback_used=False,
            duration_ms=100.0,
        )
        stats.update_from_result(result2)

        assert stats.total_recoveries == 2
        assert stats.successful_recoveries == 1
        assert stats.failed_recoveries == 1
        assert stats.success_rate == 0.5
        assert stats.average_duration_ms == 150.0  # (200 + 100) / 2


# Recovery Strategies Tests


class TestRetryStrategy:
    """Test RetryStrategy implementation."""

    @pytest.mark.asyncio
    async def test_retry_success_first_attempt(self):
        """Test retry succeeds on first attempt."""
        strategy = RetryStrategy(max_retries=3, base_delay=0.1)

        # Operation that succeeds immediately
        async def successful_operation(state: dict, **kwargs) -> dict:
            return {"status": "success"}

        context = RecoveryContext(
            workflow_id="test-1",
            node_name="generator",
            step_count=1,
            state={},
        )

        result = await strategy.execute(context, successful_operation)

        assert result.success is True
        assert result.retry_count == 0  # No retries needed
        assert result.error_fixed is True
        assert result.metadata["attempts"] == 1

    @pytest.mark.asyncio
    async def test_retry_success_after_retries(self):
        """Test retry succeeds after some failures."""
        strategy = RetryStrategy(max_retries=3, base_delay=0.05)

        # Operation that fails twice, then succeeds
        attempt_count = 0

        async def flaky_operation(state: dict, **kwargs) -> dict:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError(f"Attempt {attempt_count} failed")
            return {"status": "success"}

        context = RecoveryContext(
            workflow_id="test-1",
            node_name="generator",
            step_count=1,
            state={},
        )

        start_time = time.perf_counter()
        result = await strategy.execute(context, flaky_operation)
        duration = time.perf_counter() - start_time

        assert result.success is True
        assert result.retry_count == 2  # Two retries
        assert result.error_fixed is True
        assert attempt_count == 3

        # Should have delays (0.05s + 0.1s = 0.15s minimum)
        assert duration >= 0.15

    @pytest.mark.asyncio
    async def test_retry_all_attempts_fail(self):
        """Test retry when all attempts fail."""
        strategy = RetryStrategy(max_retries=2, base_delay=0.05)

        # Operation that always fails
        async def failing_operation(state: dict, **kwargs) -> dict:
            raise RuntimeError("Operation always fails")

        context = RecoveryContext(
            workflow_id="test-1",
            node_name="generator",
            step_count=1,
            state={},
        )

        result = await strategy.execute(context, failing_operation)

        assert result.success is False
        assert result.retry_count == 2  # All retries exhausted
        assert result.error_fixed is False
        assert "Operation always fails" in result.error_message
        assert result.metadata["attempts"] == 3  # Initial + 2 retries

    @pytest.mark.asyncio
    async def test_retry_exponential_backoff(self):
        """Test exponential backoff timing."""
        strategy = RetryStrategy(max_retries=3, base_delay=0.1)

        # Operation that always fails
        async def failing_operation(state: dict, **kwargs) -> dict:
            raise ValueError("Fail")

        context = RecoveryContext(
            workflow_id="test-1",
            node_name="generator",
            step_count=1,
            state={},
        )

        start_time = time.perf_counter()
        result = await strategy.execute(context, failing_operation)
        duration = time.perf_counter() - start_time

        # Expected delays: 0.1s (retry 1) + 0.2s (retry 2) + 0.4s (retry 3) = 0.7s
        assert duration >= 0.7
        assert result.retry_count == 3


class TestAlternativePathStrategy:
    """Test AlternativePathStrategy implementation."""

    @pytest.mark.asyncio
    async def test_alternative_path_first_alternative(self):
        """Test alternative path selects first alternative."""
        strategy = AlternativePathStrategy(
            alternatives={
                "model_template": ["template_v2", "template_simple"],
                "validator_template": ["validator_strict", "validator_basic"],
            }
        )

        context = RecoveryContext(
            workflow_id="test-1",
            node_name="generator",
            step_count=1,
            state={"template_name": "model_template"},
        )

        result = await strategy.execute(context, failed_path="model_template")

        assert result.success is True
        assert result.fallback_used is True
        assert result.error_fixed is False  # Workaround, not fix
        assert context.state["template_name"] == "template_v2"
        assert context.state["alternative_used"] is True
        assert result.metadata["alternative_used"] == "template_v2"

    @pytest.mark.asyncio
    async def test_alternative_path_no_alternatives(self):
        """Test alternative path when no alternatives exist."""
        strategy = AlternativePathStrategy(alternatives={})

        context = RecoveryContext(
            workflow_id="test-1",
            node_name="generator",
            step_count=1,
            state={},
        )

        result = await strategy.execute(context, failed_path="unknown_template")

        assert result.success is False
        assert "No alternatives available" in result.error_message


class TestGracefulDegradationStrategy:
    """Test GracefulDegradationStrategy implementation."""

    @pytest.mark.asyncio
    async def test_graceful_degradation_next_level(self):
        """Test degradation to next lower level."""
        strategy = GracefulDegradationStrategy(
            degradation_levels={
                "validation": ["full", "basic", "minimal"],
                "quality": [0.9, 0.8, 0.7],
            }
        )

        context = RecoveryContext(
            workflow_id="test-1",
            node_name="validator",
            step_count=1,
            state={},
        )

        # First degradation
        result = await strategy.execute(
            context, component="validation", current_level="full"
        )

        assert result.success is True
        assert result.fallback_used is True
        assert context.state["validation_level"] == "basic"
        assert context.state["degraded"] is True

    @pytest.mark.asyncio
    async def test_graceful_degradation_minimum_level(self):
        """Test degradation when already at minimum."""
        strategy = GracefulDegradationStrategy(
            degradation_levels={"validation": ["full", "basic", "minimal"]}
        )

        context = RecoveryContext(
            workflow_id="test-1",
            node_name="validator",
            step_count=1,
            state={},
        )

        # Already at minimum
        result = await strategy.execute(
            context, component="validation", current_level="minimal"
        )

        assert result.success is False
        assert "Already at minimum" in result.error_message


class TestErrorCorrectionStrategy:
    """Test ErrorCorrectionStrategy implementation."""

    @pytest.mark.asyncio
    async def test_error_correction_add_async(self):
        """Test error correction for missing async keyword."""
        pattern = ErrorPattern(
            pattern_id="missing_async",
            error_type=ErrorType.SYNTAX,
            regex_pattern=r"def execute",
            recovery_strategy=RecoveryStrategy.ERROR_CORRECTION,
            metadata={"fix": "add_async_keyword"},
        )

        strategy = ErrorCorrectionStrategy(error_patterns=[pattern])

        context = RecoveryContext(
            workflow_id="test-1",
            node_name="generator",
            step_count=1,
            state={},
        )

        code = "def execute_effect():\n    pass"

        result = await strategy.execute(context, code, pattern)

        assert result.success is True
        assert result.error_fixed is True
        assert "async def execute_effect()" in context.state["corrected_code"]
        assert context.state["correction_applied"] == "add_async_keyword"

    @pytest.mark.asyncio
    async def test_error_correction_no_fix_metadata(self):
        """Test error correction with no fix metadata."""
        pattern = ErrorPattern(
            pattern_id="no_fix",
            error_type=ErrorType.SYNTAX,
            regex_pattern=r"Error",
            recovery_strategy=RecoveryStrategy.ERROR_CORRECTION,
            metadata={},  # No fix metadata
        )

        strategy = ErrorCorrectionStrategy(error_patterns=[pattern])

        context = RecoveryContext(
            workflow_id="test-1",
            node_name="generator",
            step_count=1,
            state={},
        )

        result = await strategy.execute(context, "some code", pattern)

        assert result.success is False
        assert "No fix metadata" in result.error_message


class TestEscalationStrategy:
    """Test EscalationStrategy implementation."""

    @pytest.mark.asyncio
    async def test_escalation_creates_record(self):
        """Test escalation creates intervention record."""
        strategy = EscalationStrategy()

        context = RecoveryContext(
            workflow_id="test-1",
            node_name="generator",
            step_count=5,
            state={"contract": {"name": "Test"}},
            correlation_id="corr-1",
        )

        result = await strategy.execute(
            context,
            error_summary="AI quorum failed",
            suggested_action="Review contract structure",
        )

        assert result.success is True
        assert result.error_fixed is False
        assert context.state["requires_human_intervention"] is True
        assert "escalation" in context.state

        escalation_data = context.state["escalation"]
        assert escalation_data["workflow_id"] == "test-1"
        assert escalation_data["error_summary"] == "AI quorum failed"
        assert escalation_data["suggested_action"] == "Review contract structure"

    @pytest.mark.asyncio
    async def test_escalation_with_notification(self):
        """Test escalation with notification callback."""
        notified = False
        notification_data = None

        async def notification_callback(data: dict):
            nonlocal notified, notification_data
            notified = True
            notification_data = data

        strategy = EscalationStrategy(notification_callback=notification_callback)

        context = RecoveryContext(
            workflow_id="test-1",
            node_name="generator",
            step_count=1,
            state={},
        )

        result = await strategy.execute(context, error_summary="Test error")

        assert result.success is True
        assert notified is True
        assert notification_data["workflow_id"] == "test-1"


# Error Recovery Orchestrator Tests


class TestErrorRecoveryOrchestrator:
    """Test ErrorRecoveryOrchestrator main functionality."""

    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, error_orchestrator):
        """Test orchestrator initializes with default patterns."""
        assert len(error_orchestrator.error_patterns) > 0
        assert len(error_orchestrator.strategies) == 5

        # Check default patterns exist
        pattern_ids = list(error_orchestrator.error_patterns.keys())
        assert "missing_async_keyword" in pattern_ids
        assert "import_not_found" in pattern_ids

    @pytest.mark.asyncio
    async def test_error_analysis_pattern_matching(self, error_orchestrator):
        """Test error analysis and pattern matching."""
        context = RecoveryContext(
            workflow_id="test-1",
            node_name="generator",
            step_count=1,
            state={},
            exception=SyntaxError("def execute_effect missing async"),
            error_message="SyntaxError: def execute_effect",
        )

        pattern, error_type = error_orchestrator._analyze_error(context)

        assert pattern is not None
        assert pattern.pattern_id == "missing_async_keyword"
        assert error_type == ErrorType.SYNTAX

    @pytest.mark.asyncio
    async def test_error_classification(self, error_orchestrator):
        """Test error type classification."""
        # Syntax error
        error_type = error_orchestrator._classify_error_type(SyntaxError("invalid"))
        assert error_type == ErrorType.SYNTAX

        # Import error
        error_type = error_orchestrator._classify_error_type(
            ImportError("module not found")
        )
        assert error_type == ErrorType.IMPORT

        # Unknown error
        error_type = error_orchestrator._classify_error_type(Exception("unknown error"))
        assert error_type == ErrorType.RUNTIME

    @pytest.mark.asyncio
    async def test_strategy_selection(self, error_orchestrator):
        """Test recovery strategy selection."""
        context = RecoveryContext(
            workflow_id="test-1",
            node_name="generator",
            step_count=1,
            state={},
        )

        # Pattern-based selection
        pattern = ErrorPattern(
            pattern_id="test",
            error_type=ErrorType.SYNTAX,
            regex_pattern=r"Error",
            recovery_strategy=RecoveryStrategy.ERROR_CORRECTION,
        )
        strategy = error_orchestrator._select_recovery_strategy(
            pattern, ErrorType.SYNTAX, context
        )
        assert strategy == RecoveryStrategy.ERROR_CORRECTION

        # Type-based selection (no pattern)
        strategy = error_orchestrator._select_recovery_strategy(
            None, ErrorType.SYNTAX, context
        )
        assert strategy == RecoveryStrategy.ERROR_CORRECTION

        strategy = error_orchestrator._select_recovery_strategy(
            None, ErrorType.IMPORT, context
        )
        assert strategy == RecoveryStrategy.ALTERNATIVE_PATH

    @pytest.mark.asyncio
    async def test_handle_error_with_retry_success(self, error_orchestrator):
        """Test handle_error with successful retry."""
        attempt_count = 0

        async def flaky_operation(state: dict, **kwargs) -> dict:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise ValueError("Temporary failure")
            return {"status": "success"}

        context = RecoveryContext(
            workflow_id="test-1",
            node_name="generator",
            step_count=1,
            state={},
            exception=asyncio.TimeoutError("Timeout"),
            error_message="TimeoutError: operation timeout",
        )

        result = await error_orchestrator.handle_error(
            context, operation=flaky_operation
        )

        assert result.success is True
        assert result.strategy_used == RecoveryStrategy.RETRY
        assert result.error_fixed is True
        assert attempt_count == 2

    @pytest.mark.asyncio
    async def test_handle_error_performance(self, error_orchestrator):
        """Test handle_error meets performance targets (<500ms)."""

        async def fast_operation(state: dict, **kwargs) -> dict:
            return {"status": "success"}

        context = RecoveryContext(
            workflow_id="test-1",
            node_name="generator",
            step_count=1,
            state={},
            exception=RuntimeError("Test error"),
            error_message="RuntimeError: Test error",
        )

        start_time = time.perf_counter()
        result = await error_orchestrator.handle_error(
            context, operation=fast_operation
        )
        duration_ms = (time.perf_counter() - start_time) * 1000

        assert result.success is True
        # Should be well under 500ms for simple operation
        assert duration_ms < 500

    @pytest.mark.asyncio
    async def test_add_remove_error_patterns(self, error_orchestrator):
        """Test adding and removing custom error patterns."""
        initial_count = len(error_orchestrator.error_patterns)

        # Add custom pattern
        custom_pattern = ErrorPattern(
            pattern_id="custom_pattern",
            error_type=ErrorType.VALIDATION,
            regex_pattern=r"CustomError",
            recovery_strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
        )
        error_orchestrator.add_error_pattern(custom_pattern)

        assert len(error_orchestrator.error_patterns) == initial_count + 1
        assert "custom_pattern" in error_orchestrator.error_patterns

        # Remove pattern
        error_orchestrator.remove_error_pattern("custom_pattern")
        assert len(error_orchestrator.error_patterns) == initial_count
        assert "custom_pattern" not in error_orchestrator.error_patterns

    @pytest.mark.asyncio
    async def test_statistics_tracking(self, error_orchestrator):
        """Test recovery statistics tracking."""

        async def successful_operation(state: dict, **kwargs) -> dict:
            return {"status": "success"}

        context = RecoveryContext(
            workflow_id="test-1",
            node_name="generator",
            step_count=1,
            state={},
            exception=RuntimeError("Test"),
            error_message="RuntimeError: Test",
        )

        # Initial statistics
        stats = error_orchestrator.get_statistics()
        assert stats.total_recoveries == 0

        # Execute recovery
        await error_orchestrator.handle_error(context, operation=successful_operation)

        # Check updated statistics
        stats = error_orchestrator.get_statistics()
        assert stats.total_recoveries == 1
        assert stats.successful_recoveries == 1
        assert stats.success_rate == 1.0

    @pytest.mark.asyncio
    async def test_metrics_collection_integration(
        self, metrics_collector, signal_coordinator
    ):
        """Test metrics collection during error recovery."""
        orchestrator = ErrorRecoveryOrchestrator(
            metrics_collector=metrics_collector,
            signal_coordinator=signal_coordinator,
            max_retries=2,
            base_delay=0.05,
        )

        await metrics_collector.start()

        try:

            async def successful_operation(state: dict, **kwargs) -> dict:
                return {"status": "success"}

            context = RecoveryContext(
                workflow_id="test-metrics",
                node_name="generator",
                step_count=1,
                state={},
                exception=RuntimeError("Test"),
                correlation_id="test-corr-1",
            )

            result = await orchestrator.handle_error(
                context, operation=successful_operation
            )

            assert result.success is True

            # Verify metrics were collected
            stats = await metrics_collector.get_stats()
            assert stats["buffer_size"] > 0  # Some metrics should be in buffer

        finally:
            await metrics_collector.stop()

    @pytest.mark.asyncio
    async def test_signal_coordination_integration(
        self, metrics_collector, signal_coordinator
    ):
        """Test signal coordination during error recovery."""
        orchestrator = ErrorRecoveryOrchestrator(
            metrics_collector=metrics_collector,
            signal_coordinator=signal_coordinator,
            max_retries=2,
            base_delay=0.05,
        )

        async def successful_operation(state: dict, **kwargs) -> dict:
            return {"status": "success"}

        context = RecoveryContext(
            workflow_id="test-signals",
            node_name="generator",
            step_count=1,
            state={},
            exception=RuntimeError("Test"),
        )

        result = await orchestrator.handle_error(
            context, operation=successful_operation
        )

        assert result.success is True

        # Verify signal was sent
        signals = signal_coordinator.get_signal_history("test-signals")
        assert len(signals) == 1
        assert signals[0].signal_type.value == "error_recovery_completed"

    def test_reset_statistics(self, error_orchestrator):
        """Test statistics reset."""
        # Manually update statistics
        error_orchestrator.statistics.total_recoveries = 10
        error_orchestrator.statistics.successful_recoveries = 8

        # Reset
        error_orchestrator.reset_statistics()

        stats = error_orchestrator.get_statistics()
        assert stats.total_recoveries == 0
        assert stats.successful_recoveries == 0


# Performance Tests


class TestErrorRecoveryPerformance:
    """Test performance targets for error recovery."""

    @pytest.mark.asyncio
    async def test_error_analysis_performance(self, error_orchestrator):
        """Test error analysis meets <100ms target."""
        context = RecoveryContext(
            workflow_id="perf-test",
            node_name="generator",
            step_count=1,
            state={},
            exception=SyntaxError("def execute missing async"),
            error_message="SyntaxError: def execute",
        )

        start_time = time.perf_counter()
        pattern, error_type = error_orchestrator._analyze_error(context)
        duration_ms = (time.perf_counter() - start_time) * 1000

        assert duration_ms < 100  # <100ms target
        assert pattern is not None

    @pytest.mark.asyncio
    async def test_strategy_selection_performance(self, error_orchestrator):
        """Test strategy selection meets <50ms target."""
        context = RecoveryContext(
            workflow_id="perf-test",
            node_name="generator",
            step_count=1,
            state={},
        )

        pattern = ErrorPattern(
            pattern_id="test",
            error_type=ErrorType.SYNTAX,
            regex_pattern=r"Error",
            recovery_strategy=RecoveryStrategy.RETRY,
        )

        start_time = time.perf_counter()
        strategy = error_orchestrator._select_recovery_strategy(
            pattern, ErrorType.SYNTAX, context
        )
        duration_ms = (time.perf_counter() - start_time) * 1000

        assert duration_ms < 50  # <50ms target
        assert strategy == RecoveryStrategy.RETRY

    @pytest.mark.asyncio
    async def test_total_recovery_overhead(self, error_orchestrator):
        """Test total recovery overhead meets <500ms target."""

        async def fast_operation(state: dict, **kwargs) -> dict:
            return {"status": "success"}

        context = RecoveryContext(
            workflow_id="perf-test",
            node_name="generator",
            step_count=1,
            state={},
            exception=RuntimeError("Test error"),
            error_message="RuntimeError: Test error",
        )

        start_time = time.perf_counter()
        result = await error_orchestrator.handle_error(
            context, operation=fast_operation
        )
        duration_ms = (time.perf_counter() - start_time) * 1000

        assert result.success is True
        assert duration_ms < 500  # <500ms total overhead target

    @pytest.mark.asyncio
    async def test_success_rate_target(self, error_orchestrator):
        """Test 80%+ success rate for recoverable errors."""

        async def recoverable_operation(state: dict, **kwargs) -> dict:
            # Succeeds on second attempt
            if state.get("attempt_count", 0) == 0:
                state["attempt_count"] = 1
                raise ValueError("First attempt fails")
            return {"status": "success"}

        # Run 10 recovery attempts
        success_count = 0
        for i in range(10):
            context = RecoveryContext(
                workflow_id=f"success-rate-test-{i}",
                node_name="generator",
                step_count=1,
                state={},
                exception=RuntimeError(f"Test error {i}"),
            )

            result = await error_orchestrator.handle_error(
                context, operation=recoverable_operation
            )

            if result.success:
                success_count += 1

        success_rate = success_count / 10
        assert success_rate >= 0.8  # 80%+ success rate target


# Integration Tests


class TestErrorRecoveryIntegration:
    """Test error recovery integration with other components."""

    @pytest.mark.asyncio
    async def test_end_to_end_retry_workflow(self, error_orchestrator):
        """Test complete retry workflow from error to recovery."""
        attempt_count = 0

        async def operation_with_transient_error(state: dict, **kwargs) -> dict:
            nonlocal attempt_count
            attempt_count += 1

            # Fail first 2 attempts
            if attempt_count <= 2:
                raise ConnectionError(f"Connection failed (attempt {attempt_count})")

            return {"status": "success", "data": "Generated code"}

        context = RecoveryContext(
            workflow_id="e2e-retry-test",
            node_name="code_generator",
            step_count=5,
            state={"contract": {"name": "TestContract"}},
            exception=ConnectionError("Connection failed"),
            error_message="ConnectionError: Connection failed",
            correlation_id="e2e-test-1",
        )

        result = await error_orchestrator.handle_error(
            context, operation=operation_with_transient_error
        )

        assert result.success is True
        assert result.strategy_used == RecoveryStrategy.RETRY
        assert result.retry_count == 2
        assert result.error_fixed is True
        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_end_to_end_correction_workflow(self, error_orchestrator):
        """Test complete error correction workflow."""
        code_with_error = """
def execute_effect():
    return {"status": "success"}
"""

        context = RecoveryContext(
            workflow_id="e2e-correction-test",
            node_name="validator_generator",
            step_count=3,
            state={"code": code_with_error},
            exception=SyntaxError("def execute missing async"),
            error_message="SyntaxError: def execute_effect",
            correlation_id="e2e-test-2",
        )

        result = await error_orchestrator.handle_error(context, code=code_with_error)

        # Should attempt error correction
        assert result.strategy_used == RecoveryStrategy.ERROR_CORRECTION
        if result.success:
            assert "async def execute_effect" in context.state["corrected_code"]
