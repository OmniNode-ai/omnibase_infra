"""
Error Recovery Orchestrator for code generation workflows.

This module provides centralized error recovery orchestration with multiple
recovery strategies, pattern matching, and performance tracking.

Implements Pattern 7 from OMNIAGENT_AGENT_FUNCTIONALITY_RESEARCH.md.

Performance targets:
- Error analysis: <100ms
- Recovery decision: <50ms
- Total recovery overhead: <500ms
- Success rate: 80%+ for recoverable errors

Example:
    ```python
    from omninode_bridge.agents.metrics import MetricsCollector
    from omninode_bridge.agents.coordination import SignalCoordinator
    from omninode_bridge.agents.workflows import (
        ErrorRecoveryOrchestrator,
        RecoveryContext,
        ErrorPattern,
        ErrorType,
        RecoveryStrategy
    )

    # Create orchestrator
    metrics = MetricsCollector()
    signals = SignalCoordinator(state, metrics)
    orchestrator = ErrorRecoveryOrchestrator(
        metrics_collector=metrics,
        signal_coordinator=signals
    )

    # Handle error
    try:
        result = await generate_code(contract)
    except Exception as e:
        context = RecoveryContext(
            workflow_id="codegen-1",
            node_name="model_generator",
            step_count=5,
            state={"contract": contract},
            exception=e
        )

        recovery_result = await orchestrator.handle_error(context)

        if recovery_result.success:
            print(f"Recovered using {recovery_result.strategy_used}")
        else:
            print(f"Recovery failed: {recovery_result.error_message}")
    ```
"""

import logging
import time
from collections.abc import Callable
from typing import Any, Optional

from ..coordination.signals import SignalCoordinator
from ..metrics.collector import MetricsCollector
from .recovery_models import (
    ErrorPattern,
    ErrorType,
    RecoveryContext,
    RecoveryResult,
    RecoveryStatistics,
    RecoveryStrategy,
)
from .recovery_strategies import (
    AlternativePathStrategy,
    ErrorCorrectionStrategy,
    EscalationStrategy,
    GracefulDegradationStrategy,
    RetryStrategy,
)

logger = logging.getLogger(__name__)


class ErrorRecoveryOrchestrator:
    """
    Centralized error recovery orchestration.

    Coordinates multiple recovery strategies with pattern matching,
    metrics collection, and signal coordination.

    Features:
    - 5 recovery strategies (retry, alternative, degradation, correction, escalation)
    - Pattern-based error matching
    - Automatic strategy selection
    - Performance metrics tracking
    - Recovery statistics aggregation

    Performance:
    - Error analysis: <100ms
    - Recovery decision: <50ms
    - Total overhead: <500ms
    - Success rate target: 80%+ for recoverable errors

    Example:
        ```python
        orchestrator = ErrorRecoveryOrchestrator(
            metrics_collector=metrics,
            signal_coordinator=signals
        )

        # Add custom error patterns
        orchestrator.add_error_pattern(ErrorPattern(
            pattern_id="custom_syntax_error",
            error_type=ErrorType.SYNTAX,
            regex_pattern=r"SyntaxError.*missing.*async",
            recovery_strategy=RecoveryStrategy.ERROR_CORRECTION,
            metadata={"fix": "add_async_keyword"}
        ))

        # Handle error with context
        result = await orchestrator.handle_error(context)
        ```
    """

    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        signal_coordinator: Optional[SignalCoordinator] = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> None:
        """
        Initialize error recovery orchestrator.

        Args:
            metrics_collector: Optional metrics collector for tracking
            signal_coordinator: Optional signal coordinator for events
            max_retries: Maximum retry attempts (default: 3)
            base_delay: Base delay for exponential backoff (default: 1.0s)
        """
        self.metrics = metrics_collector
        self.signals = signal_coordinator
        self.max_retries = max_retries
        self.base_delay = base_delay

        # Error patterns registry
        self.error_patterns: dict[str, ErrorPattern] = {}

        # Recovery strategies
        self.strategies: dict[RecoveryStrategy, Any] = {}

        # Statistics
        self.statistics = RecoveryStatistics()

        # Initialize default patterns and strategies
        self._initialize_default_patterns()
        self._initialize_strategies()

        logger.info(
            f"ErrorRecoveryOrchestrator initialized with {len(self.error_patterns)} patterns "
            f"and {len(self.strategies)} strategies"
        )

    def add_error_pattern(self, pattern: ErrorPattern) -> None:
        """
        Add custom error pattern to registry.

        Args:
            pattern: ErrorPattern to add
        """
        self.error_patterns[pattern.pattern_id] = pattern
        logger.debug(f"Added error pattern: {pattern.pattern_id}")

    def remove_error_pattern(self, pattern_id: str) -> None:
        """
        Remove error pattern from registry.

        Args:
            pattern_id: Pattern ID to remove
        """
        if pattern_id in self.error_patterns:
            del self.error_patterns[pattern_id]
            logger.debug(f"Removed error pattern: {pattern_id}")

    async def handle_error(
        self,
        context: RecoveryContext,
        operation: Optional[Callable] = None,
        **kwargs: Any,
    ) -> RecoveryResult:
        """
        Handle error with recovery strategies.

        Main entry point for error recovery. Analyzes error, selects
        appropriate strategy, and executes recovery.

        Performance Target: <500ms total

        Args:
            context: RecoveryContext with error details
            operation: Optional operation to retry (for retry strategy)
            **kwargs: Additional arguments for specific strategies

        Returns:
            RecoveryResult with recovery outcome

        Example:
            ```python
            # Create context from exception
            context = RecoveryContext(
                workflow_id="codegen-1",
                node_name="validator_gen",
                step_count=3,
                state={"contract": {...}},
                exception=SyntaxError("missing async keyword")
            )

            # Handle error
            result = await orchestrator.handle_error(context, operation=generate_validator)

            if result.success:
                print(f"Recovered! Strategy: {result.strategy_used}")
                print(f"Duration: {result.duration_ms:.2f}ms")
            ```
        """
        start_time = time.perf_counter()

        try:
            # 1. Analyze error and match pattern (<100ms target)
            analysis_start = time.perf_counter()
            error_pattern, error_type = self._analyze_error(context)
            analysis_duration_ms = (time.perf_counter() - analysis_start) * 1000

            logger.info(
                f"[ErrorRecovery] Analyzed error for workflow '{context.workflow_id}': "
                f"type={error_type.value}, pattern={error_pattern.pattern_id if error_pattern else 'none'}, "
                f"duration={analysis_duration_ms:.2f}ms"
            )

            # Record analysis metrics
            if self.metrics:
                await self.metrics.record_timing(
                    metric_name="error_recovery_analysis_time_ms",
                    duration_ms=analysis_duration_ms,
                    tags={
                        "error_type": error_type.value,
                        "pattern_matched": str(error_pattern is not None),
                    },
                    correlation_id=context.correlation_id,
                )

            # 2. Select recovery strategy (<50ms target)
            decision_start = time.perf_counter()
            strategy = self._select_recovery_strategy(
                error_pattern, error_type, context
            )
            decision_duration_ms = (time.perf_counter() - decision_start) * 1000

            logger.info(
                f"[ErrorRecovery] Selected strategy '{strategy.value}' "
                f"for workflow '{context.workflow_id}' (decision: {decision_duration_ms:.2f}ms)"
            )

            # Record decision metrics
            if self.metrics:
                await self.metrics.record_timing(
                    metric_name="error_recovery_decision_time_ms",
                    duration_ms=decision_duration_ms,
                    tags={
                        "strategy": strategy.value,
                        "error_type": error_type.value,
                    },
                    correlation_id=context.correlation_id,
                )

            # 3. Execute recovery strategy
            execution_start = time.perf_counter()
            result = await self._execute_recovery(
                strategy=strategy,
                context=context,
                error_pattern=error_pattern,
                operation=operation,
                **kwargs,
            )
            execution_duration_ms = (time.perf_counter() - execution_start) * 1000

            # 4. Update result with pattern info
            if error_pattern:
                result.pattern_matched = error_pattern.pattern_id

            # 5. Update statistics
            self.statistics.update_from_result(result)
            self.statistics.error_types_seen[error_type] = (
                self.statistics.error_types_seen.get(error_type, 0) + 1
            )

            # 6. Record overall metrics
            total_duration_ms = (time.perf_counter() - start_time) * 1000
            result.duration_ms = total_duration_ms  # Update with total

            if self.metrics:
                await self.metrics.record_timing(
                    metric_name="error_recovery_total_time_ms",
                    duration_ms=total_duration_ms,
                    tags={
                        "strategy": result.strategy_used.value,
                        "success": str(result.success),
                        "error_type": error_type.value,
                    },
                    correlation_id=context.correlation_id,
                )

                await self.metrics.record_counter(
                    metric_name="error_recovery_attempts",
                    count=1,
                    tags={
                        "strategy": result.strategy_used.value,
                        "success": str(result.success),
                    },
                    correlation_id=context.correlation_id,
                )

            # 7. Send recovery signal
            if self.signals:
                await self.signals.signal_coordination_event(
                    coordination_id=context.workflow_id,
                    event_type="error_recovery_completed",
                    event_data={
                        "node_name": context.node_name,
                        "strategy_used": result.strategy_used.value,
                        "success": result.success,
                        "error_fixed": result.error_fixed,
                        "fallback_used": result.fallback_used,
                        "duration_ms": total_duration_ms,
                        "retry_count": result.retry_count,
                    },
                    sender_agent_id="error_recovery_orchestrator",
                )

            logger.info(
                f"[ErrorRecovery] Recovery completed for workflow '{context.workflow_id}': "
                f"success={result.success}, strategy={result.strategy_used.value}, "
                f"duration={total_duration_ms:.2f}ms"
            )

            return result

        except Exception as e:
            logger.error(
                f"[ErrorRecovery] Recovery orchestration failed for workflow '{context.workflow_id}': {e}",
                exc_info=True,
            )

            total_duration_ms = (time.perf_counter() - start_time) * 1000

            # Record failure metrics
            if self.metrics:
                await self.metrics.record_counter(
                    metric_name="error_recovery_orchestration_failures",
                    count=1,
                    correlation_id=context.correlation_id,
                )

            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.RETRY,  # Default
                retry_count=0,
                error_fixed=False,
                fallback_used=False,
                error_message=f"Recovery orchestration failed: {e!s}",
                duration_ms=total_duration_ms,
            )

    def get_statistics(self) -> RecoveryStatistics:
        """
        Get recovery statistics.

        Returns:
            RecoveryStatistics with aggregated metrics
        """
        return self.statistics

    def reset_statistics(self) -> None:
        """Reset recovery statistics."""
        self.statistics = RecoveryStatistics()
        logger.info("Recovery statistics reset")

    # Private methods

    def _analyze_error(
        self, context: RecoveryContext
    ) -> tuple[Optional[ErrorPattern], ErrorType]:
        """
        Analyze error and match to known patterns.

        Performance Target: <100ms

        Args:
            context: RecoveryContext with error details

        Returns:
            Tuple of (matched_pattern, error_type)
        """
        error_message = context.error_message

        # Try to match error patterns (sorted by priority)
        sorted_patterns = sorted(
            self.error_patterns.values(),
            key=lambda p: p.priority,
            reverse=True,  # Higher priority first
        )

        for pattern in sorted_patterns:
            if pattern.matches(error_message):
                logger.debug(
                    f"Matched error pattern: {pattern.pattern_id} "
                    f"(priority: {pattern.priority})"
                )
                return pattern, pattern.error_type

        # No pattern matched - classify by exception type
        error_type = self._classify_error_type(context.exception)

        logger.debug(f"No pattern matched, classified as: {error_type.value}")

        return None, error_type

    def _classify_error_type(self, exception: Optional[Exception]) -> ErrorType:
        """
        Classify error type from exception.

        Args:
            exception: Exception to classify

        Returns:
            ErrorType classification
        """
        if exception is None:
            return ErrorType.UNKNOWN

        exception_type = type(exception).__name__

        # Map exception types to error types
        if "SyntaxError" in exception_type:
            return ErrorType.SYNTAX
        elif "ImportError" in exception_type or "ModuleNotFoundError" in exception_type:
            return ErrorType.IMPORT
        elif "ValidationError" in exception_type or "TypeError" in exception_type:
            return ErrorType.VALIDATION
        elif (
            "TimeoutError" in exception_type or "asyncio.TimeoutError" in exception_type
        ):
            return ErrorType.TIMEOUT
        elif "TemplateError" in exception_type or "Jinja2Error" in exception_type:
            return ErrorType.TEMPLATE
        else:
            return ErrorType.RUNTIME

    def _select_recovery_strategy(
        self,
        error_pattern: Optional[ErrorPattern],
        error_type: ErrorType,
        context: RecoveryContext,
    ) -> RecoveryStrategy:
        """
        Select appropriate recovery strategy.

        Performance Target: <50ms

        Args:
            error_pattern: Matched error pattern (if any)
            error_type: Classified error type
            context: RecoveryContext

        Returns:
            Selected RecoveryStrategy
        """
        # 1. If pattern matched, use its recommended strategy
        if error_pattern:
            return error_pattern.recovery_strategy

        # 2. Otherwise, select based on error type and context
        # For syntax errors, try correction first
        if error_type == ErrorType.SYNTAX:
            return RecoveryStrategy.ERROR_CORRECTION

        # For import errors, try alternative path
        if error_type == ErrorType.IMPORT:
            return RecoveryStrategy.ALTERNATIVE_PATH

        # For validation errors, try degradation
        if error_type == ErrorType.VALIDATION:
            return RecoveryStrategy.GRACEFUL_DEGRADATION

        # For timeout errors, retry with backoff
        if error_type == ErrorType.TIMEOUT:
            return RecoveryStrategy.RETRY

        # For template errors, try alternative path
        if error_type == ErrorType.TEMPLATE:
            return RecoveryStrategy.ALTERNATIVE_PATH

        # Default: retry
        return RecoveryStrategy.RETRY

    async def _execute_recovery(
        self,
        strategy: RecoveryStrategy,
        context: RecoveryContext,
        error_pattern: Optional[ErrorPattern],
        operation: Optional[Callable],
        **kwargs: Any,
    ) -> RecoveryResult:
        """
        Execute recovery strategy.

        Args:
            strategy: Recovery strategy to execute
            context: RecoveryContext
            error_pattern: Matched error pattern (if any)
            operation: Operation to retry (if applicable)
            **kwargs: Additional strategy-specific arguments

        Returns:
            RecoveryResult from strategy execution
        """
        strategy_impl = self.strategies.get(strategy)

        if not strategy_impl:
            logger.error(f"No implementation for strategy: {strategy.value}")
            return RecoveryResult(
                success=False,
                strategy_used=strategy,
                retry_count=0,
                error_fixed=False,
                fallback_used=False,
                error_message=f"Strategy '{strategy.value}' not implemented",
                duration_ms=0.0,
            )

        try:
            # Execute strategy with appropriate arguments
            if strategy == RecoveryStrategy.RETRY:
                if not operation:
                    return RecoveryResult(
                        success=False,
                        strategy_used=strategy,
                        retry_count=0,
                        error_fixed=False,
                        fallback_used=False,
                        error_message="Retry strategy requires 'operation' argument",
                        duration_ms=0.0,
                    )
                return await strategy_impl.execute(context, operation, **kwargs)

            elif strategy == RecoveryStrategy.ERROR_CORRECTION:
                if not error_pattern:
                    # No pattern to correct - fall back to retry
                    logger.warning(
                        "No error pattern for correction, falling back to retry"
                    )
                    retry_strategy = self.strategies.get(RecoveryStrategy.RETRY)
                    if retry_strategy and operation:
                        return await retry_strategy.execute(
                            context, operation, **kwargs
                        )
                    return RecoveryResult(
                        success=False,
                        strategy_used=strategy,
                        retry_count=0,
                        error_fixed=False,
                        fallback_used=False,
                        error_message="No pattern for correction and no operation to retry",
                        duration_ms=0.0,
                    )

                code = kwargs.get("code", context.state.get("code", ""))
                return await strategy_impl.execute(
                    context, code, error_pattern, **kwargs
                )

            elif strategy == RecoveryStrategy.ALTERNATIVE_PATH:
                failed_path = kwargs.get(
                    "failed_path", context.state.get("template_name", "default")
                )
                return await strategy_impl.execute(context, failed_path, **kwargs)

            elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                component = kwargs.get("component", "validation")
                current_level = kwargs.get(
                    "current_level", context.state.get(f"{component}_level")
                )
                return await strategy_impl.execute(
                    context, component, current_level, **kwargs
                )

            elif strategy == RecoveryStrategy.ESCALATION:
                error_summary = kwargs.get("error_summary", context.error_message)
                suggested_action = kwargs.get(
                    "suggested_action", "Manual review required"
                )
                return await strategy_impl.execute(
                    context, error_summary, suggested_action, **kwargs
                )

            else:
                return RecoveryResult(
                    success=False,
                    strategy_used=strategy,
                    retry_count=0,
                    error_fixed=False,
                    fallback_used=False,
                    error_message=f"Unknown strategy: {strategy.value}",
                    duration_ms=0.0,
                )

        except Exception as e:
            logger.error(f"Strategy execution failed: {e}", exc_info=True)
            return RecoveryResult(
                success=False,
                strategy_used=strategy,
                retry_count=0,
                error_fixed=False,
                fallback_used=False,
                error_message=f"Strategy execution error: {e!s}",
                duration_ms=0.0,
            )

    def _initialize_default_patterns(self) -> None:
        """Initialize default error patterns for common code generation errors."""
        default_patterns = [
            # Syntax errors
            ErrorPattern(
                pattern_id="missing_async_keyword",
                error_type=ErrorType.SYNTAX,
                regex_pattern=r"SyntaxError.*\bdef\s+execute",
                recovery_strategy=RecoveryStrategy.ERROR_CORRECTION,
                max_retries=1,
                metadata={"fix": "add_async_keyword"},
                priority=9,
            ),
            # Import errors
            ErrorPattern(
                pattern_id="import_not_found",
                error_type=ErrorType.IMPORT,
                regex_pattern=r"(ImportError|ModuleNotFoundError).*No module named",
                recovery_strategy=RecoveryStrategy.ALTERNATIVE_PATH,
                max_retries=2,
                metadata={
                    "alternative_imports": ["typing", "collections.abc"],
                },
                priority=8,
            ),
            # Validation errors
            ErrorPattern(
                pattern_id="validation_failed",
                error_type=ErrorType.VALIDATION,
                regex_pattern=r"ValidationError.*failed.*validation",
                recovery_strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
                max_retries=2,
                priority=7,
            ),
            # Template errors
            ErrorPattern(
                pattern_id="template_rendering_error",
                error_type=ErrorType.TEMPLATE,
                regex_pattern=r"(TemplateError|UndefinedError|TemplateSyntaxError)",
                recovery_strategy=RecoveryStrategy.ALTERNATIVE_PATH,
                max_retries=2,
                priority=8,
            ),
            # Quorum errors
            ErrorPattern(
                pattern_id="quorum_consensus_failed",
                error_type=ErrorType.QUORUM,
                regex_pattern=r"(QuorumError|ConsensusError).*failed.*consensus",
                recovery_strategy=RecoveryStrategy.RETRY,
                max_retries=3,
                priority=7,
            ),
            # Timeout errors
            ErrorPattern(
                pattern_id="operation_timeout",
                error_type=ErrorType.TIMEOUT,
                regex_pattern=r"(TimeoutError|asyncio\.TimeoutError)",
                recovery_strategy=RecoveryStrategy.RETRY,
                max_retries=2,
                priority=6,
            ),
        ]

        for pattern in default_patterns:
            self.error_patterns[pattern.pattern_id] = pattern

        logger.debug(f"Initialized {len(default_patterns)} default error patterns")

    def _initialize_strategies(self) -> None:
        """Initialize recovery strategy implementations."""
        self.strategies = {
            RecoveryStrategy.RETRY: RetryStrategy(
                max_retries=self.max_retries, base_delay=self.base_delay
            ),
            RecoveryStrategy.ALTERNATIVE_PATH: AlternativePathStrategy(
                alternatives={
                    "model_template": ["template_v2", "template_simple"],
                    "validator_template": ["validator_strict", "validator_basic"],
                    "test_template": ["test_comprehensive", "test_basic"],
                }
            ),
            RecoveryStrategy.GRACEFUL_DEGRADATION: GracefulDegradationStrategy(
                degradation_levels={
                    "validation": ["full", "basic", "minimal"],
                    "quality": [0.9, 0.8, 0.7, 0.6],
                    "quorum": ["full", "partial", "single"],
                }
            ),
            RecoveryStrategy.ERROR_CORRECTION: ErrorCorrectionStrategy(
                error_patterns=list(self.error_patterns.values())
            ),
            RecoveryStrategy.ESCALATION: EscalationStrategy(
                notification_callback=None  # Can be set later
            ),
        }

        logger.debug(f"Initialized {len(self.strategies)} recovery strategies")
