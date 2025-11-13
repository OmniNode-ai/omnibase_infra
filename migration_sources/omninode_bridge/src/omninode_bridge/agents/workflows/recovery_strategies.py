"""
Recovery strategy implementations for error recovery orchestration.

This module provides concrete implementations of each recovery strategy:
- RetryStrategy: Simple retry with exponential backoff
- AlternativePathStrategy: Try alternative templates/approaches
- GracefulDegradationStrategy: Fall back to simpler generation
- ErrorCorrectionStrategy: Attempt to fix known error patterns
- EscalationStrategy: Escalate to human intervention

Performance targets:
- Individual strategy execution: <300ms
- Exponential backoff: Base delay 1s, max 8s (3 retries)
- Pattern-based correction: <100ms

Example:
    ```python
    from omninode_bridge.agents.workflows.recovery_strategies import (
        RetryStrategy,
        ErrorCorrectionStrategy
    )

    # Retry with exponential backoff
    retry_strategy = RetryStrategy(max_retries=3, base_delay=1.0)
    result = await retry_strategy.execute(operation, context)

    # Error correction
    correction_strategy = ErrorCorrectionStrategy(error_patterns)
    result = await correction_strategy.execute(operation, context, exception)
    ```
"""

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any, Optional

from .recovery_models import (
    ErrorPattern,
    RecoveryContext,
    RecoveryResult,
    RecoveryStrategy,
)

logger = logging.getLogger(__name__)


class BaseRecoveryStrategy:
    """
    Base class for recovery strategies.

    Provides common functionality for all recovery strategies.
    """

    def __init__(self, strategy_type: RecoveryStrategy) -> None:
        """
        Initialize base recovery strategy.

        Args:
            strategy_type: Type of recovery strategy
        """
        self.strategy_type = strategy_type

    async def execute(self, context: RecoveryContext, **kwargs: Any) -> RecoveryResult:
        """
        Execute recovery strategy (to be overridden).

        Args:
            context: Recovery context
            **kwargs: Strategy-specific arguments

        Returns:
            RecoveryResult with recovery outcome
        """
        raise NotImplementedError("Subclasses must implement execute()")


class RetryStrategy(BaseRecoveryStrategy):
    """
    Simple retry strategy with exponential backoff.

    Retries the operation with increasing delays between attempts:
    - Retry 1: base_delay (default: 1s)
    - Retry 2: base_delay * 2 (default: 2s)
    - Retry 3: base_delay * 4 (default: 4s)

    Performance:
    - Base delay: 1s
    - Max delay: 8s (after 3 retries)
    - Total overhead: ~7s for 3 retries

    Example:
        ```python
        strategy = RetryStrategy(max_retries=3, base_delay=1.0)

        # Define operation to retry
        async def operation(ctx: dict) -> dict:
            # Attempt generation
            return await generate_code(ctx)

        result = await strategy.execute(
            context=recovery_context,
            operation=operation
        )
        ```
    """

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0) -> None:
        """
        Initialize retry strategy.

        Args:
            max_retries: Maximum retry attempts (default: 3)
            base_delay: Base delay in seconds (default: 1.0)
        """
        super().__init__(RecoveryStrategy.RETRY)
        self.max_retries = max_retries
        self.base_delay = base_delay

    async def execute(
        self, context: RecoveryContext, operation: Callable, **kwargs: Any
    ) -> RecoveryResult:
        """
        Execute operation with exponential backoff retry.

        Args:
            context: Recovery context
            operation: Async operation to retry
            **kwargs: Additional arguments passed to operation

        Returns:
            RecoveryResult with retry outcome
        """
        start_time = time.perf_counter()
        retry_count = 0
        last_exception: Optional[Exception] = None

        for retry_count in range(self.max_retries + 1):  # +1 for initial attempt
            try:
                # Calculate delay (exponential backoff)
                if retry_count > 0:
                    delay = self.base_delay * (2 ** (retry_count - 1))
                    logger.info(
                        f"[RetryStrategy] Retry {retry_count}/{self.max_retries} "
                        f"after {delay:.1f}s delay (workflow: {context.workflow_id})"
                    )
                    await asyncio.sleep(delay)

                # Execute operation
                result = await operation(context.state, **kwargs)

                # Success!
                duration_ms = (time.perf_counter() - start_time) * 1000

                logger.info(
                    f"[RetryStrategy] Operation succeeded on attempt {retry_count + 1} "
                    f"(workflow: {context.workflow_id}, duration: {duration_ms:.2f}ms)"
                )

                return RecoveryResult(
                    success=True,
                    strategy_used=self.strategy_type,
                    retry_count=retry_count,
                    error_fixed=True,
                    fallback_used=False,
                    duration_ms=duration_ms,
                    metadata={"attempts": retry_count + 1},
                )

            except Exception as e:
                last_exception = e
                logger.warning(
                    f"[RetryStrategy] Attempt {retry_count + 1}/{self.max_retries + 1} failed: {e}"
                )

        # All retries exhausted
        duration_ms = (time.perf_counter() - start_time) * 1000

        logger.error(
            f"[RetryStrategy] All {self.max_retries + 1} attempts failed "
            f"(workflow: {context.workflow_id})"
        )

        return RecoveryResult(
            success=False,
            strategy_used=self.strategy_type,
            retry_count=retry_count,
            error_fixed=False,
            fallback_used=False,
            error_message=str(last_exception) if last_exception else "Unknown error",
            duration_ms=duration_ms,
            metadata={"attempts": retry_count + 1},
        )


class AlternativePathStrategy(BaseRecoveryStrategy):
    """
    Try alternative template or approach strategy.

    When primary approach fails, attempts alternative templates or generation methods.

    Performance:
    - Template switching: <50ms
    - Alternative generation: <200ms
    - Total overhead: <300ms

    Example:
        ```python
        strategy = AlternativePathStrategy(
            alternatives={
                "model_template": ["template_v2", "template_simple"],
                "validator_template": ["validator_strict", "validator_basic"]
            }
        )

        result = await strategy.execute(
            context=recovery_context,
            failed_path="model_template"
        )
        ```
    """

    def __init__(self, alternatives: dict[str, list[str]]) -> None:
        """
        Initialize alternative path strategy.

        Args:
            alternatives: Map of failed paths to alternative options
        """
        super().__init__(RecoveryStrategy.ALTERNATIVE_PATH)
        self.alternatives = alternatives

    async def execute(
        self, context: RecoveryContext, failed_path: str, **kwargs: Any
    ) -> RecoveryResult:
        """
        Try alternative templates/approaches.

        Args:
            context: Recovery context
            failed_path: Path/template that failed
            **kwargs: Additional arguments

        Returns:
            RecoveryResult with alternative path outcome
        """
        start_time = time.perf_counter()

        # Get alternatives for failed path
        alternatives = self.alternatives.get(failed_path, [])

        if not alternatives:
            logger.warning(
                f"[AlternativePathStrategy] No alternatives found for '{failed_path}'"
            )
            return RecoveryResult(
                success=False,
                strategy_used=self.strategy_type,
                retry_count=0,
                error_fixed=False,
                fallback_used=False,
                error_message=f"No alternatives available for '{failed_path}'",
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Try each alternative
        for alt_idx, alternative in enumerate(alternatives):
            try:
                logger.info(
                    f"[AlternativePathStrategy] Trying alternative '{alternative}' "
                    f"({alt_idx + 1}/{len(alternatives)})"
                )

                # Update state with alternative path
                context.state["template_name"] = alternative
                context.state["alternative_used"] = True

                # Success - alternative selected
                duration_ms = (time.perf_counter() - start_time) * 1000

                return RecoveryResult(
                    success=True,
                    strategy_used=self.strategy_type,
                    retry_count=alt_idx,
                    error_fixed=False,  # Workaround, not fix
                    fallback_used=True,
                    duration_ms=duration_ms,
                    metadata={
                        "alternative_used": alternative,
                        "alternatives_tried": alt_idx + 1,
                    },
                )

            except Exception as e:
                logger.warning(
                    f"[AlternativePathStrategy] Alternative '{alternative}' failed: {e}"
                )

        # All alternatives failed
        duration_ms = (time.perf_counter() - start_time) * 1000

        return RecoveryResult(
            success=False,
            strategy_used=self.strategy_type,
            retry_count=len(alternatives),
            error_fixed=False,
            fallback_used=False,
            error_message="All alternatives failed",
            duration_ms=duration_ms,
            metadata={"alternatives_tried": len(alternatives)},
        )


class GracefulDegradationStrategy(BaseRecoveryStrategy):
    """
    Graceful degradation strategy - fall back to simpler generation.

    When complex generation fails, falls back to simpler approaches:
    - Full validation → Basic validation
    - Complex model → Simple model
    - AI quorum → Single model

    Performance:
    - Degradation decision: <50ms
    - Simpler generation: <200ms
    - Total overhead: <300ms

    Example:
        ```python
        strategy = GracefulDegradationStrategy(
            degradation_levels={
                "validation": ["full", "basic", "minimal"],
                "quality": [0.9, 0.8, 0.7]
            }
        )

        result = await strategy.execute(
            context=recovery_context,
            component="validation",
            current_level="full"
        )
        ```
    """

    def __init__(self, degradation_levels: dict[str, list[Any]]) -> None:
        """
        Initialize graceful degradation strategy.

        Args:
            degradation_levels: Map of components to degradation levels
        """
        super().__init__(RecoveryStrategy.GRACEFUL_DEGRADATION)
        self.degradation_levels = degradation_levels

    async def execute(
        self,
        context: RecoveryContext,
        component: str,
        current_level: Any = None,
        **kwargs: Any,
    ) -> RecoveryResult:
        """
        Degrade to simpler approach.

        Args:
            context: Recovery context
            component: Component to degrade (validation, quality, etc.)
            current_level: Current complexity level
            **kwargs: Additional arguments

        Returns:
            RecoveryResult with degradation outcome
        """
        start_time = time.perf_counter()

        # Get degradation levels for component
        levels = self.degradation_levels.get(component, [])

        if not levels:
            return RecoveryResult(
                success=False,
                strategy_used=self.strategy_type,
                retry_count=0,
                error_fixed=False,
                fallback_used=False,
                error_message=f"No degradation levels for '{component}'",
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Find next lower level
        try:
            if current_level is not None and current_level in levels:
                current_idx = levels.index(current_level)
                if current_idx + 1 < len(levels):
                    next_level = levels[current_idx + 1]
                else:
                    # Already at lowest level
                    return RecoveryResult(
                        success=False,
                        strategy_used=self.strategy_type,
                        retry_count=0,
                        error_fixed=False,
                        fallback_used=False,
                        error_message="Already at minimum degradation level",
                        duration_ms=(time.perf_counter() - start_time) * 1000,
                    )
            else:
                # Use first (highest) level
                next_level = levels[0]

            # Update state with degraded level
            context.state[f"{component}_level"] = next_level
            context.state["degraded"] = True

            duration_ms = (time.perf_counter() - start_time) * 1000

            logger.info(
                f"[GracefulDegradationStrategy] Degraded '{component}' "
                f"from '{current_level}' to '{next_level}'"
            )

            return RecoveryResult(
                success=True,
                strategy_used=self.strategy_type,
                retry_count=0,
                error_fixed=False,  # Workaround, not fix
                fallback_used=True,
                duration_ms=duration_ms,
                metadata={
                    "component": component,
                    "previous_level": current_level,
                    "degraded_level": next_level,
                },
            )

        except Exception as e:
            logger.error(f"[GracefulDegradationStrategy] Degradation failed: {e}")
            return RecoveryResult(
                success=False,
                strategy_used=self.strategy_type,
                retry_count=0,
                error_fixed=False,
                fallback_used=False,
                error_message=str(e),
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )


class ErrorCorrectionStrategy(BaseRecoveryStrategy):
    """
    Error correction strategy - attempt to fix known error patterns.

    Applies pattern-specific corrections to fix common errors:
    - Missing async keyword → Add async
    - Import errors → Fix import statements
    - Type hint errors → Correct type annotations

    Performance:
    - Pattern matching: <50ms
    - Correction application: <100ms
    - Total overhead: <200ms

    Example:
        ```python
        patterns = [
            ErrorPattern(
                pattern_id="missing_async",
                error_type=ErrorType.SYNTAX,
                regex_pattern=r"SyntaxError.*def execute",
                recovery_strategy=RecoveryStrategy.ERROR_CORRECTION,
                metadata={"fix": "add_async_keyword"}
            )
        ]

        strategy = ErrorCorrectionStrategy(error_patterns=patterns)
        result = await strategy.execute(
            context=recovery_context,
            code="def execute_effect():",
            error_pattern=patterns[0]
        )
        ```
    """

    def __init__(self, error_patterns: list[ErrorPattern]) -> None:
        """
        Initialize error correction strategy.

        Args:
            error_patterns: List of known error patterns with corrections
        """
        super().__init__(RecoveryStrategy.ERROR_CORRECTION)
        self.error_patterns = {p.pattern_id: p for p in error_patterns}

    async def execute(
        self,
        context: RecoveryContext,
        code: str,
        error_pattern: ErrorPattern,
        **kwargs: Any,
    ) -> RecoveryResult:
        """
        Apply pattern-specific error correction.

        Args:
            context: Recovery context
            code: Code to correct
            error_pattern: Matched error pattern
            **kwargs: Additional arguments

        Returns:
            RecoveryResult with correction outcome
        """
        start_time = time.perf_counter()

        try:
            # Get correction metadata
            fix_type = error_pattern.metadata.get("fix")

            if not fix_type:
                return RecoveryResult(
                    success=False,
                    strategy_used=self.strategy_type,
                    retry_count=0,
                    error_fixed=False,
                    fallback_used=False,
                    error_message=f"No fix metadata for pattern '{error_pattern.pattern_id}'",
                    duration_ms=(time.perf_counter() - start_time) * 1000,
                    pattern_matched=error_pattern.pattern_id,
                )

            # Apply correction based on fix type
            corrected_code = await self._apply_correction(code, fix_type, error_pattern)

            # Update state with corrected code
            context.state["corrected_code"] = corrected_code
            context.state["correction_applied"] = fix_type

            duration_ms = (time.perf_counter() - start_time) * 1000

            logger.info(
                f"[ErrorCorrectionStrategy] Applied correction '{fix_type}' "
                f"for pattern '{error_pattern.pattern_id}'"
            )

            return RecoveryResult(
                success=True,
                strategy_used=self.strategy_type,
                retry_count=1,
                error_fixed=True,  # Actual fix
                fallback_used=False,
                duration_ms=duration_ms,
                pattern_matched=error_pattern.pattern_id,
                metadata={"fix_type": fix_type, "code_length": len(corrected_code)},
            )

        except Exception as e:
            logger.error(f"[ErrorCorrectionStrategy] Correction failed: {e}")
            return RecoveryResult(
                success=False,
                strategy_used=self.strategy_type,
                retry_count=1,
                error_fixed=False,
                fallback_used=False,
                error_message=str(e),
                duration_ms=(time.perf_counter() - start_time) * 1000,
                pattern_matched=error_pattern.pattern_id,
            )

    async def _apply_correction(
        self, code: str, fix_type: str, pattern: ErrorPattern
    ) -> str:
        """
        Apply specific correction to code.

        Args:
            code: Original code
            fix_type: Type of fix to apply
            pattern: Error pattern with correction metadata

        Returns:
            Corrected code
        """
        if fix_type == "add_async_keyword":
            # Add async to function definitions
            return code.replace("def execute", "async def execute")

        elif fix_type == "fix_import":
            # Fix import statement
            old_import = pattern.metadata.get("old_import", "")
            new_import = pattern.metadata.get("new_import", "")
            if old_import and new_import:
                return code.replace(old_import, new_import)

        elif fix_type == "add_type_hint":
            # Add missing type hint
            # Simple implementation - can be enhanced
            return code

        elif fix_type == "remove_duplicate":
            # Remove duplicate code
            lines = code.split("\n")
            unique_lines = []
            seen = set()
            for line in lines:
                if line not in seen:
                    unique_lines.append(line)
                    seen.add(line)
            return "\n".join(unique_lines)

        else:
            # Unknown fix type - return original
            logger.warning(f"Unknown fix type: {fix_type}")
            return code


class EscalationStrategy(BaseRecoveryStrategy):
    """
    Escalation strategy - escalate to human intervention.

    When automated recovery fails, creates intervention request for human review.

    Performance:
    - Escalation creation: <50ms
    - Notification: <100ms
    - Total overhead: <200ms

    Example:
        ```python
        strategy = EscalationStrategy(
            notification_callback=send_slack_notification
        )

        result = await strategy.execute(
            context=recovery_context,
            error_summary="AI quorum failed after 3 retries",
            suggested_action="Review contract structure"
        )
        ```
    """

    def __init__(self, notification_callback: Optional[Callable] = None) -> None:
        """
        Initialize escalation strategy.

        Args:
            notification_callback: Optional callback to notify humans
        """
        super().__init__(RecoveryStrategy.ESCALATION)
        self.notification_callback = notification_callback

    async def execute(
        self,
        context: RecoveryContext,
        error_summary: str,
        suggested_action: str = "Manual review required",
        **kwargs: Any,
    ) -> RecoveryResult:
        """
        Escalate to human intervention.

        Args:
            context: Recovery context
            error_summary: Summary of error condition
            suggested_action: Suggested action for human
            **kwargs: Additional arguments

        Returns:
            RecoveryResult with escalation outcome
        """
        start_time = time.perf_counter()

        try:
            # Create escalation record
            escalation_data = {
                "workflow_id": context.workflow_id,
                "node_name": context.node_name,
                "step_count": context.step_count,
                "error_summary": error_summary,
                "suggested_action": suggested_action,
                "state_snapshot": context.state,
                "correlation_id": context.correlation_id,
            }

            # Store escalation in context
            context.state["escalation"] = escalation_data
            context.state["requires_human_intervention"] = True

            # Send notification if callback provided
            if self.notification_callback:
                try:
                    await self.notification_callback(escalation_data)
                except Exception as e:
                    logger.warning(f"[EscalationStrategy] Notification failed: {e}")

            duration_ms = (time.perf_counter() - start_time) * 1000

            logger.warning(
                f"[EscalationStrategy] Escalated to human intervention "
                f"(workflow: {context.workflow_id}, reason: {error_summary})"
            )

            return RecoveryResult(
                success=True,  # Escalation created successfully
                strategy_used=self.strategy_type,
                retry_count=0,
                error_fixed=False,
                fallback_used=False,
                duration_ms=duration_ms,
                metadata={
                    "error_summary": error_summary,
                    "suggested_action": suggested_action,
                    "escalation_created": True,
                },
            )

        except Exception as e:
            logger.error(f"[EscalationStrategy] Escalation failed: {e}")
            return RecoveryResult(
                success=False,
                strategy_used=self.strategy_type,
                retry_count=0,
                error_fixed=False,
                fallback_used=False,
                error_message=str(e),
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )
