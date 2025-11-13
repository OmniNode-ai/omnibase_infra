"""
Recovery models for error recovery orchestration.

This module provides data models for error patterns, recovery strategies,
and recovery results used by the Error Recovery Orchestrator.

Performance targets:
- Error analysis: <100ms
- Recovery decision: <50ms
- Total recovery overhead: <500ms

Example:
    ```python
    from omninode_bridge.agents.workflows.recovery_models import (
        RecoveryStrategy,
        ErrorPattern,
        RecoveryResult,
        RecoveryContext
    )

    # Define error pattern
    pattern = ErrorPattern(
        pattern_id="syntax_error_missing_async",
        error_type=ErrorType.SYNTAX,
        regex_pattern=r"SyntaxError.*async.*def",
        recovery_strategy=RecoveryStrategy.ERROR_CORRECTION,
        max_retries=3,
        metadata={"fix": "add_async_keyword"}
    )

    # Create recovery context
    context = RecoveryContext(
        workflow_id="codegen-1",
        node_name="model_generator",
        step_count=3,
        state={"contract": {...}},
        correlation_id="abc-123"
    )

    # Record recovery result
    result = RecoveryResult(
        success=True,
        strategy_used=RecoveryStrategy.ERROR_CORRECTION,
        retry_count=1,
        error_fixed=True,
        fallback_used=False,
        duration_ms=150.5
    )
    ```
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from omninode_bridge.security import safe_compile, safe_search

# Configure logger
logger = logging.getLogger(__name__)


class RecoveryStrategy(str, Enum):
    """
    Recovery strategy types for error handling.

    Strategies are ordered from simplest (retry) to most complex (escalation).
    """

    RETRY = "retry"  # Simple retry with exponential backoff
    ALTERNATIVE_PATH = "alternative_path"  # Try different approach/template
    GRACEFUL_DEGRADATION = "graceful_degradation"  # Fall back to simpler generation
    ERROR_CORRECTION = "error_correction"  # Attempt to fix known error patterns
    ESCALATION = "escalation"  # Escalate to human intervention


class ErrorType(str, Enum):
    """
    Error types for pattern matching.

    Used to categorize errors and select appropriate recovery strategies.
    """

    SYNTAX = "syntax"  # Python syntax errors
    IMPORT = "import"  # Import resolution failures
    VALIDATION = "validation"  # Validation errors (type checking, ONEX compliance)
    RUNTIME = "runtime"  # Runtime errors during execution
    TEMPLATE = "template"  # Template rendering errors
    QUORUM = "quorum"  # AI quorum failures
    TIMEOUT = "timeout"  # Operation timeout
    UNKNOWN = "unknown"  # Unknown error type


@dataclass
class ErrorPattern:
    r"""
    Error pattern definition for pattern matching.

    Attributes:
        pattern_id: Unique pattern identifier
        error_type: Category of error (syntax, import, etc.)
        regex_pattern: Regular expression to match error message
        recovery_strategy: Recommended recovery strategy
        max_retries: Maximum retry attempts (default: 3)
        metadata: Additional metadata for recovery hints
        priority: Pattern priority (higher = more specific, 1-10)
        enabled: Enable/disable pattern (default: True)

    Example:
        ```python
        pattern = ErrorPattern(
            pattern_id="import_not_found",
            error_type=ErrorType.IMPORT,
            regex_pattern=r"ImportError.*No module named '(\w+)'",
            recovery_strategy=RecoveryStrategy.ALTERNATIVE_PATH,
            max_retries=2,
            metadata={
                "alternative_imports": ["typing", "collections.abc"],
                "fallback": "use_builtin"
            },
            priority=8
        )
        ```
    """

    pattern_id: str
    error_type: ErrorType
    regex_pattern: str
    recovery_strategy: RecoveryStrategy
    max_retries: int = 3
    metadata: dict[str, Any] = field(default_factory=dict)
    priority: int = 5  # 1-10, higher = more specific
    enabled: bool = True

    def __post_init__(self) -> None:
        """Validate pattern configuration."""
        if not self.pattern_id:
            raise ValueError("pattern_id cannot be empty")
        if not self.regex_pattern:
            raise ValueError("regex_pattern cannot be empty")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if not 1 <= self.priority <= 10:
            raise ValueError("priority must be between 1 and 10")

        # Compile regex with security validation to prevent ReDoS
        try:
            safe_compile(self.regex_pattern)
            logger.debug(
                f"Validated regex pattern for {self.pattern_id}",
                extra={"pattern_id": self.pattern_id, "error_type": self.error_type},
            )
        except Exception as e:
            logger.error(
                f"Invalid or dangerous regex pattern for {self.pattern_id}: {e}",
                extra={
                    "pattern_id": self.pattern_id,
                    "pattern_preview": self.regex_pattern[:100],
                },
            )
            raise ValueError(f"Invalid or dangerous regex pattern: {e}")

    def matches(self, error_message: str) -> bool:
        """
        Check if error message matches this pattern.

        Args:
            error_message: Error message to match

        Returns:
            True if pattern matches, False otherwise
        """
        if not self.enabled:
            return False

        try:
            # Use safe_search with timeout protection
            match = safe_search(
                self.regex_pattern, error_message, flags=re.IGNORECASE, timeout=0.5
            )
            return bool(match)
        except TimeoutError:
            logger.warning(
                f"Regex timeout for pattern {self.pattern_id}",
                extra={
                    "pattern_id": self.pattern_id,
                    "message_length": len(error_message),
                },
            )
            return False
        except Exception as e:
            logger.error(
                f"Error matching pattern {self.pattern_id}: {e}",
                extra={"pattern_id": self.pattern_id},
            )
            return False

    def extract_groups(self, error_message: str) -> dict[str, str]:
        """
        Extract named groups from error message.

        Args:
            error_message: Error message to extract from

        Returns:
            Dictionary of named groups
        """
        try:
            # Use safe_search with timeout protection
            match = safe_search(
                self.regex_pattern, error_message, flags=re.IGNORECASE, timeout=0.5
            )
            if match:
                return match.groupdict()
            return {}
        except TimeoutError:
            logger.warning(
                f"Regex timeout extracting groups for pattern {self.pattern_id}",
                extra={
                    "pattern_id": self.pattern_id,
                    "message_length": len(error_message),
                },
            )
            return {}
        except Exception as e:
            logger.error(
                f"Error extracting groups from pattern {self.pattern_id}: {e}",
                extra={"pattern_id": self.pattern_id},
            )
            return {}


@dataclass
class RecoveryContext:
    """
    Context for error recovery operations.

    Attributes:
        workflow_id: Workflow identifier
        node_name: Name of node where error occurred
        step_count: Current step count in workflow
        state: Current workflow state
        exception: Original exception (optional)
        error_message: Error message from exception
        correlation_id: Correlation ID for tracing
        metadata: Additional context metadata
        created_at: Context creation timestamp

    Example:
        ```python
        context = RecoveryContext(
            workflow_id="codegen-session-1",
            node_name="validator_generator",
            step_count=5,
            state={"contract_data": {...}, "generated_models": [...]},
            exception=SyntaxError("invalid syntax"),
            error_message="SyntaxError: invalid syntax at line 45",
            correlation_id="abc-123"
        )
        ```
    """

    workflow_id: str
    node_name: str
    step_count: int
    state: dict[str, Any]
    exception: Optional[Exception] = None
    error_message: str = ""
    correlation_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        """Validate context and extract error message if needed."""
        if not self.workflow_id:
            raise ValueError("workflow_id cannot be empty")
        if not self.node_name:
            raise ValueError("node_name cannot be empty")
        if self.step_count < 0:
            raise ValueError("step_count must be >= 0")

        # Extract error message from exception if not provided
        if self.exception and not self.error_message:
            self.error_message = str(self.exception)


@dataclass
class RecoveryResult:
    """
    Result of error recovery operation.

    Attributes:
        success: True if recovery succeeded
        strategy_used: Recovery strategy that was used
        retry_count: Number of retries performed
        error_fixed: True if error was fixed (vs. workaround)
        fallback_used: True if fallback approach was used
        error_message: Error message (if recovery failed)
        duration_ms: Recovery operation duration
        metadata: Additional result metadata
        pattern_matched: Error pattern that was matched (optional)
        created_at: Result creation timestamp

    Example:
        ```python
        result = RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.ERROR_CORRECTION,
            retry_count=2,
            error_fixed=True,
            fallback_used=False,
            duration_ms=245.5,
            metadata={
                "correction_applied": "added_async_keyword",
                "validation_passed": True
            }
        )
        ```
    """

    success: bool
    strategy_used: RecoveryStrategy
    retry_count: int
    error_fixed: bool
    fallback_used: bool
    error_message: Optional[str] = None
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    pattern_matched: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        """Validate result."""
        if self.retry_count < 0:
            raise ValueError("retry_count must be >= 0")
        if self.duration_ms < 0:
            raise ValueError("duration_ms must be >= 0")


@dataclass
class RecoveryStatistics:
    """
    Aggregated recovery statistics.

    Tracks recovery performance across multiple operations.

    Attributes:
        total_recoveries: Total recovery attempts
        successful_recoveries: Number of successful recoveries
        failed_recoveries: Number of failed recoveries
        total_retries: Total retry count across all recoveries
        average_duration_ms: Average recovery duration
        max_duration_ms: Maximum recovery duration
        min_duration_ms: Minimum recovery duration
        strategies_used: Count of each strategy used
        error_types_seen: Count of each error type seen
        patterns_matched: Count of patterns matched
        success_rate: Overall success rate (0-1)

    Example:
        ```python
        stats = RecoveryStatistics(
            total_recoveries=100,
            successful_recoveries=85,
            failed_recoveries=15,
            total_retries=150,
            average_duration_ms=325.5,
            strategies_used={
                RecoveryStrategy.RETRY: 40,
                RecoveryStrategy.ERROR_CORRECTION: 35,
                RecoveryStrategy.ALTERNATIVE_PATH: 10,
            }
        )
        print(f"Success rate: {stats.success_rate:.1%}")
        ```
    """

    total_recoveries: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    total_retries: int = 0
    average_duration_ms: float = 0.0
    max_duration_ms: float = 0.0
    min_duration_ms: float = float("inf")
    strategies_used: dict[RecoveryStrategy, int] = field(default_factory=dict)
    error_types_seen: dict[ErrorType, int] = field(default_factory=dict)
    patterns_matched: dict[str, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0-1)."""
        if self.total_recoveries == 0:
            return 0.0
        return self.successful_recoveries / self.total_recoveries

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate (0-1)."""
        return 1.0 - self.success_rate

    @property
    def average_retries_per_recovery(self) -> float:
        """Calculate average retries per recovery."""
        if self.total_recoveries == 0:
            return 0.0
        return self.total_retries / self.total_recoveries

    def update_from_result(self, result: RecoveryResult) -> None:
        """
        Update statistics from recovery result.

        Args:
            result: RecoveryResult to update from
        """
        self.total_recoveries += 1

        if result.success:
            self.successful_recoveries += 1
        else:
            self.failed_recoveries += 1

        self.total_retries += result.retry_count

        # Update duration stats
        if result.duration_ms > 0:
            self.max_duration_ms = max(self.max_duration_ms, result.duration_ms)
            self.min_duration_ms = min(self.min_duration_ms, result.duration_ms)

            # Update average (running average)
            total_duration = self.average_duration_ms * (self.total_recoveries - 1)
            total_duration += result.duration_ms
            self.average_duration_ms = total_duration / self.total_recoveries

        # Update strategy counts
        self.strategies_used[result.strategy_used] = (
            self.strategies_used.get(result.strategy_used, 0) + 1
        )

        # Update pattern counts
        if result.pattern_matched:
            self.patterns_matched[result.pattern_matched] = (
                self.patterns_matched.get(result.pattern_matched, 0) + 1
            )
