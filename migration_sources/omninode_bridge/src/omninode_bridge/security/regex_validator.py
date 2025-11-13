#!/usr/bin/env python3
"""
Regex Pattern Safety Validator for ReDoS Prevention.

Validates regex patterns before execution to prevent Regular Expression Denial of Service
(ReDoS) attacks and performance issues from catastrophic backtracking.

ONEX v2.0 Compliance:
- Pattern complexity analysis
- Timeout protection for execution
- Dangerous pattern detection
- Audit logging for security events
- Pattern compilation caching
"""

import logging
import re
import signal
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, ClassVar, Optional
from re import Pattern

from omninode_bridge.security.exceptions import (
    InputValidationError,
    SecurityValidationError,
)

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class RegexValidationResult:
    """Result of regex pattern validation."""

    is_valid: bool
    warnings: list[str]
    severity: str  # "low", "medium", "high"
    compiled_pattern: Optional[Pattern[str]] = None


class RegexValidator:
    """
    Validates regex patterns for security and performance issues.

    Features:
    - Catastrophic backtracking detection
    - Nested quantifier detection
    - Pattern complexity analysis
    - Timeout protection for execution
    - Pattern compilation caching
    - Audit logging for rejected patterns
    """

    # Dangerous regex patterns that can cause ReDoS
    # These are literal patterns to search for in the user's regex
    DANGEROUS_PATTERNS: ClassVar[list[tuple[str, str, str]]] = [
        # Nested quantifiers (HIGH severity) - look for )+, *, +, } after )
        (r"\)\s*[+*]", "Nested quantifiers detected (e.g., (x+)+)", "high"),
        (r"\)\s*\{", "Nested quantifiers with range detected", "high"),
        # Multiple quantifiers on same expression (HIGH severity)
        (r"[+*]\s*[+*]", "Multiple consecutive quantifiers detected", "high"),
        # Overlapping alternation (MEDIUM severity)
        (
            r"\(([^)]+)\|(\1)\)",
            "Alternation with identical patterns detected",
            "medium",
        ),
        # Very long quantifier ranges (MEDIUM severity)
        (r"\{\d{3,},\}", "Very large quantifier range detected", "medium"),
        # Multiple .* or .+ in sequence (MEDIUM severity)
        (r"\.\*\s*\.\*", "Multiple wildcards in sequence detected", "medium"),
        (r"\.\+\s*\.\+", "Multiple greedy matches in sequence detected", "medium"),
    ]

    # Maximum allowed pattern length
    MAX_PATTERN_LENGTH = 500

    # Maximum allowed nesting depth
    MAX_NESTING_DEPTH = 15

    # Regex execution timeout (seconds)
    EXECUTION_TIMEOUT = 1.0

    # Pattern complexity threshold
    MAX_COMPLEXITY_SCORE = 500

    def __init__(self, strict_mode: bool = False):
        """
        Initialize regex validator.

        Args:
            strict_mode: If True, reject on first high-severity issue
        """
        self.strict_mode = strict_mode

    def validate_pattern(self, pattern: str, flags: int = 0) -> RegexValidationResult:
        """
        Validate regex pattern for security and performance issues.

        Args:
            pattern: Regex pattern to validate
            flags: Regex flags (re.IGNORECASE, etc.)

        Returns:
            RegexValidationResult with validity, warnings, and compiled pattern

        Raises:
            SecurityValidationError: If pattern contains critical security issues
        """
        warnings = []
        severity_counts = {"low": 0, "medium": 0, "high": 0}

        # Check length
        if len(pattern) > self.MAX_PATTERN_LENGTH:
            raise SecurityValidationError(
                f"Regex pattern exceeds maximum length of {self.MAX_PATTERN_LENGTH} characters",
                severity="medium",
            )

        # Check for empty pattern
        if not pattern.strip():
            raise InputValidationError("Regex pattern cannot be empty")

        # Check for dangerous patterns
        for dangerous_pattern, message, severity in self.DANGEROUS_PATTERNS:
            if re.search(dangerous_pattern, pattern):
                warnings.append(message)
                severity_counts[severity] += 1

                logger.warning(
                    "Dangerous regex pattern detected",
                    extra={
                        "pattern": pattern[:100],  # Limit pattern in logs
                        "issue": message,
                        "severity": severity,
                    },
                )

                # In strict mode, reject immediately on high severity
                if self.strict_mode and severity == "high":
                    raise SecurityValidationError(
                        f"Dangerous regex pattern: {message}. Pattern: {pattern[:50]}...",
                        severity="high",
                    )

        # Calculate complexity score
        complexity_score = self._calculate_complexity(pattern)
        if complexity_score > self.MAX_COMPLEXITY_SCORE:
            warnings.append(
                f"Pattern complexity score ({complexity_score}) exceeds threshold ({self.MAX_COMPLEXITY_SCORE})"
            )
            severity_counts["high"] += 1

            if self.strict_mode:
                raise SecurityValidationError(
                    f"Regex pattern too complex (score: {complexity_score})",
                    severity="high",
                )

        # Check nesting depth
        nesting_depth = self._calculate_nesting_depth(pattern)
        if nesting_depth > self.MAX_NESTING_DEPTH:
            warnings.append(
                f"Pattern nesting depth ({nesting_depth}) exceeds maximum ({self.MAX_NESTING_DEPTH})"
            )
            severity_counts["high"] += 1

            if self.strict_mode:
                raise SecurityValidationError(
                    f"Regex pattern nesting too deep (depth: {nesting_depth})",
                    severity="high",
                )

        # Try to compile the pattern
        compiled_pattern = None
        try:
            compiled_pattern = self._safe_compile(pattern, flags)
        except re.error as e:
            raise InputValidationError(f"Invalid regex pattern: {e}")
        except Exception as e:
            logger.error(
                "Unexpected error compiling regex pattern",
                extra={"pattern": pattern[:100], "error": str(e)},
            )
            raise SecurityValidationError(
                f"Failed to compile regex pattern: {e}", severity="high"
            )

        # Determine overall severity
        if severity_counts["high"] > 0:
            overall_severity = "high"
        elif severity_counts["medium"] > 0:
            overall_severity = "medium"
        else:
            overall_severity = "low"

        # Reject if too many high-severity issues
        if severity_counts["high"] >= 2:
            raise SecurityValidationError(
                f"Regex pattern has {severity_counts['high']} high-severity security issues: {warnings}",
                severity="high",
            )

        # Log validation result
        if warnings:
            logger.info(
                f"Regex validation completed with {len(warnings)} warnings",
                extra={
                    "warning_count": len(warnings),
                    "severity": overall_severity,
                    "pattern_length": len(pattern),
                    "complexity_score": complexity_score,
                    "nesting_depth": nesting_depth,
                },
            )

        return RegexValidationResult(
            is_valid=True,
            warnings=warnings,
            severity=overall_severity,
            compiled_pattern=compiled_pattern,
        )

    def safe_match(
        self, pattern: str, text: str, flags: int = 0, timeout: float = None
    ) -> Optional[re.Match[str]]:
        """
        Safely execute regex match with validation and timeout.

        Args:
            pattern: Regex pattern
            text: Text to match against
            flags: Regex flags
            timeout: Timeout in seconds (default: EXECUTION_TIMEOUT)

        Returns:
            Match object or None

        Raises:
            SecurityValidationError: If pattern is dangerous
            TimeoutError: If execution exceeds timeout
        """
        # Validate pattern first
        validation_result = self.validate_pattern(pattern, flags)

        # Use cached compiled pattern if available
        compiled_pattern = validation_result.compiled_pattern
        if compiled_pattern is None:
            raise SecurityValidationError("Pattern validation failed", severity="high")

        # Execute with timeout
        timeout_seconds = timeout or self.EXECUTION_TIMEOUT
        try:
            with self._timeout(timeout_seconds):
                return compiled_pattern.match(text)
        except TimeoutError:
            logger.error(
                "Regex execution timeout",
                extra={
                    "pattern": pattern[:100],
                    "text_length": len(text),
                    "timeout": timeout_seconds,
                },
            )
            raise TimeoutError(
                f"Regex execution exceeded {timeout_seconds}s timeout. Pattern may cause ReDoS."
            )

    def safe_search(
        self, pattern: str, text: str, flags: int = 0, timeout: float = None
    ) -> Optional[re.Match[str]]:
        """
        Safely execute regex search with validation and timeout.

        Args:
            pattern: Regex pattern
            text: Text to search
            flags: Regex flags
            timeout: Timeout in seconds (default: EXECUTION_TIMEOUT)

        Returns:
            Match object or None

        Raises:
            SecurityValidationError: If pattern is dangerous
            TimeoutError: If execution exceeds timeout
        """
        # Validate pattern first
        validation_result = self.validate_pattern(pattern, flags)

        # Use cached compiled pattern if available
        compiled_pattern = validation_result.compiled_pattern
        if compiled_pattern is None:
            raise SecurityValidationError("Pattern validation failed", severity="high")

        # Execute with timeout
        timeout_seconds = timeout or self.EXECUTION_TIMEOUT
        try:
            with self._timeout(timeout_seconds):
                return compiled_pattern.search(text)
        except TimeoutError:
            logger.error(
                "Regex execution timeout",
                extra={
                    "pattern": pattern[:100],
                    "text_length": len(text),
                    "timeout": timeout_seconds,
                },
            )
            raise TimeoutError(
                f"Regex execution exceeded {timeout_seconds}s timeout. Pattern may cause ReDoS."
            )

    def safe_findall(
        self, pattern: str, text: str, flags: int = 0, timeout: float = None
    ) -> list[Any]:
        """
        Safely execute regex findall with validation and timeout.

        Args:
            pattern: Regex pattern
            text: Text to search
            flags: Regex flags
            timeout: Timeout in seconds (default: EXECUTION_TIMEOUT)

        Returns:
            List of matches

        Raises:
            SecurityValidationError: If pattern is dangerous
            TimeoutError: If execution exceeds timeout
        """
        # Validate pattern first
        validation_result = self.validate_pattern(pattern, flags)

        # Use cached compiled pattern if available
        compiled_pattern = validation_result.compiled_pattern
        if compiled_pattern is None:
            raise SecurityValidationError("Pattern validation failed", severity="high")

        # Execute with timeout
        timeout_seconds = timeout or self.EXECUTION_TIMEOUT
        try:
            with self._timeout(timeout_seconds):
                return compiled_pattern.findall(text)
        except TimeoutError:
            logger.error(
                "Regex execution timeout",
                extra={
                    "pattern": pattern[:100],
                    "text_length": len(text),
                    "timeout": timeout_seconds,
                },
            )
            raise TimeoutError(
                f"Regex execution exceeded {timeout_seconds}s timeout. Pattern may cause ReDoS."
            )

    @lru_cache(maxsize=256)
    def _safe_compile(self, pattern: str, flags: int = 0) -> Pattern[str]:
        """
        Compile regex pattern with caching.

        Args:
            pattern: Regex pattern
            flags: Regex flags

        Returns:
            Compiled pattern

        Raises:
            re.error: If pattern is invalid
        """
        return re.compile(pattern, flags)

    def _calculate_complexity(self, pattern: str) -> int:
        """
        Calculate complexity score for regex pattern.

        Args:
            pattern: Regex pattern

        Returns:
            Complexity score (higher = more complex)
        """
        score = 0

        # Count quantifiers
        score += pattern.count("+") * 5
        score += pattern.count("*") * 5
        score += pattern.count("{") * 3
        score += pattern.count("?") * 2

        # Count groups
        score += pattern.count("(") * 3

        # Count alternations
        score += pattern.count("|") * 4

        # Count character classes
        score += pattern.count("[") * 2

        # Count lookaheads/lookbehinds
        score += pattern.count("(?=") * 6
        score += pattern.count("(?!") * 6
        score += pattern.count("(?<=") * 7
        score += pattern.count("(?<!") * 7

        # Penalty for wildcards
        score += pattern.count(".*") * 8
        score += pattern.count(".+") * 8

        return score

    def _calculate_nesting_depth(self, pattern: str) -> int:
        """
        Calculate maximum nesting depth of groups.

        Args:
            pattern: Regex pattern

        Returns:
            Maximum nesting depth
        """
        max_depth = 0
        current_depth = 0

        i = 0
        while i < len(pattern):
            if pattern[i] == "\\":
                # Skip escaped characters
                i += 2
                continue
            elif pattern[i] == "(":
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif pattern[i] == ")":
                current_depth = max(0, current_depth - 1)
            i += 1

        return max_depth

    @staticmethod
    @contextmanager
    def _timeout(seconds: float):
        """
        Context manager for timeout protection (UNIX only).

        Args:
            seconds: Timeout duration

        Raises:
            TimeoutError: If operation exceeds timeout
        """
        # Note: signal.alarm only works on UNIX systems
        # For cross-platform support, consider using threading.Timer
        import platform

        if platform.system() == "Windows":
            # Windows doesn't support signal.alarm
            # Just yield without timeout (log warning)
            logger.warning(
                "Regex timeout protection not available on Windows",
                extra={"platform": platform.system()},
            )
            yield
            return

        def timeout_handler(signum, frame):
            raise TimeoutError("Regex execution timeout")

        # Set the signal handler
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)

        try:
            yield
        finally:
            # Restore the old handler and cancel the alarm
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)


# Global validator instance
_global_validator: Optional[RegexValidator] = None


def get_regex_validator(strict_mode: bool = False) -> RegexValidator:
    """
    Get or create global regex validator instance.

    Args:
        strict_mode: If True, reject on first high-severity issue

    Returns:
        RegexValidator instance
    """
    global _global_validator
    if _global_validator is None:
        _global_validator = RegexValidator(strict_mode=strict_mode)
    return _global_validator


# Convenience functions for common operations
def safe_compile(pattern: str, flags: int = 0) -> Pattern[str]:
    """
    Safely compile regex pattern with validation.

    Args:
        pattern: Regex pattern
        flags: Regex flags

    Returns:
        Compiled pattern

    Raises:
        SecurityValidationError: If pattern is dangerous
    """
    validator = get_regex_validator()
    result = validator.validate_pattern(pattern, flags)
    if result.compiled_pattern is None:
        raise SecurityValidationError(
            "Failed to compile regex pattern", severity="high"
        )
    return result.compiled_pattern


def safe_match(
    pattern: str, text: str, flags: int = 0, timeout: float = None
) -> Optional[re.Match[str]]:
    """
    Safely execute regex match with validation and timeout.

    Args:
        pattern: Regex pattern
        text: Text to match against
        flags: Regex flags
        timeout: Timeout in seconds

    Returns:
        Match object or None
    """
    validator = get_regex_validator()
    return validator.safe_match(pattern, text, flags, timeout)


def safe_search(
    pattern: str, text: str, flags: int = 0, timeout: float = None
) -> Optional[re.Match[str]]:
    """
    Safely execute regex search with validation and timeout.

    Args:
        pattern: Regex pattern
        text: Text to search
        flags: Regex flags
        timeout: Timeout in seconds

    Returns:
        Match object or None
    """
    validator = get_regex_validator()
    return validator.safe_search(pattern, text, flags, timeout)


def safe_findall(
    pattern: str, text: str, flags: int = 0, timeout: float = None
) -> list[Any]:
    """
    Safely execute regex findall with validation and timeout.

    Args:
        pattern: Regex pattern
        text: Text to search
        flags: Regex flags
        timeout: Timeout in seconds

    Returns:
        List of matches
    """
    validator = get_regex_validator()
    return validator.safe_findall(pattern, text, flags, timeout)
