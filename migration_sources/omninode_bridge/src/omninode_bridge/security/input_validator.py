#!/usr/bin/env python3
"""
Input Validation for Security Hardening.

Validates user prompts and inputs to prevent injection attacks, malicious code generation,
and other security vulnerabilities.

ONEX v2.0 Compliance:
- Pattern-based threat detection
- Configurable security thresholds
- Audit logging for suspicious activity
- Non-blocking warning system
"""

import logging
import re
from dataclasses import dataclass
from typing import ClassVar

from omninode_bridge.security.exceptions import (
    CommandInjectionDetected,
    DynamicCodeExecutionError,
    PathTraversalAttempt,
    SecurityValidationError,
    SQLInjectionError,
    XSSAttemptError,
)

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of input validation."""

    is_valid: bool
    warnings: list[str]
    severity: str  # "low", "medium", "high"
    sanitized_input: str


class InputValidator:
    """
    Validates user input for security threats.

    Features:
    - Command injection detection
    - Path traversal detection
    - SQL injection detection
    - XSS detection
    - Length validation
    - Sanitization without breaking legitimate use
    """

    # Pattern definitions with severity levels
    DANGEROUS_PATTERNS: ClassVar[list[tuple[str, str, str]]] = [
        # Shell metacharacters (HIGH severity) - context-aware
        (
            r";\s*(rm|curl|wget|nc|bash|sh)\s+",
            "Shell command chaining detected",
            "high",
        ),
        (r"\|\s*(nc|bash|sh|curl|wget)\s+", "Shell piping detected", "high"),
        (r"`[^`]+`", "Backtick command substitution detected", "high"),
        (r"\$\(", "Command substitution detected", "high"),
        (r"&&\s*(rm|curl|wget|nc|bash)\s+", "Shell AND operator with command", "high"),
        # Path traversal (HIGH severity)
        (r"\.\.[/\\]", "Path traversal attempt detected", "high"),
        # SQL injection (HIGH severity)
        (r"'\s*OR\s+.*=.*", "SQL injection pattern detected", "high"),
        (r"'\s*OR\s+'", "SQL injection pattern detected", "high"),
        (
            r"';?\s*(DROP|DELETE|TRUNCATE|ALTER)\s+",
            "Dangerous SQL command detected",
            "high",
        ),
        # XSS attempts (MEDIUM severity)
        (r"<script[>\s]", "XSS script tag detected", "medium"),
        (r"javascript:", "JavaScript protocol detected", "medium"),
        # Command execution (HIGH severity)
        (r"__import__\s*\(", "Dynamic import detected", "high"),
        (r"exec\s*\(", "Exec function detected", "high"),
        (r"eval\s*\(", "Eval function detected", "high"),
        # File system access (MEDIUM severity)
        (r"/etc/passwd", "Access to sensitive system file detected", "medium"),
        (r"/var/www", "Access to web root detected", "medium"),
        # Network operations (LOW severity - may be legitimate)
        (r"curl\s+", "Network request detected", "low"),
        (r"wget\s+", "Network download detected", "low"),
    ]

    # Maximum allowed prompt length
    MAX_PROMPT_LENGTH = 10_000

    # Threshold for automatic rejection
    HIGH_SEVERITY_THRESHOLD = 2  # Reject if 2+ high-severity patterns
    MEDIUM_SEVERITY_THRESHOLD = 3  # Reject if 3+ medium-severity patterns

    def __init__(self, strict_mode: bool = False):
        """
        Initialize input validator.

        Args:
            strict_mode: If True, reject on first high-severity pattern
        """
        self.strict_mode = strict_mode

    def validate_prompt(self, prompt: str) -> ValidationResult:
        """
        Validate user prompt for security issues.

        Args:
            prompt: User input prompt to validate

        Returns:
            ValidationResult with validity, warnings, severity, and sanitized input

        Raises:
            SecurityException: If prompt contains critical security violations
        """
        warnings = []
        severity_counts = {"low": 0, "medium": 0, "high": 0}

        # Check length
        if len(prompt) > self.MAX_PROMPT_LENGTH:
            raise SecurityValidationError(
                f"Prompt exceeds maximum length of {self.MAX_PROMPT_LENGTH} characters",
                severity="medium",
            )

        # Check for empty input
        if not prompt.strip():
            raise SecurityValidationError("Prompt cannot be empty", severity="low")

        # Check for dangerous patterns
        for pattern, message, severity in self.DANGEROUS_PATTERNS:
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            if matches:
                warnings.append(
                    f"{message} (found: {matches[:3]})"
                )  # Limit to 3 examples
                severity_counts[severity] += len(matches)

                logger.warning(
                    "Suspicious pattern detected in prompt",
                    extra={
                        "pattern": pattern,
                        "severity": severity,
                        "matches": matches[:3],
                        "prompt_preview": prompt[:100],
                    },
                )

                # In strict mode, reject immediately on high severity
                if self.strict_mode and severity == "high":
                    # Raise specific exception based on pattern type
                    if "Shell" in message or "command" in message.lower():
                        raise CommandInjectionDetected(message)
                    elif "SQL" in message:
                        raise SQLInjectionError(message)
                    elif "XSS" in message or "script" in message.lower():
                        raise XSSAttemptError(message)
                    elif "exec" in message.lower() or "eval" in message.lower():
                        raise DynamicCodeExecutionError(message)
                    else:
                        raise SecurityValidationError(
                            f"Critical security violation: {message}", severity="high"
                        )

        # Determine overall severity
        if severity_counts["high"] > 0:
            overall_severity = "high"
        elif severity_counts["medium"] > 0:
            overall_severity = "medium"
        else:
            overall_severity = "low"

        # Reject if too many high-severity patterns
        if severity_counts["high"] >= self.HIGH_SEVERITY_THRESHOLD:
            raise SecurityValidationError(
                f"Prompt contains {severity_counts['high']} high-severity security concerns: {warnings}",
                severity="high",
            )

        # Reject if too many medium-severity patterns
        if severity_counts["medium"] >= self.MEDIUM_SEVERITY_THRESHOLD:
            raise SecurityValidationError(
                f"Prompt contains {severity_counts['medium']} medium-severity security concerns: {warnings}",
                severity="medium",
            )

        # Sanitize input (preserve legitimate content)
        sanitized = self._sanitize_prompt(prompt)

        # Log validation result
        if warnings:
            logger.info(
                f"Input validation completed with {len(warnings)} warnings",
                extra={
                    "warning_count": len(warnings),
                    "severity": overall_severity,
                    "prompt_length": len(prompt),
                },
            )

        return ValidationResult(
            is_valid=True,
            warnings=warnings,
            severity=overall_severity,
            sanitized_input=sanitized,
        )

    def _sanitize_prompt(self, prompt: str) -> str:
        """
        Sanitize prompt without breaking legitimate use cases.

        This is intentionally light-touch to avoid breaking legitimate code generation
        requests. The goal is to remove only clearly malicious content.

        Args:
            prompt: Original prompt

        Returns:
            Sanitized prompt
        """
        sanitized = prompt

        # Remove null bytes
        sanitized = sanitized.replace("\x00", "")

        # Remove excessive whitespace
        sanitized = re.sub(r"\s+", " ", sanitized)

        # Trim
        sanitized = sanitized.strip()

        return sanitized

    def validate_file_path(self, file_path: str) -> tuple[bool, list[str]]:
        """
        Validate file path for path traversal and other issues.

        Args:
            file_path: File path to validate

        Returns:
            (is_valid, warnings)

        Raises:
            SecurityException: If path contains dangerous patterns
        """
        warnings = []

        # Check for path traversal
        if ".." in file_path:
            raise PathTraversalAttempt(
                "Path traversal detected in file path", path=file_path
            )

        # Check for absolute paths to sensitive locations
        sensitive_paths = ["/etc/", "/var/", "/root/", "~/.ssh/"]
        for sensitive in sensitive_paths:
            if file_path.startswith(sensitive):
                raise SecurityValidationError(
                    f"Access to sensitive path detected: {sensitive}",
                    severity="high",
                )

        # Check for null bytes
        if "\x00" in file_path:
            raise SecurityValidationError(
                "Null byte detected in file path", severity="high"
            )

        # Warn about absolute paths (may be legitimate)
        if file_path.startswith("/"):
            warnings.append("Absolute path detected (may be legitimate)")

        return True, warnings

    def validate_api_key(self, api_key: str) -> tuple[bool, list[str]]:
        """
        Validate API key format and detect potential issues.

        Args:
            api_key: API key to validate

        Returns:
            (is_valid, warnings)
        """
        warnings = []

        # Check length
        if len(api_key) < 16:
            warnings.append("API key is suspiciously short")

        # Check for obvious test/dummy keys
        dummy_patterns = ["test", "dummy", "fake", "example", "CHANGEME"]
        if any(pattern in api_key.lower() for pattern in dummy_patterns):
            warnings.append("API key appears to be a test/dummy key")

        # Check for spaces (usually indicates misconfiguration)
        if " " in api_key:
            warnings.append("API key contains spaces (possible misconfiguration)")

        return True, warnings
