#!/usr/bin/env python3
"""
Security Exceptions for OmniNode Bridge.

Defines specific exception types for different security violations to enable
precise error handling and security monitoring.

ONEX v2.0 Compliance:
- Type-specific exceptions for different threats
- Hierarchical exception structure
- Audit-friendly error messages
- Integration with security monitoring systems
"""

from typing import Optional

# Base Security Exceptions


class SecurityValidationError(Exception):
    """
    Base exception for all security validation failures.

    This is the root exception class for security-related issues.
    Catch this to handle all security validation errors generically.
    """

    def __init__(
        self,
        message: str,
        details: Optional[dict] = None,
        severity: str = "medium",
    ):
        """
        Initialize security validation error.

        Args:
            message: Human-readable error message
            details: Additional context about the error
            severity: Severity level ("low", "medium", "high", "critical")
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.severity = severity

    def __str__(self) -> str:
        """Return formatted error message."""
        if self.details:
            return f"{self.message} (details: {self.details})"
        return self.message

    def to_dict(self) -> dict:
        """Convert exception to dictionary for logging/API responses."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "severity": self.severity,
        }


# Input Validation Exceptions


class InputValidationError(SecurityValidationError):
    """Base exception for input validation failures."""

    def __init__(
        self,
        message: str,
        input_value: Optional[str] = None,
        path: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize input validation error.

        Args:
            message: Error message
            input_value: The invalid input (may be sanitized for logging)
            path: Path related to the error (if applicable)
            **kwargs: Additional arguments for parent class
        """
        super().__init__(message, **kwargs)
        # Store only preview of input for security (don't log full sensitive data)
        if input_value:
            self.details["input_preview"] = input_value[:50] + "..."
        if path:
            self.details["path"] = path


class CommandInjectionError(InputValidationError):
    """Detected command injection attempt in user input."""

    def __init__(self, message: str = "Command injection detected", **kwargs):
        """Initialize command injection exception."""
        super().__init__(message, severity="critical", **kwargs)


class PathTraversalError(InputValidationError):
    """Detected path traversal attempt in user input."""

    def __init__(self, message: str = "Path traversal attempt detected", **kwargs):
        """Initialize path traversal exception."""
        super().__init__(message, severity="high", **kwargs)


# Aliases for backward compatibility with test expectations
CommandInjectionDetected = CommandInjectionError
"""Alias for CommandInjectionError - raised when command injection is detected in user input."""

PathTraversalAttempt = PathTraversalError
"""Alias for PathTraversalError - raised when path traversal attempt is detected."""


class SQLInjectionError(InputValidationError):
    """Detected SQL injection attempt in user input."""

    def __init__(self, message: str = "SQL injection pattern detected", **kwargs):
        """Initialize SQL injection exception."""
        super().__init__(message, severity="critical", **kwargs)


class XSSAttemptError(InputValidationError):
    """Detected XSS (Cross-Site Scripting) attempt in user input."""

    def __init__(self, message: str = "XSS attempt detected", **kwargs):
        """Initialize XSS exception."""
        super().__init__(message, severity="high", **kwargs)


class DynamicCodeExecutionError(InputValidationError):
    """Detected dynamic code execution attempt (eval, exec, etc.)."""

    def __init__(
        self, message: str = "Dynamic code execution attempt detected", **kwargs
    ):
        """Initialize dynamic code execution exception."""
        super().__init__(message, severity="critical", **kwargs)


class ReDoSError(InputValidationError):
    """Detected Regular Expression Denial of Service (ReDoS) vulnerability."""

    def __init__(
        self,
        message: str = "ReDoS vulnerability detected in regex pattern",
        pattern: Optional[str] = None,
        **kwargs,
    ):
        """Initialize ReDoS exception."""
        super().__init__(message, severity="high", **kwargs)
        if pattern:
            # Store only preview of pattern for security
            self.details["pattern_preview"] = pattern[:100] + "..."


# Output Validation Exceptions


class OutputValidationError(SecurityValidationError):
    """Base exception for output validation failures."""

    def __init__(
        self,
        message: str,
        code_snippet: Optional[str] = None,
        line_number: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize output validation error.

        Args:
            message: Error message
            code_snippet: The problematic code
            line_number: Line number of the issue
            **kwargs: Additional arguments for parent class
        """
        super().__init__(message, **kwargs)
        if code_snippet:
            self.details["code_snippet"] = code_snippet[:100] + "..."
        if line_number:
            self.details["line_number"] = line_number


class MaliciousCodeError(OutputValidationError):
    """Detected malicious code in generated output."""

    def __init__(self, message: str = "Malicious code detected", **kwargs):
        """Initialize malicious code exception."""
        super().__init__(message, severity="critical", **kwargs)


class DangerousImportError(OutputValidationError):
    """Detected dangerous import in generated code."""

    def __init__(
        self,
        message: str = "Dangerous import detected",
        import_name: Optional[str] = None,
        **kwargs,
    ):
        """Initialize dangerous import exception."""
        super().__init__(message, severity="high", **kwargs)
        if import_name:
            self.details["import_name"] = import_name


class DangerousFunctionCallError(OutputValidationError):
    """Detected dangerous function call in generated code."""

    def __init__(
        self,
        message: str = "Dangerous function call detected",
        function_name: Optional[str] = None,
        **kwargs,
    ):
        """Initialize dangerous function call exception."""
        super().__init__(message, severity="high", **kwargs)
        if function_name:
            self.details["function_name"] = function_name


class HardcodedCredentialsError(OutputValidationError):
    """Detected hardcoded credentials in generated code."""

    def __init__(
        self,
        message: str = "Hardcoded credentials detected",
        credential_type: Optional[str] = None,
        **kwargs,
    ):
        """Initialize hardcoded credentials exception."""
        super().__init__(message, severity="critical", **kwargs)
        if credential_type:
            self.details["credential_type"] = credential_type


class SensitivePathAccessError(OutputValidationError):
    """Detected access to sensitive file system paths."""

    def __init__(
        self,
        message: str = "Sensitive path access detected",
        path: Optional[str] = None,
        **kwargs,
    ):
        """Initialize sensitive path access exception."""
        super().__init__(message, severity="medium", **kwargs)
        if path:
            self.details["path"] = path


# Path Validation Exceptions


class PathValidationError(SecurityValidationError):
    """Base exception for path validation failures."""

    def __init__(
        self,
        message: str,
        path: Optional[str] = None,
        allowed_dirs: Optional[list] = None,
        **kwargs,
    ):
        """
        Initialize path validation error.

        Args:
            message: Error message
            path: The invalid path
            allowed_dirs: List of allowed directories (if applicable)
            **kwargs: Additional arguments for parent class
        """
        super().__init__(message, **kwargs)
        if path:
            self.details["path"] = path
        if allowed_dirs:
            self.details["allowed_directories"] = allowed_dirs


class PathNotAllowedError(PathValidationError):
    """Path is not within allowed directories."""

    def __init__(
        self,
        message: str = "Path is not within allowed directories",
        path: Optional[str] = None,
        allowed_dirs: Optional[list] = None,
        **kwargs,
    ):
        """Initialize path not allowed exception."""
        # Pass both path and allowed_dirs to parent
        super().__init__(
            message, path=path, allowed_dirs=allowed_dirs, severity="high", **kwargs
        )


class BlockedPathAccessError(PathValidationError):
    """Attempted access to blocked/sensitive path."""

    def __init__(
        self,
        message: str = "Cannot access blocked path",
        path: Optional[str] = None,
        **kwargs,
    ):
        """Initialize blocked path access exception."""
        super().__init__(message, path=path, severity="critical", **kwargs)


class NullByteInPathError(PathValidationError):
    """Detected null byte in path (potential null byte injection)."""

    def __init__(
        self,
        message: str = "Null byte detected in path",
        path: Optional[str] = None,
        **kwargs,
    ):
        """Initialize null byte in path exception."""
        super().__init__(message, path=path, severity="high", **kwargs)


# API Key / Credentials Exceptions


class CredentialValidationError(SecurityValidationError):
    """Base exception for credential validation failures."""

    pass


class InvalidAPIKeyError(CredentialValidationError):
    """API key is invalid or malformed."""

    def __init__(self, message: str = "Invalid API key", **kwargs):
        """Initialize invalid API key exception."""
        super().__init__(message, severity="medium", **kwargs)


class WeakCredentialError(CredentialValidationError):
    """Credential does not meet security requirements."""

    def __init__(
        self, message: str = "Credential does not meet security requirements", **kwargs
    ):
        """Initialize weak credential exception."""
        super().__init__(message, severity="medium", **kwargs)


# Rate Limiting Exceptions


class RateLimitExceededError(SecurityValidationError):
    """Rate limit exceeded for security protection."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        limit: Optional[int] = None,
        window: Optional[int] = None,
        **kwargs,
    ):
        """Initialize rate limit exception."""
        super().__init__(message, severity="low", **kwargs)
        if limit:
            self.details["limit"] = limit
        if window:
            self.details["window_seconds"] = window


# Authentication Exceptions


class AuthenticationError(SecurityValidationError):
    """Base exception for authentication failures."""

    def __init__(self, message: str = "Authentication failed", **kwargs):
        """Initialize authentication error."""
        super().__init__(message, severity="medium", **kwargs)


class InvalidTokenError(AuthenticationError):
    """Token is invalid, expired, or malformed."""

    def __init__(self, message: str = "Invalid or expired token", **kwargs):
        """Initialize invalid token exception."""
        super().__init__(message, severity="medium", **kwargs)


class UnauthorizedAccessError(AuthenticationError):
    """User/service is not authorized to perform this action."""

    def __init__(self, message: str = "Unauthorized access", **kwargs):
        """Initialize unauthorized access exception."""
        super().__init__(message, severity="medium", **kwargs)


# Utility Functions


def create_security_exception(
    error_type: str,
    message: str,
    **kwargs,
) -> SecurityValidationError:
    """
    Factory function to create appropriate security exception.

    Args:
        error_type: Type of security error
        message: Error message
        **kwargs: Additional arguments for exception

    Returns:
        Appropriate SecurityValidationError subclass instance
    """
    exception_map = {
        "command_injection": CommandInjectionError,
        "path_traversal": PathTraversalError,
        "sql_injection": SQLInjectionError,
        "xss": XSSAttemptError,
        "dynamic_code": DynamicCodeExecutionError,
        "redos": ReDoSError,
        "malicious_code": MaliciousCodeError,
        "dangerous_import": DangerousImportError,
        "dangerous_function": DangerousFunctionCallError,
        "hardcoded_credentials": HardcodedCredentialsError,
        "sensitive_path": SensitivePathAccessError,
        "path_not_allowed": PathNotAllowedError,
        "blocked_path": BlockedPathAccessError,
        "null_byte": NullByteInPathError,
        "invalid_api_key": InvalidAPIKeyError,
        "weak_credential": WeakCredentialError,
        "rate_limit": RateLimitExceededError,
        "authentication": AuthenticationError,
        "invalid_token": InvalidTokenError,
        "unauthorized": UnauthorizedAccessError,
    }

    exception_class = exception_map.get(error_type, SecurityValidationError)
    return exception_class(message, **kwargs)


# Exception Groups for Batch Validation


class MultipleSecurityViolationsError(SecurityValidationError):
    """
    Multiple security violations detected.

    Use this when multiple security issues are found and you want to
    report them all at once rather than failing on the first one.
    """

    def __init__(
        self,
        message: str = "Multiple security violations detected",
        violations: Optional[list[SecurityValidationError]] = None,
        **kwargs,
    ):
        """
        Initialize multiple security violations exception.

        Args:
            message: Error message
            violations: List of individual violation exceptions
            **kwargs: Additional arguments for parent class
        """
        super().__init__(message, severity="high", **kwargs)
        self.violations = violations or []
        self.details["violation_count"] = len(self.violations)
        self.details["violation_types"] = [type(v).__name__ for v in self.violations]

    def get_violations_by_severity(
        self, severity: str
    ) -> list[SecurityValidationError]:
        """Get all violations of a specific severity level."""
        return [v for v in self.violations if v.severity == severity]

    def has_critical_violations(self) -> bool:
        """Check if any violations are critical severity."""
        return any(v.severity == "critical" for v in self.violations)

    def get_highest_severity(self) -> str:
        """Get the highest severity level among all violations."""
        severity_order = ["low", "medium", "high", "critical"]
        max_severity = "low"

        for violation in self.violations:
            if severity_order.index(violation.severity) > severity_order.index(
                max_severity
            ):
                max_severity = violation.severity

        return max_severity


__all__ = [
    # Base exceptions
    "SecurityValidationError",
    # Input validation
    "InputValidationError",
    "CommandInjectionError",
    "CommandInjectionDetected",  # Alias for CommandInjectionError
    "PathTraversalError",
    "PathTraversalAttempt",  # Alias for PathTraversalError
    "SQLInjectionError",
    "XSSAttemptError",
    "DynamicCodeExecutionError",
    "ReDoSError",
    # Output validation
    "OutputValidationError",
    "MaliciousCodeError",
    "DangerousImportError",
    "DangerousFunctionCallError",
    "HardcodedCredentialsError",
    "SensitivePathAccessError",
    # Path validation
    "PathValidationError",
    "PathNotAllowedError",
    "BlockedPathAccessError",
    "NullByteInPathError",
    # Credentials
    "CredentialValidationError",
    "InvalidAPIKeyError",
    "WeakCredentialError",
    # Rate limiting
    "RateLimitExceededError",
    # Authentication
    "AuthenticationError",
    "InvalidTokenError",
    "UnauthorizedAccessError",
    # Utilities
    "create_security_exception",
    "MultipleSecurityViolationsError",
]
