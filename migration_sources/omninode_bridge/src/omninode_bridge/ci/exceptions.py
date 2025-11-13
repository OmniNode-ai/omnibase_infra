"""Custom exceptions for the CI workflow system."""

from typing import Any

import yaml


class CIWorkflowError(Exception):
    """Base exception for CI workflow system errors."""

    def __init__(self, message: str, details: Any | None = None):
        """Initialize exception.

        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        """Return string representation."""
        if self.details:
            return f"{self.message}: {self.details}"
        return self.message


class WorkflowValidationError(CIWorkflowError):
    """Raised when workflow validation fails."""

    def __init__(self, message: str, validation_errors: list[str] | None = None):
        """Initialize validation error.

        Args:
            message: Error message
            validation_errors: List of specific validation errors
        """
        super().__init__(message, validation_errors)
        self.validation_errors = validation_errors or []


class WorkflowGenerationError(CIWorkflowError):
    """Raised when workflow generation fails."""

    pass


class TemplateError(CIWorkflowError):
    """Raised when template operations fail."""

    pass


class YAMLProcessingError(CIWorkflowError):
    """Raised when YAML processing fails."""

    def __init__(
        self,
        message: str,
        yaml_error: Exception | None = None,
        line_number: int | None = None,
    ):
        """Initialize YAML processing error.

        Args:
            message: Error message
            yaml_error: Original YAML error
            line_number: Line number where error occurred
        """
        details = {}
        if yaml_error:
            details["yaml_error"] = str(yaml_error)
        if line_number:
            details["line_number"] = str(line_number)

        super().__init__(message, details)
        self.yaml_error = yaml_error
        self.line_number = line_number


class ConfigurationError(CIWorkflowError):
    """Raised when configuration is invalid."""

    pass


class FileOperationError(CIWorkflowError):
    """Raised when file operations fail."""

    def __init__(
        self,
        message: str,
        file_path: str | None = None,
        operation: str | None = None,
    ):
        """Initialize file operation error.

        Args:
            message: Error message
            file_path: Path to file that caused error
            operation: File operation that failed
        """
        details = {}
        if file_path:
            details["file_path"] = file_path
        if operation:
            details["operation"] = operation

        super().__init__(message, details)
        self.file_path = file_path
        self.operation = operation


class DependencyError(CIWorkflowError):
    """Raised when dependency-related errors occur."""

    pass


class SecurityViolationError(CIWorkflowError):
    """Raised when security violations are detected."""

    def __init__(
        self, message: str, violation_type: str | None = None, severity: str = "high"
    ):
        """Initialize security violation error.

        Args:
            message: Error message
            violation_type: Type of security violation
            severity: Severity level (low, medium, high, critical)
        """
        details = {"severity": severity}
        if violation_type:
            details["violation_type"] = violation_type

        super().__init__(message, details)
        self.violation_type = violation_type
        self.severity = severity


class PerformanceError(CIWorkflowError):
    """Raised when performance issues are detected."""

    def __init__(self, message: str, performance_impact: str = "medium"):
        """Initialize performance error.

        Args:
            message: Warning message
            performance_impact: Expected performance impact (low, medium, high)
        """
        super().__init__(message, {"performance_impact": performance_impact})
        self.performance_impact = performance_impact


# Error handling utilities


def handle_yaml_error(yaml_error: Exception, context: str = "") -> YAMLProcessingError:
    """Convert YAML parsing errors to custom exceptions.

    Args:
        yaml_error: Original YAML error
        context: Additional context about where error occurred

    Returns:
        Custom YAML processing error
    """
    message = "YAML processing failed"
    if context:
        message += f" in {context}"

    # Extract line number if available
    line_number = None
    if hasattr(yaml_error, "problem_mark") and yaml_error.problem_mark:
        line_number = yaml_error.problem_mark.line + 1

    return YAMLProcessingError(message, yaml_error, line_number)


def handle_file_error(
    file_error: Exception, file_path: str, operation: str
) -> FileOperationError:
    """Convert file operation errors to custom exceptions.

    Args:
        file_error: Original file error
        file_path: Path to file that caused error
        operation: File operation that failed

    Returns:
        Custom file operation error
    """
    if isinstance(file_error, FileNotFoundError):
        message = f"File not found: {file_path}"
    elif isinstance(file_error, PermissionError):
        message = f"Permission denied accessing file: {file_path}"
    elif isinstance(file_error, IsADirectoryError):
        message = f"Expected file but found directory: {file_path}"
    else:
        message = f"File operation '{operation}' failed for {file_path}: {file_error}"

    return FileOperationError(message, file_path, operation)


def validate_required_fields(
    data: dict, required_fields: list[str], context: str = ""
) -> None:
    """Validate that required fields are present in data.

    Args:
        data: Data dictionary to validate
        required_fields: List of required field names
        context: Context for error messages

    Raises:
        WorkflowValidationError: If required fields are missing
    """
    missing_fields = [
        field for field in required_fields if field not in data or data[field] is None
    ]

    if missing_fields:
        context_str = f" in {context}" if context else ""
        message = f"Missing required fields{context_str}: {', '.join(missing_fields)}"
        raise WorkflowValidationError(message, missing_fields)


def validate_field_types(data: dict, field_types: dict, context: str = "") -> None:
    """Validate field types in data dictionary.

    Args:
        data: Data dictionary to validate
        field_types: Dictionary mapping field names to expected types
        context: Context for error messages

    Raises:
        WorkflowValidationError: If field types are incorrect
    """
    type_errors = []

    for field, expected_type in field_types.items():
        if field in data and data[field] is not None:
            if not isinstance(data[field], expected_type):
                actual_type = type(data[field]).__name__
                expected_type_name = (
                    expected_type.__name__
                    if hasattr(expected_type, "__name__")
                    else str(expected_type)
                )
                type_errors.append(
                    f"{field}: expected {expected_type_name}, got {actual_type}"
                )

    if type_errors:
        context_str = f" in {context}" if context else ""
        message = f"Type validation failed{context_str}"
        raise WorkflowValidationError(message, type_errors)


def safe_yaml_load(yaml_content: str, context: str = "") -> Any:
    """Safely load YAML content with proper error handling.

    Args:
        yaml_content: YAML content to parse
        context: Context for error messages

    Returns:
        Parsed YAML data (can be dict, list, str, etc.)

    Raises:
        YAMLProcessingError: If YAML parsing fails
    """
    import yaml

    try:
        return yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        raise handle_yaml_error(e, context)
    except Exception as e:
        raise YAMLProcessingError(f"Unexpected YAML parsing error: {e}")


def safe_file_read(file_path: str, encoding: str = "utf-8") -> str:
    """Safely read file content with proper error handling.

    Args:
        file_path: Path to file to read
        encoding: File encoding

    Returns:
        File content

    Raises:
        FileOperationError: If file reading fails
    """
    try:
        with open(file_path, encoding=encoding) as f:
            return f.read()
    except Exception as e:
        raise handle_file_error(e, file_path, "read")


def safe_file_write(file_path: str, content: str, encoding: str = "utf-8") -> None:
    """Safely write content to file with proper error handling.

    Args:
        file_path: Path to file to write
        content: Content to write
        encoding: File encoding

    Raises:
        FileOperationError: If file writing fails
    """
    from pathlib import Path

    try:
        # Ensure parent directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding=encoding) as f:
            f.write(content)
    except Exception as e:
        raise handle_file_error(e, file_path, "write")


class ErrorContext:
    """Context manager for enhanced error reporting."""

    def __init__(self, operation: str, context: str = ""):
        """Initialize error context.

        Args:
            operation: Operation being performed
            context: Additional context information
        """
        self.operation = operation
        self.context = context

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and handle exceptions."""
        if exc_type is None:
            return False

        # Re-raise custom exceptions as-is
        if isinstance(exc_val, CIWorkflowError):
            return False

        # Convert common exceptions to custom ones
        if issubclass(exc_type, yaml.YAMLError):
            raise handle_yaml_error(exc_val, self.context)
        elif issubclass(
            exc_type, FileNotFoundError | PermissionError | IsADirectoryError | IOError
        ):
            raise handle_file_error(
                exc_val, getattr(exc_val, "filename", "unknown"), self.operation
            )

        # Wrap other exceptions
        context_str = f" in {self.context}" if self.context else ""
        message = f"Operation '{self.operation}' failed{context_str}: {exc_val}"
        raise CIWorkflowError(
            message,
            {"original_exception": str(exc_val), "exception_type": exc_type.__name__},
        )
