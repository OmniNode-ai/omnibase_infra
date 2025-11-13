"""
Validation models for code generation workflow.

Provides data models for validation results, context, and error tracking.
Compliant with ONEX v2.0 standards.

Performance:
- Model instantiation: <1ms
- Model validation: <1ms
- Serialization: <5ms

Example:
    ```python
    result = ValidationResult(
        validator_name="CompletenessValidator",
        passed=True,
        score=0.95,
        errors=[],
        warnings=["Missing docstring in method foo"],
        metadata={"checked_methods": 10},
        duration_ms=45.2
    )

    context = ValidationContext(
        code_type="node",
        node_type="effect",
        contract_name="DataFetchContract",
        expected_patterns=["async def", "ModelOnexError"],
        metadata={"phase": "generation"}
    )
    ```
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4


class EnumValidationType(str, Enum):
    """Validation type enumeration."""

    COMPLETENESS = "completeness"
    QUALITY = "quality"
    ONEX_COMPLIANCE = "onex_compliance"
    SECURITY = "security"
    PERFORMANCE = "performance"


class EnumValidationSeverity(str, Enum):
    """Validation issue severity."""

    CRITICAL = "critical"  # Validation failure, blocks deployment
    ERROR = "error"  # Validation failure, should be fixed
    WARNING = "warning"  # Validation concern, consider fixing
    INFO = "info"  # Informational, no action required


@dataclass
class ValidationIssue:
    """
    Individual validation issue.

    Attributes:
        issue_id: Unique issue identifier
        severity: Issue severity level
        message: Human-readable issue description
        line_number: Line number where issue occurs (if applicable)
        column: Column number where issue occurs (if applicable)
        code_snippet: Relevant code snippet (if applicable)
        rule_name: Validation rule that detected the issue
        suggestion: Suggested fix (if available)
        metadata: Additional issue metadata

    Example:
        ```python
        issue = ValidationIssue(
            severity=EnumValidationSeverity.ERROR,
            message="Missing required method 'execute_effect'",
            line_number=45,
            rule_name="required_methods",
            suggestion="Add async def execute_effect(self, input_data) method"
        )
        ```
    """

    issue_id: str = field(default_factory=lambda: str(uuid4()))
    severity: EnumValidationSeverity = EnumValidationSeverity.ERROR
    message: str = ""
    line_number: Optional[int] = None
    column: Optional[int] = None
    code_snippet: Optional[str] = None
    rule_name: Optional[str] = None
    suggestion: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate issue data."""
        if not self.message:
            raise ValueError("message cannot be empty")


@dataclass
class ValidationResult:
    """
    Result of a single validator execution.

    Attributes:
        validator_name: Name of validator that produced this result
        validation_type: Type of validation performed
        passed: Whether validation passed
        score: Validation score (0.0-1.0, higher is better)
        errors: List of error messages (blocking issues)
        warnings: List of warning messages (non-blocking issues)
        issues: Structured validation issues with details
        metadata: Additional validation metadata
        duration_ms: Validation duration in milliseconds
        timestamp: Validation timestamp
        correlation_id: Optional correlation ID for tracing

    Example:
        ```python
        result = ValidationResult(
            validator_name="CompletenessValidator",
            validation_type=EnumValidationType.COMPLETENESS,
            passed=True,
            score=0.95,
            errors=[],
            warnings=["Missing docstring in method foo"],
            issues=[
                ValidationIssue(
                    severity=EnumValidationSeverity.WARNING,
                    message="Missing docstring",
                    line_number=45
                )
            ],
            metadata={"checked_methods": 10, "missing_docstrings": 1},
            duration_ms=45.2
        )
        ```
    """

    validator_name: str
    validation_type: EnumValidationType
    passed: bool
    score: float = 0.0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate result data."""
        if not self.validator_name:
            raise ValueError("validator_name cannot be empty")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be between 0.0 and 1.0, got {self.score}")
        if self.duration_ms < 0:
            raise ValueError(
                f"duration_ms must be non-negative, got {self.duration_ms}"
            )

    @property
    def has_errors(self) -> bool:
        """Check if validation has errors."""
        return len(self.errors) > 0 or any(
            issue.severity
            in (EnumValidationSeverity.CRITICAL, EnumValidationSeverity.ERROR)
            for issue in self.issues
        )

    @property
    def has_warnings(self) -> bool:
        """Check if validation has warnings."""
        return len(self.warnings) > 0 or any(
            issue.severity == EnumValidationSeverity.WARNING for issue in self.issues
        )

    @property
    def critical_issues(self) -> list[ValidationIssue]:
        """Get critical issues."""
        return [
            issue
            for issue in self.issues
            if issue.severity == EnumValidationSeverity.CRITICAL
        ]

    @property
    def error_issues(self) -> list[ValidationIssue]:
        """Get error issues."""
        return [
            issue
            for issue in self.issues
            if issue.severity == EnumValidationSeverity.ERROR
        ]

    @property
    def warning_issues(self) -> list[ValidationIssue]:
        """Get warning issues."""
        return [
            issue
            for issue in self.issues
            if issue.severity == EnumValidationSeverity.WARNING
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "validator_name": self.validator_name,
            "validation_type": self.validation_type.value,
            "passed": self.passed,
            "score": self.score,
            "errors": self.errors,
            "warnings": self.warnings,
            "issues": [
                {
                    "issue_id": issue.issue_id,
                    "severity": issue.severity.value,
                    "message": issue.message,
                    "line_number": issue.line_number,
                    "column": issue.column,
                    "code_snippet": issue.code_snippet,
                    "rule_name": issue.rule_name,
                    "suggestion": issue.suggestion,
                    "metadata": issue.metadata,
                }
                for issue in self.issues
            ],
            "metadata": self.metadata,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
        }


@dataclass
class ValidationContext:
    """
    Context for validation execution.

    Provides additional information to validators about what is being validated
    and what patterns/rules should be applied.

    Attributes:
        code_type: Type of code being validated (node, contract, test, etc.)
        node_type: Node type (effect, compute, reducer, orchestrator) if code_type=node
        contract_name: Contract name (if applicable)
        expected_patterns: List of expected patterns/imports
        required_methods: List of required method names
        quality_threshold: Minimum quality score required
        strict_mode: Enable strict validation (treat warnings as errors)
        metadata: Additional context metadata
        correlation_id: Optional correlation ID for tracing

    Example:
        ```python
        context = ValidationContext(
            code_type="node",
            node_type="effect",
            contract_name="DataFetchContract",
            expected_patterns=["async def", "ModelOnexError", "emit_log_event"],
            required_methods=["execute_effect", "validate_input"],
            quality_threshold=0.8,
            strict_mode=False,
            metadata={"phase": "generation", "generator": "ContractInferencer"}
        )
        ```
    """

    code_type: str = "node"
    node_type: Optional[str] = None
    contract_name: Optional[str] = None
    expected_patterns: list[str] = field(default_factory=list)
    required_methods: list[str] = field(default_factory=list)
    quality_threshold: float = 0.8
    strict_mode: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate context data."""
        if not 0.0 <= self.quality_threshold <= 1.0:
            raise ValueError(
                f"quality_threshold must be between 0.0 and 1.0, got {self.quality_threshold}"
            )


@dataclass
class ValidationSummary:
    """
    Summary of all validation results.

    Aggregates results from multiple validators into a single summary.

    Attributes:
        total_validators: Total number of validators executed
        passed_validators: Number of validators that passed
        failed_validators: Number of validators that failed
        overall_score: Overall validation score (average of all scores)
        total_errors: Total number of errors across all validators
        total_warnings: Total number of warnings across all validators
        total_duration_ms: Total validation duration in milliseconds
        results: Individual validation results
        timestamp: Summary timestamp
        correlation_id: Optional correlation ID for tracing

    Example:
        ```python
        summary = ValidationSummary(
            total_validators=3,
            passed_validators=2,
            failed_validators=1,
            overall_score=0.73,
            total_errors=5,
            total_warnings=12,
            total_duration_ms=456.7,
            results=[result1, result2, result3]
        )
        ```
    """

    total_validators: int
    passed_validators: int
    failed_validators: int
    overall_score: float
    total_errors: int
    total_warnings: int
    total_duration_ms: float
    results: dict[str, ValidationResult]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate summary data."""
        if self.total_validators < 0:
            raise ValueError(
                f"total_validators must be non-negative, got {self.total_validators}"
            )
        if not 0.0 <= self.overall_score <= 1.0:
            raise ValueError(
                f"overall_score must be between 0.0 and 1.0, got {self.overall_score}"
            )

    @property
    def passed(self) -> bool:
        """Check if all validations passed."""
        return self.failed_validators == 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_validators == 0:
            return 0.0
        return self.passed_validators / self.total_validators

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_validators": self.total_validators,
            "passed_validators": self.passed_validators,
            "failed_validators": self.failed_validators,
            "overall_score": self.overall_score,
            "total_errors": self.total_errors,
            "total_warnings": self.total_warnings,
            "total_duration_ms": self.total_duration_ms,
            "results": {
                name: result.to_dict() for name, result in self.results.items()
            },
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "passed": self.passed,
            "success_rate": self.success_rate,
        }
