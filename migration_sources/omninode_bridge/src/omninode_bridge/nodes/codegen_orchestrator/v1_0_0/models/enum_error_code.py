#!/usr/bin/env python3
"""
Error Code Enumeration for NodeCodegenOrchestrator.

Structured error codes for all failure modes in the code generation pipeline.

ONEX v2.0 Compliance:
- Enum-based naming: EnumErrorCode
- Clear categorization of failure modes
- Integration with circuit breaker and retry patterns
"""

from enum import Enum


class EnumErrorCode(str, Enum):
    """
    Structured error codes for code generation failures.

    Error categories:
    - INTELLIGENCE_* - Intelligence/RAG failures
    - VALIDATION_* - Validation failures
    - GENERATION_* - Code generation failures
    - FILE_* - File I/O failures
    - TIMEOUT_* - Timeout failures
    - SYSTEM_* - System/infrastructure failures
    """

    # Intelligence failures
    INTELLIGENCE_UNAVAILABLE = "ONEX_INTELLIGENCE_UNAVAILABLE"
    """Intelligence service unavailable (circuit breaker open or timeout)"""

    INTELLIGENCE_TIMEOUT = "ONEX_INTELLIGENCE_TIMEOUT"
    """Intelligence query exceeded timeout threshold"""

    INTELLIGENCE_DEGRADED = "ONEX_INTELLIGENCE_DEGRADED"
    """Intelligence service degraded (partial results)"""

    INTELLIGENCE_CIRCUIT_OPEN = "ONEX_INTELLIGENCE_CIRCUIT_OPEN"
    """Circuit breaker open - intelligence service unreachable"""

    # Validation failures
    VALIDATION_FAILED = "ONEX_VALIDATION_FAILED"
    """Code validation failed (linting, type checking, or tests)"""

    VALIDATION_QUALITY_BELOW_THRESHOLD = "ONEX_VALIDATION_QUALITY_BELOW_THRESHOLD"
    """Generated code quality score below minimum threshold"""

    VALIDATION_INCOMPLETE = "ONEX_VALIDATION_INCOMPLETE"
    """Validation could not complete (tooling issues)"""

    # Code generation failures
    CODE_GENERATION_FAILED = "ONEX_CODE_GENERATION_FAILED"
    """Code generation step failed (LLM error or template issue)"""

    CODE_GENERATION_TIMEOUT = "ONEX_CODE_GENERATION_TIMEOUT"
    """Code generation exceeded timeout threshold"""

    CODE_GENERATION_INCOMPLETE = "ONEX_CODE_GENERATION_INCOMPLETE"
    """Code generation produced partial results"""

    CONTRACT_GENERATION_FAILED = "ONEX_CONTRACT_GENERATION_FAILED"
    """Contract YAML generation failed"""

    PROMPT_PARSING_FAILED = "ONEX_PROMPT_PARSING_FAILED"
    """Failed to parse user prompt or extract requirements"""

    # File I/O failures
    FILE_WRITE_ERROR = "ONEX_FILE_WRITE_ERROR"
    """Failed to write generated files to disk"""

    FILE_PERMISSION_ERROR = "ONEX_FILE_PERMISSION_ERROR"
    """Insufficient permissions to write to output directory"""

    FILE_DIRECTORY_NOT_FOUND = "ONEX_FILE_DIRECTORY_NOT_FOUND"
    """Output directory does not exist or is not accessible"""

    # Timeout failures
    TIMEOUT_EXCEEDED = "ONEX_TIMEOUT_EXCEEDED"
    """Overall workflow timeout exceeded"""

    STAGE_TIMEOUT_EXCEEDED = "ONEX_STAGE_TIMEOUT_EXCEEDED"
    """Individual stage timeout exceeded"""

    # System failures
    SYSTEM_OUT_OF_MEMORY = "ONEX_SYSTEM_OUT_OF_MEMORY"
    """System ran out of memory during generation"""

    SYSTEM_RESOURCE_EXHAUSTED = "ONEX_SYSTEM_RESOURCE_EXHAUSTED"
    """System resources exhausted (CPU, disk, etc.)"""

    KAFKA_PUBLISH_FAILED = "ONEX_KAFKA_PUBLISH_FAILED"
    """Failed to publish event to Kafka"""

    UNKNOWN_ERROR = "ONEX_UNKNOWN_ERROR"
    """Unknown error occurred"""

    @property
    def is_retryable(self) -> bool:
        """Check if this error type is retryable."""
        retryable_codes = {
            self.INTELLIGENCE_UNAVAILABLE,
            self.INTELLIGENCE_TIMEOUT,
            self.INTELLIGENCE_CIRCUIT_OPEN,
            self.CODE_GENERATION_TIMEOUT,
            self.FILE_WRITE_ERROR,
            self.TIMEOUT_EXCEEDED,
            self.KAFKA_PUBLISH_FAILED,
            self.SYSTEM_RESOURCE_EXHAUSTED,
        }
        return self in retryable_codes

    @property
    def requires_circuit_breaker(self) -> bool:
        """Check if this error type should trigger circuit breaker."""
        circuit_breaker_codes = {
            self.INTELLIGENCE_UNAVAILABLE,
            self.INTELLIGENCE_TIMEOUT,
            self.INTELLIGENCE_CIRCUIT_OPEN,
            self.CODE_GENERATION_FAILED,
            self.CODE_GENERATION_TIMEOUT,
        }
        return self in circuit_breaker_codes

    @property
    def allows_partial_success(self) -> bool:
        """Check if partial success is acceptable for this error."""
        partial_success_codes = {
            self.INTELLIGENCE_DEGRADED,
            self.VALIDATION_QUALITY_BELOW_THRESHOLD,
            self.CODE_GENERATION_INCOMPLETE,
            self.VALIDATION_INCOMPLETE,
        }
        return self in partial_success_codes

    @property
    def severity(self) -> str:
        """Get error severity level."""
        critical_codes = {
            self.SYSTEM_OUT_OF_MEMORY,
            self.SYSTEM_RESOURCE_EXHAUSTED,
            self.FILE_PERMISSION_ERROR,
        }
        high_codes = {
            self.CODE_GENERATION_FAILED,
            self.VALIDATION_FAILED,
            self.CONTRACT_GENERATION_FAILED,
            self.FILE_WRITE_ERROR,
        }
        medium_codes = {
            self.INTELLIGENCE_UNAVAILABLE,
            self.TIMEOUT_EXCEEDED,
            self.CODE_GENERATION_TIMEOUT,
        }

        if self in critical_codes:
            return "CRITICAL"
        elif self in high_codes:
            return "HIGH"
        elif self in medium_codes:
            return "MEDIUM"
        else:
            return "LOW"

    def get_recovery_hint(self) -> str:
        """Get human-readable recovery suggestion."""
        recovery_hints = {
            self.INTELLIGENCE_UNAVAILABLE: "Intelligence service is unavailable. Retrying in 60s. Generation will continue with reduced intelligence.",
            self.INTELLIGENCE_TIMEOUT: "Intelligence query timed out. Retrying with shorter timeout or continuing without intelligence.",
            self.INTELLIGENCE_CIRCUIT_OPEN: "Circuit breaker is open for intelligence service. Will retry after recovery timeout.",
            self.VALIDATION_FAILED: "Code validation failed. Review linting/type checking errors and regenerate.",
            self.VALIDATION_QUALITY_BELOW_THRESHOLD: "Quality score below threshold. Consider regenerating with improved prompt.",
            self.CODE_GENERATION_FAILED: "Code generation failed. Check LLM service status and retry.",
            self.FILE_WRITE_ERROR: "Failed to write files. Check disk space and permissions.",
            self.FILE_PERMISSION_ERROR: "Permission denied. Ensure output directory is writable.",
            self.TIMEOUT_EXCEEDED: "Operation timed out. Consider breaking down into smaller tasks.",
            self.KAFKA_PUBLISH_FAILED: "Failed to publish to Kafka. Event will be retried or saved to database.",
        }
        return recovery_hints.get(
            self, "An error occurred. Please review logs for details."
        )
