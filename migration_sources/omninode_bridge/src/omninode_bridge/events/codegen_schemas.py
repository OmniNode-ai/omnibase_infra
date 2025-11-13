"""
Event schemas for autonomous code generation infrastructure.

This module defines Pydantic v2 models for events flowing between omniclaude
and omniarchon during the code generation workflow. All schemas support
versioning for backward compatibility.

Event Flow:
    omniclaude → Request Topics → omniarchon (intelligence processing)
    omniarchon → Response Topics → omniclaude (results)
    Both → Status Topics → monitoring/debugging
    NodeTestGeneratorEffect → Test Generation Topics → monitoring/observability

Schema Versioning:
    - Current version: 1.0
    - Evolution strategy: backward_compatible
    - Add new fields as optional for compatibility

Topics (16 total):
    Request Topics (4):
        - omninode_codegen_request_analyze_v1
        - omninode_codegen_request_validate_v1
        - omninode_codegen_request_pattern_v1
        - omninode_codegen_request_mixin_v1

    Response Topics (4):
        - omninode_codegen_response_analyze_v1
        - omninode_codegen_response_validate_v1
        - omninode_codegen_response_pattern_v1
        - omninode_codegen_response_mixin_v1

    Status Topic (1):
        - omninode_codegen_status_session_v1 (6 partitions)

    Test Generation Topics (3):
        - omninode_test_generation_started_v1
        - omninode_test_generation_completed_v1
        - omninode_test_generation_failed_v1

    Dead Letter Queue Topics (4):
        - omninode_codegen_request_analyze_v1_dlq
        - omninode_codegen_request_validate_v1_dlq
        - omninode_codegen_request_pattern_v1_dlq
        - omninode_codegen_request_mixin_v1_dlq
"""

from datetime import UTC, datetime
from typing import Any, ClassVar
from uuid import UUID

from pydantic import Field

from omninode_bridge.events.enums import (
    EnumAnalysisType,
    EnumNodeType,
    EnumSessionStatus,
    EnumValidationType,
)
from omninode_bridge.events.models.base import EventBase


class CodegenAnalysisRequest(EventBase):
    """
    Schema for PRD analysis requests.

    Published to: omninode_codegen_request_analyze_v1
    Flow: omniclaude → omniarchon
    """

    correlation_id: UUID = Field(..., description="Request correlation ID for tracing")
    session_id: UUID = Field(..., description="Code generation session ID")
    prd_content: str = Field(..., description="Raw PRD markdown content")
    analysis_type: EnumAnalysisType = Field(
        default=EnumAnalysisType.FULL,
        description="Type of analysis: full, partial, or quick",
    )
    workspace_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Workspace context including file paths and metadata",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Request timestamp (UTC)",
    )
    schema_version: str = Field(default="1.0", description="Event schema version")


class CodegenAnalysisResponse(EventBase):
    """
    Schema for PRD analysis responses.

    Published to: omninode_codegen_response_analyze_v1
    Flow: omniarchon → omniclaude
    """

    correlation_id: UUID = Field(..., description="Request correlation ID for matching")
    session_id: UUID = Field(..., description="Code generation session ID")
    analysis_result: dict[str, Any] = Field(
        ...,
        description="Semantic analysis results including requirements, architecture, and dependencies",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Analysis confidence score (0.0-1.0)"
    )
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Response timestamp (UTC)",
    )
    schema_version: str = Field(default="1.0", description="Event schema version")


class CodegenValidationRequest(EventBase):
    """
    Schema for code validation requests.

    Published to: omninode_codegen_request_validate_v1
    Flow: omniclaude → omniarchon
    """

    correlation_id: UUID = Field(..., description="Request correlation ID for tracing")
    session_id: UUID = Field(..., description="Code generation session ID")
    code_content: str = Field(..., description="Generated code to validate")
    node_type: EnumNodeType = Field(
        ...,
        description="Type of node: orchestrator, compute, reducer, or effect",
    )
    contracts: list[dict[str, Any]] = Field(
        default_factory=list, description="Associated contracts for compliance checking"
    )
    validation_type: EnumValidationType = Field(
        default=EnumValidationType.FULL,
        description="Type of validation: syntax, semantic, compliance, or full",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Request timestamp (UTC)",
    )
    schema_version: str = Field(default="1.0", description="Event schema version")


class CodegenValidationResponse(EventBase):
    """
    Schema for code validation responses.

    Published to: omninode_codegen_response_validate_v1
    Flow: omniarchon → omniclaude
    """

    correlation_id: UUID = Field(..., description="Request correlation ID for matching")
    session_id: UUID = Field(..., description="Code generation session ID")
    validation_result: dict[str, Any] = Field(
        ...,
        description="Validation results including errors, warnings, and suggestions",
    )
    quality_score: float = Field(
        ..., ge=0.0, le=1.0, description="Overall code quality score (0.0-1.0)"
    )
    onex_compliance_score: float = Field(
        ..., ge=0.0, le=1.0, description="ONEX compliance score (0.0-1.0)"
    )
    is_valid: bool = Field(..., description="Whether code passes all validation checks")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Response timestamp (UTC)",
    )
    schema_version: str = Field(default="1.0", description="Event schema version")


class CodegenPatternRequest(EventBase):
    """
    Schema for pattern matching requests.

    Published to: omninode_codegen_request_pattern_v1
    Flow: omniclaude → omniarchon
    """

    correlation_id: UUID = Field(..., description="Request correlation ID for tracing")
    session_id: UUID = Field(..., description="Code generation session ID")
    node_description: str = Field(
        ..., description="Description of desired node functionality"
    )
    node_type: EnumNodeType = Field(
        ...,
        description="Type of node to find: orchestrator, compute, reducer, or effect",
    )
    limit: int = Field(
        default=5, ge=1, le=20, description="Maximum number of similar nodes to return"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Request timestamp (UTC)",
    )
    schema_version: str = Field(default="1.0", description="Event schema version")


class CodegenPatternResponse(EventBase):
    """
    Schema for pattern matching responses.

    Published to: omninode_codegen_response_pattern_v1
    Flow: omniarchon → omniclaude
    """

    correlation_id: UUID = Field(..., description="Request correlation ID for matching")
    session_id: UUID = Field(..., description="Code generation session ID")
    pattern_result: list[dict[str, Any]] = Field(
        ...,
        description="List of similar nodes with similarity scores and implementation details",
    )
    total_matches: int = Field(
        ..., description="Total number of matches found (before limit)"
    )
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Response timestamp (UTC)",
    )
    schema_version: str = Field(default="1.0", description="Event schema version")


class CodegenMixinRequest(EventBase):
    """
    Schema for mixin recommendation requests.

    Published to: omninode_codegen_request_mixin_v1
    Flow: omniclaude → omniarchon
    """

    correlation_id: UUID = Field(..., description="Request correlation ID for tracing")
    session_id: UUID = Field(..., description="Code generation session ID")
    requirements: list[str] = Field(
        ..., description="List of functional requirements for the node"
    )
    node_type: EnumNodeType = Field(..., description="Type of node being generated")
    existing_mixins: list[str] = Field(
        default_factory=list,
        description="Mixins already selected (to avoid duplicates)",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Request timestamp (UTC)",
    )
    schema_version: str = Field(default="1.0", description="Event schema version")


class CodegenMixinResponse(EventBase):
    """
    Schema for mixin recommendation responses.

    Published to: omninode_codegen_response_mixin_v1
    Flow: omniarchon → omniclaude
    """

    correlation_id: UUID = Field(..., description="Request correlation ID for matching")
    session_id: UUID = Field(..., description="Code generation session ID")
    mixin_recommendations: list[dict[str, Any]] = Field(
        ...,
        description="List of recommended mixins with rationale and implementation guidance",
    )
    total_recommendations: int = Field(
        ..., description="Total number of recommendations made"
    )
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Response timestamp (UTC)",
    )
    schema_version: str = Field(default="1.0", description="Event schema version")


class CodegenStatusEvent(EventBase):
    """
    Schema for code generation status updates.

    Published to: omninode_codegen_status_session_v1
    Flow: Both omniclaude and omniarchon → monitoring/debugging
    """

    session_id: UUID = Field(..., description="Code generation session ID")
    status: EnumSessionStatus = Field(
        ...,
        description="Current status: pending, processing, completed, failed, or cancelled",
    )
    progress_percentage: float = Field(
        ..., ge=0.0, le=100.0, description="Progress percentage (0.0-100.0)"
    )
    message: str = Field(..., description="Human-readable status message")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata including current step, errors, etc.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Event timestamp (UTC)",
    )
    schema_version: str = Field(default="1.0", description="Event schema version")


class ModelEventTestGenerationStarted(EventBase):
    """
    Event: Test generation started.

    Published when NodeTestGeneratorEffect begins generating test files.
    Enables real-time progress tracking and observability of the test
    generation workflow.

    Published to: omninode_test_generation_started_v1
    Flow: NodeTestGeneratorEffect → monitoring/observability
    """

    correlation_id: UUID = Field(
        ..., description="Correlation ID for tracing across workflow"
    )
    workflow_id: UUID | None = Field(
        default=None, description="Parent workflow ID if part of larger orchestration"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Event timestamp (UTC)",
    )

    test_contract_name: str = Field(
        ..., description="Name of the test contract being processed"
    )
    test_types: list[str] = Field(
        ...,
        description="Types of tests to generate (unit, integration, contract, etc.)",
    )
    node_name: str = Field(..., description="Name of the node being tested")
    output_directory: str = Field(
        ..., description="Directory path where test files will be written"
    )

    schema_version: str = Field(default="1.0", description="Event schema version")

    model_config: ClassVar[dict] = {
        "json_schema_extra": {
            "example": {
                "correlation_id": "3f2a1b5c-8d4e-4f1a-9c3b-7e6d5a4c3b2a",
                "workflow_id": "7e8f9a0b-1c2d-3e4f-5a6b-7c8d9e0f1a2b",
                "timestamp": "2025-10-30T12:00:00Z",
                "test_contract_name": "postgres_crud_effect_tests",
                "test_types": ["unit", "integration", "contract"],
                "node_name": "NodePostgresCrudEffect",
                "output_directory": "./generated_nodes/postgres_crud_effect/tests",
                "schema_version": "1.0",
            }
        }
    }


class ModelEventTestGenerationCompleted(EventBase):
    """
    Event: Test generation completed successfully.

    Published when NodeTestGeneratorEffect successfully generates all
    requested test files. Includes quality metrics and file inventory
    for validation and observability.

    Published to: omninode_test_generation_completed_v1
    Flow: NodeTestGeneratorEffect → monitoring/observability
    """

    correlation_id: UUID = Field(
        ..., description="Correlation ID for tracing across workflow"
    )
    workflow_id: UUID | None = Field(
        default=None, description="Parent workflow ID if part of larger orchestration"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Event timestamp (UTC)",
    )

    generated_files: list[str] = Field(
        ..., description="List of generated test file paths (absolute or relative)"
    )
    file_count: int = Field(
        ..., ge=0, description="Total number of test files generated"
    )
    duration_seconds: float = Field(
        ..., ge=0.0, description="Total test generation duration in seconds"
    )
    quality_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Quality score of generated tests (0.0-1.0)",
    )
    test_coverage_estimate: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Estimated test coverage percentage (0.0-100.0)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (test counts by type, assertions generated, etc.)",
    )

    schema_version: str = Field(default="1.0", description="Event schema version")

    model_config: ClassVar[dict] = {
        "json_schema_extra": {
            "example": {
                "correlation_id": "3f2a1b5c-8d4e-4f1a-9c3b-7e6d5a4c3b2a",
                "workflow_id": "7e8f9a0b-1c2d-3e4f-5a6b-7c8d9e0f1a2b",
                "timestamp": "2025-10-30T12:05:23Z",
                "generated_files": [
                    "./generated_nodes/postgres_crud_effect/tests/test_unit.py",
                    "./generated_nodes/postgres_crud_effect/tests/test_integration.py",
                    "./generated_nodes/postgres_crud_effect/tests/test_contract.py",
                ],
                "file_count": 3,
                "duration_seconds": 45.2,
                "quality_score": 0.92,
                "test_coverage_estimate": 85.5,
                "metadata": {
                    "unit_tests": 12,
                    "integration_tests": 8,
                    "contract_tests": 5,
                    "total_assertions": 47,
                },
                "schema_version": "1.0",
            }
        }
    }


class ModelEventTestGenerationFailed(EventBase):
    """
    Event: Test generation failed.

    Published when NodeTestGeneratorEffect encounters an error that
    prevents test generation from completing. Includes detailed error
    information for debugging and alerting.

    Published to: omninode_test_generation_failed_v1
    Flow: NodeTestGeneratorEffect → monitoring/observability/alerting
    """

    correlation_id: UUID = Field(
        ..., description="Correlation ID for tracing across workflow"
    )
    workflow_id: UUID | None = Field(
        default=None, description="Parent workflow ID if part of larger orchestration"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Event timestamp (UTC)",
    )

    error_code: str = Field(
        ..., description="Error code for classification (e.g., TEMPLATE_NOT_FOUND)"
    )
    error_message: str = Field(
        ..., description="Human-readable error message describing the failure"
    )
    stack_trace: str | None = Field(
        default=None, description="Full stack trace if available (for debugging)"
    )
    failed_test_type: str | None = Field(
        default=None,
        description="Specific test type that failed (if failure was during specific test generation)",
    )
    partial_files_generated: list[str] = Field(
        default_factory=list,
        description="List of files that were successfully generated before failure",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context (node name, contract name, input parameters, etc.)",
    )

    schema_version: str = Field(default="1.0", description="Event schema version")

    model_config: ClassVar[dict] = {
        "json_schema_extra": {
            "example": {
                "correlation_id": "3f2a1b5c-8d4e-4f1a-9c3b-7e6d5a4c3b2a",
                "workflow_id": "7e8f9a0b-1c2d-3e4f-5a6b-7c8d9e0f1a2b",
                "timestamp": "2025-10-30T12:03:45Z",
                "error_code": "TEMPLATE_NOT_FOUND",
                "error_message": "Test template file not found: templates/test_unit.py.jinja2",
                "stack_trace": "Traceback (most recent call last):\n  File ...",
                "failed_test_type": "unit",
                "partial_files_generated": [],
                "metadata": {
                    "node_name": "NodePostgresCrudEffect",
                    "test_contract_name": "postgres_crud_effect_tests",
                    "template_path": "templates/test_unit.py.jinja2",
                },
                "schema_version": "1.0",
            }
        }
    }


# ================================
# Kafka Topic Constants
# ================================

# Request topics
TOPIC_CODEGEN_REQUEST_ANALYZE = "dev.omninode-bridge.codegen.request-analyze.v1"
TOPIC_CODEGEN_REQUEST_VALIDATE = "dev.omninode-bridge.codegen.request-validate.v1"
TOPIC_CODEGEN_REQUEST_PATTERN = "dev.omninode-bridge.codegen.request-pattern.v1"
TOPIC_CODEGEN_REQUEST_MIXIN = "dev.omninode-bridge.codegen.request-mixin.v1"

# Response topics
TOPIC_CODEGEN_RESPONSE_ANALYZE = "dev.omninode-bridge.codegen.response-analyze.v1"
TOPIC_CODEGEN_RESPONSE_VALIDATE = "dev.omninode-bridge.codegen.response-validate.v1"
TOPIC_CODEGEN_RESPONSE_PATTERN = "dev.omninode-bridge.codegen.response-pattern.v1"
TOPIC_CODEGEN_RESPONSE_MIXIN = "dev.omninode-bridge.codegen.response-mixin.v1"

# Status topic
TOPIC_CODEGEN_STATUS_SESSION = "dev.omninode-bridge.codegen.status-session.v1"

# Dead letter queue topics
TOPIC_CODEGEN_REQUEST_ANALYZE_DLQ = "dev.omninode-bridge.codegen.request-analyze.v1.dlq"
TOPIC_CODEGEN_REQUEST_VALIDATE_DLQ = (
    "dev.omninode-bridge.codegen.request-validate.v1.dlq"
)
TOPIC_CODEGEN_REQUEST_PATTERN_DLQ = "dev.omninode-bridge.codegen.request-pattern.v1.dlq"
TOPIC_CODEGEN_REQUEST_MIXIN_DLQ = "dev.omninode-bridge.codegen.request-mixin.v1.dlq"

# Test generation topics
TOPIC_TEST_GENERATION_STARTED = "dev.omninode-bridge.test-generation.started.v1"
TOPIC_TEST_GENERATION_COMPLETED = "dev.omninode-bridge.test-generation.completed.v1"
TOPIC_TEST_GENERATION_FAILED = "dev.omninode-bridge.test-generation.failed.v1"
