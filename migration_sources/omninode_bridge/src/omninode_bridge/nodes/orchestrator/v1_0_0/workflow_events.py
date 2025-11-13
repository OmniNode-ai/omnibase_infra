#!/usr/bin/env python3
"""
LlamaIndex Workflow Events for NodeBridgeOrchestrator.

Custom event definitions for event-driven workflow orchestration.
All events inherit from LlamaIndex Event base class and include correlation tracking.

ONEX v2.0 Compliance:
- Strong typing with Pydantic models
- Correlation ID propagation through workflow
- Timestamp tracking for performance analysis
- Namespace support for multi-tenant isolation
"""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from llama_index.core.workflow import Event
from pydantic import Field


class OrchestratorEvent(Event):
    """
    Base event for all orchestrator workflow events.

    Provides common fields for correlation tracking, timing, and namespace isolation.
    All workflow events should inherit from this base class.

    Attributes:
        correlation_id: UUID for request tracing across workflow steps
        timestamp: Event creation timestamp for performance analysis
        namespace: Multi-tenant namespace for event isolation
    """

    correlation_id: UUID = Field(..., description="Workflow correlation ID for tracing")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Event timestamp"
    )
    namespace: str = Field(
        default="omninode.bridge", description="Event namespace for isolation"
    )


# Workflow Lifecycle Events


class InputReceivedEvent(OrchestratorEvent):
    """
    Initial workflow input event.

    Maps to LlamaIndex StartEvent semantics - triggers workflow execution.
    Contains all input data needed for the stamping workflow.

    Attributes:
        content: Content to be hashed and stamped
        metadata: Additional metadata for enrichment
        enable_intelligence: Flag to enable OnexTree intelligence enrichment
    """

    content: str = Field(..., description="Content to be stamped")
    metadata: Optional[dict[str, Any]] = Field(
        default=None, description="Additional metadata"
    )
    enable_intelligence: bool = Field(
        default=False, description="Enable OnexTree intelligence enrichment"
    )


class ValidationCompletedEvent(OrchestratorEvent):
    """
    Input validation completed successfully.

    Emitted after input validation step completes. Contains validated content
    and performance metrics.

    Attributes:
        validated_content: Content after validation
        validation_time_ms: Time taken for validation in milliseconds
    """

    validated_content: str = Field(..., description="Validated content")
    validation_time_ms: float = Field(..., description="Validation time in ms")


class HashGeneratedEvent(OrchestratorEvent):
    """
    BLAKE3 hash generated successfully.

    Emitted after hash generation via MetadataStamping service.
    Contains hash value and generation metrics.

    Attributes:
        file_hash: BLAKE3 hash of content
        hash_generation_time_ms: Time taken for hash generation
        file_size_bytes: Size of content in bytes
    """

    file_hash: str = Field(..., description="BLAKE3 hash of content")
    hash_generation_time_ms: float = Field(
        ..., description="Hash generation time in ms"
    )
    file_size_bytes: int = Field(..., description="Content size in bytes")


class StampCreatedEvent(OrchestratorEvent):
    """
    Metadata stamp created successfully.

    Emitted after stamp creation with O.N.E. v0.1 compliance.
    Contains stamp ID and metadata.

    Attributes:
        stamp_id: Unique stamp identifier
        stamp_data: Stamp metadata dictionary (all string values)
        stamp_creation_time_ms: Time taken for stamp creation
    """

    stamp_id: str = Field(..., description="Unique stamp identifier")
    stamp_data: dict[str, str] = Field(..., description="Stamp metadata")
    stamp_creation_time_ms: float = Field(..., description="Stamp creation time in ms")


class IntelligenceRequestedEvent(OrchestratorEvent):
    """
    Intelligence enrichment requested from OnexTree.

    Emitted when OnexTree intelligence analysis is requested.
    Contains content and context for AI analysis.

    Attributes:
        content: Content to analyze
        file_hash: Hash of content for reference
    """

    content: str = Field(..., description="Content to analyze")
    file_hash: str = Field(..., description="Hash of content")


class IntelligenceReceivedEvent(OrchestratorEvent):
    """
    Intelligence analysis received from OnexTree.

    Emitted after OnexTree intelligence enrichment completes.
    Contains AI analysis results and confidence metrics.

    Attributes:
        intelligence_data: AI analysis results (all string values)
        intelligence_time_ms: Time taken for intelligence analysis
        confidence_score: Confidence score for analysis (0.0-1.0)
    """

    intelligence_data: dict[str, str] = Field(..., description="AI analysis results")
    intelligence_time_ms: float = Field(..., description="Intelligence time in ms")
    confidence_score: float = Field(
        ..., description="Analysis confidence score (0.0-1.0)"
    )


class PersistenceCompletedEvent(OrchestratorEvent):
    """
    Database persistence completed successfully.

    Emitted after workflow state is persisted to database.
    Contains persistence metrics and database ID.

    Attributes:
        persistence_time_ms: Time taken for database persistence
        database_id: Database record identifier
    """

    persistence_time_ms: float = Field(..., description="Persistence time in ms")
    database_id: str = Field(..., description="Database record ID")


class WorkflowCompletedEvent(OrchestratorEvent):
    """
    Workflow completed successfully.

    Maps to LlamaIndex StopEvent semantics - ends workflow execution.
    Contains final workflow results and performance metrics.

    This event is wrapped in StopEvent when returned from workflow.

    Attributes:
        stamp_id: Final stamp identifier
        file_hash: Final file hash
        stamped_content: Content with embedded stamp
        stamp_metadata: Complete stamp metadata
        intelligence_data: Optional AI analysis results
        processing_time_ms: Total workflow processing time
        workflow_steps_executed: Number of steps executed
    """

    stamp_id: str = Field(..., description="Final stamp identifier")
    file_hash: str = Field(..., description="Final file hash")
    stamped_content: str = Field(..., description="Content with embedded stamp")
    stamp_metadata: dict[str, Any] = Field(..., description="Complete stamp metadata")
    intelligence_data: Optional[dict[str, str]] = Field(
        default=None, description="Optional AI analysis results"
    )
    processing_time_ms: float = Field(..., description="Total workflow time in ms")
    workflow_steps_executed: int = Field(..., description="Number of steps executed")


class WorkflowFailedEvent(OrchestratorEvent):
    """
    Workflow failed with error.

    Emitted when workflow execution fails at any step.
    Contains error details for debugging and recovery.

    Attributes:
        error_message: Human-readable error message
        error_type: Exception type name
        failed_step: Step that failed
        processing_time_ms: Time until failure
    """

    error_message: str = Field(..., description="Human-readable error message")
    error_type: str = Field(..., description="Exception type name")
    failed_step: str = Field(..., description="Step that failed")
    processing_time_ms: float = Field(..., description="Time until failure in ms")
