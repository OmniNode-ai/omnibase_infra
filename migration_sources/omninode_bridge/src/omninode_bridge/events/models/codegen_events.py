"""
Event Models for Node Code Generation Workflow

This module defines all event schemas for the contract-first event-driven architecture
enabling parallel development across omninode_bridge, omniclaude, and omniarchon.

All events follow the contract-first principle where event schemas serve as the complete
component specification, eliminating blocking dependencies between teams.
"""

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Literal, Optional
from uuid import UUID, uuid4

from pydantic import Field

from .base import EventBase
from .typed_metrics import ModelPerModelMetrics, ModelPerNodeTypeMetrics

# ================================
# Kafka Topic Constants
# ================================

# Code generation workflow topics (ONEX v2.0 compliant naming)
TOPIC_CODEGEN_REQUESTED = "dev.omninode-bridge.codegen.generation-requested.v1"
TOPIC_CODEGEN_STARTED = "dev.omninode-bridge.codegen.generation-started.v1"
TOPIC_CODEGEN_STAGE_COMPLETED = "dev.omninode-bridge.codegen.stage-completed.v1"
TOPIC_CODEGEN_COMPLETED = "dev.omninode-bridge.codegen.generation-completed.v1"
TOPIC_CODEGEN_FAILED = "dev.omninode-bridge.codegen.generation-failed.v1"

# Metrics aggregation topics
TOPIC_CODEGEN_METRICS_RECORDED = "dev.omninode-bridge.codegen.metrics-recorded.v1"

# Pattern storage topics
TOPIC_PATTERN_STORAGE_REQUESTED = (
    "dev.omniarchon.intelligence.pattern-storage-requested.v1"
)
TOPIC_PATTERN_STORED = "dev.omniarchon.intelligence.pattern-stored.v1"

# Intelligence gathering topics
TOPIC_INTELLIGENCE_QUERY_REQUESTED = "dev.omniarchon.intelligence.query-requested.v1"
TOPIC_INTELLIGENCE_QUERY_COMPLETED = "dev.omniarchon.intelligence.query-completed.v1"

# Orchestration topics
TOPIC_ORCHESTRATOR_CHECKPOINT_REACHED = (
    "dev.omninode-bridge.codegen.checkpoint-reached.v1"
)
TOPIC_ORCHESTRATOR_CHECKPOINT_RESPONSE = (
    "dev.omninode-bridge.codegen.checkpoint-response.v1"
)


# ================================
# Event Envelope
# ================================


class OnexEnvelopeV1(EventBase):
    """
    Standard event envelope for all Kafka events.

    Provides consistent metadata, tracing, and provenance for all events
    in the omninode ecosystem.
    """

    # Envelope metadata
    envelope_version: Literal["v1"] = "v1"
    envelope_id: UUID = Field(default_factory=uuid4)
    envelope_timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Event metadata
    event_type: str = Field(..., description="NODE_GENERATION_REQUESTED|...")
    event_version: Literal["v1"] = "v1"
    correlation_id: UUID

    # Tracing
    trace_id: Optional[UUID] = None
    span_id: Optional[UUID] = None
    parent_span_id: Optional[UUID] = None

    # Source
    source_service: str = Field(..., description="omninode-bridge|omniarchon|...")
    source_node_id: Optional[UUID] = None

    # Payload
    payload: Mapping[str, object] = Field(..., description="Actual event data")

    # Provenance
    provenance: Mapping[str, object] = Field(default_factory=dict)


# ================================
# Node Generation Workflow Events
# ================================


class NodeGenerationRequestedEvent(EventBase):
    """
    Event: User requests node generation via CLI

    Publisher: CLI (cli/generate_node.py)
    Consumers: NodeCodegenOrchestrator
    Topic: omninode.codegen.requested
    """

    # Event metadata
    correlation_id: UUID = Field(..., description="Unique request identifier")
    event_id: UUID = Field(..., description="Unique event identifier")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: Literal["NODE_GENERATION_REQUESTED"] = "NODE_GENERATION_REQUESTED"

    # Request data
    prompt: str = Field(..., description="Natural language node description")
    output_directory: str = Field(..., description="Where to write generated files")
    node_type: Optional[str] = Field(
        None, description="Hint: effect|orchestrator|reducer|compute"
    )

    # User preferences
    interactive_mode: bool = Field(
        default=False, description="Enable interactive checkpoints"
    )
    enable_intelligence: bool = Field(default=True, description="Use RAG intelligence")
    enable_quorum: bool = Field(default=False, description="Use AI quorum validation")

    # Context
    user_id: Optional[str] = Field(None, description="User making request")
    session_id: Optional[UUID] = Field(None, description="Session identifier")

    # Additional configuration and metadata
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional options for code generation workflow",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for tracking and debugging",
    )


class ModelEventCodegenStarted(EventBase):
    """
    Event: Generation workflow started

    Publisher: NodeCodegenOrchestrator
    Consumers: Metrics reducer, monitoring dashboard
    Topic: omninode.codegen.started
    """

    # Event metadata
    correlation_id: UUID
    event_id: UUID
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: Literal["CODEGEN_STARTED"] = "CODEGEN_STARTED"

    # Workflow metadata
    workflow_id: UUID = Field(..., description="Unique workflow identifier")
    orchestrator_node_id: UUID = Field(..., description="Orchestrator instance")

    # Generation parameters
    prompt: str
    output_directory: str
    node_type_hint: Optional[str] = None

    # Pipeline configuration
    pipeline_stages: list[str] = Field(
        default=[
            "prompt_parsing",
            "intelligence_gathering",
            "contract_building",
            "code_generation",
            "event_bus_integration",
            "validation",
            "refinement",
            "file_writing",
        ]
    )
    estimated_duration_seconds: int = Field(default=53, description="Target: 53s")


class ModelEventCodegenStageCompleted(EventBase):
    """
    Event: Individual pipeline stage completed

    Publisher: NodeCodegenOrchestrator
    Consumers: Metrics reducer, progress tracking
    Topic: omninode.codegen.stage_completed
    """

    # Event metadata
    correlation_id: UUID
    event_id: UUID
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: Literal["CODEGEN_STAGE_COMPLETED"] = "CODEGEN_STAGE_COMPLETED"

    # Stage metadata
    workflow_id: UUID
    stage_name: str = Field(
        ..., description="prompt_parsing|intelligence_gathering|..."
    )
    stage_number: int = Field(..., ge=1, le=8)

    # Performance metrics
    duration_seconds: float
    cpu_time_seconds: Optional[float] = None
    memory_mb: Optional[float] = None

    # Stage results
    success: bool
    warnings: list[str] = Field(default_factory=list)
    stage_output: dict[str, Any] = Field(
        default_factory=dict, description="Stage-specific results"
    )


class ModelEventCodegenCompleted(EventBase):
    """
    Event: Node generation completed successfully

    Publisher: NodeCodegenOrchestrator
    Consumers: CLI, metrics reducer, pattern storage
    Topic: omninode.codegen.completed
    """

    # Event metadata
    correlation_id: UUID
    event_id: UUID
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: Literal["CODEGEN_COMPLETED"] = "CODEGEN_COMPLETED"

    # Workflow metadata
    workflow_id: UUID
    total_duration_seconds: float

    # Generated artifacts
    generated_files: list[str] = Field(..., description="List of file paths")
    node_type: str = Field(..., description="effect|orchestrator|reducer|compute")
    service_name: str = Field(..., description="e.g., 'data_services_postgrescrud'")

    # Quality metrics
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Overall quality 0-1")
    test_coverage: Optional[float] = Field(None, ge=0.0, le=1.0)
    complexity_score: Optional[float] = Field(None, description="Cyclomatic complexity")

    # Intelligence usage
    patterns_applied: list[str] = Field(
        default_factory=list, description="RAG patterns used"
    )
    intelligence_sources: list[str] = Field(
        default_factory=list, description="Qdrant|Memgraph|..."
    )

    # Model performance
    primary_model: str = Field(
        ..., description="gemini-2.5-flash|claude-3.5-sonnet|..."
    )
    total_tokens: int
    total_cost_usd: float

    # Files generated
    contract_yaml: str
    node_module: str
    models: list[str]
    enums: list[str]
    tests: list[str]


class ModelEventCodegenFailed(EventBase):
    """
    Event: Node generation failed

    Publisher: NodeCodegenOrchestrator
    Consumers: CLI, metrics reducer, alerting
    Topic: omninode.codegen.failed
    """

    # Event metadata
    correlation_id: UUID
    event_id: UUID
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: Literal["CODEGEN_FAILED"] = "CODEGEN_FAILED"

    # Workflow metadata
    workflow_id: UUID
    failed_stage: str = Field(..., description="Stage where failure occurred")
    partial_duration_seconds: float

    # Error details
    error_code: str = Field(..., description="ONEX error code")
    error_message: str
    error_context: dict[str, Any] = Field(default_factory=dict)
    stack_trace: Optional[str] = None

    # Partial artifacts (if any)
    partial_files: list[str] = Field(default_factory=list)

    # Retry metadata
    retry_count: int = Field(default=0)
    is_retryable: bool = Field(default=True)
    retry_after_seconds: Optional[int] = None


# ================================
# Metrics Aggregation Events
# ================================


class ModelEventMetricsRecorded(EventBase):
    """
    Event: Aggregated metrics recorded

    Publisher: NodeCodegenMetricsReducer
    Consumers: Analytics, dashboard, trend analysis
    Topic: omninode.codegen.metrics_recorded
    """

    # Event metadata
    correlation_id: UUID
    event_id: UUID
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: Literal["CODEGEN_METRICS_RECORDED"] = "CODEGEN_METRICS_RECORDED"

    # Aggregation window
    window_start: datetime
    window_end: datetime
    aggregation_type: str = Field(..., description="hourly|daily|weekly")

    # Performance metrics
    total_generations: int
    successful_generations: int
    failed_generations: int
    avg_duration_seconds: float
    p50_duration_seconds: float
    p95_duration_seconds: float
    p99_duration_seconds: float

    # Quality metrics
    avg_quality_score: float = Field(..., ge=0.0, le=1.0)
    avg_test_coverage: Optional[float] = None

    # Cost metrics
    total_tokens: int
    total_cost_usd: float
    avg_cost_per_generation: float

    # Model performance breakdown
    model_metrics: dict[str, ModelPerModelMetrics] = Field(
        default_factory=dict, description="Per-model performance stats"
    )

    # Node type breakdown
    node_type_metrics: dict[str, ModelPerNodeTypeMetrics] = Field(
        default_factory=dict, description="Per-node-type stats"
    )


# ================================
# Pattern Storage Events
# ================================


class PatternStorageRequestedEvent(EventBase):
    """
    Event: Request to store successful pattern

    Publisher: NodeCodegenOrchestrator (on completion)
    Consumers: Omniarchon pattern storage service
    Topic: dev.omniarchon.intelligence.pattern-storage-requested.v1
    """

    # Event metadata
    correlation_id: UUID
    event_id: UUID
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: Literal["PATTERN_STORAGE_REQUESTED"] = "PATTERN_STORAGE_REQUESTED"

    # Pattern metadata
    pattern_id: UUID = Field(..., description="Unique pattern identifier")
    node_type: str = Field(..., description="effect|orchestrator|reducer|compute")
    domain: str = Field(..., description="database|api|ml|...")

    # Code sample
    code_sample: str = Field(..., description="Primary generated code")
    contract_yaml: str = Field(..., description="Full contract YAML")

    # Quality indicators
    quality_score: float = Field(..., ge=0.0, le=1.0)
    test_coverage: Optional[float] = None
    user_rating: Optional[int] = Field(None, ge=1, le=5)

    # Intelligence metadata
    prompt: str = Field(..., description="Original user prompt")
    business_description: str
    features: list[str]
    best_practices: list[str]

    # Success indicators
    compilation_successful: bool
    tests_passing: bool
    deployed_successfully: Optional[bool] = None

    # Tags for retrieval
    tags: list[str] = Field(default_factory=list)
    embedding_text: str = Field(..., description="Text for semantic search")


class PatternStoredEvent(EventBase):
    """
    Event: Pattern stored successfully

    Publisher: Omniarchon pattern storage service
    Consumers: Metrics reducer, success tracking
    Topic: dev.omniarchon.intelligence.pattern-stored.v1
    """

    # Event metadata
    correlation_id: UUID
    event_id: UUID
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: Literal["PATTERN_STORED"] = "PATTERN_STORED"

    # Pattern metadata
    pattern_id: UUID
    storage_backend: str = Field(..., description="qdrant|memgraph|postgresql")

    # Storage details
    vector_id: Optional[str] = Field(None, description="Qdrant vector ID")
    graph_node_id: Optional[str] = Field(None, description="Memgraph node ID")
    database_row_id: Optional[int] = Field(None, description="PostgreSQL row ID")

    # Indexing metadata
    indexed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    embedding_model: str = Field(..., description="text-embedding-3-large|...")
    embedding_dimensions: int


# ================================
# Intelligence Gathering Events
# ================================


class IntelligenceQueryRequestedEvent(EventBase):
    """
    Event: Request intelligence gathering

    Publisher: NodeCodegenOrchestrator (Stage 1.5)
    Consumers: Omniarchon intelligence service
    Topic: dev.omniarchon.intelligence.query-requested.v1
    """

    # Event metadata
    correlation_id: UUID
    event_id: UUID
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: Literal["INTELLIGENCE_QUERY_REQUESTED"] = "INTELLIGENCE_QUERY_REQUESTED"

    # Query metadata
    query_id: UUID = Field(..., description="Unique query identifier")
    query_type: str = Field(..., description="patterns|best_practices|examples|...")

    # Query parameters
    prompt: str = Field(..., description="User's original prompt")
    node_type: str = Field(..., description="effect|orchestrator|reducer|compute")
    domain: str = Field(..., description="database|api|ml|...")

    # Search parameters
    top_k: int = Field(default=5, ge=1, le=20)
    min_quality_score: float = Field(default=0.8, ge=0.0, le=1.0)

    # Filter criteria
    tags: list[str] = Field(default_factory=list)
    exclude_patterns: list[UUID] = Field(default_factory=list)


class IntelligenceQueryCompletedEvent(EventBase):
    """
    Event: Intelligence query completed

    Publisher: Omniarchon intelligence service
    Consumers: NodeCodegenOrchestrator
    Topic: dev.omniarchon.intelligence.query-completed.v1
    """

    # Event metadata
    correlation_id: UUID
    event_id: UUID
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: Literal["INTELLIGENCE_QUERY_COMPLETED"] = "INTELLIGENCE_QUERY_COMPLETED"

    # Query metadata
    query_id: UUID
    query_duration_ms: float

    # Results
    patterns_found: int
    patterns: list[dict[str, Any]] = Field(
        default_factory=list, description="Retrieved patterns with scores"
    )

    # Sources used
    qdrant_results: int = Field(default=0)
    memgraph_results: int = Field(default=0)
    cache_hits: int = Field(default=0)

    # Recommendations
    best_practices: list[str] = Field(default_factory=list)
    code_examples: list[str] = Field(default_factory=list)
    performance_targets: dict[str, float] = Field(default_factory=dict)


# ================================
# Orchestration & Coordination Events
# ================================


class OrchestratorCheckpointReachedEvent(EventBase):
    """
    Event: Interactive checkpoint reached

    Publisher: NodeCodegenOrchestrator
    Consumers: CLI (interactive mode)
    Topic: dev.omninode-bridge.codegen.checkpoint-reached.v1
    """

    # Event metadata
    correlation_id: UUID
    event_id: UUID
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: Literal["ORCHESTRATOR_CHECKPOINT_REACHED"] = (
        "ORCHESTRATOR_CHECKPOINT_REACHED"
    )

    # Checkpoint metadata
    workflow_id: UUID
    checkpoint_type: str = Field(..., description="contract_review|code_review|...")
    checkpoint_number: int

    # Data to review
    artifacts: dict[str, Any] = Field(..., description="Contract, code, etc.")

    # Validation options
    options: list[str] = Field(
        default=["approve", "regenerate", "edit", "abort"],
        description="Available user actions",
    )

    # Timeout
    timeout_seconds: int = Field(default=300, description="5 minutes default")


class OrchestratorCheckpointResponseEvent(EventBase):
    """
    Event: User responded to checkpoint

    Publisher: CLI (interactive mode)
    Consumers: NodeCodegenOrchestrator
    Topic: dev.omninode-bridge.codegen.checkpoint-response.v1
    """

    # Event metadata
    correlation_id: UUID
    event_id: UUID
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: Literal["ORCHESTRATOR_CHECKPOINT_RESPONSE"] = (
        "ORCHESTRATOR_CHECKPOINT_RESPONSE"
    )

    # Response metadata
    workflow_id: UUID
    checkpoint_type: str
    checkpoint_number: int

    # User decision
    action: str = Field(..., description="approve|regenerate|edit|abort")
    feedback: Optional[str] = Field(None, description="User comments")

    # Edits (if applicable)
    edits: dict[str, Any] = Field(default_factory=dict)


# ================================
# Exports
# ================================

__all__ = [
    # Envelope
    "OnexEnvelopeV1",
    # Node generation events
    "NodeGenerationRequestedEvent",
    "ModelEventCodegenStarted",
    "ModelEventCodegenStageCompleted",
    "ModelEventCodegenCompleted",
    "ModelEventCodegenFailed",
    # Metrics events
    "ModelEventMetricsRecorded",
    # Pattern storage events
    "PatternStorageRequestedEvent",
    "PatternStoredEvent",
    # Intelligence gathering events
    "IntelligenceQueryRequestedEvent",
    "IntelligenceQueryCompletedEvent",
    # Orchestration events
    "OrchestratorCheckpointReachedEvent",
    "OrchestratorCheckpointResponseEvent",
    # Topic constants (updated to TOPIC_CODEGEN_* naming)
    "TOPIC_CODEGEN_REQUESTED",
    "TOPIC_CODEGEN_STARTED",
    "TOPIC_CODEGEN_STAGE_COMPLETED",
    "TOPIC_CODEGEN_COMPLETED",
    "TOPIC_CODEGEN_FAILED",
    "TOPIC_CODEGEN_METRICS_RECORDED",
    "TOPIC_PATTERN_STORAGE_REQUESTED",
    "TOPIC_PATTERN_STORED",
    "TOPIC_INTELLIGENCE_QUERY_REQUESTED",
    "TOPIC_INTELLIGENCE_QUERY_COMPLETED",
    "TOPIC_ORCHESTRATOR_CHECKPOINT_REACHED",
    "TOPIC_ORCHESTRATOR_CHECKPOINT_RESPONSE",
]
