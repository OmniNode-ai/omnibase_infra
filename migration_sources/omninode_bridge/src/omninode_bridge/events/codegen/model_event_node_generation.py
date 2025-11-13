#!/usr/bin/env python3
"""
Node Generation Event Models - Contract-First Development.

All event schemas for the code generation MVP workflow.
Enables parallel development across omninode_bridge, omniclaude, and omniarchon.

ONEX v2.0 Compliance:
- Model prefix naming: ModelEvent*
- Pydantic v2 validation
- UUID correlation tracking
- Timestamp tracking for all events
"""

from datetime import UTC, datetime
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import Field

from ..models.base import EventBase

# =============================================================================
# 1. Node Generation Workflow Events
# =============================================================================


class ModelEventNodeGenerationRequested(EventBase):
    """
    Event: User requests node generation via CLI.

    Publisher: CLI (cli/generate_node.py)
    Consumers: NodeCodegenOrchestrator
    Topic: dev.omninode-bridge.codegen.generation-requested.v1
    """

    # Event metadata
    correlation_id: UUID = Field(
        default_factory=uuid4, description="Unique request identifier"
    )
    event_id: UUID = Field(default_factory=uuid4, description="Unique event identifier")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: str = Field(
        default="NODE_GENERATION_REQUESTED", description="Event type identifier"
    )

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


class ModelEventNodeGenerationStarted(EventBase):
    """
    Event: Orchestrator begins generation workflow.

    Publisher: NodeCodegenOrchestrator
    Consumers: Metrics reducer, monitoring dashboard
    Topic: dev.omninode-bridge.codegen.generation-started.v1
    """

    # Event metadata
    correlation_id: UUID = Field(..., description="Unique request identifier")
    event_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: str = Field(default="NODE_GENERATION_STARTED")

    # Workflow metadata
    workflow_id: UUID = Field(..., description="Unique workflow identifier")
    orchestrator_node_id: UUID = Field(..., description="Orchestrator instance")

    # Generation parameters
    prompt: str
    output_directory: str
    node_type_hint: Optional[str]

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


class ModelEventNodeGenerationStageCompleted(EventBase):
    """
    Event: Pipeline stage completes.

    Publisher: NodeCodegenOrchestrator
    Consumers: Metrics reducer, progress tracking
    Topic: dev.omninode-bridge.codegen.stage-completed.v1
    """

    # Event metadata
    correlation_id: UUID
    event_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: str = Field(default="NODE_GENERATION_STAGE_COMPLETED")

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


class ModelEventNodeGenerationCompleted(EventBase):
    """
    Event: Node generation completed successfully.

    Publisher: NodeCodegenOrchestrator
    Consumers: CLI, metrics reducer, pattern storage
    Topic: dev.omninode-bridge.codegen.generation-completed.v1
    """

    # Event metadata
    correlation_id: UUID
    event_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: str = Field(default="NODE_GENERATION_COMPLETED")

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


class ModelEventNodeGenerationFailed(EventBase):
    """
    Event: Node generation failed.

    Publisher: NodeCodegenOrchestrator
    Consumers: CLI, metrics reducer, alerting
    Topic: dev.omninode-bridge.codegen.generation-failed.v1
    """

    # Event metadata
    correlation_id: UUID
    event_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: str = Field(default="NODE_GENERATION_FAILED")

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


# =============================================================================
# 2. Metrics Aggregation Events
# =============================================================================


class ModelEventGenerationMetricsRecorded(EventBase):
    """
    Event: Aggregated metrics recorded.

    Publisher: NodeCodegenMetricsReducer
    Consumers: Analytics, dashboard, trend analysis
    Topic: dev.omninode-bridge.codegen.metrics-recorded.v1
    """

    # Event metadata
    correlation_id: UUID
    event_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: str = Field(default="GENERATION_METRICS_RECORDED")

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
    model_metrics: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Per-model performance stats"
    )

    # Node type breakdown
    node_type_metrics: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Per-node-type stats"
    )


# =============================================================================
# 3. Pattern Storage Events
# =============================================================================


class ModelEventPatternStorageRequested(EventBase):
    """
    Event: Request to store successful pattern.

    Publisher: NodeCodegenOrchestrator (on completion)
    Consumers: Omniarchon pattern storage service
    Topic: dev.omniarchon.intelligence.pattern-storage-requested.v1
    """

    # Event metadata
    correlation_id: UUID
    event_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: str = Field(default="PATTERN_STORAGE_REQUESTED")

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


class ModelEventPatternStored(EventBase):
    """
    Event: Pattern successfully stored.

    Publisher: Omniarchon pattern storage service
    Consumers: Metrics reducer, success tracking
    Topic: dev.omniarchon.intelligence.pattern-stored.v1
    """

    # Event metadata
    correlation_id: UUID
    event_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: str = Field(default="PATTERN_STORED")

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


# =============================================================================
# 4. Intelligence Gathering Events
# =============================================================================


class ModelEventIntelligenceQueryRequested(EventBase):
    """
    Event: Request RAG intelligence for generation.

    Publisher: NodeCodegenOrchestrator (Stage 1.5)
    Consumers: Omniarchon intelligence service
    Topic: dev.omniarchon.intelligence.query-requested.v1
    """

    # Event metadata
    correlation_id: UUID
    event_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: str = Field(default="INTELLIGENCE_QUERY_REQUESTED")

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


class ModelEventIntelligenceQueryCompleted(EventBase):
    """
    Event: Intelligence gathering complete.

    Publisher: Omniarchon intelligence service
    Consumers: NodeCodegenOrchestrator
    Topic: dev.omniarchon.intelligence.query-completed.v1
    """

    # Event metadata
    correlation_id: UUID
    event_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: str = Field(default="INTELLIGENCE_QUERY_COMPLETED")

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


# =============================================================================
# 5. Orchestration & Coordination Events
# =============================================================================


class ModelEventOrchestratorCheckpointReached(EventBase):
    """
    Event: Interactive checkpoint for user validation.

    Publisher: NodeCodegenOrchestrator
    Consumers: CLI (interactive mode)
    Topic: dev.omninode-bridge.codegen.checkpoint-reached.v1
    """

    # Event metadata
    correlation_id: UUID
    event_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: str = Field(default="ORCHESTRATOR_CHECKPOINT_REACHED")

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


class ModelEventOrchestratorCheckpointResponse(EventBase):
    """
    Event: User response to checkpoint.

    Publisher: CLI (interactive mode)
    Consumers: NodeCodegenOrchestrator
    Topic: dev.omninode-bridge.codegen.checkpoint-response.v1
    """

    # Event metadata
    correlation_id: UUID
    event_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: str = Field(default="ORCHESTRATOR_CHECKPOINT_RESPONSE")

    # Response metadata
    workflow_id: UUID
    checkpoint_type: str
    checkpoint_number: int

    # User decision
    action: str = Field(..., description="approve|regenerate|edit|abort")
    feedback: Optional[str] = Field(None, description="User comments")

    # Edits (if applicable)
    edits: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Kafka Topic Constants
# =============================================================================

KAFKA_TOPICS = {
    # Node generation workflow
    "NODE_GENERATION_REQUESTED": "dev.omninode-bridge.codegen.generation-requested.v1",
    "NODE_GENERATION_STARTED": "dev.omninode-bridge.codegen.generation-started.v1",
    "NODE_GENERATION_STAGE_COMPLETED": "dev.omninode-bridge.codegen.stage-completed.v1",
    "NODE_GENERATION_COMPLETED": "dev.omninode-bridge.codegen.generation-completed.v1",
    "NODE_GENERATION_FAILED": "dev.omninode-bridge.codegen.generation-failed.v1",
    # Metrics aggregation
    "GENERATION_METRICS_RECORDED": "dev.omninode-bridge.codegen.metrics-recorded.v1",
    # Pattern storage
    "PATTERN_STORAGE_REQUESTED": "dev.omniarchon.intelligence.pattern-storage-requested.v1",
    "PATTERN_STORED": "dev.omniarchon.intelligence.pattern-stored.v1",
    # Intelligence gathering
    "INTELLIGENCE_QUERY_REQUESTED": "dev.omniarchon.intelligence.query-requested.v1",
    "INTELLIGENCE_QUERY_COMPLETED": "dev.omniarchon.intelligence.query-completed.v1",
    # Orchestration
    "ORCHESTRATOR_CHECKPOINT_REACHED": "dev.omninode-bridge.codegen.checkpoint-reached.v1",
    "ORCHESTRATOR_CHECKPOINT_RESPONSE": "dev.omninode-bridge.codegen.checkpoint-response.v1",
}
