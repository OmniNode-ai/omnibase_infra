"""
Variant metadata for template selection.

This module defines metadata for each template variant, enabling intelligent
selection based on requirements, complexity, and operational characteristics.

Performance Target: <1ms per variant lookup
Accuracy Target: >95% correct variant selection
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from ..patterns.models import EnumNodeType


class EnumTemplateVariant(str, Enum):
    """
    Template variants for different use case patterns.

    Variants are optimized for specific operational patterns:
    - MINIMAL: Learning, prototyping, simple operations (0-2 operations)
    - STANDARD: Common use cases with basic observability (2-5 operations)
    - PRODUCTION: Full observability, monitoring, production-grade (5+ operations)
    - DATABASE_HEAVY: Large database operations with connection pooling
    - API_HEAVY: HTTP clients with circuit breakers and rate limiting
    - KAFKA_HEAVY: Event-driven architecture with producers/consumers
    - ML_INFERENCE: Machine learning model inference pipelines
    - ANALYTICS: Metrics collection, aggregation, time-series
    - WORKFLOW: Multi-step orchestration with FSM and rollback
    """

    MINIMAL = "minimal"
    STANDARD = "standard"
    PRODUCTION = "production"
    DATABASE_HEAVY = "database_heavy"
    API_HEAVY = "api_heavy"
    KAFKA_HEAVY = "kafka_heavy"
    ML_INFERENCE = "ml_inference"
    ANALYTICS = "analytics"
    WORKFLOW = "workflow"


class ModelVariantMetadata(BaseModel):
    """
    Metadata describing a template variant.

    This metadata enables intelligent variant selection by providing:
    - Descriptive information about the variant's purpose
    - Applicability constraints (node types, operation counts)
    - Recommended mixins and patterns
    - Feature capabilities

    Attributes:
        variant: The template variant identifier
        node_types: Node types this variant supports
        description: What this variant is optimized for
        features: Key features included in this variant
        recommended_for: Use cases this variant excels at
        min_operations: Minimum operation count for this variant
        max_operations: Maximum operation count (None = unlimited)
        suggested_mixins: Recommended mixins for this variant
        suggested_patterns: Recommended patterns for this variant
        complexity_score: Complexity level (1-5)
        performance_characteristics: Performance metrics
        prerequisites: Required dependencies or setup
        created_at: When this variant was added
        updated_at: Last modification timestamp
    """

    # Identity
    variant: EnumTemplateVariant = Field(..., description="Template variant identifier")

    # Applicability
    node_types: list[EnumNodeType] = Field(
        ..., min_length=1, description="Node types this variant supports"
    )

    # Description
    description: str = Field(
        ..., min_length=20, description="What this variant is optimized for"
    )
    features: list[str] = Field(
        default_factory=list, description="Key features included in this variant"
    )
    recommended_for: list[str] = Field(
        default_factory=list, description="Use cases this variant excels at"
    )

    # Constraints
    min_operations: int = Field(
        default=0, ge=0, description="Minimum operation count for this variant"
    )
    max_operations: Optional[int] = Field(
        None, ge=0, description="Maximum operation count (None = unlimited)"
    )

    # Recommendations
    suggested_mixins: list[str] = Field(
        default_factory=list, description="Recommended mixins for this variant"
    )
    suggested_patterns: list[str] = Field(
        default_factory=list, description="Recommended patterns for this variant"
    )

    # Metrics
    complexity_score: int = Field(
        default=2, ge=1, le=5, description="Complexity level (1=simple, 5=complex)"
    )
    performance_characteristics: dict[str, Any] = Field(
        default_factory=dict,
        description="Performance metrics (latency, memory, throughput)",
    )

    # Technical Details
    prerequisites: list[str] = Field(
        default_factory=list, description="Required dependencies or setup"
    )

    # Metadata
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Variant creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last modification timestamp"
    )

    @field_validator(
        "features",
        "recommended_for",
        "suggested_mixins",
        "suggested_patterns",
        mode="before",
    )
    @classmethod
    def normalize_list_fields(cls, v: list[str]) -> list[str]:
        """Normalize list fields to remove duplicates and empty strings."""
        if not v:
            return []
        return sorted(set(item.strip() for item in v if item.strip()))

    def matches_requirements(
        self,
        node_type: EnumNodeType,
        operation_count: int,
        required_features: set[str],
    ) -> bool:
        """
        Check if this variant matches the given requirements.

        Args:
            node_type: Target node type
            operation_count: Number of operations in requirements
            required_features: Set of required feature tags

        Returns:
            True if variant matches requirements
        """
        # Check node type compatibility
        if node_type not in self.node_types:
            return False

        # Check operation count range
        if operation_count < self.min_operations:
            return False
        if self.max_operations is not None and operation_count > self.max_operations:
            return False

        # Check feature overlap (at least 30% match)
        if required_features:
            variant_features = set(self.features)
            if not variant_features:
                return True  # No features specified = matches all

            overlap = len(variant_features.intersection(required_features))
            ratio = overlap / len(variant_features)
            if ratio < 0.3:
                return False

        return True

    def calculate_match_score(
        self,
        node_type: EnumNodeType,
        operation_count: int,
        required_features: set[str],
    ) -> float:
        """
        Calculate match score (0.0 to 1.0) for given requirements.

        Scoring algorithm:
        - Node type match: +0.2
        - Operation count fit: +0.3 (perfect fit) to 0.0 (out of range)
        - Feature overlap: +0.5 based on overlap ratio

        Args:
            node_type: Target node type
            operation_count: Number of operations
            required_features: Set of required feature tags

        Returns:
            Match score between 0.0 and 1.0
        """
        score = 0.0

        # Node type compatibility (20%)
        if node_type in self.node_types:
            score += 0.2
        else:
            return 0.0  # No match if node type incompatible

        # Operation count fit (30%)
        if operation_count < self.min_operations:
            return 0.0  # Out of range

        if self.max_operations is None:
            # Unlimited max: score based on how close to min
            if operation_count >= self.min_operations:
                score += 0.3
        else:
            # Limited range: score based on fit within range
            if operation_count > self.max_operations:
                return 0.0  # Out of range

            range_size = self.max_operations - self.min_operations + 1
            distance_from_min = operation_count - self.min_operations
            fit_ratio = 1.0 - (distance_from_min / range_size)
            score += 0.3 * fit_ratio

        # Feature overlap (50%)
        if required_features:
            variant_features = set(self.features)
            if variant_features:
                overlap = len(variant_features.intersection(required_features))
                ratio = overlap / len(required_features)
                score += 0.5 * ratio
            else:
                score += 0.25  # No features specified = partial match
        else:
            score += 0.5  # No features required = full score

        return min(score, 1.0)

    class Config:
        frozen = False
        str_strip_whitespace = True
        json_schema_extra = {
            "example": {
                "variant": "database_heavy",
                "node_types": ["effect", "reducer"],
                "description": "Optimized for large database operations with connection pooling",
                "features": [
                    "connection_pooling",
                    "transaction_management",
                    "query_optimization",
                ],
                "recommended_for": ["High-volume data processing", "Batch operations"],
                "min_operations": 3,
                "max_operations": None,
                "suggested_mixins": [
                    "MixinConnectionPooling",
                    "MixinTransactionManagement",
                ],
                "suggested_patterns": [
                    "circuit_breaker",
                    "retry_policy",
                    "health_checks",
                ],
                "complexity_score": 4,
            }
        }


class ModelTemplateSelection(BaseModel):
    """
    Result of template variant selection.

    Represents the selected template variant with confidence score and rationale.

    Attributes:
        variant: Selected template variant
        confidence: Selection confidence (0.0 to 1.0)
        rationale: Explanation of why this variant was selected
        matched_features: Features that contributed to selection
        fallback_used: Whether fallback selection was used
        selection_time_ms: Time taken for selection (performance metric)
    """

    variant: EnumTemplateVariant = Field(..., description="Selected template variant")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Selection confidence score"
    )
    rationale: str = Field(
        ..., min_length=10, description="Why this variant was selected"
    )
    matched_features: list[str] = Field(
        default_factory=list, description="Features that matched requirements"
    )
    fallback_used: bool = Field(
        default=False, description="Whether fallback selection was used"
    )
    selection_time_ms: float = Field(
        default=0.0, ge=0.0, description="Time taken for selection (ms)"
    )

    class Config:
        frozen = True


# ============================================================================
# Variant Metadata Registry
# ============================================================================

VARIANT_METADATA_REGISTRY: dict[EnumTemplateVariant, ModelVariantMetadata] = {
    EnumTemplateVariant.MINIMAL: ModelVariantMetadata(
        variant=EnumTemplateVariant.MINIMAL,
        node_types=[
            EnumNodeType.EFFECT,
            EnumNodeType.COMPUTE,
            EnumNodeType.REDUCER,
            EnumNodeType.ORCHESTRATOR,
        ],
        description="Minimal template for learning, prototyping, and simple operations with minimal boilerplate",
        features=["basic_structure", "simple_error_handling", "minimal_logging"],
        recommended_for=[
            "Learning ONEX v2.0",
            "Rapid prototyping",
            "Simple transformations",
            "Tutorial examples",
        ],
        min_operations=0,
        max_operations=2,
        suggested_mixins=[],
        suggested_patterns=["standard_imports", "class_declaration", "type_hints"],
        complexity_score=1,
        performance_characteristics={
            "latency": "< 1ms",
            "memory": "< 10MB",
            "throughput": "1000+ ops/sec",
        },
    ),
    EnumTemplateVariant.STANDARD: ModelVariantMetadata(
        variant=EnumTemplateVariant.STANDARD,
        node_types=[
            EnumNodeType.EFFECT,
            EnumNodeType.COMPUTE,
            EnumNodeType.REDUCER,
            EnumNodeType.ORCHESTRATOR,
        ],
        description="Standard template for common use cases with basic observability and error handling",
        features=[
            "structured_logging",
            "basic_metrics",
            "error_handling",
            "health_checks",
            "consul_registration",
        ],
        recommended_for=[
            "Common CRUD operations",
            "Standard microservices",
            "Basic workflows",
            "Integration services",
        ],
        min_operations=2,
        max_operations=5,
        suggested_mixins=[],
        suggested_patterns=[
            "error_handling",
            "structured_logging",
            "health_check_mode",
            "consul_registration",
        ],
        complexity_score=2,
        performance_characteristics={
            "latency": "< 5ms",
            "memory": "< 50MB",
            "throughput": "500+ ops/sec",
        },
    ),
    EnumTemplateVariant.PRODUCTION: ModelVariantMetadata(
        variant=EnumTemplateVariant.PRODUCTION,
        node_types=[
            EnumNodeType.EFFECT,
            EnumNodeType.COMPUTE,
            EnumNodeType.REDUCER,
            EnumNodeType.ORCHESTRATOR,
        ],
        description="Production-grade template with full observability, monitoring, and resilience patterns",
        features=[
            "comprehensive_logging",
            "detailed_metrics",
            "event_publishing",
            "circuit_breakers",
            "retry_policies",
            "health_checks",
            "consul_registration",
            "lifecycle_management",
        ],
        recommended_for=[
            "Production deployments",
            "High-availability services",
            "Mission-critical operations",
            "Enterprise applications",
        ],
        min_operations=5,
        max_operations=None,
        suggested_mixins=[
            "MixinCircuitBreaker",
            "MixinRetry",
            "MixinMetrics",
        ],
        suggested_patterns=[
            "error_handling",
            "structured_logging",
            "metrics_tracking",
            "event_publishing",
            "consul_registration",
            "lifecycle_hooks",
        ],
        complexity_score=4,
        performance_characteristics={
            "latency": "< 20ms",
            "memory": "< 200MB",
            "throughput": "200+ ops/sec",
        },
    ),
    EnumTemplateVariant.DATABASE_HEAVY: ModelVariantMetadata(
        variant=EnumTemplateVariant.DATABASE_HEAVY,
        node_types=[EnumNodeType.EFFECT, EnumNodeType.REDUCER],
        description="Optimized for large database operations with connection pooling and transaction management",
        features=[
            "connection_pooling",
            "transaction_management",
            "query_optimization",
            "batch_operations",
            "connection_health_checks",
            "query_metrics",
            "prepared_statements",
        ],
        recommended_for=[
            "High-volume data processing",
            "Batch operations",
            "Data warehousing",
            "ETL pipelines",
            "Database migrations",
        ],
        min_operations=3,
        max_operations=None,
        suggested_mixins=[
            "MixinConnectionPooling",
            "MixinTransactionManagement",
            "MixinBatchProcessing",
        ],
        suggested_patterns=[
            "error_handling",
            "structured_logging",
            "metrics_tracking",
            "health_check_mode",
        ],
        complexity_score=4,
        performance_characteristics={
            "latency": "< 50ms per query",
            "memory": "< 500MB (with pooling)",
            "throughput": "1000+ queries/sec",
            "connection_pool": "10-50 connections",
        },
        prerequisites=[
            "PostgreSQL client (asyncpg)",
            "Database connection string",
            "Migration scripts",
        ],
    ),
    EnumTemplateVariant.API_HEAVY: ModelVariantMetadata(
        variant=EnumTemplateVariant.API_HEAVY,
        node_types=[EnumNodeType.EFFECT, EnumNodeType.ORCHESTRATOR],
        description="Optimized for HTTP API operations with circuit breakers, retries, and rate limiting",
        features=[
            "http_client",
            "circuit_breaker",
            "retry_logic",
            "rate_limiting",
            "timeout_management",
            "connection_pooling",
            "request_metrics",
            "response_caching",
        ],
        recommended_for=[
            "External API integrations",
            "Microservice communication",
            "Third-party service calls",
            "Webhook consumers",
            "API gateways",
        ],
        min_operations=2,
        max_operations=None,
        suggested_mixins=[
            "MixinCircuitBreaker",
            "MixinRetry",
            "MixinRateLimiting",
            "MixinCaching",
        ],
        suggested_patterns=[
            "error_handling",
            "structured_logging",
            "metrics_tracking",
            "consul_registration",
        ],
        complexity_score=3,
        performance_characteristics={
            "latency": "< 100ms per request",
            "memory": "< 100MB",
            "throughput": "100+ requests/sec",
            "concurrent_connections": "50-100",
        },
        prerequisites=[
            "HTTP client (aiohttp/httpx)",
            "Circuit breaker library",
            "Rate limiter",
        ],
    ),
    EnumTemplateVariant.KAFKA_HEAVY: ModelVariantMetadata(
        variant=EnumTemplateVariant.KAFKA_HEAVY,
        node_types=[
            EnumNodeType.EFFECT,
            EnumNodeType.ORCHESTRATOR,
            EnumNodeType.REDUCER,
        ],
        description="Optimized for event-driven architecture with Kafka producers and consumers",
        features=[
            "kafka_producer",
            "kafka_consumer",
            "batch_processing",
            "dlq_handling",
            "event_publishing",
            "consumer_groups",
            "offset_management",
            "exactly_once_semantics",
        ],
        recommended_for=[
            "Event-driven architectures",
            "Real-time data pipelines",
            "Stream processing",
            "Event sourcing",
            "CQRS patterns",
        ],
        min_operations=2,
        max_operations=None,
        suggested_mixins=[
            "MixinEventDrivenNode",
            "MixinEventPublisher",
            "MixinBatchProcessing",
        ],
        suggested_patterns=[
            "error_handling",
            "structured_logging",
            "event_publishing",
            "kafka_client_initialization",
            "lifecycle_hooks",
        ],
        complexity_score=4,
        performance_characteristics={
            "latency": "< 10ms per message",
            "memory": "< 200MB",
            "throughput": "10000+ messages/sec",
            "batch_size": "100-1000 messages",
        },
        prerequisites=[
            "Kafka cluster (Redpanda)",
            "Kafka client (aiokafka)",
            "Topic configuration",
        ],
    ),
    EnumTemplateVariant.ML_INFERENCE: ModelVariantMetadata(
        variant=EnumTemplateVariant.ML_INFERENCE,
        node_types=[EnumNodeType.EFFECT, EnumNodeType.COMPUTE],
        description="Optimized for machine learning model inference with preprocessing and batching",
        features=[
            "model_loading",
            "lazy_initialization",
            "batch_inference",
            "preprocessing",
            "postprocessing",
            "model_caching",
            "inference_metrics",
            "gpu_support",
        ],
        recommended_for=[
            "ML model serving",
            "Real-time inference",
            "Batch predictions",
            "Feature engineering",
            "Model ensembles",
        ],
        min_operations=2,
        max_operations=None,
        suggested_mixins=[
            "MixinCaching",
            "MixinBatchProcessing",
            "MixinMetrics",
        ],
        suggested_patterns=[
            "error_handling",
            "structured_logging",
            "metrics_tracking",
            "health_check_mode",
        ],
        complexity_score=4,
        performance_characteristics={
            "latency": "< 50ms per inference",
            "memory": "< 2GB (with model)",
            "throughput": "100+ inferences/sec",
            "batch_size": "1-32 samples",
        },
        prerequisites=[
            "ML framework (torch/tensorflow/onnx)",
            "Model file path or URL",
            "Preprocessing pipeline",
        ],
    ),
    EnumTemplateVariant.ANALYTICS: ModelVariantMetadata(
        variant=EnumTemplateVariant.ANALYTICS,
        node_types=[EnumNodeType.REDUCER, EnumNodeType.COMPUTE],
        description="Optimized for metrics collection, aggregation, and time-series analysis",
        features=[
            "metrics_collection",
            "aggregation",
            "time_series",
            "windowing",
            "percentile_calculation",
            "histogram_generation",
            "metric_export",
            "real_time_dashboards",
        ],
        recommended_for=[
            "Performance monitoring",
            "Business analytics",
            "Time-series analysis",
            "Real-time metrics",
            "Dashboard backends",
        ],
        min_operations=2,
        max_operations=None,
        suggested_mixins=[
            "MixinMetrics",
            "MixinAggregation",
        ],
        suggested_patterns=[
            "error_handling",
            "structured_logging",
            "metrics_tracking",
            "time_tracking",
        ],
        complexity_score=3,
        performance_characteristics={
            "latency": "< 10ms per aggregation",
            "memory": "< 100MB",
            "throughput": "1000+ metrics/sec",
            "window_size": "1s-1h",
        },
    ),
    EnumTemplateVariant.WORKFLOW: ModelVariantMetadata(
        variant=EnumTemplateVariant.WORKFLOW,
        node_types=[EnumNodeType.ORCHESTRATOR],
        description="Optimized for multi-step workflow coordination with FSM and rollback capabilities",
        features=[
            "fsm_state_management",
            "multi_step_coordination",
            "retry_logic",
            "rollback_support",
            "workflow_persistence",
            "parallel_execution",
            "conditional_branching",
            "workflow_metrics",
        ],
        recommended_for=[
            "Complex workflows",
            "Multi-step processes",
            "Saga patterns",
            "Business processes",
            "Distributed transactions",
        ],
        min_operations=3,
        max_operations=None,
        suggested_mixins=[
            "MixinEventDrivenNode",
            "MixinRetry",
            "MixinMetrics",
        ],
        suggested_patterns=[
            "error_handling",
            "structured_logging",
            "metrics_tracking",
            "event_publishing",
            "lifecycle_hooks",
        ],
        complexity_score=5,
        performance_characteristics={
            "latency": "< 100ms per step",
            "memory": "< 150MB",
            "throughput": "50+ workflows/sec",
            "max_steps": "10-50 steps",
        },
    ),
}
