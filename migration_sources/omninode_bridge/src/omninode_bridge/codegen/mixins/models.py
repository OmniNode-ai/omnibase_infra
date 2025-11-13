"""
Pydantic models for intelligent mixin selection.

Models support the Phase 3 mixin intelligence pipeline:
- Requirement analysis results
- Mixin recommendations with explanations
- Conflict detection results
- Usage statistics for adaptive learning
"""

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class ModelRequirementAnalysis(BaseModel):
    """
    Structured analysis of PRD requirements for mixin selection.

    This model contains extracted features and categorized requirement scores
    used to intelligently select mixins.
    """

    # Extracted features
    keywords: set[str] = Field(
        default_factory=set, description="Normalized keywords from text"
    )
    dependency_packages: set[str] = Field(
        default_factory=set, description="Dependency package names"
    )
    operation_types: set[str] = Field(
        default_factory=set, description="Operation categories"
    )

    # Categorized requirements (0-10 score per category)
    database_score: float = Field(
        default=0.0, ge=0.0, le=10.0, description="Database operations strength"
    )
    api_score: float = Field(
        default=0.0, ge=0.0, le=10.0, description="API client operations strength"
    )
    kafka_score: float = Field(
        default=0.0, ge=0.0, le=10.0, description="Kafka/messaging operations strength"
    )
    security_score: float = Field(
        default=0.0, ge=0.0, le=10.0, description="Security requirements strength"
    )
    observability_score: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Observability requirements strength",
    )
    resilience_score: float = Field(
        default=0.0, ge=0.0, le=10.0, description="Resilience requirements strength"
    )
    caching_score: float = Field(
        default=0.0, ge=0.0, le=10.0, description="Caching requirements strength"
    )
    performance_score: float = Field(
        default=0.0, ge=0.0, le=10.0, description="Performance optimization strength"
    )

    # Metadata
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Overall extraction confidence"
    )
    rationale: str = Field(default="", description="Why these scores were assigned")

    class Config:
        frozen = False


class ModelMixinRecommendation(BaseModel):
    """
    Recommendation for a specific mixin with explanation.

    Used to present top-K mixin recommendations to users or downstream systems.
    """

    mixin_name: str = Field(..., description="Mixin class name (e.g., MixinRetry)")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    category: str = Field(
        ..., description="Mixin category (database/resilience/observability/security)"
    )
    explanation: str = Field(
        ..., min_length=10, description="Why this mixin is recommended"
    )
    matched_requirements: list[str] = Field(
        default_factory=list, description="Which requirements matched"
    )
    prerequisites: list[str] = Field(
        default_factory=list, description="Other mixins or dependencies needed"
    )
    conflicts_with: list[str] = Field(
        default_factory=list, description="Conflicting mixins"
    )

    class Config:
        frozen = False


class ModelMixinConflict(BaseModel):
    """
    Detected conflict between mixins.

    Represents mutual exclusions, missing prerequisites, or redundancies.
    """

    type: str = Field(
        ...,
        description="Conflict type: mutual_exclusion, missing_prerequisite, redundancy",
    )
    mixin_a: str = Field(..., description="First mixin in conflict")
    mixin_b: str = Field(..., description="Second mixin in conflict")
    reason: str = Field(..., min_length=10, description="Why this is a conflict")
    resolution: str = Field(
        ...,
        description="Resolution strategy: prefer_higher_score, add_prerequisite, remove_redundant",
    )

    class Config:
        frozen = False


class ModelMixinUsageStats(BaseModel):
    """
    Usage statistics for adaptive mixin scoring.

    Tracks how often mixins are recommended, accepted, and result in successful
    code generation. Used to improve recommendations over time.
    """

    mixin_name: str = Field(..., description="Mixin class name")
    recommended_count: int = Field(default=0, ge=0, description="Times recommended")
    accepted_count: int = Field(default=0, ge=0, description="Times accepted by user")
    success_count: int = Field(
        default=0, ge=0, description="Times resulted in successful generation"
    )
    failure_count: int = Field(default=0, ge=0, description="Times resulted in failure")

    # Co-occurrence tracking
    often_used_with: dict[str, int] = Field(
        default_factory=dict,
        description="Other mixins frequently used together (mixin_name → count)",
    )

    # Performance tracking
    avg_generation_time_ms: float = Field(
        default=0.0, ge=0.0, description="Average code generation time"
    )
    avg_code_quality_score: float = Field(
        default=0.0, ge=0.0, le=5.0, description="Average code quality score (0-5)"
    )

    # Metadata
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Last update timestamp",
    )

    class Config:
        frozen = False

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0-1)."""
        if self.recommended_count == 0:
            return 0.0
        return self.success_count / self.recommended_count

    @property
    def acceptance_rate(self) -> float:
        """Calculate acceptance rate (0-1)."""
        if self.recommended_count == 0:
            return 0.0
        return self.accepted_count / self.recommended_count
