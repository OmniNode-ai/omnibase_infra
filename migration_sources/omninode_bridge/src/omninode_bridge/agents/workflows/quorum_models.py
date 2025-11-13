"""
AI Quorum data models for code validation.

Pydantic v2 models for 4-model consensus validation system.
"""

from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    """
    Configuration for an LLM model in the quorum.

    Attributes:
        model_id: Unique identifier for the model (e.g., "gemini", "glm-4.5")
        model_name: Full model name (e.g., "gemini-1.5-pro", "glm-4-plus")
        weight: Model weight in consensus calculation (0.0-10.0)
        endpoint: API endpoint URL
        api_key_env: Environment variable name for API key
        timeout: Request timeout in seconds (default 30)
        max_retries: Maximum retry attempts (default 2)
        enabled: Whether model is enabled (default True)
    """

    model_id: str = Field(..., min_length=1, max_length=50)
    model_name: str = Field(..., min_length=1, max_length=100)
    weight: float = Field(..., gt=0.0, le=10.0)
    endpoint: str = Field(..., min_length=1)
    api_key_env: str = Field(..., min_length=1)
    timeout: int = Field(default=30, gt=0, le=120)
    max_retries: int = Field(default=2, ge=0, le=5)
    enabled: bool = Field(default=True)

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, v: str) -> str:
        """Validate endpoint is a valid URL."""
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("Endpoint must start with http:// or https://")
        return v

    class Config:
        """Pydantic configuration."""

        frozen = True
        validate_assignment = True


class QuorumVote(BaseModel):
    """
    A single model's vote in the quorum.

    Attributes:
        vote_id: Unique vote identifier
        model_id: Model that cast this vote
        vote: True = pass, False = fail
        confidence: Vote confidence (0.0-1.0)
        reasoning: Detailed reasoning for the vote
        duration_ms: Time taken to generate vote
        timestamp: Vote timestamp
        error: Optional error message if vote failed
    """

    vote_id: str = Field(default_factory=lambda: str(uuid4()))
    model_id: str = Field(..., min_length=1, max_length=50)
    vote: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., min_length=1)
    duration_ms: float = Field(..., ge=0.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    error: Optional[str] = None

    class Config:
        """Pydantic configuration."""

        frozen = False
        validate_assignment = True


class QuorumResult(BaseModel):
    """
    Result of AI Quorum validation.

    Attributes:
        result_id: Unique result identifier
        passed: Whether code passed quorum validation
        consensus_score: Weighted consensus score (0.0-1.0)
        votes: List of individual model votes
        total_weight: Total weight of all participating models
        participating_weight: Total weight of models that successfully voted
        pass_threshold: Threshold required to pass (0.0-1.0)
        duration_ms: Total quorum execution time
        timestamp: Result timestamp
        correlation_id: Optional correlation ID for tracing
        metadata: Optional metadata (e.g., node_type, contract info)
    """

    result_id: str = Field(default_factory=lambda: str(uuid4()))
    passed: bool
    consensus_score: float = Field(..., ge=0.0, le=1.0)
    votes: list[QuorumVote] = Field(default_factory=list)
    total_weight: float = Field(..., gt=0.0)
    participating_weight: float = Field(..., ge=0.0)
    pass_threshold: float = Field(..., ge=0.0, le=1.0)
    duration_ms: float = Field(..., ge=0.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("participating_weight")
    @classmethod
    def validate_participating_weight(cls, v: float, info) -> float:
        """Validate participating weight doesn't exceed total weight."""
        total_weight = info.data.get("total_weight", 0.0)
        if v > total_weight:
            raise ValueError(
                f"Participating weight ({v}) cannot exceed total weight ({total_weight})"
            )
        return v

    class Config:
        """Pydantic configuration."""

        frozen = False
        validate_assignment = True

    def get_votes_summary(self) -> dict[str, Any]:
        """
        Get summary of votes.

        Returns:
            Dictionary with vote counts and statistics
        """
        total_votes = len(self.votes)
        passed_votes = sum(1 for v in self.votes if v.vote)
        failed_votes = total_votes - passed_votes
        avg_confidence = (
            sum(v.confidence for v in self.votes) / total_votes if total_votes > 0 else 0.0
        )

        return {
            "total_votes": total_votes,
            "passed_votes": passed_votes,
            "failed_votes": failed_votes,
            "avg_confidence": round(avg_confidence, 3),
            "consensus_score": round(self.consensus_score, 3),
            "passed": self.passed,
        }

    def get_vote_by_model(self, model_id: str) -> Optional[QuorumVote]:
        """
        Get vote by model ID.

        Args:
            model_id: Model identifier

        Returns:
            QuorumVote if found, None otherwise
        """
        for vote in self.votes:
            if vote.model_id == model_id:
                return vote
        return None


class ValidationContext(BaseModel):
    """
    Context for code validation.

    Attributes:
        node_type: Type of node being validated (e.g., "effect", "compute")
        contract_summary: Summary of contract requirements
        code_snippet: Optional code snippet for context
        validation_criteria: List of specific criteria to validate
        additional_context: Optional additional context
    """

    node_type: str = Field(..., min_length=1, max_length=50)
    contract_summary: str = Field(..., min_length=1)
    code_snippet: Optional[str] = None
    validation_criteria: list[str] = Field(default_factory=list)
    additional_context: dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""

        frozen = False
        validate_assignment = True
