"""
Pydantic Models for LLM Metrics and Debug Intelligence.

ONEX v2.0 compliant data models for storing and querying LLM generation
metrics, history, and learned patterns.
"""

from datetime import datetime
from typing import Any, ClassVar, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class LLMGenerationMetric(BaseModel):
    """
    LLM generation metrics for a single operation.

    Tracks token usage, costs, performance, and success/failure status
    for a single LLM generation request.
    """

    metric_id: UUID = Field(
        default_factory=uuid4, description="Unique metric identifier"
    )
    session_id: str = Field(
        ..., description="Session identifier for grouping related metrics"
    )
    correlation_id: Optional[UUID] = Field(None, description="ONEX correlation ID")
    node_type: str = Field(
        ...,
        description="Type of node being generated (effect, compute, reducer, orchestrator)",
    )
    model_tier: str = Field(..., description="Model tier (tier_1, tier_2, tier_3)")
    model_name: str = Field(
        ..., description="Specific model name (e.g., claude-sonnet-4)"
    )
    prompt_tokens: int = Field(..., ge=0, description="Number of prompt tokens")
    completion_tokens: int = Field(..., ge=0, description="Number of completion tokens")
    total_tokens: int = Field(
        ..., ge=0, description="Total tokens (prompt + completion)"
    )
    latency_ms: float = Field(
        ..., ge=0.0, description="Generation latency in milliseconds"
    )
    cost_usd: float = Field(..., ge=0.0, description="Generation cost in USD")
    success: bool = Field(..., description="Whether generation succeeded")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    metadata: Optional[dict[str, Any]] = Field(
        default_factory=dict, description="Additional metadata"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Timestamp of metric creation"
    )

    @field_validator("total_tokens")
    @classmethod
    def validate_total_tokens(cls, v: int, info) -> int:
        """Validate that total_tokens matches prompt + completion."""
        data = info.data
        if "prompt_tokens" in data and "completion_tokens" in data:
            expected = data["prompt_tokens"] + data["completion_tokens"]
            if v != expected:
                # Auto-correct if mismatch
                return expected
        return v

    class Config:
        json_schema_extra: ClassVar[dict] = {
            "example": {
                "session_id": "sess_abc123",
                "node_type": "effect",
                "model_tier": "tier_2",
                "model_name": "claude-sonnet-4",
                "prompt_tokens": 1500,
                "completion_tokens": 800,
                "total_tokens": 2300,
                "latency_ms": 3500.0,
                "cost_usd": 0.0345,
                "success": True,
                "metadata": {"validation_passed": True, "quality_score": 0.95},
            }
        }


class LLMGenerationHistory(BaseModel):
    """
    Historical record of LLM generation with prompt and output.

    Stores the actual prompt and generated text for debugging and
    pattern learning purposes.
    """

    history_id: UUID = Field(
        default_factory=uuid4, description="Unique history identifier"
    )
    metric_id: UUID = Field(..., description="Reference to LLMGenerationMetric")
    prompt_text: str = Field(..., description="Full prompt text sent to LLM")
    generated_text: str = Field(..., description="Full generated text from LLM")
    quality_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Quality score (0-1)"
    )
    validation_passed: bool = Field(..., description="Whether validation checks passed")
    validation_errors: Optional[dict[str, Any]] = Field(
        None, description="Validation error details"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Timestamp of history creation"
    )

    class Config:
        json_schema_extra: ClassVar[dict] = {
            "example": {
                "metric_id": "00000000-0000-0000-0000-000000000000",
                "prompt_text": "Generate a Python Effect node...",
                "generated_text": "class NodeMyEffect:\n    async def execute_effect(self)...",
                "quality_score": 0.92,
                "validation_passed": True,
                "validation_errors": None,
            }
        }


class LLMPattern(BaseModel):
    """
    Learned pattern from successful LLM generations.

    Stores patterns that have been successful in the past for reuse
    and optimization of future generations.
    """

    pattern_id: UUID = Field(
        default_factory=uuid4, description="Unique pattern identifier"
    )
    pattern_type: str = Field(
        ..., description="Type of pattern (prompt_template, code_structure, etc.)"
    )
    node_type: Optional[str] = Field(
        None, description="Node type this pattern applies to"
    )
    pattern_data: dict[str, Any] = Field(
        ..., description="Pattern data (template, structure, etc.)"
    )
    usage_count: int = Field(
        default=0, ge=0, description="Number of times pattern has been used"
    )
    avg_quality_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Average quality score"
    )
    success_rate: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Success rate (0-1)"
    )
    metadata: Optional[dict[str, Any]] = Field(
        default_factory=dict, description="Additional metadata"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Timestamp of pattern creation"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Timestamp of last update"
    )

    class Config:
        json_schema_extra: ClassVar[dict] = {
            "example": {
                "pattern_type": "prompt_template",
                "node_type": "effect",
                "pattern_data": {
                    "template": "Generate a Python Effect node that...",
                    "variables": ["node_name", "operation_type"],
                },
                "usage_count": 15,
                "avg_quality_score": 0.93,
                "success_rate": 0.95,
            }
        }


class MetricsSummary(BaseModel):
    """
    Aggregated metrics summary for a time period.

    Used for reporting and dashboard purposes.
    """

    total_generations: int = Field(..., ge=0, description="Total number of generations")
    successful_generations: int = Field(
        ..., ge=0, description="Number of successful generations"
    )
    failed_generations: int = Field(
        ..., ge=0, description="Number of failed generations"
    )
    total_tokens: int = Field(..., ge=0, description="Total tokens used")
    total_cost_usd: float = Field(..., ge=0.0, description="Total cost in USD")
    avg_latency_ms: float = Field(
        ..., ge=0.0, description="Average latency in milliseconds"
    )
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Success rate (0-1)")
    period_start: datetime = Field(..., description="Start of time period")
    period_end: datetime = Field(..., description="End of time period")

    class Config:
        json_schema_extra: ClassVar[dict] = {
            "example": {
                "total_generations": 150,
                "successful_generations": 142,
                "failed_generations": 8,
                "total_tokens": 450000,
                "total_cost_usd": 12.45,
                "avg_latency_ms": 3200.0,
                "success_rate": 0.947,
                "period_start": "2025-10-31T00:00:00Z",
                "period_end": "2025-10-31T23:59:59Z",
            }
        }
