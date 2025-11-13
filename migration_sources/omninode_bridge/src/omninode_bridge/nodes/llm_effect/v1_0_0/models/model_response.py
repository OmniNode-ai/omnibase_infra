#!/usr/bin/env python3
"""
LLM Response Model - ONEX v2.0 Compliant.

Response model for LLM generation operations.
"""

from datetime import UTC, datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .enum_llm_tier import EnumLLMTier


class ModelLLMResponse(BaseModel):
    """
    Response model for LLM generation operations.

    Contains generated text, model information, usage metrics, and cost tracking.
    """

    # === GENERATED CONTENT ===

    generated_text: str = Field(
        ...,
        description="LLM-generated text response",
    )

    # === MODEL INFORMATION ===

    model_used: str = Field(
        ...,
        description="Model identifier that was used (e.g., 'glm-4.5', 'llama-3.1-8b')",
    )

    tier_used: EnumLLMTier = Field(
        ...,
        description="LLM tier that was used for generation",
    )

    # === TOKEN USAGE ===

    tokens_input: int = Field(
        ...,
        description="Number of input tokens (prompt + system prompt)",
        ge=0,
    )

    tokens_output: int = Field(
        ...,
        description="Number of output tokens (generated text)",
        ge=0,
    )

    tokens_total: int = Field(
        ...,
        description="Total tokens used (input + output)",
        ge=0,
    )

    # === PERFORMANCE METRICS ===

    latency_ms: float = Field(
        ...,
        description="Generation latency in milliseconds",
        ge=0.0,
    )

    # === COST TRACKING ===

    cost_usd: float = Field(
        ...,
        description="Estimated cost in USD (based on model pricing)",
        ge=0.0,
    )

    # === QUALITY INDICATORS ===

    finish_reason: str = Field(
        ...,
        description="Completion reason (stop, length, error, content_filter)",
    )

    truncated: bool = Field(
        default=False,
        description="Whether response was truncated due to max_tokens limit",
    )

    # === ERROR HANDLING ===

    warnings: list[str] = Field(
        default_factory=list,
        description="Non-critical warnings during generation",
    )

    retry_count: int = Field(
        default=0,
        description="Number of retry attempts that were made",
        ge=0,
    )

    # === CORRELATION TRACKING ===

    correlation_id: UUID | None = Field(
        default=None,
        description="UUID for correlation tracking (from request)",
    )

    execution_id: UUID | None = Field(
        default=None,
        description="UUID for this execution instance (from request)",
    )

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Generation timestamp",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "generated_text": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
                    "model_used": "glm-4.5",
                    "tier_used": "CLOUD_FAST",
                    "tokens_input": 25,
                    "tokens_output": 45,
                    "tokens_total": 70,
                    "latency_ms": 1250.5,
                    "cost_usd": 0.000014,
                    "finish_reason": "stop",
                    "truncated": False,
                    "warnings": [],
                    "retry_count": 0,
                },
                {
                    "generated_text": "# Test truncated due to token limit...",
                    "model_used": "glm-4.5",
                    "tier_used": "CLOUD_FAST",
                    "tokens_input": 150,
                    "tokens_output": 4000,
                    "tokens_total": 4150,
                    "latency_ms": 5800.2,
                    "cost_usd": 0.00083,
                    "finish_reason": "length",
                    "truncated": True,
                    "warnings": ["Response truncated at max_tokens limit"],
                    "retry_count": 0,
                },
            ]
        }
    )


__all__ = ["ModelLLMResponse"]
