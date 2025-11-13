#!/usr/bin/env python3
"""
LLM Request Model - ONEX v2.0 Compliant.

Request model for LLM generation operations.
"""

from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from .enum_llm_tier import EnumLLMTier


class ModelLLMRequest(BaseModel):
    """
    Request model for LLM generation operations.

    Contains prompts, model tier selection, and generation parameters.
    """

    # === PROMPT CONFIGURATION ===

    prompt: str = Field(
        ...,
        description="User prompt for LLM generation",
        min_length=1,
        max_length=100000,
    )

    system_prompt: str | None = Field(
        default=None,
        description="System prompt for LLM (optional, defaults vary by tier)",
        max_length=10000,
    )

    # === TIER SELECTION ===

    tier: EnumLLMTier = Field(
        default=EnumLLMTier.CLOUD_FAST,
        description="LLM tier to use (LOCAL, CLOUD_FAST, CLOUD_PREMIUM)",
    )

    model_override: str | None = Field(
        default=None,
        description="Override default model for tier (e.g., 'glm-4.5', 'llama-3.1-8b')",
        max_length=100,
    )

    # === GENERATION PARAMETERS ===

    max_tokens: int = Field(
        default=4000,
        description="Maximum tokens to generate",
        ge=1,
        le=128000,  # GLM-4.5 max context window
    )

    temperature: float = Field(
        default=0.7,
        description="Sampling temperature (0.0 = deterministic, 2.0 = very random)",
        ge=0.0,
        le=2.0,
    )

    top_p: float = Field(
        default=0.95,
        description="Nucleus sampling probability threshold",
        ge=0.0,
        le=1.0,
    )

    # === OPERATION CONTEXT ===

    operation_type: str = Field(
        default="general",
        description="Operation type hint (node_generation, test_generation, documentation, etc.)",
        max_length=100,
    )

    context_window: str | None = Field(
        default=None,
        description="Additional context for generation (e.g., existing code, schemas)",
        max_length=200000,
    )

    # === QUALITY CONTROLS ===

    enable_streaming: bool = Field(
        default=False,
        description="Enable response streaming (not implemented in Phase 1)",
    )

    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts on API failures",
        ge=1,
        le=5,
    )

    timeout_seconds: float = Field(
        default=60.0,
        description="Request timeout in seconds",
        ge=1.0,
        le=300.0,
    )

    # === METADATA ===

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional request metadata",
    )

    # === CORRELATION TRACKING ===

    correlation_id: UUID | None = Field(
        default=None,
        description="UUID for correlation tracking (auto-generated if not provided)",
    )

    execution_id: UUID = Field(
        default_factory=uuid4,
        description="UUID for tracking this execution instance",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "prompt": "Generate a Python function to calculate Fibonacci numbers",
                    "system_prompt": "You are a helpful coding assistant specialized in Python.",
                    "tier": "CLOUD_FAST",
                    "max_tokens": 2000,
                    "temperature": 0.7,
                    "operation_type": "node_generation",
                },
                {
                    "prompt": "Write comprehensive unit tests for a distributed lock system",
                    "tier": "CLOUD_FAST",
                    "max_tokens": 4000,
                    "temperature": 0.5,
                    "operation_type": "test_generation",
                    "context_window": "# Existing code:\nclass DistributedLock:\n    ...",
                },
            ]
        }
    )


__all__ = ["ModelLLMRequest"]
