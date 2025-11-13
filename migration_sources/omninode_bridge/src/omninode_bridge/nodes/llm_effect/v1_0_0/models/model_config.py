#!/usr/bin/env python3
"""
LLM Configuration Model - ONEX v2.0 Compliant.

Configuration model for LLM Effect node initialization.
"""

from pydantic import BaseModel, ConfigDict, Field


class ModelLLMConfig(BaseModel):
    """
    Configuration model for LLM Effect node.

    Contains API credentials, model mappings, and performance settings.
    """

    # === API CREDENTIALS ===

    zai_api_key: str = Field(
        ...,
        description="Z.ai API key for GLM models",
        min_length=1,
    )

    zai_base_url: str = Field(
        default="https://api.z.ai/api/anthropic",
        description="Z.ai API base URL (Anthropic-compatible endpoint)",
    )

    ollama_base_url: str | None = Field(
        default=None,
        description="Ollama API base URL for LOCAL tier (future)",
    )

    # === MODEL CONFIGURATION ===

    default_model_local: str | None = Field(
        default=None,
        description="Default model for LOCAL tier (future: 'llama-3.1-8b')",
    )

    default_model_cloud_fast: str = Field(
        default="glm-4.5",
        description="Default model for CLOUD_FAST tier",
    )

    default_model_cloud_premium: str = Field(
        default="glm-4.6",
        description="Default model for CLOUD_PREMIUM tier",
    )

    # === CONTEXT WINDOWS (tokens) ===

    context_window_local: int = Field(
        default=8192,
        description="Context window for LOCAL tier (future)",
        ge=1,
    )

    context_window_cloud_fast: int = Field(
        default=128000,
        description="Context window for CLOUD_FAST tier (GLM-4.5)",
        ge=1,
    )

    context_window_cloud_premium: int = Field(
        default=128000,
        description="Context window for CLOUD_PREMIUM tier (GLM-4.6)",
        ge=1,
    )

    # === PRICING (per 1M tokens in USD) ===

    cost_per_1m_input_cloud_fast: float = Field(
        default=0.20,
        description="Cost per 1M input tokens for CLOUD_FAST tier",
        ge=0.0,
    )

    cost_per_1m_output_cloud_fast: float = Field(
        default=0.20,
        description="Cost per 1M output tokens for CLOUD_FAST tier",
        ge=0.0,
    )

    cost_per_1m_input_cloud_premium: float = Field(
        default=0.30,
        description="Cost per 1M input tokens for CLOUD_PREMIUM tier",
        ge=0.0,
    )

    cost_per_1m_output_cloud_premium: float = Field(
        default=0.30,
        description="Cost per 1M output tokens for CLOUD_PREMIUM tier",
        ge=0.0,
    )

    # === CIRCUIT BREAKER ===

    circuit_breaker_threshold: int = Field(
        default=5,
        description="Number of consecutive failures before circuit breaker opens",
        ge=1,
        le=20,
    )

    circuit_breaker_timeout_seconds: float = Field(
        default=60.0,
        description="Circuit breaker timeout in seconds before attempting reset",
        ge=1.0,
        le=600.0,
    )

    # === HTTP CLIENT CONFIGURATION ===

    http_timeout_seconds: float = Field(
        default=60.0,
        description="HTTP client timeout in seconds",
        ge=1.0,
        le=300.0,
    )

    http_max_connections: int = Field(
        default=10,
        description="Maximum HTTP connections in pool",
        ge=1,
        le=100,
    )

    # === RETRY CONFIGURATION ===

    max_retry_attempts: int = Field(
        default=3,
        description="Maximum retry attempts on API failures",
        ge=1,
        le=5,
    )

    retry_initial_backoff_seconds: float = Field(
        default=1.0,
        description="Initial backoff delay in seconds for retries",
        ge=0.1,
        le=10.0,
    )

    retry_backoff_multiplier: float = Field(
        default=2.0,
        description="Backoff multiplier for exponential retry",
        ge=1.0,
        le=5.0,
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "zai_api_key": "your_zai_api_key_here",  # pragma: allowlist secret
                    "zai_base_url": "https://api.z.ai/api/anthropic",
                    "default_model_cloud_fast": "glm-4.5",
                    "context_window_cloud_fast": 128000,
                    "cost_per_1m_input_cloud_fast": 0.20,
                    "circuit_breaker_threshold": 5,
                    "circuit_breaker_timeout_seconds": 60.0,
                    "max_retry_attempts": 3,
                }
            ]
        }
    )


__all__ = ["ModelLLMConfig"]
