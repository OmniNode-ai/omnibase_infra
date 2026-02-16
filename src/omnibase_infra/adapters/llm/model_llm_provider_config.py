# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Concrete Pydantic model implementing ProtocolProviderConfig.

Provides a frozen, serializable provider configuration that satisfies
the SPI structural protocol for LLM provider configuration.

Related:
    - ProtocolProviderConfig (omnibase_spi.protocols.types.protocol_llm_types)
    - OMN-2319: Implement SPI LLM protocol adapters
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelLlmProviderConfig(BaseModel):
    """Concrete implementation of ProtocolProviderConfig.

    Holds connection and authentication details for an LLM provider endpoint.

    Note on ``connection_timeout``:
        The SPI protocol ``ProtocolProviderConfig`` declares ``connection_timeout``
        as ``async def connection_timeout(self) -> int``. This model implements it
        as a plain property/attribute since Pydantic models cannot have async methods.
        The adapter layer bridges this mismatch by implementing the async protocol
        method that delegates to this attribute.

    Attributes:
        provider_name: Provider identifier (e.g. 'openai-compatible', 'ollama').
        api_key: API key for authentication. None for local providers.
        base_url: Base URL for API calls (e.g. 'http://192.168.86.201:8000').
        default_model: Default model identifier to use when not specified per-request.
        connection_timeout: Connection timeout in seconds.
        max_retries: Maximum retry attempts on transient failures.
        provider_type: Deployment type: 'local', 'external_trusted', or 'external'.

    Example:
        >>> config = ModelLlmProviderConfig(
        ...     provider_name="openai-compatible",
        ...     base_url="http://192.168.86.201:8000",
        ...     default_model="qwen2.5-coder-14b",
        ... )
        >>> config.provider_type
        'local'
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    provider_name: str = Field(
        ...,
        min_length=1,
        description="Name of the provider (e.g. 'openai-compatible', 'ollama').",
    )
    api_key: str | None = Field(
        default=None,
        description="API key for authentication. None for local providers.",
    )
    base_url: str | None = Field(
        default=None,
        description="Base URL for API calls.",
    )
    default_model: str = Field(
        default="",
        description="Default model to use when not specified per-request.",
    )
    connection_timeout: int = Field(
        default=30,
        ge=1,
        le=600,
        description="Connection timeout in seconds.",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts on transient failures.",
    )
    provider_type: str = Field(
        default="local",
        description="Deployment type: 'local', 'external_trusted', or 'external'.",
    )


__all__: list[str] = ["ModelLlmProviderConfig"]
