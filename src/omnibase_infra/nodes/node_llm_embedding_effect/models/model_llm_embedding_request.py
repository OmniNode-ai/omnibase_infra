# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""LLM embedding request model for the embedding effect node.

This module provides ModelLlmEmbeddingRequest, the input model for the LLM
embedding effect node. Supports batch embedding generation via OpenAI-compatible
and Ollama endpoints.

Related:
    - ModelLlmEmbeddingResponse: Output model with dimension uniformity validator
    - MixinLlmHttpTransport: HTTP transport mixin used by handlers
    - OMN-2112: Phase 12 embedding node
"""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelLlmEmbeddingRequest(BaseModel):
    """Input model for the LLM embedding effect node.

    Groups fields into logical sections: routing, input, streaming,
    resilience, and tracing. Field validators enforce constraints at
    construction time.

    Attributes:
        base_url: LLM provider endpoint URL.
        model: Model identifier (e.g. ``'gte-qwen2-1.5b'``).
        provider_label: Provider label for observability; not used for routing.
        texts: Texts to embed. Must contain 1 to 2048 items.
        dimensions: Optional output dimensionality override. When set, the
            provider is asked to truncate or project embeddings to this size.
        stream: Streaming is not supported for embeddings; type-level guard.
        timeout_seconds: HTTP request timeout in seconds.
        max_retries: Maximum retry attempts on transient failures.
        correlation_id: Caller-provided or auto-generated correlation ID
            for distributed tracing.
        execution_id: Unique identifier for this specific embedding call.
        metadata: Arbitrary key-value pairs for observability. Use
            ``dict(metadata)`` at point of use to convert to a mapping.

    Example:
        >>> req = ModelLlmEmbeddingRequest(
        ...     base_url="http://192.168.86.201:8002",
        ...     model="gte-qwen2-1.5b",
        ...     texts=("Hello, world!",),
        ... )
        >>> len(req.texts)
        1
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    # -- Routing --

    base_url: str = Field(
        ...,
        min_length=1,
        description="LLM provider endpoint URL.",
    )
    model: str = Field(
        ...,
        min_length=1,
        description="Model identifier (e.g. 'gte-qwen2-1.5b').",
    )
    provider_label: str = Field(
        default="",
        description="Provider label for observability; not used for routing.",
    )

    # -- Input --

    texts: tuple[str, ...] = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="Texts to embed. Must contain 1 to 2048 items.",
    )
    dimensions: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Optional output dimensionality override. When set, the provider "
            "is asked to truncate or project embeddings to this size."
        ),
    )

    # -- Streaming --

    stream: Literal[False] = Field(
        default=False,
        description="Streaming is not supported for embeddings; type-level guard.",
    )

    # -- Resilience --

    timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=600.0,
        description="HTTP request timeout in seconds.",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts on transient failures.",
    )

    # -- Tracing --

    correlation_id: UUID = Field(
        default_factory=uuid4,
        description=(
            "Caller-provided or auto-generated correlation ID for distributed tracing."
        ),
    )
    execution_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this specific embedding call.",
    )
    metadata: tuple[tuple[str, str], ...] = Field(
        default_factory=tuple,
        description="Arbitrary key-value pairs for observability. Use dict(metadata) at point of use.",
    )

    @field_validator("texts")
    @classmethod
    def _validate_texts_non_empty_strings(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        """Reject empty or whitespace-only text entries."""
        for i, text in enumerate(v):
            if not text.strip():
                raise ValueError(f"texts[{i}] must be non-empty and non-whitespace.")
        return v

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, v: str) -> str:
        """Ensure base_url uses an HTTP(S) scheme with a valid host.

        Uses ``urllib.parse.urlparse`` for robust host extraction instead of
        naive string-prefix checks alone.
        """
        if not v.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        # Ensure there's content after the scheme
        scheme_end = v.index("://") + 3
        if len(v) <= scheme_end or not v[scheme_end:].strip("/"):
            raise ValueError("base_url must include a host after the scheme")
        # Use urlparse to extract hostname
        parsed = urlparse(v)
        if not parsed.hostname:
            raise ValueError("base_url must include a valid host after the scheme")
        return v


__all__ = ["ModelLlmEmbeddingRequest"]
