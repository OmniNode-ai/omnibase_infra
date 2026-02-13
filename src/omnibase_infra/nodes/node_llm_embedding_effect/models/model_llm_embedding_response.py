# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""LLM embedding response model with dimension uniformity validator.

This module provides ModelLlmEmbeddingResponse, the output model for the
LLM embedding effect node. It enforces that all returned embeddings have
the same vector dimensionality.

Related:
    - ModelEmbedding: Core embedding model from omnibase_core
    - ModelLlmUsage: Token usage tracking
    - ModelBackendResult: Backend operation result
    - OMN-2112: Phase 12 embedding node
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from omnibase_core.models.vector import ModelEmbedding
from omnibase_infra.models.model_backend_result import ModelBackendResult
from omnibase_infra.nodes.effects.models.model_llm_usage import ModelLlmUsage
from omnibase_infra.utils import validate_timezone_aware_datetime


class ModelLlmEmbeddingResponse(BaseModel):
    """Output model for the LLM embedding effect node.

    Captures the complete result of an embedding API call including
    generated vectors, token usage, timing, and tracing metadata.

    Invariants:
        - **Dimension Uniformity**: All embeddings in the response must
          have the same vector length (dimensionality).
        - **Backend Success**: ``backend_result.success`` must be ``True``.
          Failures are raised as exceptions, not encoded in the response.
        - **Non-Empty Embeddings**: At least one embedding must be present.

    Attributes:
        status: Always ``"success"``. Non-success states are exceptions.
        embeddings: Tuple of embedding vectors. Must be non-empty.
        dimensions: Dimensionality of the embedding vectors, derived from
            the first vector's length.
        model_used: Model identifier used for embedding generation.
        provider_id: Provider-specific ID, if available.
        usage: Token usage summary. Always present; zeros allowed.
        latency_ms: End-to-end latency in milliseconds.
        retry_count: Number of retries before success.
        backend_result: Backend operation result. Must have success=True.
        correlation_id: Distributed trace correlation ID.
        execution_id: Unique execution identifier.
        timestamp: Timezone-aware completion timestamp.

    Example:
        >>> from datetime import datetime, timezone
        >>> from uuid import uuid4
        >>> from omnibase_core.models.vector import ModelEmbedding
        >>> from omnibase_infra.models.model_backend_result import ModelBackendResult
        >>> from omnibase_infra.nodes.effects.models.model_llm_usage import ModelLlmUsage
        >>> resp = ModelLlmEmbeddingResponse(
        ...     embeddings=(
        ...         ModelEmbedding(id="0", vector=[0.1, 0.2, 0.3]),
        ...     ),
        ...     dimensions=3,
        ...     model_used="gte-qwen2-1.5b",
        ...     usage=ModelLlmUsage(tokens_input=5),
        ...     latency_ms=50.0,
        ...     backend_result=ModelBackendResult(success=True, duration_ms=45.0),
        ...     correlation_id=uuid4(),
        ...     execution_id=uuid4(),
        ...     timestamp=datetime.now(timezone.utc),
        ... )
        >>> resp.dimensions
        3
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    # Status
    status: Literal["success"] = Field(
        default="success",
        description="Always 'success'. Errors are raised as exceptions.",
    )

    # Embeddings
    embeddings: tuple[ModelEmbedding, ...] = Field(
        ...,
        min_length=1,
        description="Embedding vectors. Must be non-empty.",
    )
    dimensions: int = Field(
        ...,
        ge=1,
        description="Dimensionality of the embedding vectors.",
    )

    # Model info
    model_used: str = Field(
        ...,
        min_length=1,
        description="Model identifier used for embedding generation.",
    )
    provider_id: str | None = Field(
        default=None,
        description="Provider-specific ID, if available.",
    )

    # Tokens
    usage: ModelLlmUsage = Field(
        ...,
        description="Token usage summary. Always present; zeros allowed.",
    )

    # Timing
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="End-to-end latency in milliseconds.",
    )
    retry_count: int = Field(
        default=0,
        ge=0,
        description="Number of retries before success.",
    )

    # Backend
    backend_result: ModelBackendResult = Field(
        ...,
        description="Backend operation result. Must have success=True.",
    )

    # Tracing
    correlation_id: UUID = Field(
        ...,
        description="Distributed trace correlation ID.",
    )
    execution_id: UUID = Field(
        ...,
        description="Unique execution identifier.",
    )
    timestamp: datetime = Field(
        ...,
        description="Timezone-aware completion timestamp.",
    )

    @field_validator("timestamp")
    @classmethod
    def _validate_timestamp(cls, v: datetime) -> datetime:
        """Reject naive datetimes and preserve tzinfo."""
        return validate_timezone_aware_datetime(v)

    @model_validator(mode="after")
    def _validate_uniform_dimensions(self) -> Self:
        """Enforce that all embeddings have the same vector dimensionality.

        Compares every embedding's vector length against the declared
        ``dimensions`` field. Mixed dimensionalities indicate a provider
        bug or response parsing error and must be rejected.
        """
        lengths = {len(e.vector) for e in self.embeddings}
        if len(lengths) > 1:
            msg = f"Embedding dimension mismatch: found lengths {sorted(lengths)}"
            raise ValueError(msg)
        actual_dim = lengths.pop()
        if actual_dim != self.dimensions:
            msg = (
                f"Declared dimensions ({self.dimensions}) does not match "
                f"actual vector length ({actual_dim})"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _enforce_backend_success(self) -> Self:
        """Enforce that backend_result reports success.

        Non-success backend results should be raised as exceptions by the
        handler, not encoded in the response model.
        """
        if not self.backend_result.success:
            msg = "backend_result.success must be True; errors are raised as exceptions"
            raise ValueError(msg)
        return self


__all__: list[str] = ["ModelLlmEmbeddingResponse"]
