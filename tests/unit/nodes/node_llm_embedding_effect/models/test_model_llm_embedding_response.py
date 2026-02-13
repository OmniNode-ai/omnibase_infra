# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ModelLlmEmbeddingResponse and dimension uniformity validator.

Verifies:
- Dimension uniformity: all embeddings must have the same vector length
- Mixed dimensions [768, 1024] raises ValueError
- Declared dimensions must match actual vector length
- backend_result.success must be True
- Frozen immutability
- Non-empty embeddings requirement
- Timestamp timezone awareness
- extra="forbid" enforcement

Related:
    - OMN-2113: Phase 13 embedding tests
    - OMN-2112: Phase 12 embedding node
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_core.models.vector import ModelEmbedding
from omnibase_infra.models.model_backend_result import ModelBackendResult
from omnibase_infra.nodes.effects.models.model_llm_usage import ModelLlmUsage
from omnibase_infra.nodes.node_llm_embedding_effect.models.model_llm_embedding_response import (
    ModelLlmEmbeddingResponse,
)

pytestmark = [pytest.mark.unit]


# =============================================================================
# Fixtures
# =============================================================================


def _make_embedding(idx: int, dim: int = 768) -> ModelEmbedding:
    """Create a test ModelEmbedding with given index and dimension."""
    return ModelEmbedding(
        id=str(idx),
        vector=[0.1 * (i + 1) for i in range(dim)],
        metadata={},
    )


def _valid_kwargs(dim: int = 3, count: int = 1) -> dict[str, object]:
    """Return minimal valid kwargs for constructing a response."""
    embeddings = tuple(_make_embedding(i, dim) for i in range(count))
    return {
        "embeddings": embeddings,
        "dimensions": dim,
        "model_used": "gte-qwen2-1.5b",
        "usage": ModelLlmUsage(tokens_input=5),
        "latency_ms": 50.0,
        "backend_result": ModelBackendResult(success=True, duration_ms=45.0),
        "correlation_id": uuid4(),
        "execution_id": uuid4(),
        "timestamp": datetime.now(UTC),
    }


# =============================================================================
# Happy Path Tests
# =============================================================================


class TestModelLlmEmbeddingResponseConstruction:
    """Tests for valid construction of ModelLlmEmbeddingResponse."""

    def test_minimal_valid_construction(self) -> None:
        """Minimal valid kwargs produce a valid response."""
        resp = ModelLlmEmbeddingResponse(**_valid_kwargs())
        assert resp.status == "success"
        assert len(resp.embeddings) == 1
        assert resp.dimensions == 3
        assert resp.model_used == "gte-qwen2-1.5b"
        assert resp.retry_count == 0
        assert resp.provider_id is None

    def test_batch_embeddings(self) -> None:
        """Multiple embeddings with uniform dimensions are valid."""
        resp = ModelLlmEmbeddingResponse(**_valid_kwargs(dim=768, count=5))
        assert len(resp.embeddings) == 5
        assert resp.dimensions == 768

    def test_provider_id_present(self) -> None:
        """provider_id can be set."""
        kwargs = _valid_kwargs()
        kwargs["provider_id"] = "some-provider-id"
        resp = ModelLlmEmbeddingResponse(**kwargs)
        assert resp.provider_id == "some-provider-id"

    def test_retry_count_set(self) -> None:
        """retry_count can be set."""
        kwargs = _valid_kwargs()
        kwargs["retry_count"] = 2
        resp = ModelLlmEmbeddingResponse(**kwargs)
        assert resp.retry_count == 2

    def test_timestamp_with_explicit_utc(self) -> None:
        """Timestamp with UTC timezone is accepted."""
        kwargs = _valid_kwargs()
        kwargs["timestamp"] = datetime(2025, 6, 15, 12, 0, 0, tzinfo=UTC)
        resp = ModelLlmEmbeddingResponse(**kwargs)
        assert resp.timestamp.tzinfo is not None

    def test_timestamp_with_non_utc_tz(self) -> None:
        """Timestamp with non-UTC timezone is accepted (tzinfo present)."""
        kwargs = _valid_kwargs()
        est = timezone(timedelta(hours=-5))
        kwargs["timestamp"] = datetime(2025, 6, 15, 12, 0, 0, tzinfo=est)
        resp = ModelLlmEmbeddingResponse(**kwargs)
        assert resp.timestamp.tzinfo is not None
        assert resp.timestamp.tzinfo == est


# =============================================================================
# Dimension Uniformity Validator Tests
# =============================================================================


class TestDimensionUniformityValidator:
    """Tests for _validate_uniform_dimensions model validator."""

    def test_uniform_dimensions_pass(self) -> None:
        """All embeddings with same vector length pass validation."""
        resp = ModelLlmEmbeddingResponse(**_valid_kwargs(dim=768, count=3))
        assert resp.dimensions == 768

    def test_single_embedding_passes(self) -> None:
        """Single embedding trivially passes uniformity check."""
        resp = ModelLlmEmbeddingResponse(**_valid_kwargs(dim=1024, count=1))
        assert resp.dimensions == 1024

    def test_mixed_dimensions_768_1024_rejected(self) -> None:
        """Mixed dimensions [768, 1024] raise ValueError."""
        emb_768 = _make_embedding(0, dim=768)
        emb_1024 = _make_embedding(1, dim=1024)
        kwargs = _valid_kwargs()
        kwargs["embeddings"] = (emb_768, emb_1024)
        kwargs["dimensions"] = 768
        with pytest.raises(ValidationError, match="Embedding dimension mismatch"):
            ModelLlmEmbeddingResponse(**kwargs)

    def test_mixed_dimensions_three_different_rejected(self) -> None:
        """Three different dimensions are rejected."""
        emb_a = _make_embedding(0, dim=3)
        emb_b = _make_embedding(1, dim=5)
        emb_c = _make_embedding(2, dim=7)
        kwargs = _valid_kwargs()
        kwargs["embeddings"] = (emb_a, emb_b, emb_c)
        kwargs["dimensions"] = 3
        with pytest.raises(ValidationError, match="Embedding dimension mismatch"):
            ModelLlmEmbeddingResponse(**kwargs)

    def test_declared_dimensions_mismatch_actual(self) -> None:
        """Declared dimensions != actual vector length raises ValueError."""
        emb = _make_embedding(0, dim=768)
        kwargs = _valid_kwargs()
        kwargs["embeddings"] = (emb,)
        kwargs["dimensions"] = 512  # Declared 512 but vectors are 768
        with pytest.raises(
            ValidationError, match=r"Declared dimensions.*does not match"
        ):
            ModelLlmEmbeddingResponse(**kwargs)

    def test_uniform_with_many_embeddings(self) -> None:
        """Many embeddings with identical dimensions pass validation."""
        resp = ModelLlmEmbeddingResponse(**_valid_kwargs(dim=384, count=100))
        assert resp.dimensions == 384
        assert len(resp.embeddings) == 100


# =============================================================================
# Backend Result Validator Tests
# =============================================================================


class TestBackendSuccessValidator:
    """Tests for _enforce_backend_success model validator."""

    def test_backend_success_true_passes(self) -> None:
        """backend_result.success=True passes validation."""
        resp = ModelLlmEmbeddingResponse(**_valid_kwargs())
        assert resp.backend_result.success is True

    def test_backend_success_false_rejected(self) -> None:
        """backend_result.success=False raises ValueError."""
        kwargs = _valid_kwargs()
        kwargs["backend_result"] = ModelBackendResult(
            success=False,
            error="Something failed",
            duration_ms=10.0,
        )
        with pytest.raises(
            ValidationError, match=r"backend_result\.success must be True"
        ):
            ModelLlmEmbeddingResponse(**kwargs)


# =============================================================================
# Timestamp Validation Tests
# =============================================================================


class TestTimestampValidation:
    """Tests for timestamp timezone awareness validator."""

    def test_naive_datetime_rejected(self) -> None:
        """Naive datetime (no tzinfo) is rejected."""
        kwargs = _valid_kwargs()
        kwargs["timestamp"] = datetime(2025, 6, 15, 12, 0, 0)  # naive
        with pytest.raises(ValidationError):
            ModelLlmEmbeddingResponse(**kwargs)


# =============================================================================
# Empty Embeddings Tests
# =============================================================================


class TestEmptyEmbeddings:
    """Tests for non-empty embeddings requirement."""

    def test_empty_embeddings_rejected(self) -> None:
        """Empty embeddings tuple is rejected (min_length=1)."""
        kwargs = _valid_kwargs()
        kwargs["embeddings"] = ()
        with pytest.raises(ValidationError):
            ModelLlmEmbeddingResponse(**kwargs)


# =============================================================================
# Immutability Tests
# =============================================================================


class TestResponseImmutability:
    """Tests for frozen=True enforcement."""

    def test_cannot_reassign_embeddings(self) -> None:
        """Cannot reassign embeddings after construction."""
        resp = ModelLlmEmbeddingResponse(**_valid_kwargs())
        with pytest.raises(ValidationError):
            resp.embeddings = ()  # type: ignore[misc]

    def test_cannot_reassign_dimensions(self) -> None:
        """Cannot reassign dimensions after construction."""
        resp = ModelLlmEmbeddingResponse(**_valid_kwargs())
        with pytest.raises(ValidationError):
            resp.dimensions = 999  # type: ignore[misc]


# =============================================================================
# extra="forbid" Tests
# =============================================================================


class TestResponseExtraFieldsRejected:
    """Tests for extra='forbid' enforcement."""

    def test_unknown_field_rejected(self) -> None:
        """Unknown field is rejected by extra='forbid'."""
        kwargs = _valid_kwargs()
        kwargs["unknown_field"] = "surprise"
        with pytest.raises(ValidationError):
            ModelLlmEmbeddingResponse(**kwargs)


# =============================================================================
# Field Constraint Tests
# =============================================================================


class TestFieldConstraints:
    """Tests for individual field constraints."""

    def test_dimensions_zero_rejected(self) -> None:
        """dimensions=0 is rejected (ge=1)."""
        kwargs = _valid_kwargs()
        kwargs["dimensions"] = 0
        with pytest.raises(ValidationError):
            ModelLlmEmbeddingResponse(**kwargs)

    def test_latency_ms_negative_rejected(self) -> None:
        """Negative latency is rejected (ge=0.0)."""
        kwargs = _valid_kwargs()
        kwargs["latency_ms"] = -1.0
        with pytest.raises(ValidationError):
            ModelLlmEmbeddingResponse(**kwargs)

    def test_retry_count_negative_rejected(self) -> None:
        """Negative retry_count is rejected (ge=0)."""
        kwargs = _valid_kwargs()
        kwargs["retry_count"] = -1
        with pytest.raises(ValidationError):
            ModelLlmEmbeddingResponse(**kwargs)

    def test_model_used_empty_rejected(self) -> None:
        """Empty model_used is rejected (min_length=1)."""
        kwargs = _valid_kwargs()
        kwargs["model_used"] = ""
        with pytest.raises(ValidationError):
            ModelLlmEmbeddingResponse(**kwargs)

    def test_latency_zero_valid(self) -> None:
        """latency_ms=0.0 is valid."""
        kwargs = _valid_kwargs()
        kwargs["latency_ms"] = 0.0
        resp = ModelLlmEmbeddingResponse(**kwargs)
        assert resp.latency_ms == 0.0
