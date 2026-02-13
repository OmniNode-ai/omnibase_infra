# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ModelLlmEmbeddingRequest validation.

Verifies:
- Required field validation (base_url, model, texts)
- base_url scheme and host validation
- texts tuple boundary conditions (empty, single, max 2048)
- Empty/whitespace-only text rejection
- stream=True type-level guard
- Frozen immutability
- Default field values (timeout_seconds, max_retries, correlation_id, etc.)
- extra="forbid" enforcement
- dimensions validation (ge=1)

Related:
    - OMN-2113: Phase 13 embedding tests
    - OMN-2112: Phase 12 embedding node
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.node_llm_embedding_effect.models.model_llm_embedding_request import (
    ModelLlmEmbeddingRequest,
)

pytestmark = [pytest.mark.unit]


# =============================================================================
# Fixtures
# =============================================================================


def _valid_kwargs() -> dict[str, object]:
    """Return minimal valid kwargs for constructing a request."""
    return {
        "base_url": "http://192.168.86.201:8002",
        "model": "gte-qwen2-1.5b",
        "texts": ("Hello, world!",),
    }


# =============================================================================
# Happy Path Tests
# =============================================================================


class TestModelLlmEmbeddingRequestConstruction:
    """Tests for valid construction of ModelLlmEmbeddingRequest."""

    def test_minimal_valid_construction(self) -> None:
        """Minimal kwargs produce a valid request with defaults."""
        req = ModelLlmEmbeddingRequest(**_valid_kwargs())
        assert req.base_url == "http://192.168.86.201:8002"
        assert req.model == "gte-qwen2-1.5b"
        assert req.texts == ("Hello, world!",)
        assert req.dimensions is None
        assert req.stream is False
        assert req.timeout_seconds == 30.0
        assert req.max_retries == 3
        assert req.provider_label == ""
        assert isinstance(req.correlation_id, UUID)
        assert isinstance(req.execution_id, UUID)
        assert req.metadata == ()

    def test_single_text(self) -> None:
        """Single-text tuple is valid (min_length=1)."""
        req = ModelLlmEmbeddingRequest(**_valid_kwargs())
        assert len(req.texts) == 1

    def test_batch_texts(self) -> None:
        """Multiple texts are accepted."""
        kwargs = _valid_kwargs()
        kwargs["texts"] = ("text one", "text two", "text three")
        req = ModelLlmEmbeddingRequest(**kwargs)
        assert len(req.texts) == 3

    def test_2048_texts_boundary(self) -> None:
        """2048 texts is the maximum allowed (max_length=2048)."""
        kwargs = _valid_kwargs()
        kwargs["texts"] = tuple(f"text_{i}" for i in range(2048))
        req = ModelLlmEmbeddingRequest(**kwargs)
        assert len(req.texts) == 2048

    def test_https_base_url(self) -> None:
        """HTTPS scheme is accepted."""
        kwargs = _valid_kwargs()
        kwargs["base_url"] = "https://api.openai.com"
        req = ModelLlmEmbeddingRequest(**kwargs)
        assert req.base_url == "https://api.openai.com"

    def test_base_url_with_trailing_slash(self) -> None:
        """Trailing slash in base_url is preserved (stripped at handler level)."""
        kwargs = _valid_kwargs()
        kwargs["base_url"] = "http://localhost:8002/"
        req = ModelLlmEmbeddingRequest(**kwargs)
        assert req.base_url == "http://localhost:8002/"

    def test_explicit_dimensions(self) -> None:
        """Explicit dimensions value is accepted."""
        kwargs = _valid_kwargs()
        kwargs["dimensions"] = 768
        req = ModelLlmEmbeddingRequest(**kwargs)
        assert req.dimensions == 768

    def test_custom_resilience_params(self) -> None:
        """Custom timeout and retry values are accepted."""
        kwargs = _valid_kwargs()
        kwargs["timeout_seconds"] = 60.0
        kwargs["max_retries"] = 5
        req = ModelLlmEmbeddingRequest(**kwargs)
        assert req.timeout_seconds == 60.0
        assert req.max_retries == 5

    def test_explicit_correlation_id(self) -> None:
        """Caller-provided correlation_id overrides the default."""
        cid = uuid4()
        kwargs = _valid_kwargs()
        kwargs["correlation_id"] = cid
        req = ModelLlmEmbeddingRequest(**kwargs)
        assert req.correlation_id == cid

    def test_metadata_key_value_pairs(self) -> None:
        """Metadata accepts tuple-of-tuples key-value pairs."""
        kwargs = _valid_kwargs()
        kwargs["metadata"] = (("key1", "val1"), ("key2", "val2"))
        req = ModelLlmEmbeddingRequest(**kwargs)
        assert dict(req.metadata) == {"key1": "val1", "key2": "val2"}

    def test_provider_label(self) -> None:
        """Provider label is accepted."""
        kwargs = _valid_kwargs()
        kwargs["provider_label"] = "vllm-gte"
        req = ModelLlmEmbeddingRequest(**kwargs)
        assert req.provider_label == "vllm-gte"


# =============================================================================
# Immutability Tests
# =============================================================================


class TestModelLlmEmbeddingRequestImmutability:
    """Tests for frozen=True immutability enforcement."""

    def test_frozen_base_url(self) -> None:
        """Cannot reassign base_url after construction."""
        req = ModelLlmEmbeddingRequest(**_valid_kwargs())
        with pytest.raises(ValidationError):
            req.base_url = "http://other:8000"  # type: ignore[misc]

    def test_frozen_texts(self) -> None:
        """Cannot reassign texts after construction."""
        req = ModelLlmEmbeddingRequest(**_valid_kwargs())
        with pytest.raises(ValidationError):
            req.texts = ("new text",)  # type: ignore[misc]


# =============================================================================
# base_url Validation Tests
# =============================================================================


class TestBaseUrlValidation:
    """Tests for base_url field validator."""

    def test_missing_scheme_rejected(self) -> None:
        """base_url without http:// or https:// is rejected."""
        kwargs = _valid_kwargs()
        kwargs["base_url"] = "192.168.86.201:8002"
        with pytest.raises(ValidationError, match="base_url must start with http"):
            ModelLlmEmbeddingRequest(**kwargs)

    def test_ftp_scheme_rejected(self) -> None:
        """Non-HTTP scheme is rejected."""
        kwargs = _valid_kwargs()
        kwargs["base_url"] = "ftp://192.168.86.201:8002"
        with pytest.raises(ValidationError, match="base_url must start with http"):
            ModelLlmEmbeddingRequest(**kwargs)

    def test_scheme_only_rejected(self) -> None:
        """Scheme without host is rejected."""
        kwargs = _valid_kwargs()
        kwargs["base_url"] = "http://"
        with pytest.raises(ValidationError, match="host"):
            ModelLlmEmbeddingRequest(**kwargs)

    def test_scheme_with_only_slashes_rejected(self) -> None:
        """Scheme with only trailing slashes is rejected."""
        kwargs = _valid_kwargs()
        kwargs["base_url"] = "http:///"
        with pytest.raises(ValidationError, match="host"):
            ModelLlmEmbeddingRequest(**kwargs)

    def test_empty_base_url_rejected(self) -> None:
        """Empty string base_url is rejected."""
        kwargs = _valid_kwargs()
        kwargs["base_url"] = ""
        with pytest.raises(ValidationError):
            ModelLlmEmbeddingRequest(**kwargs)


# =============================================================================
# texts Validation Tests
# =============================================================================


class TestTextsValidation:
    """Tests for texts field and validator."""

    def test_empty_tuple_rejected(self) -> None:
        """Empty texts tuple is rejected (min_length=1)."""
        kwargs = _valid_kwargs()
        kwargs["texts"] = ()
        with pytest.raises(ValidationError):
            ModelLlmEmbeddingRequest(**kwargs)

    def test_2049_texts_rejected(self) -> None:
        """2049 texts exceeds max_length=2048."""
        kwargs = _valid_kwargs()
        kwargs["texts"] = tuple(f"text_{i}" for i in range(2049))
        with pytest.raises(ValidationError):
            ModelLlmEmbeddingRequest(**kwargs)

    def test_whitespace_only_text_rejected(self) -> None:
        """Whitespace-only text entry is rejected."""
        kwargs = _valid_kwargs()
        kwargs["texts"] = ("   ",)
        with pytest.raises(ValidationError, match="non-empty and non-whitespace"):
            ModelLlmEmbeddingRequest(**kwargs)

    def test_empty_string_text_rejected(self) -> None:
        """Empty string text entry is rejected."""
        kwargs = _valid_kwargs()
        kwargs["texts"] = ("",)
        with pytest.raises(ValidationError, match="non-empty and non-whitespace"):
            ModelLlmEmbeddingRequest(**kwargs)

    def test_mixed_valid_and_whitespace_rejected(self) -> None:
        """If any text is whitespace-only, validation fails with index reference."""
        kwargs = _valid_kwargs()
        kwargs["texts"] = ("valid text", "  \t  ")
        with pytest.raises(ValidationError, match=r"texts\[1\]"):
            ModelLlmEmbeddingRequest(**kwargs)

    def test_newline_only_text_rejected(self) -> None:
        """Newline-only text entry is rejected."""
        kwargs = _valid_kwargs()
        kwargs["texts"] = ("\n\n",)
        with pytest.raises(ValidationError, match="non-empty and non-whitespace"):
            ModelLlmEmbeddingRequest(**kwargs)


# =============================================================================
# stream Guard Tests
# =============================================================================


class TestStreamGuard:
    """Tests for the stream=True type-level guard (Literal[False])."""

    def test_stream_true_rejected(self) -> None:
        """stream=True is rejected by Literal[False] type guard."""
        kwargs = _valid_kwargs()
        kwargs["stream"] = True
        with pytest.raises(ValidationError):
            ModelLlmEmbeddingRequest(**kwargs)

    def test_stream_default_is_false(self) -> None:
        """Default stream value is False."""
        req = ModelLlmEmbeddingRequest(**_valid_kwargs())
        assert req.stream is False


# =============================================================================
# dimensions Validation Tests
# =============================================================================


class TestDimensionsValidation:
    """Tests for dimensions field validation."""

    def test_dimensions_zero_rejected(self) -> None:
        """dimensions=0 is rejected (ge=1)."""
        kwargs = _valid_kwargs()
        kwargs["dimensions"] = 0
        with pytest.raises(ValidationError):
            ModelLlmEmbeddingRequest(**kwargs)

    def test_dimensions_negative_rejected(self) -> None:
        """Negative dimensions are rejected."""
        kwargs = _valid_kwargs()
        kwargs["dimensions"] = -1
        with pytest.raises(ValidationError):
            ModelLlmEmbeddingRequest(**kwargs)

    def test_dimensions_one_valid(self) -> None:
        """dimensions=1 is the minimum valid value."""
        kwargs = _valid_kwargs()
        kwargs["dimensions"] = 1
        req = ModelLlmEmbeddingRequest(**kwargs)
        assert req.dimensions == 1


# =============================================================================
# Resilience Parameter Validation Tests
# =============================================================================


class TestResilienceParameterValidation:
    """Tests for timeout_seconds and max_retries bounds."""

    def test_timeout_below_minimum_rejected(self) -> None:
        """timeout_seconds below 1.0 is rejected."""
        kwargs = _valid_kwargs()
        kwargs["timeout_seconds"] = 0.5
        with pytest.raises(ValidationError):
            ModelLlmEmbeddingRequest(**kwargs)

    def test_timeout_above_maximum_rejected(self) -> None:
        """timeout_seconds above 600.0 is rejected."""
        kwargs = _valid_kwargs()
        kwargs["timeout_seconds"] = 601.0
        with pytest.raises(ValidationError):
            ModelLlmEmbeddingRequest(**kwargs)

    def test_max_retries_negative_rejected(self) -> None:
        """Negative max_retries is rejected."""
        kwargs = _valid_kwargs()
        kwargs["max_retries"] = -1
        with pytest.raises(ValidationError):
            ModelLlmEmbeddingRequest(**kwargs)

    def test_max_retries_above_ten_rejected(self) -> None:
        """max_retries > 10 is rejected."""
        kwargs = _valid_kwargs()
        kwargs["max_retries"] = 11
        with pytest.raises(ValidationError):
            ModelLlmEmbeddingRequest(**kwargs)

    def test_max_retries_zero_valid(self) -> None:
        """max_retries=0 means no retries (single attempt)."""
        kwargs = _valid_kwargs()
        kwargs["max_retries"] = 0
        req = ModelLlmEmbeddingRequest(**kwargs)
        assert req.max_retries == 0

    def test_max_retries_ten_valid(self) -> None:
        """max_retries=10 is the maximum allowed."""
        kwargs = _valid_kwargs()
        kwargs["max_retries"] = 10
        req = ModelLlmEmbeddingRequest(**kwargs)
        assert req.max_retries == 10


# =============================================================================
# extra="forbid" Tests
# =============================================================================


class TestExtraFieldsRejected:
    """Tests for extra='forbid' enforcement."""

    def test_unknown_field_rejected(self) -> None:
        """Unknown field is rejected by extra='forbid'."""
        kwargs = _valid_kwargs()
        kwargs["unknown_field"] = "surprise"
        with pytest.raises(ValidationError):
            ModelLlmEmbeddingRequest(**kwargs)


# =============================================================================
# Required Field Tests
# =============================================================================


class TestRequiredFields:
    """Tests for required field validation."""

    def test_missing_base_url_rejected(self) -> None:
        """Missing base_url raises ValidationError."""
        kwargs = _valid_kwargs()
        del kwargs["base_url"]
        with pytest.raises(ValidationError):
            ModelLlmEmbeddingRequest(**kwargs)

    def test_missing_model_rejected(self) -> None:
        """Missing model raises ValidationError."""
        kwargs = _valid_kwargs()
        del kwargs["model"]
        with pytest.raises(ValidationError):
            ModelLlmEmbeddingRequest(**kwargs)

    def test_missing_texts_rejected(self) -> None:
        """Missing texts raises ValidationError."""
        kwargs = _valid_kwargs()
        del kwargs["texts"]
        with pytest.raises(ValidationError):
            ModelLlmEmbeddingRequest(**kwargs)

    def test_empty_model_rejected(self) -> None:
        """Empty string model is rejected (min_length=1)."""
        kwargs = _valid_kwargs()
        kwargs["model"] = ""
        with pytest.raises(ValidationError):
            ModelLlmEmbeddingRequest(**kwargs)
