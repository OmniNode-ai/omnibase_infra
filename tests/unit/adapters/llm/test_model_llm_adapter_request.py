# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for ModelLlmAdapterRequest."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.adapters.llm.model_llm_adapter_request import (
    ModelLlmAdapterRequest,
)
from omnibase_spi.protocols.types.protocol_llm_types import ProtocolLLMRequest


class TestModelLlmAdapterRequest:
    """Tests for the ModelLlmAdapterRequest Pydantic model."""

    def test_basic_request(self) -> None:
        """Basic request with required fields."""
        req = ModelLlmAdapterRequest(
            prompt="Hello, world!",
            model_name="qwen2.5-coder-14b",
        )
        assert req.prompt == "Hello, world!"
        assert req.model_name == "qwen2.5-coder-14b"
        assert req.parameters == {}
        assert req.max_tokens is None
        assert req.temperature is None

    def test_request_with_parameters(self) -> None:
        """Request with all optional fields."""
        req = ModelLlmAdapterRequest(
            prompt="Explain ONEX",
            model_name="gpt-4",
            parameters={"top_p": 0.9},
            max_tokens=500,
            temperature=0.7,
        )
        assert req.max_tokens == 500
        assert req.temperature == 0.7
        assert req.parameters == {"top_p": 0.9}

    def test_frozen_model(self) -> None:
        """Model is immutable."""
        req = ModelLlmAdapterRequest(
            prompt="Hello",
            model_name="test",
        )
        with pytest.raises(ValidationError):
            req.prompt = "Changed"  # type: ignore[misc]

    def test_empty_prompt_rejected(self) -> None:
        """Empty prompt is rejected."""
        with pytest.raises(ValidationError):
            ModelLlmAdapterRequest(
                prompt="",
                model_name="test",
            )

    def test_temperature_bounds(self) -> None:
        """Temperature must be 0.0 to 2.0."""
        with pytest.raises(ValidationError):
            ModelLlmAdapterRequest(
                prompt="Hello",
                model_name="test",
                temperature=-0.1,
            )
        with pytest.raises(ValidationError):
            ModelLlmAdapterRequest(
                prompt="Hello",
                model_name="test",
                temperature=2.1,
            )

    def test_default_parameters_is_empty_dict(self) -> None:
        """parameters defaults to an empty dict."""
        req = ModelLlmAdapterRequest(prompt="Hello", model_name="test")
        assert req.parameters == {}
        assert isinstance(req.parameters, dict)

    def test_default_max_tokens_is_none(self) -> None:
        """max_tokens defaults to None (provider default)."""
        req = ModelLlmAdapterRequest(prompt="Hello", model_name="test")
        assert req.max_tokens is None

    def test_default_temperature_is_none(self) -> None:
        """temperature defaults to None (provider default)."""
        req = ModelLlmAdapterRequest(prompt="Hello", model_name="test")
        assert req.temperature is None

    def test_satisfies_protocol(self) -> None:
        """Verify structural compatibility with ProtocolLLMRequest."""
        req = ModelLlmAdapterRequest(
            prompt="Test",
            model_name="test-model",
            parameters={"key": "value"},
            max_tokens=100,
            temperature=0.5,
        )
        assert isinstance(req, ProtocolLLMRequest)
        assert isinstance(req.prompt, str)
        assert isinstance(req.model_name, str)
        assert isinstance(req.parameters, dict)

    def test_default_timeout_seconds(self) -> None:
        """timeout_seconds defaults to 30.0 when not provided.

        Mirrors ModelLlmInferenceRequest.timeout_seconds default so that
        the adapter does not silently shorten or lengthen the contract
        timeout for callers that omit the field.
        """
        req = ModelLlmAdapterRequest(prompt="Hello", model_name="test")
        assert req.timeout_seconds == 30.0

    def test_explicit_timeout_seconds_preserved(self) -> None:
        """Caller-provided timeout_seconds is preserved verbatim."""
        req = ModelLlmAdapterRequest(
            prompt="Hello",
            model_name="test",
            timeout_seconds=120.0,
        )
        assert req.timeout_seconds == 120.0

    def test_timeout_seconds_lower_bound(self) -> None:
        """timeout_seconds below 1.0 is rejected."""
        with pytest.raises(ValidationError):
            ModelLlmAdapterRequest(
                prompt="Hello",
                model_name="test",
                timeout_seconds=0.5,
            )

    def test_timeout_seconds_upper_bound(self) -> None:
        """timeout_seconds above 600.0 is rejected."""
        with pytest.raises(ValidationError):
            ModelLlmAdapterRequest(
                prompt="Hello",
                model_name="test",
                timeout_seconds=600.1,
            )

    def test_timeout_seconds_boundary_values_accepted(self) -> None:
        """Exact bounds 1.0 and 600.0 are accepted."""
        req_min = ModelLlmAdapterRequest(
            prompt="Hello",
            model_name="test",
            timeout_seconds=1.0,
        )
        req_max = ModelLlmAdapterRequest(
            prompt="Hello",
            model_name="test",
            timeout_seconds=600.0,
        )
        assert req_min.timeout_seconds == 1.0
        assert req_max.timeout_seconds == 600.0
