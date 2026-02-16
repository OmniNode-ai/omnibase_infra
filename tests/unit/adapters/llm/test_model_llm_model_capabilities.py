# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for ModelLlmModelCapabilities."""

from __future__ import annotations

import pytest

from omnibase_infra.adapters.llm.model_llm_model_capabilities import (
    ModelLlmModelCapabilities,
)


class TestModelLlmModelCapabilities:
    """Tests for the ModelLlmModelCapabilities Pydantic model."""

    def test_full_capabilities(self) -> None:
        """Model with all capabilities specified."""
        caps = ModelLlmModelCapabilities(
            model_name="qwen2.5-coder-14b",
            supports_streaming=True,
            supports_function_calling=True,
            max_context_length=32768,
            supported_modalities=["text"],
            cost_per_1k_input_tokens=0.0,
            cost_per_1k_output_tokens=0.0,
        )
        assert caps.model_name == "qwen2.5-coder-14b"
        assert caps.supports_streaming is True
        assert caps.supports_function_calling is True
        assert caps.max_context_length == 32768
        assert "text" in caps.supported_modalities

    def test_defaults(self) -> None:
        """Default values are sensible."""
        caps = ModelLlmModelCapabilities(model_name="test-model")
        assert caps.supports_streaming is True
        assert caps.supports_function_calling is False
        assert caps.max_context_length == 4096
        assert caps.supported_modalities == ["text"]
        assert caps.cost_per_1k_input_tokens == 0.0
        assert caps.cost_per_1k_output_tokens == 0.0

    def test_frozen_model(self) -> None:
        """Model is immutable."""
        caps = ModelLlmModelCapabilities(model_name="test")
        with pytest.raises(Exception):
            caps.model_name = "other"  # type: ignore[misc]

    def test_multimodal_capabilities(self) -> None:
        """Vision model with multiple modalities."""
        caps = ModelLlmModelCapabilities(
            model_name="qwen2-vl",
            supported_modalities=["text", "vision"],
            max_context_length=65536,
        )
        assert "vision" in caps.supported_modalities

    def test_satisfies_protocol(self) -> None:
        """Verify structural compatibility with ProtocolModelCapabilities."""
        caps = ModelLlmModelCapabilities(
            model_name="test",
            supports_streaming=True,
            supports_function_calling=False,
            max_context_length=4096,
            supported_modalities=["text"],
        )
        assert isinstance(caps.model_name, str)
        assert isinstance(caps.supports_streaming, bool)
        assert isinstance(caps.supports_function_calling, bool)
        assert isinstance(caps.max_context_length, int)
        assert isinstance(caps.supported_modalities, list)
