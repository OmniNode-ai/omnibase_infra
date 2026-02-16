# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for ModelLlmHealthResponse."""

from __future__ import annotations

import pytest

from omnibase_infra.adapters.llm.model_llm_health_response import (
    ModelLlmHealthResponse,
)


class TestModelLlmHealthResponse:
    """Tests for the ModelLlmHealthResponse Pydantic model."""

    def test_healthy_response(self) -> None:
        """Healthy response with models."""
        resp = ModelLlmHealthResponse(
            is_healthy=True,
            provider_name="openai-compatible",
            response_time_ms=42.5,
            available_models=["qwen2.5-coder-14b", "qwen2.5-7b"],
        )
        assert resp.is_healthy is True
        assert resp.provider_name == "openai-compatible"
        assert resp.response_time_ms == 42.5
        assert len(resp.available_models) == 2
        assert resp.error_message is None

    def test_unhealthy_response(self) -> None:
        """Unhealthy response with error message."""
        resp = ModelLlmHealthResponse(
            is_healthy=False,
            provider_name="ollama",
            response_time_ms=5000.0,
            error_message="Connection refused",
        )
        assert resp.is_healthy is False
        assert resp.error_message == "Connection refused"

    def test_frozen_model(self) -> None:
        """Model is immutable."""
        resp = ModelLlmHealthResponse(
            is_healthy=True,
            provider_name="test",
            response_time_ms=10.0,
        )
        with pytest.raises(Exception):
            resp.is_healthy = False  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields are rejected."""
        with pytest.raises(Exception):
            ModelLlmHealthResponse(
                is_healthy=True,
                provider_name="test",
                response_time_ms=10.0,
                unknown_field="value",  # type: ignore[call-arg]
            )

    def test_default_available_models(self) -> None:
        """Available models defaults to empty list."""
        resp = ModelLlmHealthResponse(
            is_healthy=True,
            provider_name="test",
            response_time_ms=1.0,
        )
        assert resp.available_models == []

    def test_negative_response_time_rejected(self) -> None:
        """Negative response time is rejected."""
        with pytest.raises(Exception):
            ModelLlmHealthResponse(
                is_healthy=True,
                provider_name="test",
                response_time_ms=-1.0,
            )

    def test_satisfies_protocol(self) -> None:
        """Verify structural compatibility with ProtocolLLMHealthResponse."""
        resp = ModelLlmHealthResponse(
            is_healthy=True,
            provider_name="test",
            response_time_ms=10.0,
            available_models=["model-a"],
            error_message=None,
        )
        # Protocol requires these properties
        assert isinstance(resp.is_healthy, bool)
        assert isinstance(resp.provider_name, str)
        assert isinstance(resp.response_time_ms, float)
        assert isinstance(resp.available_models, list)
