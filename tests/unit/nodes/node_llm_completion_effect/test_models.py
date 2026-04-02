# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for LLM completion effect models."""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_llm_completion_effect.models import (
    ModelLLMCompletionRequest,
    ModelLLMCompletionResult,
)
from omnibase_infra.nodes.node_llm_completion_effect.models.model_llm_completion_message import (
    ModelLLMCompletionMessage,
)


@pytest.mark.unit
class TestModelLLMCompletionMessage:
    def test_frozen(self) -> None:
        msg = ModelLLMCompletionMessage(role="user", content="hello")
        with pytest.raises(Exception):
            msg.role = "system"  # type: ignore[misc]

    def test_extra_forbid(self) -> None:
        with pytest.raises(Exception):
            ModelLLMCompletionMessage(role="user", content="hi", extra_field="bad")  # type: ignore[call-arg]


@pytest.mark.unit
class TestModelLLMCompletionRequest:
    def test_defaults(self) -> None:
        req = ModelLLMCompletionRequest(
            messages=(ModelLLMCompletionMessage(role="user", content="test"),),
        )
        assert req.model == ""
        assert req.max_tokens == 1024
        assert req.temperature == 0.7
        assert req.endpoint_url == ""
        assert req.correlation_id is not None

    def test_frozen(self) -> None:
        req = ModelLLMCompletionRequest(
            messages=(ModelLLMCompletionMessage(role="user", content="test"),),
        )
        with pytest.raises(Exception):
            req.model = "new-model"  # type: ignore[misc]

    def test_extra_forbid(self) -> None:
        with pytest.raises(Exception):
            ModelLLMCompletionRequest(
                messages=(ModelLLMCompletionMessage(role="user", content="test"),),
                bogus="nope",  # type: ignore[call-arg]
            )

    def test_temperature_bounds(self) -> None:
        with pytest.raises(Exception):
            ModelLLMCompletionRequest(
                messages=(ModelLLMCompletionMessage(role="user", content="test"),),
                temperature=3.0,
            )


@pytest.mark.unit
class TestModelLLMCompletionResult:
    def test_success(self) -> None:
        cid = uuid4()
        result = ModelLLMCompletionResult(
            correlation_id=cid,
            success=True,
            content="Hello world",
            model="test-model",
            prompt_tokens=10,
            completion_tokens=5,
        )
        assert result.success is True
        assert result.content == "Hello world"
        assert result.error_message == ""

    def test_failure(self) -> None:
        cid = uuid4()
        result = ModelLLMCompletionResult(
            correlation_id=cid,
            success=False,
            error_message="timeout",
        )
        assert result.success is False
        assert result.content == ""

    def test_frozen(self) -> None:
        result = ModelLLMCompletionResult(correlation_id=uuid4(), success=True)
        with pytest.raises(Exception):
            result.success = False  # type: ignore[misc]
