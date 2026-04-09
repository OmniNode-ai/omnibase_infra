# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerDelegationRouting (delta function).

Tests cover:
    - Routing for each task type (test, research, document)
    - Fast-path routing when prompt tokens <= 24K
    - Fallback when LLM_CODER_FAST_URL is not configured
    - Error on missing required endpoint
    - System prompt assignment per task type
    - Escalation: no local endpoints configured → claude tier

Related:
    - OMN-7040: Node-based delegation pipeline
    - OMN-8029: routing_tiers.yaml config-driven routing
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

import omnibase_infra.nodes.node_delegation_routing_reducer.handlers.handler_delegation_routing as _handler_mod
from omnibase_infra.nodes.node_delegation_orchestrator.models.model_delegation_request import (
    ModelDelegationRequest,
)
from omnibase_infra.nodes.node_delegation_routing_reducer.handlers.handler_delegation_routing import (
    delta,
)

pytestmark = [pytest.mark.unit]


def _request(
    task_type: str = "test",
    prompt: str = "Write unit tests for auth.py",
    **kwargs: object,
) -> ModelDelegationRequest:
    """Build a valid ModelDelegationRequest."""
    return ModelDelegationRequest(
        prompt=prompt,
        task_type=task_type,  # type: ignore[arg-type]
        correlation_id=uuid4(),
        emitted_at=datetime.now(tz=UTC),
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.fixture(autouse=True)
def reset_config_singleton() -> None:
    """Reset the module-level config singleton before each test."""
    _handler_mod._config = None
    yield
    _handler_mod._config = None


class TestRoutingByTaskType:
    """Verify correct model selection per task type from routing_tiers.yaml."""

    def test_test_routes_to_coder_when_no_fast_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without fast-path URL, test tasks go to qwen3-coder-30b."""
        monkeypatch.setenv("LLM_CODER_URL", "http://192.168.86.201:8000")
        monkeypatch.delenv("LLM_CODER_FAST_URL", raising=False)
        # Long prompt so fast-path threshold is irrelevant
        req = _request(task_type="test", prompt="x" * 200000)
        decision = delta(req)
        assert decision.selected_model == "qwen3-coder-30b"
        assert decision.endpoint_url == "http://192.168.86.201:8000"
        assert decision.cost_tier == "low"
        assert decision.max_context_tokens == 65536

    def test_research_routes_to_coder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_CODER_URL", "http://192.168.86.201:8000")
        monkeypatch.delenv("LLM_CODER_FAST_URL", raising=False)
        req = _request(task_type="research", prompt="x" * 200000)
        decision = delta(req)
        assert decision.selected_model == "qwen3-coder-30b"
        assert decision.endpoint_url == "http://192.168.86.201:8000"

    def test_document_routes_to_deepseek(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Document tasks go to deepseek-r1-14b via LLM_CODER_FAST_URL."""
        monkeypatch.setenv("LLM_CODER_FAST_URL", "http://192.168.86.201:8001")
        req = _request(task_type="document")
        decision = delta(req)
        assert decision.selected_model == "deepseek-r1-14b"
        assert decision.endpoint_url == "http://192.168.86.201:8001"


class TestFastPathRouting:
    """Verify token-count based fast-path optimization."""

    def test_short_prompt_uses_fast_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_CODER_URL", "http://192.168.86.201:8000")
        monkeypatch.setenv("LLM_CODER_FAST_URL", "http://192.168.86.201:8001")
        # Short prompt (~10 tokens) should use fast path
        req = _request(task_type="test", prompt="Write tests for auth.py")
        decision = delta(req)
        assert decision.selected_model == "deepseek-r1-14b"
        assert decision.endpoint_url == "http://192.168.86.201:8001"
        assert decision.max_context_tokens == 24576

    def test_long_prompt_skips_fast_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_CODER_URL", "http://192.168.86.201:8000")
        monkeypatch.setenv("LLM_CODER_FAST_URL", "http://192.168.86.201:8001")
        # Prompt > 40K tokens (~160K chars) should skip fast path
        long_prompt = "x" * 200000
        req = _request(task_type="test", prompt=long_prompt)
        decision = delta(req)
        assert decision.selected_model == "qwen3-coder-30b"

    def test_fast_path_not_available_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_CODER_URL", "http://192.168.86.201:8000")
        monkeypatch.delenv("LLM_CODER_FAST_URL", raising=False)
        req = _request(task_type="test", prompt="short prompt")
        decision = delta(req)
        assert decision.selected_model == "qwen3-coder-30b"

    def test_document_uses_deepseek_fast_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Document tasks use deepseek-r1-14b (fast path in local tier)."""
        monkeypatch.setenv("LLM_CODER_FAST_URL", "http://192.168.86.201:8001")
        monkeypatch.delenv("LLM_DEEPSEEK_R1_URL", raising=False)
        req = _request(task_type="document", prompt="short prompt")
        decision = delta(req)
        # deepseek-r1-14b handles document in local tier
        assert decision.selected_model == "deepseek-r1-14b"


class TestMissingEndpoint:
    """Verify error when no tier has a configured endpoint for the task."""

    def test_missing_all_local_urls_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When all local and cloud env vars absent, only claude tier remains.

        If ANTHROPIC_API_KEY is also missing, should raise ValueError.
        """
        monkeypatch.delenv("LLM_CODER_URL", raising=False)
        monkeypatch.delenv("LLM_CODER_FAST_URL", raising=False)
        monkeypatch.delenv("LLM_DEEPSEEK_R1_URL", raising=False)
        monkeypatch.delenv("LLM_GLM_URL", raising=False)
        monkeypatch.delenv("LLM_GEMINI_FLASH_URL", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        req = _request(task_type="test")
        with pytest.raises(ValueError, match="No tier"):
            delta(req)

    def test_missing_deepseek_escalates_to_claude(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If local tier is unavailable for document, claude tier takes it."""
        monkeypatch.delenv("LLM_CODER_URL", raising=False)
        monkeypatch.delenv("LLM_CODER_FAST_URL", raising=False)
        monkeypatch.delenv("LLM_DEEPSEEK_R1_URL", raising=False)
        monkeypatch.delenv("LLM_GLM_URL", raising=False)
        monkeypatch.delenv("LLM_GEMINI_FLASH_URL", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        req = _request(task_type="document")
        decision = delta(req)
        assert decision.selected_model == "claude-sonnet-4-6"
        assert decision.cost_tier == "high"


class TestEscalationToClaudeTier:
    """Verify escalation from local → cheap_cloud → claude."""

    def test_escalates_to_claude_when_local_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LLM_CODER_URL", raising=False)
        monkeypatch.delenv("LLM_CODER_FAST_URL", raising=False)
        monkeypatch.delenv("LLM_DEEPSEEK_R1_URL", raising=False)
        monkeypatch.delenv("LLM_GLM_URL", raising=False)
        monkeypatch.delenv("LLM_GEMINI_FLASH_URL", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        req = _request(task_type="test")
        decision = delta(req)
        assert decision.selected_model == "claude-sonnet-4-6"
        assert decision.cost_tier == "high"


class TestSystemPrompts:
    """Verify system prompt assignment."""

    def test_test_system_prompt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_CODER_URL", "http://192.168.86.201:8000")
        monkeypatch.delenv("LLM_CODER_FAST_URL", raising=False)
        req = _request(task_type="test", prompt="x" * 200000)
        decision = delta(req)
        assert "test generation" in decision.system_prompt.lower()

    def test_document_system_prompt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_CODER_FAST_URL", "http://192.168.86.201:8001")
        req = _request(task_type="document")
        decision = delta(req)
        assert "documentation" in decision.system_prompt.lower()

    def test_research_system_prompt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_CODER_URL", "http://192.168.86.201:8000")
        monkeypatch.delenv("LLM_CODER_FAST_URL", raising=False)
        req = _request(task_type="research", prompt="x" * 200000)
        decision = delta(req)
        assert "research" in decision.system_prompt.lower()


class TestCorrelationIdPreserved:
    """Verify correlation_id flows through."""

    def test_correlation_id_matches_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_CODER_URL", "http://192.168.86.201:8000")
        monkeypatch.delenv("LLM_CODER_FAST_URL", raising=False)
        req = _request(task_type="test", prompt="x" * 200000)
        decision = delta(req)
        assert decision.correlation_id == req.correlation_id
