# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerDelegationRouting (delta function).

Tests cover:
    - Routing for each task type (test, research, document)
    - Fast-path routing when prompt tokens <= 24K
    - Fallback when local-deepseek backend has no endpoint
    - Error on missing all endpoints
    - System prompt assignment per task type
    - Escalation: no local endpoints configured → claude tier

Endpoint URLs are resolved from a deployed bifrost contract, not env vars.
Tests write a temporary bifrost YAML and point the handler at it via
BIFROST_CONTRACT_PATH.

Related:
    - OMN-7040: Node-based delegation pipeline
    - OMN-8029: routing_tiers.yaml config-driven routing
    - OMN-10657: Contract-driven endpoint resolution
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

import omnibase_infra.nodes.node_delegation_routing_reducer.handlers.handler_delegation_routing as _handler_mod
from omnibase_infra.errors import ProtocolConfigurationError
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


def _write_bifrost(tmp_path: Path, backends: dict[str, str]) -> str:
    """Write a bifrost contract YAML with given backend_id -> endpoint_url mappings."""
    entries = []
    for bid, url in backends.items():
        entries.append(
            f"  - backend_id: {bid}\n"
            f'    endpoint_url: "{url}"\n'
            f"    model_name: "
            "\n"
            f"    tier: local\n"
            f"    timeout_ms: 30000\n"
            f"    capabilities: []"
        )
    backends_block = "\n".join(entries) if entries else "  []"
    content = (
        "config_version: '1.1.0'\n"
        "schema_version: bifrost_delegation.v1\n"
        "backends:\n" + backends_block + "\n"
    )
    path = tmp_path / "bifrost.yaml"
    path.write_text(content)
    return str(path)


@pytest.fixture(autouse=True)
def reset_config_singleton():  # type: ignore[no-untyped-def]
    """Reset the module-level config singleton before each test."""
    _handler_mod._config = None
    _handler_mod._load_bifrost_endpoints.cache_clear()
    yield
    _handler_mod._config = None
    _handler_mod._load_bifrost_endpoints.cache_clear()


class TestRoutingByTaskType:
    """Verify correct model selection per task type from routing_tiers.yaml."""

    def test_test_routes_to_coder_when_no_fast_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without fast-path endpoint, test tasks go to qwen3-coder-30b."""
        bifrost = _write_bifrost(
            tmp_path,
            {
                "local-qwen-coder-30b": "http://192.168.86.201:8000"
            },  # onex-allow-internal-ip
        )
        monkeypatch.setenv("BIFROST_CONTRACT_PATH", bifrost)
        req = _request(task_type="test", prompt="x" * 200000)
        decision = delta(req)
        assert decision.selected_model == "qwen3-coder-30b"
        assert (
            decision.endpoint_url == "http://192.168.86.201:8000"
        )  # onex-allow-internal-ip
        assert decision.cost_tier == "low"
        assert decision.max_context_tokens == 65536

    def test_research_routes_to_coder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bifrost = _write_bifrost(
            tmp_path,
            {
                "local-qwen-coder-30b": "http://192.168.86.201:8000"
            },  # onex-allow-internal-ip
        )
        monkeypatch.setenv("BIFROST_CONTRACT_PATH", bifrost)
        req = _request(task_type="research", prompt="x" * 200000)
        decision = delta(req)
        assert decision.selected_model == "qwen3-coder-30b"
        assert (
            decision.endpoint_url == "http://192.168.86.201:8000"
        )  # onex-allow-internal-ip

    def test_document_routes_to_coder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Document tasks go to qwen3-coder-30b via the local coder backend."""
        bifrost = _write_bifrost(
            tmp_path,
            {
                "local-qwen-coder-30b": "http://192.168.86.201:8000"
            },  # onex-allow-internal-ip
        )
        monkeypatch.setenv("BIFROST_CONTRACT_PATH", bifrost)
        req = _request(task_type="document")
        decision = delta(req)
        assert decision.selected_model == "qwen3-coder-30b"
        assert (
            decision.endpoint_url == "http://192.168.86.201:8000"
        )  # onex-allow-internal-ip


class TestFastPathRouting:
    """Verify token-count based fast-path optimization."""

    def test_short_prompt_uses_fast_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bifrost = _write_bifrost(
            tmp_path,
            {
                "local-qwen-coder-30b": "http://192.168.86.201:8000",  # onex-allow-internal-ip
                "local-deepseek-r1-14b": "http://192.168.86.201:8001",  # onex-allow-internal-ip
            },
        )
        monkeypatch.setenv("BIFROST_CONTRACT_PATH", bifrost)
        req = _request(task_type="test", prompt="Write tests for auth.py")
        decision = delta(req)
        assert decision.selected_model == "qwen3-coder-30b"
        assert (
            decision.endpoint_url == "http://192.168.86.201:8000"
        )  # onex-allow-internal-ip
        assert decision.max_context_tokens == 65536

    def test_long_prompt_skips_fast_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bifrost = _write_bifrost(
            tmp_path,
            {
                "local-qwen-coder-30b": "http://192.168.86.201:8000",  # onex-allow-internal-ip
                "local-deepseek-r1-14b": "http://192.168.86.201:8001",  # onex-allow-internal-ip
            },
        )
        monkeypatch.setenv("BIFROST_CONTRACT_PATH", bifrost)
        long_prompt = "x" * 200000
        req = _request(task_type="test", prompt=long_prompt)
        decision = delta(req)
        assert decision.selected_model == "qwen3-coder-30b"

    def test_fast_path_not_available_falls_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bifrost = _write_bifrost(
            tmp_path,
            {
                "local-qwen-coder-30b": "http://192.168.86.201:8000"
            },  # onex-allow-internal-ip
        )
        monkeypatch.setenv("BIFROST_CONTRACT_PATH", bifrost)
        req = _request(task_type="test", prompt="short prompt")
        decision = delta(req)
        assert decision.selected_model == "qwen3-coder-30b"

    def test_document_uses_coder_fast_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Document tasks use qwen3-coder-30b (fast path in local tier)."""
        bifrost = _write_bifrost(
            tmp_path,
            {
                "local-qwen-coder-30b": "http://192.168.86.201:8000"
            },  # onex-allow-internal-ip
        )
        monkeypatch.setenv("BIFROST_CONTRACT_PATH", bifrost)
        req = _request(task_type="document", prompt="short prompt")
        decision = delta(req)
        assert decision.selected_model == "qwen3-coder-30b"


class TestMissingEndpoint:
    """Verify error when no tier has a configured endpoint for the task."""

    def test_missing_all_endpoints_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When all backend endpoint_urls are empty, should raise ProtocolConfigurationError."""
        bifrost = _write_bifrost(tmp_path, {})
        monkeypatch.setenv("BIFROST_CONTRACT_PATH", bifrost)
        req = _request(task_type="test")
        with pytest.raises(ProtocolConfigurationError, match="No tier"):
            delta(req)

    def test_missing_local_escalates_to_claude(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If local tier is unavailable for document, claude tier takes it."""
        bifrost = _write_bifrost(
            tmp_path, {"cloud-sonnet": "https://api.anthropic.com"}
        )
        monkeypatch.setenv("BIFROST_CONTRACT_PATH", bifrost)
        req = _request(task_type="document")
        decision = delta(req)
        assert decision.selected_model == "claude-sonnet-4-6"
        assert decision.cost_tier == "high"


class TestEscalationToClaudeTier:
    """Verify escalation from local -> cheap_cloud -> claude."""

    def test_escalates_to_claude_when_local_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bifrost = _write_bifrost(
            tmp_path, {"cloud-sonnet": "https://api.anthropic.com"}
        )
        monkeypatch.setenv("BIFROST_CONTRACT_PATH", bifrost)
        req = _request(task_type="test")
        decision = delta(req)
        assert decision.selected_model == "claude-sonnet-4-6"
        assert decision.cost_tier == "high"


class TestSystemPrompts:
    """Verify system prompt assignment."""

    def test_test_system_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bifrost = _write_bifrost(
            tmp_path,
            {
                "local-qwen-coder-30b": "http://192.168.86.201:8000"
            },  # onex-allow-internal-ip
        )
        monkeypatch.setenv("BIFROST_CONTRACT_PATH", bifrost)
        req = _request(task_type="test", prompt="x" * 200000)
        decision = delta(req)
        assert "test generation" in decision.system_prompt.lower()

    def test_document_system_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bifrost = _write_bifrost(
            tmp_path,
            {
                "local-qwen-coder-30b": "http://192.168.86.201:8000"
            },  # onex-allow-internal-ip
        )
        monkeypatch.setenv("BIFROST_CONTRACT_PATH", bifrost)
        req = _request(task_type="document")
        decision = delta(req)
        assert "documentation" in decision.system_prompt.lower()

    def test_research_system_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bifrost = _write_bifrost(
            tmp_path,
            {
                "local-qwen-coder-30b": "http://192.168.86.201:8000"
            },  # onex-allow-internal-ip
        )
        monkeypatch.setenv("BIFROST_CONTRACT_PATH", bifrost)
        req = _request(task_type="research", prompt="x" * 200000)
        decision = delta(req)
        assert "research" in decision.system_prompt.lower()


class TestCorrelationIdPreserved:
    """Verify correlation_id flows through."""

    def test_correlation_id_matches_request(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bifrost = _write_bifrost(
            tmp_path,
            {
                "local-qwen-coder-30b": "http://192.168.86.201:8000"
            },  # onex-allow-internal-ip
        )
        monkeypatch.setenv("BIFROST_CONTRACT_PATH", bifrost)
        req = _request(task_type="test", prompt="x" * 200000)
        decision = delta(req)
        assert decision.correlation_id == req.correlation_id
