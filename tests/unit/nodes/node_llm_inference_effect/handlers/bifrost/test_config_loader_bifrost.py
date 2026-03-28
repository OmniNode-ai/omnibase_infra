# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for bifrost config loader from environment variables."""

from __future__ import annotations

import pytest

from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost.config_loader_bifrost import (
    BACKEND_CODER,
    BACKEND_CODER_FAST,
    BACKEND_DEEPSEEK_R1,
    BACKEND_EMBEDDING,
    BACKEND_SMALL,
    load_bifrost_config_from_env,
)


@pytest.mark.unit
class TestLoadBifrostConfigFromEnv:
    """Tests for load_bifrost_config_from_env."""

    def test_all_endpoints_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Config includes all backends when all env vars are set."""
        monkeypatch.setenv("LLM_CODER_URL", "http://192.168.86.201:8000")
        monkeypatch.setenv("LLM_CODER_FAST_URL", "http://192.168.86.201:8001")
        monkeypatch.setenv("LLM_EMBEDDING_URL", "http://192.168.86.200:8100")
        monkeypatch.setenv("LLM_DEEPSEEK_R1_URL", "http://192.168.86.200:8101")
        monkeypatch.setenv("LLM_SMALL_URL", "http://192.168.86.105:8000")

        config = load_bifrost_config_from_env()

        assert len(config.backends) == 5
        assert BACKEND_CODER in config.backends
        assert BACKEND_CODER_FAST in config.backends
        assert BACKEND_EMBEDDING in config.backends
        assert BACKEND_DEEPSEEK_R1 in config.backends
        assert BACKEND_SMALL in config.backends

    def test_single_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Config works with only one endpoint."""
        monkeypatch.setenv("LLM_CODER_URL", "http://localhost:8000")
        monkeypatch.delenv("LLM_CODER_FAST_URL", raising=False)
        monkeypatch.delenv("LLM_EMBEDDING_URL", raising=False)
        monkeypatch.delenv("LLM_DEEPSEEK_R1_URL", raising=False)
        monkeypatch.delenv("LLM_SMALL_URL", raising=False)

        config = load_bifrost_config_from_env()

        assert len(config.backends) == 1
        assert BACKEND_CODER in config.backends
        assert config.backends[BACKEND_CODER].base_url == "http://localhost:8000"

    def test_no_endpoints_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ValueError raised when no endpoints are configured."""
        monkeypatch.delenv("LLM_CODER_URL", raising=False)
        monkeypatch.delenv("LLM_CODER_FAST_URL", raising=False)
        monkeypatch.delenv("LLM_EMBEDDING_URL", raising=False)
        monkeypatch.delenv("LLM_DEEPSEEK_R1_URL", raising=False)
        monkeypatch.delenv("LLM_SMALL_URL", raising=False)

        with pytest.raises(ValueError, match="No LLM endpoint URLs configured"):
            load_bifrost_config_from_env()

    def test_routing_rules_embedding(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Embedding routing rule routes to embedding backend."""
        monkeypatch.setenv("LLM_EMBEDDING_URL", "http://localhost:8100")
        monkeypatch.delenv("LLM_CODER_URL", raising=False)
        monkeypatch.delenv("LLM_CODER_FAST_URL", raising=False)
        monkeypatch.delenv("LLM_DEEPSEEK_R1_URL", raising=False)
        monkeypatch.delenv("LLM_SMALL_URL", raising=False)

        config = load_bifrost_config_from_env()

        embedding_rules = [
            r for r in config.routing_rules if "embedding" in r.match_operation_types
        ]
        assert len(embedding_rules) == 1
        assert BACKEND_EMBEDDING in embedding_rules[0].backend_ids

    def test_routing_rules_reasoning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reasoning routing rule routes to DeepSeek backend."""
        monkeypatch.setenv("LLM_DEEPSEEK_R1_URL", "http://localhost:8101")
        monkeypatch.delenv("LLM_CODER_URL", raising=False)
        monkeypatch.delenv("LLM_CODER_FAST_URL", raising=False)
        monkeypatch.delenv("LLM_EMBEDDING_URL", raising=False)
        monkeypatch.delenv("LLM_SMALL_URL", raising=False)

        config = load_bifrost_config_from_env()

        reasoning_rules = [
            r for r in config.routing_rules if "reasoning" in r.match_operation_types
        ]
        assert len(reasoning_rules) == 1
        assert BACKEND_DEEPSEEK_R1 in reasoning_rules[0].backend_ids

    def test_premium_code_failover(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Premium code rule includes fast coder as failover."""
        monkeypatch.setenv("LLM_CODER_URL", "http://localhost:8000")
        monkeypatch.setenv("LLM_CODER_FAST_URL", "http://localhost:8001")
        monkeypatch.delenv("LLM_EMBEDDING_URL", raising=False)
        monkeypatch.delenv("LLM_DEEPSEEK_R1_URL", raising=False)
        monkeypatch.delenv("LLM_SMALL_URL", raising=False)

        config = load_bifrost_config_from_env()

        premium_rules = [
            r for r in config.routing_rules if "premium" in r.match_cost_tiers
        ]
        assert len(premium_rules) == 1
        assert premium_rules[0].backend_ids == (BACKEND_CODER, BACKEND_CODER_FAST)

    def test_default_backends_matches_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default backends includes all configured backends."""
        monkeypatch.setenv("LLM_CODER_URL", "http://localhost:8000")
        monkeypatch.setenv("LLM_EMBEDDING_URL", "http://localhost:8100")
        monkeypatch.delenv("LLM_CODER_FAST_URL", raising=False)
        monkeypatch.delenv("LLM_DEEPSEEK_R1_URL", raising=False)
        monkeypatch.delenv("LLM_SMALL_URL", raising=False)

        config = load_bifrost_config_from_env()

        assert set(config.default_backends) == {BACKEND_CODER, BACKEND_EMBEDDING}

    def test_config_is_frozen(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returned config is immutable (frozen model)."""
        monkeypatch.setenv("LLM_CODER_URL", "http://localhost:8000")
        monkeypatch.delenv("LLM_CODER_FAST_URL", raising=False)
        monkeypatch.delenv("LLM_EMBEDDING_URL", raising=False)
        monkeypatch.delenv("LLM_DEEPSEEK_R1_URL", raising=False)
        monkeypatch.delenv("LLM_SMALL_URL", raising=False)

        config = load_bifrost_config_from_env()

        with pytest.raises(Exception):
            config.failover_attempts = 99  # type: ignore[misc]
