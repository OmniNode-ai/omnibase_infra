# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for bifrost config loader from environment variables."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost.config_loader_bifrost import (
    load_bifrost_config_from_env,
)


@pytest.mark.unit
class TestLoadBifrostConfigFromEnv:
    """Tests for load_bifrost_config_from_env."""

    def test_loads_single_backend(self) -> None:
        env = {"LLM_CODER_URL": "http://192.168.86.201:8000"}
        with patch.dict(os.environ, env, clear=True):
            config = load_bifrost_config_from_env()
        assert "local-coder-30b" in config.backends
        assert (
            config.backends["local-coder-30b"].base_url == "http://192.168.86.201:8000"
        )

    def test_loads_all_local_backends(self) -> None:
        env = {
            "LLM_CODER_URL": "http://192.168.86.201:8000",
            "LLM_CODER_FAST_URL": "http://192.168.86.201:8001",
            "LLM_EMBEDDING_URL": "http://192.168.86.201:8100",
            "LLM_DEEPSEEK_R1_URL": "http://192.168.86.200:8101",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_bifrost_config_from_env()
        assert len(config.backends) == 4
        assert "local-coder-30b" in config.backends
        assert "local-coder-14b" in config.backends
        assert "local-embedding" in config.backends
        assert "local-deepseek-r1" in config.backends

    def test_raises_when_no_backends(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="No LLM backend env vars set"):
                load_bifrost_config_from_env()

    def test_routing_rules_created_for_available_backends(self) -> None:
        env = {
            "LLM_CODER_URL": "http://192.168.86.201:8000",
            "LLM_CODER_FAST_URL": "http://192.168.86.201:8001",
            "LLM_EMBEDDING_URL": "http://192.168.86.201:8100",
            "LLM_DEEPSEEK_R1_URL": "http://192.168.86.200:8101",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_bifrost_config_from_env()
        # Should have routing rules for premium, standard, cheap, embedding, reasoning, eval
        assert len(config.routing_rules) >= 4

    def test_embedding_routing_rule(self) -> None:
        env = {"LLM_EMBEDDING_URL": "http://192.168.86.201:8100"}
        with patch.dict(os.environ, env, clear=True):
            config = load_bifrost_config_from_env()
        embedding_rules = [
            r for r in config.routing_rules if "embedding" in r.match_operation_types
        ]
        assert len(embedding_rules) == 1
        assert "local-embedding" in embedding_rules[0].backend_ids

    def test_default_backends_populated(self) -> None:
        env = {
            "LLM_CODER_URL": "http://192.168.86.201:8000",
            "LLM_CODER_FAST_URL": "http://192.168.86.201:8001",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_bifrost_config_from_env()
        assert len(config.default_backends) > 0
        # Default should prefer local-coder-14b
        assert config.default_backends[0] == "local-coder-14b"

    def test_external_backends(self) -> None:
        env = {
            "LLM_CODER_FAST_URL": "http://192.168.86.201:8001",
            "GLM_BASE_URL": "https://open.bigmodel.cn",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_bifrost_config_from_env()
        assert "glm" in config.backends
        assert config.backends["glm"].base_url == "https://open.bigmodel.cn"

    def test_config_is_frozen(self) -> None:
        env = {"LLM_CODER_URL": "http://192.168.86.201:8000"}
        with patch.dict(os.environ, env, clear=True):
            config = load_bifrost_config_from_env()
        with pytest.raises(Exception):
            config.failover_attempts = 99  # type: ignore[misc]

    def test_gemini_default_carries_v1beta_openai_path(self) -> None:
        """OMN-12664: Gemini default must be /v1beta/openai/chat/completions.

        The bare origin (``https://generativelanguage.googleapis.com``) makes the
        inference handler post to ``/v1/chat/completions`` → 404. The resolved
        default must be the complete registered endpoint carrying
        ``/v1beta/openai`` so the gateway posts it verbatim.
        """
        env = {
            "LLM_CODER_FAST_URL": "http://192.168.86.201:8001",
            "GEMINI_API_KEY": "test-key",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_bifrost_config_from_env()
        assert "gemini" in config.backends
        gemini_url = config.backends["gemini"].base_url
        assert gemini_url == (
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        )
        # Regression guard against the bare-origin default that produced the 404.
        assert "/v1beta/openai" in gemini_url
        assert gemini_url != "https://generativelanguage.googleapis.com"

    def test_gemini_explicit_base_url_override_is_preserved(self) -> None:
        """An explicit GEMINI_BASE_URL override is used verbatim (no regression)."""
        env = {
            "LLM_CODER_FAST_URL": "http://192.168.86.201:8001",
            "GEMINI_API_KEY": "test-key",
            "GEMINI_BASE_URL": (
                "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
            ),
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_bifrost_config_from_env()
        assert config.backends["gemini"].base_url == (
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        )

    def test_gemini_omitted_without_api_key(self) -> None:
        """Gemini backend is only added when the API key is present (unchanged)."""
        env = {"LLM_CODER_FAST_URL": "http://192.168.86.201:8001"}
        with patch.dict(os.environ, env, clear=True):
            config = load_bifrost_config_from_env()
        assert "gemini" not in config.backends

    def test_non_gemini_backends_have_no_path_appended(self) -> None:
        """OMN-12664 regression: local/GLM backends keep their bare base_url."""
        env = {
            "LLM_CODER_URL": "http://192.168.86.201:8000",
            "GLM_BASE_URL": "https://open.bigmodel.cn",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_bifrost_config_from_env()
        assert config.backends["local-coder-30b"].base_url == (
            "http://192.168.86.201:8000"
        )
        assert config.backends["glm"].base_url == "https://open.bigmodel.cn"
