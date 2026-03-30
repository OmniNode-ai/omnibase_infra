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
            "LLM_EMBEDDING_URL": "http://192.168.86.200:8100",
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
            "LLM_EMBEDDING_URL": "http://192.168.86.200:8100",
            "LLM_DEEPSEEK_R1_URL": "http://192.168.86.200:8101",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_bifrost_config_from_env()
        # Should have routing rules for premium, standard, cheap, embedding, reasoning, eval
        assert len(config.routing_rules) >= 4

    def test_embedding_routing_rule(self) -> None:
        env = {"LLM_EMBEDDING_URL": "http://192.168.86.200:8100"}
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
