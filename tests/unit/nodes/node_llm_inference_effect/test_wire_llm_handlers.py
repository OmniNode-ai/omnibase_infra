# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for LLM handler wiring into runtime container."""

from __future__ import annotations

import pytest

from omnibase_infra.runtime.util_container_wiring import wire_llm_handlers


@pytest.mark.unit
class TestWireLlmHandlers:
    """Tests for wire_llm_handlers."""

    @pytest.mark.asyncio
    async def test_wire_without_publisher(self) -> None:
        """Wiring without publisher registers inference + embedding handlers."""
        from omnibase_core.container import ModelONEXContainer

        container = ModelONEXContainer()
        result = await wire_llm_handlers(container)

        assert result["status"] == "success"
        assert "HandlerLlmOpenaiCompatible" in result["services"]
        assert "HandlerEmbeddingOpenaiCompatible" in result["services"]
        assert "ServiceLlmMetricsPublisher" not in result["services"]

    @pytest.mark.asyncio
    async def test_wire_with_publisher(self) -> None:
        """Wiring with publisher registers metrics-wrapped handler."""
        from omnibase_core.container import ModelONEXContainer

        async def mock_publisher(
            event_type: str, payload: object, correlation_id: str
        ) -> bool:
            return True

        container = ModelONEXContainer()
        result = await wire_llm_handlers(container, publisher=mock_publisher)

        assert result["status"] == "success"
        assert "ServiceLlmMetricsPublisher" in result["services"]
        assert "HandlerLlmOpenaiCompatible (with metrics)" in result["services"]
        assert "HandlerEmbeddingOpenaiCompatible" in result["services"]


@pytest.mark.unit
class TestLlmTopicKeys:
    """Tests for LLM topic key constants and registry resolution."""

    def test_topic_keys_defined(self) -> None:
        """LLM request topic keys are defined."""
        from omnibase_infra.topics import topic_keys

        assert hasattr(topic_keys, "LLM_INFERENCE_REQUEST")
        assert hasattr(topic_keys, "LLM_EMBEDDING_REQUEST")
        assert topic_keys.LLM_INFERENCE_REQUEST == "LLM_INFERENCE_REQUEST"
        assert topic_keys.LLM_EMBEDDING_REQUEST == "LLM_EMBEDDING_REQUEST"

    def test_topic_registry_resolves_llm_requests(self) -> None:
        """ServiceTopicRegistry resolves LLM request topics to concrete strings."""
        from omnibase_infra.topics import topic_keys
        from omnibase_infra.topics.service_topic_registry import ServiceTopicRegistry

        registry = ServiceTopicRegistry.from_defaults()

        inference_topic = registry.resolve(topic_keys.LLM_INFERENCE_REQUEST)
        embedding_topic = registry.resolve(topic_keys.LLM_EMBEDDING_REQUEST)

        assert inference_topic == "onex.cmd.omnibase-infra.llm-inference-request.v1"
        assert embedding_topic == "onex.cmd.omnibase-infra.llm-embedding-request.v1"

    def test_suffix_constants_exist(self) -> None:
        """SUFFIX_ constants are exported from topics package."""
        from omnibase_infra.topics import (
            SUFFIX_LLM_EMBEDDING_REQUEST,
            SUFFIX_LLM_INFERENCE_REQUEST,
        )

        assert (
            SUFFIX_LLM_INFERENCE_REQUEST
            == "onex.cmd.omnibase-infra.llm-inference-request.v1"
        )
        assert (
            SUFFIX_LLM_EMBEDDING_REQUEST
            == "onex.cmd.omnibase-infra.llm-embedding-request.v1"
        )


@pytest.mark.unit
class TestOllamaCleanup:
    """Tests verifying Ollama references removed from contracts."""

    def test_no_ollama_in_inference_contract(self) -> None:
        """Inference contract has no Ollama references."""
        from pathlib import Path

        contract_path = (
            Path(__file__).resolve().parents[4]
            / "src"
            / "omnibase_infra"
            / "nodes"
            / "node_llm_inference_effect"
            / "contract.yaml"
        )
        content = contract_path.read_text()
        assert "ollama" not in content.lower()

    def test_no_ollama_in_embedding_contract(self) -> None:
        """Embedding contract has no Ollama references."""
        from pathlib import Path

        contract_path = (
            Path(__file__).resolve().parents[4]
            / "src"
            / "omnibase_infra"
            / "nodes"
            / "node_llm_embedding_effect"
            / "contract.yaml"
        )
        content = contract_path.read_text()
        assert "ollama" not in content.lower()
