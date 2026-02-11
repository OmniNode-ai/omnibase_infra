# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for the RRH emit effect node handlers.

Tests collection of repo state, runtime targets, and toolchain versions.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from omnibase_infra.models.rrh import (
    ModelRRHRepoState,
    ModelRRHRuntimeTarget,
    ModelRRHToolchainVersions,
)
from omnibase_infra.nodes.node_rrh_emit_effect.handlers.handler_repo_state_collect import (
    HandlerRepoStateCollect,
)
from omnibase_infra.nodes.node_rrh_emit_effect.handlers.handler_runtime_target_collect import (
    HandlerRuntimeTargetCollect,
)
from omnibase_infra.nodes.node_rrh_emit_effect.handlers.handler_toolchain_collect import (
    HandlerToolchainCollect,
)
from omnibase_infra.nodes.node_rrh_emit_effect.node import NodeRRHEmitEffect

pytestmark = [pytest.mark.unit]

CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_rrh_emit_effect"
    / "contract.yaml"
)


# ---------------------------------------------------------------
# Contract validation
# ---------------------------------------------------------------


class TestContractValidation:
    @pytest.fixture(scope="class")
    def contract_data(self) -> dict:
        with CONTRACT_PATH.open() as f:
            data: dict = yaml.safe_load(f)
        return data

    def test_node_type_is_effect(self, contract_data: dict) -> None:
        assert contract_data.get("node_type") == "EFFECT_GENERIC"

    def test_has_three_handlers(self, contract_data: dict) -> None:
        handlers = contract_data.get("handler_routing", {}).get("handlers", [])
        assert len(handlers) == 3


# ---------------------------------------------------------------
# Node declarative check
# ---------------------------------------------------------------


class TestNodeDeclarative:
    def test_no_custom_methods(self) -> None:
        custom = [
            m
            for m in dir(NodeRRHEmitEffect)
            if not m.startswith("_") and m not in dir(NodeRRHEmitEffect.__bases__[0])
        ]
        assert custom == [], f"Node has custom methods: {custom}"


# ---------------------------------------------------------------
# HandlerRepoStateCollect
# ---------------------------------------------------------------


class TestHandlerRepoStateCollect:
    @pytest.fixture
    def handler(self) -> HandlerRepoStateCollect:
        return HandlerRepoStateCollect()

    @pytest.mark.anyio
    async def test_collects_from_real_repo(
        self, handler: HandlerRepoStateCollect
    ) -> None:
        """Integration-style test: collect state from the actual repo."""
        repo_path = str(Path(__file__).resolve().parents[3])
        result = await handler.handle(repo_path)
        assert isinstance(result, ModelRRHRepoState)
        assert result.branch  # Should have a branch
        assert result.head_sha  # Should have a SHA
        assert result.repo_root  # Should have a root path

    def test_handler_type(self, handler: HandlerRepoStateCollect) -> None:
        from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory

        assert handler.handler_type == EnumHandlerType.INFRA_HANDLER
        assert handler.handler_category == EnumHandlerTypeCategory.EFFECT

    @pytest.mark.anyio
    async def test_handles_invalid_path(self, handler: HandlerRepoStateCollect) -> None:
        result = await handler.handle("/nonexistent/path")
        assert isinstance(result, ModelRRHRepoState)
        # Should return empty values, not raise.
        assert result.branch == ""
        assert result.head_sha == ""


# ---------------------------------------------------------------
# HandlerRuntimeTargetCollect
# ---------------------------------------------------------------


class TestHandlerRuntimeTargetCollect:
    @pytest.fixture
    def handler(self) -> HandlerRuntimeTargetCollect:
        return HandlerRuntimeTargetCollect()

    @pytest.mark.anyio
    async def test_uses_overrides(self, handler: HandlerRuntimeTargetCollect) -> None:
        result = await handler.handle(
            environment="staging",
            kafka_broker="kafka:9092",
            kubernetes_context="prod",
        )
        assert isinstance(result, ModelRRHRuntimeTarget)
        assert result.environment == "staging"
        assert result.kafka_broker == "kafka:9092"
        assert result.kubernetes_context == "prod"

    @pytest.mark.anyio
    async def test_falls_back_to_env(
        self, handler: HandlerRuntimeTargetCollect
    ) -> None:
        with patch.dict("os.environ", {"ENVIRONMENT": "ci"}, clear=False):
            result = await handler.handle()
        assert result.environment == "ci"


# ---------------------------------------------------------------
# HandlerToolchainCollect
# ---------------------------------------------------------------


class TestHandlerToolchainCollect:
    @pytest.fixture
    def handler(self) -> HandlerToolchainCollect:
        return HandlerToolchainCollect()

    @pytest.mark.anyio
    async def test_collects_versions(self, handler: HandlerToolchainCollect) -> None:
        result = await handler.handle()
        assert isinstance(result, ModelRRHToolchainVersions)
        # At minimum, ruff and pytest should be installed in this project.
        assert result.ruff  # ruff is a dev dependency
        assert result.pytest  # pytest is a dev dependency

    @pytest.mark.anyio
    async def test_handler_type(self, handler: HandlerToolchainCollect) -> None:
        from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory

        assert handler.handler_type == EnumHandlerType.INFRA_HANDLER
        assert handler.handler_category == EnumHandlerTypeCategory.EFFECT
