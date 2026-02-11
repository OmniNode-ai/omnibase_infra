# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for the RRH emit effect node handlers.

Tests collection of repo state, runtime targets, and toolchain versions.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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

    @pytest.mark.anyio
    async def test_handles_repo_with_no_remote(
        self, handler: HandlerRepoStateCollect
    ) -> None:
        """When 'git remote get-url origin' fails, remote_url should be empty."""

        async def _fake_subprocess(*args: str, **_kwargs: object) -> MagicMock:
            """Simulate git commands; fail only for 'remote get-url origin'."""
            proc = MagicMock(spec=asyncio.subprocess.Process)
            # Determine which git sub-command was requested.
            # args layout: ("git", "-C", repo_path, <sub-command...>)
            sub_args = args[3:]  # everything after repo_path
            if sub_args == ("remote", "get-url", "origin"):
                proc.returncode = 2
                proc.communicate = AsyncMock(
                    return_value=(b"", b"fatal: No such remote 'origin'\n")
                )
            elif sub_args == ("rev-parse", "--abbrev-ref", "HEAD"):
                proc.returncode = 0
                proc.communicate = AsyncMock(return_value=(b"main\n", b""))
            elif sub_args == ("rev-parse", "HEAD"):
                proc.returncode = 0
                proc.communicate = AsyncMock(return_value=(b"abc123\n", b""))
            elif sub_args == ("status", "--porcelain"):
                proc.returncode = 0
                proc.communicate = AsyncMock(return_value=(b"", b""))
            elif sub_args == ("rev-parse", "--show-toplevel"):
                proc.returncode = 0
                proc.communicate = AsyncMock(return_value=(b"/fake/repo\n", b""))
            else:
                proc.returncode = 0
                proc.communicate = AsyncMock(return_value=(b"", b""))
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=_fake_subprocess):
            result = await handler.handle("/fake/repo")

        assert isinstance(result, ModelRRHRepoState)
        assert result.branch == "main"
        assert result.head_sha == "abc123"
        assert result.remote_url == ""
        assert not result.is_dirty


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

    def test_handler_type(self, handler: HandlerToolchainCollect) -> None:
        from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory

        assert handler.handler_type == EnumHandlerType.INFRA_HANDLER
        assert handler.handler_category == EnumHandlerTypeCategory.EFFECT
