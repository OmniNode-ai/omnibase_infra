# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""HandlerMCP reads its server timeouts from the real handler contract (OMN-22130)."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

import omnibase_infra
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.handlers.handler_mcp import HandlerMCP
from omnibase_infra.handlers.models.mcp import (
    ModelMcpHandlerConfig,
    ModelMcpServerTimeouts,
)
from omnibase_infra.runtime.handler_contract_config_loader import (
    extract_handler_config,
    load_handler_contract_config,
)

pytestmark = pytest.mark.integration

_CONTRACT = (
    Path(omnibase_infra.__file__).parent
    / "contracts"
    / "handlers"
    / "mcp"
    / "handler_contract.yaml"
)


def test_shipped_contract_timeouts_reach_the_handler_config() -> None:
    contract = load_handler_contract_config(_CONTRACT, "mcp")
    config = extract_handler_config(contract, "mcp", require_basic_fields=False)
    timeouts = ModelMcpServerTimeouts.model_validate(config["server_timeouts"])
    assert timeouts == ModelMcpServerTimeouts(
        shutdown_seconds=5.0, cancel_seconds=1.0, startup_seconds=2.0
    )


def test_handler_is_built_from_the_container_alone() -> None:
    container = MagicMock()
    handler = HandlerMCP(container=container)
    assert handler._container is container
    assert handler._config is None
    assert handler._initialized is False
    assert handler._server is None
    assert handler._server_task is None


async def test_shutdown_without_config_returns_immediately() -> None:
    handler = HandlerMCP(container=MagicMock())
    await asyncio.wait_for(handler.shutdown(), timeout=1.0)
    assert handler._config is None
    assert handler._initialized is False


async def test_shutdown_applies_the_configured_waits_not_the_old_defaults() -> None:
    handler = HandlerMCP(container=MagicMock())
    handler._config = ModelMcpHandlerConfig(
        auth_enabled=False,
        server_timeouts=ModelMcpServerTimeouts(
            shutdown_seconds=0.05, cancel_seconds=0.05, startup_seconds=0.05
        ),
    )
    handler._server = cast("Any", SimpleNamespace(should_exit=False))

    async def _never_stops() -> None:
        await asyncio.Event().wait()

    task = asyncio.create_task(_never_stops())
    handler._server_task = task
    started = time.monotonic()
    await handler.shutdown()
    elapsed = time.monotonic() - started
    assert task.cancelled()
    # Graceful 0.05 + forced 0.05: far under the former 5.0 + 1.0 defaults.
    assert elapsed < 1.0
    assert handler._server_task is None


def test_non_mcp_handler_type_checks_and_copies_a_declared_block() -> None:
    block = {"shutdown_seconds": 3.0}
    contract: dict[str, Any] = {
        "name": "x",
        "metadata": {"transport": {"server_timeouts": block}},
    }
    config = extract_handler_config(contract, "http", require_basic_fields=False)
    assert config["server_timeouts"] == block
    bad: dict[str, Any] = {
        "name": "x",
        "metadata": {"transport": {"server_timeouts": {"shutdown_seconds": -1}}},
    }
    with pytest.raises(ProtocolConfigurationError):
        extract_handler_config(bad, "http", require_basic_fields=False)


@pytest.mark.parametrize(
    "bad",
    [
        "five",
        {"shutdown_seconds": 0},
        {"shutdown_seconds": -1.0},
        {"unknown_seconds": 1.0},
        {"shutdown_seconds": True},
        {"cancel_seconds": False},
        {"startup_seconds": float("inf")},
        {"shutdown_seconds": float("nan")},
        {"shutdown_seconds": None},
    ],
)
def test_malformed_server_timeouts_fail_at_contract_load(bad: object) -> None:
    contract: dict[str, Any] = {
        "name": "x",
        "metadata": {"transport": {"server_timeouts": bad}},
    }
    with pytest.raises(ProtocolConfigurationError):
        extract_handler_config(contract, "mcp", require_basic_fields=False)
