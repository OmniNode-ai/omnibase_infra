# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""HandlerMCP takes its dependencies by injection (OMN-22130).

The handler is built from the container alone, and its server timeouts come
from the handler contract through ``initialize(config)``, never from literals
held in the module. The two-profile case runs the full server start and stop
path with a fake lifecycle and a fake uvicorn server, once with the in-memory
binding (Kafka off) and once with the Redpanda overlay binding (Kafka on).
"""

from __future__ import annotations

import asyncio
import inspect
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import omnibase_infra.handlers.handler_mcp as handler_mcp_module
from omnibase_infra.handlers.handler_mcp import HandlerMCP
from omnibase_infra.handlers.models.mcp import (
    ModelMcpHandlerConfig,
    ModelMcpServerTimeouts,
)
from omnibase_infra.runtime.handler_contract_config_loader import (
    extract_handler_config,
    load_handler_contract_config,
)

_CONTRACT_PATH = (
    Path(handler_mcp_module.__file__).resolve().parent.parent
    / "contracts"
    / "handlers"
    / "mcp"
    / "handler_contract.yaml"
)


def test_constructor_takes_exactly_a_required_container() -> None:
    params = inspect.signature(HandlerMCP.__init__).parameters
    assert list(params) == ["self", "container"]
    assert params["container"].default is inspect.Parameter.empty


def test_module_holds_no_server_timeout_literals() -> None:
    for name in (
        "_DEFAULT_SHUTDOWN_TIMEOUT",
        "_DEFAULT_CANCEL_TIMEOUT",
        "_DEFAULT_STARTUP_TIMEOUT",
    ):
        assert not hasattr(handler_mcp_module, name)
    for attr in ("shutdown_timeout", "cancel_timeout", "startup_timeout"):
        assert not hasattr(HandlerMCP, attr)


def test_contract_declares_the_server_timeouts() -> None:
    contract = load_handler_contract_config(_CONTRACT_PATH, "effect.mcp.handler")
    config = extract_handler_config(contract, "mcp", require_basic_fields=False)
    assert config["server_timeouts"] == {
        "shutdown_seconds": 5.0,
        "cancel_seconds": 1.0,
        "startup_seconds": 2.0,
    }
    parsed = ModelMcpHandlerConfig(**{**config, "auth_enabled": False})
    assert parsed.server_timeouts == ModelMcpServerTimeouts(
        shutdown_seconds=5.0, cancel_seconds=1.0, startup_seconds=2.0
    )


class _FakeLifecycle:
    """Stands in for MCPServerLifecycle; records what the handler gave it."""

    built: list[_FakeLifecycle] = []

    def __init__(self, container: object, config: object, bus: object) -> None:
        self.container = container
        self.config = config
        self.bus = bus
        self.registry = MagicMock(name="registry")
        self.executor = MagicMock(name="executor")
        self.stopped = False
        _FakeLifecycle.built.append(self)

    async def start(self) -> None:
        return None

    async def shutdown(self) -> None:
        self.stopped = True


class _StuckServer:
    """A uvicorn server double whose serve() never returns on its own."""

    def __init__(self, config: object) -> None:
        self.config = config
        self.should_exit = False

    async def serve(self) -> None:
        await asyncio.Event().wait()


_PROFILES = {
    "inmemory": {"kafka_enabled": False},
    "redpanda-overlay": {"kafka_enabled": True},
}


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", sorted(_PROFILES))
async def test_server_lifecycle_uses_injected_container_and_contract_timeouts(
    profile: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _FakeLifecycle.built.clear()
    monkeypatch.setattr(handler_mcp_module, "MCPServerLifecycle", _FakeLifecycle)
    monkeypatch.setattr(handler_mcp_module.uvicorn, "Server", _StuckServer)
    startup_waits: list[float] = []

    async def _ready(
        self: HandlerMCP, host: str, port: int, timeout: float, **_: object
    ) -> None:
        startup_waits.append(timeout)

    monkeypatch.setattr(HandlerMCP, "_wait_for_server_ready", _ready)

    container = MagicMock(name="container")
    handler = HandlerMCP(container=container)
    await handler.initialize(
        {
            **_PROFILES[profile],
            "auth_enabled": False,
            "host": "127.0.0.1",
            "port": 18090,
            "server_timeouts": {
                "shutdown_seconds": 0.05,
                "cancel_seconds": 0.05,
                "startup_seconds": 0.25,
            },
        }
    )

    (lifecycle,) = _FakeLifecycle.built
    assert lifecycle.container is container
    assert lifecycle.config.kafka_enabled is _PROFILES[profile]["kafka_enabled"]
    assert handler._mcp_registry is lifecycle.registry
    assert handler._mcp_executor is lifecycle.executor
    assert startup_waits == [0.25]

    started = time.monotonic()
    await handler.shutdown()
    assert time.monotonic() - started < 1.0
    assert lifecycle.stopped
    assert handler._mcp_registry is None
