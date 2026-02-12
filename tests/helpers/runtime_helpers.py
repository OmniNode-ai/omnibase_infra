# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared runtime helpers for RuntimeHostProcess testing.

Provides ``make_runtime_config`` and ``seed_mock_handlers`` utilities used by
both unit and integration tests that exercise RuntimeHostProcess lifecycle
without needing the full handler registry wiring.

These helpers were extracted from ``tests/conftest.py`` so that test modules
can import them directly without relying on conftest auto-discovery.

Usage::

    from tests.helpers.runtime_helpers import make_runtime_config, seed_mock_handlers

    config = make_runtime_config(service_name="my-test")
    process = RuntimeHostProcess(event_bus=bus, config=config)
    seed_mock_handlers(process)
    await process.start()
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from omnibase_infra.protocols.protocol_container_aware import ProtocolContainerAware


def make_runtime_config(**overrides: object) -> dict[str, object]:
    """Create a runtime config dict with default required fields and optional overrides.

    RuntimeHostProcess requires 'service_name' and 'node_name' in config for proper
    node identity construction (OMN-1602). This helper provides default values for
    these required fields while allowing specific test cases to override any config.

    Args:
        **overrides: Config keys to override or add to the default config.

    Returns:
        A config dict suitable for RuntimeHostProcess initialization.

    Example:
        >>> config = make_runtime_config()
        >>> config["service_name"]
        'test-service'

        >>> config = make_runtime_config(input_topic="custom.input")
        >>> config["input_topic"]
        'custom.input'
    """
    config: dict[str, object] = {
        "service_name": "test-service",
        "node_name": "test-node",
        "env": "test",
        "version": "v1",
    }
    config.update(overrides)
    return config


def seed_mock_handlers(
    process: object,
    *,
    handlers: dict[str, MagicMock] | None = None,
    initialized: bool = True,
) -> None:
    """Seed mock handlers on a RuntimeHostProcess to bypass fail-fast validation.

    The RuntimeHostProcess.start() method validates that handlers are registered.
    This helper sets up minimal mock handler(s) to satisfy that check, allowing
    tests to focus on other runtime functionality.

    The default mock handler includes all async lifecycle methods:
    - execute: AsyncMock for handling envelopes
    - initialize: AsyncMock for handler initialization
    - shutdown: AsyncMock for safe cleanup with await process.stop()
    - health_check: AsyncMock returning {"healthy": True}

    Args:
        process: The RuntimeHostProcess instance to seed handlers on.
            Typed as object to avoid import dependency, but must have _handlers attr.
        handlers: Optional dict of handler name to mock handler. If not provided,
            a default mock handler named "mock" is created with all lifecycle methods.
        initialized: If True (default), marks the mock handler as initialized
            so health_check returns healthy status.

    Example:
        >>> from tests.helpers.runtime_helpers import seed_mock_handlers
        >>> process = RuntimeHostProcess(event_bus=mock_event_bus)
        >>> seed_mock_handlers(process)
        >>> await process.start()  # Will not raise fail-fast error

        >>> # With custom handlers
        >>> seed_mock_handlers(process, handlers={"db": db_mock, "http": http_mock})

    Note:
        This function directly sets the private ``_handlers`` attribute.  This is
        intentional for testing purposes to bypass the normal handler registration
        flow.  RuntimeHostProcess does not expose a public API for handler seeding,
        so private access is the only option.  Do not use in production code.

    Warning:
        When providing custom handlers, ensure they have the required async methods
        (shutdown, health_check) for safe cleanup during process.stop().
    """
    if handlers is not None:
        process._handlers = handlers  # type: ignore[attr-defined]
        return

    # Create default mock handler with required async methods.
    # spec=ProtocolContainerAware constrains the mock to the handler protocol,
    # preventing tests from accidentally relying on auto-created attributes.
    # Async methods are explicitly overridden because spec alone produces
    # synchronous MagicMock stubs for protocol methods.
    mock_handler = MagicMock(spec=ProtocolContainerAware)
    mock_handler.execute = AsyncMock(return_value={"success": True, "result": "mock"})
    mock_handler.initialize = AsyncMock()
    mock_handler.shutdown = AsyncMock()
    mock_handler.health_check = AsyncMock(return_value={"healthy": True})

    # Mark as initialized for health check compatibility
    if initialized:
        mock_handler.initialized = True

    # Private attribute access: RuntimeHostProcess does not expose a public API
    # for handler seeding.  Setting _handlers directly is the only way to inject
    # mock handlers for testing.  See RuntimeHostProcess.start() for the
    # fail-fast validation that requires this.
    process._handlers = {"mock": mock_handler}  # type: ignore[attr-defined]
