# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ServiceRegistration._wire_registration_storage (OMN-5345).

Verifies that HandlerRegistrationStoragePostgres is instantiated and
registered during plugin initialization so that:
1. The node_registrations table is auto-created on first connection.
2. The handler is registered in RegistryInfraRegistrationStorage.
3. The handler is registered in the container service registry.

Tests use mocks to avoid real database connections.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

_PLUGIN_MOD = "omnibase_infra.nodes.node_registration_orchestrator.plugin"


def _make_config() -> MagicMock:
    """Build a minimal ModelDomainPluginConfig-like mock."""
    config = MagicMock()
    config.correlation_id = uuid4()
    config.container = MagicMock()
    config.container.service_registry = AsyncMock()
    config.container.service_registry.register_instance = AsyncMock()
    return config


def _make_plugin_with_pool(pool: MagicMock) -> MagicMock:
    """Construct a ServiceRegistration with an injected pool."""
    from omnibase_infra.nodes.node_registration_orchestrator.plugin import (
        ServiceRegistration,
    )

    plugin = ServiceRegistration()
    plugin._pool = pool  # type: ignore[attr-defined]
    return plugin  # type: ignore[return-value]


class TestWireRegistrationStorage:
    """Tests for _wire_registration_storage wiring method."""

    @pytest.mark.unit
    async def test_no_pool_returns_early(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No pool -> returns early with WARNING, no handler created."""
        from omnibase_infra.nodes.node_registration_orchestrator.plugin import (
            ServiceRegistration,
        )

        plugin = ServiceRegistration()
        plugin._pool = None  # type: ignore[attr-defined]
        config = _make_config()

        with caplog.at_level(logging.WARNING, logger=_PLUGIN_MOD):
            await plugin._wire_registration_storage(config)  # type: ignore[attr-defined]

        assert plugin._registration_storage is None  # type: ignore[attr-defined]
        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("pool" in r.message.lower() for r in warning_msgs)

    @pytest.mark.unit
    async def test_handler_created_with_auto_create_schema(self) -> None:
        """Handler is created with auto_create_schema=True and _ensure_pool called."""
        pool = MagicMock()
        plugin = _make_plugin_with_pool(pool)
        config = _make_config()

        mock_handler = MagicMock()
        mock_handler.handler_type = "postgresql"
        mock_handler._ensure_pool = AsyncMock()
        mock_handler.shutdown = AsyncMock()

        with (
            patch(
                "omnibase_infra.handlers.registration_storage"
                ".handler_registration_storage_postgres"
                ".HandlerRegistrationStoragePostgres",
                return_value=mock_handler,
            ) as mock_cls,
            patch(
                "omnibase_infra.nodes.node_registration_storage_effect"
                ".registry.registry_infra_registration_storage"
                ".RegistryInfraRegistrationStorage.register",
            ),
            patch(
                "omnibase_infra.nodes.node_registration_storage_effect"
                ".registry.registry_infra_registration_storage"
                ".RegistryInfraRegistrationStorage.register_handler",
            ),
            patch.dict(
                "os.environ", {"OMNIBASE_INFRA_DB_URL": "postgresql://test:5432/db"}
            ),
        ):
            await plugin._wire_registration_storage(config)  # type: ignore[attr-defined]

        # Handler should have been created with auto_create_schema=True
        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args
        assert call_kwargs[1]["auto_create_schema"] is True

        # _ensure_pool should have been called to eagerly create table
        mock_handler._ensure_pool.assert_called_once()

        # Handler should be stored on the plugin
        assert plugin._registration_storage is mock_handler  # type: ignore[attr-defined]

    @pytest.mark.unit
    async def test_failure_does_not_propagate(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """If handler creation fails, the error is caught and logged (best-effort)."""
        pool = MagicMock()
        plugin = _make_plugin_with_pool(pool)
        config = _make_config()

        with (
            patch(
                "omnibase_infra.handlers.registration_storage"
                ".handler_registration_storage_postgres"
                ".HandlerRegistrationStoragePostgres",
                side_effect=RuntimeError("connection refused"),
            ),
            caplog.at_level(logging.WARNING, logger=_PLUGIN_MOD),
        ):
            # Must NOT raise -- best-effort wiring
            await plugin._wire_registration_storage(config)  # type: ignore[attr-defined]

        assert plugin._registration_storage is None  # type: ignore[attr-defined]
        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("registration" in r.message.lower() for r in warning_msgs), (
            f"Expected WARNING about registration storage. Got: {[r.message for r in warning_msgs]}"
        )

    @pytest.mark.unit
    async def test_shutdown_cleans_up_handler(self) -> None:
        """shutdown() calls handler.shutdown() and clears the reference."""
        from omnibase_infra.nodes.node_registration_orchestrator.plugin import (
            ServiceRegistration,
        )

        plugin = ServiceRegistration()
        mock_handler = AsyncMock()
        mock_handler.shutdown = AsyncMock()
        plugin._registration_storage = mock_handler  # type: ignore[attr-defined]

        config = _make_config()
        # Need pool to be None to avoid pool.close() issues
        plugin._pool = None  # type: ignore[attr-defined]

        await plugin._do_shutdown(config)  # type: ignore[attr-defined]

        mock_handler.shutdown.assert_called_once()
        assert plugin._registration_storage is None  # type: ignore[attr-defined]
