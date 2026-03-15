# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for HandlerDb circuit breaker env var configuration.

Tests that ONEX_DB_CIRCUIT_THRESHOLD and ONEX_DB_CIRCUIT_RESET_TIMEOUT
environment variables are correctly parsed at module import time and used
when initializing the circuit breaker.

Ticket: OMN-1364
"""

from __future__ import annotations

import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
import pytest

from omnibase_core.container import ModelONEXContainer


@pytest.fixture
def mock_container() -> MagicMock:
    """Create mock ONEX container for HandlerDb tests."""
    return MagicMock(spec=ModelONEXContainer)


class TestCircuitBreakerEnvConfig:
    """Test suite for circuit breaker environment variable configuration."""

    def test_default_circuit_threshold(self) -> None:
        """Test default circuit breaker threshold is 5 when env var is not set."""
        from omnibase_infra.handlers import handler_db

        assert handler_db._DEFAULT_CIRCUIT_THRESHOLD == 5

    def test_default_circuit_reset_timeout(self) -> None:
        """Test default circuit breaker reset timeout is 30.0 when env var is not set."""
        from omnibase_infra.handlers import handler_db

        assert handler_db._DEFAULT_CIRCUIT_RESET_TIMEOUT == 30.0

    def test_custom_circuit_threshold_from_env(self) -> None:
        """Test ONEX_DB_CIRCUIT_THRESHOLD env var is parsed at module load."""
        with patch.dict("os.environ", {"ONEX_DB_CIRCUIT_THRESHOLD": "10"}):
            import omnibase_infra.handlers.handler_db as handler_db_mod

            importlib.reload(handler_db_mod)
            assert handler_db_mod._DEFAULT_CIRCUIT_THRESHOLD == 10

        # Reload to restore defaults for other tests
        importlib.reload(handler_db_mod)

    def test_custom_circuit_reset_timeout_from_env(self) -> None:
        """Test ONEX_DB_CIRCUIT_RESET_TIMEOUT env var is parsed at module load."""
        with patch.dict("os.environ", {"ONEX_DB_CIRCUIT_RESET_TIMEOUT": "60.0"}):
            import omnibase_infra.handlers.handler_db as handler_db_mod

            importlib.reload(handler_db_mod)
            assert handler_db_mod._DEFAULT_CIRCUIT_RESET_TIMEOUT == 60.0

        importlib.reload(handler_db_mod)

    def test_circuit_threshold_below_min_uses_default(self) -> None:
        """Test threshold below min_value=1 falls back to default 5."""
        with patch.dict("os.environ", {"ONEX_DB_CIRCUIT_THRESHOLD": "0"}):
            import omnibase_infra.handlers.handler_db as handler_db_mod

            importlib.reload(handler_db_mod)
            assert handler_db_mod._DEFAULT_CIRCUIT_THRESHOLD == 5

        importlib.reload(handler_db_mod)

    def test_circuit_threshold_above_max_uses_default(self) -> None:
        """Test threshold above max_value=100 falls back to default 5."""
        with patch.dict("os.environ", {"ONEX_DB_CIRCUIT_THRESHOLD": "200"}):
            import omnibase_infra.handlers.handler_db as handler_db_mod

            importlib.reload(handler_db_mod)
            assert handler_db_mod._DEFAULT_CIRCUIT_THRESHOLD == 5

        importlib.reload(handler_db_mod)

    def test_circuit_reset_timeout_below_min_uses_default(self) -> None:
        """Test reset timeout below min_value=1.0 falls back to default 30.0."""
        with patch.dict("os.environ", {"ONEX_DB_CIRCUIT_RESET_TIMEOUT": "0.5"}):
            import omnibase_infra.handlers.handler_db as handler_db_mod

            importlib.reload(handler_db_mod)
            assert handler_db_mod._DEFAULT_CIRCUIT_RESET_TIMEOUT == 30.0

        importlib.reload(handler_db_mod)

    def test_circuit_threshold_invalid_value_raises_error(self) -> None:
        """Test non-integer ONEX_DB_CIRCUIT_THRESHOLD raises ProtocolConfigurationError."""
        from omnibase_infra.errors import ProtocolConfigurationError

        with patch.dict("os.environ", {"ONEX_DB_CIRCUIT_THRESHOLD": "not_a_number"}):
            import omnibase_infra.handlers.handler_db as handler_db_mod

            with pytest.raises(ProtocolConfigurationError):
                importlib.reload(handler_db_mod)

        # Clean env and reload to restore defaults
        importlib.reload(handler_db_mod)

    def test_circuit_reset_timeout_invalid_value_raises_error(self) -> None:
        """Test non-numeric ONEX_DB_CIRCUIT_RESET_TIMEOUT raises ProtocolConfigurationError."""
        from omnibase_infra.errors import ProtocolConfigurationError

        with patch.dict(
            "os.environ", {"ONEX_DB_CIRCUIT_RESET_TIMEOUT": "not_a_number"}
        ):
            import omnibase_infra.handlers.handler_db as handler_db_mod

            with pytest.raises(ProtocolConfigurationError):
                importlib.reload(handler_db_mod)

        importlib.reload(handler_db_mod)

    @pytest.mark.asyncio
    async def test_initialize_passes_configured_values_to_circuit_breaker(
        self, mock_container: MagicMock
    ) -> None:
        """Test that initialize() passes module-level constants to _init_circuit_breaker."""
        from omnibase_infra.enums import EnumInfraTransportType
        from omnibase_infra.handlers.handler_db import (
            _DEFAULT_CIRCUIT_RESET_TIMEOUT,
            _DEFAULT_CIRCUIT_THRESHOLD,
            HandlerDb,
        )

        handler = HandlerDb(mock_container)
        mock_pool = MagicMock(spec=asyncpg.Pool)

        with (
            patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create,
            patch.object(handler, "_init_circuit_breaker") as mock_init_cb,
        ):
            mock_create.return_value = mock_pool
            await handler.initialize({"dsn": "postgresql://localhost/db"})

            mock_init_cb.assert_called_once_with(
                threshold=_DEFAULT_CIRCUIT_THRESHOLD,
                reset_timeout=_DEFAULT_CIRCUIT_RESET_TIMEOUT,
                service_name="db_handler",
                transport_type=EnumInfraTransportType.DATABASE,
            )

        # Shutdown outside patch context; pool mock needs close()
        mock_pool.close = AsyncMock()
        handler._circuit_breaker_initialized = False
        await handler.shutdown()
