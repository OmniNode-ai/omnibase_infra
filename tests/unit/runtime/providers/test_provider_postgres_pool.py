# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ProviderPostgresPool SSL passthrough (OMN-14597).

RDS enforces TLS by default; ``asyncpg.create_pool`` previously received no
``ssl`` kwarg at all, and ``ModelPostgresPoolConfig`` had no field to source
one from. These tests prove ``ssl_mode`` / ``ssl_ca_file`` are threaded
through to the real ``asyncpg.create_pool`` call, and that the default
(unset) config is unaffected — local Docker/dev must not change behavior.
"""

from __future__ import annotations

import ssl as ssl_lib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.runtime.models.model_postgres_pool_config import (
    ModelPostgresPoolConfig,
)
from omnibase_infra.runtime.providers.provider_postgres_pool import (
    ProviderPostgresPool,
    _resolve_ssl_context,
)

pytestmark = pytest.mark.unit

_MODULE = "omnibase_infra.runtime.providers.provider_postgres_pool"


def _config(**overrides: object) -> ModelPostgresPoolConfig:
    defaults: dict[str, object] = {"host": "db.example.com", "database": "testdb"}
    defaults.update(overrides)
    return ModelPostgresPoolConfig(**defaults)


class TestResolveSslContext:
    """Unit tests for the pure `_resolve_ssl_context` translation helper."""

    def test_unset_ssl_mode_returns_none(self) -> None:
        """Default config resolves to None — no CA-bundle context needed."""
        assert _resolve_ssl_context(_config()) is None

    @pytest.mark.parametrize("mode", ["disable", "allow", "prefer", "require"])
    def test_passthrough_modes_return_none(self, mode: str) -> None:
        """asyncpg resolves these sslmode strings itself; no context is built."""
        assert _resolve_ssl_context(_config(ssl_mode=mode)) is None

    def test_verify_full_builds_context_with_hostname_check_enabled(self) -> None:
        mock_context = MagicMock(spec=ssl_lib.SSLContext)
        with patch(
            f"{_MODULE}.ssl_lib.create_default_context",
            return_value=mock_context,
        ) as mock_create_ctx:
            result = _resolve_ssl_context(
                _config(ssl_mode="verify-full", ssl_ca_file="/etc/rds/ca.pem")
            )

        mock_create_ctx.assert_called_once_with(cafile="/etc/rds/ca.pem")
        assert result is mock_context
        assert mock_context.check_hostname is True

    def test_verify_ca_builds_context_with_hostname_check_disabled(self) -> None:
        mock_context = MagicMock(spec=ssl_lib.SSLContext)
        with patch(
            f"{_MODULE}.ssl_lib.create_default_context",
            return_value=mock_context,
        ):
            result = _resolve_ssl_context(
                _config(ssl_mode="verify-ca", ssl_ca_file="/etc/rds/ca.pem")
            )

        assert result is mock_context
        assert mock_context.check_hostname is False


class TestProviderCreateThreadsSsl:
    """Proves ProviderPostgresPool.create() passes `ssl=` to asyncpg.create_pool."""

    @pytest.mark.asyncio
    async def test_default_config_passes_ssl_none(self) -> None:
        """RED before this fix: no `ssl` kwarg was passed at all — RDS (which
        enforces TLS) could not be reached and there was no config field to
        change that. GREEN: `ssl=None` is now passed explicitly, which is
        behaviorally identical to omission (asyncpg's own `ssl` default is
        also None), proving local Docker/dev is unaffected.
        """
        provider = ProviderPostgresPool(_config())
        mock_create_pool = AsyncMock(return_value=AsyncMock())
        with patch(f"{_MODULE}.asyncpg.create_pool", new=mock_create_pool):
            await provider.create()

        assert mock_create_pool.call_args.kwargs["ssl"] is None

    @pytest.mark.asyncio
    async def test_require_mode_threads_string_through(self) -> None:
        provider = ProviderPostgresPool(_config(ssl_mode="require"))
        mock_create_pool = AsyncMock(return_value=AsyncMock())
        with patch(f"{_MODULE}.asyncpg.create_pool", new=mock_create_pool):
            await provider.create()

        assert mock_create_pool.call_args.kwargs["ssl"] == "require"

    @pytest.mark.asyncio
    async def test_verify_full_mode_threads_ssl_context_through(self) -> None:
        """The RDS-targeted mode: an SSLContext built from ssl_ca_file must
        reach the real asyncpg.create_pool call, not just the resolver."""
        provider = ProviderPostgresPool(
            _config(ssl_mode="verify-full", ssl_ca_file="/etc/rds/ca.pem")
        )
        mock_create_pool = AsyncMock(return_value=AsyncMock())
        mock_context = MagicMock(spec=ssl_lib.SSLContext)
        with (
            patch(f"{_MODULE}.asyncpg.create_pool", new=mock_create_pool),
            patch(
                f"{_MODULE}.ssl_lib.create_default_context",
                return_value=mock_context,
            ),
        ):
            await provider.create()

        assert mock_create_pool.call_args.kwargs["ssl"] is mock_context
