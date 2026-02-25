# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
# mypy: disable-error-code="index, operator, arg-type"
"""Unit tests for HandlerInfisical.

Tests use mocked adapter to validate handler behavior including caching,
circuit breaker, and operation routing.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from pydantic import SecretStr

from omnibase_infra.adapters._internal.adapter_infisical import (
    ModelInfisicalBatchResult,
    ModelInfisicalSecretResult,
)
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.handlers.handler_infisical import (
    HANDLER_ID_INFISICAL,
    HandlerInfisical,
)


@pytest.fixture
def mock_container() -> MagicMock:
    """Provide mock ONEX container."""
    return MagicMock()


@pytest.fixture
def infisical_config() -> dict[str, object]:
    """Provide test Infisical handler configuration."""
    return {
        "host": "https://infisical.test.com",
        "client_id": "test-client-id",
        "client_secret": "test-client-secret",
        "project_id": "00000000-0000-0000-0000-000000000123",
        "environment_slug": "test",
        "secret_path": "/",
        "cache_ttl_seconds": 60.0,
        "circuit_breaker_threshold": 3,
        "circuit_breaker_reset_timeout": 30.0,
        "circuit_breaker_enabled": False,  # Disabled in tests for simplicity
    }


class TestHandlerInfisicalProperties:
    """Test handler type classification properties."""

    def test_handler_type(self, mock_container: MagicMock) -> None:
        """Test handler_type returns INFRA_HANDLER."""
        handler = HandlerInfisical(mock_container)
        assert handler.handler_type == EnumHandlerType.INFRA_HANDLER

    def test_handler_category(self, mock_container: MagicMock) -> None:
        """Test handler_category returns EFFECT."""
        handler = HandlerInfisical(mock_container)
        assert handler.handler_category == EnumHandlerTypeCategory.EFFECT


class TestHandlerInfisicalInitialization:
    """Test handler initialization."""

    @pytest.mark.asyncio
    async def test_initialize_success(
        self, mock_container: MagicMock, infisical_config: dict[str, object]
    ) -> None:
        """Test successful initialization."""
        handler = HandlerInfisical(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_infisical.AdapterInfisical"
        ) as mock_adapter_cls:
            mock_adapter = MagicMock()
            mock_adapter_cls.return_value = mock_adapter

            await handler.initialize(infisical_config)

            assert handler._initialized
            mock_adapter.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_bad_config(self, mock_container: MagicMock) -> None:
        """Test initialization with invalid config."""
        handler = HandlerInfisical(mock_container)

        with pytest.raises(
            RuntimeHostError, match="Invalid Infisical handler configuration"
        ):
            await handler.initialize({"bad": "config"})

    @pytest.mark.asyncio
    async def test_not_initialized_execute_raises(
        self, mock_container: MagicMock
    ) -> None:
        """Test execute raises when not initialized."""
        handler = HandlerInfisical(mock_container)

        with pytest.raises(RuntimeHostError, match="not initialized"):
            await handler.execute(
                {
                    "operation": "infisical.get_secret",
                    "payload": {"secret_name": "test"},
                }
            )


class TestHandlerInfisicalGetSecret:
    """Test get_secret operation."""

    @pytest.mark.asyncio
    async def test_get_secret_success(
        self, mock_container: MagicMock, infisical_config: dict[str, object]
    ) -> None:
        """Test successful single secret retrieval."""
        handler = HandlerInfisical(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_infisical.AdapterInfisical"
        ) as mock_adapter_cls:
            mock_adapter = MagicMock()
            mock_adapter_cls.return_value = mock_adapter
            mock_adapter.get_secret.return_value = ModelInfisicalSecretResult(
                key="DB_PASS",
                value=SecretStr("secret123"),
                version=1,
                secret_path="/",
                environment="test",
            )

            await handler.initialize(infisical_config)

            result = await handler.execute(
                {
                    "operation": "infisical.get_secret",
                    "payload": {"secret_name": "DB_PASS"},
                    "correlation_id": str(uuid4()),
                }
            )

            assert result.result is not None
            assert result.result["secret_name"] == "DB_PASS"
            assert result.result["value"] == "secret123"
            assert result.result["source"] == "infisical"

    @pytest.mark.asyncio
    async def test_get_secret_missing_name(
        self, mock_container: MagicMock, infisical_config: dict[str, object]
    ) -> None:
        """Test get_secret with missing secret_name."""
        handler = HandlerInfisical(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_infisical.AdapterInfisical"
        ) as mock_adapter_cls:
            mock_adapter = MagicMock()
            mock_adapter_cls.return_value = mock_adapter

            await handler.initialize(infisical_config)

            with pytest.raises(RuntimeHostError, match="secret_name"):
                await handler.execute(
                    {
                        "operation": "infisical.get_secret",
                        "payload": {},
                    }
                )


class TestHandlerInfisicalCaching:
    """Test handler-level caching."""

    @pytest.mark.asyncio
    async def test_cache_hit(
        self, mock_container: MagicMock, infisical_config: dict[str, object]
    ) -> None:
        """Test cache hit returns cached value without adapter call."""
        handler = HandlerInfisical(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_infisical.AdapterInfisical"
        ) as mock_adapter_cls:
            mock_adapter = MagicMock()
            mock_adapter_cls.return_value = mock_adapter
            mock_adapter.get_secret.return_value = ModelInfisicalSecretResult(
                key="CACHED_KEY",
                value=SecretStr("cached-value"),
                version=1,
            )

            await handler.initialize(infisical_config)

            # First call - cache miss
            await handler.execute(
                {
                    "operation": "infisical.get_secret",
                    "payload": {"secret_name": "CACHED_KEY"},
                }
            )

            # Second call - should hit cache
            result = await handler.execute(
                {
                    "operation": "infisical.get_secret",
                    "payload": {"secret_name": "CACHED_KEY"},
                }
            )

            # Adapter should only be called once (cache hit on second)
            assert mock_adapter.get_secret.call_count == 1
            assert result.result["source"] == "cache"
            assert handler._cache_hits == 1
            assert handler._cache_misses == 1

    @pytest.mark.asyncio
    async def test_cache_disabled(self, mock_container: MagicMock) -> None:
        """Test cache disabled when TTL is 0."""
        config = {
            "host": "https://infisical.test.com",
            "client_id": "test-id",
            "client_secret": "test-secret",
            "project_id": "00000000-0000-0000-0000-000000000456",
            "cache_ttl_seconds": 0.0,  # Disabled
            "circuit_breaker_enabled": False,
        }
        handler = HandlerInfisical(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_infisical.AdapterInfisical"
        ) as mock_adapter_cls:
            mock_adapter = MagicMock()
            mock_adapter_cls.return_value = mock_adapter
            mock_adapter.get_secret.return_value = ModelInfisicalSecretResult(
                key="KEY",
                value=SecretStr("val"),
            )

            await handler.initialize(config)

            await handler.execute(
                {
                    "operation": "infisical.get_secret",
                    "payload": {"secret_name": "KEY"},
                }
            )
            await handler.execute(
                {
                    "operation": "infisical.get_secret",
                    "payload": {"secret_name": "KEY"},
                }
            )

            # Both calls should hit adapter (no caching)
            assert mock_adapter.get_secret.call_count == 2

    @pytest.mark.asyncio
    async def test_invalidate_cache_all(
        self, mock_container: MagicMock, infisical_config: dict[str, object]
    ) -> None:
        """Test invalidating all cache entries."""
        handler = HandlerInfisical(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_infisical.AdapterInfisical"
        ) as mock_adapter_cls:
            mock_adapter = MagicMock()
            mock_adapter_cls.return_value = mock_adapter
            mock_adapter.get_secret.return_value = ModelInfisicalSecretResult(
                key="K",
                value=SecretStr("v"),
            )

            await handler.initialize(infisical_config)

            await handler.execute(
                {
                    "operation": "infisical.get_secret",
                    "payload": {"secret_name": "K"},
                }
            )

            count = handler.invalidate_cache()
            assert count == 1
            assert len(handler._cache) == 0


class TestHandlerInfisicalListSecrets:
    """Test list_secrets operation."""

    @pytest.mark.asyncio
    async def test_list_secrets(
        self, mock_container: MagicMock, infisical_config: dict[str, object]
    ) -> None:
        """Test listing secrets."""
        handler = HandlerInfisical(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_infisical.AdapterInfisical"
        ) as mock_adapter_cls:
            mock_adapter = MagicMock()
            mock_adapter_cls.return_value = mock_adapter
            mock_adapter.list_secrets.return_value = [
                ModelInfisicalSecretResult(key="A", value=SecretStr("v1")),
                ModelInfisicalSecretResult(key="B", value=SecretStr("v2")),
            ]

            await handler.initialize(infisical_config)

            result = await handler.execute(
                {
                    "operation": "infisical.list_secrets",
                    "payload": {},
                }
            )

            assert result.result["count"] == 2
            assert "A" in result.result["secret_keys"]
            assert "B" in result.result["secret_keys"]


class TestHandlerInfisicalBatch:
    """Test batch secret retrieval."""

    @pytest.mark.asyncio
    async def test_batch_with_cache_mix(
        self, mock_container: MagicMock, infisical_config: dict[str, object]
    ) -> None:
        """Test batch retrieval with some cached, some fetched."""
        handler = HandlerInfisical(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_infisical.AdapterInfisical"
        ) as mock_adapter_cls:
            mock_adapter = MagicMock()
            mock_adapter_cls.return_value = mock_adapter

            # First secret fetched individually (will be cached)
            mock_adapter.get_secret.return_value = ModelInfisicalSecretResult(
                key="CACHED",
                value=SecretStr("cached-val"),
            )

            await handler.initialize(infisical_config)

            # Cache one secret
            await handler.execute(
                {
                    "operation": "infisical.get_secret",
                    "payload": {"secret_name": "CACHED"},
                }
            )

            # Batch fetch including cached + new
            mock_adapter.get_secrets_batch.return_value = ModelInfisicalBatchResult(
                secrets={
                    "NEW_KEY": ModelInfisicalSecretResult(
                        key="NEW_KEY",
                        value=SecretStr("new-val"),
                    ),
                },
                errors={},
            )

            result = await handler.execute(
                {
                    "operation": "infisical.get_secrets_batch",
                    "payload": {"secret_names": ["CACHED", "NEW_KEY"]},
                }
            )

            assert result.result["from_cache"] == 1
            assert result.result["from_fetch"] == 1
            assert "CACHED" in result.result["secrets"]
            assert "NEW_KEY" in result.result["secrets"]


class TestHandlerInfisicalOperationValidation:
    """Test operation validation."""

    @pytest.mark.asyncio
    async def test_unsupported_operation(
        self, mock_container: MagicMock, infisical_config: dict[str, object]
    ) -> None:
        """Test unsupported operation raises error."""
        handler = HandlerInfisical(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_infisical.AdapterInfisical"
        ) as mock_adapter_cls:
            mock_adapter = MagicMock()
            mock_adapter_cls.return_value = mock_adapter

            await handler.initialize(infisical_config)

            with pytest.raises(RuntimeHostError, match="not supported"):
                await handler.execute(
                    {
                        "operation": "infisical.delete_secret",
                        "payload": {},
                    }
                )

    @pytest.mark.asyncio
    async def test_missing_operation(
        self, mock_container: MagicMock, infisical_config: dict[str, object]
    ) -> None:
        """Test missing operation raises error."""
        handler = HandlerInfisical(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_infisical.AdapterInfisical"
        ) as mock_adapter_cls:
            mock_adapter = MagicMock()
            mock_adapter_cls.return_value = mock_adapter

            await handler.initialize(infisical_config)

            with pytest.raises(RuntimeHostError, match="Missing or invalid"):
                await handler.execute({"payload": {}})

    @pytest.mark.asyncio
    async def test_missing_payload(
        self, mock_container: MagicMock, infisical_config: dict[str, object]
    ) -> None:
        """Test missing payload raises error."""
        handler = HandlerInfisical(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_infisical.AdapterInfisical"
        ) as mock_adapter_cls:
            mock_adapter = MagicMock()
            mock_adapter_cls.return_value = mock_adapter

            await handler.initialize(infisical_config)

            with pytest.raises(RuntimeHostError, match="payload"):
                await handler.execute({"operation": "infisical.get_secret"})


class TestHandlerInfisicalDescribe:
    """Test describe() method."""

    def test_describe_not_initialized(self, mock_container: MagicMock) -> None:
        """Test describe before initialization."""
        handler = HandlerInfisical(mock_container)
        desc = handler.describe()
        assert desc["initialized"] is False
        assert desc["handler_type"] == "infra_handler"
        assert desc["handler_category"] == "effect"

    @pytest.mark.asyncio
    async def test_describe_initialized(
        self, mock_container: MagicMock, infisical_config: dict[str, object]
    ) -> None:
        """Test describe after initialization."""
        handler = HandlerInfisical(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_infisical.AdapterInfisical"
        ) as mock_adapter_cls:
            mock_adapter = MagicMock()
            mock_adapter_cls.return_value = mock_adapter

            await handler.initialize(infisical_config)

            desc = handler.describe()
            assert desc["initialized"] is True
            assert desc["cache_ttl_seconds"] == 60.0
            assert len(desc["supported_operations"]) == 3


class TestHandlerInfisicalShutdown:
    """Test handler shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown(
        self, mock_container: MagicMock, infisical_config: dict[str, object]
    ) -> None:
        """Test shutdown cleans up resources."""
        handler = HandlerInfisical(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_infisical.AdapterInfisical"
        ) as mock_adapter_cls:
            mock_adapter = MagicMock()
            mock_adapter_cls.return_value = mock_adapter

            await handler.initialize(infisical_config)
            assert handler._initialized

            await handler.shutdown()
            assert not handler._initialized
            assert handler._adapter is None
            mock_adapter.shutdown.assert_called_once()
