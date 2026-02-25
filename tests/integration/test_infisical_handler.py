# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
# mypy: disable-error-code="index, operator, arg-type"
"""Integration tests for HandlerInfisical.

Tests the full handler lifecycle: initialize -> execute -> shutdown.
Uses mocked adapter to avoid requiring an actual Infisical server.
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
from omnibase_infra.handlers.handler_infisical import HandlerInfisical


@pytest.fixture
def mock_container() -> MagicMock:
    """Provide mock ONEX container."""
    return MagicMock()


@pytest.fixture
def handler_config() -> dict[str, object]:
    """Full handler configuration for integration tests."""
    return {
        "host": "https://infisical.integration.test",
        "client_id": "integration-client-id",
        "client_secret": "integration-client-secret",
        "project_id": "00000000-0000-0000-0000-000000000789",
        "environment_slug": "staging",
        "secret_path": "/app",
        "cache_ttl_seconds": 120.0,
        "circuit_breaker_threshold": 5,
        "circuit_breaker_reset_timeout": 30.0,
        "circuit_breaker_enabled": True,
    }


class TestInfisicalHandlerLifecycle:
    """Integration tests for full handler lifecycle."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(
        self,
        mock_container: MagicMock,
        handler_config: dict[str, object],
    ) -> None:
        """Test initialize -> get_secret -> list_secrets -> batch -> shutdown."""
        handler = HandlerInfisical(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_infisical.AdapterInfisical"
        ) as mock_adapter_cls:
            mock_adapter = MagicMock()
            mock_adapter_cls.return_value = mock_adapter

            # Initialize
            await handler.initialize(handler_config)
            assert handler._initialized

            # 1. Get single secret
            mock_adapter.get_secret.return_value = ModelInfisicalSecretResult(
                key="API_KEY",
                value=SecretStr("sk-test-123"),
                version=5,
                secret_path="/app",
                environment="staging",
            )

            correlation_id = str(uuid4())
            result = await handler.execute(
                {
                    "operation": "infisical.get_secret",
                    "payload": {"secret_name": "API_KEY"},
                    "correlation_id": correlation_id,
                }
            )
            assert result.result["value"] == "sk-test-123"
            assert result.result["source"] == "infisical"

            # 2. Same secret again - should hit cache
            result2 = await handler.execute(
                {
                    "operation": "infisical.get_secret",
                    "payload": {"secret_name": "API_KEY"},
                }
            )
            assert result2.result["source"] == "cache"
            assert result2.result["value"] == "sk-test-123"

            # 3. List secrets
            mock_adapter.list_secrets.return_value = [
                ModelInfisicalSecretResult(key="KEY_A", value=SecretStr("a")),
                ModelInfisicalSecretResult(key="KEY_B", value=SecretStr("b")),
                ModelInfisicalSecretResult(key="KEY_C", value=SecretStr("c")),
            ]

            result3 = await handler.execute(
                {
                    "operation": "infisical.list_secrets",
                    "payload": {},
                }
            )
            assert result3.result["count"] == 3

            # 4. Batch fetch
            mock_adapter.get_secrets_batch.return_value = ModelInfisicalBatchResult(
                secrets={
                    "NEW_1": ModelInfisicalSecretResult(
                        key="NEW_1", value=SecretStr("val1")
                    ),
                    "NEW_2": ModelInfisicalSecretResult(
                        key="NEW_2", value=SecretStr("val2")
                    ),
                },
                errors={},
            )

            result4 = await handler.execute(
                {
                    "operation": "infisical.get_secrets_batch",
                    "payload": {"secret_names": ["API_KEY", "NEW_1", "NEW_2"]},
                }
            )
            # API_KEY should come from cache
            assert result4.result["from_cache"] == 1
            assert result4.result["from_fetch"] == 2

            # 5. Verify metrics
            desc = handler.describe()
            assert desc["initialized"] is True
            assert desc["cache_hits"] > 0
            assert desc["total_fetches"] > 0

            # 6. Shutdown
            await handler.shutdown()
            assert not handler._initialized
            assert handler._adapter is None

    @pytest.mark.asyncio
    async def test_cache_invalidation_lifecycle(
        self,
        mock_container: MagicMock,
        handler_config: dict[str, object],
    ) -> None:
        """Test cache invalidation forces re-fetch."""
        handler = HandlerInfisical(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_infisical.AdapterInfisical"
        ) as mock_adapter_cls:
            mock_adapter = MagicMock()
            mock_adapter_cls.return_value = mock_adapter
            mock_adapter.get_secret.return_value = ModelInfisicalSecretResult(
                key="ROTATED",
                value=SecretStr("old-value"),
            )

            await handler.initialize(handler_config)

            # Fetch and cache
            r1 = await handler.execute(
                {
                    "operation": "infisical.get_secret",
                    "payload": {"secret_name": "ROTATED"},
                }
            )
            assert r1.result["value"] == "old-value"

            # Invalidate
            handler.invalidate_cache("ROTATED")

            # Update mock to return new value
            mock_adapter.get_secret.return_value = ModelInfisicalSecretResult(
                key="ROTATED",
                value=SecretStr("new-rotated-value"),
            )

            # Re-fetch should get new value
            r2 = await handler.execute(
                {
                    "operation": "infisical.get_secret",
                    "payload": {"secret_name": "ROTATED"},
                }
            )
            assert r2.result["value"] == "new-rotated-value"
            assert r2.result["source"] == "infisical"

            await handler.shutdown()
