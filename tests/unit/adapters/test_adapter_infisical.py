# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
# mypy: disable-error-code="index, operator, arg-type"
"""Unit tests for AdapterInfisical.

Tests use mocked InfisicalSDKClient to validate adapter behavior
without requiring an actual Infisical server.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest
from pydantic import SecretStr

from omnibase_infra.adapters._internal.adapter_infisical import (
    AdapterInfisical,
    InfisicalBatchResult,
    InfisicalSecretResult,
    ModelInfisicalAdapterConfig,
)


@pytest.fixture
def adapter_config() -> ModelInfisicalAdapterConfig:
    """Provide test Infisical adapter configuration."""
    return ModelInfisicalAdapterConfig(
        host="https://infisical.example.com",
        client_id=SecretStr("test-client-id"),
        client_secret=SecretStr("test-client-secret"),
        project_id=UUID("00000000-0000-0000-0000-000000000123"),
        environment_slug="dev",
        secret_path="/",  # noqa: S106
    )


@pytest.fixture
def mock_sdk_client() -> MagicMock:
    """Provide mocked InfisicalSDKClient."""
    client = MagicMock()
    client.auth.universal_auth.login = MagicMock()
    client.secrets.get_secret_by_name = MagicMock()
    client.secrets.list_secrets = MagicMock()
    return client


class TestAdapterInfisicalInitialization:
    """Test adapter initialization and authentication."""

    def test_config_creation(self, adapter_config: ModelInfisicalAdapterConfig) -> None:
        """Test configuration model creation."""
        assert adapter_config.host == "https://infisical.example.com"
        assert adapter_config.client_id.get_secret_value() == "test-client-id"
        assert adapter_config.project_id == UUID("00000000-0000-0000-0000-000000000123")
        assert adapter_config.environment_slug == "dev"
        assert adapter_config.secret_path == "/"

    def test_adapter_not_authenticated_before_init(
        self, adapter_config: ModelInfisicalAdapterConfig
    ) -> None:
        """Test adapter is not authenticated before initialization."""
        adapter = AdapterInfisical(adapter_config)
        assert not adapter.is_authenticated

    def test_initialize_success(
        self,
        adapter_config: ModelInfisicalAdapterConfig,
        mock_sdk_client: MagicMock,
    ) -> None:
        """Test successful initialization and authentication."""
        import sys

        mock_module = MagicMock()
        mock_module.InfisicalSDKClient = MagicMock(return_value=mock_sdk_client)

        adapter = AdapterInfisical(adapter_config)

        # Patch the infisical_sdk module that initialize() imports lazily
        original = sys.modules.get("infisical_sdk")
        sys.modules["infisical_sdk"] = mock_module
        try:
            adapter.initialize()
            assert adapter.is_authenticated
            mock_module.InfisicalSDKClient.assert_called_once_with(
                host="https://infisical.example.com",
            )
            mock_sdk_client.auth.universal_auth.login.assert_called_once_with(
                client_id="test-client-id",
                client_secret="test-client-secret",  # noqa: S106
            )
        finally:
            if original is not None:
                sys.modules["infisical_sdk"] = original
            else:
                sys.modules.pop("infisical_sdk", None)

    def test_initialize_missing_sdk(
        self, adapter_config: ModelInfisicalAdapterConfig
    ) -> None:
        """Test initialization failure when SDK is not installed."""
        adapter = AdapterInfisical(adapter_config)
        import sys

        # Ensure infisical_sdk is NOT available
        original = sys.modules.get("infisical_sdk")
        sys.modules["infisical_sdk"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(
                RuntimeError, match="infisical-sdk package is not installed"
            ):
                adapter.initialize()
        finally:
            if original is not None:
                sys.modules["infisical_sdk"] = original
            else:
                sys.modules.pop("infisical_sdk", None)


class TestAdapterInfisicalGetSecret:
    """Test single secret retrieval."""

    def test_get_secret_not_initialized(
        self, adapter_config: ModelInfisicalAdapterConfig
    ) -> None:
        """Test get_secret raises when not initialized."""
        adapter = AdapterInfisical(adapter_config)
        with pytest.raises(RuntimeError, match="not initialized"):
            adapter.get_secret("my_secret")

    def test_get_secret_success(
        self, adapter_config: ModelInfisicalAdapterConfig
    ) -> None:
        """Test successful secret retrieval."""
        adapter = AdapterInfisical(adapter_config)
        mock_client = MagicMock()
        adapter._client = mock_client
        adapter._authenticated = True

        # Mock SDK response
        mock_result = MagicMock()
        mock_result.secretValue = "super-secret-value"
        mock_result.version = 3
        mock_client.secrets.get_secret_by_name.return_value = mock_result

        result = adapter.get_secret("DB_PASSWORD")

        assert isinstance(result, InfisicalSecretResult)
        assert result.key == "DB_PASSWORD"
        assert result.value.get_secret_value() == "super-secret-value"
        assert result.version == 3
        assert result.environment == "dev"

    def test_get_secret_with_overrides(
        self, adapter_config: ModelInfisicalAdapterConfig
    ) -> None:
        """Test secret retrieval with parameter overrides."""
        adapter = AdapterInfisical(adapter_config)
        mock_client = MagicMock()
        adapter._client = mock_client
        adapter._authenticated = True

        mock_result = MagicMock()
        mock_result.secretValue = "prod-value"
        mock_result.version = 1
        mock_client.secrets.get_secret_by_name.return_value = mock_result

        result = adapter.get_secret(
            "API_KEY",
            project_id="proj-456",
            environment_slug="prod",
            secret_path="/api",  # noqa: S106
        )

        assert result.value.get_secret_value() == "prod-value"
        mock_client.secrets.get_secret_by_name.assert_called_once_with(
            secret_name="API_KEY",  # noqa: S106
            project_id="proj-456",
            environment_slug="prod",
            secret_path="/api",  # noqa: S106
            expand_secret_references=True,
            view_secret_value=True,
            include_imports=True,
        )


class TestAdapterInfisicalListSecrets:
    """Test secret listing."""

    def test_list_secrets_success(
        self, adapter_config: ModelInfisicalAdapterConfig
    ) -> None:
        """Test successful secret listing."""
        adapter = AdapterInfisical(adapter_config)
        mock_client = MagicMock()
        adapter._client = mock_client
        adapter._authenticated = True

        mock_secret1 = MagicMock()
        mock_secret1.secretKey = "SECRET_A"
        mock_secret1.secretValue = "val-a"
        mock_secret1.version = 1

        mock_secret2 = MagicMock()
        mock_secret2.secretKey = "SECRET_B"
        mock_secret2.secretValue = "val-b"
        mock_secret2.version = 2

        mock_response = MagicMock()
        mock_response.secrets = [mock_secret1, mock_secret2]
        mock_client.secrets.list_secrets.return_value = mock_response

        results = adapter.list_secrets()

        assert len(results) == 2
        assert results[0].key == "SECRET_A"
        assert results[0].value.get_secret_value() == "val-a"
        assert results[1].key == "SECRET_B"


class TestAdapterInfisicalBatchFetch:
    """Test batch secret fetching."""

    def test_batch_all_success(
        self, adapter_config: ModelInfisicalAdapterConfig
    ) -> None:
        """Test batch fetch with all successes."""
        adapter = AdapterInfisical(adapter_config)
        mock_client = MagicMock()
        adapter._client = mock_client
        adapter._authenticated = True

        def mock_get(secret_name, **kwargs):
            result = MagicMock()
            result.secretValue = f"value-{secret_name}"
            result.version = 1
            return result

        mock_client.secrets.get_secret_by_name.side_effect = mock_get

        batch = adapter.get_secrets_batch(["KEY_A", "KEY_B"])

        assert isinstance(batch, InfisicalBatchResult)
        assert len(batch.secrets) == 2
        assert len(batch.errors) == 0
        assert batch.secrets["KEY_A"].value.get_secret_value() == "value-KEY_A"

    def test_batch_partial_failure(
        self, adapter_config: ModelInfisicalAdapterConfig
    ) -> None:
        """Test batch fetch with partial failures."""
        adapter = AdapterInfisical(adapter_config)
        mock_client = MagicMock()
        adapter._client = mock_client
        adapter._authenticated = True

        call_count = 0

        def mock_get(secret_name, **kwargs):
            nonlocal call_count
            call_count += 1
            if secret_name == "BAD_KEY":
                raise RuntimeError("Not found")
            result = MagicMock()
            result.secretValue = f"value-{secret_name}"
            result.version = 1
            return result

        mock_client.secrets.get_secret_by_name.side_effect = mock_get

        batch = adapter.get_secrets_batch(["GOOD_KEY", "BAD_KEY"])

        assert len(batch.secrets) == 1
        assert len(batch.errors) == 1
        assert "BAD_KEY" in batch.errors


class TestAdapterInfisicalShutdown:
    """Test adapter shutdown."""

    def test_shutdown(self, adapter_config: ModelInfisicalAdapterConfig) -> None:
        """Test shutdown clears state."""
        adapter = AdapterInfisical(adapter_config)
        adapter._client = MagicMock()
        adapter._authenticated = True

        adapter.shutdown()

        assert adapter._client is None
        assert not adapter.is_authenticated
