# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
# mypy: disable-error-code="index, operator, arg-type"
"""Unit tests for AdapterInfisical.

Tests use mocked InfisicalSDKClient to validate adapter behavior
without requiring an actual Infisical server.
"""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import UUID

import pytest
from pydantic import SecretStr

from omnibase_infra.adapters._internal.adapter_infisical import (
    AdapterInfisical,
    ModelInfisicalAdapterConfig,
    ModelInfisicalBatchResult,
    ModelInfisicalSecretResult,
)
from omnibase_infra.errors import InfraConnectionError, SecretResolutionError


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
                InfraConnectionError, match="infisical-sdk package is not installed"
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
        with pytest.raises(SecretResolutionError, match="not initialized"):
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

        assert isinstance(result, ModelInfisicalSecretResult)
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

        assert isinstance(batch, ModelInfisicalBatchResult)
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


class TestAdapterInfisicalObservability:
    """Test observability counters."""

    def test_initial_counters_zero(
        self, adapter_config: ModelInfisicalAdapterConfig
    ) -> None:
        """Test counters start at zero."""
        adapter = AdapterInfisical(adapter_config)
        assert adapter.loads_success == 0
        assert adapter.loads_failed == 0

    def test_success_counter_incremented(
        self, adapter_config: ModelInfisicalAdapterConfig
    ) -> None:
        """Test success counter incremented on successful get_secret."""
        adapter = AdapterInfisical(adapter_config)
        mock_client = MagicMock()
        adapter._client = mock_client
        adapter._authenticated = True

        mock_result = MagicMock()
        mock_result.secretValue = "val"
        mock_result.version = 1
        mock_client.secrets.get_secret_by_name.return_value = mock_result

        adapter.get_secret("KEY")
        assert adapter.loads_success == 1
        assert adapter.loads_failed == 0

    def test_failure_counter_incremented(
        self, adapter_config: ModelInfisicalAdapterConfig
    ) -> None:
        """Test failure counter incremented on failed get_secret."""
        adapter = AdapterInfisical(adapter_config)
        mock_client = MagicMock()
        adapter._client = mock_client
        adapter._authenticated = True

        mock_client.secrets.get_secret_by_name.side_effect = Exception("boom")

        with pytest.raises(SecretResolutionError):
            adapter.get_secret("KEY")
        assert adapter.loads_success == 0
        assert adapter.loads_failed == 1

    def test_list_secrets_increments_success(
        self, adapter_config: ModelInfisicalAdapterConfig
    ) -> None:
        """Test success counter incremented on successful list_secrets."""
        adapter = AdapterInfisical(adapter_config)
        mock_client = MagicMock()
        adapter._client = mock_client
        adapter._authenticated = True

        mock_response = MagicMock()
        mock_response.secrets = []
        mock_client.secrets.list_secrets.return_value = mock_response

        adapter.list_secrets()
        assert adapter.loads_success == 1

    def test_list_secrets_increments_failure(
        self, adapter_config: ModelInfisicalAdapterConfig
    ) -> None:
        """Test failure counter incremented on failed list_secrets."""
        adapter = AdapterInfisical(adapter_config)
        mock_client = MagicMock()
        adapter._client = mock_client
        adapter._authenticated = True

        mock_client.secrets.list_secrets.side_effect = Exception("boom")

        with pytest.raises(SecretResolutionError):
            adapter.list_secrets()
        assert adapter.loads_failed == 1


class TestAdapterInfisicalExtractSecretValue:
    """Test the _extract_secret_value helper."""

    def test_extracts_camel_case(
        self, adapter_config: ModelInfisicalAdapterConfig
    ) -> None:
        """Test extraction from secretValue (camelCase)."""
        adapter = AdapterInfisical(adapter_config)
        mock_result = MagicMock()
        mock_result.secretValue = "my-value"
        assert adapter._extract_secret_value(mock_result) == "my-value"

    def test_falls_back_to_snake_case(
        self, adapter_config: ModelInfisicalAdapterConfig
    ) -> None:
        """Test fallback to secret_value (snake_case) when secretValue is None."""
        adapter = AdapterInfisical(adapter_config)
        mock_result = MagicMock(spec=[])  # No attributes by default
        mock_result.secret_value = "snake-value"
        assert adapter._extract_secret_value(mock_result) == "snake-value"

    def test_empty_string_not_replaced(
        self, adapter_config: ModelInfisicalAdapterConfig
    ) -> None:
        """Test that empty string secretValue is not replaced by fallback."""
        adapter = AdapterInfisical(adapter_config)
        mock_result = MagicMock()
        mock_result.secretValue = ""
        assert adapter._extract_secret_value(mock_result) == ""


class TestAdapterInfisicalCreateSecret:
    """Test create_secret() write-path operation."""

    def test_create_secret_not_initialized(
        self, adapter_config: ModelInfisicalAdapterConfig
    ) -> None:
        """Test create_secret raises SecretResolutionError when not initialized."""
        adapter = AdapterInfisical(adapter_config)
        with pytest.raises(SecretResolutionError, match="not initialized"):
            adapter.create_secret("NEW_KEY", "new-value")

    def test_create_secret_success(
        self, adapter_config: ModelInfisicalAdapterConfig
    ) -> None:
        """Test successful secret creation with mock SDK."""
        adapter = AdapterInfisical(adapter_config)
        mock_client = MagicMock()
        adapter._client = mock_client
        adapter._authenticated = True

        adapter.create_secret("MY_KEY", "my-value")

        mock_client.secrets.create_secret_by_name.assert_called_once_with(
            secret_name="MY_KEY",  # noqa: S106
            project_id=str(adapter_config.project_id),
            environment_slug=adapter_config.environment_slug,
            secret_path=adapter_config.secret_path,
            secret_value="my-value",  # noqa: S106
        )

    def test_create_secret_with_overrides(
        self, adapter_config: ModelInfisicalAdapterConfig
    ) -> None:
        """Test create_secret uses parameter overrides over adapter defaults."""
        adapter = AdapterInfisical(adapter_config)
        mock_client = MagicMock()
        adapter._client = mock_client
        adapter._authenticated = True

        adapter.create_secret(
            "KEY",
            "value",
            project_id="override-proj",
            environment_slug="prod",
            secret_path="/prod/secrets/",  # noqa: S106
        )

        mock_client.secrets.create_secret_by_name.assert_called_once_with(
            secret_name="KEY",  # noqa: S106
            project_id="override-proj",
            environment_slug="prod",
            secret_path="/prod/secrets/",  # noqa: S106
            secret_value="value",  # noqa: S106
        )

    def test_create_secret_sdk_failure_wraps_to_infra_connection_error(
        self, adapter_config: ModelInfisicalAdapterConfig
    ) -> None:
        """Test SDK failure is wrapped as InfraConnectionError."""
        adapter = AdapterInfisical(adapter_config)
        mock_client = MagicMock()
        adapter._client = mock_client
        adapter._authenticated = True

        mock_client.secrets.create_secret_by_name.side_effect = RuntimeError(
            "SDK internal error"
        )

        with pytest.raises(InfraConnectionError, match="Failed to create secret"):
            adapter.create_secret("KEY", "value")

    def test_create_secret_empty_value(
        self, adapter_config: ModelInfisicalAdapterConfig
    ) -> None:
        """Test that empty string is a valid secret value for create_secret."""
        adapter = AdapterInfisical(adapter_config)
        mock_client = MagicMock()
        adapter._client = mock_client
        adapter._authenticated = True

        # Should not raise; empty string is a legitimate value
        adapter.create_secret("EMPTY_KEY", "")

        mock_client.secrets.create_secret_by_name.assert_called_once_with(
            secret_name="EMPTY_KEY",  # noqa: S106
            project_id=str(adapter_config.project_id),
            environment_slug=adapter_config.environment_slug,
            secret_path=adapter_config.secret_path,
            secret_value="",
        )


class TestAdapterInfisicalUpdateSecret:
    """Test update_secret() write-path operation."""

    def test_update_secret_not_initialized(
        self, adapter_config: ModelInfisicalAdapterConfig
    ) -> None:
        """Test update_secret raises SecretResolutionError when not initialized."""
        adapter = AdapterInfisical(adapter_config)
        with pytest.raises(SecretResolutionError, match="not initialized"):
            adapter.update_secret("EXISTING_KEY", "new-value")

    def test_update_secret_success(
        self, adapter_config: ModelInfisicalAdapterConfig
    ) -> None:
        """Test successful secret update with mock SDK."""
        adapter = AdapterInfisical(adapter_config)
        mock_client = MagicMock()
        adapter._client = mock_client
        adapter._authenticated = True

        adapter.update_secret("MY_KEY", "updated-value")

        mock_client.secrets.update_secret_by_name.assert_called_once_with(
            current_secret_name="MY_KEY",  # noqa: S106
            project_id=str(adapter_config.project_id),
            environment_slug=adapter_config.environment_slug,
            secret_path=adapter_config.secret_path,
            secret_value="updated-value",  # noqa: S106
        )

    def test_update_secret_with_overrides(
        self, adapter_config: ModelInfisicalAdapterConfig
    ) -> None:
        """Test update_secret uses parameter overrides over adapter defaults."""
        adapter = AdapterInfisical(adapter_config)
        mock_client = MagicMock()
        adapter._client = mock_client
        adapter._authenticated = True

        adapter.update_secret(
            "KEY",
            "value",
            project_id="override-proj",
            environment_slug="staging",
            secret_path="/staging/secrets/",  # noqa: S106
        )

        mock_client.secrets.update_secret_by_name.assert_called_once_with(
            current_secret_name="KEY",  # noqa: S106
            project_id="override-proj",
            environment_slug="staging",
            secret_path="/staging/secrets/",  # noqa: S106
            secret_value="value",  # noqa: S106
        )

    def test_update_secret_sdk_failure_wraps_to_infra_connection_error(
        self, adapter_config: ModelInfisicalAdapterConfig
    ) -> None:
        """Test SDK failure is wrapped as InfraConnectionError."""
        adapter = AdapterInfisical(adapter_config)
        mock_client = MagicMock()
        adapter._client = mock_client
        adapter._authenticated = True

        mock_client.secrets.update_secret_by_name.side_effect = RuntimeError(
            "SDK internal error"
        )

        with pytest.raises(InfraConnectionError, match="Failed to update secret"):
            adapter.update_secret("KEY", "value")

    def test_update_secret_empty_value(
        self, adapter_config: ModelInfisicalAdapterConfig
    ) -> None:
        """Test that empty string is a valid updated value for update_secret."""
        adapter = AdapterInfisical(adapter_config)
        mock_client = MagicMock()
        adapter._client = mock_client
        adapter._authenticated = True

        # Should not raise; empty string is a legitimate value
        adapter.update_secret("KEY", "")

        mock_client.secrets.update_secret_by_name.assert_called_once_with(
            current_secret_name="KEY",  # noqa: S106
            project_id=str(adapter_config.project_id),
            environment_slug=adapter_config.environment_slug,
            secret_path=adapter_config.secret_path,
            secret_value="",
        )
