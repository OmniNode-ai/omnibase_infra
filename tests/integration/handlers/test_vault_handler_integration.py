# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for HandlerVault against remote Vault infrastructure.

These tests validate HandlerVault behavior against a real HashiCorp Vault server.
They require a running Vault instance and will be skipped gracefully if Vault
is not available.

CI/CD Graceful Skip Behavior
============================

These tests skip gracefully in CI/CD environments without Vault access:

Skip Conditions (Two-Phase):
    Phase 1 - Environment Variables:
        - Skips if VAULT_ADDR not set
        - Skips if VAULT_TOKEN not set

    Phase 2 - Reachability:
        - Skips if Vault server health endpoint is unreachable
        - Uses HTTP request to /v1/sys/health with 5-second timeout

Example CI/CD Output::

    $ pytest tests/integration/handlers/test_vault_handler_integration.py -v
    test_vault_describe SKIPPED (Vault not available - VAULT_TOKEN not set)
    test_vault_write_and_read_secret SKIPPED (Vault not available - VAULT_TOKEN not set)

Test Categories
===============

- Connection Tests: Validate basic connectivity and handler metadata
- Secret CRUD Tests: Verify read/write/delete/list operations
- Error Handling Tests: Test error handling for missing secrets

Environment Variables
=====================

    VAULT_ADDR: Vault server URL (required - skip if not set)
        Example: http://localhost:8200 or http://${REMOTE_INFRA_HOST}:8200
    VAULT_TOKEN: Vault authentication token (required - skip if not set)
    VAULT_NAMESPACE: Optional Vault namespace (for Enterprise)

Remote Infrastructure:
    Vault is available on the ONEX development infrastructure server.
    See tests/infrastructure_config.py for the default REMOTE_INFRA_HOST value.
    The server IP can be overridden via the REMOTE_INFRA_HOST environment variable.

Test Isolation
==============

    - Each test uses a unique secret path to prevent collisions
    - Cleanup fixtures ensure test secrets are deleted after each test
    - Tests use a dedicated mount point (default: "secret")
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from omnibase_core.types import JsonType

    from omnibase_infra.handlers import HandlerVault

# Import fixture availability flags from conftest
from tests.integration.handlers.conftest import VAULT_AVAILABLE, VAULT_REACHABLE

# =============================================================================
# Test Configuration and Skip Conditions
# =============================================================================

# Module-level markers - skip all tests if Vault is not available
pytestmark = [
    pytest.mark.skipif(
        not VAULT_AVAILABLE,
        reason="Vault not available (VAULT_TOKEN not set)",
    ),
    pytest.mark.skipif(
        VAULT_AVAILABLE and not VAULT_REACHABLE,
        reason="Vault server not reachable at configured VAULT_ADDR",
    ),
]

# Test configuration constants
TEST_MOUNT_POINT = "secret"
TEST_SECRET_PATH_PREFIX = "integration-tests/vault-handler"


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def unique_secret_path() -> str:
    """Generate unique secret path for test isolation.

    Returns:
        Unique path under the integration test prefix.
    """
    return f"{TEST_SECRET_PATH_PREFIX}/{uuid.uuid4().hex[:12]}"


@pytest.fixture
async def cleanup_secret(
    vault_handler: HandlerVault,
    unique_secret_path: str,
) -> AsyncGenerator[str, None]:
    """Cleanup fixture to delete test secrets after test completion.

    Yields the unique_secret_path and ensures cleanup after test,
    regardless of test outcome.

    Cleanup Behavior:
        - Deletes the secret at unique_secret_path after test
        - Idempotent: safe if secret doesn't exist (test may not have created it)
        - Runs after test completion (success or failure)
        - Ignores cleanup errors to prevent test pollution

    Test Isolation:
        This fixture enables test isolation by ensuring secrets created
        during tests are cleaned up. Combined with unique_secret_path,
        this guarantees no secret path conflicts between tests.

    Args:
        vault_handler: Initialized HandlerVault fixture.
        unique_secret_path: Unique secret path fixture.

    Yields:
        The unique secret path.
    """
    yield unique_secret_path

    # Cleanup: delete the test secret
    # Idempotent: succeeds even if secret doesn't exist
    try:
        envelope: dict[str, JsonType] = {
            "operation": "vault.delete_secret",
            "payload": {
                "path": unique_secret_path,
                "mount_point": TEST_MOUNT_POINT,
            },
            "correlation_id": str(uuid.uuid4()),
        }
        await vault_handler.execute(envelope)
    except Exception:
        pass  # Ignore cleanup errors - secret may not exist or was already deleted


# =============================================================================
# Connection Tests - Validate basic connectivity
# =============================================================================


class TestHandlerVaultConnection:
    """Tests for HandlerVault connection functionality."""

    @pytest.mark.asyncio
    async def test_vault_describe(
        self,
        vault_handler: HandlerVault,
    ) -> None:
        """Test HandlerVault describe() returns handler metadata."""
        description = vault_handler.describe()

        assert description["handler_type"] == "vault"
        assert description["initialized"] is True
        assert "supported_operations" in description
        assert "vault.read_secret" in description["supported_operations"]
        assert "vault.write_secret" in description["supported_operations"]
        assert "vault.delete_secret" in description["supported_operations"]
        assert "vault.list_secrets" in description["supported_operations"]


# =============================================================================
# Secret CRUD Tests - Verify read/write/delete/list operations
# =============================================================================


class TestHandlerVaultSecretCRUD:
    """Tests for HandlerVault secret CRUD operations."""

    @pytest.mark.asyncio
    async def test_vault_write_and_read_secret(
        self,
        vault_handler: HandlerVault,
        cleanup_secret: str,
    ) -> None:
        """Test writing and reading a secret from Vault.

        Creates a secret, reads it back, and verifies the data matches.
        Cleanup fixture ensures the secret is deleted after test.
        """
        secret_path = cleanup_secret
        test_data = {
            "username": "test_user",
            "password": "test_password_123",
            "api_key": "ak_test_12345",
        }
        correlation_id = str(uuid.uuid4())

        # Write secret
        write_envelope: dict[str, JsonType] = {
            "operation": "vault.write_secret",
            "payload": {
                "path": secret_path,
                "data": test_data,
                "mount_point": TEST_MOUNT_POINT,
            },
            "correlation_id": correlation_id,
        }

        write_result = await vault_handler.execute(write_envelope)
        assert write_result.result["status"] == "success"
        assert "version" in write_result.result["payload"]

        # Read secret back
        read_envelope: dict[str, JsonType] = {
            "operation": "vault.read_secret",
            "payload": {
                "path": secret_path,
                "mount_point": TEST_MOUNT_POINT,
            },
            "correlation_id": str(uuid.uuid4()),
        }

        read_result = await vault_handler.execute(read_envelope)
        assert read_result.result["status"] == "success"

        # Verify read data matches written data
        read_data = read_result.result["payload"]["data"]
        assert read_data["username"] == test_data["username"]
        assert read_data["password"] == test_data["password"]
        assert read_data["api_key"] == test_data["api_key"]

    @pytest.mark.asyncio
    async def test_vault_delete_secret(
        self,
        vault_handler: HandlerVault,
        unique_secret_path: str,
    ) -> None:
        """Test deleting a secret from Vault.

        Creates a secret, deletes it, and verifies it no longer exists.
        """
        secret_path = unique_secret_path
        test_data = {"key": "value_to_delete"}

        # Write secret first
        write_envelope: dict[str, JsonType] = {
            "operation": "vault.write_secret",
            "payload": {
                "path": secret_path,
                "data": test_data,
                "mount_point": TEST_MOUNT_POINT,
            },
            "correlation_id": str(uuid.uuid4()),
        }
        await vault_handler.execute(write_envelope)

        # Delete the secret
        delete_envelope: dict[str, JsonType] = {
            "operation": "vault.delete_secret",
            "payload": {
                "path": secret_path,
                "mount_point": TEST_MOUNT_POINT,
            },
            "correlation_id": str(uuid.uuid4()),
        }

        delete_result = await vault_handler.execute(delete_envelope)
        assert delete_result.result["status"] == "success"
        assert delete_result.result["payload"]["deleted"] is True

        # Verify secret no longer exists (should raise SecretResolutionError)
        from omnibase_infra.errors import SecretResolutionError

        read_envelope: dict[str, JsonType] = {
            "operation": "vault.read_secret",
            "payload": {
                "path": secret_path,
                "mount_point": TEST_MOUNT_POINT,
            },
            "correlation_id": str(uuid.uuid4()),
        }

        with pytest.raises(SecretResolutionError):
            await vault_handler.execute(read_envelope)

    @pytest.mark.asyncio
    async def test_vault_list_secrets(
        self,
        vault_handler: HandlerVault,
        unique_secret_path: str,
    ) -> None:
        """Test listing secrets at a path in Vault.

        Creates multiple secrets under a path and verifies they appear in list.
        """
        # Use unique_secret_path as a directory
        base_path = unique_secret_path
        secret_names = ["secret1", "secret2", "secret3"]
        created_paths: list[str] = []

        try:
            # Create multiple secrets
            for name in secret_names:
                secret_path = f"{base_path}/{name}"
                created_paths.append(secret_path)

                write_envelope: dict[str, JsonType] = {
                    "operation": "vault.write_secret",
                    "payload": {
                        "path": secret_path,
                        "data": {"name": name},
                        "mount_point": TEST_MOUNT_POINT,
                    },
                    "correlation_id": str(uuid.uuid4()),
                }
                await vault_handler.execute(write_envelope)

            # List secrets at base path
            list_envelope: dict[str, JsonType] = {
                "operation": "vault.list_secrets",
                "payload": {
                    "path": base_path,
                    "mount_point": TEST_MOUNT_POINT,
                },
                "correlation_id": str(uuid.uuid4()),
            }

            list_result = await vault_handler.execute(list_envelope)
            assert list_result.result["status"] == "success"

            # Verify all secrets appear in list
            keys = list_result.result["payload"]["keys"]
            assert isinstance(keys, list)
            for name in secret_names:
                assert name in keys

        finally:
            # Cleanup: delete all created secrets
            for path in created_paths:
                try:
                    delete_envelope: dict[str, JsonType] = {
                        "operation": "vault.delete_secret",
                        "payload": {
                            "path": path,
                            "mount_point": TEST_MOUNT_POINT,
                        },
                        "correlation_id": str(uuid.uuid4()),
                    }
                    await vault_handler.execute(delete_envelope)
                except Exception:
                    pass  # Ignore cleanup errors

    @pytest.mark.asyncio
    async def test_vault_update_existing_secret(
        self,
        vault_handler: HandlerVault,
        cleanup_secret: str,
    ) -> None:
        """Test updating an existing secret in Vault.

        Creates a secret, updates it with new data, and verifies the update.
        """
        secret_path = cleanup_secret

        # Write initial secret
        initial_data = {"version": "1", "value": "initial"}
        write_envelope: dict[str, JsonType] = {
            "operation": "vault.write_secret",
            "payload": {
                "path": secret_path,
                "data": initial_data,
                "mount_point": TEST_MOUNT_POINT,
            },
            "correlation_id": str(uuid.uuid4()),
        }

        initial_result = await vault_handler.execute(write_envelope)
        initial_version = initial_result.result["payload"]["version"]

        # Update with new data
        updated_data = {"version": "2", "value": "updated", "new_field": "added"}
        update_envelope: dict[str, JsonType] = {
            "operation": "vault.write_secret",
            "payload": {
                "path": secret_path,
                "data": updated_data,
                "mount_point": TEST_MOUNT_POINT,
            },
            "correlation_id": str(uuid.uuid4()),
        }

        update_result = await vault_handler.execute(update_envelope)
        updated_version = update_result.result["payload"]["version"]

        # Version should be incremented
        assert updated_version > initial_version

        # Read and verify updated data
        read_envelope: dict[str, JsonType] = {
            "operation": "vault.read_secret",
            "payload": {
                "path": secret_path,
                "mount_point": TEST_MOUNT_POINT,
            },
            "correlation_id": str(uuid.uuid4()),
        }

        read_result = await vault_handler.execute(read_envelope)
        read_data = read_result.result["payload"]["data"]

        assert read_data["version"] == "2"
        assert read_data["value"] == "updated"
        assert read_data["new_field"] == "added"


# =============================================================================
# Error Handling Tests - Test error handling for edge cases
# =============================================================================


class TestHandlerVaultErrors:
    """Tests for HandlerVault error handling."""

    @pytest.mark.asyncio
    async def test_vault_read_nonexistent_secret(
        self,
        vault_handler: HandlerVault,
    ) -> None:
        """Test reading a non-existent secret raises SecretResolutionError.

        Verifies that attempting to read a secret that doesn't exist
        raises the appropriate error type.
        """
        from omnibase_infra.errors import SecretResolutionError

        nonexistent_path = f"{TEST_SECRET_PATH_PREFIX}/nonexistent-{uuid.uuid4().hex}"

        read_envelope: dict[str, JsonType] = {
            "operation": "vault.read_secret",
            "payload": {
                "path": nonexistent_path,
                "mount_point": TEST_MOUNT_POINT,
            },
            "correlation_id": str(uuid.uuid4()),
        }

        with pytest.raises(SecretResolutionError) as exc_info:
            await vault_handler.execute(read_envelope)

        # Verify error context contains expected information
        error = exc_info.value
        assert error.model.context is not None
        assert error.model.context["transport_type"].value == "vault"
        assert error.model.context["operation"] == "vault.read_secret"

    @pytest.mark.asyncio
    async def test_vault_list_nonexistent_path(
        self,
        vault_handler: HandlerVault,
    ) -> None:
        """Test listing a non-existent path raises SecretResolutionError.

        Verifies that attempting to list secrets at a path that doesn't
        exist raises the appropriate error.
        """
        from omnibase_infra.errors import SecretResolutionError

        nonexistent_path = (
            f"{TEST_SECRET_PATH_PREFIX}/nonexistent-dir-{uuid.uuid4().hex}"
        )

        list_envelope: dict[str, JsonType] = {
            "operation": "vault.list_secrets",
            "payload": {
                "path": nonexistent_path,
                "mount_point": TEST_MOUNT_POINT,
            },
            "correlation_id": str(uuid.uuid4()),
        }

        with pytest.raises(SecretResolutionError):
            await vault_handler.execute(list_envelope)


# =============================================================================
# Token Renewal Tests
# =============================================================================


class TestHandlerVaultTokenRenewal:
    """Tests for HandlerVault token renewal functionality."""

    @pytest.mark.asyncio
    async def test_vault_renew_token(
        self,
        vault_handler: HandlerVault,
    ) -> None:
        """Test token renewal operation.

        Verifies that token renewal works and returns expected response.
        Note: Token renewal may fail if using a non-renewable token.
        """
        from omnibase_infra.errors import InfraAuthenticationError

        correlation_id = str(uuid.uuid4())
        renew_envelope: dict[str, JsonType] = {
            "operation": "vault.renew_token",
            "payload": {},
            "correlation_id": correlation_id,
        }

        try:
            result = await vault_handler.execute(renew_envelope)

            # Verify success response
            assert result.result["status"] == "success"
            assert "renewable" in result.result["payload"]
            assert "lease_duration" in result.result["payload"]

        except InfraAuthenticationError:
            # Token may not be renewable (e.g., root token)
            # This is acceptable behavior for the test
            pytest.skip("Token is not renewable (may be root token)")
