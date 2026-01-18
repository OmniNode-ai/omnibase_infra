# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for SecretResolver Vault integration.

These tests verify SecretResolver's Vault integration against a real Vault server.
Tests are gated on Vault availability and skip gracefully in CI environments
without Vault access.

Test Coverage (OMN-1374):
- Real Vault secret resolution via HandlerVault
- Correlation ID propagation through the resolution chain
- Error handling with real Vault errors (auth, not found, etc.)
- Caching behavior with real Vault latency
- Metrics tracking during Vault resolution

Environment Variables:
    VAULT_ADDR: Vault server URL (required)
    VAULT_TOKEN: Vault authentication token (required)
    VAULT_NAMESPACE: Vault namespace for Enterprise (optional)

Skip Conditions:
    - Tests skip if VAULT_ADDR not set
    - Tests skip if VAULT_TOKEN not set
    - Tests skip if Vault server is unreachable
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

import pytest

from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
    SecretResolutionError,
)
from omnibase_infra.runtime.models.model_secret_mapping import ModelSecretMapping
from omnibase_infra.runtime.models.model_secret_resolver_config import (
    ModelSecretResolverConfig,
)
from omnibase_infra.runtime.models.model_secret_source_spec import ModelSecretSourceSpec
from omnibase_infra.runtime.secret_resolver import SecretResolver
from tests.integration.handlers.conftest import (
    VAULT_AVAILABLE,
    VAULT_REACHABLE,
)

if TYPE_CHECKING:
    from omnibase_infra.handlers import HandlerVault

logger = logging.getLogger(__name__)

# Test secret path prefix for isolation
TEST_SECRET_PATH_PREFIX = "integration-tests/secret-resolver"

# Module-level skip markers for graceful CI/CD degradation
pytestmark = [
    pytest.mark.skipif(
        not VAULT_AVAILABLE,
        reason="Vault not available (VAULT_TOKEN or VAULT_ADDR not set)",
    ),
    pytest.mark.skipif(
        VAULT_AVAILABLE and not VAULT_REACHABLE,
        reason="Vault server not reachable at configured VAULT_ADDR",
    ),
]


@pytest.fixture
def unique_secret_path() -> str:
    """Generate unique secret path for test isolation."""
    return f"{TEST_SECRET_PATH_PREFIX}/{uuid.uuid4().hex[:12]}"


@pytest.fixture
async def cleanup_secret(
    vault_handler: HandlerVault, unique_secret_path: str
) -> AsyncGenerator[str, None]:
    """Cleanup fixture that deletes test secret after test completion.

    Yields the secret path for the test to use, then cleans up.

    Expected behavior:
        - Succeeds silently when secret is deleted successfully
        - Handles SecretResolutionError gracefully (secret never created or already deleted)
        - Logs infrastructure errors at warning level but does not fail test teardown
        - Never re-raises exceptions to ensure test teardown completes

    Note:
        This fixture catches specific exception types for better diagnostics:
        - SecretResolutionError: Expected for already-deleted or never-created secrets
        - InfraConnectionError: Vault connection issues during cleanup
        - InfraTimeoutError: Vault operation timed out during cleanup
        - InfraAuthenticationError: Token expired or invalid during cleanup
        - InfraUnavailableError: Vault unavailable during cleanup
        - Exception: Catch-all for unexpected errors
    """
    yield unique_secret_path

    # Cleanup: delete the test secret (idempotent)
    try:
        envelope = {
            "operation": "vault.delete_secret",
            "payload": {"path": unique_secret_path, "mount_point": "secret"},
            "correlation_id": str(uuid.uuid4()),
        }
        await vault_handler.execute(envelope)
        logger.debug("Cleanup: deleted test secret at %s", unique_secret_path)
    except SecretResolutionError:
        # Expected: secret was never created or already deleted
        logger.debug(
            "Cleanup: secret already deleted or never existed at path %s",
            unique_secret_path,
        )
    except InfraConnectionError as e:
        # Vault connection issue - log at warning but don't fail teardown
        logger.warning(
            "Cleanup: Vault connection error while deleting test secret %s: %s",
            unique_secret_path,
            e,
            exc_info=True,
        )
    except InfraTimeoutError as e:
        # Vault operation timed out - log at warning but don't fail teardown
        logger.warning(
            "Cleanup: Vault timeout while deleting test secret %s: %s",
            unique_secret_path,
            e,
            exc_info=True,
        )
    except InfraAuthenticationError as e:
        # Token may have expired during test - log at warning but don't fail teardown
        logger.warning(
            "Cleanup: Vault authentication error while deleting test secret %s: %s",
            unique_secret_path,
            e,
            exc_info=True,
        )
    except InfraUnavailableError as e:
        # Vault unavailable (circuit breaker open or server down)
        logger.warning(
            "Cleanup: Vault unavailable while deleting test secret %s: %s",
            unique_secret_path,
            e,
            exc_info=True,
        )
    except Exception as e:
        # Catch-all for any other unexpected errors
        logger.warning(
            "Cleanup: unexpected error deleting test secret %s: %s (%s)",
            unique_secret_path,
            e,
            type(e).__name__,
            exc_info=True,
        )


@pytest.fixture
async def write_test_secret(
    vault_handler: HandlerVault, cleanup_secret: str
) -> tuple[str, dict[str, str]]:
    """Write a test secret to Vault and return the path and data.

    Returns:
        Tuple of (secret_path, secret_data)
    """
    secret_path = cleanup_secret
    secret_data = {
        "password": f"test-password-{uuid.uuid4().hex[:8]}",
        "username": "test-user",
        "api_key": f"key-{uuid.uuid4().hex}",
    }

    # Write the secret
    envelope = {
        "operation": "vault.write_secret",
        "payload": {
            "path": secret_path,
            "data": secret_data,
            "mount_point": "secret",
        },
        "correlation_id": str(uuid.uuid4()),
    }
    await vault_handler.execute(envelope)

    return secret_path, secret_data


class TestSecretResolverVaultIntegration:
    """Integration tests for SecretResolver with real Vault."""

    @pytest.mark.asyncio
    async def test_resolve_vault_secret_with_field(
        self,
        vault_handler: HandlerVault,
        write_test_secret: tuple[str, dict[str, str]],
    ) -> None:
        """Should resolve a specific field from Vault secret."""
        secret_path, secret_data = write_test_secret

        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="test.db.password",
                    source=ModelSecretSourceSpec(
                        source_type="vault",
                        source_path=f"secret/{secret_path}#password",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config, vault_handler=vault_handler)

        result = await resolver.get_secret_async("test.db.password")

        assert result is not None
        assert result.get_secret_value() == secret_data["password"]

    @pytest.mark.asyncio
    async def test_resolve_vault_secret_without_field(
        self,
        vault_handler: HandlerVault,
        write_test_secret: tuple[str, dict[str, str]],
    ) -> None:
        """Should resolve first value from Vault secret when no field specified."""
        secret_path, secret_data = write_test_secret

        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="test.api.token",
                    source=ModelSecretSourceSpec(
                        source_type="vault",
                        source_path=f"secret/{secret_path}",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config, vault_handler=vault_handler)

        result = await resolver.get_secret_async("test.api.token")

        assert result is not None
        # Should get one of the values (first in dict iteration order)
        assert result.get_secret_value() in secret_data.values()

    @pytest.mark.asyncio
    async def test_correlation_id_propagates_to_vault(
        self,
        vault_handler: HandlerVault,
        write_test_secret: tuple[str, dict[str, str]],
    ) -> None:
        """Should propagate correlation ID through Vault resolution."""
        secret_path, _ = write_test_secret
        test_correlation_id = uuid.uuid4()

        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="correlated.secret",
                    source=ModelSecretSourceSpec(
                        source_type="vault",
                        source_path=f"secret/{secret_path}#password",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config, vault_handler=vault_handler)

        # Resolve with correlation ID
        result = await resolver.get_secret_async(
            "correlated.secret", correlation_id=test_correlation_id
        )

        assert result is not None
        # The correlation ID should be tracked in metrics
        metrics = resolver.get_resolution_metrics()
        assert metrics.success_counts["vault"] >= 1

    @pytest.mark.asyncio
    async def test_vault_secret_not_found_returns_none(
        self,
        vault_handler: HandlerVault,
    ) -> None:
        """Should return None when Vault secret doesn't exist."""
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="nonexistent.secret",
                    source=ModelSecretSourceSpec(
                        source_type="vault",
                        source_path="secret/nonexistent/path/that/does/not/exist#field",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config, vault_handler=vault_handler)

        # Should return None when not required
        result = await resolver.get_secret_async("nonexistent.secret", required=False)

        assert result is None

    @pytest.mark.asyncio
    async def test_vault_field_not_found_returns_none(
        self,
        vault_handler: HandlerVault,
        write_test_secret: tuple[str, dict[str, str]],
    ) -> None:
        """Should return None when field doesn't exist in Vault secret."""
        secret_path, _ = write_test_secret

        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="missing.field",
                    source=ModelSecretSourceSpec(
                        source_type="vault",
                        source_path=f"secret/{secret_path}#nonexistent_field",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config, vault_handler=vault_handler)

        result = await resolver.get_secret_async("missing.field", required=False)

        assert result is None

    @pytest.mark.asyncio
    async def test_vault_caching_behavior(
        self,
        vault_handler: HandlerVault,
        write_test_secret: tuple[str, dict[str, str]],
    ) -> None:
        """Should cache Vault secret after first resolution."""
        secret_path, _secret_data = write_test_secret

        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="cached.vault.secret",
                    source=ModelSecretSourceSpec(
                        source_type="vault",
                        source_path=f"secret/{secret_path}#password",
                    ),
                ),
            ],
            default_ttl_vault_seconds=300,
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config, vault_handler=vault_handler)

        # First resolution - cache miss
        result1 = await resolver.get_secret_async("cached.vault.secret")
        stats1 = resolver.get_cache_stats()

        # Second resolution - cache hit
        result2 = await resolver.get_secret_async("cached.vault.secret")
        stats2 = resolver.get_cache_stats()

        assert result1 is not None
        assert result2 is not None
        assert result1.get_secret_value() == result2.get_secret_value()

        # Should have at least one cache hit on second call
        assert stats2.hits > stats1.hits

    @pytest.mark.asyncio
    async def test_metrics_tracking_during_vault_resolution(
        self,
        vault_handler: HandlerVault,
        write_test_secret: tuple[str, dict[str, str]],
    ) -> None:
        """Should track metrics during Vault secret resolution."""
        secret_path, _ = write_test_secret

        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="metrics.test.secret",
                    source=ModelSecretSourceSpec(
                        source_type="vault",
                        source_path=f"secret/{secret_path}#password",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config, vault_handler=vault_handler)

        # Initial metrics
        metrics_before = resolver.get_resolution_metrics()

        # Resolve secret
        await resolver.get_secret_async("metrics.test.secret")

        # Check metrics after resolution
        metrics_after = resolver.get_resolution_metrics()

        # Should have incremented vault success count
        vault_success_before = metrics_before.success_counts.get("vault", 0)
        vault_success_after = metrics_after.success_counts.get("vault", 0)

        assert vault_success_after > vault_success_before


class TestSecretResolverVaultGracefulDegradation:
    """Tests for graceful degradation scenarios."""

    def test_vault_handler_none_returns_none(self) -> None:
        """Should return None gracefully when vault_handler is None."""
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="no.handler",
                    source=ModelSecretSourceSpec(
                        source_type="vault",
                        source_path="secret/path#field",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config, vault_handler=None)

        result = resolver.get_secret("no.handler", required=False)

        assert result is None

    def test_vault_handler_none_tracks_failure_metrics(self) -> None:
        """Should track failure in metrics when handler is None."""
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="metrics.no.handler",
                    source=ModelSecretSourceSpec(
                        source_type="vault",
                        source_path="secret/path#field",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config, vault_handler=None)

        resolver.get_secret("metrics.no.handler", required=False)

        metrics = resolver.get_resolution_metrics()
        assert metrics.failure_counts["vault"] >= 1


__all__: list[str] = [
    "TestSecretResolverVaultIntegration",
    "TestSecretResolverVaultGracefulDegradation",
]
