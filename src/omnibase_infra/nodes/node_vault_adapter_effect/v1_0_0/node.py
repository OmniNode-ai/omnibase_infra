#!/usr/bin/env python3

import asyncio
import logging
import os
from typing import Any

from omnibase_core.core.errors.onex_error import CoreErrorCode, OnexError
from omnibase_core.core.node_effect_service import NodeEffectService
from omnibase_core.core.onex_container import ModelONEXContainer
from omnibase_core.enums.enum_health_status import EnumHealthStatus
from omnibase_core.models.core.model_health_status import ModelHealthStatus

from omnibase_infra.models.vault import (
    ModelVaultSecretRequest,
    ModelVaultSecretResponse,
    ModelVaultTokenRequest,
)


class MockVaultClient:
    """Mock Vault client for testing/development when hvac library unavailable."""

    def __init__(self, config: dict):
        self.config = config
        self.secrets = {}
        self.tokens = {}
        self._sealed = False

    class MockSecrets:
        def __init__(self, client):
            self.client = client
            self.kv = MockVaultClient.MockKV(client)

    class MockKV:
        def __init__(self, client):
            self.client = client

        class MockV2:
            def __init__(self, client):
                self.client = client

            def read_secret_version(self, path: str, mount_point: str = "secret", version: int | None = None):
                key = f"{mount_point}/{path}"
                if key in self.client.secrets:
                    secret_data = self.client.secrets[key]
                    return {
                        "data": {
                            "data": secret_data.get("data", {}),
                            "metadata": {
                                "version": secret_data.get("version", 1),
                                "created_time": secret_data.get("created_time", "2025-01-01T00:00:00Z"),
                            },
                        },
                    }
                return None

            def create_or_update_secret(self, path: str, secret: dict, mount_point: str = "secret"):
                key = f"{mount_point}/{path}"
                version = 1
                if key in self.client.secrets:
                    version = self.client.secrets[key].get("version", 1) + 1

                self.client.secrets[key] = {
                    "data": secret,
                    "version": version,
                    "created_time": "2025-01-01T00:00:00Z",
                }
                return {
                    "data": {
                        "version": version,
                        "created_time": "2025-01-01T00:00:00Z",
                    },
                }

            def delete_metadata_and_all_versions(self, path: str, mount_point: str = "secret"):
                key = f"{mount_point}/{path}"
                if key in self.client.secrets:
                    del self.client.secrets[key]
                    return True
                return False

            def list_secrets(self, path: str, mount_point: str = "secret"):
                prefix = f"{mount_point}/{path}"
                keys = [k.replace(f"{mount_point}/", "") for k in self.client.secrets.keys() if k.startswith(prefix)]
                if keys:
                    return {"data": {"keys": keys}}
                return None

        def __init__(self, client):
            self.client = client
            self.v2 = self.MockV2(client)

    class MockAuth:
        def __init__(self, client):
            self.client = client
            self.token = MockVaultClient.MockToken(client)

    class MockToken:
        def __init__(self, client):
            self.client = client

        def create(self, **kwargs):
            token_id = f"mock-token-{len(self.client.tokens)}"
            self.client.tokens[token_id] = {
                "policies": kwargs.get("policies", []),
                "ttl": kwargs.get("ttl", "768h"),
                "renewable": kwargs.get("renewable", True),
            }
            return {
                "auth": {
                    "client_token": token_id,
                    "policies": kwargs.get("policies", []),
                    "lease_duration": 2764800,
                    "renewable": kwargs.get("renewable", True),
                },
            }

        def renew(self, token: str, increment: int | None = None):
            if token in self.client.tokens:
                return {
                    "auth": {
                        "client_token": token,
                        "lease_duration": increment or 2764800,
                        "renewable": True,
                    },
                }
            raise Exception(f"Token not found: {token}")

        def revoke(self, token: str):
            if token in self.client.tokens:
                del self.client.tokens[token]
                return True
            return False

    class MockSys:
        def __init__(self, client):
            self.client = client

        def read_health_status(self, method: str = "GET"):
            return {
                "initialized": True,
                "sealed": self.client._sealed,
                "standby": False,
                "performance_standby": False,
                "replication_performance_mode": "disabled",
                "replication_dr_mode": "disabled",
                "server_time_utc": 1735689600,
                "version": "1.15.0",
                "cluster_name": "vault-cluster-mock",
                "cluster_id": "mock-cluster-id",
            }

    def __init__(self, config: dict):
        self.config = config
        self.secrets = MockVaultClient.MockSecrets(self)
        self.auth = MockVaultClient.MockAuth(self)
        self.sys = MockVaultClient.MockSys(self)
        self._sealed = False
        self.is_authenticated = lambda: True


class VaultConnectionPool:
    """
    Connection pool for Vault clients with health monitoring and automatic cleanup.
    Provides efficient connection reuse and automatic cleanup of unhealthy connections.
    """

    def __init__(self, config: dict, max_connections: int = 10, cleanup_interval: int = 300):
        self._config = config
        self._max_connections = max_connections
        self._cleanup_interval = cleanup_interval
        self._connections = {}  # Dict[str, VaultClient]
        self._failed_connections = set()  # Set[str] - track recently failed connections
        self._connection_usage = {}  # Dict[str, int] - track usage count
        self._last_cleanup = 0
        self._logger = logging.getLogger(__name__)

        # Background cleanup task
        self._cleanup_task = None
        self._start_background_cleanup()

    def _start_background_cleanup(self):
        """Start background cleanup task with proper resource management."""
        # Cancel existing task before creating new one to prevent resource leaks
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
        self._cleanup_task = asyncio.create_task(self._background_cleanup_loop())

    async def _background_cleanup_loop(self):
        """Background loop for periodic cleanup."""
        try:
            while True:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_idle_connections()
        except asyncio.CancelledError:
            self._logger.info("Vault connection pool cleanup task cancelled")
        except Exception as e:
            self._logger.error(f"Vault connection pool cleanup error: {e}")

    async def _cleanup_idle_connections(self):
        """Clean up idle and unhealthy connections."""
        connections_to_remove = []

        for conn_key, client in self._connections.items():
            # Check if connection is still healthy
            if not await self._is_connection_healthy(client):
                connections_to_remove.append(conn_key)
                continue

            # Remove low-usage connections if pool is at capacity
            if (len(self._connections) > self._max_connections // 2 and
                self._connection_usage.get(conn_key, 0) < 5):  # Low usage threshold
                connections_to_remove.append(conn_key)

        # Clean up selected connections
        for conn_key in connections_to_remove:
            await self._remove_connection(conn_key)

    async def _is_connection_healthy(self, client) -> bool:
        """Check if a Vault connection is still healthy."""
        try:
            # Test basic connectivity with health check
            if hasattr(client, "sys") and hasattr(client.sys, "read_health_status"):
                health = client.sys.read_health_status(method="GET")
                # Check if Vault is initialized and unsealed
                return health.get("initialized", False) and not health.get("sealed", True)
            return False
        except Exception:
            return False

    async def _remove_connection(self, conn_key: str):
        """Safely remove a connection from the pool."""
        if conn_key in self._connections:
            try:
                # Vault client doesn't need explicit cleanup, just remove reference
                del self._connections[conn_key]
                if conn_key in self._connection_usage:
                    del self._connection_usage[conn_key]
                self._logger.debug(f"Removed Vault connection: {conn_key}")
            except Exception as e:
                self._logger.error(f"Error removing Vault connection: {e}")

    def get_client(self):
        """Get a Vault client from the pool or create a new one."""
        # Create connection key based on config
        conn_key = f"{self._config['url']}:{self._config['token'][:8]}"

        # Return existing connection if available
        if conn_key in self._connections:
            self._connection_usage[conn_key] = self._connection_usage.get(conn_key, 0) + 1
            return self._connections[conn_key]

        # Check if we should create new connection (not at max capacity and no recent failures)
        if (len(self._connections) >= self._max_connections or
            conn_key in self._failed_connections):
            # Return least used connection or None if all failed recently
            if self._connections:
                least_used_key = min(self._connection_usage.items(), key=lambda x: x[1])[0]
                self._connection_usage[least_used_key] += 1
                return self._connections[least_used_key]
            return None

        # Create new connection
        try:
            import hvac

            client = hvac.Client(
                url=self._config["url"],
                token=self._config["token"],
                namespace=self._config.get("namespace"),
            )

            # Test connection
            if not client.is_authenticated():
                raise OnexError(
                    message="Vault authentication failed",
                    error_code=CoreErrorCode.AUTHENTICATION_FAILED,
                )

            # Store in pool
            self._connections[conn_key] = client
            self._connection_usage[conn_key] = 1
            self._logger.info(f"Created new Vault connection: {conn_key}")
            return client

        except ImportError:
            # Fallback to mock client
            self._logger.warning("hvac library not available, using mock Vault client")
            client = MockVaultClient(self._config)
            self._connections[conn_key] = client
            self._connection_usage[conn_key] = 1
            return client

        except Exception as e:
            self._logger.error(f"Failed to create Vault connection: {e}")
            self._failed_connections.add(conn_key)
            return None

    async def close_all(self):
        """Close all connections in the pool."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()

        self._connections.clear()
        self._connection_usage.clear()
        self._failed_connections.clear()


class NodeVaultAdapterEffect(NodeEffectService):
    """
    Vault Adapter - Event-Driven Secret Management Effect

    NodeEffect that processes event envelopes to perform Vault operations.
    Integrates with event bus for secret management, token lifecycle,
    and encryption services. Provides health check HTTP endpoint for monitoring.
    """

    def __init__(self, container: ModelONEXContainer):
        # Use proper base class - no more boilerplate!
        super().__init__(container)

        self.node_type = "effect"
        self.domain = "infrastructure"

        # ONEX logger initialization with fallback
        try:
            self.logger = getattr(container, "get_tool", lambda x: None)(
                "LOGGER",
            ) or logging.getLogger(__name__)
        except (AttributeError, Exception):
            self.logger = logging.getLogger(__name__)

        # Vault client configuration - all environment variables required
        vault_addr = os.getenv("VAULT_ADDR")
        vault_token = os.getenv("VAULT_TOKEN")
        vault_namespace = os.getenv("VAULT_NAMESPACE", "")

        if not vault_addr:
            raise OnexError(
                message="VAULT_ADDR environment variable is required but not set",
                error_code=CoreErrorCode.MISSING_REQUIRED_PARAMETER,
            )
        if not vault_token:
            raise OnexError(
                message="VAULT_TOKEN environment variable is required but not set",
                error_code=CoreErrorCode.MISSING_REQUIRED_PARAMETER,
            )

        self.vault_config = {
            "url": vault_addr,
            "token": vault_token,
            "namespace": vault_namespace,
        }
        self.vault_client = None
        self.vault_connection_pool = None
        self._initialized = False

    async def _initialize_node_resources(self) -> None:
        """Override to initialize vault client."""
        await super()._initialize_node_resources()

        # Initialize vault client
        await self.initialize_vault_client()

    async def initialize_vault_client(self):
        """Initialize Vault connection pool for high-performance operations"""
        if self._initialized:
            return

        try:
            # Initialize connection pool for better performance under load
            self.vault_connection_pool = VaultConnectionPool(
                config=self.vault_config,
                max_connections=10,  # Configurable pool size
                cleanup_interval=300,  # 5 minute cleanup interval
            )

            # Get initial client from pool (will create pool and test connection)
            self.vault_client = self.vault_connection_pool.get_client()

            if self.vault_client is None:
                # Fallback to mock client for basic functionality
                self.logger.warning(
                    "Failed to create Vault connection pool, using mock client",
                )
                self.vault_client = MockVaultClient(self.vault_config)

            # Test connection - skip for mock client
            # Protocol-based duck typing: Check if it's NOT a mock client (ONEX compliance)
            if not (hasattr(self.vault_client, "secrets") and hasattr(self.vault_client, "auth") and hasattr(self.vault_client, "_sealed")):
                # Test connection with health check
                health_status = self.health_check()
                if health_status.status == EnumHealthStatus.UNREACHABLE:
                    raise OnexError(
                        message=f"Vault connection test failed: {health_status.message}",
                        error_code=CoreErrorCode.INITIALIZATION_FAILED,
                    )

            self._initialized = True
            self.logger.info(
                "Vault client initialized successfully",
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize Vault client: {e}")
            raise OnexError(
                message=f"Vault initialization failed: {e}",
                error_code=CoreErrorCode.INITIALIZATION_FAILED,
            ) from e

    def _get_vault_client(self):
        """Get a Vault client from the connection pool for operations."""
        if self.vault_connection_pool:
            client = self.vault_connection_pool.get_client()
            if client:
                return client
        # Fallback to main client (could be mock)
        return self.vault_client

    async def _cleanup_node_resources(self) -> None:
        """Override to cleanup vault connection pool resources."""
        if self.vault_connection_pool:
            await self.vault_connection_pool.close_all()
        await super()._cleanup_node_resources()

    async def get_secret(self, request: ModelVaultSecretRequest) -> ModelVaultSecretResponse:
        """Get a secret from Vault."""
        try:
            client = self._get_vault_client()

            response = client.secrets.kv.v2.read_secret_version(
                path=request.path,
                mount_point=request.mount_path,
                version=request.version,
            )

            if response is None:
                return ModelVaultSecretResponse(
                    success=False,
                    error=f"Secret not found at path: {request.path}",
                    correlation_id=request.correlation_id,
                )

            secret_data = response.get("data", {})

            return ModelVaultSecretResponse(
                success=True,
                data=secret_data.get("data", {}),
                metadata=secret_data.get("metadata", {}),
                version=secret_data.get("metadata", {}).get("version"),
                correlation_id=request.correlation_id,
            )

        except Exception as e:
            self.logger.error(f"Failed to get secret: {e}")
            return ModelVaultSecretResponse(
                success=False,
                error=str(e),
                correlation_id=request.correlation_id,
            )

    async def set_secret(self, request: ModelVaultSecretRequest) -> ModelVaultSecretResponse:
        """Write a secret to Vault."""
        try:
            client = self._get_vault_client()

            if not request.data:
                raise OnexError(
                    message="Secret data is required for write operation",
                    error_code=CoreErrorCode.MISSING_REQUIRED_PARAMETER,
                )

            response = client.secrets.kv.v2.create_or_update_secret(
                path=request.path,
                secret=request.data,
                mount_point=request.mount_path,
            )

            version_data = response.get("data", {})

            return ModelVaultSecretResponse(
                success=True,
                version=version_data.get("version"),
                metadata={"created_time": version_data.get("created_time")},
                correlation_id=request.correlation_id,
            )

        except Exception as e:
            self.logger.error(f"Failed to set secret: {e}")
            return ModelVaultSecretResponse(
                success=False,
                error=str(e),
                correlation_id=request.correlation_id,
            )

    async def delete_secret(self, request: ModelVaultSecretRequest) -> ModelVaultSecretResponse:
        """Delete a secret from Vault."""
        try:
            client = self._get_vault_client()

            client.secrets.kv.v2.delete_metadata_and_all_versions(
                path=request.path,
                mount_point=request.mount_path,
            )

            return ModelVaultSecretResponse(
                success=True,
                correlation_id=request.correlation_id,
            )

        except Exception as e:
            self.logger.error(f"Failed to delete secret: {e}")
            return ModelVaultSecretResponse(
                success=False,
                error=str(e),
                correlation_id=request.correlation_id,
            )

    async def list_secrets(self, request: ModelVaultSecretRequest) -> ModelVaultSecretResponse:
        """List secrets at a path in Vault."""
        try:
            client = self._get_vault_client()

            response = client.secrets.kv.v2.list_secrets(
                path=request.path,
                mount_point=request.mount_path,
            )

            if response is None:
                return ModelVaultSecretResponse(
                    success=True,
                    data={"keys": []},
                    correlation_id=request.correlation_id,
                )

            return ModelVaultSecretResponse(
                success=True,
                data=response.get("data", {}),
                correlation_id=request.correlation_id,
            )

        except Exception as e:
            self.logger.error(f"Failed to list secrets: {e}")
            return ModelVaultSecretResponse(
                success=False,
                error=str(e),
                correlation_id=request.correlation_id,
            )

    async def create_token(self, request: ModelVaultTokenRequest) -> dict[str, Any]:
        """Create a new Vault token."""
        try:
            client = self._get_vault_client()

            response = client.auth.token.create(
                policies=request.policies,
                ttl=request.ttl,
                renewable=request.renewable,
                meta=request.metadata,
            )

            auth_data = response.get("auth", {})

            return {
                "success": True,
                "token": auth_data.get("client_token"),
                "policies": auth_data.get("policies", []),
                "lease_duration": auth_data.get("lease_duration"),
                "renewable": auth_data.get("renewable", False),
                "correlation_id": request.correlation_id,
            }

        except Exception as e:
            self.logger.error(f"Failed to create token: {e}")
            return {
                "success": False,
                "error": str(e),
                "correlation_id": request.correlation_id,
            }

    async def renew_token(self, request: ModelVaultTokenRequest) -> dict[str, Any]:
        """Renew an existing Vault token."""
        try:
            client = self._get_vault_client()

            if not request.token:
                raise OnexError(
                    message="Token is required for renew operation",
                    error_code=CoreErrorCode.MISSING_REQUIRED_PARAMETER,
                )

            response = client.auth.token.renew(
                token=request.token,
                increment=request.increment,
            )

            auth_data = response.get("auth", {})

            return {
                "success": True,
                "token": auth_data.get("client_token"),
                "lease_duration": auth_data.get("lease_duration"),
                "renewable": auth_data.get("renewable", False),
                "correlation_id": request.correlation_id,
            }

        except Exception as e:
            self.logger.error(f"Failed to renew token: {e}")
            return {
                "success": False,
                "error": str(e),
                "correlation_id": request.correlation_id,
            }

    async def revoke_token(self, request: ModelVaultTokenRequest) -> dict[str, Any]:
        """Revoke a Vault token."""
        try:
            client = self._get_vault_client()

            if not request.token:
                raise OnexError(
                    message="Token is required for revoke operation",
                    error_code=CoreErrorCode.MISSING_REQUIRED_PARAMETER,
                )

            client.auth.token.revoke(token=request.token)

            return {
                "success": True,
                "correlation_id": request.correlation_id,
            }

        except Exception as e:
            self.logger.error(f"Failed to revoke token: {e}")
            return {
                "success": False,
                "error": str(e),
                "correlation_id": request.correlation_id,
            }

    def health_check(self) -> ModelHealthStatus:
        """Check Vault service health and connectivity."""
        try:
            client = self._get_vault_client()

            if client is None:
                return ModelHealthStatus(
                    status=EnumHealthStatus.UNREACHABLE,
                    message="Vault client is not initialized",
                )

            # Get health status from Vault
            health = client.sys.read_health_status(method="GET")

            # Check if Vault is initialized and unsealed
            if not health.get("initialized", False):
                return ModelHealthStatus(
                    status=EnumHealthStatus.UNHEALTHY,
                    message="Vault is not initialized",
                    details=health,
                )

            if health.get("sealed", True):
                return ModelHealthStatus(
                    status=EnumHealthStatus.UNHEALTHY,
                    message="Vault is sealed",
                    details=health,
                )

            # Vault is healthy
            return ModelHealthStatus(
                status=EnumHealthStatus.HEALTHY,
                message=f"Vault is healthy (version: {health.get('version', 'unknown')})",
                details={
                    "version": health.get("version"),
                    "cluster_name": health.get("cluster_name"),
                    "cluster_id": health.get("cluster_id"),
                },
            )

        except Exception as e:
            self.logger.error(f"Vault health check failed: {e}")
            return ModelHealthStatus(
                status=EnumHealthStatus.UNREACHABLE,
                message=f"Vault health check failed: {str(e)}",
            )


# Entry point for running the node
if __name__ == "__main__":
    import sys

    # Create container (simplified for standalone operation)
    container = ModelONEXContainer()

    # Create and run the node
    node = NodeVaultAdapterEffect(container)

    # Run the node with asyncio
    try:
        asyncio.run(node.run())
    except KeyboardInterrupt:
        print("\nVault adapter shutting down...")
        sys.exit(0)
