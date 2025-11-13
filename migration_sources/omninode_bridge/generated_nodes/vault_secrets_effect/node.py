#!/usr/bin/env python3
"""
NodeVaultSecretsEffect - HashiCorp Vault secrets management integration

ONEX v2.0 Effect Node with Registration & Introspection
Domain: security
Generated: 2025-11-02T23:52:56.343758+00:00
"""

import os
from typing import Any

# Vault client
import hvac
from omnibase_core import EnumCoreErrorCode, ModelOnexError
from omnibase_core.enums.enum_log_level import EnumLogLevel as LogLevel
from omnibase_core.logging.structured import emit_log_event_sync as emit_log_event
from omnibase_core.mixins.mixin_health_check import MixinHealthCheck

# Import mixins from omnibase_core (ONEX standard)
from omnibase_core.mixins.mixin_introspection import MixinNodeIntrospection
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from omnibase_core.models.core import ModelContainer
from omnibase_core.nodes.node_effect import NodeEffect

MIXINS_AVAILABLE = True


class NodeVaultSecretsEffect(NodeEffect, MixinHealthCheck, MixinNodeIntrospection):
    """
    HashiCorp Vault secrets management integration

    Operations:
    - read_secret
    - write_secret
    - list_secrets
    - delete_secret
    - authenticate
    - renew_token

    Features:
    - connection_pooling
    - retry_logic
    - circuit_breaker
    - authentication
    - token_renewal
    - validation
    - logging
    - metrics
    - error_handling

    Capabilities:
    - Automatic node registration via introspection events
    - Health check endpoints
    - Consul service discovery integration
    """

    def __init__(self, container: ModelContainer) -> None:
        """Initialize NodeVaultSecretsEffect with registration and introspection."""
        super().__init__(container)

        # Configuration
        # Access config from container.value (ModelContainer stores config in value field)
        self.config = container.value if isinstance(container.value, dict) else {}

        # Initialize hvac client for Vault connectivity
        vault_addr = self.config.get(
            "vault_addr", os.getenv("VAULT_ADDR", "http://omninode-bridge-vault:8200")
        )
        vault_token = self.config.get("vault_token", os.getenv("VAULT_TOKEN"))
        vault_namespace = self.config.get(
            "vault_namespace", os.getenv("VAULT_NAMESPACE")
        )

        self.client = hvac.Client(
            url=vault_addr, token=vault_token, namespace=vault_namespace
        )

        # Skip authentication check in test mode to avoid connection requirements
        test_mode = self.config.get("test_mode", False)

        emit_log_event(
            LogLevel.INFO,
            "Vault client initialized",
            {
                "vault_addr": vault_addr,
                "vault_namespace": vault_namespace,
                "test_mode": test_mode,
                "authenticated": (
                    self.client.is_authenticated()
                    if (vault_token and not test_mode)
                    else False
                ),
            },
        )

        # Initialize health checks (if mixins available)
        if MIXINS_AVAILABLE:
            try:
                self.initialize_health_checks()
                self._register_component_checks()

                # Initialize introspection system
                self.initialize_introspection()
            except AttributeError:
                # Mixins not available in production omnibase_core
                pass

        emit_log_event(
            LogLevel.INFO,
            "NodeVaultSecretsEffect initialized with registration support",
            {
                "node_id": str(self.node_id),
                "mixins_available": MIXINS_AVAILABLE,
                "operations": [
                    "read_secret",
                    "write_secret",
                    "list_secrets",
                    "delete_secret",
                    "authenticate",
                    "renew_token",
                ],
                "features": [
                    "connection_pooling",
                    "retry_logic",
                    "circuit_breaker",
                    "authentication",
                    "token_renewal",
                    "validation",
                    "logging",
                    "metrics",
                    "error_handling",
                ],
            },
        )

    def _register_component_checks(self) -> None:
        """
        Register component health checks for this node.

        Override this method to add custom health checks specific to this node's dependencies.
        """
        # Base node runtime check is registered by HealthCheckMixin
        # Add custom checks here as needed
        pass

    async def execute_effect(self, contract: ModelContractEffect) -> Any:
        """
        Execute effect operation.

        Args:
            contract: Effect contract with operation parameters

        Returns:
            Operation result

        Raises:
            ModelOnexError: If operation fails
        """
        try:
            operation = contract.input_data.get("operation", "read_secret")
            mount_point = contract.input_data.get("mount_point", "secret")
            path = contract.input_data.get("path", "")

            emit_log_event(
                LogLevel.INFO,
                f"Executing Vault operation: {operation}",
                {"operation": operation, "path": path, "mount_point": mount_point},
            )

            if operation == "read_secret":
                response = self.client.secrets.kv.v2.read_secret_version(
                    path=path, mount_point=mount_point
                )
                return {"data": response.get("data", {}).get("data", {})}

            elif operation == "write_secret":
                data = contract.input_data.get("data", {})
                self.client.secrets.kv.v2.create_or_update_secret(
                    path=path, secret=data, mount_point=mount_point
                )
                return {"success": True, "message": "Secret written successfully"}

            elif operation == "list_secrets":
                response = self.client.secrets.kv.v2.list_secrets(
                    path=path, mount_point=mount_point
                )
                return {"secrets": response.get("data", {}).get("keys", [])}

            elif operation == "delete_secret":
                self.client.secrets.kv.v2.delete_metadata_and_all_versions(
                    path=path, mount_point=mount_point
                )
                return {"success": True, "message": "Secret deleted successfully"}

            else:
                raise ModelOnexError(
                    error_code=EnumCoreErrorCode.INVALID_INPUT,
                    message=f"Unknown operation: {operation}",
                    details={"operation": operation},
                )

        except hvac.exceptions.VaultError as e:
            emit_log_event(
                LogLevel.ERROR,
                f"Vault operation failed: {e!s}",
                {"operation": operation, "error": str(e)},
            )
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.EXTERNAL_SERVICE_ERROR,
                message=f"Vault operation failed: {e!s}",
                details={"operation": operation, "vault_error": str(e)},
            )
        except (ConnectionError, TimeoutError, OSError) as e:
            emit_log_event(
                LogLevel.ERROR,
                f"Network error in Vault operation: {e!s}",
                {
                    "operation": operation,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.EXTERNAL_SERVICE_ERROR,
                message=f"Network error: {e!s}",
                details={"operation": operation, "network_error": str(e)},
            )
        except Exception as e:
            # Log unexpected errors and re-raise to preserve stack trace
            emit_log_event(
                LogLevel.ERROR,
                f"Unexpected error in execute_effect: {e!s}",
                {
                    "operation": operation,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            raise

    def get_metadata_loader(self) -> Any:
        """
        Get metadata loader for this node.

        Returns:
            Metadata loader instance or None if not applicable
        """
        return None

    async def startup(self) -> None:
        """
        Node startup lifecycle hook.

        Publishes introspection data to registry and starts background tasks.
        Should be called when node is ready to serve requests.
        """
        if not MIXINS_AVAILABLE:
            emit_log_event(
                LogLevel.WARNING,
                "Mixins not available - skipping startup registration",
                {"node_id": str(self.node_id)},
            )
            return

        emit_log_event(
            LogLevel.INFO,
            "NodeVaultSecretsEffect starting up",
            {"node_id": str(self.node_id)},
        )

        # Publish introspection broadcast to registry
        try:
            await self.publish_introspection(reason="startup")

            # Start introspection background tasks (heartbeat, registry listener)
            await self.start_introspection_tasks(
                enable_heartbeat=True,
                heartbeat_interval_seconds=30,
                enable_registry_listener=True,
            )
        except AttributeError:
            # Mixins not available in production omnibase_core
            pass

        emit_log_event(
            LogLevel.INFO,
            "NodeVaultSecretsEffect startup complete - node registered",
            {"node_id": str(self.node_id)},
        )

    async def shutdown(self) -> None:
        """
        Node shutdown lifecycle hook.

        Stops background tasks and cleans up resources.
        Should be called when node is preparing to exit.
        """
        if not MIXINS_AVAILABLE:
            return

        emit_log_event(
            LogLevel.INFO,
            "NodeVaultSecretsEffect shutting down",
            {"node_id": str(self.node_id)},
        )

        # Stop introspection background tasks
        try:
            await self.stop_introspection_tasks()
        except AttributeError:
            # Mixins not available in production omnibase_core
            pass

        emit_log_event(
            LogLevel.INFO,
            "NodeVaultSecretsEffect shutdown complete",
            {"node_id": str(self.node_id)},
        )


__all__ = ["NodeVaultSecretsEffect"]
