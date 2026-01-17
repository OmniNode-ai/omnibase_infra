# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Vault-Specific Infrastructure Error Class.

This module defines the InfraVaultError class for Vault-related infrastructure
errors. It extends InfraConnectionError to provide specialized error handling
for HashiCorp Vault operations.

Security Note:
    Secret paths are automatically sanitized when stored in error context
    to prevent exposure of infrastructure topology and credential organization.
    Full paths are only logged at DEBUG level if needed for debugging.
"""

from omnibase_infra.errors.error_infra import InfraConnectionError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)
from omnibase_infra.utils.util_error_sanitization import sanitize_secret_path


class InfraVaultError(InfraConnectionError):
    """Error communicating with Vault.

    Used for Vault client initialization failures, secret operations,
    token management, and retry exhaustion scenarios.

    This error extends InfraConnectionError and is specifically designed
    for HashiCorp Vault operations. The context should use
    ``transport_type=EnumInfraTransportType.VAULT``.

    Common use cases:
        - Vault client initialization failures
        - Secret read/write operation failures
        - Token renewal or management errors
        - Connection retry exhaustion
        - Vault seal/unseal operation errors

    Security:
        Secret paths passed to this error are automatically sanitized to prevent
        exposure of infrastructure details in error messages, logs, and metrics.
        The sanitization masks path segments after the mount point, e.g.,
        "secret/data/myapp/db/creds" becomes "secret/***/***".

    Example:
        >>> from omnibase_infra.enums import EnumInfraTransportType
        >>> context = ModelInfraErrorContext(
        ...     transport_type=EnumInfraTransportType.VAULT,
        ...     operation="read_secret",
        ...     target_name="vault-primary",
        ... )
        >>> # Secret path will be sanitized to "secret/***/***" in error context
        >>> raise InfraVaultError(
        ...     "Failed to read secret from Vault",
        ...     context=context,
        ...     secret_path="secret/data/myapp/credentials",
        ... )

        >>> # Token renewal failure
        >>> context = ModelInfraErrorContext(
        ...     transport_type=EnumInfraTransportType.VAULT,
        ...     operation="renew_token",
        ...     target_name="vault-primary",
        ... )
        >>> raise InfraVaultError(
        ...     "Vault token renewal failed",
        ...     context=context,
        ...     retry_count=3,
        ... )

        >>> # Client initialization failure
        >>> context = ModelInfraErrorContext(
        ...     transport_type=EnumInfraTransportType.VAULT,
        ...     operation="initialize_client",
        ...     target_name="vault-primary",
        ... )
        >>> raise InfraVaultError(
        ...     "Failed to initialize Vault client",
        ...     context=context,
        ...     host="vault.example.com",
        ...     port=8200,
        ... )
    """

    def __init__(
        self,
        message: str,
        context: ModelInfraErrorContext | None = None,
        secret_path: str | None = None,
        **extra_context: object,
    ) -> None:
        """Initialize InfraVaultError with Vault-specific context.

        Args:
            message: Human-readable error message
            context: Bundled infrastructure context (should use VAULT transport_type)
            secret_path: Optional path to the secret that caused the error.
                This path is automatically sanitized to mask sensitive segments
                (e.g., "secret/data/myapp/creds" -> "secret/***/***").
            **extra_context: Additional context information (e.g., host, port, retry_count)

        Security:
            The secret_path is sanitized before being stored in extra_context
            to prevent exposure of infrastructure topology in error messages,
            logs, and metrics. Only the mount point is preserved.
        """
        # Include sanitized secret_path in extra_context if provided
        # This prevents exposure of full secret paths in error messages/logs
        if secret_path is not None:
            sanitized_path = sanitize_secret_path(secret_path)
            if sanitized_path:
                extra_context["secret_path"] = sanitized_path

        super().__init__(
            message=message,
            context=context,
            **extra_context,
        )


__all__: list[str] = [
    "InfraVaultError",
]
