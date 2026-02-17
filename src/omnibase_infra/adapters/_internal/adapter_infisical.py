# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Thin Infisical SDK wrapper adapter.

This adapter provides a minimal, testable interface over the ``infisicalsdk``
package. It performs NO caching, NO circuit breaking, and NO audit logging.
Those cross-cutting concerns belong to ``HandlerInfisical``.

Architecture Rule (OMN-2286):
    This adapter lives in ``_internal/`` and MUST NOT be imported directly
    by application code. All access goes through ``HandlerInfisical``.
    Only handler code and tests may import this module.

Security:
    - All secret values are wrapped in ``SecretStr`` before being returned.
    - Client credentials (client_id, client_secret) are accepted as ``SecretStr``
      and only unwrapped at the point of SDK invocation.
    - No secret values are logged at any level.
    - Error messages are sanitized via ``sanitize_secret_path`` and
      ``sanitize_error_message`` to prevent leaking infrastructure details.

.. versionadded:: 0.9.0
    Initial implementation for OMN-2286.
"""

from __future__ import annotations

import logging

from pydantic import SecretStr

from omnibase_infra.adapters.models.model_infisical_batch_result import (
    ModelInfisicalBatchResult,
)
from omnibase_infra.adapters.models.model_infisical_config import (
    ModelInfisicalAdapterConfig,
)
from omnibase_infra.adapters.models.model_infisical_secret_result import (
    ModelInfisicalSecretResult,
)
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConnectionError,
    ModelInfraErrorContext,
    SecretResolutionError,
)
from omnibase_infra.utils.util_error_sanitization import (
    sanitize_error_message,
    sanitize_secret_path,
)

logger = logging.getLogger(__name__)


class AdapterInfisical:
    """Thin wrapper around the Infisical SDK.

    This adapter handles:
    - SDK client initialization and authentication
    - Single and batch secret retrieval
    - ``SecretStr`` wrapping of all returned values

    It does NOT handle:
    - Caching (owned by handler)
    - Circuit breaking (owned by handler)
    - Retry logic (owned by handler)
    - Audit events (owned by handler)
    """

    def __init__(self, config: ModelInfisicalAdapterConfig) -> None:
        self._config = config
        self._client: object | None = None  # InfisicalSDKClient (lazy import)
        self._authenticated: bool = False
        # Observability counters
        self._loads_success: int = 0
        self._loads_failed: int = 0

    @property
    def is_authenticated(self) -> bool:
        """Whether the adapter has successfully authenticated."""
        return self._authenticated

    @property
    def loads_success(self) -> int:
        """Number of successful secret loads."""
        return self._loads_success

    @property
    def loads_failed(self) -> int:
        """Number of failed secret loads."""
        return self._loads_failed

    def initialize(self) -> None:
        """Initialize the Infisical SDK client and authenticate.

        Uses Universal Auth with machine identity credentials.

        Raises:
            InfraConnectionError: If SDK is not installed or initialization fails.
        """
        try:
            from infisical_sdk import InfisicalSDKClient
        except ImportError as e:
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.INFISICAL,
                operation="initialize",
                target_name="infisical-sdk",
            )
            raise InfraConnectionError(
                "infisical-sdk package is not installed. "
                "Install with: pip install 'infisicalsdk>=1.0.15,<2.0.0'",
                context=ctx,
            ) from e

        try:
            self._client = InfisicalSDKClient(
                host=self._config.host,
            )
            # Authenticate using Universal Auth (machine identity)
            self._client.auth.universal_auth.login(  # type: ignore[union-attr]
                client_id=self._config.client_id.get_secret_value(),
                client_secret=self._config.client_secret.get_secret_value(),
            )
            self._authenticated = True
            logger.info(
                "Infisical adapter initialized and authenticated",
            )
        except Exception as e:
            self._authenticated = False
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.INFISICAL,
                operation="initialize",
                target_name="infisical-adapter",
            )
            raise InfraConnectionError(
                f"Failed to initialize Infisical client: {sanitize_error_message(e)}",
                context=ctx,
            ) from e

    def _extract_secret_value(self, result: object) -> str:
        """Extract the secret value from an SDK result object.

        The Infisical SDK may return the value under either ``secretValue``
        (camelCase) or ``secret_value`` (snake_case) depending on the SDK
        version. This method checks both attribute names with an explicit
        ``is None`` guard so that an empty string (a valid secret value) is
        not silently replaced by the fallback attribute.

        Args:
            result: SDK result object (single secret or list entry).

        Returns:
            The raw secret value as a string.
        """
        raw_value = getattr(result, "secretValue", None)
        if raw_value is None:
            raw_value = getattr(result, "secret_value", "")
        return str(raw_value)

    def get_secret(
        self,
        secret_name: str,
        *,
        project_id: str | None = None,
        environment_slug: str | None = None,
        secret_path: str | None = None,
    ) -> ModelInfisicalSecretResult:
        """Retrieve a single secret by name.

        Args:
            secret_name: The secret key/name to retrieve.
            project_id: Override default project ID.
            environment_slug: Override default environment slug.
            secret_path: Override default secret path.

        Returns:
            ModelInfisicalSecretResult with the secret value wrapped in SecretStr.

        Raises:
            SecretResolutionError: If client is not initialized or secret not found.
        """
        if self._client is None or not self._authenticated:
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.INFISICAL,
                operation="get_secret",
                target_name="infisical-adapter",
            )
            raise SecretResolutionError(
                "Infisical adapter not initialized. Call initialize() first.",
                context=ctx,
            )

        effective_project = project_id or str(self._config.project_id)
        effective_env = environment_slug or self._config.environment_slug
        effective_path = secret_path or self._config.secret_path

        try:
            result = self._client.secrets.get_secret_by_name(  # type: ignore[attr-defined]
                secret_name=secret_name,
                project_id=effective_project,
                environment_slug=effective_env,
                secret_path=effective_path,
                expand_secret_references=True,
                view_secret_value=True,
                include_imports=True,
            )

            raw_value = self._extract_secret_value(result)
            version = getattr(result, "version", None)

            self._loads_success += 1

            return ModelInfisicalSecretResult(
                key=secret_name,
                value=SecretStr(str(raw_value)),
                version=version,
                secret_path=effective_path,
                environment=effective_env,
            )
        except Exception as e:
            self._loads_failed += 1
            sanitized_path = sanitize_secret_path(effective_path)
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.INFISICAL,
                operation="get_secret",
                target_name="infisical-adapter",
            )
            raise SecretResolutionError(
                f"Failed to retrieve secret from Infisical (path={sanitized_path})",
                context=ctx,
            ) from e

    def list_secrets(
        self,
        *,
        project_id: str | None = None,
        environment_slug: str | None = None,
        secret_path: str | None = None,
    ) -> list[ModelInfisicalSecretResult]:
        """List all secrets at the given path.

        Args:
            project_id: Override default project ID.
            environment_slug: Override default environment slug.
            secret_path: Override default secret path.

        Returns:
            List of ModelInfisicalSecretResult with values wrapped in SecretStr.

        Raises:
            SecretResolutionError: If client is not initialized.
        """
        if self._client is None or not self._authenticated:
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.INFISICAL,
                operation="list_secrets",
                target_name="infisical-adapter",
            )
            raise SecretResolutionError(
                "Infisical adapter not initialized. Call initialize() first.",
                context=ctx,
            )

        effective_project = project_id or str(self._config.project_id)
        effective_env = environment_slug or self._config.environment_slug
        effective_path = secret_path or self._config.secret_path

        try:
            result = self._client.secrets.list_secrets(  # type: ignore[attr-defined]
                project_id=effective_project,
                environment_slug=effective_env,
                secret_path=effective_path,
                expand_secret_references=True,
                view_secret_value=True,
                include_imports=True,
            )

            secrets: list[ModelInfisicalSecretResult] = []
            # The SDK returns an object with a secrets attribute (list)
            raw_secrets = getattr(result, "secrets", []) or []
            for s in raw_secrets:
                # Use explicit ``is None`` checks so that empty strings (valid
                # secret keys/values) are not silently replaced by the fallback.
                key = getattr(s, "secretKey", None)
                if key is None:
                    key = getattr(s, "secret_key", "")
                val = self._extract_secret_value(s)
                version = getattr(s, "version", None)
                secrets.append(
                    ModelInfisicalSecretResult(
                        key=str(key),
                        value=SecretStr(str(val)),
                        version=version,
                        secret_path=effective_path,
                        environment=effective_env,
                    )
                )

            self._loads_success += 1

            return secrets
        except Exception as e:
            self._loads_failed += 1
            sanitized_path = sanitize_secret_path(effective_path)
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.INFISICAL,
                operation="list_secrets",
                target_name="infisical-adapter",
            )
            raise SecretResolutionError(
                f"Failed to list secrets from Infisical (path={sanitized_path})",
                context=ctx,
            ) from e

    def get_secrets_batch(
        self,
        secret_names: list[str],
        *,
        project_id: str | None = None,
        environment_slug: str | None = None,
        secret_path: str | None = None,
    ) -> ModelInfisicalBatchResult:
        """Retrieve multiple secrets by name.

        Fetches each secret individually and collects results. Partial failures
        are captured in the errors dict without aborting the entire batch.

        Args:
            secret_names: List of secret names to retrieve.
            project_id: Override default project ID.
            environment_slug: Override default environment slug.
            secret_path: Override default secret path.

        Returns:
            ModelInfisicalBatchResult with successes and per-key errors.
        """
        batch_result = ModelInfisicalBatchResult()

        for name in secret_names:
            try:
                result = self.get_secret(
                    secret_name=name,
                    project_id=project_id,
                    environment_slug=environment_slug,
                    secret_path=secret_path,
                )
                batch_result.secrets[name] = result
            except Exception as e:
                batch_result.errors[name] = str(e)

        return batch_result

    def shutdown(self) -> None:
        """Release SDK client resources."""
        self._client = None
        self._authenticated = False
        logger.info("Infisical adapter shut down")


__all__: list[str] = [
    "AdapterInfisical",
    "ModelInfisicalBatchResult",
    "ModelInfisicalSecretResult",
    "ModelInfisicalAdapterConfig",
]
