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

.. versionadded:: 0.9.0
    Initial implementation for OMN-2286.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, SecretStr

logger = logging.getLogger(__name__)


class ModelInfisicalAdapterConfig(BaseModel):
    """Configuration for the Infisical adapter.

    Attributes:
        host: Infisical server URL.
        client_id: Machine identity client ID for Universal Auth.
        client_secret: Machine identity client secret for Universal Auth.
        project_id: Default Infisical project ID.
        environment_slug: Default environment slug (e.g., ``dev``, ``staging``, ``prod``).
        secret_path: Default secret path prefix (e.g., ``/``).
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    host: str = Field(
        default="https://app.infisical.com",
        description="Infisical server URL.",
    )
    client_id: SecretStr = Field(
        ...,
        description="Machine identity client ID for Universal Auth.",
    )
    client_secret: SecretStr = Field(
        ...,
        description="Machine identity client secret for Universal Auth.",
    )
    project_id: UUID = Field(
        ...,
        description="Default Infisical project ID.",
    )
    environment_slug: str = Field(
        default="prod",
        min_length=1,
        description="Default environment slug.",
    )
    secret_path: str = Field(
        default="/",
        description="Default secret path prefix.",
    )


@dataclass(frozen=True)
class InfisicalSecretResult:
    """Result of a single secret fetch from Infisical.

    Attributes:
        key: The secret name / key.
        value: The secret value wrapped in ``SecretStr``.
        version: The secret version (if available).
        secret_path: The path where the secret was found.
        environment: The environment slug.
    """

    key: str
    value: SecretStr
    version: int | None = None
    secret_path: str = "/"
    environment: str = ""


@dataclass
class InfisicalBatchResult:
    """Result of a batch secret fetch.

    Attributes:
        secrets: Mapping of secret name to result.
        errors: Mapping of secret name to error message for failed fetches.
    """

    secrets: dict[str, InfisicalSecretResult] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


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

    @property
    def is_authenticated(self) -> bool:
        """Whether the adapter has successfully authenticated."""
        return self._authenticated

    def initialize(self) -> None:
        """Initialize the Infisical SDK client and authenticate.

        Uses Universal Auth with machine identity credentials.

        Raises:
            RuntimeError: If SDK initialization or authentication fails.
        """
        try:
            from infisical_sdk import InfisicalSDKClient  # type: ignore[import-untyped]
        except ImportError as e:
            raise RuntimeError(
                "infisical-sdk package is not installed. "
                "Install with: pip install 'infisicalsdk>=1.0.15,<2.0.0'"
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
                extra={"host": self._config.host},
            )
        except Exception as e:
            self._authenticated = False
            # Do NOT log credentials - only the host
            raise RuntimeError(
                f"Failed to initialize Infisical client for host {self._config.host}"
            ) from e

    def get_secret(
        self,
        secret_name: str,
        *,
        project_id: str | None = None,
        environment_slug: str | None = None,
        secret_path: str | None = None,
    ) -> InfisicalSecretResult:
        """Retrieve a single secret by name.

        Args:
            secret_name: The secret key/name to retrieve.
            project_id: Override default project ID.
            environment_slug: Override default environment slug.
            secret_path: Override default secret path.

        Returns:
            InfisicalSecretResult with the secret value wrapped in SecretStr.

        Raises:
            RuntimeError: If client is not initialized or secret not found.
        """
        if self._client is None or not self._authenticated:
            raise RuntimeError(
                "Infisical adapter not initialized. Call initialize() first."
            )

        effective_project = project_id or str(self._config.project_id)
        effective_env = environment_slug or self._config.environment_slug
        effective_path = secret_path or self._config.secret_path

        try:
            result = self._client.secrets.get_secret_by_name(  # type: ignore[union-attr]
                secret_name=secret_name,
                project_id=effective_project,
                environment_slug=effective_env,
                secret_path=effective_path,
                expand_secret_references=True,
                view_secret_value=True,
                include_imports=True,
            )

            # Extract value - the SDK returns an object with secretValue attribute
            raw_value = getattr(result, "secretValue", None) or getattr(
                result, "secret_value", ""
            )
            version = getattr(result, "version", None)

            return InfisicalSecretResult(
                key=secret_name,
                value=SecretStr(str(raw_value)),
                version=version,
                secret_path=effective_path,
                environment=effective_env,
            )
        except Exception as e:
            # SECURITY: Do not log secret_name in production (reveals structure)
            raise RuntimeError(
                f"Failed to retrieve secret from Infisical (path={effective_path})"
            ) from e

    def list_secrets(
        self,
        *,
        project_id: str | None = None,
        environment_slug: str | None = None,
        secret_path: str | None = None,
    ) -> list[InfisicalSecretResult]:
        """List all secrets at the given path.

        Args:
            project_id: Override default project ID.
            environment_slug: Override default environment slug.
            secret_path: Override default secret path.

        Returns:
            List of InfisicalSecretResult with values wrapped in SecretStr.

        Raises:
            RuntimeError: If client is not initialized.
        """
        if self._client is None or not self._authenticated:
            raise RuntimeError(
                "Infisical adapter not initialized. Call initialize() first."
            )

        effective_project = project_id or str(self._config.project_id)
        effective_env = environment_slug or self._config.environment_slug
        effective_path = secret_path or self._config.secret_path

        try:
            result = self._client.secrets.list_secrets(  # type: ignore[union-attr]
                project_id=effective_project,
                environment_slug=effective_env,
                secret_path=effective_path,
                expand_secret_references=True,
                view_secret_value=True,
                include_imports=True,
            )

            secrets: list[InfisicalSecretResult] = []
            # The SDK returns an object with a secrets attribute (list)
            raw_secrets = getattr(result, "secrets", []) or []
            for s in raw_secrets:
                key = getattr(s, "secretKey", None) or getattr(s, "secret_key", "")
                val = getattr(s, "secretValue", None) or getattr(s, "secret_value", "")
                version = getattr(s, "version", None)
                secrets.append(
                    InfisicalSecretResult(
                        key=str(key),
                        value=SecretStr(str(val)),
                        version=version,
                        secret_path=effective_path,
                        environment=effective_env,
                    )
                )
            return secrets
        except Exception as e:
            raise RuntimeError(
                f"Failed to list secrets from Infisical (path={effective_path})"
            ) from e

    def get_secrets_batch(
        self,
        secret_names: list[str],
        *,
        project_id: str | None = None,
        environment_slug: str | None = None,
        secret_path: str | None = None,
    ) -> InfisicalBatchResult:
        """Retrieve multiple secrets by name.

        Fetches each secret individually and collects results. Partial failures
        are captured in the errors dict without aborting the entire batch.

        Args:
            secret_names: List of secret names to retrieve.
            project_id: Override default project ID.
            environment_slug: Override default environment slug.
            secret_path: Override default secret path.

        Returns:
            InfisicalBatchResult with successes and per-key errors.
        """
        batch_result = InfisicalBatchResult()

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
    "InfisicalBatchResult",
    "InfisicalSecretResult",
    "ModelInfisicalAdapterConfig",
]
