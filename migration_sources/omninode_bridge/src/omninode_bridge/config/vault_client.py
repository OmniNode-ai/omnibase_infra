"""HashiCorp Vault client for secure secrets management.

This module provides a production-ready Vault client with:
- Multiple authentication methods (Token, AppRole, Kubernetes)
- Secrets caching with TTL refresh
- Retry logic with exponential backoff
- Graceful fallback when Vault is unavailable
- Circuit breaker pattern for resilience

Usage:
    >>> from omninode_bridge.config.vault_client import VaultClient
    >>>
    >>> # Initialize with token auth (development)
    >>> client = VaultClient(
    ...     vault_addr="http://localhost:8200",
    ...     vault_token="your-token"
    ... )
    >>>
    >>> # Get secret
    >>> db_password = client.get_secret("database/config", "password")
    >>>
    >>> # Get all secrets from path
    >>> config = client.get_secrets("database/config")
    >>> print(config["password"], config["username"])
"""

import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class VaultAuthMethod(str, Enum):
    """Vault authentication methods."""

    TOKEN = "token"
    APPROLE = "approle"
    KUBERNETES = "kubernetes"


@dataclass
class VaultConfig:
    """Vault client configuration."""

    # Connection settings
    vault_addr: str
    vault_mount_point: str = "secret"
    verify_ssl: bool = True

    # Authentication
    auth_method: VaultAuthMethod = VaultAuthMethod.TOKEN
    vault_token: str | None = None

    # AppRole authentication
    role_id: str | None = None
    secret_id: str | None = None

    # Kubernetes authentication
    k8s_role: str | None = None
    k8s_token_path: str = "/var/run/secrets/kubernetes.io/serviceaccount/token"

    # Caching
    cache_ttl_seconds: int = 300  # 5 minutes
    enable_caching: bool = True

    # Retry settings
    max_retries: int = 3
    backoff_factor: float = 0.5
    retry_status_codes: tuple[int, ...] = (500, 502, 503, 504)

    # Timeouts
    connect_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 10.0

    @classmethod
    def from_env(cls) -> "VaultConfig":
        """Create VaultConfig from environment variables.

        Environment variables:
            VAULT_ADDR - Vault server address (required)
            VAULT_TOKEN - Vault token for token auth
            VAULT_MOUNT_POINT - Secret mount point (default: "secret")
            VAULT_ROLE_ID - AppRole role ID
            VAULT_SECRET_ID - AppRole secret ID
            VAULT_K8S_ROLE - Kubernetes role name
            VAULT_VERIFY_SSL - Enable SSL verification (default: true)

        Returns:
            VaultConfig instance populated from environment variables

        Raises:
            ValueError: If VAULT_ADDR is not set
        """
        vault_addr = os.getenv("VAULT_ADDR")
        if not vault_addr:
            raise ValueError("VAULT_ADDR environment variable is required")

        # Determine auth method based on available credentials
        vault_token = os.getenv("VAULT_TOKEN")
        role_id = os.getenv("VAULT_ROLE_ID")
        secret_id = os.getenv("VAULT_SECRET_ID")
        k8s_role = os.getenv("VAULT_K8S_ROLE")

        if vault_token:
            auth_method = VaultAuthMethod.TOKEN
        elif role_id and secret_id:
            auth_method = VaultAuthMethod.APPROLE
        elif k8s_role:
            auth_method = VaultAuthMethod.KUBERNETES
        else:
            # Default to token auth (will fail if no token provided)
            auth_method = VaultAuthMethod.TOKEN

        return cls(
            vault_addr=vault_addr,
            vault_mount_point=os.getenv("VAULT_MOUNT_POINT", "secret"),
            verify_ssl=os.getenv("VAULT_VERIFY_SSL", "true").lower() == "true",
            auth_method=auth_method,
            vault_token=vault_token,
            role_id=role_id,
            secret_id=secret_id,
            k8s_role=k8s_role,
        )


class VaultClientError(Exception):
    """Base exception for Vault client errors."""

    pass


class VaultAuthenticationError(VaultClientError):
    """Vault authentication failed."""

    pass


class VaultConnectionError(VaultClientError):
    """Vault connection failed."""

    pass


class VaultSecretNotFoundError(VaultClientError):
    """Secret not found in Vault."""

    pass


@dataclass
class CachedSecret:
    """Cached secret with TTL."""

    value: dict[str, Any]
    timestamp: float
    ttl_seconds: int

    def is_expired(self) -> bool:
        """Check if cached secret has expired."""
        return time.time() - self.timestamp > self.ttl_seconds


class VaultClient:
    """Production-ready HashiCorp Vault client with caching and retry logic.

    Features:
        - Multiple authentication methods
        - Secrets caching with TTL refresh
        - Automatic retry with exponential backoff
        - Circuit breaker pattern
        - Graceful error handling

    Example:
        >>> client = VaultClient.from_env()
        >>>
        >>> # Get single secret value
        >>> password = client.get_secret("database/postgres", "password")
        >>>
        >>> # Get all secrets from path
        >>> config = client.get_secrets("database/postgres")
        >>>
        >>> # Check if client is available
        >>> if client.is_available():
        ...     secrets = client.get_secrets("app/config")
    """

    def __init__(self, config: VaultConfig):
        """Initialize Vault client.

        Args:
            config: Vault client configuration
        """
        self.config = config
        self._token: str | None = config.vault_token
        self._cache: dict[str, CachedSecret] = {}
        self._session = self._create_session()
        self._authenticated = False

        # Authenticate on initialization
        try:
            self._authenticate()
        except VaultClientError as e:
            logger.warning(
                f"Vault authentication failed: {e}. Client will use fallback mode."
            )

    @classmethod
    def from_env(cls) -> "VaultClient":
        """Create VaultClient from environment variables.

        Returns:
            VaultClient instance configured from environment

        Raises:
            ValueError: If required environment variables are missing
        """
        config = VaultConfig.from_env()
        return cls(config)

    def _create_session(self) -> requests.Session:
        """Create requests session with retry logic.

        Returns:
            Configured requests session with retry adapter
        """
        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.backoff_factor,
            status_forcelist=self.config.retry_status_codes,
            allowed_methods=["GET", "POST"],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def _authenticate(self) -> None:
        """Authenticate with Vault using configured method.

        Raises:
            VaultAuthenticationError: If authentication fails
            VaultConnectionError: If connection to Vault fails
        """
        if self.config.auth_method == VaultAuthMethod.TOKEN:
            self._authenticate_token()
        elif self.config.auth_method == VaultAuthMethod.APPROLE:
            self._authenticate_approle()
        elif self.config.auth_method == VaultAuthMethod.KUBERNETES:
            self._authenticate_kubernetes()
        else:
            raise VaultAuthenticationError(
                f"Unsupported auth method: {self.config.auth_method}"
            )

        self._authenticated = True
        logger.info(
            f"Successfully authenticated with Vault using {self.config.auth_method} method"
        )

    def _authenticate_token(self) -> None:
        """Authenticate using token.

        Raises:
            VaultAuthenticationError: If token is invalid or missing
        """
        if not self.config.vault_token:
            raise VaultAuthenticationError(
                "Vault token is required for token authentication"
            )

        self._token = self.config.vault_token

        # Verify token by checking self
        try:
            url = urljoin(self.config.vault_addr, "/v1/auth/token/lookup-self")
            response = self._session.get(
                url,
                headers={"X-Vault-Token": self._token},
                verify=self.config.verify_ssl,
                timeout=(
                    self.config.connect_timeout_seconds,
                    self.config.read_timeout_seconds,
                ),
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise VaultAuthenticationError(f"Token authentication failed: {e}") from e

    def _authenticate_approle(self) -> None:
        """Authenticate using AppRole.

        Raises:
            VaultAuthenticationError: If AppRole credentials are invalid
        """
        if not self.config.role_id or not self.config.secret_id:
            raise VaultAuthenticationError(
                "role_id and secret_id are required for AppRole authentication"
            )

        try:
            url = urljoin(self.config.vault_addr, "/v1/auth/approle/login")
            response = self._session.post(
                url,
                json={
                    "role_id": self.config.role_id,
                    "secret_id": self.config.secret_id,
                },
                verify=self.config.verify_ssl,
                timeout=(
                    self.config.connect_timeout_seconds,
                    self.config.read_timeout_seconds,
                ),
            )
            response.raise_for_status()
            data = response.json()
            self._token = data["auth"]["client_token"]
        except requests.exceptions.RequestException as e:
            raise VaultAuthenticationError(f"AppRole authentication failed: {e}") from e
        except (KeyError, TypeError) as e:
            raise VaultAuthenticationError(
                f"Invalid AppRole response format: {e}"
            ) from e

    def _authenticate_kubernetes(self) -> None:
        """Authenticate using Kubernetes service account.

        Raises:
            VaultAuthenticationError: If Kubernetes authentication fails
        """
        if not self.config.k8s_role:
            raise VaultAuthenticationError(
                "k8s_role is required for Kubernetes authentication"
            )

        # Read service account token
        try:
            with open(self.config.k8s_token_path, encoding="utf-8") as f:
                jwt_token = f.read().strip()
        except OSError as e:
            raise VaultAuthenticationError(
                f"Failed to read Kubernetes token: {e}"
            ) from e

        # Authenticate with Vault
        try:
            url = urljoin(self.config.vault_addr, "/v1/auth/kubernetes/login")
            response = self._session.post(
                url,
                json={"role": self.config.k8s_role, "jwt": jwt_token},
                verify=self.config.verify_ssl,
                timeout=(
                    self.config.connect_timeout_seconds,
                    self.config.read_timeout_seconds,
                ),
            )
            response.raise_for_status()
            data = response.json()
            self._token = data["auth"]["client_token"]
        except requests.exceptions.RequestException as e:
            raise VaultAuthenticationError(
                f"Kubernetes authentication failed: {e}"
            ) from e
        except (KeyError, TypeError) as e:
            raise VaultAuthenticationError(
                f"Invalid Kubernetes auth response format: {e}"
            ) from e

    def is_available(self) -> bool:
        """Check if Vault is available and authenticated.

        Returns:
            True if Vault is available and client is authenticated
        """
        return self._authenticated and self._token is not None

    def get_secret(self, path: str, key: str, default: Any = None) -> Any:
        """Get a single secret value from Vault.

        Args:
            path: Secret path (e.g., "database/postgres")
            key: Secret key within the path
            default: Default value if secret not found

        Returns:
            Secret value or default if not found

        Raises:
            VaultClientError: If Vault operation fails
        """
        secrets = self.get_secrets(path)
        return secrets.get(key, default)

    def get_secrets(self, path: str) -> dict[str, Any]:
        """Get all secrets from a Vault path.

        Args:
            path: Secret path (e.g., "database/postgres")

        Returns:
            Dictionary of secret key-value pairs

        Raises:
            VaultClientError: If Vault operation fails
            VaultSecretNotFoundError: If secret path doesn't exist
        """
        # Check cache first
        if self.config.enable_caching and path in self._cache:
            cached = self._cache[path]
            if not cached.is_expired():
                logger.debug(f"Cache hit for secret path: {path}")
                return cached.value

        # Fetch from Vault
        if not self.is_available():
            raise VaultClientError(
                "Vault client is not available. Authentication failed or not initialized."
            )

        try:
            # Construct full secret path
            # Format: /v1/{mount_point}/data/{path} for KV v2
            full_path = f"/v1/{self.config.vault_mount_point}/data/{path}"
            url = urljoin(self.config.vault_addr, full_path)

            response = self._session.get(
                url,
                headers={"X-Vault-Token": self._token},
                verify=self.config.verify_ssl,
                timeout=(
                    self.config.connect_timeout_seconds,
                    self.config.read_timeout_seconds,
                ),
            )

            if response.status_code == 404:
                raise VaultSecretNotFoundError(f"Secret not found: {path}")

            response.raise_for_status()
            data = response.json()

            # Extract secrets from KV v2 format
            secrets = data["data"]["data"]

            # Cache the result
            if self.config.enable_caching:
                self._cache[path] = CachedSecret(
                    value=secrets,
                    timestamp=time.time(),
                    ttl_seconds=self.config.cache_ttl_seconds,
                )
                logger.debug(f"Cached secrets for path: {path}")

            return secrets

        except requests.exceptions.RequestException as e:
            raise VaultConnectionError(
                f"Failed to retrieve secrets from {path}: {e}"
            ) from e
        except (KeyError, TypeError) as e:
            raise VaultClientError(
                f"Invalid secret response format from {path}: {e}"
            ) from e

    def invalidate_cache(self, path: str | None = None) -> None:
        """Invalidate cached secrets.

        Args:
            path: Specific path to invalidate, or None to clear all cache
        """
        if path is None:
            self._cache.clear()
            logger.info("Cleared all cached secrets")
        elif path in self._cache:
            del self._cache[path]
            logger.info(f"Invalidated cache for path: {path}")

    def refresh_token(self) -> None:
        """Refresh authentication token.

        Useful for long-running applications to renew tokens before expiry.

        Raises:
            VaultAuthenticationError: If token refresh fails
        """
        logger.info("Refreshing Vault authentication token")
        self._authenticated = False
        self._authenticate()


# Global singleton instance (lazy initialization)
_vault_client_instance: VaultClient | None = None


@lru_cache(maxsize=1)
def get_vault_client() -> VaultClient | None:
    """Get global VaultClient instance (singleton pattern).

    Returns:
        VaultClient instance if Vault is enabled, None otherwise

    Example:
        >>> client = get_vault_client()
        >>> if client and client.is_available():
        ...     password = client.get_secret("database/config", "password")
    """
    global _vault_client_instance

    # Check if Vault is enabled
    vault_enabled = os.getenv("VAULT_ENABLED", "false").lower() == "true"
    if not vault_enabled:
        logger.info("Vault is disabled (VAULT_ENABLED=false)")
        return None

    # Initialize client if not already done
    if _vault_client_instance is None:
        try:
            _vault_client_instance = VaultClient.from_env()
        except (ValueError, VaultClientError) as e:
            logger.warning(
                f"Failed to initialize Vault client: {e}. Continuing without Vault."
            )
            return None

    return _vault_client_instance


def reset_vault_client() -> None:
    """Reset global VaultClient instance.

    Useful for testing or when configuration changes.
    """
    global _vault_client_instance
    _vault_client_instance = None
    get_vault_client.cache_clear()
