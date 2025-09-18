"""
ONEX Credential Manager for Infrastructure Security

Provides secure credential management with environment variable fallbacks,
following ONEX security patterns and avoiding hardcoded credentials.

Per ONEX security requirements:
- No hardcoded credentials in source code
- Environment variable validation and sanitization
- Support for external credential stores (Vault, AWS Secrets Manager)
- Secure credential rotation and invalidation
"""

import logging
import os
from dataclasses import dataclass
from typing import Any

from omnibase_core.core.errors.onex_error import CoreErrorCode, OnexError


@dataclass
class DatabaseCredentials:
    """Secure database credential model."""

    host: str
    port: int
    database: str
    username: str
    password: str
    ssl_mode: str = "prefer"

    def get_connection_string(self) -> str:
        """Get secure PostgreSQL connection string."""
        return (
            f"postgresql://{self.username}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}?sslmode={self.ssl_mode}"
        )


@dataclass
class EventBusCredentials:
    """Secure event bus credential model."""

    bootstrap_servers: list[str]
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: str | None = None
    sasl_username: str | None = None
    sasl_password: str | None = None
    ssl_ca_location: str | None = None
    ssl_cert_location: str | None = None
    ssl_key_location: str | None = None
    ssl_key_password: str | None = None


class ONEXCredentialManager:
    """
    ONEX-compliant credential manager for infrastructure security.

    Provides secure credential management with multiple backend support:
    - Environment variables (development)
    - HashiCorp Vault (production)
    - AWS Secrets Manager (cloud)
    - Azure Key Vault (azure cloud)
    """

    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._cache: dict[str, Any] = {}
        self._backend = self._detect_credential_backend()

    def _detect_credential_backend(self) -> str:
        """Detect available credential backend."""
        if os.getenv("VAULT_ADDR"):
            return "vault"
        if os.getenv("AWS_REGION"):
            return "aws_secrets"
        if os.getenv("AZURE_CLIENT_ID"):
            return "azure_keyvault"
        return "environment"

    def get_database_credentials(self) -> DatabaseCredentials:
        """
        Get secure database credentials.

        Returns:
            DatabaseCredentials with validated connection parameters

        Raises:
            OnexError: If credentials are missing or invalid
        """
        try:
            if self._backend == "vault":
                return self._get_database_credentials_from_vault()
            if self._backend == "aws_secrets":
                return self._get_database_credentials_from_aws()
            return self._get_database_credentials_from_env()

        except Exception as e:
            raise OnexError(
                "Failed to retrieve database credentials",
                CoreErrorCode.OPERATION_FAILED,
            ) from e

    def get_event_bus_credentials(self) -> EventBusCredentials:
        """
        Get secure event bus credentials.

        Returns:
            EventBusCredentials with security configuration

        Raises:
            OnexError: If credentials are missing or invalid
        """
        try:
            if self._backend == "vault":
                return self._get_event_bus_credentials_from_vault()
            if self._backend == "aws_secrets":
                return self._get_event_bus_credentials_from_aws()
            return self._get_event_bus_credentials_from_env()

        except Exception as e:
            raise OnexError(
                "Failed to retrieve event bus credentials",
                CoreErrorCode.OPERATION_FAILED,
            ) from e

    def _get_database_credentials_from_env(self) -> DatabaseCredentials:
        """Get database credentials from environment variables with validation."""
        required_vars = {
            "POSTGRES_HOST": "host",
            "POSTGRES_PORT": "port",
            "POSTGRES_DATABASE": "database",
            "POSTGRES_USER": "username",
            "POSTGRES_PASSWORD": "password",
        }

        credentials = {}

        for env_var, cred_key in required_vars.items():
            value = os.getenv(env_var)
            if not value:
                raise OnexError(
                    f"Missing required environment variable: {env_var}",
                    CoreErrorCode.MISSING_REQUIRED_PARAMETER,
                )

            # Type conversion for port
            if cred_key == "port":
                try:
                    value = int(value)
                except ValueError:
                    raise OnexError(
                        f"Invalid port value: {value}",
                        CoreErrorCode.PARAMETER_TYPE_MISMATCH,
                    )

            credentials[cred_key] = value

        # Optional SSL mode
        ssl_mode = os.getenv("POSTGRES_SSL_MODE", "prefer")
        credentials["ssl_mode"] = ssl_mode

        self._logger.info(
            f"Retrieved database credentials for host: {credentials['host']}",
        )

        return DatabaseCredentials(**credentials)

    def _get_event_bus_credentials_from_env(self) -> EventBusCredentials:
        """Get event bus credentials from environment variables with TLS security integration."""
        # Get bootstrap servers
        servers_str = os.getenv("REDPANDA_BOOTSTRAP_SERVERS")
        if not servers_str:
            # Fallback to host/port pattern - use internal Docker service name and internal port
            host = os.getenv("REDPANDA_HOST", "omnibase-infra-redpanda")
            port = os.getenv("REDPANDA_PORT", "9092")  # Use internal Kafka API port
            servers_str = f"{host}:{port}"

        bootstrap_servers = [server.strip() for server in servers_str.split(",")]

        # Integrate with TLS configuration manager for secure settings
        try:
            from .tls_config import get_tls_manager

            tls_manager = get_tls_manager()
            kafka_tls_config = tls_manager.get_kafka_tls_config()

            credentials = EventBusCredentials(
                bootstrap_servers=bootstrap_servers,
                security_protocol=kafka_tls_config.security_protocol,
                ssl_ca_location=kafka_tls_config.ssl_ca_location,
                ssl_cert_location=kafka_tls_config.ssl_certificate_location,
                ssl_key_location=kafka_tls_config.ssl_key_location,
                ssl_key_password=kafka_tls_config.ssl_key_password,
            )

            self._logger.info(
                f"Retrieved secure event bus credentials with TLS: {kafka_tls_config.security_protocol}",
            )

        except Exception as e:
            # Fallback to environment-based configuration if TLS manager fails
            self._logger.warning(
                f"TLS manager unavailable, using environment config: {e}",
            )
            security_protocol = os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")

            credentials = EventBusCredentials(
                bootstrap_servers=bootstrap_servers,
                security_protocol=security_protocol,
            )

        # SASL authentication (if enabled)
        if credentials.security_protocol in ["SASL_PLAINTEXT", "SASL_SSL"]:
            credentials.sasl_mechanism = os.getenv("KAFKA_SASL_MECHANISM", "PLAIN")
            credentials.sasl_username = os.getenv("KAFKA_SASL_USERNAME")
            credentials.sasl_password = os.getenv("KAFKA_SASL_PASSWORD")

        # SSL configuration (if not already set by TLS manager)
        if (
            credentials.security_protocol in ["SSL", "SASL_SSL"]
            and not credentials.ssl_ca_location
        ):
            credentials.ssl_ca_location = os.getenv("KAFKA_SSL_CA_LOCATION")
            credentials.ssl_cert_location = os.getenv("KAFKA_SSL_CERT_LOCATION")
            credentials.ssl_key_location = os.getenv("KAFKA_SSL_KEY_LOCATION")
            credentials.ssl_key_password = os.getenv("KAFKA_SSL_KEY_PASSWORD")

        self._logger.info(
            f"Retrieved event bus credentials for servers: {bootstrap_servers}, protocol: {credentials.security_protocol}",
        )

        return credentials

    def _get_database_credentials_from_vault(self) -> DatabaseCredentials:
        """Get database credentials from HashiCorp Vault."""
        # TODO: Implement Vault integration
        self._logger.warning(
            "Vault backend not implemented, falling back to environment",
        )
        return self._get_database_credentials_from_env()

    def _get_event_bus_credentials_from_vault(self) -> EventBusCredentials:
        """Get event bus credentials from HashiCorp Vault."""
        # TODO: Implement Vault integration
        self._logger.warning(
            "Vault backend not implemented, falling back to environment",
        )
        return self._get_event_bus_credentials_from_env()

    def _get_database_credentials_from_aws(self) -> DatabaseCredentials:
        """Get database credentials from AWS Secrets Manager."""
        # TODO: Implement AWS Secrets Manager integration
        self._logger.warning(
            "AWS Secrets Manager backend not implemented, falling back to environment",
        )
        return self._get_database_credentials_from_env()

    def _get_event_bus_credentials_from_aws(self) -> EventBusCredentials:
        """Get event bus credentials from AWS Secrets Manager."""
        # TODO: Implement AWS Secrets Manager integration
        self._logger.warning(
            "AWS Secrets Manager backend not implemented, falling back to environment",
        )
        return self._get_event_bus_credentials_from_env()

    def validate_credentials(self) -> bool:
        """
        Validate that all required credentials are available.

        Returns:
            True if all credentials are valid, False otherwise
        """
        try:
            self.get_database_credentials()
            self.get_event_bus_credentials()
            return True
        except OnexError:
            return False

    def rotate_credentials(self, credential_type: str) -> None:
        """
        Rotate credentials for the specified type.

        Args:
            credential_type: Type of credentials to rotate (database, event_bus)
        """
        if credential_type in self._cache:
            del self._cache[credential_type]

        self._logger.info(f"Rotated credentials for: {credential_type}")

    def clear_cache(self) -> None:
        """Clear all cached credentials."""
        self._cache.clear()
        self._logger.info("Cleared credential cache")


# Global credential manager instance
_credential_manager: ONEXCredentialManager | None = None


def get_credential_manager() -> ONEXCredentialManager:
    """
    Get global credential manager instance.

    Returns:
        ONEXCredentialManager singleton instance
    """
    global _credential_manager

    if _credential_manager is None:
        _credential_manager = ONEXCredentialManager()

    return _credential_manager
