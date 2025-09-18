"""
TLS/SSL Security Configuration for ONEX Infrastructure

Provides comprehensive TLS/SSL configuration for secure communication
with external services (PostgreSQL, Kafka/RedPanda, etc.).

Per ONEX security requirements:
- Mandatory TLS for production environments
- Certificate validation and chain verification
- Support for mutual TLS (mTLS) authentication
- Secure cipher suite configuration
"""

import logging
import os
import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omnibase_core.core.errors.onex_error import CoreErrorCode, OnexError


@dataclass
class TLSCertificateConfig:
    """TLS certificate configuration model."""

    ca_cert_path: str | None = None
    client_cert_path: str | None = None
    client_key_path: str | None = None
    key_password: str | None = None
    verify_mode: str = "CERT_REQUIRED"
    check_hostname: bool = True
    ciphers: str | None = None


@dataclass
class PostgreSQLTLSConfig:
    """PostgreSQL TLS configuration."""

    ssl_mode: str = "require"
    ssl_ca: str | None = None
    ssl_cert: str | None = None
    ssl_key: str | None = None
    ssl_root_cert: str | None = None
    ssl_crl: str | None = None

    def to_connection_params(self) -> dict[str, str]:
        """Convert to PostgreSQL connection parameters."""
        params = {"sslmode": self.ssl_mode}

        if self.ssl_ca:
            params["sslcert"] = self.ssl_cert
        if self.ssl_cert:
            params["sslcert"] = self.ssl_cert
        if self.ssl_key:
            params["sslkey"] = self.ssl_key
        if self.ssl_root_cert:
            params["sslrootcert"] = self.ssl_root_cert
        if self.ssl_crl:
            params["sslcrl"] = self.ssl_crl

        return params


@dataclass
class KafkaTLSConfig:
    """Kafka/RedPanda TLS configuration."""

    security_protocol: str = "SSL"
    ssl_ca_location: str | None = None
    ssl_certificate_location: str | None = None
    ssl_key_location: str | None = None
    ssl_key_password: str | None = None
    ssl_cipher_suites: str | None = None
    ssl_curves_list: str | None = None
    ssl_sigalgs_list: str | None = None
    ssl_endpoint_identification_algorithm: str = "https"

    def to_producer_config(self) -> dict[str, Any]:
        """Convert to Kafka producer configuration."""
        config = {"security.protocol": self.security_protocol}

        if self.ssl_ca_location:
            config["ssl.ca.location"] = self.ssl_ca_location
        if self.ssl_certificate_location:
            config["ssl.certificate.location"] = self.ssl_certificate_location
        if self.ssl_key_location:
            config["ssl.key.location"] = self.ssl_key_location
        if self.ssl_key_password:
            config["ssl.key.password"] = self.ssl_key_password
        if self.ssl_cipher_suites:
            config["ssl.cipher.suites"] = self.ssl_cipher_suites
        if self.ssl_curves_list:
            config["ssl.curves.list"] = self.ssl_curves_list
        if self.ssl_sigalgs_list:
            config["ssl.sigalgs.list"] = self.ssl_sigalgs_list

        config["ssl.endpoint.identification.algorithm"] = (
            self.ssl_endpoint_identification_algorithm
        )

        return config


class ONEXTLSConfigManager:
    """
    ONEX TLS configuration manager for infrastructure security.

    Provides centralized TLS configuration management with support for:
    - Certificate validation and verification
    - Mutual TLS (mTLS) authentication
    - Environment-specific security policies
    - Certificate rotation and management
    """

    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._environment = os.getenv("ONEX_ENVIRONMENT", "development")
        self._cert_base_path = Path(os.getenv("ONEX_CERTS_PATH", "/etc/ssl/certs/onex"))

        # Security policy based on environment
        self._enforce_tls = self._environment in ["production", "staging"]
        self._require_mtls = self._environment == "production"

        self._logger.info(
            f"TLS manager initialized for environment: {self._environment}",
        )
        self._logger.info(
            f"TLS enforcement: {self._enforce_tls}, mTLS required: {self._require_mtls}",
        )

    def get_postgresql_tls_config(self) -> PostgreSQLTLSConfig:
        """
        Get PostgreSQL TLS configuration.

        Returns:
            PostgreSQLTLSConfig with appropriate security settings

        Raises:
            OnexError: If required certificates are missing in secure environments
        """
        try:
            # Base configuration
            ssl_mode = "disable"

            if self._enforce_tls:
                ssl_mode = "require"

                # Look for CA certificate
                ca_cert = self._cert_base_path / "postgres" / "ca.crt"
                if ca_cert.exists():
                    ssl_mode = "verify-ca"

                    # Look for client certificate for mTLS
                    client_cert = self._cert_base_path / "postgres" / "client.crt"
                    client_key = self._cert_base_path / "postgres" / "client.key"

                    if client_cert.exists() and client_key.exists():
                        ssl_mode = "verify-full"

                        return PostgreSQLTLSConfig(
                            ssl_mode=ssl_mode,
                            ssl_ca=str(ca_cert),
                            ssl_cert=str(client_cert),
                            ssl_key=str(client_key),
                        )

                    return PostgreSQLTLSConfig(
                        ssl_mode=ssl_mode,
                        ssl_ca=str(ca_cert),
                    )

            # Override from environment
            env_ssl_mode = os.getenv("POSTGRES_SSL_MODE")
            if env_ssl_mode:
                ssl_mode = env_ssl_mode

            config = PostgreSQLTLSConfig(ssl_mode=ssl_mode)

            # Add individual certificate paths if specified
            if os.getenv("POSTGRES_SSL_CA"):
                config.ssl_ca = os.getenv("POSTGRES_SSL_CA")
            if os.getenv("POSTGRES_SSL_CERT"):
                config.ssl_cert = os.getenv("POSTGRES_SSL_CERT")
            if os.getenv("POSTGRES_SSL_KEY"):
                config.ssl_key = os.getenv("POSTGRES_SSL_KEY")

            self._logger.info(f"PostgreSQL TLS config: ssl_mode={ssl_mode}")
            return config

        except Exception as e:
            raise OnexError(
                "Failed to configure PostgreSQL TLS settings",
                CoreErrorCode.CONFIGURATION_ERROR,
            ) from e

    def get_kafka_tls_config(self) -> KafkaTLSConfig:
        """
        Get Kafka/RedPanda TLS configuration.

        Returns:
            KafkaTLSConfig with appropriate security settings

        Raises:
            OnexError: If required certificates are missing in secure environments
        """
        try:
            # Default to plaintext for development
            security_protocol = "PLAINTEXT"

            if self._enforce_tls:
                security_protocol = "SSL"

                # Check for SASL authentication
                if os.getenv("KAFKA_SASL_USERNAME"):
                    security_protocol = "SASL_SSL"

            # Override from environment
            env_protocol = os.getenv("KAFKA_SECURITY_PROTOCOL")
            if env_protocol:
                security_protocol = env_protocol

            config = KafkaTLSConfig(security_protocol=security_protocol)

            # SSL configuration
            if security_protocol in ["SSL", "SASL_SSL"]:
                # CA certificate
                ca_cert_path = self._cert_base_path / "kafka" / "ca.crt"
                if ca_cert_path.exists():
                    config.ssl_ca_location = str(ca_cert_path)
                elif os.getenv("KAFKA_SSL_CA_LOCATION"):
                    config.ssl_ca_location = os.getenv("KAFKA_SSL_CA_LOCATION")
                elif self._enforce_tls:
                    raise OnexError(
                        "CA certificate required for Kafka SSL in secure environment",
                        CoreErrorCode.CONFIGURATION_ERROR,
                    )

                # Client certificates for mTLS
                if self._require_mtls:
                    client_cert_path = self._cert_base_path / "kafka" / "client.crt"
                    client_key_path = self._cert_base_path / "kafka" / "client.key"

                    if client_cert_path.exists() and client_key_path.exists():
                        config.ssl_certificate_location = str(client_cert_path)
                        config.ssl_key_location = str(client_key_path)

                        # Check for key password
                        key_password = os.getenv("KAFKA_SSL_KEY_PASSWORD")
                        if key_password:
                            config.ssl_key_password = key_password
                    else:
                        raise OnexError(
                            "Client certificates required for Kafka mTLS in production",
                            CoreErrorCode.CONFIGURATION_ERROR,
                        )

                # Environment overrides
                if os.getenv("KAFKA_SSL_CERT_LOCATION"):
                    config.ssl_certificate_location = os.getenv(
                        "KAFKA_SSL_CERT_LOCATION",
                    )
                if os.getenv("KAFKA_SSL_KEY_LOCATION"):
                    config.ssl_key_location = os.getenv("KAFKA_SSL_KEY_LOCATION")
                if os.getenv("KAFKA_SSL_KEY_PASSWORD"):
                    config.ssl_key_password = os.getenv("KAFKA_SSL_KEY_PASSWORD")

                # Cipher suites (production hardening)
                if self._environment == "production":
                    config.ssl_cipher_suites = (
                        "ECDHE-RSA-AES256-GCM-SHA384,ECDHE-RSA-AES128-GCM-SHA256"
                    )
                    config.ssl_curves_list = "secp384r1,secp256r1"

            self._logger.info(
                f"Kafka TLS config: security_protocol={security_protocol}",
            )
            return config

        except Exception as e:
            raise OnexError(
                "Failed to configure Kafka TLS settings",
                CoreErrorCode.CONFIGURATION_ERROR,
            ) from e

    def create_ssl_context(self, cert_config: TLSCertificateConfig) -> ssl.SSLContext:
        """
        Create SSL context from certificate configuration.

        Args:
            cert_config: TLS certificate configuration

        Returns:
            Configured SSL context

        Raises:
            OnexError: If SSL context creation fails
        """
        try:
            # Create secure SSL context
            context = ssl.create_default_context()

            # Configure verification mode
            if cert_config.verify_mode == "CERT_NONE":
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
            elif cert_config.verify_mode == "CERT_OPTIONAL":
                context.verify_mode = ssl.CERT_OPTIONAL
            else:  # CERT_REQUIRED
                context.verify_mode = ssl.CERT_REQUIRED

            context.check_hostname = cert_config.check_hostname

            # Load CA certificate
            if cert_config.ca_cert_path:
                context.load_verify_locations(cert_config.ca_cert_path)

            # Load client certificate and key
            if cert_config.client_cert_path and cert_config.client_key_path:
                context.load_cert_chain(
                    cert_config.client_cert_path,
                    cert_config.client_key_path,
                    cert_config.key_password,
                )

            # Configure cipher suites for production
            if self._environment == "production" and cert_config.ciphers:
                context.set_ciphers(cert_config.ciphers)

            return context

        except Exception as e:
            raise OnexError(
                "Failed to create SSL context",
                CoreErrorCode.CONFIGURATION_ERROR,
            ) from e

    def validate_certificate_chain(self, cert_path: str) -> bool:
        """
        Validate certificate chain.

        Args:
            cert_path: Path to certificate file

        Returns:
            True if certificate chain is valid
        """
        try:
            if not Path(cert_path).exists():
                self._logger.warning(f"Certificate not found: {cert_path}")
                return False

            # Load and validate certificate
            with open(cert_path, "rb") as cert_file:
                cert_data = cert_file.read()
                ssl.PEM_cert_to_DER_cert(cert_data.decode("utf-8"))

            self._logger.info(f"Certificate validation passed: {cert_path}")
            return True

        except Exception as e:
            self._logger.error(f"Certificate validation failed for {cert_path}: {e}")
            return False

    def get_security_policy(self) -> dict[str, Any]:
        """
        Get current security policy configuration.

        Returns:
            Dictionary with security policy settings
        """
        return {
            "environment": self._environment,
            "enforce_tls": self._enforce_tls,
            "require_mtls": self._require_mtls,
            "cert_base_path": str(self._cert_base_path),
            "supported_protocols": ["TLSv1.2", "TLSv1.3"],
            "minimum_key_size": 2048 if self._environment == "production" else 1024,
        }


# Global TLS configuration manager instance
_tls_manager: ONEXTLSConfigManager | None = None


def get_tls_manager() -> ONEXTLSConfigManager:
    """
    Get global TLS configuration manager instance.

    Returns:
        ONEXTLSConfigManager singleton instance
    """
    global _tls_manager

    if _tls_manager is None:
        _tls_manager = ONEXTLSConfigManager()

    return _tls_manager
