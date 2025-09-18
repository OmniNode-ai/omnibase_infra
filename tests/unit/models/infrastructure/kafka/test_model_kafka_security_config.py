"""Test suite for Kafka security configuration models."""

import pytest
from pydantic import ValidationError

from omnibase_infra.models.infrastructure.kafka.model_kafka_security_config import (
    ModelKafkaSecurityConfig,
)
from omnibase_infra.models.infrastructure.kafka.model_kafka_ssl_config import (
    ModelKafkaSSLConfig,
)
from omnibase_infra.models.infrastructure.kafka.model_kafka_sasl_config import (
    ModelKafkaSASLConfig,
)


class TestModelKafkaSSLConfig:
    """Test cases for Kafka SSL configuration model."""

    def test_create_default_ssl_config(self):
        """Test creating SSL config with default values."""
        ssl_config = ModelKafkaSSLConfig()

        assert ssl_config.ssl_check_hostname is True
        assert ssl_config.ssl_cafile is None
        assert ssl_config.ssl_certfile is None
        assert ssl_config.ssl_keyfile is None
        assert ssl_config.ssl_password is None
        assert ssl_config.ssl_protocol == "TLSv1_2"

    def test_create_full_ssl_config(self):
        """Test creating SSL config with all fields."""
        ssl_config = ModelKafkaSSLConfig(
            ssl_check_hostname=False,
            ssl_cafile="/path/to/ca.pem",
            ssl_certfile="/path/to/cert.pem",
            ssl_keyfile="/path/to/key.pem",
            ssl_password="secret123",
            ssl_crlfile="/path/to/crl.pem",
            ssl_ciphers="ECDHE+AESGCM",
            ssl_protocol="TLSv1_3",
            ssl_context="custom_context"
        )

        assert ssl_config.ssl_check_hostname is False
        assert ssl_config.ssl_cafile == "/path/to/ca.pem"
        assert ssl_config.ssl_certfile == "/path/to/cert.pem"
        assert ssl_config.ssl_keyfile == "/path/to/key.pem"
        assert ssl_config.ssl_password == "secret123"
        assert ssl_config.ssl_protocol == "TLSv1_3"

    @pytest.mark.parametrize("protocol", ["TLSv1_2", "TLSv1_3", "SSLv23"])
    def test_ssl_protocol_options(self, protocol):
        """Test different SSL protocol options."""
        ssl_config = ModelKafkaSSLConfig(ssl_protocol=protocol)
        assert ssl_config.ssl_protocol == protocol


class TestModelKafkaSASLConfig:
    """Test cases for Kafka SASL configuration model."""

    def test_create_default_sasl_config(self):
        """Test creating SASL config with default values."""
        sasl_config = ModelKafkaSASLConfig()

        assert sasl_config.sasl_mechanism == "PLAIN"
        assert sasl_config.sasl_plain_username is None
        assert sasl_config.sasl_plain_password is None
        assert sasl_config.sasl_kerberos_service_name == "kafka"

    def test_create_plain_sasl_config(self):
        """Test creating PLAIN SASL configuration."""
        sasl_config = ModelKafkaSASLConfig(
            sasl_mechanism="PLAIN",
            sasl_plain_username="testuser",
            sasl_plain_password="testpass"
        )

        assert sasl_config.sasl_mechanism == "PLAIN"
        assert sasl_config.sasl_plain_username == "testuser"
        assert sasl_config.sasl_plain_password == "testpass"

    @pytest.mark.parametrize("mechanism", ["PLAIN", "SCRAM-SHA-256", "SCRAM-SHA-512", "GSSAPI"])
    def test_sasl_mechanisms(self, mechanism):
        """Test different SASL authentication mechanisms."""
        sasl_config = ModelKafkaSASLConfig(sasl_mechanism=mechanism)
        assert sasl_config.sasl_mechanism == mechanism

    def test_kerberos_sasl_config(self):
        """Test Kerberos SASL configuration."""
        sasl_config = ModelKafkaSASLConfig(
            sasl_mechanism="GSSAPI",
            sasl_kerberos_service_name="kafka",
            sasl_kerberos_domain_name="EXAMPLE.COM"
        )

        assert sasl_config.sasl_mechanism == "GSSAPI"
        assert sasl_config.sasl_kerberos_service_name == "kafka"
        assert sasl_config.sasl_kerberos_domain_name == "EXAMPLE.COM"


class TestModelKafkaSecurityConfig:
    """Test cases for Kafka security configuration model."""

    def test_create_default_security_config(self):
        """Test creating security config with default values."""
        security_config = ModelKafkaSecurityConfig()

        assert security_config.security_protocol == "PLAINTEXT"
        assert security_config.ssl_config is None
        assert security_config.sasl_config is None
        assert security_config.enable_auto_commit is True
        assert security_config.auto_commit_interval_ms == 5000

    @pytest.mark.parametrize("protocol", ["PLAINTEXT", "SSL", "SASL_PLAINTEXT", "SASL_SSL"])
    def test_security_protocols(self, protocol):
        """Test different security protocols."""
        security_config = ModelKafkaSecurityConfig(security_protocol=protocol)
        assert security_config.security_protocol == protocol

    def test_ssl_only_configuration(self):
        """Test SSL-only security configuration."""
        ssl_config = ModelKafkaSSLConfig(
            ssl_cafile="/path/to/ca.pem",
            ssl_protocol="TLSv1_3"
        )

        security_config = ModelKafkaSecurityConfig(
            security_protocol="SSL",
            ssl_config=ssl_config
        )

        assert security_config.security_protocol == "SSL"
        assert security_config.ssl_config is not None
        assert security_config.ssl_config.ssl_cafile == "/path/to/ca.pem"
        assert security_config.sasl_config is None

    def test_sasl_only_configuration(self):
        """Test SASL-only security configuration."""
        sasl_config = ModelKafkaSASLConfig(
            sasl_mechanism="PLAIN",
            sasl_plain_username="user",
            sasl_plain_password="pass"
        )

        security_config = ModelKafkaSecurityConfig(
            security_protocol="SASL_PLAINTEXT",
            sasl_config=sasl_config
        )

        assert security_config.security_protocol == "SASL_PLAINTEXT"
        assert security_config.sasl_config is not None
        assert security_config.sasl_config.sasl_mechanism == "PLAIN"
        assert security_config.ssl_config is None

    def test_ssl_and_sasl_configuration(self):
        """Test combined SSL and SASL security configuration."""
        ssl_config = ModelKafkaSSLConfig(ssl_protocol="TLSv1_3")
        sasl_config = ModelKafkaSASLConfig(sasl_mechanism="SCRAM-SHA-256")

        security_config = ModelKafkaSecurityConfig(
            security_protocol="SASL_SSL",
            ssl_config=ssl_config,
            sasl_config=sasl_config
        )

        assert security_config.security_protocol == "SASL_SSL"
        assert security_config.ssl_config is not None
        assert security_config.sasl_config is not None

    def test_timeout_configurations(self):
        """Test various timeout configurations."""
        security_config = ModelKafkaSecurityConfig(
            session_timeout_ms=30000,
            heartbeat_interval_ms=10000,
            max_poll_interval_ms=600000,
            request_timeout_ms=60000
        )

        assert security_config.session_timeout_ms == 30000
        assert security_config.heartbeat_interval_ms == 10000
        assert security_config.max_poll_interval_ms == 600000
        assert security_config.request_timeout_ms == 60000

    def test_model_integration(self):
        """Test that all three models work together correctly."""
        # Create SSL config
        ssl_config = ModelKafkaSSLConfig(
            ssl_check_hostname=True,
            ssl_protocol="TLSv1_3"
        )

        # Create SASL config
        sasl_config = ModelKafkaSASLConfig(
            sasl_mechanism="SCRAM-SHA-512",
            sasl_plain_username="secure_user",
            sasl_plain_password="secure_pass"
        )

        # Create main security config
        security_config = ModelKafkaSecurityConfig(
            security_protocol="SASL_SSL",
            ssl_config=ssl_config,
            sasl_config=sasl_config,
            enable_auto_commit=False,
            auto_commit_interval_ms=10000
        )

        # Verify integration
        assert security_config.security_protocol == "SASL_SSL"
        assert security_config.ssl_config.ssl_protocol == "TLSv1_3"
        assert security_config.sasl_config.sasl_mechanism == "SCRAM-SHA-512"
        assert security_config.enable_auto_commit is False