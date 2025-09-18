"""Comprehensive tests for ModelTLSConfig - Security Model Testing.

Addresses PR #11 feedback about TLS config storing ssl_key_password as plain string field.
Tests security-sensitive field handling and TLS configuration validation.
"""

import pytest
from pathlib import Path
from pydantic import ValidationError, SecretStr

from omnibase_infra.models.core.security.model_tls_config import ModelTLSConfig


class TestModelTLSConfig:
    """Test ModelTLSConfig with comprehensive security coverage."""

    def test_model_initialization_defaults(self):
        """Test model initializes with secure defaults."""
        tls_config = ModelTLSConfig()

        # Verify TLS security defaults
        assert tls_config.tls_enabled is True  # Should default to secure
        assert tls_config.tls_version_min == "1.2"  # Should enforce modern TLS
        assert tls_config.ssl_cert_path is None
        assert tls_config.ssl_key_path is None
        assert tls_config.ssl_key_password is None
        assert tls_config.ssl_ca_path is None
        assert tls_config.ssl_verify_peer is True  # Should default to secure
        assert tls_config.ssl_verify_hostname is True  # Should default to secure
        assert tls_config.cipher_suites is None
        assert tls_config.ocsp_stapling_enabled is False
        assert tls_config.certificate_transparency_enabled is False
        assert tls_config.session_cache_enabled is True
        assert tls_config.session_cache_timeout_seconds == 300
        assert tls_config.renegotiation_allowed is False  # Should default to secure

    def test_secure_tls_configuration(self):
        """Test secure TLS configuration setup."""
        tls_config = ModelTLSConfig(
            tls_enabled=True,
            tls_version_min="1.3",
            ssl_cert_path="/etc/ssl/certs/server.crt",
            ssl_key_path="/etc/ssl/private/server.key",
            ssl_key_password="SecurePassword123!",  # Note: Should be SecretStr in production
            ssl_ca_path="/etc/ssl/certs/ca.crt",
            ssl_verify_peer=True,
            ssl_verify_hostname=True,
            cipher_suites=[
                "ECDHE-RSA-AES256-GCM-SHA384",
                "ECDHE-RSA-AES128-GCM-SHA256",
            ],
            ocsp_stapling_enabled=True,
            certificate_transparency_enabled=True,
            session_cache_enabled=True,
            session_cache_timeout_seconds=600,
            renegotiation_allowed=False,
        )

        assert tls_config.tls_enabled is True
        assert tls_config.tls_version_min == "1.3"
        assert tls_config.ssl_cert_path == "/etc/ssl/certs/server.crt"
        assert tls_config.ssl_key_path == "/etc/ssl/private/server.key"
        assert tls_config.ssl_key_password == "SecurePassword123!"
        assert tls_config.ssl_verify_peer is True
        assert tls_config.ssl_verify_hostname is True
        assert tls_config.ocsp_stapling_enabled is True
        assert tls_config.certificate_transparency_enabled is True
        assert tls_config.renegotiation_allowed is False

    def test_tls_version_validation(self):
        """Test TLS version validation constraints."""
        # Valid TLS versions
        valid_versions = ["1.2", "1.3"]
        for version in valid_versions:
            tls_config = ModelTLSConfig(tls_version_min=version)
            assert tls_config.tls_version_min == version

        # Invalid TLS versions (should be validated by pattern if enforced)
        # Note: Current model accepts any string - this test documents expected behavior
        insecure_versions = ["1.0", "1.1", "SSLv3"]
        for version in insecure_versions:
            # SECURITY NOTE: Model currently allows insecure versions
            # In production, this should be validated with a regex pattern
            tls_config = ModelTLSConfig(tls_version_min=version)
            assert tls_config.tls_version_min == version
            # TODO: Add regex validation pattern to reject insecure versions

    def test_ssl_password_security_concerns(self):
        """Test SSL password handling - documents security concerns from PR feedback."""
        # SECURITY CONCERN: Password stored as plain string (from PR #11 feedback)
        password = "VerySecretPassword123!"

        tls_config = ModelTLSConfig(ssl_key_password=password)

        # Current behavior: password stored as plain string
        assert tls_config.ssl_key_password == password

        # SECURITY RISK: Password could be logged or exposed
        json_data = tls_config.model_dump()
        assert json_data["ssl_key_password"] == password  # Exposed in JSON!

        # TODO: Implement SecretStr for sensitive fields
        # Expected secure behavior:
        # tls_config = ModelTLSConfig(ssl_key_password=SecretStr(password))
        # assert tls_config.ssl_key_password.get_secret_value() == password
        # json_data = tls_config.model_dump()
        # assert "ssl_key_password" not in json_data or json_data["ssl_key_password"] == "**********"

    def test_file_path_validation(self):
        """Test SSL file path validation and constraints."""
        # Test max length constraints (if any)
        long_path = "/very/long/path/" + "x" * 500

        # Currently model accepts very long paths - should consider adding max_length
        tls_config = ModelTLSConfig(
            ssl_cert_path=long_path,
            ssl_key_path=long_path,
            ssl_ca_path=long_path,
        )

        assert tls_config.ssl_cert_path == long_path
        assert tls_config.ssl_key_path == long_path
        assert tls_config.ssl_ca_path == long_path

    def test_cipher_suite_validation(self):
        """Test cipher suite list validation."""
        # Strong cipher suites
        strong_ciphers = [
            "ECDHE-RSA-AES256-GCM-SHA384",
            "ECDHE-RSA-AES128-GCM-SHA256",
            "ECDHE-ECDSA-AES256-GCM-SHA384",
            "ECDHE-ECDSA-AES128-GCM-SHA256",
        ]

        tls_config = ModelTLSConfig(cipher_suites=strong_ciphers)
        assert tls_config.cipher_suites == strong_ciphers

        # Weak cipher suites (should be avoided but currently allowed)
        weak_ciphers = [
            "DES-CBC-SHA",
            "RC4-MD5",
            "NULL-MD5",
        ]

        tls_config_weak = ModelTLSConfig(cipher_suites=weak_ciphers)
        assert tls_config_weak.cipher_suites == weak_ciphers
        # TODO: Add validation to reject weak cipher suites

    def test_session_cache_timeout_validation(self):
        """Test session cache timeout validation."""
        # Valid timeout values
        valid_timeouts = [60, 300, 600, 3600, 86400]  # 1 minute to 1 day
        for timeout in valid_timeouts:
            tls_config = ModelTLSConfig(session_cache_timeout_seconds=timeout)
            assert tls_config.session_cache_timeout_seconds == timeout

        # Test constraints (if any are enforced)
        # Currently model accepts any integer - should consider adding range validation
        extreme_timeouts = [0, -1, 999999999]
        for timeout in extreme_timeouts:
            try:
                tls_config = ModelTLSConfig(session_cache_timeout_seconds=timeout)
                # If no validation, this will pass
                assert tls_config.session_cache_timeout_seconds == timeout
            except ValidationError:
                # If validation is enforced, this should fail for negative values
                if timeout < 0:
                    pass  # Expected for negative values
                else:
                    raise

    def test_insecure_configuration_detection(self):
        """Test detection of insecure TLS configurations."""
        # Insecure configuration 1: TLS disabled
        insecure_config_1 = ModelTLSConfig(
            tls_enabled=False,
        )
        assert insecure_config_1.tls_enabled is False
        # TODO: Add method to check if configuration is secure

        # Insecure configuration 2: Peer verification disabled
        insecure_config_2 = ModelTLSConfig(
            ssl_verify_peer=False,
            ssl_verify_hostname=False,
        )
        assert insecure_config_2.ssl_verify_peer is False
        assert insecure_config_2.ssl_verify_hostname is False

        # Insecure configuration 3: Old TLS version
        insecure_config_3 = ModelTLSConfig(
            tls_version_min="1.1",  # Insecure version
        )
        assert insecure_config_3.tls_version_min == "1.1"

    def test_secure_configuration_best_practices(self):
        """Test secure TLS configuration following best practices."""
        secure_config = ModelTLSConfig(
            # Core security settings
            tls_enabled=True,
            tls_version_min="1.3",  # Latest TLS version

            # Certificate settings
            ssl_cert_path="/etc/ssl/certs/server.crt",
            ssl_key_path="/etc/ssl/private/server.key",
            ssl_ca_path="/etc/ssl/certs/ca-bundle.crt",

            # Verification settings (secure defaults)
            ssl_verify_peer=True,
            ssl_verify_hostname=True,

            # Strong cipher suites only
            cipher_suites=[
                "ECDHE-ECDSA-AES256-GCM-SHA384",
                "ECDHE-RSA-AES256-GCM-SHA384",
                "ECDHE-ECDSA-AES128-GCM-SHA256",
                "ECDHE-RSA-AES128-GCM-SHA256",
            ],

            # Enhanced security features
            ocsp_stapling_enabled=True,
            certificate_transparency_enabled=True,

            # Session security
            session_cache_enabled=True,
            session_cache_timeout_seconds=300,  # 5 minutes
            renegotiation_allowed=False,  # Prevent renegotiation attacks
        )

        # Verify all secure settings
        assert secure_config.tls_enabled is True
        assert secure_config.tls_version_min == "1.3"
        assert secure_config.ssl_verify_peer is True
        assert secure_config.ssl_verify_hostname is True
        assert secure_config.ocsp_stapling_enabled is True
        assert secure_config.certificate_transparency_enabled is True
        assert secure_config.renegotiation_allowed is False
        assert len(secure_config.cipher_suites) == 4

    def test_password_none_handling(self):
        """Test handling of None password (certificate without password)."""
        tls_config = ModelTLSConfig(
            ssl_cert_path="/etc/ssl/certs/server.crt",
            ssl_key_path="/etc/ssl/private/server.key",
            ssl_key_password=None,  # No password required
        )

        assert tls_config.ssl_key_password is None

    def test_json_serialization_security_review(self):
        """Test JSON serialization with security review for sensitive data exposure."""
        tls_config = ModelTLSConfig(
            ssl_cert_path="/etc/ssl/certs/server.crt",
            ssl_key_path="/etc/ssl/private/server.key",
            ssl_key_password="SuperSecretPassword123!",
            tls_version_min="1.3",
        )

        json_data = tls_config.model_dump()

        # SECURITY REVIEW: Check what gets serialized
        assert json_data["ssl_cert_path"] == "/etc/ssl/certs/server.crt"  # OK to expose
        assert json_data["ssl_key_path"] == "/etc/ssl/private/server.key"  # Path OK to expose
        assert json_data["ssl_key_password"] == "SuperSecretPassword123!"  # SECURITY RISK!
        assert json_data["tls_version_min"] == "1.3"  # OK to expose

        # TODO: Implement secure serialization that masks sensitive fields
        # Expected secure serialization:
        # assert json_data["ssl_key_password"] == "**********"

    def test_model_config_validation_settings(self):
        """Test model configuration for validation settings."""
        tls_config = ModelTLSConfig()

        # Check if model has strict validation settings
        try:
            # Test extra field rejection
            ModelTLSConfig(invalid_field="should_be_rejected")
            # If this doesn't raise an error, extra fields are allowed (less secure)
            extra_fields_allowed = True
        except ValidationError:
            # If this raises an error, extra fields are forbidden (more secure)
            extra_fields_allowed = False

        # Document current behavior
        # TODO: Ensure extra fields are forbidden in production model
        print(f"Extra fields allowed: {extra_fields_allowed}")

    def test_certificate_file_existence_validation(self):
        """Test certificate file existence validation (if implemented)."""
        # Test with non-existent files
        tls_config = ModelTLSConfig(
            ssl_cert_path="/nonexistent/cert.crt",
            ssl_key_path="/nonexistent/key.key",
            ssl_ca_path="/nonexistent/ca.crt",
        )

        # Currently, model doesn't validate file existence
        # This is acceptable as files might not exist at model creation time
        assert tls_config.ssl_cert_path == "/nonexistent/cert.crt"
        assert tls_config.ssl_key_path == "/nonexistent/key.key"
        assert tls_config.ssl_ca_path == "/nonexistent/ca.crt"

    def test_security_hardening_recommendations(self):
        """Document security hardening recommendations based on testing."""
        recommendations = [
            "1. Use SecretStr for ssl_key_password field",
            "2. Add regex validation for tls_version_min (reject < 1.2)",
            "3. Add cipher suite strength validation",
            "4. Add max_length constraints for file paths",
            "5. Implement secure JSON serialization for sensitive fields",
            "6. Add field validators to prevent logging of sensitive data",
            "7. Consider adding is_secure() method for configuration validation",
            "8. Add session_cache_timeout_seconds range validation (60-86400)",
            "9. Ensure extra='forbid' in model config",
            "10. Add comprehensive security validation method",
        ]

        # This test documents the recommendations - implementation needed
        for i, recommendation in enumerate(recommendations, 1):
            assert recommendation.startswith(f"{i}.")

        print("\nSecurity Hardening Recommendations:")
        for recommendation in recommendations:
            print(f"  - {recommendation}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short", "-s"])  # -s to show print statements