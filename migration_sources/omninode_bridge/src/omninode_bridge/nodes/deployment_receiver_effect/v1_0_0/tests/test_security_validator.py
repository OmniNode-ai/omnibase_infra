#!/usr/bin/env python3
"""
Unit tests for security validator.
Tests HMAC authentication, BLAKE3 checksum validation, and IP whitelisting.
"""

import hashlib
import hmac
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

try:
    import blake3

    BLAKE3_AVAILABLE = True
except ImportError:
    BLAKE3_AVAILABLE = False

from ..models.model_auth import ModelAuthCredentials
from ..security_validator import SecurityValidator


class TestSecurityValidator:
    """Test security validation functionality."""

    @pytest.fixture
    def validator(self):
        """Create security validator with test secret."""
        return SecurityValidator(
            secret_key="test-secret-key-for-hmac",
            allowed_ip_ranges=["192.168.86.0/24", "10.0.0.0/8"],
        )

    @pytest.fixture
    def valid_credentials(self):
        """Create valid authentication credentials."""
        sender_id = uuid4()
        auth_token = "a" * 32
        secret_key = "test-secret-key-for-hmac"
        message = b"test-message"

        # Generate valid signature
        signature = hmac.new(
            secret_key.encode("utf-8"), message, hashlib.sha256
        ).hexdigest()

        return ModelAuthCredentials(
            sender_id=sender_id,
            auth_token=auth_token,
            signature=signature,
            sender_ip="192.168.86.101",
        )

    def test_init(self, validator):
        """Test validator initialization."""
        assert validator.secret_key == b"test-secret-key-for-hmac"
        assert len(validator.allowed_ip_ranges) == 2
        assert len(validator.parsed_ranges) == 2

    def test_hmac_valid_signature(self, validator, valid_credentials):
        """Test HMAC validation with valid signature."""
        message = b"test-message"
        result = validator.validate_hmac_signature(valid_credentials, message)

        assert result.is_valid is True
        assert result.error_code is None
        assert result.sender_id == valid_credentials.sender_id

    def test_hmac_invalid_signature(self, validator, valid_credentials):
        """Test HMAC validation with invalid signature."""
        # Change signature to make it invalid
        invalid_credentials = valid_credentials.model_copy()
        invalid_credentials.signature = "invalid" + valid_credentials.signature

        message = b"test-message"
        result = validator.validate_hmac_signature(invalid_credentials, message)

        assert result.is_valid is False
        assert result.error_code == "AUTHENTICATION_FAILED"
        assert "mismatch" in result.error_message.lower()

    def test_hmac_wrong_message(self, validator, valid_credentials):
        """Test HMAC validation with wrong message."""
        # Signature is valid but for different message
        result = validator.validate_hmac_signature(
            valid_credentials, b"different-message"
        )

        assert result.is_valid is False
        assert result.error_code == "AUTHENTICATION_FAILED"

    @pytest.mark.skipif(not BLAKE3_AVAILABLE, reason="blake3 library not installed")
    def test_checksum_valid(self, validator):
        """Test BLAKE3 checksum validation with valid checksum."""
        # Create temporary file with known content
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            content = b"test file content for checksum validation"
            f.write(content)
            temp_path = f.name

        try:
            # Calculate expected checksum
            expected_checksum = blake3.blake3(content).hexdigest()

            # Validate checksum
            result = validator.validate_checksum(temp_path, expected_checksum)

            assert result.is_valid is True
            assert result.expected_checksum == expected_checksum
            assert result.actual_checksum == expected_checksum
            assert result.error_message is None

        finally:
            Path(temp_path).unlink()

    @pytest.mark.skipif(not BLAKE3_AVAILABLE, reason="blake3 library not installed")
    def test_checksum_invalid(self, validator):
        """Test BLAKE3 checksum validation with invalid checksum."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            f.write(b"test content")
            temp_path = f.name

        try:
            # Use wrong checksum
            wrong_checksum = "a" * 64

            result = validator.validate_checksum(temp_path, wrong_checksum)

            assert result.is_valid is False
            assert result.expected_checksum == wrong_checksum
            assert result.actual_checksum != wrong_checksum
            assert "mismatch" in result.error_message.lower()

        finally:
            Path(temp_path).unlink()

    def test_checksum_file_not_found(self, validator):
        """Test BLAKE3 checksum validation with non-existent file."""
        result = validator.validate_checksum("/nonexistent/file.tar", "a" * 64)

        assert result.is_valid is False
        assert "not found" in result.error_message.lower()

    def test_ip_whitelist_allowed_local(self, validator):
        """Test IP whitelist with allowed local network IP."""
        result = validator.validate_ip_whitelist("192.168.86.101")

        assert result.is_allowed is True
        assert result.ip_address == "192.168.86.101"
        assert result.matched_range == "192.168.86.0/24"
        assert result.error_message is None

    def test_ip_whitelist_allowed_private(self, validator):
        """Test IP whitelist with allowed private network IP."""
        result = validator.validate_ip_whitelist("10.0.5.100")

        assert result.is_allowed is True
        assert result.ip_address == "10.0.5.100"
        assert result.matched_range == "10.0.0.0/8"

    def test_ip_whitelist_denied(self, validator):
        """Test IP whitelist with denied public IP."""
        result = validator.validate_ip_whitelist("8.8.8.8")

        assert result.is_allowed is False
        assert result.ip_address == "8.8.8.8"
        assert result.matched_range is None
        assert "not in allowed ranges" in result.error_message

    def test_ip_whitelist_invalid_ip(self, validator):
        """Test IP whitelist with invalid IP address."""
        result = validator.validate_ip_whitelist("invalid-ip")

        assert result.is_allowed is False
        assert "Invalid IP address" in result.error_message

    @pytest.mark.skipif(not BLAKE3_AVAILABLE, reason="blake3 library not installed")
    def test_validate_all_success(self, validator, valid_credentials):
        """Test complete validation with all checks passing."""
        # Create temporary file with known content
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            content = b"test content for full validation"
            f.write(content)
            temp_path = f.name

        try:
            expected_checksum = blake3.blake3(content).hexdigest()
            message = b"test-message"

            # Generate valid signature
            signature = hmac.new(
                validator.secret_key, message, hashlib.sha256
            ).hexdigest()

            credentials = ModelAuthCredentials(
                sender_id=uuid4(),
                auth_token="a" * 32,
                signature=signature,
                sender_ip="192.168.86.101",
            )

            auth_result, checksum_result, ip_result = validator.validate_all(
                credentials, temp_path, expected_checksum, message
            )

            assert auth_result.is_valid is True
            assert checksum_result.is_valid is True
            assert ip_result.is_allowed is True

        finally:
            Path(temp_path).unlink()

    def test_validate_all_auth_failure(self, validator, valid_credentials):
        """Test complete validation with authentication failure."""
        # Invalid credentials
        invalid_credentials = valid_credentials.model_copy()
        invalid_credentials.signature = "invalid"

        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            f.write(b"content")
            temp_path = f.name

        try:
            auth_result, checksum_result, ip_result = validator.validate_all(
                invalid_credentials, temp_path, "a" * 64, b"test-message"
            )

            # Auth should fail, but other checks still run
            assert auth_result.is_valid is False

        finally:
            Path(temp_path).unlink()

    def test_validate_all_no_ip(self, validator, valid_credentials):
        """Test complete validation without IP address (local testing)."""
        # Remove IP from credentials
        credentials = valid_credentials.model_copy()
        credentials.sender_ip = None

        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            f.write(b"content")
            temp_path = f.name

        try:
            message = b"test-message"
            signature = hmac.new(
                validator.secret_key, message, hashlib.sha256
            ).hexdigest()
            credentials.signature = signature

            auth_result, checksum_result, ip_result = validator.validate_all(
                credentials, temp_path, "a" * 64, message
            )

            # IP check should pass for local testing
            assert ip_result.is_allowed is True
            assert ip_result.ip_address == "unknown"

        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
