# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for Ed25519 key loader utilities.

Tests verify that:
    - Valid Ed25519 PEM files load successfully (private and public)
    - File-not-found raises ProtocolConfigurationError
    - Invalid PEM data raises ProtocolConfigurationError
    - Wrong key type (e.g., RSA) raises ProtocolConfigurationError
    - Unreadable files raise ProtocolConfigurationError

Related Tickets:
    - OMN-1899: Runtime gateway envelope signing
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.asymmetric.rsa import generate_private_key
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.gateway.utils.util_key_loader import (
    load_private_key_from_pem,
    load_public_key_from_pem,
)

pytestmark = pytest.mark.integration


# =============================================================================
# Helpers
# =============================================================================


def _write_ed25519_keypair(tmp_path: Path) -> tuple[Path, Path]:
    """Generate an Ed25519 keypair and write both keys as PEM files.

    Args:
        tmp_path: Temporary directory from pytest fixture.

    Returns:
        Tuple of (private_key_path, public_key_path).
    """
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    private_pem = private_key.private_bytes(
        encoding=Encoding.PEM,
        format=PrivateFormat.PKCS8,
        encryption_algorithm=NoEncryption(),
    )
    public_pem = public_key.public_bytes(
        encoding=Encoding.PEM,
        format=PublicFormat.SubjectPublicKeyInfo,
    )

    private_path = tmp_path / "private.pem"
    public_path = tmp_path / "public.pem"

    private_path.write_bytes(private_pem)
    public_path.write_bytes(public_pem)

    return private_path, public_path


def _write_rsa_keypair(tmp_path: Path) -> tuple[Path, Path]:
    """Generate an RSA keypair and write both keys as PEM files.

    Args:
        tmp_path: Temporary directory from pytest fixture.

    Returns:
        Tuple of (private_key_path, public_key_path).
    """
    private_key = generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    public_key = private_key.public_key()

    private_pem = private_key.private_bytes(
        encoding=Encoding.PEM,
        format=PrivateFormat.PKCS8,
        encryption_algorithm=NoEncryption(),
    )
    public_pem = public_key.public_bytes(
        encoding=Encoding.PEM,
        format=PublicFormat.SubjectPublicKeyInfo,
    )

    private_path = tmp_path / "rsa_private.pem"
    public_path = tmp_path / "rsa_public.pem"

    private_path.write_bytes(private_pem)
    public_path.write_bytes(public_pem)

    return private_path, public_path


# =============================================================================
# Tests: load_private_key_from_pem - Happy Path
# =============================================================================


class TestLoadPrivateKeyFromPemSuccess:
    """Happy-path tests for loading Ed25519 private keys."""

    def test_load_private_key_from_pem_success(self, tmp_path: Path) -> None:
        """Valid Ed25519 PEM file loads successfully and returns Ed25519PrivateKey."""
        # Arrange
        private_path, _ = _write_ed25519_keypair(tmp_path)

        # Act
        key = load_private_key_from_pem(private_path)

        # Assert
        assert isinstance(key, Ed25519PrivateKey)

    def test_load_private_key_produces_working_key(self, tmp_path: Path) -> None:
        """Loaded private key can sign data successfully."""
        # Arrange
        private_path, _ = _write_ed25519_keypair(tmp_path)

        # Act
        key = load_private_key_from_pem(private_path)
        signature = key.sign(b"test message")

        # Assert
        assert isinstance(signature, bytes)
        assert len(signature) == 64  # Ed25519 signatures are 64 bytes


# =============================================================================
# Tests: load_private_key_from_pem - Error Branches
# =============================================================================


class TestLoadPrivateKeyFromPemErrors:
    """Error branch tests for loading Ed25519 private keys."""

    def test_load_private_key_file_not_found(self, tmp_path: Path) -> None:
        """Non-existent file raises ProtocolConfigurationError."""
        # Arrange
        nonexistent = tmp_path / "does_not_exist.pem"

        # Act & Assert
        with pytest.raises(ProtocolConfigurationError, match="does not exist"):
            load_private_key_from_pem(nonexistent)

    def test_load_private_key_invalid_pem_data(self, tmp_path: Path) -> None:
        """File containing non-PEM data raises ProtocolConfigurationError."""
        # Arrange
        bad_file = tmp_path / "garbage.pem"
        bad_file.write_bytes(b"this is not a PEM file at all")

        # Act & Assert
        with pytest.raises(ProtocolConfigurationError, match="Invalid PEM format"):
            load_private_key_from_pem(bad_file)

    def test_load_private_key_wrong_key_type_rsa(self, tmp_path: Path) -> None:
        """RSA private key PEM raises ProtocolConfigurationError (wrong type)."""
        # Arrange
        rsa_private_path, _ = _write_rsa_keypair(tmp_path)

        # Act & Assert
        with pytest.raises(ProtocolConfigurationError, match="Key is not Ed25519"):
            load_private_key_from_pem(rsa_private_path)

    def test_load_private_key_read_error(self, tmp_path: Path) -> None:
        """OSError during file read raises ProtocolConfigurationError."""
        # Arrange
        key_path = tmp_path / "unreadable.pem"
        key_path.write_bytes(b"dummy")

        # Act & Assert
        with patch.object(Path, "read_bytes", side_effect=OSError("Permission denied")):
            with pytest.raises(
                ProtocolConfigurationError, match="Failed to read private key file"
            ):
                load_private_key_from_pem(key_path)

    def test_load_private_key_empty_file(self, tmp_path: Path) -> None:
        """Empty file raises ProtocolConfigurationError."""
        # Arrange
        empty_file = tmp_path / "empty.pem"
        empty_file.write_bytes(b"")

        # Act & Assert
        with pytest.raises(ProtocolConfigurationError, match="Invalid PEM format"):
            load_private_key_from_pem(empty_file)

    def test_load_private_key_public_key_in_private_slot(self, tmp_path: Path) -> None:
        """Public key PEM loaded as private key raises ProtocolConfigurationError."""
        # Arrange
        _, public_path = _write_ed25519_keypair(tmp_path)

        # Act & Assert - a public key PEM is not a valid private key PEM
        with pytest.raises(ProtocolConfigurationError):
            load_private_key_from_pem(public_path)


# =============================================================================
# Tests: load_public_key_from_pem - Happy Path
# =============================================================================


class TestLoadPublicKeyFromPemSuccess:
    """Happy-path tests for loading Ed25519 public keys."""

    def test_load_public_key_from_pem_success(self, tmp_path: Path) -> None:
        """Valid Ed25519 public key PEM file loads successfully."""
        # Arrange
        _, public_path = _write_ed25519_keypair(tmp_path)

        # Act
        key = load_public_key_from_pem(public_path)

        # Assert
        assert isinstance(key, Ed25519PublicKey)

    def test_load_public_key_can_verify(self, tmp_path: Path) -> None:
        """Loaded public key can verify signatures from the matching private key."""
        # Arrange
        private_path, public_path = _write_ed25519_keypair(tmp_path)
        private_key = load_private_key_from_pem(private_path)
        public_key = load_public_key_from_pem(public_path)

        # Act
        message = b"test message for verification"
        signature = private_key.sign(message)

        # Assert - should not raise
        public_key.verify(signature, message)


# =============================================================================
# Tests: load_public_key_from_pem - Error Branches
# =============================================================================


class TestLoadPublicKeyFromPemErrors:
    """Error branch tests for loading Ed25519 public keys."""

    def test_load_public_key_file_not_found(self, tmp_path: Path) -> None:
        """Non-existent file raises ProtocolConfigurationError."""
        # Arrange
        nonexistent = tmp_path / "does_not_exist.pem"

        # Act & Assert
        with pytest.raises(ProtocolConfigurationError, match="does not exist"):
            load_public_key_from_pem(nonexistent)

    def test_load_public_key_invalid_pem_data(self, tmp_path: Path) -> None:
        """File containing non-PEM data raises ProtocolConfigurationError."""
        # Arrange
        bad_file = tmp_path / "garbage.pem"
        bad_file.write_bytes(b"not a valid PEM file")

        # Act & Assert
        with pytest.raises(ProtocolConfigurationError, match="Invalid PEM format"):
            load_public_key_from_pem(bad_file)

    def test_load_public_key_wrong_key_type_rsa(self, tmp_path: Path) -> None:
        """RSA public key PEM raises ProtocolConfigurationError (wrong type)."""
        # Arrange
        _, rsa_public_path = _write_rsa_keypair(tmp_path)

        # Act & Assert
        with pytest.raises(ProtocolConfigurationError, match="Key is not Ed25519"):
            load_public_key_from_pem(rsa_public_path)

    def test_load_public_key_read_error(self, tmp_path: Path) -> None:
        """OSError during file read raises ProtocolConfigurationError."""
        # Arrange
        key_path = tmp_path / "unreadable.pem"
        key_path.write_bytes(b"dummy")

        # Act & Assert
        with patch.object(Path, "read_bytes", side_effect=OSError("Permission denied")):
            with pytest.raises(
                ProtocolConfigurationError, match="Failed to read public key file"
            ):
                load_public_key_from_pem(key_path)

    def test_load_public_key_empty_file(self, tmp_path: Path) -> None:
        """Empty file raises ProtocolConfigurationError."""
        # Arrange
        empty_file = tmp_path / "empty.pem"
        empty_file.write_bytes(b"")

        # Act & Assert
        with pytest.raises(ProtocolConfigurationError, match="Invalid PEM format"):
            load_public_key_from_pem(empty_file)

    def test_load_public_key_private_key_in_public_slot(self, tmp_path: Path) -> None:
        """Private key PEM loaded as public key raises ProtocolConfigurationError."""
        # Arrange
        private_path, _ = _write_ed25519_keypair(tmp_path)

        # Act & Assert - a private key PEM is not a valid public key PEM
        with pytest.raises(ProtocolConfigurationError):
            load_public_key_from_pem(private_path)


# =============================================================================
# Tests: Error Context Verification
# =============================================================================


class TestKeyLoaderErrorContext:
    """Tests verifying that errors include proper context metadata."""

    def test_private_key_error_has_correlation_id(self, tmp_path: Path) -> None:
        """ProtocolConfigurationError from private key loader includes correlation_id."""
        # Arrange
        nonexistent = tmp_path / "missing_private.pem"

        # Act & Assert
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_private_key_from_pem(nonexistent)

        # correlation_id is set on the error model
        assert exc_info.value.correlation_id is not None

    def test_private_key_error_has_operation_in_model_context(
        self, tmp_path: Path
    ) -> None:
        """ProtocolConfigurationError includes operation in model context."""
        # Arrange
        nonexistent = tmp_path / "missing_private.pem"

        # Act & Assert
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_private_key_from_pem(nonexistent)

        # Operation is passed as structured context to the error model
        assert (
            exc_info.value.model.context.get("operation") == "load_private_key_from_pem"
        )

    def test_public_key_error_has_correlation_id(self, tmp_path: Path) -> None:
        """ProtocolConfigurationError from public key loader includes correlation_id."""
        # Arrange
        nonexistent = tmp_path / "missing_public.pem"

        # Act & Assert
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_public_key_from_pem(nonexistent)

        assert exc_info.value.correlation_id is not None

    def test_public_key_error_has_operation_in_model_context(
        self, tmp_path: Path
    ) -> None:
        """ProtocolConfigurationError includes operation in model context."""
        # Arrange
        nonexistent = tmp_path / "missing_public.pem"

        # Act & Assert
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_public_key_from_pem(nonexistent)

        assert (
            exc_info.value.model.context.get("operation") == "load_public_key_from_pem"
        )
