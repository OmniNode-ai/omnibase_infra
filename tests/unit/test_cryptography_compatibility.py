"""Test cryptography 46.x compatibility.

Verifies cryptography package upgrade (46.0.3) doesn't break existing
functionality. Addresses CVE-2024-12797 security fix validation.

The cryptography package is a transitive dependency via omnibase_core 0.6.2.
This codebase directly uses:
- hashlib.sha256 for content-derived IDs (registration_reducer.py)

See: pyproject.toml lines 57-61 for upgrade rationale.
"""

import hashlib
from uuid import UUID

import pytest


class TestCryptographyCompatibility:
    """Verify cryptography package functions correctly after upgrade."""

    def test_cryptography_import(self) -> None:
        """Verify cryptography package imports successfully."""
        import cryptography

        # Verify version is 46.x as specified in pyproject.toml
        version = cryptography.__version__
        assert version.startswith("46."), (
            f"Expected cryptography 46.x, got {version}. "
            "Update pyproject.toml constraint if intentional."
        )

    def test_cryptography_hazmat_primitives_available(self) -> None:
        """Verify hazmat primitives module is accessible.

        This tests the core cryptographic primitives module which is
        commonly used by dependent packages.
        """
        from cryptography.hazmat.primitives import hashes

        # Verify SHA256 algorithm is available
        sha256 = hashes.SHA256()
        assert sha256.name == "sha256"
        assert sha256.digest_size == 32
        assert sha256.block_size == 64

    def test_cryptography_fernet_available(self) -> None:
        """Verify Fernet symmetric encryption is available.

        Fernet is commonly used for symmetric encryption and may be
        used by transitive dependencies.
        """
        from cryptography.fernet import Fernet

        # Generate a key and verify basic operations
        key = Fernet.generate_key()
        assert isinstance(key, bytes)
        assert len(key) == 44  # Base64-encoded 32-byte key

        # Verify encryption/decryption round-trip
        fernet = Fernet(key)
        plaintext = b"test message"
        ciphertext = fernet.encrypt(plaintext)
        decrypted = fernet.decrypt(ciphertext)
        assert decrypted == plaintext


class TestHashlibCompatibility:
    """Verify hashlib operations used in codebase work correctly.

    The registration_reducer.py uses hashlib.sha256 for content-derived IDs.
    This validates that functionality continues to work.
    """

    def test_hashlib_sha256_basic(self) -> None:
        """Verify basic SHA256 hashing works."""
        content = "test-content"
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # SHA256 produces 64 hex characters (256 bits / 4 bits per hex char)
        assert len(content_hash) == 64
        assert all(c in "0123456789abcdef" for c in content_hash)

    def test_hashlib_sha256_deterministic(self) -> None:
        """Verify SHA256 produces deterministic results.

        Content-derived IDs require consistent hashing.
        """
        content = "node-123|EFFECT|2024-01-01T00:00:00"

        hash1 = hashlib.sha256(content.encode("utf-8")).hexdigest()
        hash2 = hashlib.sha256(content.encode("utf-8")).hexdigest()

        assert hash1 == hash2, "SHA256 should be deterministic"

    def test_content_derived_uuid_pattern(self) -> None:
        """Verify content-derived UUID generation pattern works.

        This replicates the logic in RegistrationReducer._generate_registration_id
        to ensure the upgrade doesn't break ID generation.
        """
        # Sample input matching the pattern in registration_reducer.py
        node_id = "test-node-123"
        node_type = "EFFECT"
        timestamp_iso = "2024-01-01T00:00:00+00:00"

        # Build canonical content with pipe delimiter
        canonical_content = f"{node_id}|{node_type}|{timestamp_iso}"

        # Compute SHA256 and convert to UUID format
        content_hash = hashlib.sha256(canonical_content.encode("utf-8")).hexdigest()

        # Take first 32 hex chars (128 bits) and format as UUID
        uuid_hex = content_hash[:32]
        uuid_str = (
            f"{uuid_hex[:8]}-{uuid_hex[8:12]}-{uuid_hex[12:16]}-"
            f"{uuid_hex[16:20]}-{uuid_hex[20:32]}"
        )

        # Verify it produces a valid UUID
        result_uuid = UUID(uuid_str)
        assert str(result_uuid) == uuid_str

        # Verify determinism - same input should produce same UUID
        content_hash2 = hashlib.sha256(canonical_content.encode("utf-8")).hexdigest()
        uuid_hex2 = content_hash2[:32]
        uuid_str2 = (
            f"{uuid_hex2[:8]}-{uuid_hex2[8:12]}-{uuid_hex2[12:16]}-"
            f"{uuid_hex2[16:20]}-{uuid_hex2[20:32]}"
        )
        assert uuid_str == uuid_str2, "Content-derived UUIDs should be deterministic"


class TestSecretsModuleAvailable:
    """Verify secrets module is available for secure random operations.

    While this codebase may not directly use the secrets module,
    it's the recommended approach for cryptographic random values
    and should be available.
    """

    def test_secrets_token_hex(self) -> None:
        """Verify secrets.token_hex works for secure random hex strings."""
        import secrets

        token = secrets.token_hex(16)
        assert len(token) == 32  # 16 bytes = 32 hex chars
        assert all(c in "0123456789abcdef" for c in token)

    def test_secrets_token_bytes(self) -> None:
        """Verify secrets.token_bytes works for secure random bytes."""
        import secrets

        token = secrets.token_bytes(32)
        assert isinstance(token, bytes)
        assert len(token) == 32

    def test_secrets_compare_digest(self) -> None:
        """Verify constant-time comparison is available.

        This is important for timing-attack resistant comparisons.
        """
        import secrets

        a = b"test-secret-value"
        b_same = b"test-secret-value"
        b_diff = b"different-value"

        assert secrets.compare_digest(a, b_same) is True
        assert secrets.compare_digest(a, b_diff) is False
