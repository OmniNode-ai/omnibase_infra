"""
Signature validation for O.N.E. v0.1 protocol.

This module provides ed25519 signature validation and
message integrity verification.
"""

import base64
import hashlib
import hmac
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SignatureValidator:
    """
    Validator for ed25519 signatures and message integrity.

    Provides signature verification and trusted public key management
    for O.N.E. protocol security.
    """

    def __init__(self):
        """Initialize signature validator."""
        self.public_keys: dict[str, str] = {}
        self.trusted_key_ids: set = set()
        self._ed25519_available = self._check_ed25519_support()

    def _check_ed25519_support(self) -> bool:
        """Check if ed25519 support is available."""
        import importlib.util

        if importlib.util.find_spec("ed25519") is not None:
            return True
        else:
            logger.warning("ed25519 module not available. Using HMAC-SHA256 fallback.")
            return False

    def verify_ed25519_signature(
        self, message: bytes, signature: str, public_key: str
    ) -> bool:
        """
        Verify ed25519 signature.

        Args:
            message: Message bytes to verify
            signature: Base64 encoded signature
            public_key: Base64 encoded public key

        Returns:
            bool: True if signature valid
        """
        if not self._ed25519_available:
            logger.warning("ed25519 not available, using HMAC fallback")
            return self.verify_hmac_signature(message, signature, public_key)

        try:
            import ed25519

            # Decode signature and key
            sig_bytes = base64.b64decode(signature)
            key_bytes = base64.b64decode(public_key)

            # Create verifying key and verify
            verifying_key = ed25519.VerifyingKey(key_bytes)
            verifying_key.verify(sig_bytes, message)

            logger.debug("ed25519 signature verification successful")
            return True

        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False

    def verify_hmac_signature(
        self, message: bytes, signature: str, secret_key: str
    ) -> bool:
        """
        Verify HMAC-SHA256 signature (fallback method).

        Args:
            message: Message bytes to verify
            signature: Base64 encoded signature
            secret_key: Secret key for HMAC

        Returns:
            bool: True if signature valid
        """
        try:
            # Decode signature
            provided_signature = base64.b64decode(signature)

            # Calculate expected signature
            expected_signature = hmac.new(
                secret_key.encode("utf-8"), message, hashlib.sha256
            ).digest()

            # Compare signatures
            if hmac.compare_digest(provided_signature, expected_signature):
                logger.debug("HMAC-SHA256 signature verification successful")
                return True
            else:
                logger.warning("HMAC-SHA256 signature mismatch")
                return False

        except Exception as e:
            logger.error(f"HMAC signature verification failed: {e}")
            return False

    def validate_message_integrity(
        self, message: dict, signature: str, public_key: str
    ) -> bool:
        """
        Validate message integrity with signature.

        Args:
            message: Message dictionary
            signature: Base64 encoded signature
            public_key: Base64 encoded public key

        Returns:
            bool: True if message integrity validated
        """
        try:
            # Create canonical representation
            message_bytes = self._canonicalize_message(message)

            # Verify signature
            if self._ed25519_available:
                return self.verify_ed25519_signature(
                    message_bytes, signature, public_key
                )
            else:
                return self.verify_hmac_signature(message_bytes, signature, public_key)

        except Exception as e:
            logger.error(f"Message integrity validation failed: {e}")
            return False

    def _canonicalize_message(self, message: dict) -> bytes:
        """
        Create canonical byte representation of message.

        Args:
            message: Message dictionary

        Returns:
            bytes: Canonical message bytes
        """
        # Remove signature fields if present
        clean_message = {
            k: v
            for k, v in message.items()
            if k not in ["signature", "public_key", "sig", "key"]
        }

        # Create deterministic JSON
        canonical = json.dumps(
            clean_message, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )

        return canonical.encode("utf-8")

    def add_trusted_public_key(self, key_id: str, public_key: str):
        """
        Add trusted public key.

        Args:
            key_id: Key identifier
            public_key: Base64 encoded public key
        """
        self.public_keys[key_id] = public_key
        self.trusted_key_ids.add(key_id)
        logger.info(f"Added trusted public key: {key_id}")

    def remove_trusted_public_key(self, key_id: str):
        """
        Remove trusted public key.

        Args:
            key_id: Key identifier
        """
        if key_id in self.public_keys:
            del self.public_keys[key_id]
            self.trusted_key_ids.discard(key_id)
            logger.info(f"Removed trusted public key: {key_id}")

    def get_public_key(self, key_id: str) -> Optional[str]:
        """
        Get public key by ID.

        Args:
            key_id: Key identifier

        Returns:
            str: Public key or None if not found
        """
        return self.public_keys.get(key_id)

    def is_trusted_key(self, key_id: str) -> bool:
        """
        Check if key ID is trusted.

        Args:
            key_id: Key identifier

        Returns:
            bool: True if key is trusted
        """
        return key_id in self.trusted_key_ids

    def generate_signature(
        self, message: bytes, private_key: Optional[str] = None
    ) -> Optional[tuple[str, str]]:
        """
        Generate signature for message (for testing).

        Args:
            message: Message to sign
            private_key: Private key (base64 encoded)

        Returns:
            tuple: (signature, public_key) or None if failed
        """
        if not self._ed25519_available:
            # Generate HMAC signature for testing
            secret_key = private_key or "test_secret_key"
            signature = hmac.new(
                secret_key.encode("utf-8"), message, hashlib.sha256
            ).digest()
            return (base64.b64encode(signature).decode("utf-8"), secret_key)

        try:
            import ed25519

            if private_key:
                # Use provided private key
                private_key_bytes = base64.b64decode(private_key)
                signing_key = ed25519.SigningKey(private_key_bytes)
            else:
                # Generate new key pair
                signing_key, verifying_key = ed25519.create_keypair()

            # Sign message
            signature = signing_key.sign(message)
            public_key = signing_key.verifying_key.to_bytes()

            return (
                base64.b64encode(signature).decode("utf-8"),
                base64.b64encode(public_key).decode("utf-8"),
            )

        except Exception as e:
            logger.error(f"Signature generation failed: {e}")
            return None

    def validate_signature_headers(
        self, headers: dict[str, str]
    ) -> tuple[bool, Optional[str]]:
        """
        Validate signature headers from request.

        Args:
            headers: Request headers

        Returns:
            tuple: (is_valid, error_message)
        """
        # Check for required headers
        signature = headers.get("X-ONF-Signature") or headers.get("x-onf-signature")
        public_key = headers.get("X-ONF-Public-Key") or headers.get("x-onf-public-key")
        key_id = headers.get("X-ONF-Key-Id") or headers.get("x-onf-key-id")

        if not signature:
            return False, "Missing X-ONF-Signature header"

        if not public_key and not key_id:
            return False, "Missing X-ONF-Public-Key or X-ONF-Key-Id header"

        # Validate base64 encoding
        try:
            base64.b64decode(signature)
            if public_key:
                base64.b64decode(public_key)
        except (ValueError, TypeError):
            # Base64 decoding failed due to invalid encoding or incorrect data type
            return False, "Invalid base64 encoding in signature headers"

        return True, None
