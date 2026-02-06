# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Envelope validation service for runtime gateway.

This module provides a service for validating inbound envelope signatures and
enforcing realm boundaries. It uses Ed25519 signature verification and Blake3
hashing from omnibase_core crypto utilities.

The validator performs the following checks:
    1. Realm matching - envelope realm must match expected gateway realm
    2. Signer lookup - signer must be in trusted public keys registry
    3. Hash verification - recomputed payload hash must match signature hash
    4. Signature verification - Ed25519 signature must be valid

Exports:
    ServiceEnvelopeValidator: Service for validating inbound envelope signatures and realm.
    ValidationResult: Result of envelope validation with error details.

Example:
    >>> from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    >>> from omnibase_core.models.envelope.model_message_envelope import ModelMessageEnvelope
    >>>
    >>> # Create validator with trusted signers
    >>> validator = ServiceEnvelopeValidator(
    ...     expected_realm="production",
    ...     public_keys={"runtime-1": public_key_1, "runtime-2": public_key_2},
    ...     reject_unsigned=True,
    ... )
    >>>
    >>> # Validate an envelope
    >>> result = validator.validate_envelope(envelope)
    >>> if result.is_valid:
    ...     print("Envelope is valid")
    ... else:
    ...     print(f"Validation failed: {result.error_code} - {result.error_message}")

"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

from omnibase_core.crypto.crypto_blake3_hasher import hash_canonical_json
from omnibase_core.crypto.crypto_ed25519_signer import InvalidSignature, verify_base64
from omnibase_infra.gateway.enums import EnumValidationErrorCode

if TYPE_CHECKING:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    from omnibase_core.models.envelope.model_message_envelope import (
        ModelMessageEnvelope,
    )

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationResult:
    """Result of envelope validation.

    This immutable dataclass captures the outcome of envelope validation,
    including success/failure status and detailed error information when
    validation fails.

    Attributes:
        is_valid: True if envelope passed all validation checks.
        error_code: Error code from EnumValidationErrorCode when validation fails.
            None when is_valid is True.
        error_message: Human-readable error description when validation fails.
            None when is_valid is True.

    Example:
        >>> # Successful validation
        >>> result = ValidationResult(is_valid=True)
        >>> bool(result)  # True
        >>>
        >>> # Failed validation
        >>> result = ValidationResult(
        ...     is_valid=False,
        ...     error_code=EnumValidationErrorCode.REALM_MISMATCH,
        ...     error_message="Expected realm 'production', got 'staging'",
        ... )
        >>> bool(result)  # False

    """

    is_valid: bool
    error_code: EnumValidationErrorCode | None = None
    error_message: str | None = None

    def __bool__(self) -> bool:
        """Allow using result in boolean context.

        Warning:
            **Non-standard __bool__ behavior**: Returns ``True`` only when
            ``is_valid`` is True. Differs from typical dataclass behavior
            where instances are always truthy.

        Returns:
            True if validation passed, False otherwise.

        """
        return self.is_valid

    @classmethod
    def success(cls) -> ValidationResult:
        """Create a successful validation result.

        Returns:
            ValidationResult with is_valid=True and no error details.

        """
        return cls(is_valid=True)

    @classmethod
    def failure(
        cls,
        error_code: EnumValidationErrorCode,
        error_message: str,
    ) -> ValidationResult:
        """Create a failed validation result.

        Args:
            error_code: The specific error code identifying the failure.
            error_message: Human-readable description of the failure.

        Returns:
            ValidationResult with is_valid=False and error details.

        """
        return cls(
            is_valid=False,
            error_code=error_code,
            error_message=error_message,
        )


class ServiceEnvelopeValidator:
    """Service for validating inbound envelope signatures and realm.

    This service validates inbound envelopes at runtime boundaries by:
        1. Checking realm matches the expected gateway realm
        2. Looking up the signer's public key in the trusted registry
        3. Recomputing the payload hash using Blake3
        4. Verifying the Ed25519 signature

    The validator maintains a registry of trusted signers identified by their
    runtime_id and their corresponding Ed25519 public keys.

    Attributes:
        expected_realm: The realm this gateway expects envelopes to belong to.
        reject_unsigned: If True, envelopes without signatures are rejected.
            Defaults to True for secure-by-default behavior.

    Example:
        >>> from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        >>>
        >>> validator = ServiceEnvelopeValidator(
        ...     expected_realm="production",
        ...     public_keys={"runtime-1": public_key},
        ...     reject_unsigned=True,
        ... )
        >>>
        >>> result = validator.validate_envelope(envelope)
        >>> if not result:
        ...     print(f"Validation failed: {result.error_code}")

    Security Considerations:
        - Always enable reject_unsigned in production environments
        - Regularly rotate and audit trusted public keys
        - Log validation failures for security monitoring

    """

    def __init__(
        self,
        expected_realm: str,
        public_keys: dict[str, Ed25519PublicKey] | None = None,
        reject_unsigned: bool = True,
    ) -> None:
        """Initialize the envelope validator.

        Args:
            expected_realm: The realm this gateway expects all inbound envelopes
                to belong to. Envelopes with a different realm are rejected.
            public_keys: Dictionary mapping runtime_id to Ed25519PublicKey for
                trusted signers. Defaults to empty dict if not provided.
            reject_unsigned: If True (default), envelopes without signatures
                are rejected with MISSING_SIGNATURE error. Set to False only
                for development or internal-only deployments.

        Example:
            >>> validator = ServiceEnvelopeValidator(
            ...     expected_realm="production",
            ...     public_keys={"runtime-1": pub_key_1, "runtime-2": pub_key_2},
            ... )

        """
        self._expected_realm = expected_realm
        self._public_keys: dict[str, Ed25519PublicKey] = dict(public_keys or {})
        self._public_keys_lock = threading.Lock()
        self._reject_unsigned = reject_unsigned

        logger.debug(
            "ServiceEnvelopeValidator initialized",
            extra={
                "expected_realm": expected_realm,
                "trusted_signers": list(self._public_keys.keys()),
                "reject_unsigned": reject_unsigned,
            },
        )

    @property
    def expected_realm(self) -> str:
        """Return the expected realm for this gateway.

        Returns:
            The realm string this gateway expects in inbound envelopes.

        """
        return self._expected_realm

    @property
    def reject_unsigned(self) -> bool:
        """Return whether unsigned envelopes are rejected.

        Returns:
            True if unsigned envelopes are rejected, False otherwise.

        """
        return self._reject_unsigned

    @property
    def trusted_signer_count(self) -> int:
        """Return the number of trusted signers registered.

        Returns:
            Count of runtime_ids in the trusted public keys registry.

        """
        with self._public_keys_lock:
            return len(self._public_keys)

    def validate_envelope(
        self,
        envelope: ModelMessageEnvelope,
    ) -> ValidationResult:
        """Validate envelope signature and realm.

        Performs the following validation steps in order:
            1. Check realm matches expected_realm
            2. Check signature exists (if reject_unsigned is True)
            3. Look up signer's public key
            4. Recompute payload hash using Blake3
            5. Verify hash matches signature's payload_hash
            6. Verify Ed25519 signature

        Args:
            envelope: The inbound envelope to validate. Must have realm set.
                If signed, must have signature with signer, payload_hash,
                and signature fields.

        Returns:
            ValidationResult indicating success or failure with error details.

        Error Codes:
            - REALM_MISMATCH: envelope.realm != expected_realm
            - MISSING_SIGNATURE: envelope.signature is None and reject_unsigned
            - UNKNOWN_SIGNER: signer not in public_keys registry
            - HASH_MISMATCH: recomputed hash != signature.payload_hash
            - INVALID_SIGNATURE: Ed25519 verification failed

        Example:
            >>> result = validator.validate_envelope(envelope)
            >>> if result.is_valid:
            ...     process_envelope(envelope)
            ... else:
            ...     log_validation_failure(result.error_code, result.error_message)

        """
        # Extract correlation_id from envelope trace_id or auto-generate
        correlation_id = str(envelope.trace_id) if envelope.trace_id else str(uuid4())

        # Step 1: Check realm matches
        if envelope.realm != self._expected_realm:
            error_msg = (
                f"Expected realm '{self._expected_realm}', got '{envelope.realm}'"
            )
            logger.warning(
                "Envelope realm mismatch",
                extra={
                    "correlation_id": correlation_id,
                    "expected_realm": self._expected_realm,
                    "actual_realm": envelope.realm,
                    "runtime_id": envelope.runtime_id,
                    "trace_id": str(envelope.trace_id) if envelope.trace_id else None,
                },
            )
            return ValidationResult.failure(
                error_code=EnumValidationErrorCode.REALM_MISMATCH,
                error_message=error_msg,
            )

        # Step 2: Check signature exists (if required)
        if envelope.signature is None:
            if self._reject_unsigned:
                error_msg = "Envelope has no signature and reject_unsigned is enabled"
                logger.warning(
                    "Missing envelope signature",
                    extra={
                        "correlation_id": correlation_id,
                        "runtime_id": envelope.runtime_id,
                        "trace_id": (
                            str(envelope.trace_id) if envelope.trace_id else None
                        ),
                    },
                )
                return ValidationResult.failure(
                    error_code=EnumValidationErrorCode.MISSING_SIGNATURE,
                    error_message=error_msg,
                )
            # Unsigned envelope is allowed
            logger.debug(
                "Accepting unsigned envelope (reject_unsigned=False)",
                extra={
                    "correlation_id": correlation_id,
                    "runtime_id": envelope.runtime_id,
                    "trace_id": str(envelope.trace_id) if envelope.trace_id else None,
                },
            )
            return ValidationResult.success()

        # Step 3: Look up signer's public key
        signer = envelope.signature.signer
        with self._public_keys_lock:
            public_key = self._public_keys.get(signer)
            trusted_signers_snapshot = list(self._public_keys.keys())
        if public_key is None:
            error_msg = f"Unknown signer '{signer}' not in trusted public keys"
            logger.warning(
                "Unknown signer in envelope",
                extra={
                    "correlation_id": correlation_id,
                    "signer": signer,
                    "runtime_id": envelope.runtime_id,
                    "trace_id": str(envelope.trace_id) if envelope.trace_id else None,
                    "trusted_signers": trusted_signers_snapshot,
                },
            )
            return ValidationResult.failure(
                error_code=EnumValidationErrorCode.UNKNOWN_SIGNER,
                error_message=error_msg,
            )

        # Step 4: Recompute payload hash
        # The payload needs to be serialized to canonical JSON for hashing
        # envelope.payload is already a dict-like structure
        payload_dict = envelope.payload
        if hasattr(payload_dict, "model_dump"):
            # Pydantic model - convert to dict
            payload_dict = payload_dict.model_dump(mode="json")

        computed_hash = hash_canonical_json(payload_dict)

        # Step 5: Verify hash matches
        if computed_hash != envelope.signature.payload_hash:
            error_msg = (
                f"Payload hash mismatch: expected '{envelope.signature.payload_hash}', "
                f"computed '{computed_hash}'"
            )
            logger.warning(
                "Envelope payload hash mismatch",
                extra={
                    "correlation_id": correlation_id,
                    "signer": signer,
                    "expected_hash": envelope.signature.payload_hash,
                    "computed_hash": computed_hash,
                    "runtime_id": envelope.runtime_id,
                    "trace_id": str(envelope.trace_id) if envelope.trace_id else None,
                },
            )
            return ValidationResult.failure(
                error_code=EnumValidationErrorCode.HASH_MISMATCH,
                error_message=error_msg,
            )

        # Step 6: Verify Ed25519 signature
        # The message that was signed is the payload hash
        message = envelope.signature.payload_hash.encode("utf-8")
        signature_b64 = envelope.signature.signature

        try:
            # Get raw public key bytes for verification
            public_key_bytes = public_key.public_bytes_raw()
            is_valid = verify_base64(public_key_bytes, message, signature_b64)
            if not is_valid:
                error_msg = "Ed25519 signature verification returned False"
                logger.warning(
                    "Envelope signature verification failed",
                    extra={
                        "correlation_id": correlation_id,
                        "signer": signer,
                        "runtime_id": envelope.runtime_id,
                        "trace_id": (
                            str(envelope.trace_id) if envelope.trace_id else None
                        ),
                    },
                )
                return ValidationResult.failure(
                    error_code=EnumValidationErrorCode.INVALID_SIGNATURE,
                    error_message=error_msg,
                )
        except InvalidSignature as e:
            error_msg = f"Ed25519 signature verification raised InvalidSignature: {e}"
            logger.warning(
                "Envelope signature invalid",
                extra={
                    "correlation_id": correlation_id,
                    "signer": signer,
                    "runtime_id": envelope.runtime_id,
                    "trace_id": str(envelope.trace_id) if envelope.trace_id else None,
                    "error": str(e),
                },
            )
            return ValidationResult.failure(
                error_code=EnumValidationErrorCode.INVALID_SIGNATURE,
                error_message=error_msg,
            )
        except Exception as e:
            # Catch any other crypto errors
            error_msg = f"Signature verification error: {type(e).__name__}: {e}"
            logger.warning(
                "Envelope signature verification error",
                extra={
                    "correlation_id": correlation_id,
                    "signer": signer,
                    "runtime_id": envelope.runtime_id,
                    "trace_id": str(envelope.trace_id) if envelope.trace_id else None,
                    "error_type": type(e).__name__,
                    "error": str(e),
                },
            )
            return ValidationResult.failure(
                error_code=EnumValidationErrorCode.INVALID_SIGNATURE,
                error_message=error_msg,
            )

        # All validations passed
        logger.debug(
            "Envelope validation successful",
            extra={
                "correlation_id": correlation_id,
                "signer": signer,
                "runtime_id": envelope.runtime_id,
                "trace_id": str(envelope.trace_id) if envelope.trace_id else None,
            },
        )
        return ValidationResult.success()

    def add_trusted_signer(
        self,
        runtime_id: str,
        public_key: Ed25519PublicKey,
    ) -> None:
        """Add a trusted signer's public key to the registry.

        Registers a new trusted signer or updates an existing signer's public
        key. This enables dynamic addition of trusted runtimes without
        recreating the validator.

        Args:
            runtime_id: The unique identifier for the runtime.
            public_key: The Ed25519 public key for signature verification.

        Example:
            >>> from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            ...     Ed25519PrivateKey,
            ... )
            >>>
            >>> # Generate a key pair
            >>> private_key = Ed25519PrivateKey.generate()
            >>> public_key = private_key.public_key()
            >>>
            >>> # Add to validator
            >>> validator.add_trusted_signer("new-runtime", public_key)

        Note:
            If runtime_id already exists, the public key is replaced.
            This can be used for key rotation scenarios.

        """
        with self._public_keys_lock:
            is_update = runtime_id in self._public_keys
            self._public_keys[runtime_id] = public_key
            total_signers = len(self._public_keys)

        logger.info(
            "Trusted signer %s: %s",
            "updated" if is_update else "added",
            runtime_id,
            extra={
                "runtime_id": runtime_id,
                "action": "update" if is_update else "add",
                "total_signers": total_signers,
            },
        )

    def remove_trusted_signer(self, runtime_id: str) -> bool:
        """Remove a trusted signer from the registry.

        Removes a runtime from the trusted signers registry. Envelopes signed
        by this runtime will be rejected with UNKNOWN_SIGNER error.

        Args:
            runtime_id: The unique identifier for the runtime to remove.

        Returns:
            True if the signer was removed, False if not found.

        Example:
            >>> removed = validator.remove_trusted_signer("decommissioned-runtime")
            >>> if removed:
            ...     print("Signer removed successfully")

        """
        with self._public_keys_lock:
            if runtime_id in self._public_keys:
                del self._public_keys[runtime_id]
                total_signers = len(self._public_keys)
                removed = True
            else:
                removed = False

        if removed:
            logger.info(
                "Trusted signer removed: %s",
                runtime_id,
                extra={
                    "runtime_id": runtime_id,
                    "total_signers": total_signers,
                },
            )
        return removed

    def is_trusted_signer(self, runtime_id: str) -> bool:
        """Check if a runtime_id is a trusted signer.

        Args:
            runtime_id: The runtime identifier to check.

        Returns:
            True if the runtime_id has a registered public key.

        """
        with self._public_keys_lock:
            return runtime_id in self._public_keys


__all__: list[str] = ["ServiceEnvelopeValidator", "ValidationResult"]
