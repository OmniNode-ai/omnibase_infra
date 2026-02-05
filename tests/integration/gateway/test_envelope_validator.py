# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for ServiceEnvelopeValidator.

Tests verify that:
    - Valid envelope passes validation
    - Invalid signature is rejected (INVALID_SIGNATURE)
    - Missing signature is rejected (MISSING_SIGNATURE)
    - Realm mismatch is rejected (REALM_MISMATCH)
    - Tampered payload is rejected (HASH_MISMATCH -> INVALID_SIGNATURE)
    - Unknown signer is rejected (UNKNOWN_SIGNER)

Related Tickets:
    - OMN-1899: Runtime gateway envelope signing
"""

from __future__ import annotations

import base64

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from omnibase_core.models.envelope.model_envelope_signature import (
    ModelEnvelopeSignature,
)
from omnibase_core.models.envelope.model_message_envelope import ModelMessageEnvelope
from omnibase_infra.gateway import (
    EnumValidationErrorCode,
    ServiceEnvelopeSigner,
    ServiceEnvelopeValidator,
    ValidationResult,
)

from .conftest import ModelTestPayload


class TestEnvelopeValidatorValidEnvelope:
    """Tests for valid envelope validation."""

    def test_valid_envelope_passes_validation(
        self,
        envelope_signer: ServiceEnvelopeSigner,
        envelope_validator: ServiceEnvelopeValidator,
        sample_payload: ModelTestPayload,
    ) -> None:
        """Valid signed envelope passes all validation checks."""
        # Arrange
        envelope = envelope_signer.sign_envelope(
            payload=sample_payload,
            bus_id="test-bus",
        )

        # Act
        result = envelope_validator.validate_envelope(envelope)

        # Assert
        assert result.is_valid is True
        assert result.error_code is None
        assert result.error_message is None

    def test_validation_result_is_truthy_when_valid(
        self,
        envelope_signer: ServiceEnvelopeSigner,
        envelope_validator: ServiceEnvelopeValidator,
        sample_payload: ModelTestPayload,
    ) -> None:
        """ValidationResult is truthy when validation passes."""
        # Arrange
        envelope = envelope_signer.sign_envelope(
            payload=sample_payload,
            bus_id="test-bus",
        )

        # Act
        result = envelope_validator.validate_envelope(envelope)

        # Assert
        assert bool(result) is True
        # Can use in boolean context
        if result:
            validated = True
        else:
            validated = False
        assert validated is True

    def test_validation_result_success_factory(self) -> None:
        """ValidationResult.success() creates valid result."""
        # Act
        result = ValidationResult.success()

        # Assert
        assert result.is_valid is True
        assert result.error_code is None
        assert result.error_message is None
        assert bool(result) is True


class TestEnvelopeValidatorInvalidSignature:
    """Tests for invalid signature detection."""

    def test_tampered_signature_is_rejected(
        self,
        envelope_signer: ServiceEnvelopeSigner,
        envelope_validator: ServiceEnvelopeValidator,
        sample_payload: ModelTestPayload,
    ) -> None:
        """Envelope with tampered signature is rejected."""
        # Arrange
        envelope = envelope_signer.sign_envelope(
            payload=sample_payload,
            bus_id="test-bus",
        )

        # Tamper with the signature
        assert envelope.signature is not None
        tampered_signature = base64.b64encode(b"tampered-signature-data").decode()
        tampered_envelope_sig = ModelEnvelopeSignature(
            algorithm=envelope.signature.algorithm,
            signer=envelope.signature.signer,
            payload_hash=envelope.signature.payload_hash,
            signature=tampered_signature,
        )
        # Create new envelope with tampered signature
        tampered_envelope: ModelMessageEnvelope[ModelTestPayload] = (
            ModelMessageEnvelope[ModelTestPayload](
                realm=envelope.realm,
                runtime_id=envelope.runtime_id,
                bus_id=envelope.bus_id,
                trace_id=envelope.trace_id,
                emitted_at=envelope.emitted_at,
                signature=tampered_envelope_sig,
                payload=envelope.payload,
            )
        )

        # Act
        result = envelope_validator.validate_envelope(tampered_envelope)

        # Assert
        assert result.is_valid is False
        assert result.error_code == EnumValidationErrorCode.INVALID_SIGNATURE
        assert result.error_message is not None

    def test_signature_from_different_key_is_rejected(
        self,
        envelope_validator: ServiceEnvelopeValidator,
        sample_payload: ModelTestPayload,
    ) -> None:
        """Envelope signed with different key is rejected."""
        # Arrange - create a different keypair
        different_private_key = Ed25519PrivateKey.generate()
        different_signer = ServiceEnvelopeSigner(
            realm="test",  # Same realm to pass realm check
            runtime_id="test-runtime-001",  # Same runtime_id to pass signer lookup
            private_key=different_private_key,
        )

        envelope = different_signer.sign_envelope(
            payload=sample_payload,
            bus_id="test-bus",
        )

        # Act - validator has different public key
        result = envelope_validator.validate_envelope(envelope)

        # Assert
        assert result.is_valid is False
        assert result.error_code == EnumValidationErrorCode.INVALID_SIGNATURE


class TestEnvelopeValidatorMissingSignature:
    """Tests for missing signature detection.

    Note: The ModelMessageEnvelope model requires a signature field (not optional).
    These tests verify the validator's configuration properties rather than
    testing actual unsigned envelope handling, since the model enforces
    signature presence at construction time.
    """

    def test_reject_unsigned_configuration_is_respected(
        self,
        ed25519_public_key: Ed25519PublicKey,
    ) -> None:
        """Validator respects reject_unsigned configuration."""
        # Arrange & Assert - verify configuration is stored correctly
        strict_validator = ServiceEnvelopeValidator(
            expected_realm="test",
            public_keys={"test-runtime-001": ed25519_public_key},
            reject_unsigned=True,
        )
        permissive_validator = ServiceEnvelopeValidator(
            expected_realm="test",
            public_keys={"test-runtime-001": ed25519_public_key},
            reject_unsigned=False,
        )

        # Assert configuration is respected
        assert strict_validator.reject_unsigned is True
        assert permissive_validator.reject_unsigned is False

    def test_validator_initialization_with_reject_unsigned_true(
        self,
        ed25519_public_key: Ed25519PublicKey,
    ) -> None:
        """Validator initializes correctly with reject_unsigned=True."""
        # Act
        validator = ServiceEnvelopeValidator(
            expected_realm="test",
            public_keys={"runtime-1": ed25519_public_key},
            reject_unsigned=True,
        )

        # Assert
        assert validator.reject_unsigned is True
        assert validator.expected_realm == "test"
        assert validator.trusted_signer_count == 1

    def test_validator_initialization_with_reject_unsigned_false(
        self,
        ed25519_public_key: Ed25519PublicKey,
    ) -> None:
        """Validator initializes correctly with reject_unsigned=False."""
        # Act
        validator = ServiceEnvelopeValidator(
            expected_realm="test",
            public_keys={"runtime-1": ed25519_public_key},
            reject_unsigned=False,
        )

        # Assert
        assert validator.reject_unsigned is False


class TestEnvelopeValidatorRealmMismatch:
    """Tests for realm mismatch detection."""

    def test_realm_mismatch_is_rejected(
        self,
        envelope_signer: ServiceEnvelopeSigner,
        ed25519_public_key: Ed25519PublicKey,
        sample_payload: ModelTestPayload,
    ) -> None:
        """Envelope with wrong realm is rejected."""
        # Arrange - create validator expecting different realm
        different_realm_validator = ServiceEnvelopeValidator(
            expected_realm="production",  # Different from signer's "test"
            public_keys={"test-runtime-001": ed25519_public_key},
            reject_unsigned=True,
        )

        envelope = envelope_signer.sign_envelope(
            payload=sample_payload,
            bus_id="test-bus",
        )

        # Act
        result = different_realm_validator.validate_envelope(envelope)

        # Assert
        assert result.is_valid is False
        assert result.error_code == EnumValidationErrorCode.REALM_MISMATCH
        assert result.error_message is not None
        assert "production" in result.error_message
        assert "test" in result.error_message

    def test_realm_check_happens_before_signature_check(
        self,
        ed25519_public_key: Ed25519PublicKey,
        sample_payload: ModelTestPayload,
    ) -> None:
        """Realm mismatch is detected before signature validation."""
        # Arrange - envelope with wrong realm and invalid signature
        from datetime import UTC, datetime
        from uuid import uuid4

        validator = ServiceEnvelopeValidator(
            expected_realm="production",
            public_keys={"test-runtime-001": ed25519_public_key},
            reject_unsigned=True,
        )

        # Use a valid-looking hash (64 hex chars) even though it's fake
        # to pass model validation - realm check happens before hash verification
        fake_hash = "a" * 64
        fake_signature = base64.b64encode(b"invalid-signature-data").decode()

        wrong_realm_envelope: ModelMessageEnvelope[ModelTestPayload] = (
            ModelMessageEnvelope[ModelTestPayload](
                realm="wrong-realm",  # Wrong realm
                runtime_id="test-runtime-001",
                bus_id="test-bus",
                trace_id=uuid4(),
                emitted_at=datetime.now(UTC),
                signature=ModelEnvelopeSignature(
                    algorithm="ed25519",
                    signer="test-runtime-001",
                    payload_hash=fake_hash,
                    signature=fake_signature,
                ),
                payload=sample_payload,
            )
        )

        # Act
        result = validator.validate_envelope(wrong_realm_envelope)

        # Assert - should fail on realm, not signature
        assert result.is_valid is False
        assert result.error_code == EnumValidationErrorCode.REALM_MISMATCH


class TestEnvelopeValidatorUnknownSigner:
    """Tests for unknown signer detection."""

    def test_unknown_signer_is_rejected(
        self,
        ed25519_public_key: Ed25519PublicKey,
        sample_payload: ModelTestPayload,
    ) -> None:
        """Envelope from unknown signer is rejected."""
        # Arrange - validator with different signer registered
        validator = ServiceEnvelopeValidator(
            expected_realm="test",
            public_keys={"known-runtime": ed25519_public_key},  # Different signer
            reject_unsigned=True,
        )

        # Create envelope from unknown signer
        unknown_private_key = Ed25519PrivateKey.generate()
        unknown_signer = ServiceEnvelopeSigner(
            realm="test",
            runtime_id="unknown-runtime",  # Not in validator's public_keys
            private_key=unknown_private_key,
        )

        envelope = unknown_signer.sign_envelope(
            payload=sample_payload,
            bus_id="test-bus",
        )

        # Act
        result = validator.validate_envelope(envelope)

        # Assert
        assert result.is_valid is False
        assert result.error_code == EnumValidationErrorCode.UNKNOWN_SIGNER
        assert result.error_message is not None
        assert "unknown-runtime" in result.error_message


class TestEnvelopeValidatorTamperedPayload:
    """Tests for tampered payload detection."""

    def test_tampered_payload_is_rejected(
        self,
        envelope_signer: ServiceEnvelopeSigner,
        envelope_validator: ServiceEnvelopeValidator,
        sample_payload: ModelTestPayload,
    ) -> None:
        """Envelope with tampered payload is rejected due to hash mismatch."""
        # Arrange
        envelope = envelope_signer.sign_envelope(
            payload=sample_payload,
            bus_id="test-bus",
        )

        # Create envelope with different payload but original signature
        tampered_payload = ModelTestPayload(
            action="tampered",  # Changed
            resource_id="resource-123",
            data={"key1": "value1", "key2": "value2"},
        )

        assert envelope.signature is not None
        tampered_envelope: ModelMessageEnvelope[ModelTestPayload] = (
            ModelMessageEnvelope[ModelTestPayload](
                realm=envelope.realm,
                runtime_id=envelope.runtime_id,
                bus_id=envelope.bus_id,
                trace_id=envelope.trace_id,
                emitted_at=envelope.emitted_at,
                signature=envelope.signature,  # Original signature
                payload=tampered_payload,  # Tampered payload
            )
        )

        # Act
        result = envelope_validator.validate_envelope(tampered_envelope)

        # Assert
        assert result.is_valid is False
        assert result.error_code == EnumValidationErrorCode.HASH_MISMATCH


class TestEnvelopeValidatorSignerManagement:
    """Tests for dynamic signer management."""

    def test_add_trusted_signer(
        self,
        envelope_validator: ServiceEnvelopeValidator,
    ) -> None:
        """Can add trusted signer dynamically."""
        # Arrange
        new_private_key = Ed25519PrivateKey.generate()
        new_public_key = new_private_key.public_key()

        # Act
        envelope_validator.add_trusted_signer("new-runtime", new_public_key)

        # Assert
        assert envelope_validator.is_trusted_signer("new-runtime") is True
        assert envelope_validator.trusted_signer_count >= 2  # Original + new

    def test_remove_trusted_signer(
        self,
        envelope_validator: ServiceEnvelopeValidator,
    ) -> None:
        """Can remove trusted signer dynamically."""
        # Arrange
        new_private_key = Ed25519PrivateKey.generate()
        new_public_key = new_private_key.public_key()
        envelope_validator.add_trusted_signer("temp-runtime", new_public_key)

        # Act
        removed = envelope_validator.remove_trusted_signer("temp-runtime")

        # Assert
        assert removed is True
        assert envelope_validator.is_trusted_signer("temp-runtime") is False

    def test_remove_nonexistent_signer_returns_false(
        self,
        envelope_validator: ServiceEnvelopeValidator,
    ) -> None:
        """Removing nonexistent signer returns False."""
        # Act
        removed = envelope_validator.remove_trusted_signer("nonexistent-runtime")

        # Assert
        assert removed is False

    def test_is_trusted_signer(
        self,
        envelope_validator: ServiceEnvelopeValidator,
    ) -> None:
        """Can check if signer is trusted."""
        # Assert
        assert envelope_validator.is_trusted_signer("test-runtime-001") is True
        assert envelope_validator.is_trusted_signer("unknown-runtime") is False


class TestEnvelopeValidatorProperties:
    """Tests for validator property accessors."""

    def test_expected_realm_property(
        self,
        envelope_validator: ServiceEnvelopeValidator,
    ) -> None:
        """Expected realm property returns configured realm."""
        assert envelope_validator.expected_realm == "test"

    def test_reject_unsigned_property(
        self,
        envelope_validator: ServiceEnvelopeValidator,
    ) -> None:
        """Reject unsigned property returns configured value."""
        assert envelope_validator.reject_unsigned is True

    def test_trusted_signer_count_property(
        self,
        envelope_validator: ServiceEnvelopeValidator,
    ) -> None:
        """Trusted signer count returns number of registered signers."""
        assert envelope_validator.trusted_signer_count >= 1


class TestValidationResultFactory:
    """Tests for ValidationResult factory methods."""

    def test_failure_factory_creates_invalid_result(self) -> None:
        """ValidationResult.failure() creates invalid result with details."""
        # Act
        result = ValidationResult.failure(
            error_code=EnumValidationErrorCode.REALM_MISMATCH,
            error_message="Test error message",
        )

        # Assert
        assert result.is_valid is False
        assert result.error_code == EnumValidationErrorCode.REALM_MISMATCH
        assert result.error_message == "Test error message"
        assert bool(result) is False

    def test_validation_result_is_falsy_when_invalid(self) -> None:
        """ValidationResult is falsy when validation fails."""
        # Arrange
        result = ValidationResult.failure(
            error_code=EnumValidationErrorCode.INVALID_SIGNATURE,
            error_message="Signature invalid",
        )

        # Assert
        assert bool(result) is False
        # Can use in boolean context
        if result:
            validated = True
        else:
            validated = False
        assert validated is False
