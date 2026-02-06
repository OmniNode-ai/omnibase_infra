# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for ServiceEnvelopeSigner.

Tests verify that:
    - Signing a payload produces valid ModelMessageEnvelope
    - Signature contains correct realm, runtime_id, bus_id
    - Payload hash is deterministic (same payload = same hash)
    - Signature is verifiable with public key

Related Tickets:
    - OMN-1899: Runtime gateway envelope signing
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from omnibase_core.crypto.crypto_blake3_hasher import hash_canonical_json
from omnibase_core.crypto.crypto_ed25519_signer import verify_base64
from omnibase_core.models.envelope.model_message_envelope import ModelMessageEnvelope
from omnibase_infra.gateway import ServiceEnvelopeSigner

from .conftest import ModelTestPayload

pytestmark = pytest.mark.integration


class TestEnvelopeSignerBasic:
    """Basic tests for ServiceEnvelopeSigner functionality."""

    def test_sign_envelope_returns_model_message_envelope(
        self,
        envelope_signer: ServiceEnvelopeSigner,
        sample_payload: ModelTestPayload,
    ) -> None:
        """Signing a payload produces a valid ModelMessageEnvelope."""
        # Arrange
        bus_id = "test-bus-001"

        # Act
        envelope = envelope_signer.sign_envelope(
            payload=sample_payload,
            bus_id=bus_id,
        )

        # Assert
        assert isinstance(envelope, ModelMessageEnvelope)
        assert envelope.payload == sample_payload
        assert envelope.signature is not None

    def test_sign_envelope_sets_realm_correctly(
        self,
        envelope_signer: ServiceEnvelopeSigner,
        sample_payload: ModelTestPayload,
    ) -> None:
        """Signature contains correct realm from signer configuration."""
        # Arrange
        bus_id = "test-bus-001"

        # Act
        envelope = envelope_signer.sign_envelope(
            payload=sample_payload,
            bus_id=bus_id,
        )

        # Assert
        assert envelope.realm == envelope_signer.realm
        assert envelope.realm == "test"

    def test_sign_envelope_sets_runtime_id_correctly(
        self,
        envelope_signer: ServiceEnvelopeSigner,
        sample_payload: ModelTestPayload,
    ) -> None:
        """Signature contains correct runtime_id from signer configuration."""
        # Arrange
        bus_id = "test-bus-001"

        # Act
        envelope = envelope_signer.sign_envelope(
            payload=sample_payload,
            bus_id=bus_id,
        )

        # Assert
        assert envelope.runtime_id == envelope_signer.runtime_id
        assert envelope.runtime_id == "test-runtime-001"
        # Signer in signature should match runtime_id
        assert envelope.signature is not None
        assert envelope.signature.signer == envelope_signer.runtime_id

    def test_sign_envelope_sets_bus_id_correctly(
        self,
        envelope_signer: ServiceEnvelopeSigner,
        sample_payload: ModelTestPayload,
    ) -> None:
        """Envelope contains the bus_id passed to sign_envelope."""
        # Arrange
        bus_id = "custom-bus-id-123"

        # Act
        envelope = envelope_signer.sign_envelope(
            payload=sample_payload,
            bus_id=bus_id,
        )

        # Assert
        assert envelope.bus_id == bus_id

    def test_sign_envelope_generates_trace_id_when_not_provided(
        self,
        envelope_signer: ServiceEnvelopeSigner,
        sample_payload: ModelTestPayload,
    ) -> None:
        """Trace ID is auto-generated when not provided."""
        # Arrange
        bus_id = "test-bus-001"

        # Act
        envelope = envelope_signer.sign_envelope(
            payload=sample_payload,
            bus_id=bus_id,
        )

        # Assert
        assert envelope.trace_id is not None
        assert isinstance(envelope.trace_id, UUID)

    def test_sign_envelope_uses_provided_trace_id(
        self,
        envelope_signer: ServiceEnvelopeSigner,
        sample_payload: ModelTestPayload,
    ) -> None:
        """Trace ID from argument is used when provided."""
        # Arrange
        bus_id = "test-bus-001"
        trace_id = uuid4()

        # Act
        envelope = envelope_signer.sign_envelope(
            payload=sample_payload,
            bus_id=bus_id,
            trace_id=trace_id,
        )

        # Assert
        assert envelope.trace_id == trace_id

    def test_sign_envelope_sets_algorithm_to_ed25519(
        self,
        envelope_signer: ServiceEnvelopeSigner,
        sample_payload: ModelTestPayload,
    ) -> None:
        """Signature algorithm is always 'ed25519'."""
        # Arrange
        bus_id = "test-bus-001"

        # Act
        envelope = envelope_signer.sign_envelope(
            payload=sample_payload,
            bus_id=bus_id,
        )

        # Assert
        assert envelope.signature is not None
        assert envelope.signature.algorithm == "ed25519"


class TestEnvelopeSignerHashDeterminism:
    """Tests for payload hash determinism."""

    def test_same_payload_produces_same_hash(
        self,
        envelope_signer: ServiceEnvelopeSigner,
    ) -> None:
        """Same payload should produce the same hash every time."""
        # Arrange
        payload1 = ModelTestPayload(
            action="created",
            resource_id="resource-abc",
            data={"key": "value"},
        )
        payload2 = ModelTestPayload(
            action="created",
            resource_id="resource-abc",
            data={"key": "value"},
        )
        bus_id = "test-bus"

        # Act
        envelope1 = envelope_signer.sign_envelope(payload=payload1, bus_id=bus_id)
        envelope2 = envelope_signer.sign_envelope(payload=payload2, bus_id=bus_id)

        # Assert
        assert envelope1.signature is not None
        assert envelope2.signature is not None
        assert envelope1.signature.payload_hash == envelope2.signature.payload_hash

    def test_different_payload_produces_different_hash(
        self,
        envelope_signer: ServiceEnvelopeSigner,
    ) -> None:
        """Different payloads should produce different hashes."""
        # Arrange
        payload1 = ModelTestPayload(
            action="created",
            resource_id="resource-abc",
            data={"key": "value1"},
        )
        payload2 = ModelTestPayload(
            action="created",
            resource_id="resource-abc",
            data={"key": "value2"},  # Different value
        )
        bus_id = "test-bus"

        # Act
        envelope1 = envelope_signer.sign_envelope(payload=payload1, bus_id=bus_id)
        envelope2 = envelope_signer.sign_envelope(payload=payload2, bus_id=bus_id)

        # Assert
        assert envelope1.signature is not None
        assert envelope2.signature is not None
        assert envelope1.signature.payload_hash != envelope2.signature.payload_hash

    def test_payload_hash_matches_blake3_canonical_json(
        self,
        envelope_signer: ServiceEnvelopeSigner,
        sample_payload: ModelTestPayload,
    ) -> None:
        """Payload hash should match independently computed Blake3 hash."""
        # Arrange
        bus_id = "test-bus"
        payload_dict = sample_payload.model_dump(mode="json")

        # Act
        envelope = envelope_signer.sign_envelope(
            payload=sample_payload,
            bus_id=bus_id,
        )
        expected_hash = hash_canonical_json(payload_dict)

        # Assert
        assert envelope.signature is not None
        assert envelope.signature.payload_hash == expected_hash


class TestEnvelopeSignerSignatureVerification:
    """Tests for signature verification with public key."""

    def test_signature_verifiable_with_matching_public_key(
        self,
        envelope_signer: ServiceEnvelopeSigner,
        ed25519_public_key: Ed25519PublicKey,
        sample_payload: ModelTestPayload,
    ) -> None:
        """Signature should be verifiable with the matching public key."""
        # Arrange
        bus_id = "test-bus"

        # Act
        envelope = envelope_signer.sign_envelope(
            payload=sample_payload,
            bus_id=bus_id,
        )

        # Verify signature manually
        assert envelope.signature is not None
        public_key_bytes = ed25519_public_key.public_bytes_raw()
        message = envelope.signature.payload_hash.encode("utf-8")
        signature_b64 = envelope.signature.signature

        # Assert
        is_valid = verify_base64(public_key_bytes, message, signature_b64)
        assert is_valid is True

    def test_signature_not_verifiable_with_different_public_key(
        self,
        envelope_signer: ServiceEnvelopeSigner,
        sample_payload: ModelTestPayload,
    ) -> None:
        """Signature should NOT be verifiable with a different public key."""
        # Arrange
        bus_id = "test-bus"
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        different_key = Ed25519PrivateKey.generate()
        different_public_key = different_key.public_key()

        # Act
        envelope = envelope_signer.sign_envelope(
            payload=sample_payload,
            bus_id=bus_id,
        )

        # Verify signature with different key
        assert envelope.signature is not None
        public_key_bytes = different_public_key.public_bytes_raw()
        message = envelope.signature.payload_hash.encode("utf-8")
        signature_b64 = envelope.signature.signature

        # Assert - verification should fail
        is_valid = verify_base64(public_key_bytes, message, signature_b64)
        assert is_valid is False


class TestEnvelopeSignerOptionalFields:
    """Tests for optional envelope fields."""

    def test_sign_envelope_with_causality_id(
        self,
        envelope_signer: ServiceEnvelopeSigner,
        sample_payload: ModelTestPayload,
    ) -> None:
        """Causality ID is set when provided."""
        # Arrange
        bus_id = "test-bus"
        causality_id = uuid4()

        # Act
        envelope = envelope_signer.sign_envelope(
            payload=sample_payload,
            bus_id=bus_id,
            causality_id=causality_id,
        )

        # Assert
        assert envelope.causality_id == causality_id

    def test_sign_envelope_with_tenant_id(
        self,
        envelope_signer: ServiceEnvelopeSigner,
        sample_payload: ModelTestPayload,
    ) -> None:
        """Tenant ID is set when provided."""
        # Arrange
        bus_id = "test-bus"
        tenant_id = "tenant-abc-123"

        # Act
        envelope = envelope_signer.sign_envelope(
            payload=sample_payload,
            bus_id=bus_id,
            tenant_id=tenant_id,
        )

        # Assert
        assert envelope.tenant_id == tenant_id

    def test_sign_envelope_without_optional_fields(
        self,
        envelope_signer: ServiceEnvelopeSigner,
        sample_payload: ModelTestPayload,
    ) -> None:
        """Optional fields default to None when not provided."""
        # Arrange
        bus_id = "test-bus"

        # Act
        envelope = envelope_signer.sign_envelope(
            payload=sample_payload,
            bus_id=bus_id,
        )

        # Assert
        assert envelope.causality_id is None
        assert envelope.tenant_id is None
        assert envelope.emitter_identity is None


class TestEnvelopeSignerProperties:
    """Tests for signer property accessors."""

    def test_realm_property_returns_configured_realm(
        self,
        envelope_signer: ServiceEnvelopeSigner,
    ) -> None:
        """Realm property returns the configured realm."""
        assert envelope_signer.realm == "test"

    def test_runtime_id_property_returns_configured_runtime_id(
        self,
        envelope_signer: ServiceEnvelopeSigner,
    ) -> None:
        """Runtime ID property returns the configured runtime_id."""
        assert envelope_signer.runtime_id == "test-runtime-001"


class TestEnvelopeSignerSignDict:
    """Tests for sign_dict method (dict payload signing).

    Related Tickets:
        - OMN-1899: Runtime gateway envelope signing for dict payloads
    """

    def test_sign_dict_returns_model_message_envelope(
        self,
        envelope_signer: ServiceEnvelopeSigner,
    ) -> None:
        """Signing a dict payload produces a valid ModelMessageEnvelope."""
        # Arrange
        bus_id = "test-bus-001"
        payload = {"action": "created", "resource_id": "123", "value": 42}

        # Act
        envelope = envelope_signer.sign_dict(
            payload=payload,
            bus_id=bus_id,
        )

        # Assert
        assert isinstance(envelope, ModelMessageEnvelope)
        assert envelope.payload == payload
        assert envelope.signature is not None

    def test_sign_dict_sets_realm_correctly(
        self,
        envelope_signer: ServiceEnvelopeSigner,
    ) -> None:
        """Dict signature contains correct realm from signer configuration."""
        # Arrange
        bus_id = "test-bus-001"
        payload = {"key": "value"}

        # Act
        envelope = envelope_signer.sign_dict(
            payload=payload,
            bus_id=bus_id,
        )

        # Assert
        assert envelope.realm == envelope_signer.realm
        assert envelope.realm == "test"

    def test_sign_dict_generates_trace_id_when_not_provided(
        self,
        envelope_signer: ServiceEnvelopeSigner,
    ) -> None:
        """Trace ID is auto-generated when not provided."""
        # Arrange
        bus_id = "test-bus"
        payload = {"key": "value"}

        # Act
        envelope = envelope_signer.sign_dict(
            payload=payload,
            bus_id=bus_id,
        )

        # Assert
        assert envelope.trace_id is not None
        assert isinstance(envelope.trace_id, UUID)

    def test_sign_dict_uses_provided_trace_id(
        self,
        envelope_signer: ServiceEnvelopeSigner,
    ) -> None:
        """Trace ID is set correctly when provided."""
        # Arrange
        bus_id = "test-bus"
        trace_id = uuid4()
        payload = {"key": "value"}

        # Act
        envelope = envelope_signer.sign_dict(
            payload=payload,
            bus_id=bus_id,
            trace_id=trace_id,
        )

        # Assert
        assert envelope.trace_id == trace_id

    def test_sign_dict_hash_is_deterministic(
        self,
        envelope_signer: ServiceEnvelopeSigner,
    ) -> None:
        """Same dict payload produces same hash (deterministic)."""
        # Arrange
        bus_id = "test-bus"
        payload = {"action": "test", "id": 123}

        # Act
        envelope1 = envelope_signer.sign_dict(payload=payload, bus_id=bus_id)
        envelope2 = envelope_signer.sign_dict(payload=payload, bus_id=bus_id)

        # Assert - hashes should be identical
        assert envelope1.signature is not None
        assert envelope2.signature is not None
        assert envelope1.signature.payload_hash == envelope2.signature.payload_hash

    def test_sign_dict_signature_verifiable(
        self,
        envelope_signer: ServiceEnvelopeSigner,
        ed25519_public_key: Ed25519PublicKey,
    ) -> None:
        """Dict signature is verifiable with matching public key."""
        # Arrange
        bus_id = "test-bus"
        payload = {"action": "verify", "data": [1, 2, 3]}

        # Act
        envelope = envelope_signer.sign_dict(
            payload=payload,
            bus_id=bus_id,
        )

        # Verify signature
        assert envelope.signature is not None
        public_key_bytes = ed25519_public_key.public_bytes_raw()
        message = envelope.signature.payload_hash.encode("utf-8")
        signature_b64 = envelope.signature.signature

        # Assert - verification should succeed
        is_valid = verify_base64(public_key_bytes, message, signature_b64)
        assert is_valid is True

    def test_sign_dict_matches_canonical_json_hash(
        self,
        envelope_signer: ServiceEnvelopeSigner,
    ) -> None:
        """Dict payload hash matches direct Blake3 canonical JSON hash."""
        # Arrange
        bus_id = "test-bus"
        payload = {"name": "test", "value": 42}

        # Act
        envelope = envelope_signer.sign_dict(
            payload=payload,
            bus_id=bus_id,
        )

        # Calculate expected hash directly
        expected_hash = hash_canonical_json(payload)

        # Assert
        assert envelope.signature is not None
        assert envelope.signature.payload_hash == expected_hash
