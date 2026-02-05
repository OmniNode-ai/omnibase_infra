# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Envelope Signer Service.

This module provides a service for signing outbound envelopes with Ed25519 signatures.
The signer uses Blake3 for hashing and Ed25519 for cryptographic signatures, ensuring
message integrity and authenticity.

Signing Process:
    1. Serialize payload to canonical JSON (deterministic key ordering)
    2. Hash the canonical JSON with Blake3
    3. Sign the hash with Ed25519 private key
    4. Build ModelEnvelopeSignature with algorithm, signer, hash, and signature
    5. Return ModelMessageEnvelope containing payload and signature

Security Considerations:
    - Private keys must be kept secure and never logged
    - Canonical JSON ensures consistent hashing across systems
    - Blake3 provides fast, cryptographically secure hashing
    - Ed25519 provides strong, fast signature generation

Exports:
    ServiceEnvelopeSigner: Service class for envelope signing operations

Example:
    >>> from pathlib import Path
    >>> from omnibase_infra.gateway.utils import load_private_key_from_pem
    >>> from omnibase_infra.gateway.services import ServiceEnvelopeSigner
    >>>
    >>> private_key = load_private_key_from_pem(Path("/etc/onex/keys/private.pem"))
    >>> signer = ServiceEnvelopeSigner(
    ...     realm="dev",
    ...     runtime_id="runtime-dev-001",
    ...     private_key=private_key,
    ... )
    >>>
    >>> envelope = signer.sign_envelope(
    ...     payload=my_event,
    ...     bus_id="event-bus-main",
    ... )
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TypeVar
from uuid import UUID, uuid4

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
)
from pydantic import BaseModel

from omnibase_core.crypto.crypto_blake3_hasher import hash_canonical_json
from omnibase_core.crypto.crypto_ed25519_signer import sign_base64
from omnibase_core.models.envelope.model_emitter_identity import ModelEmitterIdentity
from omnibase_core.models.envelope.model_envelope_signature import (
    ModelEnvelopeSignature,
)
from omnibase_core.models.envelope.model_message_envelope import ModelMessageEnvelope

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ServiceEnvelopeSigner:
    """Service for signing outbound envelopes with Ed25519.

    This service wraps a Pydantic payload in a ModelMessageEnvelope with a
    cryptographic signature. The signature ensures message integrity and
    authenticity, allowing recipients to verify the message was sent by
    a trusted runtime and has not been tampered with.

    The signing process uses:
        - Blake3 for fast, secure hashing of the canonical JSON payload
        - Ed25519 for asymmetric signature generation
        - Canonical JSON serialization for deterministic hashing

    Attributes:
        realm: Routing boundary identifier for message routing.
        runtime_id: Unique identifier for this gateway instance.

    Thread Safety:
        This service is thread-safe. The Ed25519 signing operation is
        stateless and can be called concurrently from multiple threads.

    Example:
        >>> from pathlib import Path
        >>> from omnibase_infra.gateway.utils import load_private_key_from_pem
        >>>
        >>> private_key = load_private_key_from_pem(Path("/etc/onex/keys/private.pem"))
        >>> signer = ServiceEnvelopeSigner(
        ...     realm="prod",
        ...     runtime_id="runtime-prod-001",
        ...     private_key=private_key,
        ... )
        >>>
        >>> # Sign an event payload
        >>> envelope = signer.sign_envelope(
        ...     payload=my_event,
        ...     bus_id="event-bus-main",
        ...     trace_id=trace_id,
        ... )
        >>> print(envelope.signature.algorithm)  # 'ed25519'
    """

    def __init__(
        self,
        realm: str,
        runtime_id: str,
        private_key: Ed25519PrivateKey,
    ) -> None:
        """Initialize the envelope signer.

        Args:
            realm: Routing boundary identifier (e.g., "dev", "staging", "prod").
                Used in the envelope to indicate the message's routing domain.
            runtime_id: Unique identifier for this gateway instance (e.g.,
                "runtime-dev-001"). Used as the signer identifier in signatures.
            private_key: Ed25519 private key for signing operations. The key
                is used to generate signatures for outbound envelopes.

        Note:
            The private key bytes are extracted and stored internally. The
            original Ed25519PrivateKey object is not retained, minimizing
            the security surface.
        """
        self._realm = realm
        self._runtime_id = runtime_id
        # Extract raw 32-byte private key for signing operations
        self._private_key_bytes = private_key.private_bytes(
            encoding=Encoding.Raw,
            format=PrivateFormat.Raw,
            encryption_algorithm=NoEncryption(),
        )

        logger.debug(
            "ServiceEnvelopeSigner initialized",
            extra={
                "realm": realm,
                "runtime_id": runtime_id,
            },
        )

    @property
    def realm(self) -> str:
        """Return the configured realm identifier.

        Returns:
            The routing boundary identifier for this signer.
        """
        return self._realm

    @property
    def runtime_id(self) -> str:
        """Return the configured runtime identifier.

        Returns:
            The unique gateway instance identifier.
        """
        return self._runtime_id

    def sign_envelope(  # noqa: ONEX-PATTERN-PARAMS - Optional params with defaults are acceptable
        self,
        payload: T,
        bus_id: str,
        trace_id: UUID | None = None,
        causality_id: UUID | None = None,
        tenant_id: str | None = None,
        emitter_identity: ModelEmitterIdentity | None = None,
    ) -> ModelMessageEnvelope[T]:
        """Sign a payload and wrap in ModelMessageEnvelope.

        Creates a signed envelope containing the payload with a cryptographic
        signature. The signature is computed over the canonical JSON
        representation of the payload, ensuring deterministic hashing.

        Signing Process:
            1. Serialize payload to dictionary using Pydantic's model_dump()
            2. Hash the dictionary using canonical JSON serialization with Blake3
            3. Sign the hash bytes with Ed25519 private key
            4. Build ModelEnvelopeSignature with algorithm, signer, hash, signature
            5. Construct ModelMessageEnvelope with all fields

        Args:
            payload: Pydantic BaseModel instance to sign and wrap. The payload
                is serialized to canonical JSON for hashing.
            bus_id: Identifier for the event bus this envelope is destined for.
            trace_id: Optional trace identifier for distributed tracing. If not
                provided, a new UUID is generated.
            causality_id: Optional causality chain identifier for event correlation.
            tenant_id: Optional tenant identifier for multi-tenant routing.
            emitter_identity: Optional emitter identity for attribution.

        Returns:
            ModelMessageEnvelope containing the payload and cryptographic signature.
            The envelope includes all routing metadata and can be serialized for
            transmission.

        Example:
            >>> envelope = signer.sign_envelope(
            ...     payload=MyEvent(action="created", resource_id="123"),
            ...     bus_id="main-bus",
            ...     trace_id=UUID("12345678-1234-5678-1234-567812345678"),
            ... )
            >>> print(envelope.signature.algorithm)
            'ed25519'
            >>> print(envelope.realm)
            'dev'
        """
        # Generate trace_id if not provided
        effective_trace_id = trace_id if trace_id is not None else uuid4()

        # Step 1: Serialize payload to dictionary
        payload_dict = payload.model_dump(mode="json")

        # Step 2: Hash with Blake3 using canonical JSON
        # hash_canonical_json expects dict[str, object]
        payload_hash = hash_canonical_json(payload_dict)

        # Step 3: Sign the hash with Ed25519
        # sign_base64 expects bytes for the data to sign
        signature_b64 = sign_base64(
            self._private_key_bytes,
            payload_hash.encode("utf-8"),
        )

        # Step 4: Build ModelEnvelopeSignature
        envelope_signature = ModelEnvelopeSignature(
            algorithm="ed25519",
            signer=self._runtime_id,
            payload_hash=payload_hash,
            signature=signature_b64,
        )

        # Step 5: Build and return ModelMessageEnvelope
        envelope: ModelMessageEnvelope[T] = ModelMessageEnvelope[T](
            realm=self._realm,
            runtime_id=self._runtime_id,
            bus_id=bus_id,
            trace_id=effective_trace_id,
            emitted_at=datetime.now(UTC),
            signature=envelope_signature,
            payload=payload,
            causality_id=causality_id,
            tenant_id=tenant_id,
            emitter_identity=emitter_identity,
        )

        logger.debug(
            "Signed envelope",
            extra={
                "trace_id": str(effective_trace_id),
                "realm": self._realm,
                "runtime_id": self._runtime_id,
                "bus_id": bus_id,
                "payload_hash": payload_hash[:16] + "...",  # Truncate for logging
            },
        )

        return envelope

    def sign_dict(  # noqa: ONEX-PATTERN-PARAMS - Optional params with defaults are acceptable
        self,
        payload: dict[str, object],
        bus_id: str,
        trace_id: UUID | None = None,
        causality_id: UUID | None = None,
        tenant_id: str | None = None,
        emitter_identity: ModelEmitterIdentity | None = None,
    ) -> ModelMessageEnvelope[dict[str, object]]:
        """Sign a dict payload and wrap in ModelMessageEnvelope.

        This is a variant of sign_envelope() that accepts raw dict payloads
        instead of Pydantic BaseModel instances. Useful for signing envelope
        dicts that don't have a corresponding model.

        Signing Process:
            1. Hash the dict using canonical JSON serialization with Blake3
            2. Sign the hash bytes with Ed25519 private key
            3. Build ModelEnvelopeSignature with algorithm, signer, hash, signature
            4. Construct ModelMessageEnvelope with all fields

        Args:
            payload: Dict payload to sign and wrap. The payload is serialized
                to canonical JSON for hashing.
            bus_id: Identifier for the event bus this envelope is destined for.
            trace_id: Optional trace identifier for distributed tracing. If not
                provided, a new UUID is generated.
            causality_id: Optional causality chain identifier for event correlation.
            tenant_id: Optional tenant identifier for multi-tenant routing.
            emitter_identity: Optional emitter identity for attribution.

        Returns:
            ModelMessageEnvelope containing the dict payload and cryptographic
            signature.

        Example:
            >>> envelope = signer.sign_dict(
            ...     payload={"action": "created", "resource_id": "123"},
            ...     bus_id="main-bus",
            ...     trace_id=UUID("12345678-1234-5678-1234-567812345678"),
            ... )
            >>> print(envelope.signature.algorithm)
            'ed25519'

        .. versionadded:: OMN-1899
            Added to support signing dict envelopes in runtime host.
        """
        # Generate trace_id if not provided
        effective_trace_id = trace_id if trace_id is not None else uuid4()

        # Step 1: Hash with Blake3 using canonical JSON
        # payload is already a dict, no need to call model_dump()
        payload_hash = hash_canonical_json(payload)

        # Step 2: Sign the hash with Ed25519
        signature_b64 = sign_base64(
            self._private_key_bytes,
            payload_hash.encode("utf-8"),
        )

        # Step 3: Build ModelEnvelopeSignature
        envelope_signature = ModelEnvelopeSignature(
            algorithm="ed25519",
            signer=self._runtime_id,
            payload_hash=payload_hash,
            signature=signature_b64,
        )

        # Step 4: Build and return ModelMessageEnvelope
        envelope: ModelMessageEnvelope[dict[str, object]] = ModelMessageEnvelope(
            realm=self._realm,
            runtime_id=self._runtime_id,
            bus_id=bus_id,
            trace_id=effective_trace_id,
            emitted_at=datetime.now(UTC),
            signature=envelope_signature,
            payload=payload,
            causality_id=causality_id,
            tenant_id=tenant_id,
            emitter_identity=emitter_identity,
        )

        logger.debug(
            "Signed dict envelope",
            extra={
                "trace_id": str(effective_trace_id),
                "realm": self._realm,
                "runtime_id": self._runtime_id,
                "bus_id": bus_id,
                "payload_hash": payload_hash[:16] + "...",  # Truncate for logging
            },
        )

        return envelope


__all__: list[str] = [
    "ServiceEnvelopeSigner",
]
