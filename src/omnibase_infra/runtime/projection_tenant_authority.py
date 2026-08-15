# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cryptographically verified tenant authority for projection writes.

The public event models in this repository are data containers, not proof that
authentication occurred.  In particular, ``ModelEventEnvelope`` security
context and metadata fields can be supplied by any deserializer.  This module
therefore mints a sealed-construction capability only after the canonical core
``ModelMessageEnvelope`` signature verifies.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from uuid import UUID

from pydantic import BaseModel

if TYPE_CHECKING:
    from omnibase_core.protocols.crypto.protocol_key_provider import (
        ProtocolKeyProvider,
    )


_AUTHORITY_MINT = object()


def _tenant_context_error(message: str) -> Exception:
    from omnibase_infra.errors.error_projection import ProjectionTenantContextError

    return ProjectionTenantContextError(message)


def parse_canonical_tenant_uuid(value: object, *, authority: str) -> UUID:
    """Parse a non-zero canonical UUID without accepting slugs or sentinels."""
    if not isinstance(value, str) or value != value.strip():
        raise _tenant_context_error(
            f"Projection tenant context from {authority} is not a canonical UUID"
        )
    try:
        tenant_id = UUID(value)
    except ValueError as exc:
        raise _tenant_context_error(
            f"Projection tenant context from {authority} is malformed"
        ) from exc
    if str(tenant_id) != value or tenant_id.int == 0:
        raise _tenant_context_error(
            f"Projection tenant context from {authority} is not a canonical UUID"
        )
    return tenant_id


@dataclass(frozen=True, slots=True, init=False)
class VerifiedProjectionTenantAuthority:
    """Opaque, in-process capability produced by signature verification.

    The constructor is intentionally sealed.  Callers obtain an instance only
    through :func:`verify_signed_projection_tenant_authority`; request-shaped
    dictionaries and materialized envelopes cannot be coerced into this type.
    """

    tenant_id: UUID
    trace_id: UUID
    runtime_id: str
    realm: str
    bus_id: str
    emitted_at: datetime
    event_envelope_id: UUID
    event_payload_hash: str

    def __init__(
        self,
        *,
        tenant_id: UUID,
        trace_id: UUID,
        runtime_id: str,
        realm: str,
        bus_id: str,
        emitted_at: datetime,
        event_envelope_id: UUID,
        event_payload_hash: str,
        _mint: object,
    ) -> None:
        if _mint is not _AUTHORITY_MINT:
            raise TypeError(
                "VerifiedProjectionTenantAuthority can only be minted by "
                "signature verification"
            )
        object.__setattr__(self, "tenant_id", tenant_id)
        object.__setattr__(self, "trace_id", trace_id)
        object.__setattr__(self, "runtime_id", runtime_id)
        object.__setattr__(self, "realm", realm)
        object.__setattr__(self, "bus_id", bus_id)
        object.__setattr__(self, "emitted_at", emitted_at)
        object.__setattr__(self, "event_envelope_id", event_envelope_id)
        object.__setattr__(self, "event_payload_hash", event_payload_hash)


@runtime_checkable
class ProtocolProjectionTenantBindingResolver(Protocol):
    """Authoritative mapping from a verified signer scope to one tenant UUID."""

    def resolve_tenant_id(
        self,
        *,
        runtime_id: str,
        realm: str,
        bus_id: str,
    ) -> UUID | None:
        """Return the tenant authorized for this signer scope, or ``None``."""
        ...


def verify_signed_projection_tenant_authority(
    envelope: object,
    key_provider: ProtocolKeyProvider,
    tenant_binding_resolver: ProtocolProjectionTenantBindingResolver,
) -> VerifiedProjectionTenantAuthority:
    """Verify a core signed envelope and mint its canonical UUID authority.

    ``ModelMessageEnvelope`` signs the tenant identifier together with the
    runtime, realm, bus, trace, timestamp, and payload hash.  Unknown signers,
    bad signatures, altered payloads/metadata, missing tenants, and non-UUID
    tenants all fail closed before a database adapter can be built.
    """
    from omnibase_core.models.envelope.model_message_envelope import (
        ModelMessageEnvelope,
    )

    if not isinstance(envelope, ModelMessageEnvelope):
        raise _tenant_context_error(
            "Tenant projection authority requires a signed ModelMessageEnvelope"
        )
    try:
        verified = envelope.verify_signature(key_provider)
    except Exception as exc:
        raise _tenant_context_error(
            "Projection tenant envelope signature verification failed"
        ) from exc
    if not verified:
        raise _tenant_context_error(
            "Projection tenant envelope signature verification failed"
        )
    tenant_id = parse_canonical_tenant_uuid(
        envelope.tenant_id,
        authority="verified signed message envelope",
    )
    authorized_tenant_id = tenant_binding_resolver.resolve_tenant_id(
        runtime_id=envelope.runtime_id,
        realm=envelope.realm,
        bus_id=envelope.bus_id,
    )
    if not isinstance(authorized_tenant_id, UUID):
        raise _tenant_context_error(
            "Projection envelope signer has no authoritative tenant binding"
        )
    if authorized_tenant_id != tenant_id:
        raise _tenant_context_error(
            "Projection envelope tenant does not match its signer binding"
        )
    from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope

    event_envelope = envelope.payload
    if not isinstance(event_envelope, ModelEventEnvelope):
        try:
            event_envelope = ModelEventEnvelope[object].model_validate(event_envelope)
        except Exception as exc:
            raise _tenant_context_error(
                "Signed projection authority must wrap a valid ModelEventEnvelope"
            ) from exc
    if (
        not isinstance(event_envelope.correlation_id, UUID)
        or event_envelope.correlation_id != envelope.trace_id
    ):
        raise _tenant_context_error(
            "Signed projection trace does not match event correlation UUID"
        )
    return VerifiedProjectionTenantAuthority(
        tenant_id=tenant_id,
        trace_id=envelope.trace_id,
        runtime_id=envelope.runtime_id,
        realm=envelope.realm,
        bus_id=envelope.bus_id,
        emitted_at=envelope.emitted_at,
        event_envelope_id=event_envelope.envelope_id,
        event_payload_hash=envelope.signature.payload_hash,
        _mint=_AUTHORITY_MINT,
    )


def assert_projection_tenant_authority_matches_event(
    authority: VerifiedProjectionTenantAuthority,
    event_envelope: object,
) -> None:
    """Bind a verified capability to the exact event being dispatched."""
    from omnibase_core.crypto.crypto_blake3_hasher import hash_canonical_json
    from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope

    if not isinstance(event_envelope, ModelEventEnvelope):
        raise _tenant_context_error(
            "Verified tenant authority requires a typed dispatch envelope"
        )
    if (
        event_envelope.envelope_id != authority.event_envelope_id
        or event_envelope.correlation_id != authority.trace_id
    ):
        raise _tenant_context_error(
            "Verified tenant authority does not match the dispatched envelope"
        )
    payload: BaseModel = event_envelope
    actual_hash = hash_canonical_json(payload.model_dump(mode="json"))
    if actual_hash != authority.event_payload_hash:
        raise _tenant_context_error(
            "Verified tenant authority payload does not match the dispatched envelope"
        )


__all__ = [
    "ProtocolProjectionTenantBindingResolver",
    "VerifiedProjectionTenantAuthority",
    "assert_projection_tenant_authority_matches_event",
    "parse_canonical_tenant_uuid",
    "verify_signed_projection_tenant_authority",
]
