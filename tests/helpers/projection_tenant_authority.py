# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Real Ed25519 projection-authority fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

from omnibase_core.crypto.crypto_ed25519_signer import generate_keypair
from omnibase_core.models.envelope.model_message_envelope import ModelMessageEnvelope
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.runtime.projection_tenant_authority import (
    VerifiedProjectionTenantAuthority,
    verify_signed_projection_tenant_authority,
)


class InMemoryKeyProvider:
    """Minimal test provider implementing the core key-provider protocol."""

    def __init__(self, keys: dict[str, bytes] | None = None) -> None:
        self._keys = dict(keys or {})

    def get_public_key(self, runtime_id: str) -> bytes | None:
        return self._keys.get(runtime_id)

    def register_key(self, runtime_id: str, public_key: bytes) -> None:
        if len(public_key) != 32:
            raise ValueError("Ed25519 public keys must be 32 bytes")
        self._keys[runtime_id] = public_key

    def has_key(self, runtime_id: str) -> bool:
        return runtime_id in self._keys

    def list_runtime_ids(self) -> list[str]:
        return sorted(self._keys)


@dataclass(frozen=True)
class StaticTenantBindingResolver:
    runtime_id: str
    realm: str
    bus_id: str
    tenant_id: UUID

    def resolve_tenant_id(
        self,
        *,
        runtime_id: str,
        realm: str,
        bus_id: str,
    ) -> UUID | None:
        if (runtime_id, realm, bus_id) != (
            self.runtime_id,
            self.realm,
            self.bus_id,
        ):
            return None
        return self.tenant_id


@dataclass(frozen=True)
class SignedTenantAuthorityFixture:
    envelope: ModelMessageEnvelope[ModelEventEnvelope[dict[str, object]]]
    key_provider: InMemoryKeyProvider
    binding_resolver: StaticTenantBindingResolver

    def verify(self) -> VerifiedProjectionTenantAuthority:
        return verify_signed_projection_tenant_authority(
            self.envelope,
            self.key_provider,
            self.binding_resolver,
        )


def signed_tenant_authority_fixture(
    tenant_id: UUID,
    *,
    payload: dict[str, object] | None = None,
    event_envelope: ModelEventEnvelope[dict[str, object]] | None = None,
    runtime_id: str = "tenant-gateway-proof",
    realm: str = "test",
    bus_id: str = "proof-bus",
) -> SignedTenantAuthorityFixture:
    keypair = generate_keypair()
    typed_event = event_envelope or ModelEventEnvelope[dict[str, object]](
        payload=payload or {},
        correlation_id=uuid4(),
    )
    if typed_event.correlation_id is None:
        raise ValueError("projection authority fixture requires correlation_id")
    envelope = ModelMessageEnvelope[
        ModelEventEnvelope[dict[str, object]]
    ].create_signed(
        realm=realm,
        runtime_id=runtime_id,
        bus_id=bus_id,
        tenant_id=str(tenant_id),
        payload=typed_event,
        trace_id=typed_event.correlation_id,
        private_key=keypair.private_key_bytes,
        emitted_at=datetime.now(UTC),
    )
    return SignedTenantAuthorityFixture(
        envelope=envelope,
        key_provider=InMemoryKeyProvider({runtime_id: keypair.public_key_bytes}),
        binding_resolver=StaticTenantBindingResolver(
            runtime_id=runtime_id,
            realm=realm,
            bus_id=bus_id,
            tenant_id=tenant_id,
        ),
    )


def verified_tenant_authority(
    tenant_id: UUID,
) -> VerifiedProjectionTenantAuthority:
    return signed_tenant_authority_fixture(tenant_id).verify()


def verified_tenant_dispatch(
    tenant_id: UUID,
) -> tuple[VerifiedProjectionTenantAuthority, ModelEventEnvelope[dict[str, object]]]:
    fixture = signed_tenant_authority_fixture(tenant_id)
    return fixture.verify(), fixture.envelope.payload


__all__ = [
    "InMemoryKeyProvider",
    "SignedTenantAuthorityFixture",
    "StaticTenantBindingResolver",
    "signed_tenant_authority_fixture",
    "verified_tenant_authority",
    "verified_tenant_dispatch",
]
