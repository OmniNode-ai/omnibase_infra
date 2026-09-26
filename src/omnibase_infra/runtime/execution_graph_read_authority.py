# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Sealed authority for a signed, gateway-admitted execution-graph read.

An event envelope's tenant is attribution, not authorization. Only a trusted
gateway's signed outer message can mint this request-time read capability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from omnibase_core.models.envelope.model_message_envelope import ModelMessageEnvelope
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.execution_graph_replay.model_execution_graph_request import (
    ModelExecutionGraphRequest,
)

if TYPE_CHECKING:
    from omnibase_core.protocols.crypto.protocol_key_provider import (
        ProtocolKeyProvider,
    )

_AUTHORITY_MINT = object()


class ExecutionGraphReadAuthorityError(PermissionError):
    """The request lacks trusted, matching tenant and correlation proof."""


@dataclass(frozen=True, slots=True)
class TrustedGatewaySignerScope:
    """One gateway signing identity admitted for graph reads."""

    runtime_id: str
    realm: str
    bus_id: str

    def __post_init__(self) -> None:
        if not self.runtime_id or not self.realm or not self.bus_id:
            raise ValueError("Trusted gateway signer scope fields must be non-empty")


@dataclass(frozen=True, slots=True)
class TrustedExecutionGraphGatewayPolicy:
    """Explicit allowlist; key possession by some other runtime is insufficient."""

    scopes: frozenset[TrustedGatewaySignerScope]

    def __post_init__(self) -> None:
        if not self.scopes or any(
            type(scope) is not TrustedGatewaySignerScope for scope in self.scopes
        ):
            raise ValueError("Graph read requires a non-empty typed gateway allowlist")


@dataclass(frozen=True, slots=True, init=False)
class VerifiedExecutionGraphReadAuthority:
    """In-process capability tied to one signed tenant/correlation request."""

    tenant_id: UUID
    correlation_id: UUID
    request: ModelExecutionGraphRequest
    signer_scope: TrustedGatewaySignerScope
    payload_hash: str

    def __init__(
        self,
        *,
        tenant_id: UUID,
        correlation_id: UUID,
        request: ModelExecutionGraphRequest,
        signer_scope: TrustedGatewaySignerScope,
        payload_hash: str,
        _mint: object,
    ) -> None:
        if _mint is not _AUTHORITY_MINT:
            raise TypeError(
                "Graph read authority requires gateway signature verification"
            )
        object.__setattr__(self, "tenant_id", tenant_id)
        object.__setattr__(self, "correlation_id", correlation_id)
        object.__setattr__(self, "request", request)
        object.__setattr__(self, "signer_scope", signer_scope)
        object.__setattr__(self, "payload_hash", payload_hash)


def _canonical_uuid(value: object, *, name: str) -> UUID:
    if not isinstance(value, str) or value != value.strip():
        raise ExecutionGraphReadAuthorityError(f"Graph read {name} is not canonical")
    try:
        parsed = UUID(value)
    except ValueError as exc:
        raise ExecutionGraphReadAuthorityError(
            f"Graph read {name} is not canonical"
        ) from exc
    if parsed.int == 0 or str(parsed) != value:
        raise ExecutionGraphReadAuthorityError(f"Graph read {name} is not canonical")
    return parsed


def verify_signed_execution_graph_read_authority(
    envelope: object,
    key_provider: ProtocolKeyProvider,
    policy: TrustedExecutionGraphGatewayPolicy,
) -> VerifiedExecutionGraphReadAuthority:
    """Verify the outer signature and every inner/outer identity binding."""
    if not isinstance(envelope, ModelMessageEnvelope):
        raise ExecutionGraphReadAuthorityError(
            "Graph read requires a signed ModelMessageEnvelope"
        )
    scope = TrustedGatewaySignerScope(
        runtime_id=envelope.runtime_id,
        realm=envelope.realm,
        bus_id=envelope.bus_id,
    )
    if scope not in policy.scopes:
        raise ExecutionGraphReadAuthorityError(
            "Graph read signer is not a trusted gateway"
        )
    try:
        verified = envelope.verify_signature(key_provider)
    except Exception as exc:
        raise ExecutionGraphReadAuthorityError(
            "Graph read signature verification failed"
        ) from exc
    if not verified:
        raise ExecutionGraphReadAuthorityError(
            "Graph read signature verification failed"
        )

    tenant_id = _canonical_uuid(envelope.tenant_id, name="tenant")
    if not isinstance(envelope.payload, dict):
        raise ExecutionGraphReadAuthorityError(
            "Graph read signed payload is not a canonical envelope"
        )
    try:
        inner = ModelEventEnvelope[object].model_validate(envelope.payload)
    except Exception as exc:
        raise ExecutionGraphReadAuthorityError(
            "Graph read signed payload is not a canonical envelope"
        ) from exc
    inner_tenant = _canonical_uuid(inner.tenant_id, name="inner tenant")
    if inner_tenant != tenant_id:
        raise ExecutionGraphReadAuthorityError(
            "Graph read inner tenant conflicts with signed tenant"
        )
    if inner.correlation_id != envelope.trace_id:
        raise ExecutionGraphReadAuthorityError(
            "Graph read inner correlation conflicts with signed trace"
        )
    try:
        request = ModelExecutionGraphRequest.model_validate(inner.payload)
    except Exception as exc:
        raise ExecutionGraphReadAuthorityError(
            "Graph read signed payload does not contain a valid request"
        ) from exc
    if request.correlation_id != envelope.trace_id:
        raise ExecutionGraphReadAuthorityError(
            "Graph read request correlation conflicts with signed trace"
        )
    return VerifiedExecutionGraphReadAuthority(
        tenant_id=tenant_id,
        correlation_id=envelope.trace_id,
        request=request,
        signer_scope=scope,
        payload_hash=envelope.signature.payload_hash,
        _mint=_AUTHORITY_MINT,
    )


__all__ = [
    "ExecutionGraphReadAuthorityError",
    "TrustedExecutionGraphGatewayPolicy",
    "TrustedGatewaySignerScope",
    "VerifiedExecutionGraphReadAuthority",
    "verify_signed_execution_graph_read_authority",
]
