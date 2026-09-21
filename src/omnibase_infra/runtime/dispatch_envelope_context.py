# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""In-process typed context channels for one materialized dispatch."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.runtime.projection_tenant_authority import (
    VerifiedProjectionTenantAuthority,
)

# The dispatch payload remains strictly JSON-safe.  The original typed envelope
# travels beside it only to preserve transport identity such as envelope_id.  It
# is explicitly NOT an authentication source.
_CURRENT_DISPATCH_ENVELOPE: ContextVar[ModelEventEnvelope[object] | None] = ContextVar(
    "onex_current_dispatch_envelope",
    default=None,
)

# Authentication is a separate channel.  Only the opaque capability minted by
# canonical signature verification may be bound here.
_CURRENT_PROJECTION_TENANT_AUTHORITY: ContextVar[
    VerifiedProjectionTenantAuthority | None
] = ContextVar("onex_current_projection_tenant_authority", default=None)

# OMN-18955 (producer half of OMN-18905).  The SOURCE message's own Kafka
# coordinates, as a (partition, offset) pair.
#
# WHY A CONTEXT CHANNEL AND NOT THE ENVELOPE.  ``ModelEventEnvelope`` declares
# no partition or offset field and is ``extra="forbid"``, so the pair cannot
# ride it without a core release and an ``envelope_version`` bump.  The
# coordinates exist -- ``ModelEventMessage`` carries them straight off the
# aiokafka record -- but the consume callback rebuilds the envelope from
# ``message.value`` alone, and that is the single line at which they leave the
# path.  A task-local channel bound in the frame that still holds the record
# carries them the rest of the way, which is exactly how
# ``onex_active_consumer_flow_key`` already reaches the same dispatcher.
#
# WHY IT MATTERS.  A projection writer under in-process dispatch builds its
# snapshot-delta ``MessageMeta`` from these two values.  With neither injected
# it publishes every delta at partition 0 / offset 0, and the consuming
# SnapshotCache drops a delta whose ``source_offset`` is not greater than the
# cached one for the same source topic and partition -- so every delta after
# the FIRST for a given key is discarded as an idempotent replay and the
# exposure freezes while sitting at lag zero.  Measured on the .201 dev lane:
# 6,210,195 lifetime drops on the consumer-flow exposure, and a readiness
# endpoint answering 503 because it correctly refuses to call that healthy.
_CURRENT_SOURCE_COORDINATE: ContextVar[tuple[int, str] | None] = ContextVar(
    "onex_current_source_coordinate",
    default=None,
)


@contextmanager
def bind_dispatch_envelope(envelope: object) -> Iterator[None]:
    """Bind only a typed envelope and restore the prior context on every exit."""
    authoritative = envelope if isinstance(envelope, ModelEventEnvelope) else None
    token = _CURRENT_DISPATCH_ENVELOPE.set(authoritative)
    try:
        yield
    finally:
        _CURRENT_DISPATCH_ENVELOPE.reset(token)


def current_dispatch_envelope() -> ModelEventEnvelope[object] | None:
    """Return the typed envelope bound to the current dispatcher invocation."""
    return _CURRENT_DISPATCH_ENVELOPE.get()


@contextmanager
def bind_projection_tenant_authority(
    authority: VerifiedProjectionTenantAuthority,
) -> Iterator[None]:
    """Bind one verified capability and restore the prior value on exit."""
    if type(authority) is not VerifiedProjectionTenantAuthority:
        raise TypeError("projection tenant authority must be a verified capability")
    token = _CURRENT_PROJECTION_TENANT_AUTHORITY.set(authority)
    try:
        yield
    finally:
        _CURRENT_PROJECTION_TENANT_AUTHORITY.reset(token)


def current_projection_tenant_authority() -> VerifiedProjectionTenantAuthority | None:
    """Return the capability bound by a trusted ingress verification boundary."""
    return _CURRENT_PROJECTION_TENANT_AUTHORITY.get()


@contextmanager
def bind_source_coordinate(message: object) -> Iterator[None]:
    """Bind the source record's ``(partition, offset)`` for this dispatch.

    ABSENT IS A STATEMENT, NOT A ZERO.  Nothing is bound unless the record
    carries BOTH values, so a transport that cannot report coordinates leaves
    the reader with no key rather than a defaulted ``0`` -- and a defaulted
    zero is the exact defect this channel exists to remove (OMN-18955).  The
    same discipline governs ``_envelope_timestamp`` and ``_tenant_id`` at the
    projection dispatch site.

    The offset is kept as the ``str`` the transport model declares rather than
    coerced here: the reader already parses it, and narrowing at two places
    invites the two from disagreeing.
    """
    partition = getattr(message, "partition", None)
    offset = getattr(message, "offset", None)
    if not isinstance(partition, int) or isinstance(partition, bool) or offset is None:
        yield
        return
    token = _CURRENT_SOURCE_COORDINATE.set((partition, str(offset)))
    try:
        yield
    finally:
        _CURRENT_SOURCE_COORDINATE.reset(token)


def current_source_coordinate() -> tuple[int, str] | None:
    """Return the source record's ``(partition, offset)``, or ``None``."""
    return _CURRENT_SOURCE_COORDINATE.get()


__all__ = [
    "bind_dispatch_envelope",
    "bind_projection_tenant_authority",
    "bind_source_coordinate",
    "current_dispatch_envelope",
    "current_projection_tenant_authority",
    "current_source_coordinate",
]
