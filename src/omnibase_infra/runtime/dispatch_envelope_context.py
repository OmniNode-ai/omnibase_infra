# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""In-process typed context channels for one materialized dispatch.

A SOURCE-COORDINATE CHANNEL DOES NOT BELONG HERE, and one was removed from this
module under OMN-18918 rather than left beside its replacement. The source
message's partition and offset reach a projection writer as an explicit
``ModelMessageDeliveryContext`` on ``ProtocolDispatchEngine.dispatch``, threaded
from the consume loop that holds the record -- not as ambient task-local state.
The operator ruled for the typed parameter over the context channel on
2026-09-20, on the ground that partition and offset are facts about a DELIVERY
and the protocol should say what it carries. Two mechanisms writing the same two
payload keys is the failure that ruling exists to prevent: the later writer wins
silently and no test sees it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.runtime.projection_tenant_authority import (
    VerifiedProjectionTenantAuthority,
)

if TYPE_CHECKING:
    from omnibase_infra.runtime.execution_graph_read_authority import (
        VerifiedExecutionGraphReadAuthority,
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
_CURRENT_EXECUTION_GRAPH_READ_AUTHORITY: ContextVar[object | None] = ContextVar(
    "onex_current_execution_graph_read_authority", default=None
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
def bind_execution_graph_read_authority(
    authority: VerifiedExecutionGraphReadAuthority,
) -> Iterator[None]:
    """Bind exactly one verified graph-read capability for one dispatch."""
    from omnibase_infra.runtime.execution_graph_read_authority import (
        VerifiedExecutionGraphReadAuthority,
    )

    if type(authority) is not VerifiedExecutionGraphReadAuthority:
        raise TypeError("execution graph authority must be a verified capability")
    token = _CURRENT_EXECUTION_GRAPH_READ_AUTHORITY.set(authority)
    try:
        yield
    finally:
        _CURRENT_EXECUTION_GRAPH_READ_AUTHORITY.reset(token)


def current_execution_graph_read_authority() -> (
    VerifiedExecutionGraphReadAuthority | None
):
    """Return the verified graph-read capability for this dispatch, if any."""
    return _CURRENT_EXECUTION_GRAPH_READ_AUTHORITY.get()  # type: ignore[return-value]


__all__ = [
    "bind_dispatch_envelope",
    "bind_execution_graph_read_authority",
    "bind_projection_tenant_authority",
    "current_dispatch_envelope",
    "current_execution_graph_read_authority",
    "current_projection_tenant_authority",
]
