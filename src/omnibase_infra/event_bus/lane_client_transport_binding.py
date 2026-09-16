# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The one seam a CLI has for telling the bus which lane it selected.

OMN-18432. ``RuntimeLocal`` owns event-bus construction and reaches the
provider through exactly one call: ``bus_cls.from_bootstrap(<address>)``, a
single positional string. Everything else it could carry is closed --
``KNOWN_BACKEND_KEYS`` in ``omnibase_core`` is a five-element frozenset, and
``omnibase_core`` must not name an ``omnibase_infra`` symbol on any path
(compat -> core -> spi -> infra, enforced by sdk-boundary-check CI). So a
process that has resolved a lane's protocol, mechanism and credential has no
argument to put them in.

TWO SHAPES WERE AVAILABLE, AND WHY THIS ONE
    Widening ``KNOWN_BACKEND_KEYS`` and passing the transport through
    ``backend_overrides`` is the shape with no process-scoped state. It is also
    an ``omnibase_core`` change, which means a core release and a pin bump in
    every consumer before the dev lane becomes reachable from a developer
    shell -- and it would put a credential VALUE into a ``dict[str, str]`` that
    flows through a layer whose whole job is to know nothing about transports.

    The other shape, taken here, is an explicitly scoped binding the process
    entry point establishes around its own dispatch. It is process-scoped
    state, which is a real cost and is why the three properties below are
    enforced rather than documented.

THE THREE PROPERTIES THAT MAKE IT SAFE
    1. ADDRESS-MATCHED. ``resolve_lane_client_transport`` answers only for the
       exact broker address the binding names. A binding for one lane can never
       be applied to another broker, so "the last lane's credential leaked onto
       the next connection" is not a reachable state.
    2. SCOPED. It is a context manager, cleared on the way out including on an
       exception. The credential does not outlive the dispatch it was resolved
       for.
    3. SINGLE. A second binding inside a first is REFUSED, not stacked. One
       process dispatching to two lanes at once is a mistake, and silently
       choosing the inner one for the caller is how it would stay invisible.

Nothing here reads the environment and nothing here reads a file: the caller
resolves the transport and states it. That keeps the resolution order -- store,
then environment, then refuse -- in one place (``cli/delegate_lane_credentials``)
instead of split across a factory nobody calls directly.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from omnibase_infra.event_bus.model_lane_client_transport import (
    ModelLaneClientTransport,
)

__all__ = [
    "LaneClientTransportBindingError",
    "bind_lane_client_transport",
    "bound_lane_client_transport",
    "resolve_lane_client_transport",
]


class LaneClientTransportBindingError(RuntimeError):
    """A second lane transport was bound while one was already active."""


#: A ``ContextVar`` rather than a module global, for a reason this design has
#: to answer for: an asyncio process can have more than one task in flight, and
#: a plain global would let one task's credential be visible to another task
#: building a client at the same instant. A context variable is task-local, so
#: the binding reaches exactly the call tree that established it.
_BOUND: ContextVar[ModelLaneClientTransport | None] = ContextVar(
    "onex_lane_client_transport", default=None
)


@contextmanager
def bind_lane_client_transport(
    transport: ModelLaneClientTransport,
) -> Iterator[ModelLaneClientTransport]:
    """Make ``transport`` resolvable for its own broker, for this scope only.

    Raises:
        LaneClientTransportBindingError: a transport is already bound. Two
            lanes in one process is not a selection; refusing is what keeps it
            from being silently resolved by nesting order.
    """
    already = _BOUND.get()
    if already is not None:
        message = (
            f"a lane transport is already bound for lane {already.lane!r} at "
            f"{already.bootstrap_servers}; refusing to bind lane "
            f"{transport.lane!r} at {transport.bootstrap_servers} over it. One "
            "process dispatches to one lane."
        )
        raise LaneClientTransportBindingError(message)
    token = _BOUND.set(transport)
    try:
        yield transport
    finally:
        _BOUND.reset(token)


def bound_lane_client_transport() -> ModelLaneClientTransport | None:
    """The active binding, whatever address it names, or ``None``.

    For callers that want to REPORT what is bound. Anything that builds a
    client must use :func:`resolve_lane_client_transport` instead, so the
    address match is never skipped.
    """
    return _BOUND.get()


def resolve_lane_client_transport(
    bootstrap_servers: str,
) -> ModelLaneClientTransport | None:
    """The bound transport for this exact broker address, or ``None``.

    ``None`` is the unbound answer AND the wrong-address answer, on purpose:
    both mean "this connection was not the one the caller resolved a lane for",
    and both must leave the existing environment-sourced construction path
    exactly as it was.
    """
    bound = _BOUND.get()
    if bound is None:
        return None
    if bound.bootstrap_servers != bootstrap_servers.strip():
        return None
    return bound
