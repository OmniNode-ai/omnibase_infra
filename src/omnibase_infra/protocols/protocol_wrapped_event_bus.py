# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Structural protocol for a bus wrapper that names the bus it forwards to.

OMN-21753. ``LazyStartRuntimeTransaction`` wraps the runtime bus in a
``RecordingBus`` so that it can stop the subscriptions of one contract on a
rollback. Every other call is forwarded, but a capability check such as
``supports_offset_handover`` cannot see through a forwarding ``__getattr__``:
``isinstance`` against a ``runtime_checkable`` Protocol reads members with
``inspect.getattr_static``, and against a concrete bus class it reads the
wrapper's own type. A wrapper that implements this protocol says which bus it
forwards to, and the pre-subscribe preflight asks that bus about handover and
consume concurrency, the bus the declarations are honoured on.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["ProtocolWrappedEventBus"]


@runtime_checkable
class ProtocolWrappedEventBus(Protocol):
    """A bus wrapper that forwards every call to ``wrapped_event_bus``."""

    @property
    def wrapped_event_bus(self) -> object:
        """The bus this wrapper forwards to."""
        ...
