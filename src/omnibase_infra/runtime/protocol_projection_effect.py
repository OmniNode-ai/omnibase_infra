# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ProtocolProjectionEffect — synchronous projection effect protocol.

Defines the interface that NodeProjectionEffect must implement (OMN-2508).
The runtime calls ``execute()`` synchronously — blocking until the projection
is persisted — before publishing any intents to Kafka.

Design rationale (OMN-2363 / OMN-2510):
    Projection failure **must** block Kafka publish.  If the projection write
    fails and the runtime published anyway, downstream consumers would observe
    state that has not been persisted, creating an irrecoverable split-brain.

    The synchronous boundary is intentional: ``execute()`` has no ``async``
    keyword so callers cannot inadvertently fire-and-forget the call.  Any
    underlying async I/O is bridged inside the implementation (e.g., via
    ``asyncio.run_coroutine_threadsafe`` or a thread-pool executor).

Migration note (OMN-2510 follow-up):
    Once omnibase_spi>=0.11.0 ships NodeProjectionEffect, this protocol
    should be replaced by the canonical one from omnibase_spi.  The runtime
    import in service_dispatch_result_applier should then point there.

Related:
    - OMN-2508: NodeProjectionEffect implementation (omnibase_spi)
    - OMN-2509: Reducer emits ModelProjectionIntent (omnibase_core)
    - OMN-2510: Runtime wires projection before Kafka publish (this ticket)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from omnibase_infra.models.projection.model_projection_intent import (
        ModelProjectionIntent,
    )
    from omnibase_infra.runtime.models.model_projection_result_local import (
        ModelProjectionResultLocal,
    )


@runtime_checkable
class ProtocolProjectionEffect(Protocol):
    """Protocol for synchronous projection effects.

    Implementations persist a projection before returning so that the runtime
    can guarantee Kafka publish only happens after the projection is durable.

    Contract:
        - ``execute()`` is **synchronous** — no ``async def``.
        - ``execute()`` raises ``ProjectionError`` (or any ``Exception``) on
          failure.  The runtime treats any raised exception as a signal NOT to
          publish to Kafka.
        - ``execute()`` returns ``ModelProjectionResultLocal`` with
          ``success=True`` on success.

    Thread safety:
        Implementations are responsible for their own concurrency concerns.
        The runtime calls ``execute()`` from the async event loop via
        ``asyncio.get_event_loop().run_in_executor()`` if the implementation
        performs blocking I/O, OR the implementation itself bridges to async
        internally.
    """

    def execute(
        self,
        intent: ModelProjectionIntent,
    ) -> ModelProjectionResultLocal:
        """Execute the projection synchronously.

        Persist the projection described by ``intent`` to the backing store
        before returning.  The caller (DispatchResultApplier) will block on
        the return value and will NOT publish to Kafka if this raises.

        Args:
            intent: The projection intent carrying subject, aggregate_id,
                projection_type, payload, and causation chain.

        Returns:
            ModelProjectionResultLocal with ``success=True`` and an optional
            ``artifact_ref`` pointing to the persisted artifact.

        Raises:
            ProjectionError: On projection failure (preferred).
            Exception: Any other exception also blocks Kafka publish.
        """
        ...


__all__ = ["ProtocolProjectionEffect"]
