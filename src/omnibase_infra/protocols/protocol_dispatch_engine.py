# Copyright (c) 2025 OmniNode Team
"""Protocol definition for dispatch engine interface.

This module defines the ProtocolDispatchEngine interface for dispatch engines
that route event envelopes to appropriate handlers based on topic. The primary
implementation is MessageDispatchEngine in the runtime module.

Architecture:
    The dispatch engine sits between event bus consumption and handler execution:

    Kafka/EventBus -> EventBusSubcontractWiring -> ProtocolDispatchEngine -> Handlers

    The wiring layer uses this protocol for duck-typed dispatch engine injection,
    enabling both production MessageDispatchEngine and test mocks.

Thread Safety:
    Implementations MUST be thread-safe for concurrent async dispatch calls.
    Multiple coroutines may invoke dispatch() simultaneously from event bus
    callbacks.

    Design Requirements:
        - Use asyncio.Lock for shared mutable state (e.g., routing tables)
        - Dispatch operations should be stateless where possible
        - Handler invocations may run concurrently

Related:
    - OMN-1621: Runtime consumes event_bus subcontract for contract-driven wiring
    - EventBusSubcontractWiring: Uses this protocol for dispatch engine injection
    - MessageDispatchEngine: Production implementation of this protocol

.. versionadded:: 0.2.5
"""

from __future__ import annotations

__all__ = ["ProtocolDispatchEngine"]

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
    from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult


@runtime_checkable
class ProtocolDispatchEngine(Protocol):
    """Protocol for dispatch engines that route event envelopes to handlers.

    Dispatch engines are responsible for routing incoming event envelopes
    to the appropriate handlers based on topic and message type. The protocol
    enables duck-typed injection of dispatch engines into the event bus wiring
    layer.

    Protocol Verification:
        Per ONEX conventions, protocol compliance is verified via duck typing
        rather than isinstance checks:

        .. code-block:: python

            # Verify required method exists and is callable
            if hasattr(engine, 'dispatch') and callable(engine.dispatch):
                wiring = EventBusSubcontractWiring(
                    event_bus=event_bus,
                    dispatch_engine=engine,
                    environment="dev",
                )
            else:
                raise TypeError("Object does not implement ProtocolDispatchEngine")

    Example:
        .. code-block:: python

            from omnibase_infra.protocols import ProtocolDispatchEngine
            from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope

            class MyDispatchEngine:
                '''Custom dispatch engine implementation.'''

                async def dispatch(
                    self,
                    topic: str,
                    envelope: ModelEventEnvelope[object],
                ) -> ModelDispatchResult | None:
                    '''Dispatch envelope to appropriate handler.'''
                    handler = self._resolve_handler(topic, envelope)
                    return await handler.handle(envelope)

            # Verify protocol compliance via duck typing
            engine = MyDispatchEngine()
            assert hasattr(engine, 'dispatch') and callable(engine.dispatch)

    Note:
        Method bodies in this Protocol use ``...`` (Ellipsis) rather than
        ``raise NotImplementedError()``. This is the standard Python convention
        for ``typing.Protocol`` classes per PEP 544.

    .. versionadded:: 0.2.5
    """

    async def dispatch(
        self,
        topic: str,
        envelope: ModelEventEnvelope[object],
    ) -> ModelDispatchResult | None:
        """Dispatch an event envelope to the appropriate handler(s).

        Routes the envelope to handlers registered for the given topic.
        The dispatch engine resolves handlers based on topic patterns,
        message types, and other routing criteria.

        Typing Note (ModelEventEnvelope[object]):
            The envelope parameter uses ``ModelEventEnvelope[object]`` instead of
            ``Any`` per CLAUDE.md guidance: "Use ``object`` for generic payloads".

            This is intentional:
            - CLAUDE.md mandates "NEVER use ``Any``" and specifies ``object`` for
              generic payloads that can accept multiple event types
            - The dispatch engine handles envelopes from multiple topics with
              different payload types
            - Handler implementations use ``isinstance()`` type guards for
              payload-specific processing

        Thread Safety:
            This method MUST be safe for concurrent calls from multiple
            coroutines. The event bus wiring layer may dispatch messages
            from multiple topic subscriptions concurrently.

        Args:
            topic: The full topic name from which the envelope was consumed.
                Used for routing and logging context.
                Example: "dev.onex.evt.node.introspected.v1"
            envelope: The deserialized event envelope containing the payload.
                The payload type varies by topic/event type.

        Returns:
            ModelDispatchResult if the implementation returns dispatch metrics/results,
            None if the implementation does not track results. Callers should not
            depend on the return value for correctness.

        Raises:
            InfraDispatchError: If no handler is registered for the topic/message type.
            OnexError: For handler execution failures (implementation-specific).

        Example:
            .. code-block:: python

                # Event bus wiring callback
                async def on_message(message: ProtocolEventMessage) -> None:
                    envelope = deserialize(message)
                    await dispatch_engine.dispatch(topic, envelope)
        """
        ...

    async def dispatch_with_transaction(
        self,
        *,
        topic: str,
        envelope: ModelEventEnvelope[object],
        tx: object,
    ) -> ModelDispatchResult | None:
        """Dispatch an event envelope with database transaction context.

        This method enables transaction-scoped dispatch for correct idempotency
        semantics. The transaction parameter allows handlers to participate in
        the same database transaction as the idempotency insert, ensuring
        exactly-once processing.

        Design Decision:
            The ``tx`` parameter is typed as ``object`` rather than a specific
            database type (e.g., ``asyncpg.Connection``) to avoid leaking
            infrastructure types into the protocol. This keeps the protocol
            generic and allows future flexibility for different database backends.

            Implementers should type-narrow ``tx`` via duck-typing (checking for
            required methods) rather than isinstance checks, per ONEX conventions:

            .. code-block:: python

                from omnibase_core.errors import OnexError

                async def dispatch_with_transaction(
                    self,
                    *,
                    topic: str,
                    envelope: ModelEventEnvelope[object],
                    tx: object,
                ) -> ModelDispatchResult | None:
                    # Duck-type check for required database connection methods
                    if not (hasattr(tx, 'execute') and callable(tx.execute)):
                        raise OnexError(
                            "Transaction object must implement execute() method"
                        )
                    # Use tx with duck-typed interface...

        Thread Safety:
            This method MUST be safe for concurrent calls from multiple
            coroutines. The transaction context (``tx``) is typically
            connection-scoped and should not be shared across coroutines.

        Args:
            topic: The full topic name from which the envelope was consumed.
                Used for routing and logging context.
                Example: "dev.onex.evt.node.introspected.v1"
            envelope: The deserialized event envelope containing the payload.
                The payload type varies by topic/event type.
            tx: Database transaction/connection context. Typed as ``object``
                to avoid infrastructure type leakage; implementations should
                type-narrow based on their database backend (e.g.,
                ``asyncpg.Connection``, ``aiosqlite.Connection``).

        Returns:
            ModelDispatchResult if the implementation returns dispatch metrics/results,
            None if the implementation does not track results. Callers should not
            depend on the return value for correctness.

        Raises:
            InfraDispatchError: If no handler is registered for the topic/message type.
            OnexError: For handler execution failures or if ``tx`` does not implement
                the required database connection interface (duck-type validation).

        Example:
            .. code-block:: python

                # Idempotency consumer with transaction context
                async with pool.acquire() as conn:
                    async with conn.transaction():
                        # Insert idempotency record and dispatch in same transaction
                        await insert_idempotency_record(conn, message_id)
                        await dispatch_engine.dispatch_with_transaction(
                            topic=topic,
                            envelope=envelope,
                            tx=conn,
                        )
                        # Both committed atomically

        Related:
            - OMN-1740: Transaction-scoped dispatch for idempotency
            - dispatch(): Non-transactional dispatch method

        .. versionadded:: 0.2.9
        """
        ...
