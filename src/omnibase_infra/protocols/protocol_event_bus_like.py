# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Event Bus Protocol for Introspection.

This module provides the minimal protocol interface for event bus compatibility
with infrastructure components like ServiceTimeoutEmitter and MixinNodeIntrospection.

Thread Safety:
    Implementations of ProtocolEventBusLike MUST be thread-safe for concurrent
    async calls. Multiple coroutines may invoke publish methods simultaneously.

    Design Requirements:
        - **Connection Pooling**: Use connection pools for underlying transports
        - **Async-Safe Clients**: Ensure underlying clients (aiokafka, etc.) are async-safe
        - **No Shared Mutable State**: Avoid instance-level caches that could race

    Recommended Patterns:
        - Use asyncio.Lock for any shared mutable state (e.g., circuit breakers)
        - Use MixinAsyncCircuitBreaker for fault tolerance
        - Keep publish operations stateless where possible

Related:
    - docs/architecture/CIRCUIT_BREAKER_THREAD_SAFETY.md: Circuit breaker patterns
    - EventBusKafka: Production implementation with circuit breaker integration
    - EventBusInmemory: Simple implementation for testing

.. versionadded:: 0.4.0
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ProtocolEventBusLike(Protocol):
    """Protocol for event bus compatibility.

    This protocol defines the minimal interface required for an event bus
    to be used with infrastructure components. Any object implementing either
    ``publish_envelope`` or ``publish`` method is compatible.

    The mixin prefers ``publish_envelope`` when available, falling back
    to ``publish`` for raw bytes publishing.

    Thread Safety:
        Implementations MUST be thread-safe for concurrent async calls.

        **Guarantees implementers MUST provide:**
            - Concurrent calls to publish methods are safe
            - Internal state (if any) is protected by appropriate locks
            - Underlying transport clients are async-safe

        **Locking recommendations:**
            - Use asyncio.Lock for shared mutable state
            - Use MixinAsyncCircuitBreaker for fault tolerance (optional)
            - Connection state management should use async-safe patterns

        **What callers can assume:**
            - Multiple coroutines can call publish methods concurrently
            - Order of message delivery may not match call order (Kafka partitioning)
            - Each publish is an independent operation

    Note:
        Method bodies in this Protocol use ``...`` (Ellipsis) rather than
        ``raise NotImplementedError()``. This is the standard Python convention
        for ``typing.Protocol`` classes per PEP 544.

    See Also:
        - docs/architecture/CIRCUIT_BREAKER_THREAD_SAFETY.md: Thread safety patterns
        - MixinAsyncCircuitBreaker: Recommended for production implementations
    """

    async def publish_envelope(
        self,
        envelope: object,
        topic: str,
    ) -> None:
        """Publish an event envelope to a topic.

        Thread Safety:
            This method MUST be safe for concurrent calls from multiple
            coroutines. Implementations should not rely on call ordering.

        Envelope Structure:
            The envelope is typically a Pydantic model (ModelEventEnvelope
            from omnibase_core) with the following fields:

            - correlation_id: UUID for distributed tracing
            - event_type: String identifying the event type
            - payload: The event data (dict or Pydantic model)
            - metadata: Optional metadata dict
            - timestamp: When the event was created

            Implementations should support:
            - Pydantic v2 models (model_dump method)
            - Pydantic v1 models (dict method)
            - Plain dict objects

        Args:
            envelope: The event envelope/model to publish. Typically
                ModelEventEnvelope, but any Pydantic model or dict is supported.
            topic: The topic to publish to.
        """
        ...

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
    ) -> None:
        """Publish raw bytes to a topic (fallback method).

        Thread Safety:
            This method MUST be safe for concurrent calls from multiple
            coroutines. Implementations should not rely on call ordering.

        Args:
            topic: The topic to publish to.
            key: Optional message key as bytes.
            value: The message value as bytes.
        """
        ...


__all__: list[str] = ["ProtocolEventBusLike"]
