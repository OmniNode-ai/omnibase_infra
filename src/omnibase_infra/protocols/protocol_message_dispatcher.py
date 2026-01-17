# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Protocol for Message Dispatchers in the ONEX Dispatch Engine.

This module defines the ProtocolMessageDispatcher protocol for category-based
message dispatchers. Dispatchers are the execution units that process messages
after routing in the dispatch engine.

Thread Safety:
    Dispatcher implementations may be invoked concurrently from the dispatch engine.
    The same dispatcher instance may be called from multiple coroutines simultaneously.

    Design Requirements:
        - Stateless Dispatchers (Recommended): Keep dispatchers stateless by
          extracting all needed data from the envelope. This is the safest
          approach and requires no synchronization.
        - Stateful Dispatchers: If state is required, use appropriate
          synchronization primitives (asyncio.Lock for async state).

Related:
    - OMN-934: Dispatcher registry for message dispatch engine
    - RegistryDispatcher: Registry for managing dispatcher registrations
    - ModelDispatchResult: Result model for dispatch operations

.. versionadded:: 0.5.0
"""

from __future__ import annotations

__all__ = ["ProtocolMessageDispatcher"]

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from omnibase_core.enums import EnumNodeKind
from omnibase_infra.enums import EnumMessageCategory
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult

if TYPE_CHECKING:
    from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope


@runtime_checkable
class ProtocolMessageDispatcher(Protocol):
    """
    Protocol for category-based message dispatchers in the dispatch engine.

    Message dispatchers are the execution units that process messages after routing.
    Each dispatcher is classified by:
    - category: The message category it handles (EVENT, COMMAND, INTENT)
    - message_types: Specific message types it accepts (empty = all)
    - node_kind: The ONEX node kind this dispatcher represents

    Protocol Verification:
        Per ONEX conventions, protocol compliance is verified via duck typing rather
        than isinstance checks. Verify required methods and properties exist:

        **Validation Approaches**:

        1. **Duck Typing Check** (recommended for quick structural validation):
           Use when you need to verify an object implements the dispatcher interface
           before passing it to components that expect a dispatcher.

           .. code-block:: python

               # Verify required properties and methods exist
               if (hasattr(dispatcher, 'dispatcher_id') and
                   hasattr(dispatcher, 'category') and
                   hasattr(dispatcher, 'handle') and callable(dispatcher.handle)):
                   registry.register_dispatcher(dispatcher)
               else:
                   raise TypeError("Object does not implement ProtocolMessageDispatcher")

        2. **RegistryDispatcher Validation** (comprehensive validation):
           The ``RegistryDispatcher.register_dispatcher()`` method performs thorough
           validation including:
           - All required properties exist and have correct types
           - Execution shape is valid (category -> node_kind combination)
           - ``handle()`` method is callable

           This is the recommended approach for production registration as it
           provides detailed error messages for debugging.

        **Note**: For complete type safety, use static type checking (mypy)
        in addition to duck typing verification.

    Example:
        .. code-block:: python

            from omnibase_core.enums.enum_node_kind import EnumNodeKind
            from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
            from omnibase_infra.enums import EnumMessageCategory
            from omnibase_infra.enums.enum_dispatch_status import EnumDispatchStatus
            from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
            from omnibase_infra.protocols import ProtocolMessageDispatcher

            class UserEventDispatcher:
                '''Dispatcher for user-related events.'''

                @property
                def dispatcher_id(self) -> str:
                    return "user-event-dispatcher"

                @property
                def category(self) -> EnumMessageCategory:
                    return EnumMessageCategory.EVENT

                @property
                def message_types(self) -> set[str]:
                    return {"UserCreated", "UserUpdated", "UserDeleted"}

                @property
                def node_kind(self) -> EnumNodeKind:
                    return EnumNodeKind.REDUCER

                async def handle(
                    self, envelope: ModelEventEnvelope[object]
                ) -> ModelDispatchResult:
                    # Process the event
                    return ModelDispatchResult(
                        status=EnumDispatchStatus.SUCCESS,
                        topic="user.events",
                        dispatcher_id=self.dispatcher_id,
                    )

            # Verify protocol compliance via duck typing (per ONEX conventions)
            dispatcher = UserEventDispatcher()
            assert hasattr(dispatcher, 'dispatcher_id')
            assert hasattr(dispatcher, 'category')
            assert hasattr(dispatcher, 'handle') and callable(dispatcher.handle)

    Attributes:
        dispatcher_id: Unique identifier for this dispatcher.
        category: The message category this dispatcher processes.
        message_types: Specific message types this dispatcher accepts.
            Empty set means dispatcher accepts all message types in its category.
        node_kind: The ONEX node kind this dispatcher represents.

    Note:
        Method bodies in this Protocol use ``...`` (Ellipsis) rather than
        ``raise NotImplementedError()``. This is the standard Python convention
        for ``typing.Protocol`` classes per PEP 544.

    .. versionadded:: 0.5.0
    """

    @property
    def dispatcher_id(self) -> str:
        """
        Return the unique identifier for this dispatcher.

        The dispatcher ID is used for:
        - Registration and lookup in the registry
        - Tracing and observability
        - Error reporting and debugging

        Returns:
            str: Unique dispatcher identifier (e.g., "user-event-dispatcher")
        """
        ...

    @property
    def category(self) -> EnumMessageCategory:
        """
        Return the message category this dispatcher processes.

        Dispatchers are classified by the category of messages they can handle:
        - EVENT: Past-tense immutable facts
        - COMMAND: Imperative action requests
        - INTENT: Goal-oriented desires

        Returns:
            EnumMessageCategory: The message category (EVENT, COMMAND, or INTENT)
        """
        ...

    @property
    def message_types(self) -> set[str]:
        """
        Return the specific message types this dispatcher accepts.

        When empty, the dispatcher accepts all message types within its category.
        When non-empty, only the listed message types are accepted.

        Returns:
            set[str]: Set of accepted message types, or empty for all types
        """
        ...

    @property
    def node_kind(self) -> EnumNodeKind:
        """
        Return the ONEX node kind this dispatcher represents.

        The node kind determines valid execution shapes:
        - REDUCER: Handles EVENT messages for state aggregation
        - ORCHESTRATOR: Handles EVENT and COMMAND messages for coordination
        - EFFECT: Handles INTENT and COMMAND messages for external I/O

        Returns:
            EnumNodeKind: The node kind (REDUCER, ORCHESTRATOR, EFFECT, etc.)
        """
        ...

    async def handle(
        self,
        envelope: ModelEventEnvelope[object],
    ) -> ModelDispatchResult:
        """
        Handle the given envelope and return a dispatch result.

        This is the primary execution method. The dispatcher receives an input
        envelope, processes it according to its category and node kind,
        and returns a dispatch result indicating success or failure.

        Typing Note (ModelEventEnvelope[object]):
            The envelope parameter uses ``ModelEventEnvelope[object]`` instead of
            ``Any`` per CLAUDE.md guidance: "Use ``object`` for generic payloads".

            This is intentional:
            - CLAUDE.md mandates "NEVER use ``Any``" and specifies ``object`` for
              generic payloads that can accept multiple event types
            - Generic dispatchers must handle multiple event types at runtime;
              the dispatch engine routes based on topic/category/message_type,
              not payload shape
            - Payload extraction uses ``isinstance()`` type guards for runtime
              safety (see dispatcher implementations)
            - ``object`` provides better type safety than ``Any`` while allowing
              the flexibility required for polymorphic dispatch

            For type-specific processing, dispatcher implementations should use
            type guards to narrow the payload type:

            .. code-block:: python

                payload = envelope.payload
                if isinstance(payload, SpecificEventType):
                    # Type-safe processing here
                    process_specific_event(payload)

        Args:
            envelope: The input envelope containing the message to process.

        Returns:
            ModelDispatchResult: The result of the dispatch operation.
        """
        ...
