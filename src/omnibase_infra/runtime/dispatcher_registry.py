# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Dispatcher Registry for Message Dispatch Engine.

This module provides the DispatcherRegistry class and ProtocolMessageDispatcher protocol
for managing dispatcher registrations in the dispatch engine. Dispatchers are the execution
units that process messages after routing.

Design Pattern:
    The DispatcherRegistry follows the "freeze after init" pattern (like EnvelopeRouter):
    1. Registration phase: Register dispatchers during startup (single-threaded)
    2. Freeze: Call freeze() to prevent further modifications
    3. Execution phase: Thread-safe read access for dispatcher lookup

    This pattern ensures:
    - No runtime registration overhead (no locking on reads)
    - Thread-safe concurrent access after freeze
    - Clear separation between configuration and execution phases

Thread Safety:
    - Registration methods are protected by threading.Lock
    - After freeze(), the registry is read-only and thread-safe
    - Execution shape validation occurs at registration time

Related:
    - OMN-934: Dispatcher registry for message dispatch engine
    - EnvelopeRouter: Uses similar freeze-after-init pattern
    - ModelDispatcherRegistration: Dispatcher metadata model
    - ModelExecutionShapeValidation: Validates execution shapes

.. versionadded:: 0.4.0
"""

from __future__ import annotations

__all__ = [
    "ProtocolMessageDispatcher",
    "DispatcherRegistry",
]

import logging
import threading
from collections import defaultdict
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from uuid import uuid4

from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_core.models.errors.model_onex_error import ModelOnexError

from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.models.validation.model_execution_shape_validation import (
    ModelExecutionShapeValidation,
)

if TYPE_CHECKING:
    from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope

logger = logging.getLogger(__name__)


@runtime_checkable
class ProtocolMessageDispatcher(Protocol):
    """
    Protocol for category-based message dispatchers in the dispatch engine.

    Message dispatchers are the execution units that process messages after routing.
    Each dispatcher is classified by:
    - category: The message category it handles (EVENT, COMMAND, INTENT)
    - message_types: Specific message types it accepts (empty = all)
    - node_kind: The ONEX node kind this dispatcher represents

    Thread Safety:
        WARNING: Dispatcher implementations may be invoked concurrently from the
        dispatch engine. The same dispatcher instance may be called from multiple
        coroutines simultaneously.

        Design Requirements:
            - **Stateless Dispatchers (Recommended)**: Keep dispatchers stateless by
              extracting all needed data from the envelope. This is the safest
              approach and requires no synchronization.
            - **Stateful Dispatchers**: If state is required, use appropriate
              synchronization primitives (asyncio.Lock for async state).

    Example:
        .. code-block:: python

            from omnibase_infra.runtime.dispatcher_registry import ProtocolMessageDispatcher
            from omnibase_infra.enums import EnumMessageCategory
            from omnibase_core.enums.enum_node_kind import EnumNodeKind

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

            # Verify protocol compliance
            dispatcher: ProtocolMessageDispatcher = UserEventDispatcher()
            assert isinstance(dispatcher, ProtocolMessageDispatcher)

    Attributes:
        dispatcher_id: Unique identifier for this dispatcher.
        category: The message category this dispatcher processes.
        message_types: Specific message types this dispatcher accepts.
            Empty set means dispatcher accepts all message types in its category.
        node_kind: The ONEX node kind this dispatcher represents.

    Note:
        Method bodies in this Protocol use ``...`` (Ellipsis) rather than
        ``raise NotImplementedError()``. This is the standard Python convention
        for ``typing.Protocol`` classes per PEP 544. Protocol classes define
        structural subtyping (duck typing) interfaces, not inheritance-based
        abstract base classes. Use ``raise NotImplementedError`` only for
        ``abc.ABC`` abstract base classes, not for Protocol definitions.

    .. versionadded:: 0.4.0
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

        Example:
            .. code-block:: python

                @property
                def dispatcher_id(self) -> str:
                    return "order-processor"
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

        Example:
            .. code-block:: python

                @property
                def category(self) -> EnumMessageCategory:
                    return EnumMessageCategory.EVENT
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

        Example:
            .. code-block:: python

                @property
                def message_types(self) -> set[str]:
                    # Accept only specific event types
                    return {"UserCreated", "UserUpdated"}

                @property
                def message_types(self) -> set[str]:
                    # Accept all event types in category
                    return set()
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

        Example:
            .. code-block:: python

                @property
                def node_kind(self) -> EnumNodeKind:
                    return EnumNodeKind.REDUCER
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

        Args:
            envelope: The input envelope containing the message to process.
                The payload contains category-specific data.

        Returns:
            ModelDispatchResult: The result of the dispatch operation with:
                - status: Success, error, timeout, etc.
                - dispatcher_id: This dispatcher's ID
                - duration_ms: Processing time
                - error_message: Error details if failed

        Example:
            .. code-block:: python

                async def handle(
                    self, envelope: ModelEventEnvelope[object]
                ) -> ModelDispatchResult:
                    try:
                        # Process the event
                        result = await self._process_event(envelope.payload)

                        return ModelDispatchResult(
                            status=EnumDispatchStatus.SUCCESS,
                            topic="user.events",
                            dispatcher_id=self.dispatcher_id,
                            output_count=1,
                        )
                    except Exception as e:
                        return ModelDispatchResult(
                            status=EnumDispatchStatus.DISPATCHER_ERROR,
                            topic="user.events",
                            dispatcher_id=self.dispatcher_id,
                            error_message=str(e),
                        )
        """
        ...


class DispatchEntryInternal:
    """
    Internal entry for a registered dispatcher.

    Stores the dispatcher instance and its registration metadata.
    This is an implementation detail and not part of the public API.
    """

    __slots__ = ("dispatcher", "message_types", "registration_id")

    def __init__(
        self,
        dispatcher: ProtocolMessageDispatcher,
        message_types: set[str],
        registration_id: str,
    ) -> None:
        self.dispatcher = dispatcher
        self.message_types = message_types
        self.registration_id = registration_id


class DispatcherRegistry:
    """
    Thread-safe registry for message dispatchers with freeze pattern.

    The DispatcherRegistry manages dispatcher registrations for the dispatch engine.
    It stores dispatchers by category and message type, validates execution shapes
    at registration time, and provides efficient lookup for dispatching.

    Design Pattern:
        The registry follows the "freeze after init" pattern:
        1. Registration phase: Register dispatchers during startup
        2. Freeze: Call freeze() to lock the registry
        3. Execution phase: Thread-safe reads for dispatcher lookup

    Thread Safety:
        - Registration methods are protected by threading.Lock
        - After freeze(), the registry is read-only and safe for concurrent access
        - Execution shape validation occurs at registration time

    Execution Shape Validation:
        At registration time, the registry validates that the dispatcher's category
        and node_kind combination forms a valid execution shape per ONEX standards:
        - EVENT -> REDUCER (valid)
        - EVENT -> ORCHESTRATOR (valid)
        - COMMAND -> ORCHESTRATOR (valid)
        - COMMAND -> EFFECT (valid)
        - INTENT -> EFFECT (valid)
        - Other combinations are rejected

    Example:
        .. code-block:: python

            from omnibase_core.runtime import DispatcherRegistry

            # 1. Create registry and register dispatchers
            registry = DispatcherRegistry()
            registry.register_dispatcher(user_event_dispatcher)
            registry.register_dispatcher(order_command_dispatcher)

            # 2. Freeze to prevent modifications
            registry.freeze()

            # 3. Look up dispatchers (thread-safe after freeze)
            dispatchers = registry.get_dispatchers(
                category=EnumMessageCategory.EVENT,
                message_type="UserCreated",
            )

    Attributes:
        _dispatchers_by_category: Dispatchers organized by category -> list of entries
        _dispatchers_by_id: Dispatchers indexed by dispatcher_id for fast lookup
        _frozen: If True, registration is disabled
        _registration_lock: Lock protecting registration methods

    See Also:
        - :class:`ProtocolMessageDispatcher`: Dispatcher protocol definition
        - :class:`~omnibase_core.runtime.envelope_router.EnvelopeRouter`:
          Similar freeze-after-init pattern
        - :class:`~omnibase_infra.models.validation.model_execution_shape_validation.ModelExecutionShapeValidation`:
          Execution shape validation

    .. versionadded:: 0.4.0
    """

    def __init__(self) -> None:
        """
        Initialize DispatcherRegistry with empty registries.

        Creates empty dispatcher registries. Dispatchers must be registered before
        dispatch. Call ``freeze()`` after registration to prevent further
        modifications and enable safe concurrent access.
        """
        # Dispatchers organized by category
        self._dispatchers_by_category: dict[
            EnumMessageCategory, list[DispatchEntryInternal]
        ] = defaultdict(list)
        # Dispatchers indexed by dispatcher_id for fast lookup and duplicate detection
        self._dispatchers_by_id: dict[str, DispatchEntryInternal] = {}
        # Frozen flag
        self._frozen: bool = False
        # Lock protects registration methods
        self._registration_lock: threading.Lock = threading.Lock()

    def register_dispatcher(
        self,
        dispatcher: ProtocolMessageDispatcher,
        message_types: set[str] | None = None,
    ) -> None:
        """
        Register a dispatcher for message dispatch.

        Registers the dispatcher and validates that its category/node_kind
        combination forms a valid execution shape.

        Args:
            dispatcher: A dispatcher implementing ProtocolMessageDispatcher.
            message_types: Optional override for message types.
                If None, uses dispatcher.message_types property.
                If empty set, dispatcher accepts all message types in category.

        Raises:
            ModelOnexError: If registry is frozen (INVALID_STATE).
            ModelOnexError: If dispatcher is None (INVALID_PARAMETER).
            ModelOnexError: If dispatcher lacks required properties (INVALID_PARAMETER).
            ModelOnexError: If dispatcher_id is already registered (DUPLICATE_REGISTRATION).
            ModelOnexError: If execution shape is invalid (VALIDATION_FAILED).

        Example:
            .. code-block:: python

                registry = DispatcherRegistry()

                # Register with dispatcher's message_types
                registry.register_dispatcher(user_event_dispatcher)

                # Register with custom message_types
                registry.register_dispatcher(
                    order_dispatcher,
                    message_types={"OrderCreated", "OrderUpdated"},
                )

                # After registration, freeze
                registry.freeze()

        Thread Safety:
            This method is protected by an internal lock to ensure atomic
            validation and registration.

        .. versionadded:: 0.4.0
        """
        # Validate dispatcher outside lock
        self._validate_dispatcher(dispatcher)

        # Get dispatcher properties
        dispatcher_id = dispatcher.dispatcher_id
        category = dispatcher.category
        node_kind = dispatcher.node_kind
        effective_message_types = (
            message_types if message_types is not None else dispatcher.message_types
        )

        # Validate execution shape outside lock
        self._validate_execution_shape(dispatcher_id, category, node_kind)

        # Create registration entry
        registration_id = str(uuid4())
        entry = DispatchEntryInternal(
            dispatcher=dispatcher,
            message_types=effective_message_types,
            registration_id=registration_id,
        )

        # Lock for atomic frozen check + registration
        with self._registration_lock:
            if self._frozen:
                raise ModelOnexError(
                    message="Cannot register dispatcher: DispatcherRegistry is frozen. "
                    "Registration is not allowed after freeze() has been called.",
                    error_code=EnumCoreErrorCode.INVALID_STATE,
                )

            if dispatcher_id in self._dispatchers_by_id:
                raise ModelOnexError(
                    message=f"Dispatcher with ID '{dispatcher_id}' is already registered. "
                    "Use unregister_dispatcher() first to replace.",
                    error_code=EnumCoreErrorCode.DUPLICATE_REGISTRATION,
                )

            # Register in both indexes
            self._dispatchers_by_id[dispatcher_id] = entry
            self._dispatchers_by_category[category].append(entry)

            logger.debug(
                "Registered dispatcher '%s' for category '%s' with %d message types",
                dispatcher_id,
                category.value,
                len(effective_message_types) if effective_message_types else 0,
            )

    def unregister_dispatcher(self, dispatcher_id: str) -> bool:
        """
        Unregister a dispatcher by its ID.

        Removes the dispatcher from the registry. Returns True if the dispatcher
        was found and removed, False if not found.

        Args:
            dispatcher_id: The unique identifier of the dispatcher to remove.

        Returns:
            bool: True if dispatcher was found and removed, False if not found.

        Raises:
            ModelOnexError: If registry is frozen (INVALID_STATE).

        Example:
            .. code-block:: python

                registry = DispatcherRegistry()
                registry.register_dispatcher(dispatcher)

                # Remove the dispatcher
                removed = registry.unregister_dispatcher("my-dispatcher")
                assert removed is True

                # Try to remove again
                removed = registry.unregister_dispatcher("my-dispatcher")
                assert removed is False

        Thread Safety:
            This method is protected by an internal lock.

        .. versionadded:: 0.4.0
        """
        with self._registration_lock:
            if self._frozen:
                raise ModelOnexError(
                    message="Cannot unregister dispatcher: DispatcherRegistry is frozen. "
                    "Modification is not allowed after freeze() has been called.",
                    error_code=EnumCoreErrorCode.INVALID_STATE,
                )

            if dispatcher_id not in self._dispatchers_by_id:
                return False

            entry = self._dispatchers_by_id.pop(dispatcher_id)

            # Remove from category index
            category = entry.dispatcher.category
            category_list = self._dispatchers_by_category[category]
            self._dispatchers_by_category[category] = [
                e for e in category_list if e.registration_id != entry.registration_id
            ]

            logger.debug("Unregistered dispatcher '%s'", dispatcher_id)
            return True

    def get_dispatchers(
        self,
        category: EnumMessageCategory,
        message_type: str | None = None,
    ) -> list[ProtocolMessageDispatcher]:
        """
        Get dispatchers that can process the given category and message type.

        Returns dispatchers matching the category and optionally filtering by
        message type. Dispatchers with empty message_types accept all message
        types in their category.

        Args:
            category: The message category to look up.
            message_type: Optional specific message type to filter by.

        Returns:
            list[ProtocolMessageDispatcher]: List of matching dispatchers.
                Empty list if no dispatchers match.

        Raises:
            ModelOnexError: If registry is not frozen (INVALID_STATE).

        Example:
            .. code-block:: python

                registry = DispatcherRegistry()
                registry.register_dispatcher(user_dispatcher)
                registry.freeze()

                # Get all EVENT dispatchers
                dispatchers = registry.get_dispatchers(EnumMessageCategory.EVENT)

                # Get dispatchers for specific message type
                dispatchers = registry.get_dispatchers(
                    EnumMessageCategory.EVENT,
                    message_type="UserCreated",
                )

        Thread Safety:
            This method is safe for concurrent access after freeze().

        .. versionadded:: 0.4.0
        """
        # Enforce freeze contract for thread safety
        if not self._frozen:
            raise ModelOnexError(
                message="get_dispatchers() called before freeze(). "
                "Registration MUST complete and freeze() MUST be called before lookup. "
                "This is required for thread safety.",
                error_code=EnumCoreErrorCode.INVALID_STATE,
            )

        entries = self._dispatchers_by_category.get(category, [])
        result: list[ProtocolMessageDispatcher] = []

        for entry in entries:
            # Check if dispatcher accepts this message type
            if message_type is None:
                # No type filter - include all dispatchers for category
                result.append(entry.dispatcher)
            elif not entry.message_types:
                # Empty message_types means accept all
                result.append(entry.dispatcher)
            elif message_type in entry.message_types:
                # Specific message type matches
                result.append(entry.dispatcher)

        return result

    def get_dispatcher_by_id(
        self, dispatcher_id: str
    ) -> ProtocolMessageDispatcher | None:
        """
        Get a dispatcher by its unique ID.

        Args:
            dispatcher_id: The dispatcher's unique identifier.

        Returns:
            ProtocolMessageDispatcher or None if not found.

        Raises:
            ModelOnexError: If registry is not frozen (INVALID_STATE).

        Example:
            .. code-block:: python

                registry = DispatcherRegistry()
                registry.register_dispatcher(my_dispatcher)
                registry.freeze()

                dispatcher = registry.get_dispatcher_by_id("my-dispatcher")
                if dispatcher:
                    result = await dispatcher.handle(envelope)

        Thread Safety:
            This method is safe for concurrent access after freeze().

        .. versionadded:: 0.4.0
        """
        if not self._frozen:
            raise ModelOnexError(
                message="get_dispatcher_by_id() called before freeze(). "
                "Registration MUST complete and freeze() MUST be called before lookup.",
                error_code=EnumCoreErrorCode.INVALID_STATE,
            )

        entry = self._dispatchers_by_id.get(dispatcher_id)
        return entry.dispatcher if entry else None

    def freeze(self) -> None:
        """
        Freeze the registry to prevent further modifications.

        Once frozen, any calls to ``register_dispatcher()`` or ``unregister_dispatcher()``
        will raise ModelOnexError with INVALID_STATE error code. This enforces
        the read-only-after-init pattern for thread safety.

        The freeze operation is idempotent - calling freeze() multiple times
        has no additional effect.

        Example:
            .. code-block:: python

                registry = DispatcherRegistry()
                registry.register_dispatcher(dispatcher)

                # Freeze to prevent modifications
                registry.freeze()
                assert registry.is_frozen

                # Subsequent registration attempts raise INVALID_STATE
                registry.register_dispatcher(another_dispatcher)  # Raises!

        Note:
            This is a one-way operation - there is no ``unfreeze()`` method
            by design, as unfreezing would defeat the thread-safety guarantees.

        Thread Safety:
            This method is protected by an internal lock to ensure atomic
            setting of the frozen flag.

        .. versionadded:: 0.4.0
        """
        with self._registration_lock:
            self._frozen = True

    @property
    def is_frozen(self) -> bool:
        """
        Check if the registry is frozen.

        Returns:
            bool: True if frozen and registration is disabled,
                False if registration is still allowed.

        Example:
            .. code-block:: python

                registry = DispatcherRegistry()
                assert not registry.is_frozen

                registry.freeze()
                assert registry.is_frozen

        .. versionadded:: 0.4.0
        """
        return self._frozen

    @property
    def dispatcher_count(self) -> int:
        """
        Get the total number of registered dispatchers.

        Returns:
            int: Number of registered dispatchers.

        Example:
            .. code-block:: python

                registry = DispatcherRegistry()
                assert registry.dispatcher_count == 0

                registry.register_dispatcher(dispatcher)
                assert registry.dispatcher_count == 1

        .. versionadded:: 0.4.0
        """
        return len(self._dispatchers_by_id)

    def _validate_dispatcher(
        self, dispatcher: ProtocolMessageDispatcher | None
    ) -> None:
        """
        Validate that a dispatcher meets the ProtocolMessageDispatcher requirements.

        Args:
            dispatcher: The dispatcher to validate.

        Raises:
            ModelOnexError: If dispatcher is None or lacks required properties.
        """
        if dispatcher is None:
            raise ModelOnexError(
                message="Cannot register None dispatcher. Dispatcher is required.",
                error_code=EnumCoreErrorCode.INVALID_PARAMETER,
            )

        # Validate dispatcher_id property
        if not hasattr(dispatcher, "dispatcher_id"):
            raise ModelOnexError(
                message="Dispatcher must have 'dispatcher_id' property. "
                "Ensure dispatcher implements ProtocolMessageDispatcher interface.",
                error_code=EnumCoreErrorCode.INVALID_PARAMETER,
            )

        dispatcher_id = dispatcher.dispatcher_id
        if not isinstance(dispatcher_id, str) or not dispatcher_id:
            raise ModelOnexError(
                message=f"Dispatcher dispatcher_id must be non-empty string, got {type(dispatcher_id).__name__}",
                error_code=EnumCoreErrorCode.INVALID_PARAMETER,
            )

        # Validate category property
        if not hasattr(dispatcher, "category"):
            raise ModelOnexError(
                message="Dispatcher must have 'category' property. "
                "Ensure dispatcher implements ProtocolMessageDispatcher interface.",
                error_code=EnumCoreErrorCode.INVALID_PARAMETER,
            )

        category = dispatcher.category
        if not isinstance(category, EnumMessageCategory):
            raise ModelOnexError(
                message=f"Dispatcher category must be EnumMessageCategory, got {type(category).__name__}",
                error_code=EnumCoreErrorCode.INVALID_PARAMETER,
            )

        # Validate node_kind property
        if not hasattr(dispatcher, "node_kind"):
            raise ModelOnexError(
                message="Dispatcher must have 'node_kind' property. "
                "Ensure dispatcher implements ProtocolMessageDispatcher interface.",
                error_code=EnumCoreErrorCode.INVALID_PARAMETER,
            )

        node_kind = dispatcher.node_kind
        if not isinstance(node_kind, EnumNodeKind):
            raise ModelOnexError(
                message=f"Dispatcher node_kind must be EnumNodeKind, got {type(node_kind).__name__}",
                error_code=EnumCoreErrorCode.INVALID_PARAMETER,
            )

        # Validate message_types property
        if not hasattr(dispatcher, "message_types"):
            raise ModelOnexError(
                message="Dispatcher must have 'message_types' property. "
                "Ensure dispatcher implements ProtocolMessageDispatcher interface.",
                error_code=EnumCoreErrorCode.INVALID_PARAMETER,
            )

        message_types = dispatcher.message_types
        if not isinstance(message_types, set):
            raise ModelOnexError(
                message=f"Dispatcher message_types must be set[str], got {type(message_types).__name__}",
                error_code=EnumCoreErrorCode.INVALID_PARAMETER,
            )

        # Validate handle method is callable
        if not hasattr(dispatcher, "handle") or not callable(
            getattr(dispatcher, "handle", None)
        ):
            raise ModelOnexError(
                message="Dispatcher must have callable 'handle' method. "
                "Ensure dispatcher implements ProtocolMessageDispatcher interface.",
                error_code=EnumCoreErrorCode.INVALID_PARAMETER,
            )

    def _validate_execution_shape(
        self,
        dispatcher_id: str,
        category: EnumMessageCategory,
        node_kind: EnumNodeKind,
    ) -> None:
        """
        Validate that the dispatcher's category/node_kind forms a valid execution shape.

        Uses ModelExecutionShapeValidation to check ONEX architectural compliance.

        Args:
            dispatcher_id: Dispatcher ID for error messages.
            category: The message category.
            node_kind: The target node kind.

        Raises:
            ModelOnexError: If execution shape is not valid (VALIDATION_FAILED).
        """
        validation = ModelExecutionShapeValidation.validate_shape(
            source_category=category,
            target_node_kind=node_kind,
        )

        if not validation.is_allowed:
            raise ModelOnexError(
                message=f"Dispatcher '{dispatcher_id}' has invalid execution shape: "
                f"{category.value} -> {node_kind.value}. {validation.rationale}",
                error_code=EnumCoreErrorCode.VALIDATION_FAILED,
            )

    def __str__(self) -> str:
        """
        Human-readable string representation.

        Returns:
            str: Format "DispatcherRegistry[dispatchers=N, frozen=bool]"
        """
        return f"DispatcherRegistry[dispatchers={len(self._dispatchers_by_id)}, frozen={self._frozen}]"

    def __repr__(self) -> str:
        """
        Detailed representation for debugging.

        Returns:
            str: Detailed format including dispatcher IDs and categories.
        """
        dispatcher_ids = list(self._dispatchers_by_id.keys())
        categories = list(self._dispatchers_by_category.keys())

        # Limit output for large registries
        if len(dispatcher_ids) > 10:
            dispatcher_repr = f"<{len(dispatcher_ids)} dispatchers>"
        else:
            dispatcher_repr = repr(dispatcher_ids)

        if len(categories) > 5:
            category_repr = f"<{len(categories)} categories>"
        else:
            category_repr = repr([c.value for c in categories])

        return (
            f"DispatcherRegistry(dispatchers={dispatcher_repr}, "
            f"categories={category_repr}, frozen={self._frozen})"
        )
