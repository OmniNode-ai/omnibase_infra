# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Dispatcher Registry for Message Dispatch Engine.

This module provides the RegistryDispatcher class and ProtocolMessageDispatcher protocol
for managing dispatcher registrations in the dispatch engine. Dispatchers are the execution
units that process messages after routing.

Design Pattern:
    The RegistryDispatcher follows the "freeze after init" pattern (like EnvelopeRouter):
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
    "RegistryDispatcher",
    "ProtocolMessageDispatcher",
]

import logging
import threading
from collections import defaultdict
from uuid import uuid4

from omnibase_core.enums import EnumCoreErrorCode, EnumNodeKind
from omnibase_core.models.errors import ModelOnexError
from omnibase_infra.enums import EnumMessageCategory
from omnibase_infra.models.validation.model_execution_shape_validation import (
    ModelExecutionShapeValidation,
)
from omnibase_infra.protocols.protocol_message_dispatcher import (
    ProtocolMessageDispatcher,
)

logger = logging.getLogger(__name__)


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


class RegistryDispatcher:
    """
    Thread-safe registry for message dispatchers with freeze pattern.

    The RegistryDispatcher manages dispatcher registrations for the dispatch engine.
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

            from omnibase_infra.runtime import RegistryDispatcher

            # 1. Create registry and register dispatchers
            registry = RegistryDispatcher()
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
        Initialize RegistryDispatcher with empty registries.

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

                registry = RegistryDispatcher()

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
                    message="Cannot register dispatcher: RegistryDispatcher is frozen. "
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

                registry = RegistryDispatcher()
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
                    message="Cannot unregister dispatcher: RegistryDispatcher is frozen. "
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

                registry = RegistryDispatcher()
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

                registry = RegistryDispatcher()
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

                registry = RegistryDispatcher()
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

                registry = RegistryDispatcher()
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

                registry = RegistryDispatcher()
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

        This method provides comprehensive validation using duck typing patterns.
        It validates:

        - Property values have correct types (e.g., dispatcher_id is a non-empty str)
        - EnumMessageCategory and EnumNodeKind are the actual enum instances
        - message_types is a set (not just any iterable)
        - handle method is callable

        **When to Use Each Validation Approach**:

        - ``hasattr() + callable()``: Quick structural check via duck typing,
          suitable for type guards or early rejection of obviously invalid objects.
        - ``_validate_dispatcher()``: Comprehensive validation with detailed error
          messages, used internally by ``register_dispatcher()``.

        Args:
            dispatcher: The dispatcher to validate.

        Raises:
            ModelOnexError: If dispatcher is None or lacks required properties.
                Error codes used:
                - INVALID_PARAMETER: Missing or invalid property
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
            str: Format "RegistryDispatcher[dispatchers=N, frozen=bool]"
        """
        return f"RegistryDispatcher[dispatchers={len(self._dispatchers_by_id)}, frozen={self._frozen}]"

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
            f"RegistryDispatcher(dispatchers={dispatcher_repr}, "
            f"categories={category_repr}, frozen={self._frozen})"
        )
