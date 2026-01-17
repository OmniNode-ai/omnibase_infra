# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry-based intent model for dynamic type resolution.

This module provides:
- ModelRegistryIntent: Base model for all registration intents
- RegistryIntent: Decorator-based registry for intent type discovery

Design Pattern:
    Instead of maintaining explicit union types like:
        _IntentUnion = ModelConsulIntent | ModelPostgresIntent

    Intent models self-register via decorator:
        @RegistryIntent.register("consul")
        class ModelConsulRegistrationIntent(ModelRegistryIntent): ...

    The registry resolves types dynamically during Pydantic validation,
    enabling new intent types to be added without modifying existing code.

IMPORTANT - Union Sync Requirement (OMN-1007):
    When adding a new intent type, you MUST update BOTH:

    1. Register with @RegistryIntent.register("kind") decorator (this file)
    2. Add the model to ModelRegistrationIntent union in model_registration_intent.py

    The RegistryIntent enables dynamic type resolution for ModelReducerExecutionResult,
    while ModelRegistrationIntent is a static discriminated union used for Pydantic
    field type hints in protocols (e.g., ProtocolEffect.execute_intent).

    Use validate_union_registry_sync() from model_registration_intent.py in tests
    to verify the two systems stay in sync.

Related:
    - model_registration_intent.py: ModelRegistrationIntent discriminated union
    - ProtocolRegistrationIntent: Protocol for duck-typed function signatures
    - validate_union_registry_sync(): Test helper for sync validation
    - OMN-1007: Union reduction refactoring
"""

from __future__ import annotations

import threading
import warnings
from collections.abc import Callable
from typing import ClassVar
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError


class RegistryIntent:
    """Registry for intent model types with decorator-based registration.

    This registry enables dynamic type resolution for Pydantic validation
    without requiring explicit union type definitions.

    Example:
        @RegistryIntent.register("consul")
        class ModelConsulRegistrationIntent(ModelRegistryIntent):
            kind: Literal["consul"] = "consul"
            ...

        # Later, resolve type from kind
        intent_cls = RegistryIntent.get_type("consul")
        intent = intent_cls.model_validate(data)

    Thread Safety:
        All registration and mutation operations are protected by a class-level
        threading.Lock to ensure thread-safe access. Read operations (get_type,
        get_all_types, is_registered) are atomic dictionary operations and safe
        for concurrent access. The lock ensures that check-and-set operations
        in register() are atomic, preventing race conditions.
    """

    _types: ClassVar[dict[str, type[ModelRegistryIntent]]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def register(cls, kind: str) -> Callable[[type], type]:
        """Decorator to register an intent model type.

        Args:
            kind: The intent kind identifier (e.g., "consul", "postgres").
                  Must match the model's `kind` field value.

        Returns:
            Decorator function that registers the class and returns it unchanged.

        Raises:
            ValueError: If kind is already registered (prevents duplicates).

        Example:
            @RegistryIntent.register("consul")
            class ModelConsulRegistrationIntent(ModelRegistryIntent):
                kind: Literal["consul"] = "consul"
                service_name: str
                ...
        """

        def decorator(intent_cls: type) -> type:
            # Thread-safe registration with atomic check-and-set
            with cls._lock:
                if kind in cls._types:
                    context = ModelInfraErrorContext(
                        transport_type=EnumInfraTransportType.RUNTIME,
                        operation="register_intent",
                        correlation_id=uuid4(),
                    )
                    raise ProtocolConfigurationError(
                        f"Intent kind '{kind}' already registered to {cls._types[kind].__name__}. "
                        f"Cannot register {intent_cls.__name__}.",
                        context=context,
                    )
                cls._types[kind] = intent_cls
            return intent_cls

        return decorator

    @classmethod
    def get_type(cls, kind: str) -> type[ModelRegistryIntent]:
        """Get the intent model class for a given kind.

        Args:
            kind: The intent kind identifier.

        Returns:
            The registered intent model class.

        Raises:
            KeyError: If kind is not registered.

        Example:
            intent_cls = RegistryIntent.get_type("consul")
            intent = intent_cls.model_validate({"kind": "consul", ...})
        """
        if kind not in cls._types:
            registered = ", ".join(sorted(cls._types.keys())) or "(none)"
            raise KeyError(
                f"Unknown intent kind '{kind}'. Registered kinds: {registered}"
            )
        return cls._types[kind]

    @classmethod
    def get_all_types(cls) -> dict[str, type[ModelRegistryIntent]]:
        """Get all registered intent types.

        Returns:
            Dict mapping kind strings to intent model classes.

        Example:
            all_types = RegistryIntent.get_all_types()
            for kind, intent_cls in all_types.items():
                print(f"{kind}: {intent_cls.__name__}")
        """
        return dict(cls._types)

    @classmethod
    def is_registered(cls, kind: str) -> bool:
        """Check if a kind is registered.

        Args:
            kind: The intent kind identifier.

        Returns:
            True if registered, False otherwise.

        Example:
            if RegistryIntent.is_registered("consul"):
                intent_cls = RegistryIntent.get_type("consul")
        """
        return kind in cls._types

    @classmethod
    def clear(cls) -> None:
        """Clear all registered types.

        Warning:
            This method is intended for **testing purposes only**.
            Calling it in production code will emit a warning.
            It breaks the immutability guarantee after startup.

        Thread Safety:
            This method is protected by the class-level lock to ensure
            thread-safe clearing of the registry.
        """
        warnings.warn(
            "RegistryIntent.clear() is intended for testing only. "
            "Do not use in production code.",
            UserWarning,
            stacklevel=2,
        )
        with cls._lock:
            cls._types.clear()


class ModelRegistryIntent(BaseModel):
    """Base model for all registry-managed registration intents.

    All concrete intent models MUST:
    1. Inherit from this base class
    2. Use @RegistryIntent.register("kind") decorator
    3. Define kind as Literal["kind"] with matching default
    4. Be frozen (immutable) for thread safety

    This base class defines the common interface that all intents share,
    enabling type-safe access to common fields without type narrowing.

    Example:
        @RegistryIntent.register("consul")
        class ModelConsulRegistrationIntent(ModelRegistryIntent):
            kind: Literal["consul"] = "consul"
            service_name: str
            ...

    Attributes:
        kind: Intent kind identifier used for type discrimination.
        operation: The operation to perform (e.g., 'register', 'upsert', 'deregister').
        node_id: The node ID this intent applies to.
        correlation_id: Correlation ID for distributed tracing.

    Related:
        - RegistryIntent: Registration decorator and type lookup
        - ProtocolRegistrationIntent: Protocol for duck-typed signatures
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    kind: str
    """Intent kind identifier used for type discrimination."""

    operation: str
    """The operation to perform (e.g., 'register', 'upsert', 'deregister')."""

    node_id: UUID
    """The node ID this intent applies to."""

    correlation_id: UUID
    """Correlation ID for distributed tracing."""


__all__ = [
    "RegistryIntent",
    "ModelRegistryIntent",
]
