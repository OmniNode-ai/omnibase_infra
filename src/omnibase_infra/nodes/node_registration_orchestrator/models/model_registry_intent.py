# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry-based intent model for dynamic type resolution.

This module provides:
- ModelRegistryIntent: Base model for all registration intents
- IntentRegistry: Decorator-based registry for intent type discovery

Design Pattern:
    Instead of maintaining explicit union types like:
        _IntentUnion = ModelConsulIntent | ModelPostgresIntent

    Intent models self-register via decorator:
        @IntentRegistry.register("consul")
        class ModelConsulRegistrationIntent(ModelRegistryIntent): ...

    The registry resolves types dynamically during Pydantic validation,
    enabling new intent types to be added without modifying existing code.

Related:
    - ProtocolRegistrationIntent: Protocol for duck-typed function signatures
    - OMN-1007: Union reduction refactoring
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class IntentRegistry:
    """Registry for intent model types with decorator-based registration.

    This registry enables dynamic type resolution for Pydantic validation
    without requiring explicit union type definitions.

    Example:
        @IntentRegistry.register("consul")
        class ModelConsulRegistrationIntent(ModelRegistryIntent):
            kind: Literal["consul"] = "consul"
            ...

        # Later, resolve type from kind
        intent_cls = IntentRegistry.get_type("consul")
        intent = intent_cls.model_validate(data)

    Thread Safety:
        The registry is populated at module import time (class definition).
        After startup, it is read-only and thread-safe for concurrent access.
    """

    _types: ClassVar[dict[str, type[ModelRegistryIntent]]] = {}

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
            @IntentRegistry.register("consul")
            class ModelConsulRegistrationIntent(ModelRegistryIntent):
                kind: Literal["consul"] = "consul"
                service_name: str
                ...
        """

        def decorator(intent_cls: type) -> type:
            if kind in cls._types:
                raise ValueError(
                    f"Intent kind '{kind}' already registered to {cls._types[kind].__name__}. "
                    f"Cannot register {intent_cls.__name__}."
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
            intent_cls = IntentRegistry.get_type("consul")
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
            all_types = IntentRegistry.get_all_types()
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
            if IntentRegistry.is_registered("consul"):
                intent_cls = IntentRegistry.get_type("consul")
        """
        return kind in cls._types

    @classmethod
    def clear(cls) -> None:
        """Clear all registered types.

        Warning:
            This method is intended for testing purposes only.
            Do not use in production code as it breaks the immutability
            guarantee after startup.
        """
        cls._types.clear()


class ModelRegistryIntent(BaseModel):
    """Base model for all registry-managed registration intents.

    All concrete intent models MUST:
    1. Inherit from this base class
    2. Use @IntentRegistry.register("kind") decorator
    3. Define kind as Literal["kind"] with matching default
    4. Be frozen (immutable) for thread safety

    This base class defines the common interface that all intents share,
    enabling type-safe access to common fields without type narrowing.

    Example:
        @IntentRegistry.register("consul")
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
        - IntentRegistry: Registration decorator and type lookup
        - ProtocolRegistrationIntent: Protocol for duck-typed signatures
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str
    """Intent kind identifier used for type discrimination."""

    operation: str
    """The operation to perform (e.g., 'register', 'upsert', 'deregister')."""

    node_id: UUID
    """The node ID this intent applies to."""

    correlation_id: UUID
    """Correlation ID for distributed tracing."""


__all__ = [
    "IntentRegistry",
    "ModelRegistryIntent",
]
