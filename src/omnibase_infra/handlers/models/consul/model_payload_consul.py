# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry-based Consul payload model for dynamic type resolution.

This module provides:
- ModelPayloadConsul: Base model for all Consul handler payloads
- RegistryPayloadConsul: Decorator-based registry for payload type discovery

Design Pattern:
    Instead of maintaining explicit union types like:
        ConsulPayload = ModelConsulKVGetFoundPayload | ModelConsulRegisterPayload | ...

    Payload models self-register via decorator:
        @RegistryPayloadConsul.register("kv_get_found")
        class ModelConsulKVGetFoundPayload(ModelPayloadConsul): ...

    The registry resolves types dynamically during Pydantic validation,
    enabling new payload types to be added without modifying existing code.

Related:
    - EnumConsulOperationType: Operation type enum for discriminator
    - OMN-1007: Union reduction refactoring
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

from pydantic import BaseModel, ConfigDict


class RegistryPayloadConsul:
    """Registry for Consul payload model types with decorator-based registration.

    This registry enables dynamic type resolution for Pydantic validation
    without requiring explicit union type definitions.

    Example:
        @RegistryPayloadConsul.register("kv_get_found")
        class ModelConsulKVGetFoundPayload(ModelPayloadConsul):
            operation_type: Literal["kv_get_found"] = "kv_get_found"
            ...

        # Later, resolve type from operation_type
        payload_cls = RegistryPayloadConsul.get_type("kv_get_found")
        payload = payload_cls.model_validate(data)

    Thread Safety:
        The registry is populated at module import time (class definition).
        After startup, it is read-only and thread-safe for concurrent access.
    """

    _types: ClassVar[dict[str, type[ModelPayloadConsul]]] = {}

    @classmethod
    def register(cls, operation_type: str) -> Callable[[type], type]:
        """Decorator to register a Consul payload model type.

        Args:
            operation_type: The operation type identifier (e.g., "kv_get_found",
                          "register"). Must match the model's `operation_type`
                          field value.

        Returns:
            Decorator function that registers the class and returns it unchanged.

        Raises:
            ValueError: If operation_type is already registered (prevents duplicates).

        Example:
            @RegistryPayloadConsul.register("kv_get_found")
            class ModelConsulKVGetFoundPayload(ModelPayloadConsul):
                operation_type: Literal["kv_get_found"] = "kv_get_found"
                key: str
                value: str | None
                ...
        """

        def decorator(payload_cls: type) -> type:
            if operation_type in cls._types:
                raise ValueError(
                    f"Consul payload operation_type '{operation_type}' already registered "
                    f"to {cls._types[operation_type].__name__}. "
                    f"Cannot register {payload_cls.__name__}."
                )
            cls._types[operation_type] = payload_cls
            return payload_cls

        return decorator

    @classmethod
    def get_type(cls, operation_type: str) -> type[ModelPayloadConsul]:
        """Get the payload model class for a given operation type.

        Args:
            operation_type: The operation type identifier.

        Returns:
            The registered payload model class.

        Raises:
            KeyError: If operation_type is not registered.

        Example:
            payload_cls = RegistryPayloadConsul.get_type("kv_get_found")
            payload = payload_cls.model_validate({"operation_type": "kv_get_found", ...})
        """
        if operation_type not in cls._types:
            registered = ", ".join(sorted(cls._types.keys())) or "(none)"
            raise KeyError(
                f"Unknown Consul operation_type '{operation_type}'. "
                f"Registered types: {registered}"
            )
        return cls._types[operation_type]

    @classmethod
    def get_all_types(cls) -> dict[str, type[ModelPayloadConsul]]:
        """Get all registered Consul payload types.

        Returns:
            Dict mapping operation_type strings to payload model classes.

        Example:
            all_types = RegistryPayloadConsul.get_all_types()
            for op_type, payload_cls in all_types.items():
                print(f"{op_type}: {payload_cls.__name__}")
        """
        return dict(cls._types)

    @classmethod
    def is_registered(cls, operation_type: str) -> bool:
        """Check if an operation type is registered.

        Args:
            operation_type: The operation type identifier.

        Returns:
            True if registered, False otherwise.

        Example:
            if RegistryPayloadConsul.is_registered("kv_get_found"):
                payload_cls = RegistryPayloadConsul.get_type("kv_get_found")
        """
        return operation_type in cls._types

    @classmethod
    def clear(cls) -> None:
        """Clear all registered types.

        Warning:
            This method is intended for testing purposes only.
            Do not use in production code as it breaks the immutability
            guarantee after startup.
        """
        cls._types.clear()


class ModelPayloadConsul(BaseModel):
    """Base model for all registry-managed Consul payloads.

    All concrete Consul payload models MUST:
    1. Inherit from this base class
    2. Use @RegistryPayloadConsul.register("operation_type") decorator
    3. Define operation_type as Literal["type"] with matching default
    4. Be frozen (immutable) for thread safety

    This base class defines the common interface that all payloads share,
    enabling type-safe access to common fields without type narrowing.

    Example:
        @RegistryPayloadConsul.register("kv_get_found")
        class ModelConsulKVGetFoundPayload(ModelPayloadConsul):
            operation_type: Literal["kv_get_found"] = "kv_get_found"
            key: str
            value: str | None
            ...

    Attributes:
        operation_type: Operation type identifier used for type discrimination.

    Related:
        - RegistryPayloadConsul: Registration decorator and type lookup
        - EnumConsulOperationType: Enum of valid operation types
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    operation_type: str
    """Operation type identifier used for type discrimination."""


__all__: list[str] = [
    "ModelPayloadConsul",
    "RegistryPayloadConsul",
]
