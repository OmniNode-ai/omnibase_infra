# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry-based Vault payload model for dynamic type resolution.

This module provides:
- ModelPayloadVault: Base model for all Vault handler payloads
- RegistryPayloadVault: Decorator-based registry for payload type discovery

Design Pattern:
    Instead of maintaining explicit union types like:
        VaultPayload = ModelVaultSecretPayload | ModelVaultWritePayload | ...

    Payload models self-register via decorator:
        @RegistryPayloadVault.register("read_secret")
        class ModelVaultSecretPayload(ModelPayloadVault): ...

    The registry resolves types dynamically during Pydantic validation,
    enabling new payload types to be added without modifying existing code.

Related:
    - EnumVaultOperationType: Operation type enum for discriminator
    - OMN-1007: Union reduction refactoring
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

from pydantic import BaseModel, ConfigDict


class RegistryPayloadVault:
    """Registry for Vault payload model types with decorator-based registration.

    This registry enables dynamic type resolution for Pydantic validation
    without requiring explicit union type definitions.

    Example:
        @RegistryPayloadVault.register("read_secret")
        class ModelVaultSecretPayload(ModelPayloadVault):
            operation_type: Literal[EnumVaultOperationType.READ_SECRET]
            ...

        # Later, resolve type from operation_type
        payload_cls = RegistryPayloadVault.get_type("read_secret")
        payload = payload_cls.model_validate(data)

    Thread Safety:
        The registry is populated at module import time (class definition).
        After startup, it is read-only and thread-safe for concurrent access.
    """

    _types: ClassVar[dict[str, type[ModelPayloadVault]]] = {}

    @classmethod
    def register(cls, operation_type: str) -> Callable[[type], type]:
        """Decorator to register a Vault payload model type.

        Args:
            operation_type: The Vault operation type identifier (e.g., "read_secret").
                           Must match the model's `operation_type` field value.

        Returns:
            Decorator function that registers the class and returns it unchanged.

        Raises:
            ValueError: If operation_type is already registered (prevents duplicates).

        Example:
            @RegistryPayloadVault.register("read_secret")
            class ModelVaultSecretPayload(ModelPayloadVault):
                operation_type: Literal[EnumVaultOperationType.READ_SECRET]
                path: str
                ...
        """

        def decorator(payload_cls: type) -> type:
            if operation_type in cls._types:
                raise ValueError(
                    f"Vault payload operation_type '{operation_type}' already registered "
                    f"to {cls._types[operation_type].__name__}. "
                    f"Cannot register {payload_cls.__name__}."
                )
            cls._types[operation_type] = payload_cls
            return payload_cls

        return decorator

    @classmethod
    def get_type(cls, operation_type: str) -> type[ModelPayloadVault]:
        """Get the payload model class for a given operation type.

        Args:
            operation_type: The Vault operation type identifier.

        Returns:
            The registered payload model class.

        Raises:
            KeyError: If operation_type is not registered.

        Example:
            payload_cls = RegistryPayloadVault.get_type("read_secret")
            payload = payload_cls.model_validate({"operation_type": "read_secret", ...})
        """
        if operation_type not in cls._types:
            registered = ", ".join(sorted(cls._types.keys())) or "(none)"
            raise KeyError(
                f"Unknown Vault operation_type '{operation_type}'. "
                f"Registered types: {registered}"
            )
        return cls._types[operation_type]

    @classmethod
    def get_all_types(cls) -> dict[str, type[ModelPayloadVault]]:
        """Get all registered Vault payload types.

        Returns:
            Dict mapping operation_type strings to payload model classes.

        Example:
            all_types = RegistryPayloadVault.get_all_types()
            for op_type, payload_cls in all_types.items():
                print(f"{op_type}: {payload_cls.__name__}")
        """
        return dict(cls._types)

    @classmethod
    def is_registered(cls, operation_type: str) -> bool:
        """Check if an operation type is registered.

        Args:
            operation_type: The Vault operation type identifier.

        Returns:
            True if registered, False otherwise.

        Example:
            if RegistryPayloadVault.is_registered("read_secret"):
                payload_cls = RegistryPayloadVault.get_type("read_secret")
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


class ModelPayloadVault(BaseModel):
    """Base model for all registry-managed Vault payloads.

    All concrete Vault payload models MUST:
    1. Inherit from this base class
    2. Use @RegistryPayloadVault.register("operation_type") decorator
    3. Define operation_type field with appropriate Literal type
    4. Be frozen (immutable) for thread safety

    This base class defines the common interface that all payloads share,
    enabling type-safe access to common fields without type narrowing.

    Example:
        @RegistryPayloadVault.register("read_secret")
        class ModelVaultSecretPayload(ModelPayloadVault):
            operation_type: Literal[EnumVaultOperationType.READ_SECRET]
            path: str
            data: dict[str, str]
            ...

    Attributes:
        operation_type: Operation type identifier used for type discrimination.

    Related:
        - RegistryPayloadVault: Registration decorator and type lookup
        - EnumVaultOperationType: Enum defining valid operation types
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    operation_type: str
    """Operation type identifier used for type discrimination."""


__all__ = [
    "ModelPayloadVault",
    "RegistryPayloadVault",
]
