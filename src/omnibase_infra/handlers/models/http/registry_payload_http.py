# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry for HTTP payload types with decorator-based registration.

This module provides:
- RegistryPayloadHttp: Decorator-based registry for payload type discovery

Design Pattern:
    Instead of maintaining explicit union types like:
        HttpPayload = ModelHttpGetPayload | ModelHttpPostPayload | ...

    Payload models self-register via decorator:
        @RegistryPayloadHttp.register("get")
        class ModelHttpGetPayload(ModelPayloadHttp): ...

    The registry resolves types dynamically during Pydantic validation,
    enabling new payload types to be added without modifying existing code.

Related:
    - ModelPayloadHttp: Base model for HTTP payloads
    - EnumHttpOperationType: Operation type enum for discriminator
    - OMN-1007: Union reduction refactoring
"""

from __future__ import annotations

import threading
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar, TypeVar

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError

if TYPE_CHECKING:
    from omnibase_infra.handlers.models.http.model_payload_http import (
        ModelPayloadHttp,
    )

# TypeVar bound to ModelPayloadHttp to preserve subclass type through decorator
_HttpPayloadT = TypeVar("_HttpPayloadT", bound="ModelPayloadHttp")


class RegistryPayloadHttp:
    """Registry for HTTP payload model types with decorator-based registration.

    This registry enables dynamic type resolution for Pydantic validation
    without requiring explicit union type definitions.

    Example:
        @RegistryPayloadHttp.register("get")
        class ModelHttpGetPayload(ModelPayloadHttp):
            operation_type: Literal["get"] = "get"
            ...

        # Later, resolve type from operation_type
        payload_cls = RegistryPayloadHttp.get_type("get")
        payload = payload_cls.model_validate(data)

    Thread Safety:
        All registration and mutation operations are protected by a class-level
        threading.Lock to ensure thread-safe access. Read operations (get_type,
        get_all_types, is_registered) are atomic dictionary operations and safe
        for concurrent access. The lock ensures that check-and-set operations
        in register() are atomic, preventing race conditions.
    """

    _types: ClassVar[dict[str, type[ModelPayloadHttp]]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def register(
        cls, operation_type: str
    ) -> Callable[[type[_HttpPayloadT]], type[_HttpPayloadT]]:
        """Decorator to register an HTTP payload model type.

        Args:
            operation_type: The operation type identifier (e.g., "get", "post").
                           Must match the model's `operation_type` field value.

        Returns:
            Decorator function that registers the class and returns it unchanged.

        Raises:
            ValueError: If operation_type is already registered (prevents duplicates).

        Example:
            @RegistryPayloadHttp.register("get")
            class ModelHttpGetPayload(ModelPayloadHttp):
                operation_type: Literal["get"] = "get"
                url: str
                ...
        """

        def decorator(
            payload_cls: type[_HttpPayloadT],
        ) -> type[_HttpPayloadT]:
            # Runtime type validation (outside lock - class inspection is read-only)
            from omnibase_infra.handlers.models.http.model_payload_http import (
                ModelPayloadHttp,
            )

            if not issubclass(payload_cls, ModelPayloadHttp):
                context = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.HTTP,
                    operation="register_payload_type",
                )
                raise ProtocolConfigurationError(
                    f"Registered class {payload_cls.__name__!r} must be a subclass "
                    f"of ModelPayloadHttp, got {payload_cls.__mro__}",
                    context=context,
                )

            # Thread-safe registration with atomic check-and-set
            with cls._lock:
                if operation_type in cls._types:
                    context = ModelInfraErrorContext(
                        transport_type=EnumInfraTransportType.HTTP,
                        operation="register_payload_type",
                    )
                    raise ProtocolConfigurationError(
                        f"HTTP payload operation_type '{operation_type}' already registered "
                        f"to {cls._types[operation_type].__name__}. "
                        f"Cannot register {payload_cls.__name__}.",
                        context=context,
                    )
                cls._types[operation_type] = payload_cls
            return payload_cls

        return decorator

    @classmethod
    def get_type(cls, operation_type: str) -> type[ModelPayloadHttp]:
        """Get the payload model class for a given operation type.

        Args:
            operation_type: The operation type identifier.

        Returns:
            The registered payload model class.

        Raises:
            KeyError: If operation_type is not registered.

        Example:
            payload_cls = RegistryPayloadHttp.get_type("get")
            payload = payload_cls.model_validate({"operation_type": "get", ...})
        """
        if operation_type not in cls._types:
            registered = ", ".join(sorted(cls._types.keys())) or "(none)"
            raise KeyError(
                f"Unknown HTTP operation_type '{operation_type}'. "
                f"Registered types: {registered}"
            )
        return cls._types[operation_type]

    @classmethod
    def get_all_types(cls) -> dict[str, type[ModelPayloadHttp]]:
        """Get all registered HTTP payload types.

        Returns:
            Dict mapping operation_type strings to payload model classes.

        Example:
            all_types = RegistryPayloadHttp.get_all_types()
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
            if RegistryPayloadHttp.is_registered("get"):
                payload_cls = RegistryPayloadHttp.get_type("get")
        """
        return operation_type in cls._types

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
            "RegistryPayloadHttp.clear() is intended for testing only. "
            "Do not use in production code.",
            UserWarning,
            stacklevel=2,
        )
        with cls._lock:
            cls._types.clear()


__all__ = [
    "RegistryPayloadHttp",  # HTTP payload registry
]
