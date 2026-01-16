# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Base Consul payload model for registry-based type resolution.

This module provides:
- ModelPayloadConsul: Base model for all Consul handler payloads

Design Pattern:
    Instead of maintaining explicit union types like:
        ConsulPayload = ModelConsulKVGetFoundPayload | ModelConsulRegisterPayload | ...

    Payload models self-register via decorator:
        @RegistryPayloadConsul.register("kv_get_found")
        class ModelConsulKVGetFoundPayload(ModelPayloadConsul): ...

    The registry resolves types dynamically during Pydantic validation,
    enabling new payload types to be added without modifying existing code.

Related:
    - RegistryPayloadConsul: Decorator-based registry for payload type discovery
    - EnumConsulOperationType: Operation type enum for discriminator
    - OMN-1007: Union reduction refactoring
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


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

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    operation_type: str
    """Operation type identifier used for type discrimination."""


__all__: list[str] = [
    "ModelPayloadConsul",
]
