# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Base HTTP payload model for registry-based type resolution.

This module provides:
- ModelPayloadHttp: Base model for all HTTP handler payloads

Design Pattern:
    Instead of maintaining explicit union types like:
        HttpPayload = ModelHttpGetPayload | ModelHttpPostPayload | ...

    Payload models self-register via decorator:
        @RegistryPayloadHttp.register("get")
        class ModelHttpGetPayload(ModelPayloadHttp): ...

    The registry resolves types dynamically during Pydantic validation,
    enabling new payload types to be added without modifying existing code.

Related:
    - RegistryPayloadHttp: Decorator-based registry for payload type discovery
    - EnumHttpOperationType: Operation type enum for discriminator
    - OMN-1007: Union reduction refactoring
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelPayloadHttp(BaseModel):
    """Base model for all registry-managed HTTP payloads.

    All concrete HTTP payload models MUST:
    1. Inherit from this base class
    2. Use @RegistryPayloadHttp.register("operation_type") decorator
    3. Define operation_type as Literal["type"] with matching default
    4. Be frozen (immutable) for thread safety

    This base class defines the common interface that all HTTP payloads share,
    enabling type-safe access to common fields without type narrowing.

    Example:
        @RegistryPayloadHttp.register("get")
        class ModelHttpGetPayload(ModelPayloadHttp):
            operation_type: Literal["get"] = "get"
            url: str
            status_code: int
            ...

    Attributes:
        operation_type: Operation type identifier used for type discrimination.

    Related:
        - RegistryPayloadHttp: Registration decorator and type lookup
        - EnumHttpOperationType: Enum for operation type values
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    operation_type: str
    """Operation type identifier used for type discrimination."""


__all__ = [
    "ModelPayloadHttp",
]
