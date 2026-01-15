# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Base Vault payload model for registry-based type resolution.

This module provides:
- ModelPayloadVault: Base model for all Vault handler payloads

Design Pattern:
    Instead of maintaining explicit union types like:
        VaultPayload = ModelVaultSecretPayload | ModelVaultWritePayload | ...

    Payload models self-register via decorator:
        @RegistryPayloadVault.register("read_secret")
        class ModelVaultSecretPayload(ModelPayloadVault): ...

    The registry resolves types dynamically during Pydantic validation,
    enabling new payload types to be added without modifying existing code.

Related:
    - RegistryPayloadVault: Decorator-based registry for payload type discovery
    - EnumVaultOperationType: Operation type enum for discriminator
    - OMN-1007: Union reduction refactoring
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


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

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    operation_type: str
    """Operation type identifier used for type discrimination."""


__all__ = [
    "ModelPayloadVault",
]
