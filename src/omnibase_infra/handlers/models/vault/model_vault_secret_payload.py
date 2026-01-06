# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Vault Secret Payload Model.

This module provides the Pydantic model for vault.read_secret operation results.
"""

from __future__ import annotations

from typing import Literal

from omnibase_core.types import JsonType
from pydantic import ConfigDict, Field

from omnibase_infra.handlers.models.vault.enum_vault_operation_type import (
    EnumVaultOperationType,
)
from omnibase_infra.handlers.models.vault.model_payload_vault import (
    ModelPayloadVault,
    RegistryPayloadVault,
)


@RegistryPayloadVault.register("read_secret")
class ModelVaultSecretPayload(ModelPayloadVault):
    """Payload for vault.read_secret operation result.

    Contains the secret data retrieved from Vault KV v2 secrets engine
    along with metadata about the secret version.

    Attributes:
        operation_type: Discriminator set to "read_secret"
        data: The secret data as a key-value dictionary
        metadata: Vault metadata about the secret (version, created_time, etc.)

    Example:
        >>> payload = ModelVaultSecretPayload(
        ...     data={"username": "admin", "password": "secret"},
        ...     metadata={"version": 1, "created_time": "2025-01-01T00:00:00Z"},
        ... )
        >>> print(payload.data["username"])
        'admin'
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    operation_type: Literal[EnumVaultOperationType.READ_SECRET] = Field(
        default=EnumVaultOperationType.READ_SECRET,
        description="Operation type discriminator",
    )
    data: dict[str, JsonType] = Field(
        description="Secret data as key-value dictionary",
    )
    metadata: dict[str, JsonType] = Field(
        default_factory=dict,
        description="Vault metadata about the secret",
    )


__all__: list[str] = ["ModelVaultSecretPayload"]
