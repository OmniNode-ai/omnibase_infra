# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Vault Write Payload Model.

This module provides the Pydantic model for vault.write_secret operation results.
"""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field

from omnibase_infra.handlers.models.vault.enum_vault_operation_type import (
    EnumVaultOperationType,
)
from omnibase_infra.handlers.models.vault.model_payload_vault import (
    ModelPayloadVault,
)
from omnibase_infra.handlers.models.vault.registry_payload_vault import (
    RegistryPayloadVault,
)


@RegistryPayloadVault.register("write_secret")
class ModelVaultWritePayload(ModelPayloadVault):
    """Payload for vault.write_secret operation result.

    Contains confirmation of secret write operation with version information.

    Attributes:
        operation_type: Discriminator set to "write_secret"
        version: The version number of the newly written secret
        created_time: Timestamp when the secret version was created

    Example:
        >>> payload = ModelVaultWritePayload(
        ...     version=2,
        ...     created_time="2025-01-01T00:00:00Z",
        ... )
        >>> print(payload.version)
        2
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    operation_type: Literal[EnumVaultOperationType.WRITE_SECRET] = Field(
        default=EnumVaultOperationType.WRITE_SECRET,
        description="Operation type discriminator",
    )
    version: int | None = Field(
        default=None,
        description="Version number of the written secret",
    )
    created_time: str | None = Field(
        default=None,
        description="Timestamp when the secret version was created",
    )


__all__: list[str] = ["ModelVaultWritePayload"]
