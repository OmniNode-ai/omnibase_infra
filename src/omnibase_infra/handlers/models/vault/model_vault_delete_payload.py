# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Vault Delete Payload Model.

This module provides the Pydantic model for vault.delete_secret operation results.
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


@RegistryPayloadVault.register("delete_secret")
class ModelVaultDeletePayload(ModelPayloadVault):
    """Payload for vault.delete_secret operation result.

    Contains confirmation of secret deletion.

    Attributes:
        operation_type: Discriminator set to "delete_secret"
        deleted: Whether the secret was successfully deleted

    Example:
        >>> payload = ModelVaultDeletePayload(deleted=True)
        >>> print(payload.deleted)
        True
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    operation_type: Literal[EnumVaultOperationType.DELETE_SECRET] = Field(
        default=EnumVaultOperationType.DELETE_SECRET,
        description="Operation type discriminator",
    )
    deleted: bool = Field(
        description="Whether the secret was successfully deleted",
    )


__all__: list[str] = ["ModelVaultDeletePayload"]
