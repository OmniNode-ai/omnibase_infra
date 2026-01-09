# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Vault List Payload Model.

This module provides the Pydantic model for vault.list_secrets operation results.
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


@RegistryPayloadVault.register("list_secrets")
class ModelVaultListPayload(ModelPayloadVault):
    """Payload for vault.list_secrets operation result.

    Contains the list of secret keys at the specified path.

    Attributes:
        operation_type: Discriminator set to "list_secrets"
        keys: List of secret key names at the path

    Example:
        >>> payload = ModelVaultListPayload(keys=["db/", "api/", "config"])
        >>> print(payload.keys)
        ['db/', 'api/', 'config']
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    operation_type: Literal[EnumVaultOperationType.LIST_SECRETS] = Field(
        default=EnumVaultOperationType.LIST_SECRETS,
        description="Operation type discriminator",
    )
    keys: list[str] = Field(
        default_factory=list,
        description="List of secret key names at the path",
    )


__all__: list[str] = ["ModelVaultListPayload"]
