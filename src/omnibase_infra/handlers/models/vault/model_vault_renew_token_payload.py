# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Vault Renew Token Payload Model.

This module provides the Pydantic model for vault.renew_token operation results.
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


@RegistryPayloadVault.register("renew_token")
class ModelVaultRenewTokenPayload(ModelPayloadVault):
    """Payload for vault.renew_token operation result.

    Contains token renewal information including new TTL.

    Attributes:
        operation_type: Discriminator set to "renew_token"
        renewable: Whether the token can be renewed again
        lease_duration: New token TTL in seconds

    Example:
        >>> payload = ModelVaultRenewTokenPayload(
        ...     renewable=True,
        ...     lease_duration=3600,
        ... )
        >>> print(payload.lease_duration)
        3600
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    operation_type: Literal[EnumVaultOperationType.RENEW_TOKEN] = Field(
        default=EnumVaultOperationType.RENEW_TOKEN,
        description="Operation type discriminator",
    )
    renewable: bool = Field(
        default=False,
        description="Whether the token can be renewed again",
    )
    lease_duration: int = Field(
        default=0,
        ge=0,
        description="New token TTL in seconds",
    )


__all__: list[str] = ["ModelVaultRenewTokenPayload"]
