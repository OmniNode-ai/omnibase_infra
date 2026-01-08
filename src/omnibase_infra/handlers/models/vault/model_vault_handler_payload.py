# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Vault Handler Payload Model.

This module provides the discriminated union type for Vault operation payloads
and the wrapper model that contains them.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag

from omnibase_infra.handlers.models.vault.model_vault_delete_payload import (
    ModelVaultDeletePayload,
)
from omnibase_infra.handlers.models.vault.model_vault_list_payload import (
    ModelVaultListPayload,
)
from omnibase_infra.handlers.models.vault.model_vault_renew_token_payload import (
    ModelVaultRenewTokenPayload,
)
from omnibase_infra.handlers.models.vault.model_vault_secret_payload import (
    ModelVaultSecretPayload,
)
from omnibase_infra.handlers.models.vault.model_vault_write_payload import (
    ModelVaultWritePayload,
)


def _vault_payload_discriminator(value: object) -> str:
    """Discriminator function for VaultPayload union.

    Determines which payload type to use based on the operation_type field.

    Args:
        value: The raw value being validated (dict or model instance)

    Returns:
        The discriminator tag value matching the operation_type
    """
    if isinstance(value, dict):
        operation_type = value.get("operation_type", "read_secret")
    else:
        operation_type = getattr(value, "operation_type", "read_secret")

    # Handle both enum values and string values
    if hasattr(operation_type, "value"):
        return str(operation_type.value)
    return str(operation_type)


# Discriminated union of all Vault payload types
VaultPayload = Annotated[
    Annotated[ModelVaultSecretPayload, Tag("read_secret")]
    | Annotated[ModelVaultWritePayload, Tag("write_secret")]
    | Annotated[ModelVaultDeletePayload, Tag("delete_secret")]
    | Annotated[ModelVaultListPayload, Tag("list_secrets")]
    | Annotated[ModelVaultRenewTokenPayload, Tag("renew_token")],
    Discriminator(_vault_payload_discriminator),
]


class ModelVaultHandlerPayload(BaseModel):
    """Wrapper model for Vault handler payloads.

    Contains the discriminated union of all possible Vault operation payloads.
    This allows the HandlerVault to return a consistent payload wrapper while
    the actual payload type varies based on the operation.

    Attributes:
        data: The operation-specific payload (discriminated union)

    Example:
        >>> from omnibase_infra.handlers.models.vault import ModelVaultSecretPayload
        >>> secret_payload = ModelVaultSecretPayload(
        ...     data={"username": "admin"},
        ...     metadata={"version": 1},
        ... )
        >>> wrapper = ModelVaultHandlerPayload(data=secret_payload)
        >>> print(wrapper.data.operation_type)
        <EnumVaultOperationType.READ_SECRET: 'read_secret'>
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    data: VaultPayload = Field(
        description="Operation-specific payload",
    )


__all__: list[str] = [
    "ModelVaultHandlerPayload",
    "VaultPayload",
]
