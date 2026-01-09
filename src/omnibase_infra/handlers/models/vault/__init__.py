# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Vault Handler Models Module.

This module exports Pydantic models for Vault handler request/response structures.
All models are strongly typed to eliminate Any usage.

Exports:
    EnumVaultOperationType: Discriminator enum for Vault operation types
    ModelPayloadVault: Base model for registry-managed Vault payloads
    RegistryPayloadVault: Decorator-based registry for payload type discovery
    ModelVaultSecretPayload: Payload for vault.read_secret result
    ModelVaultWritePayload: Payload for vault.write_secret result
    ModelVaultDeletePayload: Payload for vault.delete_secret result
    ModelVaultListPayload: Payload for vault.list_secrets result
    ModelVaultRenewTokenPayload: Payload for vault.renew_token result
    VaultPayload: Discriminated union of all Vault payload types
    ModelVaultHandlerPayload: Wrapper containing discriminated union payload
"""

from omnibase_infra.handlers.models.vault.enum_vault_operation_type import (
    EnumVaultOperationType,
)
from omnibase_infra.handlers.models.vault.model_payload_vault import (
    ModelPayloadVault,
)
from omnibase_infra.handlers.models.vault.model_vault_delete_payload import (
    ModelVaultDeletePayload,
)
from omnibase_infra.handlers.models.vault.model_vault_handler_config import (
    ModelVaultHandlerConfig,
)
from omnibase_infra.handlers.models.vault.model_vault_handler_payload import (
    ModelVaultHandlerPayload,
    VaultPayload,
)
from omnibase_infra.handlers.models.vault.model_vault_list_payload import (
    ModelVaultListPayload,
)
from omnibase_infra.handlers.models.vault.model_vault_renew_token_payload import (
    ModelVaultRenewTokenPayload,
)
from omnibase_infra.handlers.models.vault.model_vault_retry_config import (
    ModelVaultRetryConfig,
)
from omnibase_infra.handlers.models.vault.model_vault_secret_payload import (
    ModelVaultSecretPayload,
)
from omnibase_infra.handlers.models.vault.model_vault_write_payload import (
    ModelVaultWritePayload,
)
from omnibase_infra.handlers.models.vault.registry_payload_vault import (
    RegistryPayloadVault,
)

__all__: list[str] = [
    "EnumVaultOperationType",
    "ModelPayloadVault",
    "ModelVaultDeletePayload",
    "ModelVaultHandlerConfig",
    "ModelVaultHandlerPayload",
    "ModelVaultListPayload",
    "ModelVaultRenewTokenPayload",
    "ModelVaultRetryConfig",
    "ModelVaultSecretPayload",
    "ModelVaultWritePayload",
    "RegistryPayloadVault",
    "VaultPayload",
]
