# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Vault Operation Type Enum.

Defines the discriminator enum for Vault operation types, used in the
discriminated union for Vault handler responses.
"""

from __future__ import annotations

from enum import Enum


class EnumVaultOperationType(str, Enum):
    """Vault operation type discriminator.

    Each value corresponds to a specific Vault operation type and its
    associated payload model in the VaultPayload discriminated union.

    Attributes:
        READ_SECRET: Read secret operation (ModelVaultSecretPayload)
        WRITE_SECRET: Write secret operation (ModelVaultWritePayload)
        DELETE_SECRET: Delete secret operation (ModelVaultDeletePayload)
        LIST_SECRETS: List secrets operation (ModelVaultListPayload)
        RENEW_TOKEN: Renew token operation (ModelVaultRenewTokenPayload)
    """

    READ_SECRET = "read_secret"
    WRITE_SECRET = "write_secret"
    DELETE_SECRET = "delete_secret"
    LIST_SECRETS = "list_secrets"
    RENEW_TOKEN = "renew_token"


__all__: list[str] = ["EnumVaultOperationType"]
