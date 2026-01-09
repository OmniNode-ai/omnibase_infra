# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Vault Secret Payload Model.

This module provides the Pydantic model for vault.read_secret operation results.

JsonType Recursion Fix (OMN-1274):
    This module uses ``dict[str, object]`` for the ``data`` and ``metadata``
    fields instead of the recursive ``JsonType`` type alias. Here is why:

    **The Original Problem:**
    ``JsonType`` was defined as a recursive type alias::

        JsonType = dict[str, "JsonType"] | list["JsonType"] | str | int | float | bool | None

    Pydantic 2.x performs eager schema generation at class definition time.
    When encountering recursive type aliases, it attempts infinite expansion::

        JsonType -> dict[str, JsonType] | list[JsonType] | ...
                 -> dict[str, dict[str, JsonType] | ...] | ...
                 -> ... (RecursionError)

    **Why dict[str, object] is Correct Here:**
    Vault secrets are always structured as key-value dictionaries:

    - ``data``: Secret key-value pairs (e.g., {"username": "...", "password": "..."})
    - ``metadata``: Vault metadata (version, created_time, etc.)

    Neither field needs to support arrays or primitives at the root level,
    making ``dict[str, object]`` the semantically correct choice.

    Using ``dict[str, object]`` provides:
    - Correct semantics: Vault secrets are always dictionaries
    - Type safety: Pydantic validates the outer structure
    - No recursion: ``object`` avoids the recursive type expansion

    **Caveats:**
    - Values are typed as ``object`` (no static type checking on values)
    - Callers should validate expected secret structure at runtime
    - For full JSON support (any JSON value), use ``JsonType`` from
      ``omnibase_core.types`` (now fixed via TypeAlias pattern)

See Also:
    - ADR: ``docs/decisions/adr-any-type-pydantic-workaround.md`` (historical)
    - Pydantic issue: https://github.com/pydantic/pydantic/issues/3278
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
    data: dict[str, object] = Field(
        description="Secret data as key-value dictionary",
    )
    metadata: dict[str, object] = Field(
        default_factory=dict,
        description="Vault metadata about the secret",
    )


__all__: list[str] = ["ModelVaultSecretPayload"]
