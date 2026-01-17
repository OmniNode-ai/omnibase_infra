# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Model for secret source specification.

.. versionadded:: 0.8.0
    Initial implementation for OMN-764.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Canonical type alias for secret source types.
# Reuse this type across all secret-related models for consistency.
SecretSourceType = Literal["env", "vault", "file"]


class ModelSecretSourceSpec(BaseModel):
    """Source specification for a single secret.

    Defines where and how to retrieve a secret value from a specific source.

    Attributes:
        source_type: The type of secret source (env, vault, or file).
        source_path: The path or key to the secret within the source.

    Examples:
        Environment variable::

            ModelSecretSourceSpec(source_type="env", source_path="POSTGRES_PASSWORD")

        Vault secret::

            ModelSecretSourceSpec(
                source_type="vault",
                source_path="secret/data/database/postgres#password"
            )

        File-based secret::

            ModelSecretSourceSpec(source_type="file", source_path="/run/secrets/db_pass")
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    source_type: SecretSourceType = Field(
        ...,
        description="Type of secret source: 'env' for environment variables, "
        "'vault' for HashiCorp Vault, 'file' for file-based secrets.",
    )
    source_path: str = Field(
        ...,
        min_length=1,
        description="Path or key to the secret. Format depends on source_type: "
        "env=VAR_NAME, vault=path/to/secret#key, file=/path/to/file.",
    )


__all__: list[str] = ["ModelSecretSourceSpec", "SecretSourceType"]
