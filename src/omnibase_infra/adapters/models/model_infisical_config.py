# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Infisical adapter configuration model.

.. versionadded:: 0.9.0
    Initial implementation for OMN-2286.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class ModelInfisicalAdapterConfig(BaseModel):
    """Configuration for the Infisical adapter.

    Attributes:
        host: Infisical server URL.
        client_id: Machine identity client ID for Universal Auth.
        client_secret: Machine identity client secret for Universal Auth.
        project_id: Default Infisical project ID.
        environment_slug: Default environment slug (e.g., ``dev``, ``staging``, ``prod``).
        secret_path: Default secret path prefix (e.g., ``/``).
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    host: str = Field(
        default="https://app.infisical.com",
        description="Infisical server URL.",
    )
    client_id: SecretStr = Field(
        ...,
        description="Machine identity client ID for Universal Auth.",
    )
    client_secret: SecretStr = Field(
        ...,
        description="Machine identity client secret for Universal Auth.",
    )
    project_id: UUID = Field(
        ...,
        description="Default Infisical project ID.",
    )
    environment_slug: str = Field(
        default="prod",
        min_length=1,
        description="Default environment slug.",
    )
    secret_path: str = Field(
        default="/",
        description="Default secret path prefix.",
    )
