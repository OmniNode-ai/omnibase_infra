# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Catalog-observed column state for application-domain enforcement."""

from pydantic import BaseModel, ConfigDict, Field


class ModelApplicationDatabaseColumnState(BaseModel):
    """One exact column shape after a migration is applied."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., pattern=r"^[a-z_][a-z0-9_]*$")
    data_type: str = Field(..., min_length=1)
    nullable: bool
    default_expression: str | None = None
    generated_expression: str | None = None


__all__ = ["ModelApplicationDatabaseColumnState"]
