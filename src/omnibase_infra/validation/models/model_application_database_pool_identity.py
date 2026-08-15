# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Read-only identity evidence from one application PostgreSQL pool."""

from pydantic import BaseModel, ConfigDict, Field


class ModelApplicationDatabasePoolIdentity(BaseModel):
    """Exact current database and user observed through one real pool."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pool: str = Field(..., pattern=r"^[a-z_][a-z0-9_]*$")
    current_database: str = Field(..., pattern=r"^[a-z_][a-z0-9_]*$")
    current_user: str = Field(..., pattern=r"^[a-z_][a-z0-9_]*$")


__all__ = ["ModelApplicationDatabasePoolIdentity"]
