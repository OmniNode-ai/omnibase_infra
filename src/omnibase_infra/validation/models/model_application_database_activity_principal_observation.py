# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""One activity observation in an application-database evidence result."""

from pydantic import BaseModel, ConfigDict, Field


class ModelApplicationDatabaseActivityPrincipalObservation(BaseModel):
    """One exact ``(datname, usename)`` pair with its sample count."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    principal: str = Field(..., pattern=r"^[a-z_][a-z0-9_]*$")
    observation_count: int = Field(..., ge=1)


__all__ = ["ModelApplicationDatabaseActivityPrincipalObservation"]
