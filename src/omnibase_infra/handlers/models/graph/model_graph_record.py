# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Graph record model for Cypher query results."""

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.types import JsonDict


class ModelGraphRecord(BaseModel):
    """Single record from a Cypher query result.

    Attributes:
        data: Record data as key-value pairs
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    data: JsonDict = Field(description="Record data as key-value pairs")


__all__: list[str] = ["ModelGraphRecord"]
