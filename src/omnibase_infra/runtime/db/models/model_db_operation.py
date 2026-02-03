# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Database operation model for SQL operations."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.runtime.db.models.model_db_param import ModelDbParam
from omnibase_infra.runtime.db.models.model_db_return import ModelDbReturn
from omnibase_infra.runtime.db.models.model_db_safety_policy import ModelDbSafetyPolicy


class ModelDbOperation(BaseModel):
    """Individual database operation definition.

    Attributes:
        mode: Operation mode ("read", "write", "delete")
        sql: SQL query template with positional parameters ($1, $2, ...)
        params: Parameter definitions (name -> ModelDbParam)
        returns: Return type specification
        safety_policy: Safety constraints for this operation
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: str = Field(..., description="Operation mode (read/write/delete)")
    sql: str = Field(..., description="SQL query template")
    params: dict[str, ModelDbParam] = Field(
        default_factory=dict, description="Parameter definitions"
    )
    returns: ModelDbReturn = Field(
        default_factory=ModelDbReturn, description="Return type specification"
    )
    safety_policy: ModelDbSafetyPolicy = Field(
        default_factory=ModelDbSafetyPolicy, description="Safety constraints"
    )
