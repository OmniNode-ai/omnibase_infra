# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Database repository contract model."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.runtime.db.models.model_db_operation import ModelDbOperation


class ModelDbRepositoryContract(BaseModel):
    """Complete database repository contract.

    Defines all operations, tables, and configuration for a database repository.

    Attributes:
        name: Repository name (used for identification)
        engine: Database engine ("postgres", "mysql", etc.)
        database_ref: Reference to database connection configuration
        tables: List of tables this repository operates on
        models: Mapping of model names to their module paths
        ops: Mapping of operation names to their definitions
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., description="Repository name")
    engine: str = Field(default="postgres", description="Database engine")
    database_ref: str = Field(..., description="Database connection reference")
    tables: list[str] = Field(default_factory=list, description="Tables operated on")
    models: dict[str, str] = Field(
        default_factory=dict, description="Model name -> module path mapping"
    )
    ops: dict[str, ModelDbOperation] = Field(
        default_factory=dict, description="Operation name -> definition mapping"
    )
