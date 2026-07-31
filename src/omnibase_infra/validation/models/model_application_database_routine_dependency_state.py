# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed catalog state for one routine reachable from an authority surface."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ModelApplicationDatabaseRoutineDependencyState(BaseModel):
    """Catalog-resolved routine metadata used for transitive authority analysis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    object_id: int = Field(..., gt=0)
    namespace: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    language: str = Field(..., min_length=1)
    source_body: str | None
    argument_type_ids: tuple[int, ...] = ()
    argument_names: tuple[str | None, ...] = ()
    returns_trigger: bool = False
    referenced_routine_ids: tuple[int, ...] = ()
    referenced_target_columns: tuple[str, ...] = ()
    references_target_whole_row: bool = False

    @field_validator("namespace", "name", "language")
    @classmethod
    def normalize_catalog_name(cls, value: str) -> str:
        """Normalize catalog identifiers without accepting empty values."""
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("catalog names must not be blank")
        return normalized

    @model_validator(mode="after")
    def validate_unique_dependencies(
        self,
    ) -> ModelApplicationDatabaseRoutineDependencyState:
        """Reject ambiguous duplicate dependency evidence."""
        if self.argument_names and len(self.argument_names) != len(
            self.argument_type_ids
        ):
            raise ValueError(
                "argument names must be empty or align exactly with argument type IDs"
            )
        if any(name is not None and not name.strip() for name in self.argument_names):
            raise ValueError("argument names must not contain blank values")
        if len(set(self.referenced_routine_ids)) != len(self.referenced_routine_ids):
            raise ValueError("referenced routine IDs must be unique")
        if len(set(self.referenced_target_columns)) != len(
            self.referenced_target_columns
        ):
            raise ValueError("referenced target columns must be unique")
        return self


__all__ = ["ModelApplicationDatabaseRoutineDependencyState"]
