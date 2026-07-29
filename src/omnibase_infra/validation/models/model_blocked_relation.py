# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Explicit unresolved relation from a migration ownership manifest."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.validation.enums.enum_application_relation_kind import (
    EnumApplicationRelationKind,
)


class ModelBlockedRelation(BaseModel):
    """Unresolved live-only relation retained as an explicit hard blocker."""

    model_config = ConfigDict(frozen=True, extra="allow")

    name: str = Field(..., min_length=1)
    kind: EnumApplicationRelationKind
    reason: str = Field(..., min_length=1)

    @field_validator("kind", mode="before")
    @classmethod
    def normalize_kind(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        return value.strip().lower().replace("-", "_").replace(" ", "_")


__all__ = ["ModelBlockedRelation"]
