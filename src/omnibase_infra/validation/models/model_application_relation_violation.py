# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""One deterministic application-relation ownership failure."""

from pydantic import BaseModel, ConfigDict

from omnibase_infra.validation.enums.enum_application_relation_violation import (
    EnumApplicationRelationViolation,
)


class ModelApplicationRelationViolation(BaseModel):
    """One deterministic global ownership validation failure."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: EnumApplicationRelationViolation
    message: str
    relation_name: str | None = None
    source_paths: tuple[str, ...] = ()


__all__ = ["ModelApplicationRelationViolation"]
