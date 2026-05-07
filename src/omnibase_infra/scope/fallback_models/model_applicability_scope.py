# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Compatibility subset of the core applicability scope model."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.scope.fallback_models.model_scope_predicate import (
    ModelScopePredicate,
)


class ModelApplicabilityScope(BaseModel):
    """Compatibility subset of the core applicability scope model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    applies_when: ModelScopePredicate = Field(default_factory=ModelScopePredicate)
    disabled_when: ModelScopePredicate = Field(default_factory=ModelScopePredicate)


__all__ = ["ModelApplicabilityScope"]
