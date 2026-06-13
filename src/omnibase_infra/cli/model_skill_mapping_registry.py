# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Declarative ``onex skill`` mapping registry (OMN-13097).

The full skill→node registry loaded from ``skill_mapping.yaml``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.cli.model_skill_mapping import ModelSkillMapping

__all__ = ["ModelSkillMappingRegistry"]


class ModelSkillMappingRegistry(BaseModel):
    """The full declarative skill→node registry loaded from YAML."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    skills: tuple[ModelSkillMapping, ...] = Field(
        ...,
        description="All declared skill mappings.",
    )

    @model_validator(mode="after")
    def _validate_unique_names(self) -> ModelSkillMappingRegistry:
        names = [s.skill_name for s in self.skills]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(
                f"duplicate skill_name(s) in mapping registry: {', '.join(duplicates)}"
            )
        return self

    def get(self, skill_name: str) -> ModelSkillMapping | None:
        """Return the mapping for ``skill_name`` or None when absent."""
        for mapping in self.skills:
            if mapping.skill_name == skill_name:
                return mapping
        return None
