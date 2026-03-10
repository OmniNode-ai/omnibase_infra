# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from omnibase_infra.registry.models.model_artifact_registry_entry import (
    ModelArtifactRegistryEntry,
)


class ModelArtifactRegistry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    version: str
    description: str = ""
    artifacts: list[ModelArtifactRegistryEntry]
