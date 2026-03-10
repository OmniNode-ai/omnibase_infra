# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.registry.models.model_source_trigger import ModelSourceTrigger


class ModelArtifactRegistryEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    artifact_id: UUID
    artifact_type: Literal[
        "doc",
        "design_spec",
        "runbook",
        "roadmap",
        "reference",
        "migration_note",
        "release_note",
    ]
    title: str
    path: str
    repo: str
    owner_hint: str | None = None
    update_policy: Literal["none", "warn", "require", "strict"] = "warn"
    source_triggers: list[ModelSourceTrigger] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    last_verified: datetime | None = None
