from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelSourceTrigger(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    pattern: str
    change_scope: Literal["structural", "semantic", "any"] = "any"
    match_fields: list[str] = Field(default_factory=list)


class ModelArtifactRegistryEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    artifact_id: str
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


class ModelArtifactRegistry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    version: str
    description: str = ""
    artifacts: list[ModelArtifactRegistryEntry]
