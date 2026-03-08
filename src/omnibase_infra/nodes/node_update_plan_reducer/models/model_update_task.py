# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Domain event model for update tasks within a plan."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelUpdateTask(BaseModel):
    """A single task within an update plan.

    Each task targets a specific artifact and specifies what action
    is needed (human review, regeneration, stub creation, etc.).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    title: str
    target_artifact_id: str
    task_type: Literal[
        "patch_existing",
        "regenerate",
        "create_stub",
        "human_author",
        "request_waiver",
    ]
    blocking: bool = False
    depends_on: list[str] = Field(default_factory=list)
    owner_hint: str | None = None
    status: Literal["planned", "completed", "waived"] = "planned"
