# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Fail-closed input contract for an image-only dev effects replacement."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
ImageId = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
SourceSha = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
SourceRepo = Literal["omnibase_core", "omnibase_compat", "omnimarket"]


class ModelEffectsDeployPlan(BaseModel):
    """An immutable candidate, exact current state, and an image-only boundary.

    Source pins cover the three workspace siblings, not every distribution.
    The executor additionally compares installed dependency and shared runtime
    content against the prior image. Unknown or incompatible inputs refuse.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["1"]
    ticket_id: Annotated[str, Field(pattern=r"^OMN-[1-9][0-9]*$")]
    reason: Annotated[str, Field(min_length=10, max_length=1000)]
    compose_project: Literal["omnibase-infra"]
    service: Literal["runtime-effects"]
    expected_container_id: Sha256
    expected_image_id: ImageId
    candidate_image_id: ImageId
    compose_files: Annotated[tuple[Path, ...], Field(min_length=1)]
    compose_working_dir: Path
    expected_compose_sha256: Sha256
    source_pins: dict[SourceRepo, SourceSha]
    infra_source_sha: SourceSha
    source_clones_root: Path
    hotpatch_ledger: Path
    # Match the existing effects readiness allowance; a short fixed deadline
    # would roll back a normal subscription bootstrap before it could finish.
    health_timeout_seconds: Annotated[int, Field(ge=1, le=1800)] = 1800

    @field_validator("reason")
    @classmethod
    def meaningful_reason(cls, value: str) -> str:
        if len(value.strip()) < 10:
            raise ValueError("reason must contain a concrete rollout justification")
        return value

    @model_validator(mode="after")
    def validate_boundary(self) -> Self:
        if self.expected_image_id == self.candidate_image_id:
            raise ValueError("candidate must differ from the current immutable image")
        if set(self.source_pins) != {"omnibase_core", "omnibase_compat", "omnimarket"}:
            raise ValueError("source_pins must name all three workspace siblings")
        paths = (
            *self.compose_files,
            self.compose_working_dir,
            self.source_clones_root,
            self.hotpatch_ledger,
        )
        if any(not path.is_absolute() or ".." in path.parts for path in paths):
            raise ValueError("compose paths must be absolute and traversal-free")
        if len(set(self.compose_files)) != len(self.compose_files):
            raise ValueError("compose file chain must not contain duplicates")
        return self

    @classmethod
    def load(cls, path: Path) -> Self:
        """Validate the serialized plan before any deployment operation."""
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
