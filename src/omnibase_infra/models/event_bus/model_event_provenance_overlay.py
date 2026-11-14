# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The provenance a deployment's overlay declares, as raw text (OMN-23981)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field


class ModelEventProvenanceOverlay(BaseModel):
    """Where the deployment says this process runs from; every field may be absent.

    Raw text on purpose: the stamp handler validates it into the core
    provenance model, so a refused value is a reason and never an exception.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The process variable the deployment's private overlay sets for each field.
    VARIABLES: ClassVar[tuple[tuple[str, str], ...]] = (
        ("lane", "ONEX_PROVENANCE_LANE"),
        ("lane_kind", "ONEX_PROVENANCE_LANE_KIND"),
        ("host", "ONEX_PROVENANCE_HOST"),
        ("install_source", "ONEX_PROVENANCE_INSTALL_SOURCE"),
        ("revision", "ONEX_PROVENANCE_REVISION"),
    )

    lane: str | None = Field(default=None, description="dev, stability or other.")
    lane_kind: str | None = Field(
        default=None, description="Kind of emitter; required only for lane other."
    )
    host: str | None = Field(default=None, description="Host the process runs on.")
    install_source: str | None = Field(
        default=None, description="tag, branch, worktree or hand."
    )
    revision: str | None = Field(
        default=None, description="Source revision the process was built from."
    )

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> Self:
        """Read the overlay out of a process environment; absent variables stay None."""
        return cls(**{field: environ.get(name) for field, name in cls.VARIABLES})


__all__: list[str] = ["ModelEventProvenanceOverlay"]
