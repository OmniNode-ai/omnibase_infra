# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""How long each kind of step of a lab proof may run, declared in the plan node's contract.

Ticket: OMN-21623
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

MAX_STEP_SECONDS = 7200
"""Upper bound of any step (two hours): a contract typo cannot hang a lane for days."""


class ModelLabProofStepTimeouts(BaseModel):
    """The wall-clock bound of each kind of step the plan renders, in seconds.

    The handler holds no number: each step takes its bound from here. Every field
    is required, so a contract missing one fails at load naming it, never at the
    first proof that needed it (CLAUDE.md rule 8).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    default_seconds: int = Field(
        ge=1,
        le=MAX_STEP_SECONDS,
        description="A step of no kind named below (a git or file operation).",
    )
    git_fetch_seconds: int = Field(
        ge=1, le=MAX_STEP_SECONDS, description="A shallow fetch of a clone."
    )
    focused_tests_seconds: int = Field(
        ge=1, le=MAX_STEP_SECONDS, description="The PR's changed test files."
    )
    generate_seconds: int = Field(
        ge=1, le=MAX_STEP_SECONDS, description="Rendering the bundle's compose file."
    )
    build_base_seconds: int = Field(
        ge=1,
        le=MAX_STEP_SECONDS,
        description="Building the runtime image at the pinned infra commit.",
    )
    build_override_seconds: int = Field(
        ge=1,
        le=MAX_STEP_SECONDS,
        description="Installing the commit under test over the image's package.",
    )
    up_seconds: int = Field(
        ge=1, le=MAX_STEP_SECONDS, description="Booting the bundle."
    )
    health_probe_seconds: int = Field(
        ge=1,
        le=MAX_STEP_SECONDS,
        description="One Docker health inspection (the retry loop repeats it).",
    )
    identity_seconds: int = Field(
        ge=1,
        le=MAX_STEP_SECONDS,
        description="Hashing the installed package tree inside a container.",
    )
    import_smoke_seconds: int = Field(
        ge=1,
        le=MAX_STEP_SECONDS,
        description="Importing every consumer module inside a container.",
    )
    wiring_logs_seconds: int = Field(
        ge=1, le=MAX_STEP_SECONDS, description="Reading one container's startup log."
    )
    down_seconds: int = Field(
        ge=1,
        le=MAX_STEP_SECONDS,
        description="Stopping the bundle and deleting its volumes.",
    )
    remove_images_seconds: int = Field(
        ge=1, le=MAX_STEP_SECONDS, description="Deleting the run images."
    )
    remove_workdir_seconds: int = Field(
        ge=1, le=MAX_STEP_SECONDS, description="Deleting the run directory."
    )


__all__ = ["MAX_STEP_SECONDS", "ModelLabProofStepTimeouts"]
