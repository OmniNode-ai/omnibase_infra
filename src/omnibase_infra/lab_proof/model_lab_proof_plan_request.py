# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Everything the plan handler reads to render one proof run.

Ticket: OMN-19572
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.lab_proof.model_lab_proof_bundle_policy import (
    ModelLabProofBundlePolicy,
)
from omnibase_infra.lab_proof.model_lab_proof_profile_variant import (
    ModelLabProofProfileVariant,
)
from omnibase_infra.lab_proof.model_lab_proof_subject import (
    SHA_PATTERN,
    ModelLabProofSubject,
)


class ModelLabProofPlanRequest(BaseModel):
    """The profile row, the subject, the host facts and the bundle policy.

    ``lane_root`` is the prover's own directory on the host
    (``~/prove-<lane>``, interim recipes common frame 4); the run's working
    directory is always ``<lane_root>/<run_key>`` and its logs
    ``<lane_root>/logs/<run_key>``, so teardown removes only what this run made.
    ``harness_root`` is the omnibase_infra clone running the proof; the derived
    image's Dockerfile is read from it, so the runtime image itself is built
    from ``infra_sha`` alone and carries nothing of the harness.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    profile_key: str = Field(min_length=1)
    profile_version: int = Field(ge=1)
    profile_repo: str = Field(min_length=1)
    variant: ModelLabProofProfileVariant
    subject: ModelLabProofSubject
    run_key: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{5,62}$")
    host: str = Field(min_length=1)
    lane_root: str = Field(pattern=r"^/[A-Za-z0-9_./-]*/prove-[a-z0-9-]+$")
    harness_root: str = Field(pattern=r"^/[A-Za-z0-9_./-]+$")
    infra_sha: str = Field(pattern=SHA_PATTERN)
    model_endpoint_url: str = Field(
        pattern=r"^https?://[A-Za-z0-9.-]+:[0-9]+/v1/chat/completions$"
    )
    positive_control_project: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]+$")
    negative_control: bool = False
    bundle: ModelLabProofBundlePolicy

    @model_validator(mode="after")
    def _harness_in_lane(self) -> ModelLabProofPlanRequest:
        if not self.harness_root.startswith(self.lane_root + "/"):
            raise ValueError("harness_root must be inside lane_root")
        if (
            self.harness_root == f"{self.lane_root}/{self.run_key}"
            or self.harness_root.startswith(f"{self.lane_root}/{self.run_key}/")
        ):
            raise ValueError("harness_root must not be inside the run directory")
        return self


__all__ = ["ModelLabProofPlanRequest"]
