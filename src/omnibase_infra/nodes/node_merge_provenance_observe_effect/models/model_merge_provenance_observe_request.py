# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What to observe: one commit of one repository (OMN-19927)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelMergeProvenanceObserveRequest(BaseModel):
    """Observe the merge-group runs recorded for ``sha``.

    ``workflow_path`` and ``summary_job`` name the one required context a
    queue landing is gated on in this repository: the ``CI`` workflow's
    ``CI Summary`` job. They are fields, not constants, so the node serves any
    repository whose queue gates on a different aggregate.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    repository: str = Field(
        pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$",
        description="owner/name of the repository.",
    )
    sha: str = Field(
        pattern=r"^[0-9a-f]{40}$", description="Full 40-character commit sha."
    )
    workflow_path: str = Field(
        default=".github/workflows/ci.yml",
        description="Workflow whose merge-group runs count as validation.",
    )
    summary_job: str = Field(
        default="CI Summary",
        description="The job whose success the merge queue gates on.",
    )


__all__: list[str] = ["ModelMergeProvenanceObserveRequest"]
