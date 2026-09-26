# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""What is being proved: one repository, one pull request, one commit.

Ticket: OMN-19572
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

SHA_PATTERN = r"^[0-9a-f]{40}$"


class ModelLabProofSubject(BaseModel):
    """The pull request and the exact commit under test.

    ``proved_sha`` is the head for a proof and the merge base for a base
    control; ``fetch_ref`` is what is fetched to reach it (the PR head ref,
    or the sha itself), and the fetched commit must equal ``proved_sha``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    repo: str = Field(pattern=r"^OmniNode-ai/[A-Za-z0-9_.-]+$")
    pr_number: int = Field(ge=1)
    head_sha: str = Field(pattern=SHA_PATTERN)
    base_sha: str = Field(pattern=SHA_PATTERN)
    proved_sha: str = Field(pattern=SHA_PATTERN)
    fetch_ref: str = Field(pattern=r"^(refs/pull/[0-9]+/head|[0-9a-f]{40})$")
    changed_files: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _proved_is_head_or_base(self) -> ModelLabProofSubject:
        if self.proved_sha not in (self.head_sha, self.base_sha):
            raise ValueError("proved_sha must be the head or the base")
        if self.fetch_ref.startswith("refs/pull/"):
            if self.fetch_ref != f"refs/pull/{self.pr_number}/head":
                raise ValueError("fetch_ref names a different pull request")
            if self.proved_sha != self.head_sha:
                raise ValueError("a PR head ref can only prove the head")
        elif self.fetch_ref != self.proved_sha:
            raise ValueError("a sha fetch_ref must equal proved_sha")
        return self

    @property
    def is_base_control(self) -> bool:
        """True when this run proves the merge base, not the head."""
        return self.proved_sha != self.head_sha


__all__ = ["SHA_PATTERN", "ModelLabProofSubject"]
