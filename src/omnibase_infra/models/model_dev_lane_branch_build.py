# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One dev-lane build of an unmerged pull request's branch (OMN-22996)."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_LIFETIME = timedelta(days=7)

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REPO = re.compile(r"^[a-z][a-z0-9_]*$")
_REQUESTER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,79}$")


class ModelDevLaneBranchBuild(BaseModel):
    """A pull request's branch the dev lane was built from, recorded after the build.

    The deploy agent writes it when a dev job that declared an open pull request
    (``open_pr``, ``<repo>#<number>@<head sha>``) verified the lane. It names the
    repository, pull request, branch and head, the command that asked and the job
    that built it, and an expiry at most seven days out, so a build whose pull
    request never merges still returns the lane to dev.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    repo: str
    pr: int = Field(gt=0)
    branch: str = Field(max_length=200)
    head: str
    requested_by: str
    job_id: str = Field(min_length=1, max_length=80)
    recorded_at: datetime
    expires_at: datetime

    @field_validator("repo")
    @classmethod
    def _repo_shape(cls, value: str) -> str:
        if not _REPO.match(value):
            raise ValueError("repo must be a repository name such as omnibase_infra")
        return value

    @field_validator("head")
    @classmethod
    def _head_sha(cls, value: str) -> str:
        if not _COMMIT.match(value):
            raise ValueError("head must be a full 40 character commit sha")
        return value

    @field_validator("requested_by")
    @classmethod
    def _requester_shape(cls, value: str) -> str:
        if not _REQUESTER.match(value):
            raise ValueError("requested_by must be a lane or agent token")
        return value

    @field_validator("recorded_at", "expires_at")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamps must carry a timezone")
        return value.astimezone(UTC)

    def model_post_init(self, __context: object) -> None:
        if self.expires_at <= self.recorded_at:
            raise ValueError("expires_at must be after recorded_at")
        if self.expires_at > self.recorded_at + MAX_LIFETIME:
            raise ValueError(
                f"expires_at is more than {MAX_LIFETIME.days} days after recording"
            )

    @property
    def key(self) -> str:
        """The identity of the build: repository, pull request and head."""
        return f"{self.repo}#{self.pr}@{self.head}"
