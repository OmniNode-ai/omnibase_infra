# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The branch whose code the dev deploy agent loads, until it expires (OMN-23039)."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_LIFETIME = timedelta(days=7)

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_TICKET = re.compile(r"^OMN-\d+$")
_LANE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
#: ``main`` is release-synced and never tracked (OMN-16442); ``dev`` is the tracking
#: ref a pin overrides, so pinning it would change nothing.
_REFUSED_BRANCHES = frozenset({"main", "master", "dev"})


class ModelDevLaneCodeRefPin(BaseModel):
    """A fix to the agent's own code, installed from its branch before the merge.

    The record names the branch whose code the agent loads, the head it was
    registered at, the pull request, ticket and lane that registered it, and an
    expiry at most seven days out. It lives in the agent's state directory, not
    in the repository, because it has to exist before the merge.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    branch: str = Field(min_length=1, max_length=200)
    head: str
    pr: int = Field(gt=0)
    ticket: str
    registered_by: str
    registered_at: datetime
    expires_at: datetime

    @field_validator("branch")
    @classmethod
    def _branch_shape(cls, value: str) -> str:
        if value.startswith("origin/"):
            raise ValueError("branch must be a bare name, not a remote-tracking ref")
        if value in _REFUSED_BRANCHES:
            raise ValueError(f"branch {value!r} cannot be pinned")
        return value

    @field_validator("head")
    @classmethod
    def _head_sha(cls, value: str) -> str:
        if not _COMMIT.match(value):
            raise ValueError("head must be a full 40 character commit sha")
        return value

    @field_validator("ticket")
    @classmethod
    def _ticket_shape(cls, value: str) -> str:
        if not _TICKET.match(value):
            raise ValueError("ticket must look like OMN-123")
        return value

    @field_validator("registered_by")
    @classmethod
    def _lane_shape(cls, value: str) -> str:
        if not _LANE_TOKEN.match(value):
            raise ValueError("registered_by must be a lane token")
        return value

    @field_validator("registered_at", "expires_at")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamps must carry a timezone")
        return value.astimezone(UTC)

    def model_post_init(self, __context: object) -> None:
        if self.expires_at <= self.registered_at:
            raise ValueError("expires_at must be after registered_at")
        if self.expires_at > self.registered_at + MAX_LIFETIME:
            raise ValueError(
                f"expires_at is more than {MAX_LIFETIME.days} days after registration"
            )
