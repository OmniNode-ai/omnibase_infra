# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One declared dev-lane branch install (OMN-17427)."""

from __future__ import annotations

import re
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.models.model_dev_lane_branch_install_file import (
    ModelDevLaneBranchInstallFile,
)

COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{2,79}$")
_REPO = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_TICKET = re.compile(r"^OMN-\d+$")
_SERVICE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_LANE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


class ModelDevLaneBranchInstall(BaseModel):
    """A fix installed from its branch into running dev-lane containers.

    The record is declared before the merge, so it names where the fix comes
    from (repository, PR, branch and head), where it lands (services and the
    container path and sha256 of each file) and when it stops being re-applied
    (an expiry, or the merge).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    repo: str
    pr: int = Field(gt=0)
    branch: str = Field(min_length=1, max_length=200)
    head: str
    ticket: str
    registered_by: str
    services: tuple[str, ...] = Field(min_length=1)
    files: tuple[ModelDevLaneBranchInstallFile, ...] = Field(min_length=1)
    restart: bool = True
    registered_at: datetime
    expires_at: datetime

    @field_validator("id")
    @classmethod
    def _id_shape(cls, value: str) -> str:
        if not _ID.match(value):
            raise ValueError("id must be a lowercase token of 3 to 80 characters")
        return value

    @field_validator("repo")
    @classmethod
    def _repo_shape(cls, value: str) -> str:
        if not _REPO.match(value):
            raise ValueError("repo must be a repository name such as omnimarket")
        return value

    @field_validator("head")
    @classmethod
    def _head_sha(cls, value: str) -> str:
        if not COMMIT_PATTERN.match(value):
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

    @field_validator("services")
    @classmethod
    def _services_shape(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value) or not all(_SERVICE.match(s) for s in value):
            raise ValueError("services must be distinct compose service names")
        return value

    @field_validator("files")
    @classmethod
    def _files_distinct(
        cls, value: tuple[ModelDevLaneBranchInstallFile, ...]
    ) -> tuple[ModelDevLaneBranchInstallFile, ...]:
        if len({f.path for f in value}) != len(value):
            raise ValueError("files must name distinct container paths")
        return value

    @field_validator("registered_at", "expires_at")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamps must carry a timezone")
        return value.astimezone(UTC)
