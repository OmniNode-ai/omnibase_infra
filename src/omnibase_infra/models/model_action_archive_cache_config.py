# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed action-archive cache config for the runner fleet.

OMN-24299 -- the self-hosted runner's action-archive cache recorded as fleet
config source-of-truth. The runner reads the env var
``ACTIONS_RUNNER_ACTION_ARCHIVE_CACHE`` and, for a step that uses
``owner/repo@<resolved sha>``, looks for the archive file
``<cache dir>/<owner>_<repo>/<resolved sha>.tar.gz`` and copies it instead of
downloading from GitHub. When the archive is absent the runner downloads as
today (fail-open), so a stale/absent cache degrades to a cold download rather
than failing or mis-resolving a job.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelActionArchiveCacheConfig(BaseModel):
    """Action-archive cache component for the self-hosted runner fleet."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    active: bool = Field(
        default=False,
        description=(
            "Whether the action-archive cache is populated and the runner is "
            "expected to resolve action archives from it instead of GitHub."
        ),
    )
    staging_root: str = Field(
        ...,
        min_length=1,
        description=(
            "Deploy-host directory the fill writes the archives into. Must "
            "start with '/' or '~/' (the fill expands '~/' with os.path.expanduser)."
        ),
    )
    container_path: str = Field(
        ...,
        min_length=1,
        description=(
            "Absolute path where each runner container sees the cache, read "
            "only. Must be absolute."
        ),
    )
    local_owner: str = Field(
        ...,
        min_length=1,
        description=(
            "GitHub owner whose action repos are read from the canonical clone "
            "of the same repo name instead of GitHub."
        ),
    )
    remote_url_template: str = Field(
        default="https://github.com/{repo}.git",
        min_length=1,
        description=(
            "Template for third-party action repo URLs; must contain the text "
            "{repo}. Where third-party action repos are fetched from, once per ref."
        ),
    )
    fetch_timeout_seconds: int = Field(
        default=120, ge=1, description="Timeout for a single git call."
    )
    refresh_interval_seconds: int = Field(
        ...,
        ge=1,
        description="Cadence a periodic fill should run at.",
    )

    @field_validator("staging_root")
    @classmethod
    def _validate_staging_root(cls, value: str) -> str:
        if not value.startswith(("/", "~/")):
            raise ValueError("staging_root must start with '/' or '~/'")
        return value

    @field_validator("container_path")
    @classmethod
    def _validate_container_path(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError("container_path must be an absolute path")
        return value

    @field_validator("remote_url_template")
    @classmethod
    def _validate_remote_url_template(cls, value: str) -> str:
        if "{repo}" not in value:
            raise ValueError("remote_url_template must contain the text {repo}")
        return value


__all__ = ["ModelActionArchiveCacheConfig"]
