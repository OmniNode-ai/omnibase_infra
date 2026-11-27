# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request to restore the cache mounts of idle runner containers to the runner user (OMN-26113)."""

from pydantic import BaseModel, ConfigDict, Field


class ModelRunnerCacheOwnershipRestoreRequest(BaseModel):
    """Which containers are runners, who owns their caches, and how long each docker call may take."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    container_match: str
    uid: int = Field(default=1001, ge=1)
    gid: int = Field(default=1001, ge=1)
    docker_timeout_seconds: int = Field(default=60, gt=0)
    chown_timeout_seconds: int = Field(default=900, gt=0)
