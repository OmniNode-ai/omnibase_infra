# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A fix worker's push, carrying the fence of the lease it holds (OMN-21426)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelFencedPushRequest(BaseModel):
    """Push ``worktree``'s HEAD to ``branch`` after checking ``fence`` is live.

    The fence is re-checked after acceptance. The push timeout bounds the
    intervening window; ``--force-with-lease`` requires ``expected_remote_sha``
    to remain the remote branch head. It cannot enforce an etcd fence: a stale
    push is reported by the result for quarantine, not prevented.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    kind: str = Field(min_length=1)
    resource: str = Field(min_length=1, pattern=r"^[^/]+$")
    fence: int = Field(ge=1)
    worktree: str = Field(min_length=1)
    remote: str = Field(default="origin", min_length=1, pattern=r"^[^-]")
    branch: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
    expected_remote_sha: str = Field(pattern=r"^[0-9a-f]{40}$")


__all__ = ["ModelFencedPushRequest"]
