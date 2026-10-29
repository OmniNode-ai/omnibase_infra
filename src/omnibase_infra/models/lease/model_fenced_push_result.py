# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The answer to a fenced push (OMN-21426)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

FencedPushRefusal = Literal[
    "",
    "LEASE_FENCE_STALE",
    "LEASE_EXPIRED",
    "LEASE_UNREADABLE",
    "PUSH_REJECTED",
]


class ModelFencedPushResult(BaseModel):
    """``pushed``, or ``refused`` with the reason; a lease refusal made no push.

    A pushed result with ``fence_held_after_push=False`` needs quarantine: the
    post-check found the grant lost or unreadable. Git cannot enforce an etcd
    fence atomically, so an intervening stale push is reported, not prevented.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    outcome: Literal["pushed", "refused"]
    reason: FencedPushRefusal = ""
    kind: str
    resource: str
    fence: int = Field(ge=0)
    fence_held_after_push: bool = Field(
        description="True only when an accepted push's post-check reads the fence OK"
    )
    detail: str = ""


__all__ = ["FencedPushRefusal", "ModelFencedPushResult"]
