# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One path a runtime lane's work uses, declared as data (OMN-26357).

A path is a delegation check, the gateway, the scm write nodes, the work ledger,
deploy, health or onboarding. It names the contracts and topics that path's probe
or command drives; the contracts it reaches from there through topics are computed
from the contract files, never spelled here.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelRuntimeLaneUsePath(BaseModel):
    """The seeds of one path a lane needs: the contracts and topics it starts from."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    id: str = Field(min_length=1, description="The path's id, for example a check id.")
    nodes: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Contract names the path calls by name.",
    )
    topics: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Topics the path publishes to or waits on.",
    )
