# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed Actions tool-cache durability config for the runner fleet.

OMN-16053 (OMN-14027 C2) — records that ``RUNNER_TOOL_CACHE`` lives in the
container filesystem (not a volume), so a fleet ``--force-recreate`` wipes the
warm cache on every runner at once and hands the next wave a fleet-wide cold
CPython+uv download. Fleet recreates must be bracketed by the seed script.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelToolCacheConfig(BaseModel):
    """Actions tool-cache durability record for the self-hosted runner fleet."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    path: str = Field(
        ...,
        min_length=1,
        description="RUNNER_TOOL_CACHE path inside the runner containers.",
    )
    durable: bool = Field(
        ...,
        description=(
            "Whether the tool cache survives a container recreate. False today: "
            "the path is container-filesystem, not a volume."
        ),
    )
    snapshot_root: str = Field(
        ...,
        min_length=1,
        description="Host directory holding the canonical tool-cache snapshot.",
    )
    seed_script: str = Field(
        ...,
        min_length=1,
        description="Repo-relative script that re-seeds the cache after a recreate.",
    )
    recreate_procedure: str = Field(
        ...,
        min_length=1,
        description="Repo-relative runbook documenting the bracketed recreate.",
    )


__all__ = ["ModelToolCacheConfig"]
