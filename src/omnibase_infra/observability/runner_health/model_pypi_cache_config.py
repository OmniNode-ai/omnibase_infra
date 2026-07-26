# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed PyPI pull-through cache endpoint config for the runner fleet.

OMN-14027 C1 — the runner fleet's egress cache endpoint (devpi) recorded as
fleet config source-of-truth. This is a shovel-ready/inert record: ``active``
stays ``False`` until the soak-gated rollout stands up the cache
(``docker/docker-compose.pypi-cache.yml``) and wires the runner env. Recording
the endpoint here does not activate it.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelPyPICacheConfig(BaseModel):
    """PyPI pull-through cache endpoint for the self-hosted runner fleet."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    active: bool = Field(
        default=False,
        description=(
            "Whether the fleet is wired to the cache. Stays False until the "
            "OMN-14027 soak-gated rollout activates the devpi service and the "
            "matching UV_DEFAULT_INDEX runner env."
        ),
    )
    host: str = Field(
        ...,
        min_length=1,
        description="Cache host (Tailscale MagicDNS name, matching runner_host).",
    )
    port: int = Field(..., ge=1, le=65535, description="Published devpi port.")
    simple_index_url: str = Field(
        ...,
        min_length=1,
        description="devpi root/pypi pull-through simple index URL.",
    )
    fallback_index_url: str = Field(
        ...,
        min_length=1,
        description=(
            "PyPI fallback index so a cache miss/outage degrades, not fails-closed."
        ),
    )
    target_hit_rate: float = Field(
        default=0.90,
        gt=0.0,
        le=1.0,
        description="Steady-state wheel-cache hit-rate acceptance target.",
    )


__all__ = ["ModelPyPICacheConfig"]
