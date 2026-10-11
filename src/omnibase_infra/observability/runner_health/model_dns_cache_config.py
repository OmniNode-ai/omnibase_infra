# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed local DNS-cache endpoint config for the runner fleet.

OMN-15736 — the runner fleet's local caching DNS resolver (unbound) recorded
as fleet config source-of-truth. This is a shovel-ready/inert record:
``active`` stays ``False`` until the operator-gated rollout stands up the
cache (``docker/docker-compose.dns-cache.yml``) and wires a canary subset of
runners' ``dns:`` directive (``docker/docker-compose.dns-canary.yml``).
Recording the endpoint here does not activate it.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelDnsCacheConfig(BaseModel):
    """Local caching DNS resolver endpoint for the self-hosted runner fleet."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    active: bool = Field(
        default=False,
        description=(
            "Whether the fleet is wired to the cache. Stays False until the "
            "OMN-15736 operator-gated rollout activates the unbound service "
            "and repoints runner `dns:` directives."
        ),
    )
    host: str = Field(
        ...,
        min_length=1,
        description=(
            "Cache host as a raw IP. `dns:` (the compose/Docker container "
            "directive) requires a raw IP, not a Tailscale MagicDNS hostname, "
            "unlike the pypi_cache endpoint above."
        ),
    )
    port: int = Field(..., ge=1, le=65535, description="Published DNS port (53).")
    upstream_forwarders: tuple[str, ...] = Field(
        ...,
        min_length=1,
        description=(
            "Existing systemd-resolved upstream + fallback resolvers the "
            "cache forwards to, unchanged — the cache is additive, not a "
            "resolver replacement."
        ),
    )
    target_hit_rate: float = Field(
        default=0.80,
        gt=0.0,
        le=1.0,
        description="Steady-state cache hit-rate acceptance target (AC1).",
    )
    target_dns_failure_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Acceptance target for DNS-class job failures on canary runners "
            "over the busy-window observation period (AC3)."
        ),
    )


__all__ = ["ModelDnsCacheConfig"]
