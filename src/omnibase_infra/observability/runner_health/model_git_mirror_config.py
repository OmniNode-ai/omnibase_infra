# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed local git-mirror config for the runner fleet.

OMN-16053 (OMN-14027 C2) — the runner host's bare git mirrors recorded as fleet
config source-of-truth. The mirror is a delta reducer, never a source of truth:
``actions/checkout`` still resolves the exact requested SHA against github.com
over its own authenticated remote, and the runner-side pre-seed is fail-open at
every step, so a stale/absent mirror degrades to a cold clone rather than
failing or mis-resolving a job.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelGitMirrorConfig(BaseModel):
    """Local git-mirror component for the self-hosted runner fleet."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    active: bool = Field(
        default=False,
        description=(
            "Whether the mirror daemon + refresh timer are deployed and the "
            "job-started pre-seed is expected to hydrate from them."
        ),
    )
    bind_address: str = Field(
        ...,
        min_length=1,
        description=(
            "git-daemon listen address. Docker bridge gateway ONLY — git:// is "
            "unauthenticated, so private-repo mirrors must not be reachable "
            "beyond the runner containers."
        ),
    )
    docker_network: str = Field(
        ...,
        min_length=1,
        description="Docker network whose gateway address the daemon binds.",
    )
    port: int = Field(..., ge=1, le=65535, description="git-daemon port.")
    mirror_root: str = Field(
        ...,
        min_length=1,
        description="Host directory holding the bare <repo>.git mirrors.",
    )
    refresh_interval_seconds: int = Field(
        ...,
        ge=1,
        description="Upstream refresh cadence (systemd timer interval).",
    )
    serialized: bool = Field(
        ...,
        description=(
            "One upstream fetch per repo per interval for the whole fleet, "
            "enforced by systemd Type=oneshot plus an flock in the refresh "
            "script. This serialization is the entire point of the component."
        ),
    )
    repos: tuple[str, ...] = Field(
        ...,
        min_length=1,
        description="Mirrored repositories, descending order of CI job volume.",
    )
    runner_allowlist: str = Field(
        default="ALL",
        min_length=1,
        description=(
            "Space-separated RUNNER_NAME allowlist consumed by the pre-seed, "
            "or ALL for fleet-wide."
        ),
    )
    kill_switch_env: str = Field(
        ...,
        min_length=1,
        description="Per-job env var honoured by the pre-seed as a kill switch.",
    )


__all__ = ["ModelGitMirrorConfig"]
