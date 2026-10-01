# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One additional runner pool on a declared host (OMN-19895)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelRunnerFleetPool(BaseModel):
    """A second (third, ...) class of runner on a host that already has one.

    A host row carries one ``runner_name_prefix``. That was enough while every
    host carried one class of runner, and it stopped being enough when the
    action fleet was spread off the primary host (OMN-19895): .202 already
    carries ``omnipc2-verify-runner`` and now also carries CI runners and a
    customer-plane runner, each of which needs its own name prefix so names
    never collide and the fleet probe can tell which host a runner is on. The
    primary host likewise carries verify and customer-plane runners whose names
    no inventory row declared, so the probe counted them toward no host.

    A pool is the same three facts a host row carries for its first class:
    a name prefix, a declared count and the classes it serves. Its prefix obeys
    the same uniqueness and non-nesting rules as every host prefix, checked
    across the whole inventory by ``ModelRunnerFleetConfig``.

    Where its services are defined: in the primary host's compose file when the
    pool is on the primary host, otherwise in
    ``docker/docker-compose.runners-<runner_name_prefix>.yml``, the same rule a
    non-primary host row follows.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    runner_name_prefix: str = Field(
        ...,
        min_length=1,
        description="Prefix for this pool's runner and container names; unique across the inventory.",
    )
    expected_count: int = Field(
        ...,
        ge=0,
        description="Declared steady-state runner count of this pool.",
    )
    classes: tuple[str, ...] = Field(
        ...,
        min_length=1,
        description=(
            "Runner classes this pool serves (for example 'action', 'verify', "
            "'customer-plane'). Never empty, for the same reason as a host row."
        ),
    )


__all__ = ["ModelRunnerFleetPool"]
