# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One declared runner host in the fleet inventory (OMN-17477)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.observability.runner_health.enum_runner_host_arch import (
    EnumRunnerHostArch,
)


class ModelRunnerFleetHost(BaseModel):
    """A single host that carries self-hosted runners.

    The fleet was single-host by construction until this model existed: the
    config carried scalar ``runner_host`` / ``runner_name_prefix`` /
    ``expected_count`` fields, so a second host was not unconfigured, it was
    unrepresentable.

    Three of these fields exist because a second host needs something a scalar
    cannot express:

    ``arch``
        Which CPU the host is. The lab's second and third hosts are arm64 Macs,
        and a job that assumes amd64 landing on one of them fails in a way that
        looks like a flaky test rather than a placement error. It is declared
        here so the registered runner can carry a matching ``arch-*`` label and
        a workload can pin an architecture when it needs one.

    ``runner_name_prefix``
        Runner and container names are ``<prefix>-<N>``. Two hosts sharing a
        prefix collide on both, and the collision is silent -- the loser
        re-registers over the winner. The prefix is per host so the namespace
        is partitioned by construction.

    ``classes``
        Which runner classes this host carries. Capacity is not fungible: a
        verify-class runner cannot pick up an action-class job. Counting one in
        the other's total would raise the action fleet's degraded floor by
        capacity that can never satisfy it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    host: str = Field(
        ...,
        min_length=1,
        description=(
            "Stable address of the host. Prefer a Tailscale MagicDNS name over "
            "a LAN address so deploy and monitor flows do not break when the "
            "host's address changes."
        ),
    )
    arch: EnumRunnerHostArch = Field(
        ...,
        description="CPU architecture of the host, in Docker's TARGETARCH spelling.",
    )
    runner_name_prefix: str = Field(
        ...,
        min_length=1,
        description="Prefix for this host's runner and container names; unique across hosts.",
    )
    expected_count: int = Field(
        ...,
        ge=0,
        description=(
            "Declared steady-state runner count on this host, across the classes "
            "below. Zero is legal and means a host that is inventoried but "
            "carries no runners yet."
        ),
    )
    classes: tuple[str, ...] = Field(
        ...,
        min_length=1,
        description=(
            "Runner classes this host carries (for example 'action', 'deploy', "
            "'verify'). Never empty: a host carrying no class is capacity "
            "nothing can ever use, which is a declaration error rather than a "
            "neutral default."
        ),
    )

    @property
    def arch_label(self) -> str:
        """The routing label a runner on this host registers with."""
        return f"arch-{self.arch.value}"


__all__ = ["ModelRunnerFleetHost"]
