# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declared ownership contract for the bounded Docker network janitor."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field


class ModelNetworkOwnershipRule(BaseModel):
    """A single declared ownership rule for janitor-reclaimable networks.

    Ownership is *opt-in and explicit*. A network is only ever a deletion
    candidate when its name matches ``name_pattern`` (a full-match regex) and
    it is older than ``min_age_seconds``. Anything that does not match a
    declared rule defaults to preserve — a naming mistake can never become
    destructive.

    The pattern is anchored with ``fullmatch`` semantics at evaluation time,
    so ``omnibase-infra-boot-.*`` will not accidentally match an unrelated
    network that merely contains that substring.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    name: str = Field(..., min_length=1, description="Human-readable rule identifier")
    name_pattern: str = Field(
        ...,
        min_length=1,
        description="Regex (fullmatch) for network names this rule owns",
    )
    min_age_seconds: int = Field(
        ...,
        ge=0,
        description="Network must be older than this to be reclaim-eligible",
    )
    description: str = Field(
        default="", description="Why these networks are safe to reclaim when stale"
    )

    def matches_name(self, network_name: str) -> bool:
        """Return True if ``network_name`` is owned by this rule (full match)."""
        return re.fullmatch(self.name_pattern, network_name) is not None


# Canonical ownership contract for the runner fleet.
#
# The runtime-boot and migration CI workflows create ephemeral compose stacks
# under per-run project names (``omnibase-infra-boot-<run>-<attempt>`` and
# ``omnibase-infra-<boot-id>``). Docker derives a network named
# ``<project>_<network>`` or ``<project>-network`` from those projects. When a
# job dies before its teardown step runs, the network leaks. These rules let
# the janitor reclaim ONLY those clearly-owned, idle, aged-out leftovers.
#
# Age threshold (2h) is deliberately conservative: a CI boot job has a 20m
# timeout, so any owned network older than 2h with no attached containers is
# unambiguously abandoned.
DEFAULT_OWNERSHIP_RULES: tuple[ModelNetworkOwnershipRule, ...] = (
    ModelNetworkOwnershipRule(
        name="runtime-boot-compose-networks",
        # Matches both `<project>_<net>` and `<project>-network` derivations of
        # the per-run compose project names used by reusable-runtime-boot.yml.
        name_pattern=r"omnibase-infra-boot-[a-z0-9][a-z0-9_-]*",
        min_age_seconds=2 * 60 * 60,
        description=(
            "Ephemeral compose networks from reusable-runtime-boot.yml; "
            "leaked when a boot job dies before its teardown step."
        ),
    ),
)


__all__ = ["ModelNetworkOwnershipRule", "DEFAULT_OWNERSHIP_RULES"]
