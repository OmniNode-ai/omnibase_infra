# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Typed routing policy — the node's contract `config:` block, parsed.

WHY THIS MODEL EXISTS RATHER THAN A DICT. The thresholds decide where every CI
job in a repository executes. Read as a mapping they are a silent-default
hazard: a missing key reads as "no constraint" and a routing gate quietly stops
gating (CLAUDE.md rule 8). Every field below is REQUIRED and typed, so a
contract missing one fails at load with the field's name rather than at the
first run that needed it.

WHERE THE VALUES COME FROM. The node's own ``contract.yaml`` `config:` block,
which is the declaration of record. They are not read from an environment
variable, not hardcoded in a workflow file, and not duplicated in a second
config file.

Ticket: OMN-18412
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelCIRunnerRoutePolicy(BaseModel):
    """The declared routing thresholds, labels and refusal policy."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    policy_version: int = Field(
        gt=0,
        description="Version of the declared policy, echoed on every decision so a "
        "recorded decision can be read against the policy that produced it.",
    )
    runner_group: str = Field(
        min_length=1,
        description="Self-hosted runner group the capacity observation is taken from.",
    )
    lab_labels: tuple[str, ...] = Field(
        min_length=1,
        description="Labels that name the lab fleet. Never invented by the decision: "
        "they may only be chosen when the seam ceiling already carries them.",
    )
    hosted_labels: tuple[str, ...] = Field(
        min_length=1,
        description="Labels that name GitHub-hosted compute; the fallback placement.",
    )
    min_idle_runners: int = Field(
        ge=0,
        description="Headroom FLOOR, not a capacity match: below this many idle "
        "runners, routing stops adding load to the fleet.",
    )
    max_busy_fraction: float = Field(
        gt=0.0,
        le=1.0,
        description="Busy/online fraction at or above which the fleet is saturated.",
    )
    min_online_fraction: float = Field(
        gt=0.0,
        le=1.0,
        description="Fraction of the DECLARED fleet inventory below which the fleet "
        "is not trusted at all. A fraction rather than a count because the count "
        "goes stale the moment the fleet is resized: the previous literal was 60 "
        "against an 88-runner fleet, and when the fleet was capped to 60 that same "
        "literal became equal to the whole fleet, so a single runner going offline "
        "would have refused the fleet on every run. Expressed against the inventory "
        "it tracks a resize instead. A separate and lower floor than the fleet "
        "canary's own health threshold: the canary asks whether the fleet is "
        "healthy, this asks whether it is big enough to route onto at all.",
    )
    lab_record_max_age_seconds: int = Field(
        gt=0,
        description="Freshness bound on the lab-load observation. Older is UNKNOWN, "
        "and unknown routes hosted -- never 'assume ample'.",
    )
    max_lab_load_ratio: float = Field(
        gt=0.0,
        description="Lab host load1/cores above which the lab is saturated.",
    )
    min_lab_free_mem_mib: int = Field(
        ge=0,
        description="Lab host free memory floor. Load ranks; memory ADMITS.",
    )
    capacity_downgrade_reasons: tuple[str, ...] = Field(
        min_length=1,
        description="The reasons that mean 'the fleet is unavailable for THIS run'. "
        "Only these may be reversed for a repository that may not run hosted. "
        "Reversing a trust reason instead would place untrusted code on the fleet, "
        "which no capacity argument may ever do.",
    )
    private_repo_hosted_placement: str = Field(
        pattern="^(refuse)$",
        description="What to do when a PRIVATE repository's only allowed placement "
        "is hosted. 'refuse' is the only accepted value: the 2026-09-14 operator "
        "ruling forbids that placement, and a hosted job in a private repository "
        "does not execute in any case. Declared rather than implied so the rule is "
        "readable in the contract instead of only in the handler.",
    )


__all__ = ["ModelCIRunnerRoutePolicy"]
