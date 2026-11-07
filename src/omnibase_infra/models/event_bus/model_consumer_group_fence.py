# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The fence a consumer-group handover takes on its source group (OMN-22714).

A fold retires a consumer's container and the runtime takes its group's
position over. If the retired group still has a member when the runtime reads
its commit, that member goes on consuming past the copied position: under the
writers' rule the records are projected twice, and under ``last_ran`` an effect
runs twice, or the retired side's later commit is lost to a second fold. The
one-runtime plan's rule (T8, and every fold after it): nothing takes a group's
position over until the retired group is confirmed to have no members.

The handover therefore describes the retired group before it reads a single
offset and proceeds only when the group holds no member. A member that has just
left without a clean leave stays listed until its session times out, so the
fence waits up to ``wait_seconds``, describing every ``poll_seconds``, and then
refuses. A group that cannot be described is refused too, never read as empty.

Declared under the handover, with these defaults::

    consumer_group_handover:
      from_group_id: agent-observability-postgres
      from_auto_offset_reset: earliest
      fence:
        wait_seconds: 60
        poll_seconds: 2
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ModelConsumerGroupFence"]


class ModelConsumerGroupFence(BaseModel):
    """How long a handover waits for its source group to have no members."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    wait_seconds: float = Field(
        default=60.0,
        ge=0.0,
        description=(
            "Longest wait for the retired group to empty before the handover "
            "refuses; covers a member that left without a clean leave and is "
            "listed until its session times out."
        ),
    )
    poll_seconds: float = Field(
        default=2.0,
        gt=0.0,
        description="Interval between two describes of the retired group.",
    )
