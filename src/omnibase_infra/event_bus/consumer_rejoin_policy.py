# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pure decision function for "is this consumer wedged, and should it rejoin?" (OMN-18640).

No I/O, no clock of its own, no aiokafka import. Every input arrives as a
measured :class:`ModelConsumerPollObservation`; the output is a derived
:class:`ModelConsumerStallVerdict`. That split is what makes the 2026-09-17 and
2026-09-18 incidents replayable as a unit test against their recorded numbers
rather than only against a live broker.

**What the detector keys on, and why it is not "no records".** A consumer that
returns no records may be caught up. The discriminator recorded on OMN-18640 is
the pair of facts that disagreed with every green surface during both outages:
the fetch position did not move while the log end offset did. So a stall
requires a POSITIVE backlog that the consumer is not draining, measured against
the partition leaders, which answer independently of the group coordinator the
wedged client cannot reach.

**What it deliberately does NOT key on.** Not the count of
``GroupCoordinatorNotAvailableError`` lines: aiokafka retries the coordinator
inside its own background task and never raises those to the consuming loop, so
counting them means scraping a third-party logger -- a works-by-convention
surface that breaks silently on a dependency bump. The offset evidence is both
stronger and stable across aiokafka versions. The "N consecutive" bound the
brief asks for is applied to confirmed stall EVALUATIONS instead, which bounds
a transient reading exactly as a coordinator-error count would.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.health.enum_consumer_stall_reason import (
    EnumConsumerStallReason,
)
from omnibase_infra.models.health.model_consumer_poll_observation import (
    ModelConsumerPollObservation,
)
from omnibase_infra.models.health.model_consumer_stall_verdict import (
    ModelConsumerStallVerdict,
)


class ModelConsumerRejoinPolicy(BaseModel):
    """Thresholds governing stall detection and the forced rejoin.

    Built from ``ModelKafkaEventBusConfig``; see the field docstrings there for
    why each default is what it is. Declared defaults only -- no environment
    fallback, so a lane cannot quietly disarm the recovery.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    stall_seconds: float = Field(
        ...,
        description="Seconds without a delivered record before a stall is possible.",
        gt=0.0,
    )
    required_consecutive_stalls: int = Field(
        ...,
        description="Confirmations required before a rejoin is ordered.",
        ge=1,
    )
    rejoin_cooldown_seconds: float = Field(
        ...,
        description="Minimum seconds between forced rejoins of the same group.",
        ge=0.0,
    )


def evaluate_consumer_stall(
    observation: ModelConsumerPollObservation,
    policy: ModelConsumerRejoinPolicy,
    *,
    prior_consecutive_stalls: int,
) -> ModelConsumerStallVerdict:
    """Classify one observation.

    Args:
        observation: Measured state of the consumer at this evaluation point.
        policy: Thresholds to apply.
        prior_consecutive_stalls: Confirmations accumulated by earlier
            evaluations. Any non-stalled verdict returns zero, so the caller
            can carry the returned count forward without its own reset logic.

    Returns:
        A verdict whose ``should_rejoin`` is derived, never asserted.
    """
    required = policy.required_consecutive_stalls

    def _clear(reason: EnumConsumerStallReason) -> ModelConsumerStallVerdict:
        return ModelConsumerStallVerdict(
            reason=reason,
            consecutive_stalls=0,
            required_consecutive_stalls=required,
        )

    # A consumer that delivered a record inside the window is working. This is
    # checked first and unconditionally: no other signal can override direct
    # evidence of progress.
    if observation.seconds_since_last_record < policy.stall_seconds:
        return _clear(EnumConsumerStallReason.NOT_STALLED_PROGRESSING)

    # The end-offset probe failed, so the leaders are unreachable. Whatever is
    # wrong is not a client that can see the brokers and not its coordinator,
    # and recreating into an unreachable broker only churns connections.
    if not observation.broker_reachable:
        return _clear(EnumConsumerStallReason.NOT_STALLED_BROKER_UNREACHABLE)

    if observation.assigned_partitions == 0:
        reason = EnumConsumerStallReason.STALLED_NO_ASSIGNMENT
    elif observation.backlog_records > 0:
        reason = EnumConsumerStallReason.STALLED_BACKLOG_NOT_DRAINING
    else:
        # Assigned, caught up, quiet. This is the positive control: an idle
        # consumer must never be recreated, however long it stays quiet.
        return _clear(EnumConsumerStallReason.NOT_STALLED_IDLE)

    # The signature holds. Bound the recreate rate before confirming it: a
    # fault that survives a rejoin must not turn into a recreate loop. The
    # confirmation count is held, not cleared, so the cooldown delays the next
    # rejoin rather than resetting the evidence for it.
    since_rejoin = observation.seconds_since_last_rejoin
    if since_rejoin is not None and since_rejoin < policy.rejoin_cooldown_seconds:
        return ModelConsumerStallVerdict(
            reason=EnumConsumerStallReason.NOT_STALLED_WITHIN_COOLDOWN,
            consecutive_stalls=prior_consecutive_stalls,
            required_consecutive_stalls=required,
        )

    return ModelConsumerStallVerdict(
        reason=reason,
        consecutive_stalls=prior_consecutive_stalls + 1,
        required_consecutive_stalls=required,
    )


__all__ = ["ModelConsumerRejoinPolicy", "evaluate_consumer_stall"]
