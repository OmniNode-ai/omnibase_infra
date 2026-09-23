# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The one declaration of how a Kafka CONNECT is retried (OMN-18925).

Two surfaces in this repo open a producer against the same broker, and before
this model they disagreed about what to do when that connect did not land:

* :class:`~omnibase_infra.event_bus.event_bus_kafka.EventBusKafka`, the
  transport the delegate CLI resolves through ``RuntimeLocal``;
* :class:`~omnibase_infra.runtime.providers.provider_kafka_producer.ProviderKafkaProducer`,
  the contract-DI surface materialized into the long-lived runtime host
  process.

Both were single-shot. The event-bus surface *looked* like it was not: it
declares ``max_retry_attempts`` and ``retry_backoff_base``, its ``start()``
docstring promises "connection retry" and "connection failures raise ... after
retries", and its retry loop is real — but that loop wraps PUBLISH, never
connect. A reader checking whether connect was protected found the fields, the
docstring and the loop, and all three agreed with each other and not with the
code. That is worse than an absent retry, because it stops the reader looking.

**What this cost, measured.** On 2026-09-21 at 14:35:54Z the dev Redpanda
emitted ``Reactor stalled for 17222 ms on shard 0`` under host CPU starvation
on ``.201`` — 66 GB of swap in use, load 32.54 on 32 CPUs, 60 concurrent CI
runners. The broker never restarted and never refused; a Seastar reactor that
cannot be scheduled simply stops answering. Against a single ``wait_for`` with
a 10-second deadline that stall is unsurvivable **by construction**: no value
of the deadline that is also a sane deadline outlives a 17-second stall, so the
only fix that works is a second attempt.

**Why one model rather than two sets of fields.** AC-3 of the filing ticket
asks that both surfaces "resolve from one declaration so they cannot drift",
and the drift it is guarding against has already happened once on this exact
pair: ``max_request_size`` had to be pinned across both surfaces under
OMN-16267 after they disagreed about the broker's ``message.max.bytes``. The
same two surfaces, the same broker, the same class of silent disagreement. So
the retry policy is declared once, here, read from the same ``KAFKA_*``
variables the event-bus config already documents, and both surfaces consume it
rather than each carrying their own defaults.

**What this model deliberately does NOT do.** It does not retry forever, does
not widen anybody's deadline, and does not decide *whether* a given exception
is worth retrying — that is the caller's judgement about its own failure modes
and stays at the call site. It only answers "how many more times, how long
between, and what is the total I have promised not to exceed".

.. versionadded:: OMN-18925
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from omnibase_infra.event_bus.models.config.model_kafka_event_bus_config import (
    ModelKafkaEventBusConfig,
)

__all__ = ["ModelKafkaConnectRetryPolicy"]

# The dials are deliberately the ones ModelKafkaEventBusConfig already
# publishes for its publish-side retry, and they are read FROM that config
# rather than from the environment a second time. A connect that backed off
# on a different dial from the publish that follows it would be two policies
# wearing one name, which is the drift AC-3 names.
#
# The defaults below are stated only so this model is constructible on its
# own in a test. In production both come from the bus config, whose bounds
# on these two fields are identical to the ones declared here -- checked, not
# assumed -- so routing through it narrows nothing.
_DEFAULT_MAX_RETRY_ATTEMPTS = 3
_DEFAULT_RETRY_BACKOFF_BASE = 1.0


class ModelKafkaConnectRetryPolicy(BaseModel):
    """How many times a Kafka connect is retried, and how long between tries."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    max_retry_attempts: int = Field(
        default=_DEFAULT_MAX_RETRY_ATTEMPTS,
        ge=0,
        le=10,
        description=(
            "Retries AFTER the first attempt, so the total number of connect "
            "attempts is this plus one. Zero restores the pre-OMN-18925 "
            "single-shot behaviour exactly, which is what makes the negative "
            "control expressible. Shares KAFKA_MAX_RETRY_ATTEMPTS with the "
            "event bus's publish-side retry by design (AC-3)."
        ),
    )
    retry_backoff_base: float = Field(
        default=_DEFAULT_RETRY_BACKOFF_BASE,
        ge=0.001,
        le=60.0,
        description=(
            "Base seconds for exponential backoff between connect attempts: "
            "the delay before retry N is base * 2**N. The lower bound allows "
            "a test to collapse the schedule to microseconds without mocking "
            "the clock."
        ),
    )
    attempt_timeout_seconds: float = Field(
        ...,
        gt=0.0,
        le=300.0,
        description=(
            "The per-attempt connect deadline, supplied by the surface rather "
            "than defaulted here: the event bus and the DI provider have "
            "genuinely different per-attempt budgets (30s and 10s), and "
            "flattening them would be a behaviour change smuggled into a "
            "refactor. What this model unifies is the retry SHAPE, not the "
            "deadline each surface already chose."
        ),
    )

    @property
    def total_attempts(self) -> int:
        """Connect attempts this policy permits, first attempt included."""
        return self.max_retry_attempts + 1

    def backoff_delays(self) -> tuple[float, ...]:
        """Delays between consecutive attempts, in order.

        One shorter than :attr:`total_attempts`: there is no delay after the
        final attempt, because nothing follows it.
        """
        return tuple(
            self.retry_backoff_base * (2**index)
            for index in range(self.max_retry_attempts)
        )

    @property
    def total_bound_seconds(self) -> float:
        """Worst-case wall clock this policy can spend before giving up.

        Every attempt times out AND every backoff is served. Reported rather
        than assumed because it is the number that has to stay under the
        caller's own deadline, and because AC-2 asks the refusal to name the
        bound it exceeded — a bound nobody can compute is one nobody can cite.
        """
        return self.attempt_timeout_seconds * self.total_attempts + sum(
            self.backoff_delays()
        )

    @classmethod
    def from_bus_config(
        cls,
        config: ModelKafkaEventBusConfig,
        *,
        attempt_timeout_seconds: float,
    ) -> ModelKafkaConnectRetryPolicy:
        """Resolve the policy from the overlay-resolved event-bus config.

        The two dials are taken from the config object rather than read out
        of the process environment a second time here. That is not a
        stylistic choice: a direct environment read for a name the overlay
        already resolves is the drift AC-3 names, wearing the disguise of
        agreement -- the two reads would agree only for as long as nobody
        changed how the overlay resolves.
        ``ModelKafkaEventBusConfig.default()`` already applies the
        ``KAFKA_*`` overrides, so a caller that has no config of its own
        still gets the operator's values through one path instead of two.

        Args:
            config: The overlay-resolved Kafka config carrying
                ``max_retry_attempts`` and ``retry_backoff_base``.
            attempt_timeout_seconds: The calling surface's own per-attempt
                connect deadline.

        Raises:
            ValueError: If the resolved values are out of range for a retry
                policy. Note that a NON-NUMERIC dial never reaches here: the
                bus config's own override path logs a warning and keeps its
                default for an unparseable value. That is pre-existing
                behaviour shared with every other dial it resolves, and
                changing it for these two alone would make them behave
                differently from their neighbours -- which is the disagreement
                this model exists to remove. It is worth knowing about and is
                not this change's to fix.
        """
        try:
            return cls(
                max_retry_attempts=config.max_retry_attempts,
                retry_backoff_base=config.retry_backoff_base,
                attempt_timeout_seconds=attempt_timeout_seconds,
            )
        except ValidationError as exc:
            msg = f"Invalid Kafka connect retry configuration: {exc}"
            raise ValueError(msg) from exc
