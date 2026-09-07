# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Aggregate Kafka consumer fetch-memory budget (OMN-17888).

WHAT THIS BOUNDS, AND WHY IT IS NOT ``max_partition_fetch_bytes``
----------------------------------------------------------------

``EventBusKafka.subscribe()`` creates one ``AIOKafkaConsumer`` per
``(topic, group_id)``, and group ids are scoped per topic
(``.__t.<topic>``) on purpose, to keep a shared group from rebalancing on
every subscription. So the number of live consumers equals the number of
subscribed topics -- ~355 on the .201 dev lane, against a broker carrying
1657 topics at one partition each.

Every one of those consumers passes ``max_partition_fetch_bytes`` (1_048_588,
kept aligned to the producer's ``max_request_size`` by OMN-16267) and,
before this model existed, passed nothing for ``fetch_max_bytes`` -- so each
inherited aiokafka's 52_428_800 default. With one partition per consumer the
effective per-fetch cap is ``min(52_428_800, 1_048_588) = 1_048_588``, and
nothing anywhere bounded the *sum*. Measured on the dev lane: a 694 -> 1512 MB
burst in six seconds at subscription start, then SIGKILL by the cgroup OOM
killer at anon-rss ~1.55 GB against ``memory.max`` 1_610_612_736.

The one-line "fix" -- lowering ``max_partition_fetch_bytes`` -- regresses
OMN-16267 into silent data loss, because that knob is the one carrying the
guarantee that no record the producer can send is unfetchable. This model
bounds the *aggregate* instead, through ``fetch_max_bytes``, which carries the
opposite and explicitly documented guarantee (KIP-74 ``minOneMessage``):

    "This is not an absolute maximum, if the first message in the first
    non-empty partition of the fetch is larger than this value, the message
    will still be returned to ensure that the consumer can make progress."
    -- aiokafka ``AIOKafkaConsumer`` docstring, ``fetch_max_bytes``

So ``fetch_max_bytes`` may legally sit *below* ``max_partition_fetch_bytes``:
a record between the two is still delivered, one fetch at a time. aiokafka
performs no validation coupling the two; they are independent
``FetchRequest`` fields.

THE ARITHMETIC
--------------

::

    L = memory limit in bytes            (source: cgroup, or declared)
    B = memory_fraction * L              (aggregate fetch budget)
    k = in_flight_fetches_per_broker     (one in-flight response + one
                                          buffered FetchResult per node)
    b = brokers_per_consumer
    N = max_concurrent_consumers         (ENFORCED as a subscription cap, so
                                          the divisor is a bound and not an
                                          estimate)

    fetch_max_bytes = floor(B / (k * b * N))

    invariant, asserted here and in the unit tests:
        n * k * b * fetch_max_bytes <= B    for every n <= N

No number in this file has a default. A budget that is not fully declared is a
``ValidationError``; a limit that cannot be resolved is a
``ProtocolConfigurationError``. Neither is ever replaced by an assumed number
-- that assumption is what a silent OOM is made of.

HONEST LIMIT
------------

``fetch_max_bytes`` is a soft cap by design (that is the progress guarantee
above). Per-fetch cost is ``max(fetch_max_bytes, one batch <= max_request_size)``.
If every subscribed partition simultaneously held a record near
``max_request_size``, the aggregate reverts to the unbounded figure. The bound
is therefore a bound on the *steady state*, and the record-size distribution on
the target broker is the thing that makes it a bound in practice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.enums import EnumInfraTransportType, EnumKafkaFetchBudgetSource
from omnibase_infra.errors import (
    ModelInfraErrorContext,
    ProtocolConfigurationError,
)

# cgroup v2 only. cgroup v1 exposes the limit at a different path and an
# unconstrained host reports a sentinel; both resolve to "cannot read a limit
# here", which is a refusal, not a reason to assume one.
CGROUP_V2_MEMORY_MAX_PATH: Final[Path] = Path("/sys/fs/cgroup/memory.max")

# Name of the single deployment variable carrying the whole typed budget as
# JSON. One variable rather than five keeps the compose surface to a single
# fail-closed `${VAR:?...}` line and makes a partially-declared budget
# impossible to express. This module NAMES the variable (so the error text can
# tell an operator what to set) but never READS it: environment resolution is
# the runtime kernel's job, not a config model's.
FETCH_BUDGET_ENV_VAR: Final[str] = "ONEX_KAFKA_CONSUMER_FETCH_BUDGET_JSON"


def _config_error(
    message: str, parameter: str, value: object
) -> ProtocolConfigurationError:
    return ProtocolConfigurationError(
        message,
        context=ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.KAFKA,
            operation="resolve_consumer_fetch_budget",
            target_name="kafka_consumer_fetch_budget",
        ),
        parameter=parameter,
        value=value,
    )


class ModelKafkaConsumerFetchBudget(BaseModel):
    """Declared inputs to the aggregate consumer fetch bound.

    No field carries a numeric default: the bound must derive from a declared
    policy and a real limit, never from a number that happened to be typed
    into a source file. The single ``None`` default on ``memory_limit_bytes``
    is the absence of a declared limit, not a value -- and the model validator
    rejects that absence for the one source that consumes it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    source: EnumKafkaFetchBudgetSource = Field(
        description=(
            "Where the memory limit comes from. CONTAINER_CGROUP_LIMIT reads "
            "the live cgroup v2 limit at bus construction; DECLARED_BYTES uses "
            "memory_limit_bytes."
        ),
    )
    memory_limit_bytes: int | None = Field(
        # None is not a fallback value: it is the *absence* of a declared
        # limit, and _limit_matches_source rejects it outright for the one
        # source that needs it. Defaulting it is what lets the cgroup source
        # be expressed without writing a second, driftable copy of the limit.
        default=None,
        description=(
            "The memory limit in bytes. Required when source is "
            "DECLARED_BYTES; must be omitted when source is "
            "CONTAINER_CGROUP_LIMIT, where a second, possibly-stale copy of "
            "the limit would be exactly the drift this model exists to "
            "prevent."
        ),
        ge=1024 * 1024,
    )
    memory_fraction: float = Field(
        description=(
            "Fraction of the memory limit the aggregate consumer fetch "
            "buffers may occupy. The remainder covers the per-consumer client "
            "baseline (AIOKafkaClient + cluster metadata + coordinator + "
            "heartbeat task), decode, handler state and the interpreter."
        ),
        gt=0.0,
        le=1.0,
    )
    max_concurrent_consumers: int = Field(
        description=(
            "Hard cap on live consumers, and the divisor of the aggregate "
            "budget. EventBusKafka refuses to start consumer N+1 rather than "
            "quietly exceeding the bound -- a loud, attributable startup "
            "failure instead of a silent SIGKILL."
        ),
        ge=1,
    )
    brokers_per_consumer: int = Field(
        description=(
            "Broker nodes a single consumer may hold an in-flight fetch "
            "against. aiokafka groups fetch requests per leader node and skips "
            "a node that already has one in flight, so this multiplies the "
            "per-consumer fetch cost. 1 on a single-node Redpanda; the broker "
            "count on a multi-node cluster."
        ),
        ge=1,
    )
    in_flight_fetches_per_broker: int = Field(
        description=(
            "Fetch responses resident per consumer per broker node. 2 is the "
            "conservative figure: one in-flight response plus one buffered "
            "FetchResult. aiokafka's MemoryRecords wraps the same response "
            "byte slice, so the true figure is nearer 1."
        ),
        ge=1,
    )

    @model_validator(mode="after")
    def _limit_matches_source(self) -> ModelKafkaConsumerFetchBudget:
        """The limit must be declared exactly when, and only when, it is used."""
        if self.source is EnumKafkaFetchBudgetSource.DECLARED_BYTES:
            if self.memory_limit_bytes is None:
                msg = (
                    "memory_limit_bytes is required when source is "
                    "DECLARED_BYTES -- the budget has no limit to divide"
                )
                raise ValueError(msg)
        elif self.memory_limit_bytes is not None:
            msg = (
                "memory_limit_bytes must be omitted when source is "
                "CONTAINER_CGROUP_LIMIT -- the limit is read live from the "
                "cgroup so it cannot drift from the one the kernel enforces"
            )
            raise ValueError(msg)
        return self

    def resolve_memory_limit_bytes(self) -> int:
        """Resolve the memory limit, or refuse.

        Returns:
            The memory limit in bytes.

        Raises:
            ProtocolConfigurationError: source is CONTAINER_CGROUP_LIMIT and the
                cgroup v2 limit file is absent, unreadable, unparseable, or
                reports ``max`` (unconstrained). No assumed limit is
                substituted in any of those cases.
        """
        if self.source is EnumKafkaFetchBudgetSource.DECLARED_BYTES:
            # Guaranteed non-None by _limit_matches_source.
            assert self.memory_limit_bytes is not None
            return self.memory_limit_bytes

        try:
            raw = CGROUP_V2_MEMORY_MAX_PATH.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise _config_error(
                f"cannot read the cgroup v2 memory limit at "
                f"{CGROUP_V2_MEMORY_MAX_PATH} ({exc}). The aggregate Kafka "
                f"consumer fetch bound has no limit to divide and this process "
                f"refuses to assume one. Either run under a memory-limited "
                f"cgroup v2 container, or declare "
                f"source={EnumKafkaFetchBudgetSource.DECLARED_BYTES.value} with "
                f"memory_limit_bytes.",
                parameter="source",
                value=self.source.value,
            ) from exc

        if raw == "max":
            raise _config_error(
                f"{CGROUP_V2_MEMORY_MAX_PATH} reads 'max' -- this cgroup is "
                f"unconstrained, so there is no limit to derive the aggregate "
                f"Kafka consumer fetch bound from. Declare "
                f"source={EnumKafkaFetchBudgetSource.DECLARED_BYTES.value} with "
                f"an explicit memory_limit_bytes instead.",
                parameter="source",
                value=self.source.value,
            )

        try:
            limit = int(raw)
        except ValueError as exc:
            raise _config_error(
                f"{CGROUP_V2_MEMORY_MAX_PATH} is not an integer byte count "
                f"(read {raw!r})",
                parameter="source",
                value=self.source.value,
            ) from exc

        if limit < 1024 * 1024:
            raise _config_error(
                f"{CGROUP_V2_MEMORY_MAX_PATH} reports {limit} bytes, below the "
                f"1 MiB floor -- refusing to derive a fetch bound from it",
                parameter="source",
                value=self.source.value,
            )
        return limit

    def resolve_fetch_max_bytes(self) -> int:
        """Compute the per-consumer ``fetch_max_bytes`` from the declared budget.

        Returns:
            ``floor(memory_fraction * L / (k * b * N))``.

        Raises:
            ProtocolConfigurationError: the limit cannot be resolved, or the
                declared policy divides down to a bound too small to make
                progress on.
        """
        limit = self.resolve_memory_limit_bytes()
        divisor = (
            self.in_flight_fetches_per_broker
            * self.brokers_per_consumer
            * self.max_concurrent_consumers
        )
        fetch_max_bytes = int(self.memory_fraction * limit) // divisor

        # A bound under a Kafka record header is not a bound, it is a stall
        # dressed as one: every fetch would return exactly one record via the
        # minOneMessage path. Refuse loudly rather than ship that.
        minimum_useful_bytes = 16 * 1024
        if fetch_max_bytes < minimum_useful_bytes:
            raise _config_error(
                f"the declared budget divides to fetch_max_bytes="
                f"{fetch_max_bytes} bytes, below the {minimum_useful_bytes} "
                f"byte floor: memory_fraction={self.memory_fraction} x "
                f"limit={limit} / (in_flight_fetches_per_broker="
                f"{self.in_flight_fetches_per_broker} x brokers_per_consumer="
                f"{self.brokers_per_consumer} x max_concurrent_consumers="
                f"{self.max_concurrent_consumers}). Raise memory_fraction or "
                f"lower max_concurrent_consumers.",
                parameter="memory_fraction",
                value=self.memory_fraction,
            )
        return fetch_max_bytes

    @classmethod
    def from_declaration(cls, raw: str | None) -> ModelKafkaConsumerFetchBudget:
        """Parse a declared budget, or refuse.

        ``raw`` is the value the runtime kernel resolved for
        ``ONEX_KAFKA_CONSUMER_FETCH_BUDGET_JSON`` -- rendered from
        ``contracts/services/runtime_policy.contract.yaml`` into
        ``docker/runtime-policy.env`` and passed by compose in the fail-closed
        ``${VAR:?...}`` form, so a runtime container that has not declared a
        budget does not start at all.

        Args:
            raw: The declared budget as JSON, or None/blank when undeclared.

        Returns:
            The declared budget.

        Raises:
            ProtocolConfigurationError: ``raw`` is absent, blank, or does not
                parse to a complete budget. There is no fallback value -- an
                undeclared budget is the unbounded state this exists to remove.
        """
        if raw is None or not raw.strip():
            raise _config_error(
                f"{FETCH_BUDGET_ENV_VAR} is not set. The aggregate Kafka "
                f"consumer fetch bound is contract-declared and has no "
                f"default: it is rendered from "
                f"contracts/services/runtime_policy.contract.yaml into "
                f"docker/runtime-policy.env and passed through compose. An "
                f"unbounded aggregate fetch is what OOM-killed the runtime "
                f"(OMN-17888), so this refuses rather than choosing a number.",
                parameter=FETCH_BUDGET_ENV_VAR,
                value=raw,
            )
        try:
            return cls.model_validate_json(raw)
        except ValueError as exc:
            raise _config_error(
                f"{FETCH_BUDGET_ENV_VAR} does not parse to a complete "
                f"consumer fetch budget: {exc}",
                parameter=FETCH_BUDGET_ENV_VAR,
                value="<redacted: see contract>",
            ) from exc
