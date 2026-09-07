# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Aggregate consumer fetch-memory bound (OMN-17888).

``EventBusKafka`` creates one ``AIOKafkaConsumer`` per subscribed topic. On the
.201 dev lane that is ~355 live consumers, each passing
``max_partition_fetch_bytes=1_048_588`` and -- before this change -- nothing for
``fetch_max_bytes``, so each inherited aiokafka's 52_428_800 default and nothing
bounded the SUM. Measured consequence: 694 -> 1512 MB in six seconds at
subscription start, then SIGKILL by the cgroup OOM killer at anon-rss ~1.55 GB
against ``memory.max`` 1_610_612_736.

These tests pin the fix and, just as importantly, pin the thing the fix must NOT
do: ``max_partition_fetch_bytes`` still equals the producer's
``max_request_size`` at every construction site, because that knob is the one
carrying the OMN-16267 guarantee that no record the producer can send is
unfetchable. A future lane that "fixes memory" by lowering it fails this file.

The container memory limit is INJECTED here, never read from the host: a suite
whose verdict depends on which machine runs it proves nothing.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiokafka.errors import UnknownTopicOrPartitionError

from omnibase_infra.enums import EnumKafkaFetchBudgetSource
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import (
    ModelKafkaConsumerFetchBudget,
    ModelKafkaEventBusConfig,
)

TEST_BOOTSTRAP_SERVERS: str = "localhost:9092"

# The .201 dev lane's real limit: docker inspect .HostConfig.Memory, and
# docker/docker-compose.infra.yml x-runtime-base deploy.resources.limits.memory
# 1536M. Injected, not read -- see the module docstring.
DEV_LANE_MEMORY_LIMIT_BYTES: int = 1_610_612_736
DEV_LANE_MEMORY_FRACTION: float = 0.15
DEV_LANE_MAX_CONSUMERS: int = 512
DEV_LANE_IN_FLIGHT_PER_BROKER: int = 2


def _budget(
    *,
    max_concurrent_consumers: int = DEV_LANE_MAX_CONSUMERS,
    brokers_per_consumer: int = 1,
) -> ModelKafkaConsumerFetchBudget:
    return ModelKafkaConsumerFetchBudget(
        source=EnumKafkaFetchBudgetSource.DECLARED_BYTES,
        memory_limit_bytes=DEV_LANE_MEMORY_LIMIT_BYTES,
        memory_fraction=DEV_LANE_MEMORY_FRACTION,
        max_concurrent_consumers=max_concurrent_consumers,
        brokers_per_consumer=brokers_per_consumer,
        in_flight_fetches_per_broker=DEV_LANE_IN_FLIGHT_PER_BROKER,
    )


def _config(
    budget: ModelKafkaConsumerFetchBudget | None = None,
) -> ModelKafkaEventBusConfig:
    return ModelKafkaEventBusConfig(
        bootstrap_servers=TEST_BOOTSTRAP_SERVERS,
        consumer_fetch_budget=budget if budget is not None else _budget(),
    )


@pytest.fixture
def mock_producer() -> AsyncMock:
    producer = AsyncMock()
    producer.start = AsyncMock()
    producer.stop = AsyncMock()
    producer._closed = False
    return producer


def _consumer_cls(count: int = 1) -> MagicMock:
    consumers = []
    for _ in range(count):
        consumer = AsyncMock()
        consumer.start = AsyncMock()
        consumer.stop = AsyncMock()
        consumers.append(consumer)
    return MagicMock(side_effect=consumers)


@pytest.mark.unit
class TestConsumerFetchMemoryBound:
    """The aggregate bound reaches every consumer and holds at every N."""

    @pytest.mark.asyncio
    async def test_fetch_max_bytes_passed_to_initial_construction_site(
        self, mock_producer: AsyncMock
    ) -> None:
        """The first AIOKafkaConsumer() gets the derived fetch_max_bytes.

        RED on the parent commit: the kwarg is never passed at all
        (``grep -rn fetch_max_bytes src/ tests/`` returns zero rows there;
        positive control -- ``max_partition_fetch_bytes`` returns 13).
        """
        consumer_cls = _consumer_cls()
        with (
            patch(
                "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
                return_value=mock_producer,
            ),
            patch(
                "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer",
                consumer_cls,
            ),
        ):
            config = _config()
            bus = EventBusKafka(config=config)
            await bus._start_consumer_for_topic("events", "my-group")

            kwargs = consumer_cls.call_args.kwargs
            assert "fetch_max_bytes" in kwargs
            assert config.consumer_fetch_budget is not None
            assert (
                kwargs["fetch_max_bytes"]
                == config.consumer_fetch_budget.resolve_fetch_max_bytes()
            )
            # The dev-lane instantiation, spelled out so a change to any input
            # shows up here as a number and not just as a recomputed equality.
            assert kwargs["fetch_max_bytes"] == 235_929

    @pytest.mark.asyncio
    async def test_fetch_max_bytes_passed_to_metadata_retry_construction_site(
        self, mock_producer: AsyncMock
    ) -> None:
        """The consumer recreated inside the metadata-retry loop gets it too.

        Both construction sites, not just the first -- the recreate path is how
        every topic whose metadata has not propagated yet gets its consumer, so
        a bound applied only to the first site leaks on exactly the cold-start
        burst that OOM-killed the runtime.
        """
        failing = AsyncMock()
        failing.start = AsyncMock(
            side_effect=UnknownTopicOrPartitionError("metadata not ready")
        )
        failing.stop = AsyncMock()
        succeeding = AsyncMock()
        succeeding.start = AsyncMock()
        succeeding.stop = AsyncMock()
        consumer_cls = MagicMock(side_effect=[failing, succeeding])

        with (
            patch(
                "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
                return_value=mock_producer,
            ),
            patch(
                "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer",
                consumer_cls,
            ),
            patch(
                "omnibase_infra.event_bus.event_bus_kafka.asyncio.sleep",
                AsyncMock(),
            ),
        ):
            config = _config()
            bus = EventBusKafka(config=config)
            await bus._start_consumer_for_topic("events", "my-group")

            assert consumer_cls.call_count == 2
            recreated = consumer_cls.call_args_list[1].kwargs
            assert config.consumer_fetch_budget is not None
            assert (
                recreated["fetch_max_bytes"]
                == config.consumer_fetch_budget.resolve_fetch_max_bytes()
            )

    @pytest.mark.parametrize("n", [1, 64, 355, 512])
    @pytest.mark.asyncio
    async def test_aggregate_bound_holds_at_every_n_up_to_the_cap(
        self, mock_producer: AsyncMock, n: int
    ) -> None:
        """n consumers x k x b x fetch_max_bytes <= fraction x limit, for n <= N.

        RED on the parent commit at every n above 4: with aiokafka's 52_428_800
        default the left side at n=355 is 37_224_448_000 bytes against a
        241_591_910 byte budget -- off by 154x.

        n=355 is the count measured live on the dev lane from the container's
        own "Updating subscribed topics" log lines.
        """
        consumer_cls = _consumer_cls(count=n)
        with (
            patch(
                "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
                return_value=mock_producer,
            ),
            patch(
                "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer",
                consumer_cls,
            ),
        ):
            bus = EventBusKafka(config=_config())
            for index in range(n):
                await bus._start_consumer_for_topic(f"topic-{index}", f"group-{index}")

            fetch_values = {
                call.kwargs["fetch_max_bytes"] for call in consumer_cls.call_args_list
            }
            assert len(fetch_values) == 1, "the bound must be uniform across consumers"
            fetch_max_bytes = fetch_values.pop()

            budget_bytes = int(DEV_LANE_MEMORY_FRACTION * DEV_LANE_MEMORY_LIMIT_BYTES)
            resident = n * DEV_LANE_IN_FLIGHT_PER_BROKER * 1 * fetch_max_bytes
            assert resident <= budget_bytes, (
                f"{n} consumers x {DEV_LANE_IN_FLIGHT_PER_BROKER} in-flight x "
                f"{fetch_max_bytes} B = {resident} B exceeds the declared "
                f"{budget_bytes} B fetch budget"
            )

    @pytest.mark.parametrize("n", [1, 64, 355, 512])
    @pytest.mark.asyncio
    async def test_per_partition_floor_is_preserved_alongside_the_bound(
        self, mock_producer: AsyncMock, n: int
    ) -> None:
        """OMN-16267 regression guard. GREEN before AND after, deliberately.

        ``max_partition_fetch_bytes`` still equals the producer's
        ``max_request_size`` at every construction site. The aggregate bound
        sits BELOW it, which is legal and intended: aiokafka documents
        ``fetch_max_bytes`` as "not an absolute maximum, if the first message in
        the first non-empty partition of the fetch is larger than this value,
        the message will still be returned to ensure that the consumer can make
        progress" (KIP-74 minOneMessage). Lowering the per-partition knob
        instead would reintroduce RecordTooLargeError / skip-and-advance -- and
        would fail this test, which is the point of pinning it here.
        """
        consumer_cls = _consumer_cls(count=n)
        with (
            patch(
                "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
                return_value=mock_producer,
            ),
            patch(
                "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer",
                consumer_cls,
            ),
        ):
            config = _config()
            bus = EventBusKafka(config=config)
            for index in range(n):
                await bus._start_consumer_for_topic(f"topic-{index}", f"group-{index}")

            for call in consumer_cls.call_args_list:
                assert (
                    call.kwargs["max_partition_fetch_bytes"] == config.max_request_size
                )
                assert call.kwargs["max_partition_fetch_bytes"] >= 1_048_588
                assert (
                    call.kwargs["fetch_max_bytes"]
                    < call.kwargs["max_partition_fetch_bytes"]
                ), "the aggregate cap is expected to sit below the per-partition floor"

    @pytest.mark.asyncio
    async def test_broker_fanout_shrinks_the_per_consumer_share(
        self, mock_producer: AsyncMock
    ) -> None:
        """brokers_per_consumer divides the share; the invariant still holds.

        aiokafka groups fetch requests per leader node and skips a node that
        already has one in flight, so a three-broker cluster holds three times
        the resident fetch bytes per consumer.
        """
        consumer_cls = _consumer_cls(count=2)
        with (
            patch(
                "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
                return_value=mock_producer,
            ),
            patch(
                "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer",
                consumer_cls,
            ),
        ):
            single = EventBusKafka(config=_config(_budget(brokers_per_consumer=1)))
            await single._start_consumer_for_topic("a", "g-a")
            single_broker_bytes = consumer_cls.call_args.kwargs["fetch_max_bytes"]

            triple = EventBusKafka(config=_config(_budget(brokers_per_consumer=3)))
            await triple._start_consumer_for_topic("b", "g-b")
            triple_broker_bytes = consumer_cls.call_args.kwargs["fetch_max_bytes"]

        assert triple_broker_bytes == single_broker_bytes // 3

    @pytest.mark.asyncio
    async def test_subscription_cap_is_enforced(self, mock_producer: AsyncMock) -> None:
        """Consumer N+1 is refused by name, not silently allowed.

        RED on the parent commit: the consumer count is unbounded, so the
        divisor in the budget arithmetic would be an estimate rather than a
        bound. Refusing converts a silent SIGKILL into an attributable
        configuration error.
        """
        cap = 4
        consumer_cls = _consumer_cls(count=cap + 1)
        with (
            patch(
                "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
                return_value=mock_producer,
            ),
            patch(
                "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer",
                consumer_cls,
            ),
        ):
            bus = EventBusKafka(
                config=_config(_budget(max_concurrent_consumers=cap)),
            )
            for index in range(cap):
                await bus._start_consumer_for_topic(f"topic-{index}", f"group-{index}")

            with pytest.raises(ProtocolConfigurationError) as excinfo:
                await bus._start_consumer_for_topic("one-too-many", "group-extra")

        message = str(excinfo.value)
        assert "max_concurrent_consumers" in message
        assert str(cap) in message
        assert "one-too-many" in message

    @pytest.mark.asyncio
    async def test_no_declared_budget_leaves_the_kwarg_absent(
        self, mock_producer: AsyncMock
    ) -> None:
        """A config with no budget passes no aggregate cap at all.

        The library path (CLI relays, focused tests) constructs a handful of
        consumers and is not what OOM-killed the runtime. It gets aiokafka's own
        behaviour rather than a bound this repo invented for it. The runtime
        construction path is the one that refuses -- see
        ``test_select_event_bus_refuses_without_a_declared_budget``.
        """
        consumer_cls = _consumer_cls()
        with (
            patch(
                "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
                return_value=mock_producer,
            ),
            patch(
                "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer",
                consumer_cls,
            ),
        ):
            bus = EventBusKafka(
                config=ModelKafkaEventBusConfig(
                    bootstrap_servers=TEST_BOOTSTRAP_SERVERS
                ),
            )
            await bus._start_consumer_for_topic("events", "my-group")

        assert "fetch_max_bytes" not in consumer_cls.call_args.kwargs
        assert (
            consumer_cls.call_args.kwargs["max_partition_fetch_bytes"] == 1_048_588
        ), "OMN-16267 holds on the unbudgeted path too"


@pytest.mark.unit
class TestFetchBudgetResolution:
    """The budget refuses rather than assuming, on every unresolvable input."""

    def test_declared_bytes_requires_the_limit(self) -> None:
        with pytest.raises(ValueError, match="memory_limit_bytes is required"):
            ModelKafkaConsumerFetchBudget(
                source=EnumKafkaFetchBudgetSource.DECLARED_BYTES,
                memory_fraction=0.15,
                max_concurrent_consumers=512,
                brokers_per_consumer=1,
                in_flight_fetches_per_broker=2,
            )

    def test_cgroup_source_rejects_a_second_copy_of_the_limit(self) -> None:
        """A declared limit alongside the cgroup source is drift waiting to happen."""
        with pytest.raises(ValueError, match="must be omitted"):
            ModelKafkaConsumerFetchBudget(
                source=EnumKafkaFetchBudgetSource.CONTAINER_CGROUP_LIMIT,
                memory_limit_bytes=DEV_LANE_MEMORY_LIMIT_BYTES,
                memory_fraction=0.15,
                max_concurrent_consumers=512,
                brokers_per_consumer=1,
                in_flight_fetches_per_broker=2,
            )

    def test_partial_budget_is_a_validation_error(self) -> None:
        """No field silently defaults to a number."""
        with pytest.raises(ValueError):
            ModelKafkaConsumerFetchBudget(  # type: ignore[call-arg]
                source=EnumKafkaFetchBudgetSource.DECLARED_BYTES,
                memory_limit_bytes=DEV_LANE_MEMORY_LIMIT_BYTES,
                memory_fraction=0.15,
            )

    def test_unreadable_cgroup_limit_refuses_rather_than_assuming(
        self, tmp_path: object
    ) -> None:
        """An absent cgroup file is a refusal, never an assumed limit.

        This is the case that fires on the launching macOS host and on any
        unconstrained runner. Refusing here is what keeps a unit run from
        silently deriving a bound from a limit that does not exist.
        """
        budget = ModelKafkaConsumerFetchBudget(
            source=EnumKafkaFetchBudgetSource.CONTAINER_CGROUP_LIMIT,
            memory_fraction=0.15,
            max_concurrent_consumers=512,
            brokers_per_consumer=1,
            in_flight_fetches_per_broker=2,
        )
        with (
            patch(
                "omnibase_infra.event_bus.models.config."
                "model_kafka_consumer_fetch_budget.CGROUP_V2_MEMORY_MAX_PATH"
            ) as path_mock,
            pytest.raises(ProtocolConfigurationError, match="cannot read the cgroup"),
        ):
            path_mock.read_text.side_effect = FileNotFoundError("no such file")
            budget.resolve_memory_limit_bytes()

    def test_unconstrained_cgroup_refuses(self) -> None:
        """``max`` means unconstrained, which is not a limit to divide."""
        budget = ModelKafkaConsumerFetchBudget(
            source=EnumKafkaFetchBudgetSource.CONTAINER_CGROUP_LIMIT,
            memory_fraction=0.15,
            max_concurrent_consumers=512,
            brokers_per_consumer=1,
            in_flight_fetches_per_broker=2,
        )
        with (
            patch(
                "omnibase_infra.event_bus.models.config."
                "model_kafka_consumer_fetch_budget.CGROUP_V2_MEMORY_MAX_PATH"
            ) as path_mock,
            pytest.raises(ProtocolConfigurationError, match="unconstrained"),
        ):
            path_mock.read_text.return_value = "max\n"
            budget.resolve_memory_limit_bytes()

    def test_cgroup_limit_is_read_live(self) -> None:
        """The dev-lane limit read from the cgroup gives the dev-lane bound."""
        budget = ModelKafkaConsumerFetchBudget(
            source=EnumKafkaFetchBudgetSource.CONTAINER_CGROUP_LIMIT,
            memory_fraction=DEV_LANE_MEMORY_FRACTION,
            max_concurrent_consumers=DEV_LANE_MAX_CONSUMERS,
            brokers_per_consumer=1,
            in_flight_fetches_per_broker=DEV_LANE_IN_FLIGHT_PER_BROKER,
        )
        with patch(
            "omnibase_infra.event_bus.models.config."
            "model_kafka_consumer_fetch_budget.CGROUP_V2_MEMORY_MAX_PATH"
        ) as path_mock:
            path_mock.read_text.return_value = f"{DEV_LANE_MEMORY_LIMIT_BYTES}\n"
            assert budget.resolve_memory_limit_bytes() == DEV_LANE_MEMORY_LIMIT_BYTES
            assert budget.resolve_fetch_max_bytes() == 235_929

    def test_a_budget_that_divides_to_nothing_is_refused(self) -> None:
        """A bound under a Kafka record header is a stall dressed as a bound."""
        budget = ModelKafkaConsumerFetchBudget(
            source=EnumKafkaFetchBudgetSource.DECLARED_BYTES,
            memory_limit_bytes=DEV_LANE_MEMORY_LIMIT_BYTES,
            memory_fraction=0.001,
            max_concurrent_consumers=100_000,
            brokers_per_consumer=1,
            in_flight_fetches_per_broker=2,
        )
        with pytest.raises(ProtocolConfigurationError, match="below the"):
            budget.resolve_fetch_max_bytes()

    def test_from_declaration_refuses_when_undeclared(self) -> None:
        """No declared value, no assumed budget -- the point of the change."""
        with pytest.raises(ProtocolConfigurationError, match="is not set"):
            ModelKafkaConsumerFetchBudget.from_declaration(None)
        with pytest.raises(ProtocolConfigurationError, match="is not set"):
            ModelKafkaConsumerFetchBudget.from_declaration("   ")

    def test_from_declaration_round_trips_the_rendered_contract_value(self) -> None:
        """The exact string the renderer emits parses back to the budget.

        Kept byte-identical to ``docker/runtime-policy.env``'s
        ``ONEX_KAFKA_CONSUMER_FETCH_BUDGET_JSON`` so a contract edit that breaks
        the round trip fails here rather than at container start on the lane.
        """
        rendered = (
            '{"source":"container_cgroup_limit","memory_fraction":0.15,'
            '"max_concurrent_consumers":512,"brokers_per_consumer":1,'
            '"in_flight_fetches_per_broker":2}'
        )
        budget = ModelKafkaConsumerFetchBudget.from_declaration(rendered)
        assert budget.source is EnumKafkaFetchBudgetSource.CONTAINER_CGROUP_LIMIT
        assert budget.memory_limit_bytes is None
        assert budget.max_concurrent_consumers == 512
