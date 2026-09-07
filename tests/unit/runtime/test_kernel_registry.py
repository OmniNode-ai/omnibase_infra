# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for kernel registry auto-configuration (OMN-7076).

Tests that the kernel resolves the event bus from the registry based on
backend probes, not inline if/else creation.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from omnibase_infra.backends.auto_configure import (
    EventBusResolutionAmbiguousError,
    select_event_bus,
)
from omnibase_infra.backends.enum_probe_state import EnumProbeState
from omnibase_infra.backends.model_probe_result import ModelProbeResult
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.event_bus.models.config import ModelKafkaConsumerFetchBudget

pytestmark = pytest.mark.unit

# OMN-17888: select_event_bus refuses to construct an EventBusKafka without a
# declared aggregate consumer fetch budget. Every Kafka-resolving call below
# therefore passes one. DECLARED_BYTES with the .201 dev lane's real 1536 MiB
# limit, injected -- a unit test must never read the host's own cgroup.
DECLARED_FETCH_BUDGET = ModelKafkaConsumerFetchBudget.from_declaration(
    '{"source":"declared_bytes","memory_limit_bytes":1610612736,'
    '"memory_fraction":0.15,"max_concurrent_consumers":512,'
    '"brokers_per_consumer":1,"in_flight_fetches_per_broker":2}'
)


class TestKernelRegistryResolution:
    """Test that the kernel resolves event bus from registry."""

    def test_registry_resolves_inmemory_when_kafka_unavailable(self) -> None:
        """When Kafka is unreachable, kernel auto-falls back to in-memory bus."""
        kafka_probe_result = ModelProbeResult(
            state=EnumProbeState.DISCOVERED,
            reason="TCP connect to localhost:59999 failed",
            backend_label="event_bus_kafka",
        )
        with (
            patch(
                "omnibase_infra.backends.auto_configure.probe_kafka",
                return_value=kafka_probe_result,
            ),
            patch.dict("os.environ", {}, clear=False) as env,
        ):
            env.pop("ONEX_EVENT_BUS_TYPE", None)
            bus = select_event_bus(
                kafka_bootstrap_servers=None,
                environment="test",
                consumer_group="test-group",
            )
            assert type(bus).__name__ == "EventBusInmemory"

    def test_registry_resolves_kafka_when_healthy(self) -> None:
        """When Kafka is healthy, kernel selects EventBusKafka."""
        kafka_probe_result = ModelProbeResult(
            state=EnumProbeState.AUTHORITATIVE,
            reason="Kafka healthy with 5 topics, brokers match config",
            backend_label="event_bus_kafka",
        )
        with (
            patch(
                "omnibase_infra.backends.auto_configure.probe_kafka",
                return_value=kafka_probe_result,
            ),
            patch.dict("os.environ", {}, clear=False) as env,
        ):
            env.pop("ONEX_EVENT_BUS_TYPE", None)
            bus = select_event_bus(
                kafka_bootstrap_servers="localhost:9092",
                environment="test",
                consumer_group="test-group",
                consumer_fetch_budget=DECLARED_FETCH_BUDGET,
            )
            assert type(bus).__name__ == "EventBusKafka"

    def test_registry_applies_kafka_environment_overrides(self) -> None:
        """Explicit Kafka config still honors KAFKA_* runtime env overrides."""
        kafka_probe_result = ModelProbeResult(
            state=EnumProbeState.AUTHORITATIVE,
            reason="Kafka healthy with 5 topics, brokers match config",
            backend_label="event_bus_kafka",
        )
        with (
            patch(
                "omnibase_infra.backends.auto_configure.probe_kafka",
                return_value=kafka_probe_result,
            ),
            patch.dict(
                "os.environ",
                {
                    "KAFKA_INSTANCE_ID": "runtime-effects",
                    "ONEX_EVENT_BUS_TYPE": "",
                },
            ),
        ):
            bus = select_event_bus(
                kafka_bootstrap_servers="localhost:9092",
                environment="test",
                consumer_group="test-group",
                consumer_fetch_budget=DECLARED_FETCH_BUDGET,
            )
            assert type(bus).__name__ == "EventBusKafka"
            assert bus.config.instance_id == "runtime-effects"

    def test_env_var_is_ignored_at_the_construction_seam(self) -> None:
        """ONEX_EVENT_BUS_TYPE no longer forces anything (OMN-17304).

        Pre-ruling this test pinned env=inmemory beating an AUTHORITATIVE
        probe. The var holds no tier now: with no explicit ``bus_type`` and no
        config, the probe decides — the set value is warned about and ignored.
        """
        kafka_probe_result = ModelProbeResult(
            state=EnumProbeState.AUTHORITATIVE,
            reason="Kafka healthy with 5 topics, brokers match config",
            backend_label="event_bus_kafka",
        )
        with (
            patch(
                "omnibase_infra.backends.auto_configure.probe_kafka",
                return_value=kafka_probe_result,
            ),
            patch.dict("os.environ", {"ONEX_EVENT_BUS_TYPE": "inmemory"}),
        ):
            bus = select_event_bus(
                kafka_bootstrap_servers="localhost:9092",
                environment="test",
                consumer_group="test-group",
                consumer_fetch_budget=DECLARED_FETCH_BUDGET,
            )
            assert type(bus).__name__ == "EventBusKafka"

    def test_reachable_with_explicit_servers_refuses_to_guess(self) -> None:
        """OMN-16678: REACHABLE is indeterminate — the kernel path must not guess.

        This test previously asserted the opposite ("still try Kafka"). That
        mapping was one half of a contradiction: the delegate path resolved the
        IDENTICAL probe state to in-memory. Since ``probe_kafka`` degrades any
        Stage-2 metadata failure — a 2s ``list_topics`` timeout against a
        perfectly healthy broker included — to REACHABLE, whichever branch ran
        was decided by transient network timing, not by configuration
        (measured 14 kafka / 6 inmemory over 20 unchanged-env calls,
        ``knowledge-base#59``). Both paths now refuse and name the ambiguity;
        an operator who wants the old "try Kafka anyway" behavior states it
        explicitly (``bus_type="kafka"`` / ``--bus kafka``) or declares it in
        the runtime config (OMN-17304 — the env var holds no tier).
        """
        kafka_probe_result = ModelProbeResult(
            state=EnumProbeState.REACHABLE,
            reason="TCP reachable but topic list failed",
            backend_label="event_bus_kafka",
        )
        with (
            patch(
                "omnibase_infra.backends.auto_configure.probe_kafka",
                return_value=kafka_probe_result,
            ),
            patch.dict("os.environ", {}, clear=False) as env,
        ):
            env.pop("ONEX_EVENT_BUS_TYPE", None)
            with pytest.raises(EventBusResolutionAmbiguousError) as excinfo:
                select_event_bus(
                    kafka_bootstrap_servers="localhost:9092",
                    environment="test",
                    consumer_group="test-group",
                )
            assert "REACHABLE" in str(excinfo.value)

    def test_reachable_is_resolvable_by_the_documented_remedy(self) -> None:
        """The remedy the error message names actually works (OMN-17304).

        The documented remedies are the explicit argument and the declared
        config; the explicit argument is the one reachable through
        ``select_event_bus``'s own signature.
        """
        kafka_probe_result = ModelProbeResult(
            state=EnumProbeState.REACHABLE,
            reason="TCP reachable but topic list failed",
            backend_label="event_bus_kafka",
        )
        with (
            patch(
                "omnibase_infra.backends.auto_configure.probe_kafka",
                return_value=kafka_probe_result,
            ),
            patch.dict("os.environ", {}, clear=False) as env,
        ):
            env.pop("ONEX_EVENT_BUS_TYPE", None)
            bus = select_event_bus(
                bus_type="kafka",
                kafka_bootstrap_servers="localhost:9092",
                environment="test",
                consumer_group="test-group",
                consumer_fetch_budget=DECLARED_FETCH_BUDGET,
            )
            assert type(bus).__name__ == "EventBusKafka"

    def test_explicit_bus_type_argument_outranks_the_probe(self) -> None:
        """Tier 1 of the shared order is reachable from the in-process caller too."""
        kafka_probe_result = ModelProbeResult(
            state=EnumProbeState.AUTHORITATIVE,
            reason="Kafka healthy with 5 topics, brokers match config",
            backend_label="event_bus_kafka",
        )
        with (
            patch(
                "omnibase_infra.backends.auto_configure.probe_kafka",
                return_value=kafka_probe_result,
            ),
            patch.dict("os.environ", {"ONEX_EVENT_BUS_TYPE": "kafka"}),
        ):
            bus = select_event_bus(
                bus_type="inmemory",
                kafka_bootstrap_servers="localhost:9092",
                environment="test",
                consumer_group="test-group",
            )
            assert type(bus).__name__ == "EventBusInmemory"


@pytest.mark.unit
class TestKafkaConsumerFetchBudgetIsRequiredAtTheConstructionSeam:
    """OMN-17888: the Kafka construction seam refuses an undeclared budget."""

    def test_select_event_bus_refuses_without_a_declared_budget(self) -> None:
        """No budget, no Kafka bus -- and the error names what to declare.

        This is the seam the OOM happened behind: the process that subscribes
        to hundreds of topics at once. Refusing here is what makes the bound a
        requirement rather than a suggestion.
        """
        kafka_probe_result = ModelProbeResult(
            state=EnumProbeState.AUTHORITATIVE,
            reason="Kafka healthy with 5 topics, brokers match config",
            backend_label="event_bus_kafka",
        )
        with (
            patch(
                "omnibase_infra.backends.auto_configure.probe_kafka",
                return_value=kafka_probe_result,
            ),
            pytest.raises(ProtocolConfigurationError) as excinfo,
        ):
            select_event_bus(
                bus_type="kafka",
                kafka_bootstrap_servers="localhost:9092",
                environment="test",
                consumer_group="test-group",
            )
        message = str(excinfo.value)
        assert "consumer_fetch_budget" in message
        assert "ONEX_KAFKA_CONSUMER_FETCH_BUDGET_JSON" in message

    def test_inmemory_bus_needs_no_budget(self) -> None:
        """The in-memory bus holds no fetch buffers, so it is not gated.

        Requiring a container memory limit for local, brokerless development
        would gate the cheap path on a fact only the deployed path has.
        """
        kafka_probe_result = ModelProbeResult(
            state=EnumProbeState.DISCOVERED,
            reason="TCP connect failed",
            backend_label="event_bus_kafka",
        )
        with patch(
            "omnibase_infra.backends.auto_configure.probe_kafka",
            return_value=kafka_probe_result,
        ):
            bus = select_event_bus(
                bus_type="inmemory",
                kafka_bootstrap_servers=None,
                environment="test",
                consumer_group="test-group",
            )
        assert type(bus).__name__ == "EventBusInmemory"

    def test_declared_budget_reaches_the_constructed_bus(self) -> None:
        """The declared budget is carried onto the bus config, not dropped."""
        kafka_probe_result = ModelProbeResult(
            state=EnumProbeState.AUTHORITATIVE,
            reason="Kafka healthy with 5 topics, brokers match config",
            backend_label="event_bus_kafka",
        )
        with patch(
            "omnibase_infra.backends.auto_configure.probe_kafka",
            return_value=kafka_probe_result,
        ):
            bus = select_event_bus(
                bus_type="kafka",
                kafka_bootstrap_servers="localhost:9092",
                environment="test",
                consumer_group="test-group",
                consumer_fetch_budget=DECLARED_FETCH_BUDGET,
            )
        budget = bus.config.consumer_fetch_budget
        assert budget is not None
        assert budget.max_concurrent_consumers == 512
        assert budget.resolve_fetch_max_bytes() == 235_929
