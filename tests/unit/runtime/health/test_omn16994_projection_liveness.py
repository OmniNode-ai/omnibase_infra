# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Projection liveness folds into the runtime health verdict (OMN-16994).

RED-first reproduction of the silent-black-hole class documented in
``docs/tracking/2026-08-29-hook-emission-ledger-trace.md`` hop 6 and deferred
out of OMN-16843 as its AC6.

Two distinct masks, both live-confirmed on ``.201`` on 2026-08-29:

1. **Unattached.** A contract declaring a projection (``db_io.db_tables``)
   fails to prepare its handler, so it never subscribes. ``ServiceRuntimeHealthMonitor`` then reads its expected
   consumer groups from the LIVE bus registry, which by construction contains
   only the subscriptions that DID attach — so the missing projection drops out
   of the expectation set and ``topic_coverage`` reports "All N expected
   consumer group(s) covered". Nineteen contracts were unattached on every
   compose lane for months under exactly this reading (OMN-16843).

2. **Fully DLQing.** ``node_projection_session_replay`` attached at zero lag,
   consumed every event and routed 100% of them to the quarantine sink on a
   Postgres auth failure. Offsets commit on the DLQ route, so lag reads 0 and
   ``/health`` returned ``status: "healthy"`` with ``failed_handlers: {}``.

Related Tickets:
    - OMN-16994: this ticket (OMN-16843 AC6, deferred)
    - OMN-16843: compose-lane internal DSN wiring, whose AC6 this picks up
    - OMN-16690: the read-under-write-declaration class that produced mask 2
    - OMN-16777: the consumer-flow counters this reads the DLQ ratio from
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from omnibase_core.models.contracts.subcontracts.model_db_ownership_subcontract import (
    ModelDbOwnershipSubcontract,
)
from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_infra.models.observability import (
    ModelConsumerFlowDelta,
    ModelNodeFlowWindow,
)
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
)
from omnibase_infra.runtime.health.projection_liveness import (
    DLQ_SATURATION_MIN_MESSAGES,
    describe_dlq_saturation,
    describe_projection_attachment,
    evaluate_projection_liveness,
    select_kernel_nonwriting_projections,
    select_projection_contracts,
)
from omnibase_infra.runtime.observability import (
    get_consumer_flow_counters,
    reset_consumer_flow_counters,
)
from omnibase_infra.services.service_runtime_health_monitor import (
    ConsumerGroupSnapshot,
    ServiceRuntimeHealthMonitor,
)

PROJECTION_TOPIC = "onex.evt.omniclaude.tool-executed.v1"
ORCHESTRATOR_TOPIC = "onex.cmd.omnimarket.contract-sweep-start.v1"


def _table(name: str = "session_replay") -> ModelDbTableDeclaration:
    return ModelDbTableDeclaration(
        name=name,
        database_ref="application",
        schema="omninode_internal",
        migration="0001_init.sql",
        access="read_write",
        role="projection_target",
    )


def _projection_contract(
    *,
    name: str = "node_projection_session_replay",
    topic: str = PROJECTION_TOPIC,
    package_name: str = "omnimarket",
    consumer_purpose: str | None = None,
    plugin_managed: bool = False,
    requires_cloud_gateway: bool = False,
    with_db_io: bool = True,
) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="REDUCER",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=__file__,
        entry_point_name=name,
        package_name=package_name,
        requires_cloud_gateway=requires_cloud_gateway,
        event_bus=ModelEventBusWiring(
            subscribe_topics=(topic,),
            publish_topics=(),
            consumer_purpose=consumer_purpose,
            plugin_managed=plugin_managed,
        ),
        db_io=(
            ModelDbOwnershipSubcontract(db_tables=[_table()]) if with_db_io else None
        ),
    )


def _non_projection_contract(
    *, name: str = "node_contract_sweep", topic: str = ORCHESTRATOR_TOPIC
) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="ORCHESTRATOR",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=__file__,
        entry_point_name=name,
        package_name="omnimarket",
        event_bus=ModelEventBusWiring(subscribe_topics=(topic,), publish_topics=()),
    )


def _window(
    *,
    topic: str,
    messages_in: int,
    messages_dlq: int,
    sequence: int = 1,
    consumer_group: str = "runtime.projection.consume.v1",
) -> ModelNodeFlowWindow:
    node_id = uuid4()
    start = datetime(2026, 8, 29, 12, 0, tzinfo=UTC) + timedelta(seconds=30 * sequence)
    end = start + timedelta(seconds=30)
    return ModelNodeFlowWindow(
        node_id=node_id,
        window_start=start,
        window_end=end,
        window_sequence=sequence,
        consumer_deltas=(
            ModelConsumerFlowDelta(
                consumer_group=consumer_group,
                topic=topic,
                node_id=node_id,
                window_start=start,
                window_end=end,
                window_sequence=sequence,
                messages_in=messages_in,
                messages_out=0,
                messages_dlq=messages_dlq,
                handler_errors=messages_dlq,
            ),
        ),
    )


# =============================================================================
# select_projection_contracts
# =============================================================================


@pytest.mark.unit
class TestSelectProjectionContracts:
    """Which contracts are in scope. Over-selection would red the whole fleet."""

    def test_db_io_tables_make_a_contract_a_projection(self) -> None:
        manifest = ModelAutoWiringManifest(
            contracts=(_projection_contract(),), errors=()
        )
        selected = select_projection_contracts(manifest)
        assert [p.name for p in selected] == ["node_projection_session_replay"]
        assert selected[0].subscribe_topics == (PROJECTION_TOPIC,)

    def test_raw_event_consumer_purpose_alone_is_out_of_scope(self) -> None:
        """``consumer_purpose: audit|projection`` without ``db_io`` is excluded.

        ``handler_wiring`` skips the Kafka subscription for these outright
        unless the kernel registered a result applier for that exact contract
        name (``_raw_event_projection_enabled``), and the applier registry is
        not visible from a health cycle. Selecting them would report a
        permanent, fleet-wide, false outage — the fastest way to get a real
        health signal switched off.
        """
        manifest = ModelAutoWiringManifest(
            contracts=(
                _projection_contract(
                    name="node_ledger_projection_compute",
                    consumer_purpose="audit",
                    with_db_io=False,
                ),
            ),
            errors=(),
        )
        assert select_projection_contracts(manifest) == ()

    def test_db_io_projection_with_raw_purpose_is_still_in_scope(self) -> None:
        """``db_io.db_tables`` is the discriminator; purpose does not veto it."""
        manifest = ModelAutoWiringManifest(
            contracts=(_projection_contract(consumer_purpose="projection"),), errors=()
        )
        assert [p.name for p in select_projection_contracts(manifest)] == [
            "node_projection_session_replay"
        ]

    def test_plain_orchestrator_is_not_a_projection(self) -> None:
        manifest = ModelAutoWiringManifest(
            contracts=(_non_projection_contract(),), errors=()
        )
        assert select_projection_contracts(manifest) == ()

    def test_plugin_managed_projection_is_out_of_scope(self) -> None:
        """The domain plugin owns the subscription; the bus registry never sees it."""
        manifest = ModelAutoWiringManifest(
            contracts=(_projection_contract(plugin_managed=True),), errors=()
        )
        assert select_projection_contracts(manifest) == ()

    def test_cloud_gateway_projection_is_out_of_scope(self) -> None:
        """Deliberately unwired on lanes with no cloud mirroring (OMN-13809)."""
        manifest = ModelAutoWiringManifest(
            contracts=(_projection_contract(requires_cloud_gateway=True),), errors=()
        )
        assert select_projection_contracts(manifest) == ()

    def test_a_kernel_nonwriting_contract_leaves_select_projection_contracts_scope(
        self,
    ) -> None:
        """The fourth documented exclusion (OMN-17562).

        Once the kernel stops SUBSCRIBING a projection it will never dispatch,
        that projection's topics correctly leave the live bus registry. This
        selector is manifest-derived while ``attached_topics`` is the live
        registry, so leaving the contract in scope would simply trade
        ``projection_write_path`` DEGRADED for ``projection_attachment``
        DEGRADED on every one of them — a fleet-wide false outage reported by a
        different dimension.
        """
        manifest = ModelAutoWiringManifest(
            contracts=(_projection_contract(),), errors=()
        )
        assert (
            select_projection_contracts(
                manifest,
                kernel_nonwriting=frozenset({"node_projection_session_replay"}),
            )
            == ()
        )

    def test_the_excluded_half_is_still_selectable_by_name(self) -> None:
        """Excluded is not invisible: the write-path leg still names them.

        The exclusion removes them from the ATTACHMENT scope, not from the
        health surface. An operator must still be able to read which
        projections this kernel deliberately does not serve.
        """
        manifest = ModelAutoWiringManifest(
            contracts=(_projection_contract(),), errors=()
        )
        selected = select_kernel_nonwriting_projections(
            manifest, frozenset({"node_projection_session_replay"})
        )
        assert [p.name for p in selected] == ["node_projection_session_replay"]
        assert selected[0].subscribe_topics == (PROJECTION_TOPIC,)

    def test_a_name_that_is_not_a_declared_projection_is_never_selected(self) -> None:
        """A stale or foreign ledger entry cannot manufacture an unlookupable name.

        The narrowing that used to live inside ``evaluate_projection_liveness``
        (``in_scope & dispatch_skipped``) moves here, to the point where a name
        is resolved to a contract.
        """
        manifest = ModelAutoWiringManifest(
            contracts=(_projection_contract(),), errors=()
        )
        assert (
            select_kernel_nonwriting_projections(
                manifest, frozenset({"node_projection_from_another_process"})
            )
            == ()
        )

    def test_a_non_projection_contract_is_never_selected_as_nonwriting(self) -> None:
        """The two selectors share one discriminator; neither can drift."""
        manifest = ModelAutoWiringManifest(
            contracts=(_non_projection_contract(),), errors=()
        )
        assert (
            select_kernel_nonwriting_projections(
                manifest, frozenset({"node_contract_sweep"})
            )
            == ()
        )


# =============================================================================
# evaluate_projection_liveness
# =============================================================================


@pytest.mark.unit
class TestEvaluateProjectionLiveness:
    """The pure verdict. No clock, no bus, no database."""

    def test_unattached_projection_is_named(self) -> None:
        projections = select_projection_contracts(
            ModelAutoWiringManifest(contracts=(_projection_contract(),), errors=())
        )
        verdict = evaluate_projection_liveness(
            projections=projections,
            attached_topics=frozenset({ORCHESTRATOR_TOPIC}),
            flow_windows=(),
        )
        assert verdict.attachment_evaluated is True
        assert verdict.unattached_projections == ("node_projection_session_replay",)

    def test_attached_projection_is_not_named(self) -> None:
        projections = select_projection_contracts(
            ModelAutoWiringManifest(contracts=(_projection_contract(),), errors=())
        )
        verdict = evaluate_projection_liveness(
            projections=projections,
            attached_topics=frozenset({PROJECTION_TOPIC}),
            flow_windows=(),
        )
        assert verdict.unattached_projections == ()

    def test_empty_attached_topics_is_unknown_not_a_failure(self) -> None:
        """An empty registry means "we cannot tell", never "nothing attached"."""
        projections = select_projection_contracts(
            ModelAutoWiringManifest(contracts=(_projection_contract(),), errors=())
        )
        verdict = evaluate_projection_liveness(
            projections=projections,
            attached_topics=frozenset(),
            flow_windows=(),
        )
        assert verdict.attachment_evaluated is False
        assert verdict.unattached_projections == ()

    def test_fully_dlqing_projection_is_named(self) -> None:
        projections = select_projection_contracts(
            ModelAutoWiringManifest(contracts=(_projection_contract(),), errors=())
        )
        windows = (
            _window(
                topic=PROJECTION_TOPIC,
                messages_in=DLQ_SATURATION_MIN_MESSAGES,
                messages_dlq=DLQ_SATURATION_MIN_MESSAGES,
            ),
        )
        verdict = evaluate_projection_liveness(
            projections=projections,
            attached_topics=frozenset({PROJECTION_TOPIC}),
            flow_windows=windows,
        )
        assert verdict.saturation_evaluated is True
        assert verdict.dlq_saturated_projections == ("node_projection_session_replay",)

    def test_partial_dlq_is_not_saturation(self) -> None:
        projections = select_projection_contracts(
            ModelAutoWiringManifest(contracts=(_projection_contract(),), errors=())
        )
        windows = (
            _window(
                topic=PROJECTION_TOPIC,
                messages_in=DLQ_SATURATION_MIN_MESSAGES * 2,
                messages_dlq=DLQ_SATURATION_MIN_MESSAGES,
            ),
        )
        verdict = evaluate_projection_liveness(
            projections=projections,
            attached_topics=frozenset({PROJECTION_TOPIC}),
            flow_windows=windows,
        )
        assert verdict.dlq_saturated_projections == ()

    def test_single_poison_pill_below_the_floor_is_not_saturation(self) -> None:
        """One malformed event must not flip a runtime for a whole cycle."""
        projections = select_projection_contracts(
            ModelAutoWiringManifest(contracts=(_projection_contract(),), errors=())
        )
        windows = (_window(topic=PROJECTION_TOPIC, messages_in=1, messages_dlq=1),)
        verdict = evaluate_projection_liveness(
            projections=projections,
            attached_topics=frozenset({PROJECTION_TOPIC}),
            flow_windows=windows,
        )
        assert verdict.dlq_saturated_projections == ()

    def test_saturation_accumulates_across_the_observation_window(self) -> None:
        """The floor is met by the window set, not by any single 30 s tick."""
        projections = select_projection_contracts(
            ModelAutoWiringManifest(contracts=(_projection_contract(),), errors=())
        )
        per_window = max(1, DLQ_SATURATION_MIN_MESSAGES // 2)
        windows = tuple(
            _window(
                topic=PROJECTION_TOPIC,
                messages_in=per_window,
                messages_dlq=per_window,
                sequence=index + 1,
            )
            for index in range(3)
        )
        verdict = evaluate_projection_liveness(
            projections=projections,
            attached_topics=frozenset({PROJECTION_TOPIC}),
            flow_windows=windows,
        )
        assert verdict.dlq_saturated_projections == ("node_projection_session_replay",)
        assert verdict.observed_window_count == 3

    def test_idle_projection_is_not_saturated(self) -> None:
        """Zero in, zero DLQ is observed-idle, which is a fact, not a failure."""
        projections = select_projection_contracts(
            ModelAutoWiringManifest(contracts=(_projection_contract(),), errors=())
        )
        windows = (_window(topic=PROJECTION_TOPIC, messages_in=0, messages_dlq=0),)
        verdict = evaluate_projection_liveness(
            projections=projections,
            attached_topics=frozenset({PROJECTION_TOPIC}),
            flow_windows=windows,
        )
        assert verdict.dlq_saturated_projections == ()

    def test_no_windows_is_unknown_not_a_failure(self) -> None:
        projections = select_projection_contracts(
            ModelAutoWiringManifest(contracts=(_projection_contract(),), errors=())
        )
        verdict = evaluate_projection_liveness(
            projections=projections,
            attached_topics=frozenset({PROJECTION_TOPIC}),
            flow_windows=(),
        )
        assert verdict.saturation_evaluated is False

    def test_non_projection_topic_saturation_is_out_of_scope(self) -> None:
        projections = select_projection_contracts(
            ModelAutoWiringManifest(contracts=(_projection_contract(),), errors=())
        )
        windows = (
            _window(
                topic=ORCHESTRATOR_TOPIC,
                messages_in=DLQ_SATURATION_MIN_MESSAGES,
                messages_dlq=DLQ_SATURATION_MIN_MESSAGES,
            ),
        )
        verdict = evaluate_projection_liveness(
            projections=projections,
            attached_topics=frozenset({PROJECTION_TOPIC}),
            flow_windows=windows,
        )
        assert verdict.dlq_saturated_projections == ()


@pytest.mark.unit
class TestDimensionDetails:
    """Details are the durable record — a bare count forces a log dig."""

    def test_attachment_detail_names_the_projection(self) -> None:
        verdict = evaluate_projection_liveness(
            projections=select_projection_contracts(
                ModelAutoWiringManifest(contracts=(_projection_contract(),), errors=())
            ),
            attached_topics=frozenset({ORCHESTRATOR_TOPIC}),
            flow_windows=(),
        )
        assert "node_projection_session_replay" in describe_projection_attachment(
            verdict
        )

    def test_saturation_detail_names_the_projection(self) -> None:
        verdict = evaluate_projection_liveness(
            projections=select_projection_contracts(
                ModelAutoWiringManifest(contracts=(_projection_contract(),), errors=())
            ),
            attached_topics=frozenset({PROJECTION_TOPIC}),
            flow_windows=(
                _window(
                    topic=PROJECTION_TOPIC,
                    messages_in=DLQ_SATURATION_MIN_MESSAGES,
                    messages_dlq=DLQ_SATURATION_MIN_MESSAGES,
                ),
            ),
        )
        assert "node_projection_session_replay" in describe_dlq_saturation(verdict)


# =============================================================================
# Retained flow windows on the process-scoped accumulator
# =============================================================================


@pytest.mark.unit
class TestRetainedFlowWindows:
    """The health monitor and the heartbeat tick run on different clocks.

    ``drain()`` resets the counters, so a monitor reading them live would race
    the heartbeat and usually see a fraction of a window. Retaining the CLOSED
    windows is what makes the ratio readable off-cycle.
    """

    def setup_method(self) -> None:
        reset_consumer_flow_counters()

    def teardown_method(self) -> None:
        reset_consumer_flow_counters()

    def test_drained_windows_are_retained(self) -> None:
        counters = get_consumer_flow_counters()
        node_id = uuid4()
        base = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)
        counters.record_in("group-a", PROJECTION_TOPIC, 5)
        assert counters.drain(node_id=node_id, now=base) is None  # priming
        counters.record_in("group-a", PROJECTION_TOPIC, 5)
        counters.record_dlq("group-a", PROJECTION_TOPIC, 5)
        counters.drain(node_id=node_id, now=base + timedelta(seconds=30))

        retained = counters.retained_windows.snapshot()
        assert len(retained) == 1
        delta = retained[0].consumer_deltas[0]
        assert (delta.messages_in, delta.messages_dlq) == (5, 5)

    def test_retention_is_bounded(self) -> None:
        from omnibase_infra.runtime.observability.consumer_flow_counters import (
            RETAINED_FLOW_WINDOW_COUNT,
        )

        counters = get_consumer_flow_counters()
        node_id = uuid4()
        base = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)
        counters.drain(node_id=node_id, now=base)
        for index in range(RETAINED_FLOW_WINDOW_COUNT + 5):
            counters.drain(
                node_id=node_id, now=base + timedelta(seconds=30 * (index + 1))
            )

        assert len(counters.retained_windows.snapshot()) == RETAINED_FLOW_WINDOW_COUNT

    def test_non_carrier_drain_retains_nothing(self) -> None:
        counters = get_consumer_flow_counters()
        carrier = uuid4()
        base = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)
        counters.drain(node_id=carrier, now=base)
        counters.drain(node_id=uuid4(), now=base + timedelta(seconds=30))
        assert counters.retained_windows.snapshot() == ()


# =============================================================================
# ServiceRuntimeHealthMonitor — the surface the trace read green
# =============================================================================


@pytest.mark.unit
class TestMonitorFoldsProjectionLiveness:
    """`/health` must stop reporting healthy over a dead projection."""

    def setup_method(self) -> None:
        reset_consumer_flow_counters()

    def teardown_method(self) -> None:
        reset_consumer_flow_counters()

    @staticmethod
    def _snapshots(group_ids: list[str]) -> list[ConsumerGroupSnapshot]:
        return [ConsumerGroupSnapshot(group_id=g, state="STABLE") for g in group_ids]

    @pytest.mark.asyncio
    async def test_unattached_projection_degrades_the_verdict(self) -> None:
        """The OMN-16843 mask: 19 unattached projections, verdict HEALTHY.

        The live bus registry contains ONLY the orchestrator's subscription.
        Before OMN-16994 the monitor replaced its manifest-derived expectation
        set with that registry, so the missing projection was not merely
        un-flagged — it was no longer expected at all.
        """
        manifest = ModelAutoWiringManifest(
            contracts=(_projection_contract(), _non_projection_contract()),
            errors=(),
        )
        live_group = f"runtime.sweep.consume.v1.__t.{ORCHESTRATOR_TOPIC}"
        bus = MagicMock(spec=ProtocolEventBusLike)
        bus.get_consumer_groups.return_value = {
            (ORCHESTRATOR_TOPIC, "runtime.sweep.consume.v1"): live_group
        }
        monitor = ServiceRuntimeHealthMonitor(
            event_bus=bus, bootstrap_servers="localhost:9092"
        )

        with (
            patch(
                "omnibase_infra.services.service_runtime_health_monitor._discover_contracts",
                return_value=manifest,
            ),
            patch(
                "omnibase_infra.services.service_runtime_health_monitor._list_consumer_group_snapshots",
                return_value=self._snapshots([live_group]),
            ),
        ):
            event = await monitor.run_once()

        assert event.status == "DEGRADED"
        assert event.unattached_projection_count == 1
        dimension = next(
            d for d in event.dimensions if d.name == "projection_attachment"
        )
        assert dimension.status == "DEGRADED"
        assert "node_projection_session_replay" in dimension.detail

    @pytest.mark.asyncio
    async def test_fully_dlqing_projection_degrades_the_verdict(self) -> None:
        """The hop-6 mask: attached, lag 0, 100% quarantined, verdict HEALTHY."""
        manifest = ModelAutoWiringManifest(
            contracts=(_projection_contract(),), errors=()
        )
        live_group = f"runtime.projection.consume.v1.__t.{PROJECTION_TOPIC}"
        bus = MagicMock(spec=ProtocolEventBusLike)
        bus.get_consumer_groups.return_value = {
            (PROJECTION_TOPIC, "runtime.projection.consume.v1"): live_group
        }
        counters = get_consumer_flow_counters()
        node_id = uuid4()
        base = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)
        counters.drain(node_id=node_id, now=base)
        counters.record_in(
            "runtime.projection.consume.v1",
            PROJECTION_TOPIC,
            DLQ_SATURATION_MIN_MESSAGES,
        )
        counters.record_dlq(
            "runtime.projection.consume.v1",
            PROJECTION_TOPIC,
            DLQ_SATURATION_MIN_MESSAGES,
        )
        counters.drain(node_id=node_id, now=base + timedelta(seconds=30))

        monitor = ServiceRuntimeHealthMonitor(
            event_bus=bus, bootstrap_servers="localhost:9092"
        )
        with (
            patch(
                "omnibase_infra.services.service_runtime_health_monitor._discover_contracts",
                return_value=manifest,
            ),
            patch(
                "omnibase_infra.services.service_runtime_health_monitor._list_consumer_group_snapshots",
                return_value=self._snapshots([live_group]),
            ),
        ):
            event = await monitor.run_once()

        assert event.status == "DEGRADED"
        assert event.dlq_saturated_projection_count == 1
        dimension = next(
            d for d in event.dimensions if d.name == "projection_dlq_saturation"
        )
        assert dimension.status == "DEGRADED"
        assert "node_projection_session_replay" in dimension.detail

    @pytest.mark.asyncio
    async def test_attached_and_flowing_projection_stays_healthy(self) -> None:
        manifest = ModelAutoWiringManifest(
            contracts=(_projection_contract(),), errors=()
        )
        live_group = f"runtime.projection.consume.v1.__t.{PROJECTION_TOPIC}"
        bus = MagicMock(spec=ProtocolEventBusLike)
        bus.get_consumer_groups.return_value = {
            (PROJECTION_TOPIC, "runtime.projection.consume.v1"): live_group
        }
        counters = get_consumer_flow_counters()
        node_id = uuid4()
        base = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)
        counters.drain(node_id=node_id, now=base)
        counters.record_in(
            "runtime.projection.consume.v1",
            PROJECTION_TOPIC,
            DLQ_SATURATION_MIN_MESSAGES,
        )
        counters.drain(node_id=node_id, now=base + timedelta(seconds=30))

        monitor = ServiceRuntimeHealthMonitor(
            event_bus=bus, bootstrap_servers="localhost:9092"
        )
        with (
            patch(
                "omnibase_infra.services.service_runtime_health_monitor._discover_contracts",
                return_value=manifest,
            ),
            patch(
                "omnibase_infra.services.service_runtime_health_monitor._list_consumer_group_snapshots",
                return_value=self._snapshots([live_group]),
            ),
        ):
            event = await monitor.run_once()

        assert event.status == "HEALTHY"
        assert event.unattached_projection_count == 0
        assert event.dlq_saturated_projection_count == 0


# =============================================================================
# End-to-end: induce the real failure, read the real endpoint
# =============================================================================


@pytest.mark.unit
class TestInducedCredentialFailureFlipsTheEndpoint:
    """The OMN-16994 done-proof, driven through the production seams.

    Nothing here hand-feeds a counter. The failure is induced by calling the
    same ``_route_projection_error_to_dlq`` arm the projection dispatch callback
    calls when a handler raises — which is precisely what a Postgres
    ``PermissionError``/auth failure does — under the same
    ``active_flow_key`` binding the auto-wiring boundary establishes. The
    counters, the window drain, the monitor cycle and the HTTP payload are all
    the real ones.
    """

    def setup_method(self) -> None:
        reset_consumer_flow_counters()

    def teardown_method(self) -> None:
        reset_consumer_flow_counters()

    @pytest.mark.asyncio
    async def test_health_endpoint_flips_and_names_the_projection(self) -> None:
        import json
        from unittest.mock import AsyncMock

        from aiohttp import web

        from omnibase_infra.runtime.auto_wiring.handler_wiring import (
            _route_projection_error_to_dlq,
        )
        from omnibase_infra.runtime.health.runtime_health_block import (
            RUNTIME_HEALTH_DETAIL_KEY,
        )
        from omnibase_infra.runtime.observability import active_flow_key
        from omnibase_infra.services.health_checker import ServiceHealth

        consumer_group = "runtime.projection.consume.v1"
        counters = get_consumer_flow_counters()
        node_id = uuid4()
        base = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)
        counters.drain(node_id=node_id, now=base)  # priming

        dlq_bus = MagicMock(spec=ProtocolEventBusLike)
        dlq_bus.publish = AsyncMock(return_value=None)
        for _ in range(DLQ_SATURATION_MIN_MESSAGES):
            counters.record_in(consumer_group, PROJECTION_TOPIC)
            with active_flow_key(consumer_group, PROJECTION_TOPIC):
                routed = await _route_projection_error_to_dlq(
                    dlq_bus,
                    [],
                    {"payload": {"session_id": "s-1"}},
                    "HandlerProjectionSessionReplay",
                    "PermissionError: password authentication failed for user "
                    '"omninode_runtime"',
                )
            assert routed is True
        counters.drain(node_id=node_id, now=base + timedelta(seconds=30))

        manifest = ModelAutoWiringManifest(
            contracts=(_projection_contract(),), errors=()
        )
        live_group = f"{consumer_group}.__t.{PROJECTION_TOPIC}"
        bus = MagicMock(spec=ProtocolEventBusLike)
        bus.get_consumer_groups.return_value = {
            (PROJECTION_TOPIC, consumer_group): live_group
        }
        monitor = ServiceRuntimeHealthMonitor(
            event_bus=bus, bootstrap_servers="localhost:9092"
        )
        with (
            patch(
                "omnibase_infra.services.service_runtime_health_monitor._discover_contracts",
                return_value=manifest,
            ),
            patch(
                "omnibase_infra.services.service_runtime_health_monitor._list_consumer_group_snapshots",
                return_value=[
                    ConsumerGroupSnapshot(group_id=live_group, state="STABLE")
                ],
            ),
        ):
            await monitor.run_once()

        # The process itself is perfectly alive — this is exactly what
        # RuntimeHostProcess.health_check() reported on the stability lane while
        # the projection quarantined every event it took.
        runtime = MagicMock()
        runtime.health_check = AsyncMock(
            return_value={
                "healthy": True,
                "degraded": False,
                "is_running": True,
                "runtime_attached": True,
                "event_bus_healthy": True,
                "failed_handlers": {},
                "skipped_handlers": {},
            }
        )
        server = ServiceHealth(runtime=runtime, version="0.0.0")
        server.set_runtime_health_provider(lambda: monitor.latest_event)

        response = await server._handle_health(MagicMock(spec=web.Request))
        assert response.text is not None
        body = json.loads(response.text)

        assert body["status"] != "healthy"
        block = body["details"][RUNTIME_HEALTH_DETAIL_KEY]
        assert block["dlq_saturated_projection_count"] == 1
        saturation = next(
            d for d in block["dimensions"] if d["name"] == "projection_dlq_saturation"
        )
        assert "node_projection_session_replay" in saturation["detail"]
