# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The kernel stops subscribing projections it will never dispatch (OMN-17562).

The defect
----------
OMN-15905 made the shared kernel refuse to dispatch a handler with the
standalone-runner shape: that handler owns its own consume loop and its own DB
pool, so a dedicated writer process runs it. OMN-17519 did the same for a
handler entry assigned zero topics. Both branches are correct.

Neither branch touched the *subscription*. The kernel still joined the consumer
group for the contract's topics, took every message, returned ``None`` before
any handler ran, and committed the offset. Consume, ack, write nothing, no DLQ
record, no terminal event, no ERROR log. Measured on the ``.201`` dev and
stability-test lanes 2026-09-02: ``projection_count=37 nonwriting=15`` on the
main runtime profile and ``13/2`` on effects, byte-identical on both lanes.

The events were not merely unwritten — they were *destroyed*. A committed
offset on a topic no process will re-read is unrecoverable, so deploying the
writer later cannot backfill.

The fix
-------
Extend the treatment ``plugin_managed`` (OMN-10864), ``requires_cloud_gateway``
(OMN-13809) and raw-event projection contracts already get: a contract with no
live dispatcher in THIS process subscribes zero topics. Its events then
accumulate on the broker, replayable by the writer that eventually runs.

Both subscribe seams must apply it, because they are independent paths:

* :func:`_prepare_contract_wiring` — the immediate path, which computes
  ``subscription_topics`` up front.
* :func:`subscribe_wired_contract_topics` — the deferred path taken by the real
  kernel boot (``subscribe_immediately=False``, subscribe after the dispatch
  engine is frozen), which re-derives eligibility from the wiring REPORT.

The mixed-contract guard
------------------------
The naive rule — "the contract is in the dispatch-skipped ledger, so
unsubscribe it" — silently unsubscribes contracts that genuinely dispatch.
``projection_pattern_learning`` and ``projection_routing_decision`` each declare
ONE subscribe topic and TWO handler entries: the in-process
``HandlerProjection*`` (``event_model`` set, takes the topic via the
``len(topics) == 1`` branch) and the ``*ProjectionRunner`` (no ``event_model``,
takes it via the ``event_model is None`` branch). Both own it. The runner entry
is recorded as skipped; the sibling is live-dispatched and writes rows.

So the decision is per (contract, handler entry) and the contract-level fact is
"EVERY entry is a no-op", carried as a typed field on ``PreparedWiring`` and
``ModelContractWiringResult`` — never read back out of the process-global
ledger at the subscribe seam.

Related Tickets:
    - OMN-17562: this module — the kernel stops subscribing what it cannot dispatch
    - OMN-17448: the ledger and the ``projection_write_path`` health dimension
    - OMN-16874: ``_is_standalone_projection_runner``, the runner-branch predicate
    - OMN-17519: ``_projection_dispatch_owned_elsewhere``, the zero-route branch
    - OMN-15905: why the kernel must not dispatch a standalone runner
    - OMN-10864: ``plugin_managed``, the treatment this extends
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_core.models.contracts.subcontracts.model_db_ownership_subcontract import (
    ModelDbOwnershipSubcontract,
)
from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    subscribe_wired_contract_topics,
    wire_from_manifest,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from tests.helpers.application_db_topology import (
    application_topology,
    configure_projection_dsns,
)

_THIS_MODULE = "tests.unit.runtime.auto_wiring.test_omn17562_subscription_skip"
_TOPIC = "onex.evt.omnimarket.projection-subject.v1"
_RUNNER_ONLY = "node_projection_runner_owned"
_MIXED = "node_projection_mixed_entries"
_PATCH_IMPORT_HANDLER = (
    "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class"
)


# ---------------------------------------------------------------------------
# Handlers — shaped, never mocked. ``_is_standalone_projection_runner`` reads
# four structural properties, and a MagicMock satisfies every ``callable()``
# check by accident, which would make these tests pass against a predicate that
# had stopped working.
# ---------------------------------------------------------------------------


class _OwnedDb:
    def connect(self) -> None: ...

    def close(self) -> None: ...


class StandaloneRunnerHandler:
    """Runner shape: own consume loop, own projection entrypoint, own pool."""

    topics: tuple[str, ...] = (_TOPIC,)

    def __init__(self) -> None:
        self.db = _OwnedDb()

    def project_event(self, *_: object, **__: object) -> bool:
        return True

    def run(self) -> None: ...


class InProcessProjectionHandler:
    """No runner shape: the kernel dispatches this one for real."""

    def handle(self, input_data: dict[str, Any]) -> dict[str, Any]:
        return {"rows_upserted": 1, "echo": input_data}


def _handler_classes() -> dict[str, type]:
    return {
        "StandaloneRunnerHandler": StandaloneRunnerHandler,
        "InProcessProjectionHandler": InProcessProjectionHandler,
    }


def _import_by_name(_module: str, class_name: str) -> type:
    return _handler_classes()[class_name]


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


def _projection_table() -> ModelDbTableDeclaration:
    """A relation the shipped topology really grants, so the arm resolves."""
    return ModelDbTableDeclaration(
        name="delegation_routing_tenant_overlay",
        database_ref="application",
        schema="tenant",
        migration="0001_create_delegation_routing_tenant_overlay.sql",
        access="read_write",
        role="tenant_routing_overlay",
    )


def _entry(handler_name: str, *, with_event_model: bool) -> ModelHandlerRoutingEntry:
    return ModelHandlerRoutingEntry(
        handler=ModelHandlerRef(name=handler_name, module=_THIS_MODULE),
        event_model=(
            ModelHandlerRef(name="ModelProjectionSubject", module=_THIS_MODULE)
            if with_event_model
            else None
        ),
        operation=None,
    )


def _contract(
    name: str, entries: tuple[ModelHandlerRoutingEntry, ...]
) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="REDUCER",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/contract.yaml"),
        entry_point_name=name,
        package_name="omnimarket",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(_TOPIC,),
            publish_topics=(),
        ),
        db_io=ModelDbOwnershipSubcontract(db_tables=[_projection_table()]),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=entries,
        ),
    )


def _runner_only_contract() -> ModelDiscoveredContract:
    """The OMN-15905 shape with NO in-process sibling: nothing dispatches here."""
    return _contract(
        _RUNNER_ONLY,
        (_entry("StandaloneRunnerHandler", with_event_model=False),),
    )


def _mixed_contract() -> ModelDiscoveredContract:
    """The ``projection_pattern_learning`` shape: one runner, one LIVE sibling.

    One subscribe topic, two entries, and BOTH own it —
    ``InProcessProjectionHandler`` via ``len(topics) == 1`` and
    ``StandaloneRunnerHandler`` via ``event_model is None``.
    """
    return _contract(
        _MIXED,
        (
            _entry("InProcessProjectionHandler", with_event_model=True),
            _entry("StandaloneRunnerHandler", with_event_model=False),
        ),
    )


@pytest.fixture(autouse=True)
def _dsns(monkeypatch: pytest.MonkeyPatch) -> None:
    """The live projection arm refuses to build without a DSN per binding."""
    configure_projection_dsns(monkeypatch, url="postgresql://fixture/omn17562")


def _event_bus() -> MagicMock:
    bus = MagicMock(spec=ProtocolEventBusLike)
    bus.subscribe = AsyncMock(return_value=AsyncMock())
    return bus


async def _wire(
    contract: ModelDiscoveredContract,
    bus: MagicMock,
    *,
    subscribe_immediately: bool = True,
) -> tuple[Any, MessageDispatchEngine, ModelAutoWiringManifest]:
    manifest = ModelAutoWiringManifest(contracts=(contract,))
    engine = MessageDispatchEngine()
    with patch(_PATCH_IMPORT_HANDLER, side_effect=_import_by_name):
        report = await wire_from_manifest(
            manifest,
            engine,
            event_bus=bus,
            environment="local",
            subscribe_immediately=subscribe_immediately,
            topology=application_topology(),
        )
    return report, engine, manifest


# ===========================================================================
# The immediate seam — _prepare_contract_wiring
# ===========================================================================


@pytest.mark.unit
class TestImmediateSubscribeSeam:
    @pytest.mark.asyncio
    async def test_contract_with_no_live_dispatcher_subscribes_zero_topics(
        self,
    ) -> None:
        """RED before the fix: the kernel joined the group and ate every message.

        The dispatcher is still registered — the contract is WIRED, the routes
        exist, and a future routing change surfaces on them. Only the Kafka
        subscription is withheld, exactly as ``plugin_managed`` withholds it.
        """
        bus = _event_bus()
        report, _engine, _manifest = await _wire(_runner_only_contract(), bus)

        result = next(r for r in report.results if r.contract_name == _RUNNER_ONLY)
        assert result.topics_subscribed == (), (
            "the kernel subscribed a contract it dispatches nothing for — every "
            "message is consumed, acked and destroyed"
        )
        bus.subscribe.assert_not_called()
        assert result.dispatchers_registered, (
            "the dispatcher registration must be unchanged; this ticket withholds "
            "the SUBSCRIPTION, not the route"
        )

    @pytest.mark.asyncio
    async def test_a_mixed_contract_keeps_its_subscription(self) -> None:
        """RED-guard: one live sibling is enough to keep consuming.

        ``projection_pattern_learning`` / ``projection_routing_decision``. A
        rule keyed on the contract name in the dispatch-skipped ledger would
        unsubscribe these two and stop a projection that writes rows today.
        """
        bus = _event_bus()
        report, _engine, _manifest = await _wire(_mixed_contract(), bus)

        result = next(r for r in report.results if r.contract_name == _MIXED)
        assert result.topics_subscribed == (_TOPIC,), (
            "a contract with a LIVE in-process entry must keep consuming"
        )
        bus.subscribe.assert_called()

    @pytest.mark.asyncio
    async def test_the_contract_level_fact_is_carried_on_the_wiring_result(
        self,
    ) -> None:
        """The subscribe seams read a typed field, never the process-global ledger.

        The ledger is keyed by contract and written by every process that wires
        one; the subscribe decision is about THIS wiring pass. Reading the
        ledger here would couple two facts that can legitimately differ.
        """
        bus = _event_bus()
        runner_report, _e1, _m1 = await _wire(_runner_only_contract(), bus)
        mixed_report, _e2, _m2 = await _wire(_mixed_contract(), _event_bus())

        runner = next(
            r for r in runner_report.results if r.contract_name == _RUNNER_ONLY
        )
        mixed = next(r for r in mixed_report.results if r.contract_name == _MIXED)

        assert runner.nonwriting_handlers == ("StandaloneRunnerHandler",)
        assert runner.has_no_live_dispatcher is True
        assert mixed.nonwriting_handlers == ("StandaloneRunnerHandler",)
        assert mixed.has_no_live_dispatcher is False


# ===========================================================================
# The deferred seam — subscribe_wired_contract_topics
# ===========================================================================


@pytest.mark.unit
class TestDeferredSubscribeSeam:
    @pytest.mark.asyncio
    async def test_deferred_subscribe_path_applies_the_same_skip(self) -> None:
        """The real kernel boot subscribes AFTER freezing the dispatch engine.

        ``wire_from_manifest(..., subscribe_immediately=False)`` leaves
        ``subscription_topics`` unread by this path: it re-derives eligibility
        from the report. A fix applied only to the immediate seam would change
        nothing on any deployed lane.
        """
        bus = _event_bus()
        report, engine, manifest = await _wire(
            _runner_only_contract(), bus, subscribe_immediately=False
        )
        assert bus.subscribe.call_count == 0  # nothing subscribed yet, by design

        subscribed = await subscribe_wired_contract_topics(
            manifest, report, engine, bus, environment="local"
        )

        assert _RUNNER_ONLY not in subscribed
        bus.subscribe.assert_not_called()

    @pytest.mark.asyncio
    async def test_deferred_path_still_subscribes_a_mixed_contract(self) -> None:
        bus = _event_bus()
        report, engine, manifest = await _wire(
            _mixed_contract(), bus, subscribe_immediately=False
        )

        subscribed = await subscribe_wired_contract_topics(
            manifest, report, engine, bus, environment="local"
        )

        assert subscribed.get(_MIXED) == (_TOPIC,)
