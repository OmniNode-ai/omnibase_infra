# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The third way a projection persists nothing, and reads HEALTHY (OMN-17448).

RED-first reproduction of the mask measured live on the ``.201`` dev lane on
2026-09-01, immediately after the OMN-17374 lane-apply deploy.

The shape
---------
``node_projection_tenant_registry`` and ``node_projection_delegation`` were both
subscribed, both consuming, both advancing to LAG 0 — and neither ever executed
its ``project_event``. ``handler_wiring``'s auto-wired dispatch callback opens
with ``if is_projection_runner: return None`` (deliberate, OMN-15905: a handler
with the standalone-runner shape owns its own consume loop and DB pool, so the
sanctioned way to run it is a dedicated writer process outside the kernel). The
Kafka consumer still commits. Net: **consume, ack, write nothing, no DLQ record,
no terminal event, no ERROR log.**

Live evidence, quoted from the ticket: a well-formed ``TENANT_CREATED``
published to ``onex.tenant.events`` at offset 37 moved the consumer group's
CURRENT-OFFSET 37 → 38 at LAG 0, left ``tenant_registry_mirror`` at 0 rows,
left the malformed-DLQ topic at HWM 0, and produced zero log lines for that
correlation id in any of the four runtime containers.

Why OMN-16994's two masks cannot see it
---------------------------------------
Both, by construction rather than by accident:

* ``unattached_projections`` — the topic IS attached. The subscription is real;
  only the dispatch is a no-op.
* ``dlq_saturated_projections`` — nothing raises, so ``messages_dlq`` stays 0
  and the ratio never reaches 1.0.

``service_runtime_health_monitor`` therefore logged ``status=HEALTHY ...
projections=13 unattached_projections=0`` throughout, while
``node_projection_tenant_registry`` had no writer on ANY lane.

What is asserted here
---------------------
Only what a single kernel process can observe: "this process subscribes X and
dispatches nothing for it." It deliberately does NOT assert "X has no writer
anywhere" — a kernel cannot see a sibling Deployment, and claiming otherwise
would report a permanent false outage on every lane where the standalone writer
IS correctly deployed. The corpus-level "every subscribing lane has a deployed
writer" assertion is a static gate over deployment manifests (OMN-17448 AC5).

Related Tickets:
    - OMN-17448: this ticket
    - OMN-15905: why the skip exists, and the standalone-writer fix pattern
    - OMN-16994: the two masks this one sits alongside
    - OMN-16874: ``_is_standalone_projection_runner``, the recorded predicate
"""

from __future__ import annotations

from typing import Any

import pytest

from omnibase_core.models.contracts.subcontracts.model_db_ownership_subcontract import (
    ModelDbOwnershipSubcontract,
)
from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
)
from omnibase_infra.runtime.health.projection_liveness import (
    describe_projection_write_path,
    evaluate_projection_liveness,
    select_projection_contracts,
)
from omnibase_infra.runtime.projection_dispatch_ledger import (
    dispatch_skipped_projections,
    record_dispatch_skipped_projection,
    reset_dispatch_skipped_projections,
)

TENANT_TOPIC = "onex.tenant.events"
REGISTRY_PROJECTION = "node_projection_tenant_registry"


@pytest.fixture(autouse=True)
def _clean_ledger() -> Any:
    """The ledger is process-local; never let one test's wiring leak into another."""
    reset_dispatch_skipped_projections()
    yield
    reset_dispatch_skipped_projections()


def _registry_contract(
    *, name: str = REGISTRY_PROJECTION, topic: str = TENANT_TOPIC
) -> ModelDiscoveredContract:
    """The real shape: one subscribed topic, one written relation."""
    return ModelDiscoveredContract(
        name=name,
        node_type="REDUCER",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=__file__,
        entry_point_name=name,
        package_name="omnimarket",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(topic,),
            publish_topics=(),
        ),
        db_io=ModelDbOwnershipSubcontract(
            db_tables=[
                ModelDbTableDeclaration(
                    name="tenant_registry_mirror",
                    database_ref="application",
                    schema="omninode_internal",
                    migration="0000_create_tenant_registry_mirror.sql",
                    access="write",
                    role="registry_mirror",
                )
            ]
        ),
    )


def _projections() -> tuple[Any, ...]:
    return select_projection_contracts(
        ModelAutoWiringManifest(contracts=(_registry_contract(),), errors=())
    )


# =============================================================================
# The ledger
# =============================================================================


@pytest.mark.unit
class TestProjectionDispatchLedger:
    """Process-local record of the standalone-runner wiring branch."""

    def test_recorded_name_is_reported(self) -> None:
        record_dispatch_skipped_projection(REGISTRY_PROJECTION)
        assert dispatch_skipped_projections() == frozenset({REGISTRY_PROJECTION})

    def test_empty_ledger_is_empty(self) -> None:
        assert dispatch_skipped_projections() == frozenset()

    def test_blank_name_is_not_stored(self) -> None:
        """An unnamed entry cannot be rendered on a detail an operator can look up."""
        record_dispatch_skipped_projection("   ")
        assert dispatch_skipped_projections() == frozenset()

    def test_recording_is_idempotent(self) -> None:
        record_dispatch_skipped_projection(REGISTRY_PROJECTION)
        record_dispatch_skipped_projection(REGISTRY_PROJECTION)
        assert dispatch_skipped_projections() == frozenset({REGISTRY_PROJECTION})


# =============================================================================
# The verdict — this is the RED half
# =============================================================================


@pytest.mark.unit
class TestNonWritingProjectionIsNamed:
    """The live .201 shape: attached, lag 0, DLQ 0, and nothing written."""

    def test_the_two_prior_masks_both_read_clean_on_this_shape(self) -> None:
        """RED anchor: without the write-path leg, this state reports nothing.

        Attached (so not unattached) and never raising (so never DLQ-saturated).
        This is exactly the state the monitor called HEALTHY while the mirror
        held 0 rows — asserted here so a future change that reintroduces the
        mask fails loudly instead of quietly.
        """
        verdict = evaluate_projection_liveness(
            projections=_projections(),
            attached_topics=frozenset({TENANT_TOPIC}),
            flow_windows=(),
            dispatch_skipped=frozenset({REGISTRY_PROJECTION}),
        )
        assert verdict.unattached_projections == ()
        assert verdict.dlq_saturated_projections == ()

    def test_dispatch_skipped_projection_is_named(self) -> None:
        verdict = evaluate_projection_liveness(
            projections=_projections(),
            attached_topics=frozenset({TENANT_TOPIC}),
            flow_windows=(),
            dispatch_skipped=frozenset({REGISTRY_PROJECTION}),
        )
        assert verdict.nonwriting_projections == (REGISTRY_PROJECTION,)

    def test_dispatched_projection_is_not_named(self) -> None:
        verdict = evaluate_projection_liveness(
            projections=_projections(),
            attached_topics=frozenset({TENANT_TOPIC}),
            flow_windows=(),
            dispatch_skipped=frozenset(),
        )
        assert verdict.nonwriting_projections == ()

    def test_a_ledger_entry_out_of_contract_scope_is_not_named(self) -> None:
        """A stale or foreign entry must not manufacture an unlookupable name."""
        verdict = evaluate_projection_liveness(
            projections=_projections(),
            attached_topics=frozenset({TENANT_TOPIC}),
            flow_windows=(),
            dispatch_skipped=frozenset({"node_projection_from_another_process"}),
        )
        assert verdict.nonwriting_projections == ()

    def test_detail_names_the_projection_and_says_what_is_missing(self) -> None:
        verdict = evaluate_projection_liveness(
            projections=_projections(),
            attached_topics=frozenset({TENANT_TOPIC}),
            flow_windows=(),
            dispatch_skipped=frozenset({REGISTRY_PROJECTION}),
        )
        detail = describe_projection_write_path(verdict)
        assert REGISTRY_PROJECTION in detail
        assert "dedicated writer" in detail

    def test_clean_detail_states_the_count(self) -> None:
        verdict = evaluate_projection_liveness(
            projections=_projections(),
            attached_topics=frozenset({TENANT_TOPIC}),
            flow_windows=(),
            dispatch_skipped=frozenset(),
        )
        assert describe_projection_write_path(verdict) == (
            "All 1 declared projection(s) dispatch in-process"
        )


# =============================================================================
# The wiring seam records it on the branch that decides it
# =============================================================================


class _StandaloneRunnerShaped:
    """A handler with the four properties ``_is_standalone_projection_runner`` reads.

    Deliberately shaped rather than mocked: the predicate is structural
    (``project_event`` + ``run`` + ``topics`` + a handler-owned DB adapter with
    ``connect``/``close``), and a MagicMock satisfies every ``callable()`` check
    by accident, which would make the test pass against a predicate that had
    stopped working.
    """

    topics: tuple[str, ...] = (TENANT_TOPIC,)

    class _Db:
        def connect(self) -> None: ...

        def close(self) -> None: ...

    def __init__(self) -> None:
        self.db = self._Db()

    def project_event(self, *_: object, **__: object) -> bool:
        return True

    def run(self) -> None: ...


class _InProcessHandler:
    """No runner shape: the kernel dispatches this one for real."""

    def handle(self, input_data: dict[str, object]) -> dict[str, object]:
        return input_data


@pytest.mark.unit
class TestWiringSeamRecordsTheSkip:
    """The record must come from the same branch that returns ``None``."""

    def _build(self, handler: object, contract_name: str) -> None:
        from omnibase_infra.runtime.auto_wiring.handler_wiring import (
            ProjectionDatabaseTarget,
            _make_projection_dispatch_callback,
        )

        _make_projection_dispatch_callback(
            handler,
            ProjectionDatabaseTarget(
                tables=(),
                table_targets=(),
                physical_database="omnidash_analytics",
            ),
            (TENANT_TOPIC,),
            contract_name=contract_name,
        )

    def test_standalone_runner_contract_is_recorded(self) -> None:
        self._build(_StandaloneRunnerShaped(), REGISTRY_PROJECTION)
        assert REGISTRY_PROJECTION in dispatch_skipped_projections()

    def test_in_process_handler_contract_is_not_recorded(self) -> None:
        self._build(_InProcessHandler(), "node_projection_dispatched_in_kernel")
        assert dispatch_skipped_projections() == frozenset()
