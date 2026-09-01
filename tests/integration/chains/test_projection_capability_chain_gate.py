# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Event Chain Gate — the PROJECTION ARM's CAPABILITY SEAM (OMN-16814).

Why this module exists.

OMN-16690: ``node_hook_event_capture/contract.yaml`` declared ``access: write``
while ``handler_hook_event_capture.py`` called ``db.query()``. The runtime read
seam refused, the projection arm swallowed the ``PermissionError`` and DLQ'd the
event, the caller kept seeing ``202``, and the quarantine sink reached a high
water mark of **8,878,932**. It ran for roughly a week behind green CI.

The platform testing inventory names why nothing caught it
(``docs/tracking/2026-08-27-event-chain-testing-inventory.md`` §7.1):

    The golden-chain DB double carries **no capability enforcement**. […] So the
    golden chain test passed, on every PR, while the live chain DLQ'd every
    single event. This is not a gap in coverage — it is a gap in *fidelity*, and
    it makes a contract/handler capability mismatch invisible to all 194
    golden-chain tests **by construction**.

That is not a claim about some other repo's tests. It is true of this repo's
too: ``tests/integration/runtime/test_projection_handler_db_injection_integration.py``
injects a ``FakeDb`` through ``_build_projection_db_adapter`` whose ``upsert`` /
``query`` do no access checking at all, so it is green for *every* declared
``access`` value.

``omnimarket#2164`` (merged 2026-08-27) closed the statically-visible half with
``check_projection_contract_access.py``: a handler that *textually* calls
``db.query()`` must declare ``access: read_write``. It cannot see a read reached
through a helper, a base class, or an adapter — and nothing anywhere proved that
the *runtime* refusal is still wired, or that a mismatched contract turns a
whole **chain** red rather than passing green while every event dead-letters.

Where the seam boundary is drawn, and why there.

The refusal lives in ``ProjectionTableOperation._assert_read_declared`` /
``_assert_write_declared`` (``handler_wiring.py``), and both fire **before** the
adapter touches a connection. So the entire capability decision is reachable
with zero database contact. This suite therefore drives:

* the real ``_prepare_handler_wiring`` arm selection,
* the real ``_make_projection_dispatch_callback``,
* the real ``_build_projection_db_adapter``,
* the real ``ProjectionDatabaseOperations`` domain-operation routing, and
* the real ``ProjectionTableOperation`` access assertions and SQL construction,

and substitutes exactly one thing: the **DBAPI driver**. ``psycopg2`` is
replaced in ``sys.modules`` by a recording double, the same technique
``test_sync_psycopg2_adapter_preserves_text_array_lists`` already uses in this
repo. That substitution is outside the seam under test by construction — it is
the connection layer, not the capability layer — which is the distinction the
OMN-16004 seam goldens draw with ``RecordingPublisher``.

Substituting the DB adapter *object* instead, as the existing suite does, is
precisely the fidelity gap this module closes. Do not "simplify" this file by
patching ``_build_projection_db_adapter``; that deletes the only thing it tests.

What is asserted.

Both directions, on the real chain: a declared-``write`` table refusing a read,
and a declared-``read`` table refusing a write. A one-sided fixture could pass by
accident. The negative rows are the negative control — they are committed as
assertions that the failure mode is DETECTABLE, never as broken chain rows.

Zero infrastructure: no broker, no database, no service container, no lane.

Non-goals (see OMN-16814). This module changes no runtime behavior. It pins
what the runtime does so that a change to it is a visible diff rather than a
silent one.

That reservation has since been cashed. OMN-17379 implements inventory §7.2 —
that a ``PermissionError`` from the access seam is a *contract* defect and
should fail the consumer loudly rather than DLQ every event forever — and the
negative rows below moved with it, from asserting a dead-letter copy to
asserting a propagated ``ProjectionNotMaterializedError`` that withholds the
offset. The rationale, and why the DLQ leg is not merely redundant on that path
but actively wrong, is on
``test_a_capability_mismatch_turns_the_chain_red_and_withholds_the_offset``.
The live defect that forced it: ``pr_merged_events`` sat 24 days behind the
topic it consumes at ``TOTAL-LAG 0``, because the projection arm caught the
write-path error, logged it, DLQ'd it and returned — and a callback that
returns normally IS an ACK.
"""

from __future__ import annotations

import json
import logging
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast
from unittest.mock import patch

import pytest
from pydantic import BaseModel, ConfigDict

from omnibase_core.models.contracts.subcontracts.model_db_ownership_subcontract import (
    ModelDbOwnershipSubcontract,
)
from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_core.models.contracts.subcontracts.model_event_bus_subcontract import (
    ModelEventBusSubcontract,
)
from omnibase_core.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_core.protocols.event_bus.protocol_event_bus_subscriber import (
    ProtocolEventBusSubscriber,
)
from omnibase_core.services.service_handler_resolver import ServiceHandlerResolver
from omnibase_core.services.service_local_handler_ownership_query import (
    ServiceLocalHandlerOwnershipQuery,
)
from omnibase_infra.errors import ProjectionNotMaterializedError
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.auto_wiring.handler_wiring import _prepare_handler_wiring
from omnibase_infra.runtime.auto_wiring.models import (
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)
from omnibase_infra.runtime.event_bus_subcontract_wiring import (
    EventBusSubcontractWiring,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from omnibase_infra.runtime.service_dispatch_result_applier import DispatchResultApplier
from omnibase_spi.protocols.runtime import ProtocolDispatchEngine
from tests.helpers.application_db_topology import (
    application_topology,
    configure_projection_dsns,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_THIS_MODULE = "tests.integration.chains.test_projection_capability_chain_gate"

ENTRY_TOPIC = "onex.evt.omnibase-infra.capability-seam-heartbeat.v1"  # onex-topic-allow: this suite's own synthetic entry topic
TERMINAL_TOPIC = "onex.evt.omnibase-infra.capability-seam-projected.v1"  # onex-topic-allow: this suite's own terminal topic
# The contract-declared DLQ. The projection arm's CONTENT-failure router
# publishes here (a malformed payload the event itself is at fault for). Since
# OMN-17379 a capability refusal is NOT routed here — it is a write-path
# failure, so the offset is withheld instead. The negative rows watch this topic
# to assert that ABSENCE, which is what keeps a re-introduced DLQ leg on the
# withheld-offset path from being an invisible regression.
DLQ_TOPIC = "onex.dlq.omnibase-infra.capability-seam.v1"  # onex-topic-allow: this suite's own DLQ sink

# A REAL table from the shipped local topology profile, chosen deliberately.
# `_require_projection_binding_privileges` refuses to wire a projection whose
# principal lacks declared privileges, and the only sanctioned grant-synthesis
# helper in the tree (`_topology_with_unshipped_grants`) refuses outright for a
# table the platform already ships. So a fabricated table name cannot reach the
# capability seam at all — the wiring rejects it three layers earlier, and the
# row would prove nothing about `access`. `omninode_internal.projection_watermarks`
# is granted INSERT/SELECT/UPDATE to `omninode_runtime` by the shipped profile,
# which is what lets the DECLARED `access` be the only variable in this matrix.
TABLE_NAME = "projection_watermarks"
TABLE_SCHEMA = "omninode_internal"


# ---------------------------------------------------------------------------
# The chain matrix
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapabilityCase:
    """One (declared access x attempted operation) pair driven through the chain.

    Attributes:
        case_id: pytest parameter id.
        declared_access: the contract's ``db_io.db_tables[].access``. This is
            the ONLY thing that varies between the admitted and refused rows —
            same handler, same wiring, same wire bytes.
        operation: what the handler attempts, ``"query"`` or ``"upsert"``.
        admitted: whether the declaration admits the operation.
        refusal_fragment: the exact substring the production seam puts in its
            ``PermissionError``. Asserting the seam's own wording is what keeps
            this test anchored to the refusal instead of to a local mirror of
            its logic.
    """

    case_id: str
    declared_access: str
    operation: str
    admitted: bool
    refusal_fragment: str = ""


CAPABILITY_CASES: tuple[CapabilityCase, ...] = (
    # ---- Admitted: the declaration covers the operation. ----
    CapabilityCase(
        case_id="read_write-declared-query-admitted",
        declared_access="read_write",
        operation="query",
        admitted=True,
    ),
    CapabilityCase(
        case_id="read_write-declared-upsert-admitted",
        declared_access="read_write",
        operation="upsert",
        admitted=True,
    ),
    # ---- Refused: the OMN-16690 shape, and its mirror image. ----
    CapabilityCase(
        case_id="write-declared-query-refused-OMN-16690-shape",
        declared_access="write",
        operation="query",
        admitted=False,
        refusal_fragment="declares access='write'; read refused",
    ),
    CapabilityCase(
        case_id="read-declared-upsert-refused",
        declared_access="read",
        operation="upsert",
        admitted=False,
        refusal_fragment="declares access='read'; write refused",
    ),
)


# ---------------------------------------------------------------------------
# The handler under chain
# ---------------------------------------------------------------------------


class ModelCapabilitySeamHeartbeat(BaseModel):
    """The wire event the projection folds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    edge_id: str
    health_status: str


class ProtocolInjectedProjectionDb(Protocol):
    """The surface the projection arm injects as ``input_data['_db']``.

    Declared so the handler below can be typed against the adapter without a
    ``type: ignore``. It is a description of the real
    ``ProjectionDatabaseOperations`` router, not a substitute for it — the
    object bound at runtime is the production one.
    """

    def query(
        self, table: str, filters: dict[str, object] | None = None
    ) -> list[dict[str, object]]: ...

    def upsert(self, table: str, conflict_key: str, row: dict[str, object]) -> bool: ...


class HandlerCapabilitySeamProjection:
    """A projection handler in the exact shape the projection arm expects.

    ``input_data['_db']`` is the adapter the arm injects. The handler performs
    ONE governed operation and returns ``rows_upserted`` — the OMN-13360 gate the
    arm reads to decide whether to emit a terminal event.

    ``_operation`` is a class attribute rather than a constructor argument
    because the arm builds nothing: ``_import_handler_class`` is patched to
    return this class and the wiring instantiates it with no arguments, exactly
    as it does in production.
    """

    _operation: str = "query"

    def handle(self, input_data: dict[str, object]) -> dict[str, object]:
        db = cast("ProtocolInjectedProjectionDb", input_data.pop("_db"))
        input_data.pop("_event_type", None)
        input_data.pop("_topic", None)
        input_data.pop("_envelope_id", None)

        if self._operation == "query":
            # The OMN-16690 call. On a contract declaring access='write' the
            # real seam refuses HERE, before any connection is touched.
            rows = db.query(TABLE_NAME, {"edge_id": input_data.get("edge_id")})
            return {"rows_upserted": 1, "rows_read": len(rows)}

        db.upsert(TABLE_NAME, "edge_id", dict(input_data))
        return {"rows_upserted": 1}


# ---------------------------------------------------------------------------
# The DBAPI double — the ONLY substituted object, and deliberately below the seam
# ---------------------------------------------------------------------------


@dataclass
class RecordedSql:
    """Every statement the real adapter built and handed to the driver.

    Non-empty proves the chain reached the database layer; empty on a refused
    row proves the refusal happened BEFORE any statement was constructed, which
    is the property that makes the seam a real gate rather than a late check.
    """

    statements: list[str] = field(default_factory=list)


def _install_recording_psycopg2(
    monkeypatch: pytest.MonkeyPatch, recorded: RecordedSql
) -> None:
    """Replace the psycopg2 DRIVER, leaving every layer above it real."""

    class FakeJson:
        def __init__(self, value: object) -> None:
            self.value = value

    class FakeCursor:
        def __enter__(self) -> FakeCursor:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def execute(self, sql: str, params: object | None = None) -> None:
            recorded.statements.append(sql)

        def fetchall(self) -> list[dict[str, object]]:
            return []

        def fetchone(self) -> tuple[str, str]:
            return ("omninode_runtime", "omnidash_analytics")

    class FakeConnection:
        closed = False
        autocommit = False

        def cursor(self, *args: object, **kwargs: object) -> FakeCursor:
            return FakeCursor()

        def commit(self) -> None:
            return None

        def rollback(self) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    fake_extras = types.SimpleNamespace(
        Json=FakeJson,
        RealDictCursor=object,
        register_uuid=lambda: None,
    )
    fake_psycopg2 = types.SimpleNamespace(
        connect=lambda dsn: FakeConnection(),
        extras=fake_extras,
    )
    monkeypatch.setitem(sys.modules, "psycopg2", fake_psycopg2)
    monkeypatch.setitem(sys.modules, "psycopg2.extras", fake_extras)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


@dataclass
class CapabilityChainRun:
    """What one chain execution produced on the bus and at the driver."""

    terminal_messages: list[bytes] = field(default_factory=list)
    dlq_messages: list[bytes] = field(default_factory=list)
    recorded: RecordedSql = field(default_factory=RecordedSql)


def _contract(case: CapabilityCase, contract_path: Path) -> ModelDiscoveredContract:
    """Build the discovered contract, varying ONLY ``access``.

    Everything else — handler, event model, topics, table identity — is
    identical across all four rows, so a difference in outcome can only be
    attributed to the declared capability.
    """
    return ModelDiscoveredContract(
        name="node_capability_seam_chain_gate",
        node_type="REDUCER_GENERIC",
        contract_version=ModelContractVersion(major=0, minor=1, patch=0),
        contract_path=contract_path,
        entry_point_name="node_capability_seam_chain_gate",
        package_name="omnibase-infra-chain-gate",
        # The projection arm emits a terminal ONLY when the contract's
        # terminal_event is also in publish_topics (handler_wiring.py:8486-8492).
        # Without it a positive row would silently assert nothing, because the
        # "no terminal" observation would be identical to a refused row's.
        terminal_event=TERMINAL_TOPIC,
        event_bus=ModelEventBusWiring(
            subscribe_topics=(ENTRY_TOPIC,),
            publish_topics=(TERMINAL_TOPIC,),
            dlq_topics=(DLQ_TOPIC,),
        ),
        db_io=ModelDbOwnershipSubcontract(
            db_tables=[
                ModelDbTableDeclaration(
                    name=TABLE_NAME,
                    database_ref="application",
                    schema=TABLE_SCHEMA,
                    migration=f"tests/{TABLE_NAME}.sql",
                    access=case.declared_access,
                    role=f"{TABLE_NAME}_projection",
                )
            ]
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="operation_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name="HandlerCapabilitySeamProjection",
                        module=_THIS_MODULE,
                    ),
                    event_model=ModelHandlerRef(
                        name="ModelCapabilitySeamHeartbeat",
                        module=_THIS_MODULE,
                    ),
                    operation="capability_seam_projection",
                ),
            ),
        ),
    )


CONTRACT_YAML = f"""
name: "node_capability_seam_chain_gate"
node_type: "REDUCER_GENERIC"
event_bus:
  subscribe_topics:
    - "{ENTRY_TOPIC}"
  publish_topics:
    - "{TERMINAL_TOPIC}"
  dlq_topics:
    - "{DLQ_TOPIC}"
terminal_event: "{TERMINAL_TOPIC}"
"""


async def _run_capability_chain(
    case: CapabilityCase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> CapabilityChainRun:
    """Drive one (access x operation) pair from raw wire bytes to the seam."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(CONTRACT_YAML, encoding="utf-8")

    run = CapabilityChainRun()
    _install_recording_psycopg2(monkeypatch, run.recorded)

    contract = _contract(case, contract_path)
    assert contract.db_io is not None and contract.db_io.db_tables, (
        f"[{case.case_id}] gate precondition not met: the parsed contract "
        "carries no db_io.db_tables, so the projection arm would not be "
        "selected and this row would pass vacuously. Fix the fixture or the "
        "core pin — do not delete the row."
    )
    assert contract.db_io.db_tables[0].access == case.declared_access, (
        f"[{case.case_id}] the declared access did not survive parsing "
        f"({contract.db_io.db_tables[0].access!r} != {case.declared_access!r}); "
        "the row would be asserting against the wrong capability."
    )

    assert contract.handler_routing is not None
    entry = contract.handler_routing.handlers[0]

    bus = EventBusInmemory(environment="chain-gate", group="chain-gate")
    await bus.start()

    async def _collect_terminal(message: object) -> None:
        run.terminal_messages.append(cast("bytes", getattr(message, "value", b"")))

    async def _collect_dlq(message: object) -> None:
        run.dlq_messages.append(cast("bytes", getattr(message, "value", b"")))

    await bus.subscribe(
        TERMINAL_TOPIC,
        on_message=_collect_terminal,
        group_id=f"chain-gate-terminal-{case.case_id}",
    )
    await bus.subscribe(
        DLQ_TOPIC,
        on_message=_collect_dlq,
        group_id=f"chain-gate-dlq-{case.case_id}",
    )

    engine = MessageDispatchEngine()

    handler_cls = type(
        "HandlerCapabilitySeamProjectionBound",
        (HandlerCapabilitySeamProjection,),
        {"_operation": case.operation},
    )

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=handler_cls,
    ):
        prepared = _prepare_handler_wiring(
            contract=contract,
            entry=entry,
            dispatch_engine=engine,
            resolver=ServiceHandlerResolver(),
            ownership_query=ServiceLocalHandlerOwnershipQuery(
                local_node_names=frozenset({contract.name})
            ),
            event_bus=bus,
            container=None,
            topology=application_topology(),
        )

    assert prepared.quarantine_reason is None, (
        f"[{case.case_id}] wiring quarantined the handler before a single "
        f"message was dispatched: {prepared.quarantine_reason} "
        f"({prepared.quarantine_detail})"
    )

    engine.register_dispatcher(
        dispatcher_id=prepared.dispatcher_id,
        dispatcher=prepared.dispatcher,
        category=prepared.category,
        message_types=prepared.message_types,
    )
    engine.register_route(
        ModelDispatchRoute(
            route_id=f"{case.case_id}-route",
            topic_pattern=ENTRY_TOPIC,
            message_category=prepared.category,
            handler_id=prepared.dispatcher_id,
        )
    )
    engine.freeze()

    applier = DispatchResultApplier(
        event_bus=cast("ProtocolEventBusLike", bus),
        output_topic=TERMINAL_TOPIC,
        allowed_output_topics=(TERMINAL_TOPIC,),
    )

    wiring = EventBusSubcontractWiring(
        event_bus=cast("ProtocolEventBusSubscriber", bus),
        dispatch_engine=cast("ProtocolDispatchEngine", engine),
        environment="chain-gate",
        node_name="capability-seam-chain-gate",
        service="omnibase-infra",
        version="v1",
        result_applier=applier,
    )
    await wiring.wire_subscriptions(
        ModelEventBusSubcontract(
            version=ModelSemVer(major=1, minor=0, patch=0),
            subscribe_topics=[ENTRY_TOPIC],
            publish_topics=[TERMINAL_TOPIC],
        ),
        "capability-seam-chain-gate",
    )

    envelope_json = {
        "payload": {"edge_id": "tenant-beta-edge-1", "health_status": "HEALTHY"},
        "event_type": ENTRY_TOPIC,
        "correlation_id": "7a300827-0000-4000-8000-00000000c0de",
        "source_tool": "chain-gate",
    }
    try:
        await bus.publish(
            ENTRY_TOPIC,
            key=None,
            value=json.dumps(envelope_json).encode("utf-8"),
        )
    finally:
        await wiring.cleanup()
        await bus.close()

    return run


@pytest.fixture(autouse=True)
def _projection_dsns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the projection arm DSNs so it constructs.

    Set here rather than inherited from the shell: a dev shell exports
    ``OMNIDASH_ANALYTICS_DB_URL`` and a CI runner does not, which is the exact
    OMN-16796 environment-dependence class. A row that passes locally and fails
    on the runner for this reason is a fixture bug, not a finding.
    """
    configure_projection_dsns(monkeypatch, url="postgresql://fixture/omn16814")


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [pytest.param(c, id=c.case_id) for c in CAPABILITY_CASES if c.admitted],
)
async def test_a_declaration_that_admits_the_operation_lets_the_chain_complete(
    case: CapabilityCase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC1 — the positive half, through the real ``ProjectionTableOperation``.

    A terminal event on the bus is the arm's own OMN-13360-gated statement that
    the projection wrote; recorded SQL is independent proof that the chain
    actually reached the database layer rather than short-circuiting somewhere
    above it and reporting success.
    """
    run = await _run_capability_chain(case, tmp_path, monkeypatch)

    assert run.dlq_messages == [], (
        f"[{case.case_id}] a declaration that ADMITS {case.operation!r} still "
        f"dead-lettered the event: {run.dlq_messages!r}"
    )
    assert run.terminal_messages, (
        f"[{case.case_id}] no terminal event reached {TERMINAL_TOPIC}; the "
        "chain did not complete even though the contract admits the operation."
    )
    assert run.recorded.statements, (
        f"[{case.case_id}] the chain emitted a terminal event without the "
        "adapter ever executing a statement — the projection reported a write "
        "it did not perform, which is the OMN-13360 false-positive shape."
    )


@pytest.mark.parametrize(
    "case",
    [pytest.param(c, id=c.case_id) for c in CAPABILITY_CASES if not c.admitted],
)
async def test_a_capability_mismatch_turns_the_chain_red_and_withholds_the_offset(
    case: CapabilityCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """AC2 + AC4 — the negative control, both directions.

    This is the OMN-16690 signature and its mirror image, driven end to end. It
    is committed as a PASSING assertion that the failure mode is detectable,
    never as a broken chain row.

    OMN-17379 CHANGED THE CARRIER OF THAT DETECTION, DELIBERATELY.

    This module's header names the change and reserves this exact spot for it:
    inventory §7.2 — "a ``PermissionError`` from the access seam is a *contract*
    defect and should fail the consumer loudly rather than DLQ every event
    forever" — was called correct and left alone so that "that change, when
    someone makes it, is a visible diff rather than a silent one." OMN-17379 is
    that change, and this is that diff.

    The old row asserted the refused event reached ``DLQ_TOPIC``. It no longer
    does, and must not: the projection arm now distinguishes a CONTENT failure
    (a malformed payload — the event's own defect, which redelivery can never
    repair, so DLQ-and-advance stays correct) from a WRITE-PATH failure (this
    one — the runtime's defect, where the event is well-formed and still owed a
    row). For the second class it raises ``ProjectionNotMaterializedError``,
    which ``EventBusKafka._dispatch_to_subscriber`` classifies offset-unsafe
    unconditionally and rewinds the partition on.

    Why the dead-letter copy is not merely redundant but wrong here: a withheld
    offset means the record is redelivered on a loop until an operator repairs
    the declaration, and a DLQ leg on that path publishes one copy per
    redelivery. The 8,878,932-message quarantine high-water mark in this
    module's header is what that looks like at scale. The record's home is its
    own topic, uncommitted.

    Four separate facts are asserted because any one alone is weak:

    1. **No terminal.** The chain does not report success. This is the half the
       live outage got wrong at the HTTP boundary (callers saw ``202``).
    2. **The refusal propagates as ``ProjectionNotMaterializedError``, naming
       ``PermissionError``.** This is the detectable evidence, and it is
       strictly louder than the DLQ copy it replaces: the old behavior returned
       normally, which the consume boundary reads as an ACK, so the offset
       advanced past an event that was never written. That silent ACK is what
       let ``pr_merged_events`` sit 24 days behind its topic at TOTAL-LAG 0.
    3. **No dead-letter copy.** Asserted positively, so a future re-introduction
       of the DLQ leg on this path is a red row rather than an invisible
       regression back toward the 8.9M sink.
    4. **Zero SQL was constructed.** The refusal happened at the capability
       seam, before any statement existed — so this is a real gate, not a late
       check that would have already touched the database.
    """
    with caplog.at_level(logging.ERROR):
        run = await _run_capability_chain(case, tmp_path, monkeypatch)

    assert run.terminal_messages == [], (
        f"[{case.case_id}] the chain published a terminal event for an "
        f"operation the contract declares access={case.declared_access!r} does "
        "NOT permit. That is a projection advertising a write the runtime "
        "refused — the OMN-16690 failure mode, un-detected."
    )

    refusals = [
        record.exc_info[1]
        for record in caplog.records
        if record.exc_info is not None
        and isinstance(record.exc_info[1], ProjectionNotMaterializedError)
    ]
    assert refusals, (
        f"[{case.case_id}] the refused event produced no "
        "ProjectionNotMaterializedError anywhere on the chain. The capability "
        "refusal is then undetectable AND the offset advances past it — "
        "strictly worse than the OMN-16690 behavior this row exists to pin, "
        f"which at least left a dead-letter copy. Records: {caplog.records!r}"
    )

    refusal_text = str(refusals[0])
    assert "PermissionError" in refusal_text, (
        f"[{case.case_id}] the propagated refusal does not name "
        "PermissionError, so an operator cannot tell a capability refusal from "
        f"a malformed event — the two need different fixes. Text: "
        f"{refusal_text!r}"
    )
    # MEASURED, not stylistic. Neutering `_assert_read_declared` /
    # `_assert_write_declared` on the production class does NOT make the
    # PermissionError disappear — the chain then fails a SECOND, independent
    # refusal ("Projection operation has no declared workload binding"), because
    # a write-declared table has no read binding to resolve either. So an
    # assertion of the form `"PermissionError" in refusal_text` STAYS GREEN with
    # the capability seam switched off, and would gate nothing.
    #
    # Asserting the seam's own wording is what makes this row non-vacuous.
    # Verified by running exactly that experiment: with both assertions stubbed
    # to `lambda self: None`, these two rows go RED on this line and only this
    # line. Do not relax it to an exception-type check.
    assert case.refusal_fragment in refusal_text, (
        f"[{case.case_id}] the propagated refusal does not carry the seam's own "
        f"wording {case.refusal_fragment!r}. Asserting the production wording "
        "rather than a re-implementation of the rule is what keeps this row "
        f"anchored to the real seam. Text: {refusal_text!r}"
    )

    assert run.dlq_messages == [], (
        f"[{case.case_id}] the refused event was ALSO dead-lettered: "
        f"{run.dlq_messages!r}. On a withheld-offset path that publishes one "
        "copy per redelivery for as long as the declaration stays broken — the "
        "shape that took the quarantine sink to 8,878,932 messages. The record "
        "is preserved by the uncommitted offset, not by a copy (OMN-17379)."
    )

    assert run.recorded.statements == [], (
        f"[{case.case_id}] the adapter executed {run.recorded.statements!r} "
        "despite the capability refusal. The seam must refuse BEFORE building a "
        "statement; a refusal after the fact is not a gate."
    )


async def test_the_refusal_comes_from_the_production_seam_not_a_local_mirror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC3 — anchor the assertion to the seam, not to a copy of its rule.

    ``omnimarket/src/omnimarket/nodes/node_projection_live_events/tests/
    test_projection_live_events.py`` re-implements ``access not in {"read",
    "read_write"}`` and asserts its own copy. That test was green throughout the
    OMN-16690 outage, because a mirror of a rule cannot tell you whether the
    rule is still wired.

    Here the capability decision is exercised by importing the production
    classes and driving them directly: same objects the chain above used, no
    reimplementation. If ``ProjectionTableOperation`` stops refusing, this fails
    — which is the property a mirror cannot have.
    """
    from omnibase_infra.runtime.auto_wiring.handler_wiring import (
        ProjectionTableOperation,
    )

    # Both refusals are methods on the production class, not constants copied
    # into this file. Their absence is itself a finding.
    assert hasattr(ProjectionTableOperation, "_assert_read_declared")
    assert hasattr(ProjectionTableOperation, "_assert_write_declared")

    refused = await _run_capability_chain(
        CAPABILITY_CASES[2], tmp_path / "mirror-check", monkeypatch
    )
    admitted = await _run_capability_chain(
        CAPABILITY_CASES[0], tmp_path / "mirror-check-admitted", monkeypatch
    )

    # The ONLY difference between these two runs is the contract's declared
    # access string. Same handler, same operation, same wire bytes, same wiring.
    # So the divergence below is attributable to the capability seam and to
    # nothing else.
    assert refused.terminal_messages == [] and admitted.terminal_messages
    assert refused.recorded.statements == [] and admitted.recorded.statements
