# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18419 AC4, tree half: the declared TREE grades green through two real dispatches.

Why this file exists
--------------------
OMN-18419 landed two independent halves. The PUBLISH half — that the
pattern-B worker command and the state_io outbox record the envelope whose
consumption caused them — is pinned against the real dispatch path by
``tests/integration/runtime/test_chain_causal_edge_seam_omn18419.py``. The TREE
half — that the replay grades each recorded edge against the parent the
contract DECLARES, rather than against whatever sat at ``index - 1`` — was
pinned only by unit tests over the pure grading module, on rows the test
itself constructed, plus a one-off replay run by hand. AC4's falsifier names
exactly that: "it asserts only on hand-built rows rather than on evidence
produced by a dispatch."

So nothing this file asserts on is built by this file.

Two real dispatches, and what each one produces
-----------------------------------------------
1. The evidence. Each of the chain's five hops is published as a
   ``ModelEventMessage`` carrying real ONEX headers — ``message_id`` is the
   hop's envelope identity, ``parent_message_id`` is the causal edge — and is
   driven through ``HandlerLedgerProjection.handle()`` ->
   ``IntentEffectDispatchBridge`` -> ``HandlerLedgerAppend``, the same three
   hops the live ledger-projection path takes: the canonical definition-B
   entrypoint receiving its contract-declared ``event_model`` typed, then the
   single generic intent effect the kernel DERIVES for every audit consumer,
   then the write effect's own persistence. The ``public.event_ledger`` rows
   the chain writer later reads back are the OUTPUT of that dispatch; no SQL
   in this file writes them. What this leg does NOT cover is Kafka and the
   auto-wiring decision that subscribes the projection node at all — the same
   boundary ``tests/integration/ledger/test_ledger_e2e_pipeline.py`` states
   for itself, and not a boundary this file's subject sits on.

2. The subject. One terminal ``delegate-skill-completed`` envelope is published
   on a real ``EventBusInmemory`` behind a real ``wire_from_manifest`` +
   ``MessageDispatchEngine``, which constructs and invokes the REAL
   ``HandlerDelegationChainLedger``. Its ``handle()``, its ``event_ledger``
   read, ``assemble_replay_and_verify`` and its ``public.ledger_chain`` write
   are all inherited untouched; the subclass below exists only because the
   auto-wiring resolver quarantines a handler whose required ``container``
   argument it cannot supply, and it carries the test database's DSN.

The declaration is NOT supplied by this test
--------------------------------------------
``declared_chain`` is left unset, so the handler loads the topology from its
own ``contract.yaml`` — which is what makes this a regression test rather than
a restatement. On the parent commit of the OMN-18419 fix that file declares a
four-topic LINE and the replay re-derives each expected parent positionally;
the same five published hops then grade red, in both directions:

* the positive case fails — the branch hop records the HEAD as its parent and
  the line replay expects the routing decision, and the intermediate
  ``delegation-request.v1`` hop is neither declared nor read back at all;
* the negative control fails — a ``delegate-skill-completed`` mis-parented to
  the hop that precedes it in TIME is exactly what a positional re-derivation
  calls correct, so it grades GREEN there and this file asserts it is red.

Measured RED-first against parent ``76c8658f1``; the evidence is on OMN-18419.

The positive control
--------------------
The negative control is the positive control's falsifier and vice versa: both
cases publish the same five hops through the same two dispatches and differ in
one header. A change that made every hop green regardless of its recorded edge
would pass the first case and fail the second; a change that made every hop red
would fail the first. The four unrelated hops of the negative case are asserted
GREEN in the same act, so a blanket-red is not mistaken for the branch verdict.

Environment
-----------
``OMNIBASE_INFRA_DB_URL`` pointing at a database with the forward migrations
applied — the environment ci.yml's ``integration-guard`` job builds, which is
where this proof is wired. Skips when Postgres is unset or unreachable, like
the sibling real-DB proofs; with ``OMN18419_REQUIRE_PG=1`` (set by that CI
step) an unset or unreachable Postgres FAILS instead of skipping, so a runner
reading only the exit status cannot take a skip for a pass that asserted
nothing. In GitHub Actions it always runs: that job provisions its own
throwaway Postgres and exports the DSN it just built, so gating on the host the
DSN happens to resolve to would turn a routing detail into a silent skip on the
one job that owns this proof. Outside Actions it runs only under the explicit
opt-in ``OMN18419_ALLOW_LAB_PG=1``, and that opt-in is honoured only against
loopback Postgres — a non-loopback host under it FAILS before the database is
probed, so a mistyped DSN can neither reach a shared database nor read as a
pass. Every row it writes is scoped to a correlation id minted in the test and
deleted afterwards, from both relations, on failure as well as success.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar, cast
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import asyncpg
import pytest

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.handlers.handler_delegation_chain_ledger import (
    HandlerDelegationChainLedger,
)
from omnibase_infra.nodes.node_ledger_projection_compute.handlers.handler_ledger_projection import (
    HandlerLedgerProjection,
)
from omnibase_infra.nodes.node_ledger_write_effect.handlers.handler_ledger_append import (
    HandlerLedgerAppend,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)
from omnibase_infra.runtime.intent_effects import IntentEffectDispatchBridge
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from tests.helpers.util_postgres import PostgresConfig, check_postgres_reachable

pytestmark = [
    pytest.mark.integration,
    pytest.mark.postgres,
]

_postgres_config = PostgresConfig.from_env()

_REQUIRE_PG_ENV = "OMN18419_REQUIRE_PG"
REQUIRE_PG = os.environ.get(_REQUIRE_PG_ENV) == "1"
_ALLOW_LAB_PG_ENV = "OMN18419_ALLOW_LAB_PG"
ALLOW_LAB_PG = os.environ.get(_ALLOW_LAB_PG_ENV) == "1"
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

# The chain, in the order the envelopes are observed on the wire. The DECLARED
# parent of each hop is the contract's business, not this file's -- these are
# only the topics the test publishes on.
_TOPIC_DELEGATE_SKILL = "onex.cmd.omnimarket.delegate-skill.v1"
_TOPIC_DELEGATION_REQUEST = "onex.cmd.omnibase-infra.delegation-request.v1"
_TOPIC_ROUTING_REQUEST = "onex.cmd.omnibase-infra.delegation-routing-request.v1"
_TOPIC_ROUTING_DECISION = "onex.evt.omnibase-infra.routing-decision.v1"
_TOPIC_COMPLETED = "onex.evt.omnimarket.delegate-skill-completed.v1"

_PUBLISHED_TOPICS = (
    _TOPIC_DELEGATE_SKILL,
    _TOPIC_DELEGATION_REQUEST,
    _TOPIC_ROUTING_REQUEST,
    _TOPIC_ROUTING_DECISION,
    _TOPIC_COMPLETED,
)


def _postgres_available() -> bool:
    return _postgres_config.is_configured and check_postgres_reachable(
        _postgres_config,
        timeout=5.0,
    )


def _postgres_host_is_loopback() -> bool:
    return _postgres_config.host in _LOOPBACK_HOSTS


@pytest.fixture(autouse=True)
def _postgres_required_when_flagged() -> None:
    """Refuse, skip or run -- in that order, and never silently.

    An explicit opt-in that cannot be honoured is an operator error, not a
    skip: it fails before the database is probed at all.
    """
    if ALLOW_LAB_PG and not _postgres_host_is_loopback():
        pytest.fail(
            f"{_ALLOW_LAB_PG_ENV}=1 permits a non-Actions run only against loopback "
            f"Postgres; OMNIBASE_INFRA_DB_URL host {_postgres_config.host!r} is not "
            "loopback. Refusing to write event_ledger or ledger_chain rows."
        )
    if not _postgres_available():
        message = (
            "PostgreSQL not available: OMNIBASE_INFRA_DB_URL is unset, malformed, "
            "or unreachable (point it at a database migrated with "
            "scripts/run-migrations.py)."
        )
        if REQUIRE_PG:
            pytest.fail(
                f"{_REQUIRE_PG_ENV}=1 but {message} This proof fails closed rather "
                "than skipping, so a Postgres-absent run cannot read as a pass."
            )
        pytest.skip(message)
    if os.environ.get("GITHUB_ACTIONS") == "true":
        # The integration-guard job provisions its own throwaway Postgres and
        # exports the DSN it just built, so the database is this job's by
        # construction. The host it resolves to is 127.0.0.1 or the container
        # gateway depending on where the runner executes, and gating on that
        # would turn a routing detail into a silent skip on the one job that
        # is supposed to run this proof.
        return
    if not ALLOW_LAB_PG:
        pytest.skip(
            "OMN-18419 writes event_ledger and ledger_chain rows; outside GitHub "
            "Actions it runs only under the explicit lab opt-in "
            f"{_ALLOW_LAB_PG_ENV}=1, against loopback Postgres."
        )


@pytest.fixture
def postgres_dsn() -> str:
    return _postgres_config.build_dsn()


@pytest.fixture
async def db_pool(postgres_dsn: str) -> AsyncIterator[asyncpg.Pool]:
    """A readback pool that is NOT the pool the code under test writes through."""
    pool = await asyncpg.create_pool(postgres_dsn, min_size=1, max_size=2, timeout=10.0)
    try:
        yield pool
    finally:
        await pool.close()


@pytest.fixture
async def written_correlation_ids(db_pool: asyncpg.Pool) -> AsyncIterator[list[UUID]]:
    """Every correlation this test writes, deleted from both relations after."""
    correlation_ids: list[UUID] = []
    try:
        yield correlation_ids
    finally:
        if correlation_ids:
            async with db_pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM public.ledger_chain WHERE correlation_id = ANY($1::text[])",
                    [str(value) for value in correlation_ids],
                )
                await conn.execute(
                    "DELETE FROM public.event_ledger WHERE correlation_id = ANY($1::uuid[])",
                    [str(value) for value in correlation_ids],
                )


class HandlerUnderTest(HandlerDelegationChainLedger):
    """The REAL chain writer, with a DSN and nothing else changed.

    ``handle()``, ``_read_observed``, ``assemble_replay_and_verify`` and
    ``_persist_rows`` are all inherited untouched -- unlike the OMN-18398
    harness in this directory, which stubs both database edges because the
    relation is its input rather than its subject. Here both edges are the
    point. ``declared_chain`` is deliberately not passed, so the topology comes
    from the node's own contract.

    The constructor takes no required parameter because the auto-wiring
    resolver quarantines a handler whose required ``container`` argument it
    cannot supply in this harness.
    """

    dsn: ClassVar[str] = ""
    instances: ClassVar[list[HandlerUnderTest]] = []

    def __init__(self) -> None:
        super().__init__(cast("Any", object()), db_dsn=type(self).dsn)
        type(self).instances.append(self)


def _chain_writer_contract() -> ModelDiscoveredContract:
    """This node's real shape: bus-triggered effect, no publish_topics."""
    return ModelDiscoveredContract(
        name="node_delegation_chain_ledger_effect",
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path(
            "/tmp/node_delegation_chain_ledger_effect/contract.yaml"  # noqa: S108
        ),
        entry_point_name="node_delegation_chain_ledger_effect",
        package_name="omnibase_infra",
        event_bus=ModelEventBusWiring(subscribe_topics=(_TOPIC_COMPLETED,)),
        handler_routing=ModelHandlerRouting(
            routing_strategy="topic_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name="HandlerUnderTest",
                        module=__name__,
                    ),
                    event_model=ModelHandlerRef(
                        name="ModelDelegationTerminalPayload",
                        module=(
                            "omnibase_infra.nodes."
                            "node_delegation_chain_ledger_effect.models"
                        ),
                    ),
                    topic=_TOPIC_COMPLETED,
                    operation="delegation_chain.record",
                    message_category="event",
                ),
            ),
        ),
    )


async def _publish_chain_into_event_ledger(
    *,
    correlation_id: UUID,
    completed_parent_is_head: bool,
    postgres_dsn: str,
) -> dict[str, UUID]:
    """Drive the five hops through the REAL ledger-projection dispatch path.

    Returns each hop's envelope identity so the caller can assert on the edges
    the chain writer recorded WITHOUT re-reading them from the writer's own
    output. ``completed_parent_is_head`` selects the branch (the declared,
    correct edge) or the line (the mis-parented negative control).
    """
    envelope_ids = {topic: uuid4() for topic in _PUBLISHED_TOPICS}
    parents: dict[str, UUID | None] = {
        _TOPIC_DELEGATE_SKILL: None,
        _TOPIC_DELEGATION_REQUEST: envelope_ids[_TOPIC_DELEGATE_SKILL],
        _TOPIC_ROUTING_REQUEST: envelope_ids[_TOPIC_DELEGATION_REQUEST],
        _TOPIC_ROUTING_DECISION: envelope_ids[_TOPIC_ROUTING_REQUEST],
        _TOPIC_COMPLETED: (
            envelope_ids[_TOPIC_DELEGATE_SKILL]
            if completed_parent_is_head
            else envelope_ids[_TOPIC_ROUTING_DECISION]
        ),
    }

    container = MagicMock()
    append_handler = HandlerLedgerAppend(container, postgres_dsn)
    await append_handler.initialize({})
    projection_handler = HandlerLedgerProjection(container)
    bridge = IntentEffectDispatchBridge(append_handler)

    # The observed ORDER is evidence the chain writer reads back (it orders by
    # event timestamp), so each hop is stamped a second apart in the order it
    # is published rather than left to collide on NOW().
    base = datetime.now(UTC)
    try:
        for index, topic in enumerate(_PUBLISHED_TOPICS):
            headers = ModelEventHeaders(
                correlation_id=correlation_id,
                message_id=envelope_ids[topic],
                parent_message_id=parents[topic],
                event_type=topic,
                source="omn18419-ac4-dispatch-proof",
                timestamp=base + timedelta(seconds=index),
            )
            message = ModelEventMessage(
                topic=topic,
                key=str(correlation_id).encode("utf-8"),
                value=json.dumps({"hop_index": index}).encode("utf-8"),
                headers=headers,
                partition=0,
                offset=str(int(uuid4().int % (2**62))),
            )
            output = await projection_handler.handle(message)
            intent = output.result
            assert intent is not None, (
                f"the ledger projection emitted no intent for {topic!r}; the "
                "evidence this proof grades would not exist"
            )
            await bridge.execute(intent.payload, correlation_id=correlation_id)
    finally:
        await append_handler.shutdown()

    return envelope_ids


async def _dispatch_chain_writer(*, correlation_id: UUID, postgres_dsn: str) -> None:
    """Publish the terminal event through the REAL wiring and dispatch engine."""
    HandlerUnderTest.dsn = postgres_dsn
    HandlerUnderTest.instances = []

    bus = EventBusInmemory(environment="test", group="omn18419-tree-replay")
    await bus.start()
    try:
        engine = MessageDispatchEngine()
        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=HandlerUnderTest,
        ):
            await wire_from_manifest(
                ModelAutoWiringManifest(contracts=(_chain_writer_contract(),)),
                engine,
                event_bus=bus,
                environment="local",
            )
        engine.freeze()

        envelope = ModelEventEnvelope[object](
            payload={"correlation_id": str(correlation_id), "status": "completed"},
            correlation_id=correlation_id,
            event_type="omnimarket.delegate-skill-completed",
        )
        # EventBusInmemory awaits subscriber callbacks inline before returning,
        # so publish() is the synchronization point for the dispatch.
        await bus.publish(
            _TOPIC_COMPLETED,
            None,
            envelope.model_dump_json().encode("utf-8"),
            None,
        )
    finally:
        for handler in HandlerUnderTest.instances:
            await handler.shutdown()
        await bus.close()

    assert len(HandlerUnderTest.instances) == 1, (
        "the wiring did not construct exactly one chain writer, so nothing was "
        f"dispatched: {HandlerUnderTest.instances!r}"
    )


async def _read_chain_rows(
    db_pool: asyncpg.Pool, correlation_id: UUID
) -> tuple[dict[str, object], ...]:
    """Read back what the dispatch wrote, ordered as the canary reads it."""
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT hop_index, hop, replay_green, verifier_verdict,
                   envelope_id, parent_envelope_id, replay_detail
            FROM public.ledger_chain
            WHERE correlation_id = $1
            ORDER BY hop_index
            """,
            str(correlation_id),
        )
    return tuple(dict(row) for row in rows)


def _describe(rows: Sequence[dict[str, object]]) -> str:
    return "\n".join(
        f"  {row['hop_index']} {row['hop']} replay_green={row['replay_green']} "
        f"verifier={row['verifier_verdict']} parent={row['parent_envelope_id']} "
        f"detail={row['replay_detail']!r}"
        for row in rows
    )


@pytest.mark.asyncio
async def test_declared_tree_grades_green_through_the_real_dispatch(
    db_pool: asyncpg.Pool,
    postgres_dsn: str,
    written_correlation_ids: list[UUID],
) -> None:
    """Every recorded edge of the declared TREE replays green.

    The branch is the whole point: ``delegate-skill-completed`` is observed
    LAST and caused by the HEAD, so a replay that re-derives each expected
    parent from ``index - 1`` grades a correct edge red. The assertion names
    the branch edge explicitly rather than only counting greens, so a change
    that stopped reading the recorded parent at all would still fail here.
    """
    correlation_id = uuid4()
    written_correlation_ids.append(correlation_id)

    envelope_ids = await _publish_chain_into_event_ledger(
        correlation_id=correlation_id,
        completed_parent_is_head=True,
        postgres_dsn=postgres_dsn,
    )
    await _dispatch_chain_writer(
        correlation_id=correlation_id, postgres_dsn=postgres_dsn
    )

    rows = await _read_chain_rows(db_pool, correlation_id)

    assert [row["hop"] for row in rows] == list(_PUBLISHED_TOPICS), (
        "the chain writer did not record the five published hops in the order "
        f"they were observed:\n{_describe(rows)}"
    )
    assert all(row["replay_green"] for row in rows), (
        "a correct chain graded red -- every edge published here is the edge "
        f"the contract declares:\n{_describe(rows)}"
    )
    assert all(row["verifier_verdict"] == "pass" for row in rows), (
        f"tier 2 did not pass on every observed hop:\n{_describe(rows)}"
    )

    branch = next(row for row in rows if row["hop"] == _TOPIC_COMPLETED)
    assert branch["parent_envelope_id"] == str(envelope_ids[_TOPIC_DELEGATE_SKILL]), (
        "the branch hop's recorded parent is not the chain head, so this run "
        "proves nothing about the tree: "
        f"{branch['parent_envelope_id']!r}"
    )
    assert branch["parent_envelope_id"] != str(envelope_ids[_TOPIC_ROUTING_DECISION]), (
        "the branch hop's parent is the hop that precedes it in TIME, which is "
        "the line topology this proof exists to distinguish from the tree"
    )

    intermediate = next(row for row in rows if row["hop"] == _TOPIC_DELEGATION_REQUEST)
    assert intermediate["parent_envelope_id"] == str(
        envelope_ids[_TOPIC_DELEGATE_SKILL]
    ), (
        "the delegation-request hop was not read back with the head as its "
        f"parent:\n{_describe(rows)}"
    )


@pytest.mark.asyncio
async def test_a_mis_parented_hop_grades_red_and_its_siblings_do_not(
    db_pool: asyncpg.Pool,
    postgres_dsn: str,
    written_correlation_ids: list[UUID],
) -> None:
    """The negative control: the LINE answer for the branch hop is refused.

    This chain differs from the green one by a single header --
    ``delegate-skill-completed`` records the routing decision, the hop that
    precedes it in time, instead of the head that caused it. A positional
    re-derivation calls that correct, which is precisely the defect. The four
    unrelated hops are asserted GREEN in the same act so a blanket red cannot
    be mistaken for this verdict.
    """
    correlation_id = uuid4()
    written_correlation_ids.append(correlation_id)

    envelope_ids = await _publish_chain_into_event_ledger(
        correlation_id=correlation_id,
        completed_parent_is_head=False,
        postgres_dsn=postgres_dsn,
    )
    await _dispatch_chain_writer(
        correlation_id=correlation_id, postgres_dsn=postgres_dsn
    )

    rows = await _read_chain_rows(db_pool, correlation_id)
    assert [row["hop"] for row in rows] == list(_PUBLISHED_TOPICS), (
        f"the chain writer did not record the five published hops:\n{_describe(rows)}"
    )

    mis_parented = next(row for row in rows if row["hop"] == _TOPIC_COMPLETED)
    assert mis_parented["parent_envelope_id"] == str(
        envelope_ids[_TOPIC_ROUTING_DECISION]
    ), (
        "the negative control did not record the edge it set out to record, so "
        f"its verdict is about something else:\n{_describe(rows)}"
    )
    assert not mis_parented["replay_green"], (
        "a hop whose recorded parent is NOT the parent its contract declares "
        "graded green -- the recorded edge is being re-derived from position "
        f"rather than from the declaration:\n{_describe(rows)}"
    )
    assert _TOPIC_DELEGATE_SKILL in str(mis_parented["replay_detail"]), (
        "the red carries no statement of which declared parent it expected: "
        f"{mis_parented['replay_detail']!r}"
    )

    siblings = [row for row in rows if row["hop"] != _TOPIC_COMPLETED]
    assert len(siblings) == 4, f"expected four sibling hops:\n{_describe(rows)}"
    assert all(row["replay_green"] for row in siblings), (
        "one mis-parented hop turned the whole chain red, so the red above is "
        f"not evidence about that hop:\n{_describe(rows)}"
    )


# ---------------------------------------------------------------------------
# OMN-18916 AC4: the observed chain is legitimately LONGER than five hops
#
# Two causes, both measured on the .201 dev lane, both graded red before this
# change because tier 2 compared observed index i against declared index i:
#
#   RETRY       the attempts ladder climbs a rung, so a second routing
#               request is published with its OWN envelope and its own
#               correct parent. A real hop that must be kept.
#   REDELIVERY  the identical envelope arrives twice at adjacent offsets.
#               One hop projected twice, which must be collapsed.
#
# Driven through the same real projection-and-dispatch path the OMN-18419
# proofs above use, so what is graded is evidence this file produced rather
# than ledger_chain rows it wrote by hand.
# ---------------------------------------------------------------------------


async def _publish_longer_chain_into_event_ledger(
    *,
    correlation_id: UUID,
    postgres_dsn: str,
    redeliver_instead_of_retry: bool,
) -> dict[str, UUID]:
    """Publish a chain longer than the declaration, one way or the other.

    With ``redeliver_instead_of_retry`` the extra pair carries the SAME
    envelope ids as the first pair, which is a redelivery. Without it the
    extra pair carries fresh ids and correct parents, which is a retry. The
    two shapes differ in exactly one respect, so a fix that conflated them
    cannot pass both cases.
    """
    first = {topic: uuid4() for topic in _PUBLISHED_TOPICS}
    retry_request = (
        first[_TOPIC_ROUTING_REQUEST] if redeliver_instead_of_retry else uuid4()
    )
    retry_decision = (
        first[_TOPIC_ROUTING_DECISION] if redeliver_instead_of_retry else uuid4()
    )

    # head, request, routing round 1, routing round 2, terminal off the head.
    sequence: list[tuple[str, UUID, UUID | None]] = [
        (_TOPIC_DELEGATE_SKILL, first[_TOPIC_DELEGATE_SKILL], None),
        (
            _TOPIC_DELEGATION_REQUEST,
            first[_TOPIC_DELEGATION_REQUEST],
            first[_TOPIC_DELEGATE_SKILL],
        ),
        (
            _TOPIC_ROUTING_REQUEST,
            first[_TOPIC_ROUTING_REQUEST],
            first[_TOPIC_DELEGATION_REQUEST],
        ),
        (
            _TOPIC_ROUTING_DECISION,
            first[_TOPIC_ROUTING_DECISION],
            first[_TOPIC_ROUTING_REQUEST],
        ),
        (_TOPIC_ROUTING_REQUEST, retry_request, first[_TOPIC_DELEGATION_REQUEST]),
        (_TOPIC_ROUTING_DECISION, retry_decision, retry_request),
        (_TOPIC_COMPLETED, first[_TOPIC_COMPLETED], first[_TOPIC_DELEGATE_SKILL]),
    ]

    container = MagicMock()
    append_handler = HandlerLedgerAppend(container, postgres_dsn)
    await append_handler.initialize({})
    projection_handler = HandlerLedgerProjection(container)
    bridge = IntentEffectDispatchBridge(append_handler)

    base = datetime.now(UTC)
    try:
        for index, (topic, envelope_id, parent) in enumerate(sequence):
            headers = ModelEventHeaders(
                correlation_id=correlation_id,
                message_id=envelope_id,
                parent_message_id=parent,
                event_type=topic,
                source="omn18916-ac4-dispatch-proof",
                timestamp=base + timedelta(seconds=index),
            )
            message = ModelEventMessage(
                topic=topic,
                key=str(correlation_id).encode("utf-8"),
                value=json.dumps({"hop_index": index}).encode("utf-8"),
                headers=headers,
                partition=0,
                offset=str(int(uuid4().int % (2**62))),
            )
            output = await projection_handler.handle(message)
            intent = output.result
            assert intent is not None, (
                f"the ledger projection emitted no intent for {topic!r}; the "
                "evidence this proof grades would not exist"
            )
            await bridge.execute(intent.payload, correlation_id=correlation_id)
    finally:
        await append_handler.shutdown()

    return first


@pytest.mark.asyncio
async def test_a_retried_chain_grades_green_through_the_real_dispatch(
    db_pool: asyncpg.Pool,
    postgres_dsn: str,
    written_correlation_ids: list[UUID],
) -> None:
    """AC1/AC4. Seven observed hops, every edge correct, nothing red.

    On the parent commit the two hops of the second routing round and the
    terminal after them are compared against the wrong declared positions,
    so a chain in which nothing is wrong grades fail and skip.
    """
    correlation_id = uuid4()
    written_correlation_ids.append(correlation_id)

    await _publish_longer_chain_into_event_ledger(
        correlation_id=correlation_id,
        postgres_dsn=postgres_dsn,
        redeliver_instead_of_retry=False,
    )
    await _dispatch_chain_writer(
        correlation_id=correlation_id, postgres_dsn=postgres_dsn
    )

    rows = await _read_chain_rows(db_pool, correlation_id)

    assert len(rows) == 7, (
        "a retry is a real second attempt and must be kept as its own hop; "
        f"collapsing it would erase an attempt that happened:\n{_describe(rows)}"
    )
    assert all(row["replay_green"] for row in rows), (
        f"a causally correct retried chain graded red:\n{_describe(rows)}"
    )
    assert all(row["verifier_verdict"] == "pass" for row in rows), (
        "tier 2 graded a legitimate retry as fail or skip, which is the "
        f"positional defect this ticket removes:\n{_describe(rows)}"
    )


@pytest.mark.asyncio
async def test_a_redelivered_hop_is_recorded_once_through_the_real_dispatch(
    db_pool: asyncpg.Pool,
    postgres_dsn: str,
    written_correlation_ids: list[UUID],
) -> None:
    """AC2/AC4. The same envelope twice is one hop, not two.

    The contrast with the retry proof above is the whole assertion: identical
    input shape, identical hop count on the wire, and a different row count
    in ``ledger_chain`` because the envelope ids differ in one case and not
    the other.
    """
    correlation_id = uuid4()
    written_correlation_ids.append(correlation_id)

    await _publish_longer_chain_into_event_ledger(
        correlation_id=correlation_id,
        postgres_dsn=postgres_dsn,
        redeliver_instead_of_retry=True,
    )
    await _dispatch_chain_writer(
        correlation_id=correlation_id, postgres_dsn=postgres_dsn
    )

    rows = await _read_chain_rows(db_pool, correlation_id)

    assert len(rows) == 5, (
        "seven hops were published but two were byte-identical redeliveries, "
        f"so five distinct envelopes were observed:\n{_describe(rows)}"
    )
    assert [row["hop_index"] for row in rows] == [0, 1, 2, 3, 4], (
        "hop_index must stay dense after a collapse, or the canary's own "
        f"ORDER BY hop_index reads the gap as a missing hop:\n{_describe(rows)}"
    )
    assert all(row["replay_green"] for row in rows), _describe(rows)
    assert all(row["verifier_verdict"] == "pass" for row in rows), _describe(rows)


# ---------------------------------------------------------------------------
# OMN-18964: a RE-ROUTED delegation grades green through the real dispatch.
#
# The orchestrator issues a repeat routing request while consuming a failing
# quality-gate-result or a retryable inference-response, and records that
# envelope as its parent. Both parents are published here through the same
# real projection path, so the chain writer reads them out of event_ledger
# exactly as it does on the lane. Measured on the .201 dev lane before the
# fix: every re-routed chain replayed red, every other chain green.
# ---------------------------------------------------------------------------

_TOPIC_QUALITY_GATE_RESULT = "onex.evt.omnibase-infra.quality-gate-result.v1"
_TOPIC_INFERENCE_RESPONSE = "onex.evt.omnibase-infra.inference-response.v1"


async def _publish_sequence_into_event_ledger(
    *,
    correlation_id: UUID,
    postgres_dsn: str,
    sequence: Sequence[tuple[str, UUID, UUID | None]],
) -> None:
    """Drive an explicit (topic, envelope, parent) sequence through the projection."""
    container = MagicMock()
    append_handler = HandlerLedgerAppend(container, postgres_dsn)
    await append_handler.initialize({})
    projection_handler = HandlerLedgerProjection(container)
    bridge = IntentEffectDispatchBridge(append_handler)

    base = datetime.now(UTC)
    try:
        for index, (topic, envelope_id, parent) in enumerate(sequence):
            headers = ModelEventHeaders(
                correlation_id=correlation_id,
                message_id=envelope_id,
                parent_message_id=parent,
                event_type=topic,
                source="omn18964-reroute-dispatch-proof",
                timestamp=base + timedelta(seconds=index),
            )
            message = ModelEventMessage(
                topic=topic,
                key=str(correlation_id).encode("utf-8"),
                value=json.dumps({"hop_index": index}).encode("utf-8"),
                headers=headers,
                partition=0,
                offset=str(int(uuid4().int % (2**62))),
            )
            output = await projection_handler.handle(message)
            intent = output.result
            assert intent is not None, (
                f"the ledger projection emitted no intent for {topic!r}; the "
                "evidence this proof grades would not exist"
            )
            await bridge.execute(intent.payload, correlation_id=correlation_id)
    finally:
        await append_handler.shutdown()


def _rerouted_sequence(
    *, publish_reroute_parents: bool
) -> list[tuple[str, UUID, UUID | None]]:
    """Head, request, round 1, a gate re-route, an inference re-route, terminal."""
    ids = {
        name: uuid4()
        for name in ("skill", "req", "rr1", "rd1", "qg", "rr2", "ir", "rr3")
    }
    terminal = uuid4()
    sequence: list[tuple[str, UUID, UUID | None]] = [
        (_TOPIC_DELEGATE_SKILL, ids["skill"], None),
        (_TOPIC_DELEGATION_REQUEST, ids["req"], ids["skill"]),
        (_TOPIC_ROUTING_REQUEST, ids["rr1"], ids["req"]),
        (_TOPIC_ROUTING_DECISION, ids["rd1"], ids["rr1"]),
    ]
    if publish_reroute_parents:
        # Its own parent (the gate request) is never projected or graded.
        sequence.append((_TOPIC_QUALITY_GATE_RESULT, ids["qg"], uuid4()))
    sequence.append((_TOPIC_ROUTING_REQUEST, ids["rr2"], ids["qg"]))
    if publish_reroute_parents:
        sequence.append((_TOPIC_INFERENCE_RESPONSE, ids["ir"], uuid4()))
    sequence.append((_TOPIC_ROUTING_REQUEST, ids["rr3"], ids["ir"]))
    sequence.append((_TOPIC_COMPLETED, terminal, ids["skill"]))
    return sequence


@pytest.mark.asyncio
async def test_a_rerouted_chain_grades_green_through_the_real_dispatch(
    db_pool: asyncpg.Pool,
    postgres_dsn: str,
    written_correlation_ids: list[UUID],
) -> None:
    """OMN-18964. Both re-route kinds close; the parent evidence is not a row."""
    correlation_id = uuid4()
    written_correlation_ids.append(correlation_id)

    await _publish_sequence_into_event_ledger(
        correlation_id=correlation_id,
        postgres_dsn=postgres_dsn,
        sequence=_rerouted_sequence(publish_reroute_parents=True),
    )
    await _dispatch_chain_writer(
        correlation_id=correlation_id, postgres_dsn=postgres_dsn
    )

    rows = await _read_chain_rows(db_pool, correlation_id)
    assert [row["hop"] for row in rows] == [
        _TOPIC_DELEGATE_SKILL,
        _TOPIC_DELEGATION_REQUEST,
        _TOPIC_ROUTING_REQUEST,
        _TOPIC_ROUTING_DECISION,
        _TOPIC_ROUTING_REQUEST,
        _TOPIC_ROUTING_REQUEST,
        _TOPIC_COMPLETED,
    ], f"re-route parent evidence must not be written as a hop:\n{_describe(rows)}"
    assert all(row["replay_green"] for row in rows), (
        f"a causally correct re-routed chain graded red:\n{_describe(rows)}"
    )
    assert all(row["verifier_verdict"] == "pass" for row in rows), _describe(rows)


@pytest.mark.asyncio
async def test_a_reroute_whose_parent_was_never_projected_grades_red(
    db_pool: asyncpg.Pool,
    postgres_dsn: str,
    written_correlation_ids: list[UUID],
) -> None:
    """Negative control: the declaration alone does not green a re-route."""
    correlation_id = uuid4()
    written_correlation_ids.append(correlation_id)

    await _publish_sequence_into_event_ledger(
        correlation_id=correlation_id,
        postgres_dsn=postgres_dsn,
        sequence=_rerouted_sequence(publish_reroute_parents=False),
    )
    await _dispatch_chain_writer(
        correlation_id=correlation_id, postgres_dsn=postgres_dsn
    )

    rows = await _read_chain_rows(db_pool, correlation_id)
    greens = [row["replay_green"] for row in rows]
    assert greens == [True, True, True, True, False, False, True], (
        f"only the two re-routes may be red without their parents:\n{_describe(rows)}"
    )
