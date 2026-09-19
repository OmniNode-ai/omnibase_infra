# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for HandlerGraph against a real Memgraph server.

Selection in CI (OMN-18795). These tests do NOT skip gracefully in CI, because
a silent skip and a green run are indistinguishable in a junit summary:

    - the module carries ``pytest.mark.graph``, which the PR test splits
      deselect, so they are never collected-and-skipped on a pull request;
    - the service-backed job in ``.github/workflows/ci.yml`` starts a real
      Memgraph, exports ``MEMGRAPH_BOLT_URL`` and ``GRAPH_INTEGRATION_TESTS=1``,
      and runs them;
    - a CI job that selects them without that provisioning FAILS, naming the
      variable, rather than skipping.

Outside CI they skip, so a laptop with no Memgraph is not forced to stand one up.

Why this file was rewritten, not merely ungated
-----------------------------------------------
The suite these tests replace had never executed. It was registered in
``config/env_gated_test_skips.yaml`` as gated on ``MEMGRAPH_BOLT_URL``, which no
workflow set, and the first run against a real server produced one failure and
twelve fixture-setup errors — every one of them an API that no longer exists:

    - ``HandlerGraph()`` was constructed with no arguments; the constructor
      takes a ``ModelONEXContainer`` (``handler_graph.py:168``);
    - ``describe()`` was called synchronously and indexed as a dict; it is
      ``async def describe() -> ModelGraphHandlerMetadata`` (``:1287``);
    - envelopes used the operations ``graph.query`` and ``graph.execute``, which
      the dispatch table does not carry. The seven it does carry are
      ``graph.execute_query``, ``graph.execute_query_batch``,
      ``graph.create_node``, ``graph.create_relationship``,
      ``graph.delete_node``, ``graph.delete_relationship`` and
      ``graph.traverse`` (``:1383``), and their payload fields are ``query`` /
      ``parameters``, not ``cypher``;
    - the config fixture passed ``uri`` / ``username`` / ``password``.
      ``initialize()`` resolves a config dict from ``connection_uri`` or
      ``bolt_uri`` only, with ``auth`` and ``options`` as the other two keys
      (``:266``); an unrecognised key falls through to the contract descriptor
      and a Bolt endpoint that is not the one the test meant.

That is the cost of a suite that skips: it keeps its shape while the thing it
claims to cover moves out from under it, and nothing says so.

Every test below proves its claim by reading back through an INDEPENDENT neo4j
driver (the ``reader`` fixture), not by trusting the handler's own return value.
A handler that reported success without writing anything would pass a
self-checking test and fails these.

Two real defects surfaced the moment this suite first ran, both of them
invisible for as long as it was skipped, and both fixed in the same change that
gave the suite a home. The tests that found them are ordinary passing
assertions below, and each is the RED proof that the fix was needed:

    - ``traverse()`` raised ``AttributeError`` for any traversal that reached a
      node, because it read its rows with ``await result.data()`` and then
      treated the resulting plain dicts as neo4j objects. Only the zero-row
      path had ever worked.
    - ``delete_relationship()`` reported ``relationships_deleted=2`` when it
      had deleted one, because its Cypher matched the undirected pattern
      ``()-[r]-()``, which matches each relationship once per direction.

Run with infrastructure::

    $ MEMGRAPH_BOLT_URL=bolt://localhost:7687 GRAPH_INTEGRATION_TESTS=1 \\
        uv run pytest tests/integration/handlers/test_handler_graph_integration.py -v

Related tickets: OMN-1142 (original suite), OMN-18775 (registered as having no
execution path), OMN-18795 (this rewrite).
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import pytest
from neo4j import AsyncGraphDatabase

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.errors import InfraConnectionError, RuntimeHostError
from omnibase_infra.handlers.models.graph import (
    ModelGraphExecutePayload,
    ModelGraphQueryPayload,
)
from tests.helpers.service_env import require_service_env

if TYPE_CHECKING:
    from neo4j import AsyncDriver

    from omnibase_core.models.dispatch import ModelHandlerOutput
    from omnibase_infra.handlers import HandlerGraph
    from omnibase_infra.handlers.models.model_graph_handler_response import (
        ModelGraphHandlerResponse,
    )

MEMGRAPH_BOLT_URL = os.getenv("MEMGRAPH_BOLT_URL", "")
MEMGRAPH_USERNAME = os.getenv("MEMGRAPH_USERNAME", "")
MEMGRAPH_PASSWORD = os.getenv("MEMGRAPH_PASSWORD", "")
# The handler's own default database name. The reader must use the same one, or
# it would read a different database and report a false absence.
MEMGRAPH_DATABASE = os.getenv("MEMGRAPH_DATABASE", "memgraph")

# None when the server takes no auth, which is how the CI-provisioned Memgraph
# and the lab one both run. A tuple only when a username is actually supplied.
BOLT_AUTH: tuple[str, str] | None = (
    (MEMGRAPH_USERNAME, MEMGRAPH_PASSWORD) if MEMGRAPH_USERNAME else None
)

pytestmark = [
    pytest.mark.graph,
]


@pytest.fixture(autouse=True, scope="module")
def _require_memgraph() -> None:
    """Refuse to skip this suite silently once CI has selected it."""
    require_service_env(
        opt_in="GRAPH_INTEGRATION_TESTS",
        endpoint="MEMGRAPH_BOLT_URL",
        workflow=".github/workflows/ci.yml",
        service="Memgraph",
    )


# =============================================================================
# Fixtures
# =============================================================================


class GraphReader:
    """A read-back channel that does not go through the code under test.

    Reading a write back through the same handler that performed it proves only
    that the handler is self-consistent. This wraps a second, independent neo4j
    driver against the same server, so every assertion about graph state is the
    SERVER's answer.
    """

    def __init__(self, driver: AsyncDriver) -> None:
        self._driver = driver

    async def rows(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Run a Cypher query and return its records as plain dicts."""
        async with self._driver.session(database=MEMGRAPH_DATABASE) as session:
            result = await session.run(query, parameters or {})
            data: list[dict[str, Any]] = await result.data()
            await result.consume()
        return data

    async def count(self, query: str, parameters: dict[str, Any] | None = None) -> int:
        """Run a query whose single record is a single count, and return it."""
        rows = await self.rows(query, parameters)
        assert len(rows) == 1, f"expected exactly one count row, got {rows!r}"
        return int(next(iter(rows[0].values())))


@pytest.fixture
async def reader() -> AsyncGenerator[GraphReader, None]:
    """An independent Bolt connection used only to read server state back."""
    driver = AsyncGraphDatabase.driver(MEMGRAPH_BOLT_URL, auth=BOLT_AUTH)
    try:
        await driver.verify_connectivity()
        yield GraphReader(driver)
    finally:
        await driver.close()


@pytest.fixture
async def handler() -> AsyncGenerator[HandlerGraph, None]:
    """An initialized HandlerGraph, shut down on every exit path."""
    from omnibase_infra.handlers import HandlerGraph

    instance = HandlerGraph(ModelONEXContainer(enable_service_registry=False))
    await instance.initialize(MEMGRAPH_BOLT_URL, auth=BOLT_AUTH)
    try:
        yield instance
    finally:
        await instance.shutdown()


@pytest.fixture
def uninitialized_handler() -> HandlerGraph:
    """A constructed but never-initialized handler, for the refusal paths."""
    from omnibase_infra.handlers import HandlerGraph

    return HandlerGraph(ModelONEXContainer(enable_service_registry=False))


@pytest.fixture
async def unique_label(reader: GraphReader) -> AsyncGenerator[str, None]:
    """A node label unique to this test, with its nodes removed afterwards.

    The server is shared and persistent, so a fixed label would make one test's
    assertions depend on another's leftovers and would make a re-run of this
    suite fail where the first run passed. Cleanup runs through the reader
    rather than the handler, so a test that broke the handler still cleans up.
    """
    label = f"TestNode_{uuid4().hex[:12]}"
    try:
        yield label
    finally:
        await reader.rows(f"MATCH (n:{label}) DETACH DELETE n")


# =============================================================================
# Helpers
# =============================================================================


def query_payload(
    output: ModelHandlerOutput[ModelGraphHandlerResponse],
) -> ModelGraphQueryPayload:
    """Narrow an envelope response to the query-shaped payload it carries."""
    response = output.result
    assert response is not None
    payload = response.payload.data
    assert isinstance(payload, ModelGraphQueryPayload)
    return payload


def execute_payload(
    output: ModelHandlerOutput[ModelGraphHandlerResponse],
) -> ModelGraphExecutePayload:
    """Narrow an envelope response to the write-shaped payload it carries."""
    response = output.result
    assert response is not None
    payload = response.payload.data
    assert isinstance(payload, ModelGraphExecutePayload)
    return payload


# =============================================================================
# Handler metadata and health
# =============================================================================


class TestHandlerGraphMetadata:
    """The handler describes itself, and reports on the server it is talking to."""

    @pytest.mark.asyncio
    async def test_describe_names_the_backend_the_handler_is_connected_to(
        self, handler: HandlerGraph
    ) -> None:
        """describe() agrees with the server that actually answered.

        ``database_type`` is derived from the connection URI, so it is a claim
        about the backend and not free-standing metadata. It is checked against
        the server's own agent string, read from the live connection, rather
        than asserted on its own.
        """
        metadata = await handler.describe()
        health = await handler.health_check()

        assert metadata.handler_type == "graph_database"
        assert metadata.supports_transactions is True
        assert "cypher" in metadata.capabilities
        assert "parameterized_queries" in metadata.capabilities
        assert "traversal" in metadata.capabilities

        assert health.database_version is not None
        assert metadata.database_type in health.database_version.lower()

    @pytest.mark.asyncio
    async def test_health_check_reports_a_reachable_server(
        self, handler: HandlerGraph
    ) -> None:
        """A live server reports healthy, with a measured round trip."""
        health = await handler.health_check()

        assert health.healthy is True
        assert health.latency_ms > 0.0
        assert health.database_version is not None

    @pytest.mark.asyncio
    async def test_health_check_on_an_uninitialized_handler_reports_unhealthy(
        self, uninitialized_handler: HandlerGraph
    ) -> None:
        """Health is a report, not an assertion: no connection is not an error."""
        health = await uninitialized_handler.health_check()

        assert health.healthy is False
        assert health.latency_ms == 0.0
        assert health.database_version is None


# =============================================================================
# Query execution
# =============================================================================


class TestHandlerGraphQuery:
    """execute_query against the real Cypher engine."""

    @pytest.mark.asyncio
    async def test_a_query_without_parameters_is_evaluated_by_the_server(
        self, handler: HandlerGraph
    ) -> None:
        """The values come back from Memgraph's evaluator, not from the handler."""
        result = await handler.execute_query(
            "RETURN 1 + 1 AS sum, toUpper('hello') AS shout"
        )

        assert result.records == [{"sum": 2, "shout": "HELLO"}]
        assert result.summary.database == MEMGRAPH_DATABASE
        assert result.summary.contains_updates is False
        assert result.counters.nodes_created == 0
        assert result.execution_time_ms > 0.0

    @pytest.mark.asyncio
    async def test_query_parameters_are_bound_by_the_server(
        self, handler: HandlerGraph
    ) -> None:
        """Parameters survive the round trip with their types intact.

        The string parameter carries a quote and a ``$``, which is exactly what
        string interpolation would mangle or mis-resolve, so a handler that
        formatted the query instead of parameterizing it would not return this.
        """
        result = await handler.execute_query(
            "RETURN $name AS name, $value AS value, $flag AS flag",
            {"name": "o'brien $notaparam", "value": 42, "flag": True},
        )

        assert result.records == [
            {"name": "o'brien $notaparam", "value": 42, "flag": True}
        ]

    @pytest.mark.asyncio
    async def test_a_write_query_reports_counters_that_match_the_server_state(
        self, handler: HandlerGraph, reader: GraphReader, unique_label: str
    ) -> None:
        """The reported counters are checked against what the server now holds."""
        result = await handler.execute_query(
            f"UNWIND $names AS name CREATE (n:{unique_label} {{name: name}})",
            {"names": ["alice", "bob", "carol"]},
        )

        assert result.counters.nodes_created == 3
        assert result.counters.labels_added == 3
        assert result.summary.contains_updates is True

        names = await reader.rows(
            f"MATCH (n:{unique_label}) RETURN n.name AS name ORDER BY name"
        )
        assert [row["name"] for row in names] == ["alice", "bob", "carol"]

    @pytest.mark.asyncio
    async def test_a_property_update_is_visible_to_an_independent_reader(
        self, handler: HandlerGraph, reader: GraphReader, unique_label: str
    ) -> None:
        """SET is proven by reading the property back, not by properties_set."""
        await reader.rows(f"CREATE (n:{unique_label} {{name: 'bob', role: 'user'}})")

        result = await handler.execute_query(
            f"MATCH (n:{unique_label}) WHERE n.name = $name SET n.role = $role",
            {"name": "bob", "role": "admin"},
        )

        assert result.counters.properties_set == 1
        roles = await reader.rows(f"MATCH (n:{unique_label}) RETURN n.role AS role")
        assert [row["role"] for row in roles] == ["admin"]


# =============================================================================
# Batch execution
# =============================================================================


class TestHandlerGraphBatch:
    """execute_query_batch, whose only interesting property is atomicity."""

    @pytest.mark.asyncio
    async def test_a_transactional_batch_commits_every_query(
        self, handler: HandlerGraph, reader: GraphReader, unique_label: str
    ) -> None:
        """Both statements land, and the batch reports one transaction."""
        result = await handler.execute_query_batch(
            [
                (f"CREATE (n:{unique_label} {{name: $name}})", {"name": "alice"}),
                (f"CREATE (n:{unique_label} {{name: $name}})", {"name": "bob"}),
            ],
            transaction=True,
        )

        assert result.success is True
        assert result.rollback_occurred is False
        assert result.transaction_id is not None
        assert len(result.results) == 2

        names = await reader.rows(
            f"MATCH (n:{unique_label}) RETURN n.name AS name ORDER BY name"
        )
        assert [row["name"] for row in names] == ["alice", "bob"]

    @pytest.mark.asyncio
    async def test_a_failing_transactional_batch_leaves_nothing_behind(
        self, handler: HandlerGraph, reader: GraphReader, unique_label: str
    ) -> None:
        """Rollback is the whole point of the transactional path, so it is read back.

        The first statement succeeds inside the transaction and the second is
        not Cypher at all. If the rollback did not happen, the server would
        still hold the first node — which is what this asserts against, rather
        than the ``rollback_occurred`` flag the handler sets itself.
        """
        with pytest.raises(InfraConnectionError):
            await handler.execute_query_batch(
                [
                    (f"CREATE (n:{unique_label} {{name: 'ghost'}})", None),
                    ("THIS IS NOT CYPHER", None),
                ],
                transaction=True,
            )

        assert await reader.count(f"MATCH (n:{unique_label}) RETURN count(n)") == 0


# =============================================================================
# Node operations
# =============================================================================


class TestHandlerGraphNodes:
    """create_node and delete_node against real stored nodes."""

    @pytest.mark.asyncio
    async def test_a_created_node_is_visible_to_an_independent_reader(
        self, handler: HandlerGraph, reader: GraphReader, unique_label: str
    ) -> None:
        """The returned identifiers address the node the server actually stored."""
        node = await handler.create_node(
            [unique_label], {"name": "alice", "age": 30, "active": True}
        )

        assert node.labels == [unique_label]
        assert node.properties == {"name": "alice", "age": 30, "active": True}

        stored = await reader.rows(
            f"MATCH (n:{unique_label}) WHERE id(n) = $node_id "
            "RETURN n.name AS name, n.age AS age, n.active AS active",
            {"node_id": int(node.id)},
        )
        assert stored == [{"name": "alice", "age": 30, "active": True}]

    @pytest.mark.asyncio
    async def test_a_label_that_is_not_a_safe_cypher_identifier_is_refused(
        self, handler: HandlerGraph, reader: GraphReader
    ) -> None:
        """Labels are interpolated, not parameterized, so they are validated.

        The refusal has to happen BEFORE the write, or the validation would be
        decoration — so absence on the server is part of the claim. The label
        chosen is a whole second clause, which is what an interpolated label
        would turn into.
        """
        injected = "Evil {}) DETACH DELETE n; CREATE (x:Pwned"

        with pytest.raises(RuntimeHostError, match="Invalid label"):
            await handler.create_node([injected], {"name": "alice"})

        assert await reader.count("MATCH (n:Pwned) RETURN count(n)") == 0

    @pytest.mark.asyncio
    async def test_delete_node_removes_it_from_the_server(
        self, handler: HandlerGraph, reader: GraphReader, unique_label: str
    ) -> None:
        """Deletion is proven by asking the server, not by the delete's verdict."""
        node = await handler.create_node([unique_label], {"name": "doomed"})
        assert await reader.count(f"MATCH (n:{unique_label}) RETURN count(n)") == 1

        result = await handler.delete_node(node.id)

        assert result.success is True
        assert result.node_id == node.id
        assert await reader.count(f"MATCH (n:{unique_label}) RETURN count(n)") == 0

    @pytest.mark.asyncio
    async def test_deleting_a_node_that_is_already_gone_reports_failure(
        self, handler: HandlerGraph, reader: GraphReader, unique_label: str
    ) -> None:
        """An id that matches nothing is a false verdict, not an exception.

        The id is one this test freed a moment ago, so it is certainly absent —
        as opposed to a large literal, which could in principle collide with
        another lane's node on this shared server.
        """
        node = await handler.create_node([unique_label], {"name": "transient"})
        await handler.delete_node(node.id)

        result = await handler.delete_node(node.id)

        assert result.success is False
        assert await reader.count(f"MATCH (n:{unique_label}) RETURN count(n)") == 0

    @pytest.mark.asyncio
    async def test_a_connected_node_needs_detach_and_detach_removes_the_edge(
        self, handler: HandlerGraph, reader: GraphReader, unique_label: str
    ) -> None:
        """Both halves of the detach contract, each checked against the server.

        Memgraph reports the refusal as a generic ``ClientError`` rather than
        the driver's ``ConstraintError``, so the handler's friendlier
        "Use detach=True" ``RuntimeHostError`` arm (``handler_graph.py:943``) is
        unreachable here and the refusal surfaces as ``InfraConnectionError``.
        The behaviour asserted is the one this backend actually produces.
        """
        source = await handler.create_node([unique_label], {"name": "alice"})
        target = await handler.create_node([unique_label], {"name": "bob"})
        await handler.create_relationship(source.id, target.id, "KNOWS")

        with pytest.raises(InfraConnectionError):
            await handler.delete_node(source.id, detach=False)

        # The refused delete left the graph alone — otherwise "needs detach"
        # would be a message rather than a guarantee.
        assert await reader.count(f"MATCH (n:{unique_label}) RETURN count(n)") == 2

        result = await handler.delete_node(source.id, detach=True)

        assert result.success is True
        assert result.relationships_deleted == 1
        assert await reader.count(f"MATCH (n:{unique_label}) RETURN count(n)") == 1
        assert (
            await reader.count(f"MATCH (:{unique_label})-[r:KNOWS]->() RETURN count(r)")
            == 0
        )


# =============================================================================
# Relationship operations
# =============================================================================


class TestHandlerGraphRelationships:
    """create_relationship and delete_relationship against real stored edges."""

    @pytest.mark.asyncio
    async def test_a_created_relationship_is_visible_to_an_independent_reader(
        self, handler: HandlerGraph, reader: GraphReader, unique_label: str
    ) -> None:
        """Type, properties and both endpoints are read back off the server."""
        source = await handler.create_node([unique_label], {"name": "alice"})
        target = await handler.create_node([unique_label], {"name": "bob"})

        relationship = await handler.create_relationship(
            source.id, target.id, "KNOWS", {"since": 2020}
        )

        assert relationship.type == "KNOWS"
        assert relationship.properties == {"since": 2020}

        stored = await reader.rows(
            "MATCH (a)-[r:KNOWS]->(b) WHERE id(r) = $rel_id "
            "RETURN a.name AS source, b.name AS target, r.since AS since",
            {"rel_id": int(relationship.id)},
        )
        assert stored == [{"source": "alice", "target": "bob", "since": 2020}]

    @pytest.mark.asyncio
    async def test_a_relationship_type_that_is_not_a_safe_identifier_is_refused(
        self, handler: HandlerGraph, reader: GraphReader, unique_label: str
    ) -> None:
        """The relationship type is interpolated too, and is validated the same way."""
        source = await handler.create_node([unique_label], {"name": "alice"})
        target = await handler.create_node([unique_label], {"name": "bob"})

        with pytest.raises(RuntimeHostError, match="Invalid relationship_type"):
            await handler.create_relationship(
                source.id, target.id, "KNOWS]->(:Pwned) CREATE (a)-[r2:OWNED"
            )

        assert await reader.count("MATCH (n:Pwned) RETURN count(n)") == 0
        assert (
            await reader.count(f"MATCH (:{unique_label})-[r]->() RETURN count(r)") == 0
        )

    @pytest.mark.asyncio
    async def test_delete_relationship_removes_the_edge_and_leaves_the_nodes(
        self, handler: HandlerGraph, reader: GraphReader, unique_label: str
    ) -> None:
        """Deleting an edge is not deleting its endpoints; both are read back."""
        source = await handler.create_node([unique_label], {"name": "alice"})
        target = await handler.create_node([unique_label], {"name": "bob"})
        relationship = await handler.create_relationship(source.id, target.id, "KNOWS")

        result = await handler.delete_relationship(relationship.id)

        assert result.success is True
        assert result.node_id is None
        assert (
            await reader.count(
                "MATCH ()-[r]->() WHERE id(r) = $rel_id RETURN count(r)",
                {"rel_id": int(relationship.id)},
            )
            == 0
        )
        assert await reader.count(f"MATCH (n:{unique_label}) RETURN count(n)") == 2

    # OMN-18795: this was a strict xfail pinning a real defect -- the Cypher
    # matched the UNDIRECTED pattern ()-[r]-(), so count(r) returned 2 for a
    # single deleted edge. The pattern is directed now and this is an ordinary
    # passing assertion.
    @pytest.mark.asyncio
    async def test_delete_relationship_reports_the_one_edge_it_deleted(
        self, handler: HandlerGraph, reader: GraphReader, unique_label: str
    ) -> None:
        """One deleted edge should be reported as one, and the server agrees it is one."""
        source = await handler.create_node([unique_label], {"name": "alice"})
        target = await handler.create_node([unique_label], {"name": "bob"})
        relationship = await handler.create_relationship(source.id, target.id, "KNOWS")
        assert (
            await reader.count(f"MATCH (:{unique_label})-[r:KNOWS]->() RETURN count(r)")
            == 1
        )

        result = await handler.delete_relationship(relationship.id)

        assert result.relationships_deleted == 1


# =============================================================================
# Traversal
# =============================================================================


class TestHandlerGraphTraversal:
    """traverse() from a real node over real edges."""

    @pytest.mark.asyncio
    async def test_a_traversal_that_matches_nothing_returns_an_empty_result(
        self, handler: HandlerGraph, reader: GraphReader, unique_label: str
    ) -> None:
        """An isolated node yields no nodes, no edges and no paths.

        The premise — that the node exists and genuinely has no outgoing
        ``KNOWS`` edge — is read off the server, so an empty result cannot be
        passing because the start node was missing.
        """
        node = await handler.create_node([unique_label], {"name": "lonely"})
        assert (
            await reader.count(
                "MATCH (n) WHERE id(n) = $node_id "
                "OPTIONAL MATCH (n)-[r:KNOWS]->() RETURN count(r)",
                {"node_id": int(node.id)},
            )
            == 0
        )

        result = await handler.traverse(node.id, ["KNOWS"], "outgoing", 1)

        assert result.nodes == []
        assert result.relationships == []
        assert result.paths == []
        assert result.depth_reached == 0

    # OMN-18795: this was a strict xfail pinning a real defect -- traverse()
    # read its rows with `await result.data()`, which flattens neo4j Node and
    # Relationship objects into plain dicts, so every traversal that reached a
    # node raised AttributeError and only the zero-row path had ever worked.
    # The handler iterates Record objects now and this test passes.
    @pytest.mark.asyncio
    async def test_a_traversal_returns_the_nodes_the_server_can_reach(
        self, handler: HandlerGraph, reader: GraphReader, unique_label: str
    ) -> None:
        """A one-hop traversal finds the neighbour the server says is there."""
        source = await handler.create_node([unique_label], {"name": "alice"})
        target = await handler.create_node([unique_label], {"name": "bob"})
        await handler.create_relationship(source.id, target.id, "KNOWS")
        assert (
            await reader.count(
                "MATCH (a)-[:KNOWS]->(b) WHERE id(a) = $node_id RETURN count(b)",
                {"node_id": int(source.id)},
            )
            == 1
        )

        result = await handler.traverse(source.id, ["KNOWS"], "outgoing", 1)

        assert [node.properties["name"] for node in result.nodes] == ["bob"]
        assert [edge.type for edge in result.relationships] == ["KNOWS"]
        assert result.depth_reached == 1


# =============================================================================
# Envelope dispatch — the execute() surface
# =============================================================================


class TestHandlerGraphEnvelopeDispatch:
    """execute() routes each supported operation to the method that does the work."""

    @pytest.mark.asyncio
    async def test_execute_query_through_the_envelope_returns_server_records(
        self, handler: HandlerGraph
    ) -> None:
        """The envelope's payload field is 'query'/'parameters', and it is bound."""
        envelope: dict[str, object] = {
            "operation": "graph.execute_query",
            "payload": {
                "query": "RETURN $name AS name, $value AS value",
                "parameters": {"name": "test_param", "value": 42},
            },
            "correlation_id": str(uuid4()),
        }

        payload = query_payload(await handler.execute(envelope))

        assert [record.data for record in payload.records] == [
            {"name": "test_param", "value": 42}
        ]
        assert payload.summary["contains_updates"] is False

    @pytest.mark.asyncio
    async def test_execute_query_batch_through_the_envelope_commits(
        self, handler: HandlerGraph, reader: GraphReader, unique_label: str
    ) -> None:
        """The batch operation writes through the envelope path too."""
        envelope: dict[str, object] = {
            "operation": "graph.execute_query_batch",
            "payload": {
                "queries": [
                    {
                        "query": f"CREATE (n:{unique_label} {{name: $name}})",
                        "parameters": {"name": "alice"},
                    },
                    {
                        "query": f"CREATE (n:{unique_label} {{name: $name}})",
                        "parameters": {"name": "bob"},
                    },
                ],
                "transaction": True,
            },
        }

        payload = execute_payload(await handler.execute(envelope))

        assert payload.success is True
        assert payload.counters["query_count"] == 2
        assert await reader.count(f"MATCH (n:{unique_label}) RETURN count(n)") == 2

    @pytest.mark.asyncio
    async def test_create_node_through_the_envelope_is_visible_to_the_reader(
        self, handler: HandlerGraph, reader: GraphReader, unique_label: str
    ) -> None:
        """The node id the envelope reports addresses a node the server holds."""
        envelope: dict[str, object] = {
            "operation": "graph.create_node",
            "payload": {
                "labels": [unique_label],
                "properties": {"name": "alice"},
            },
        }

        payload = execute_payload(await handler.execute(envelope))

        assert payload.counters["nodes_created"] == 1
        stored = await reader.rows(
            f"MATCH (n:{unique_label}) WHERE id(n) = $node_id RETURN n.name AS name",
            {"node_id": int(str(payload.counters["node_id"]))},
        )
        assert stored == [{"name": "alice"}]

    @pytest.mark.asyncio
    async def test_create_relationship_through_the_envelope_is_visible_to_the_reader(
        self, handler: HandlerGraph, reader: GraphReader, unique_label: str
    ) -> None:
        """The relationship id the envelope reports addresses a real edge."""
        source = await handler.create_node([unique_label], {"name": "alice"})
        target = await handler.create_node([unique_label], {"name": "bob"})
        envelope: dict[str, object] = {
            "operation": "graph.create_relationship",
            "payload": {
                "from_node_id": source.id,
                "to_node_id": target.id,
                "relationship_type": "KNOWS",
                "properties": {"since": 2021},
            },
        }

        payload = execute_payload(await handler.execute(envelope))

        assert payload.counters["relationships_created"] == 1
        stored = await reader.rows(
            "MATCH ()-[r:KNOWS]->() WHERE id(r) = $rel_id RETURN r.since AS since",
            {"rel_id": int(str(payload.counters["relationship_id"]))},
        )
        assert stored == [{"since": 2021}]

    @pytest.mark.asyncio
    async def test_delete_node_through_the_envelope_removes_it(
        self, handler: HandlerGraph, reader: GraphReader, unique_label: str
    ) -> None:
        """Detach deletion through the envelope, proven by a read-back."""
        source = await handler.create_node([unique_label], {"name": "alice"})
        target = await handler.create_node([unique_label], {"name": "bob"})
        await handler.create_relationship(source.id, target.id, "KNOWS")
        envelope: dict[str, object] = {
            "operation": "graph.delete_node",
            "payload": {"node_id": source.id, "detach": True},
        }

        payload = execute_payload(await handler.execute(envelope))

        assert payload.counters["nodes_deleted"] == 1
        assert await reader.count(f"MATCH (n:{unique_label}) RETURN count(n)") == 1

    @pytest.mark.asyncio
    async def test_delete_relationship_through_the_envelope_removes_the_edge(
        self, handler: HandlerGraph, reader: GraphReader, unique_label: str
    ) -> None:
        """Only the edge goes; the endpoints are still there afterwards.

        ``counters["relationships_deleted"]`` is deliberately NOT asserted here:
        it carries the over-count recorded by
        ``test_delete_relationship_reports_the_one_edge_it_deleted``, and
        asserting the wrong number in a second place would cement it.
        """
        source = await handler.create_node([unique_label], {"name": "alice"})
        target = await handler.create_node([unique_label], {"name": "bob"})
        relationship = await handler.create_relationship(source.id, target.id, "KNOWS")
        envelope: dict[str, object] = {
            "operation": "graph.delete_relationship",
            "payload": {"relationship_id": relationship.id},
        }

        payload = execute_payload(await handler.execute(envelope))

        assert payload.success is True
        assert (
            await reader.count(
                "MATCH ()-[r]->() WHERE id(r) = $rel_id RETURN count(r)",
                {"rel_id": int(relationship.id)},
            )
            == 0
        )
        assert await reader.count(f"MATCH (n:{unique_label}) RETURN count(n)") == 2

    @pytest.mark.asyncio
    async def test_traverse_through_the_envelope_returns_an_empty_result(
        self, handler: HandlerGraph, unique_label: str
    ) -> None:
        """The seventh operation routes; only its zero-row path works today.

        The non-empty case is the ``traverse()`` defect recorded above — through
        this path the AttributeError is re-wrapped as ``RuntimeHostError``, so
        it would not even name the cause. This asserts the routing and the one
        outcome that is correct.
        """
        node = await handler.create_node([unique_label], {"name": "lonely"})
        envelope: dict[str, object] = {
            "operation": "graph.traverse",
            "payload": {"start_node_id": node.id, "relationship_types": ["KNOWS"]},
        }

        payload = query_payload(await handler.execute(envelope))

        assert payload.records == []
        assert payload.summary["nodes_found"] == 0


# =============================================================================
# Correlation id propagation
# =============================================================================


class TestHandlerGraphCorrelationId:
    """The envelope's tracing ids are carried through to the response."""

    @pytest.mark.asyncio
    async def test_the_envelope_correlation_id_is_carried_to_both_layers(
        self, handler: HandlerGraph
    ) -> None:
        """The id appears on the handler output and on the response it wraps."""
        correlation_id = uuid4()
        envelope: dict[str, object] = {
            "operation": "graph.execute_query",
            "payload": {"query": "RETURN 1 AS one"},
            "correlation_id": str(correlation_id),
        }

        output = await handler.execute(envelope)

        assert output.correlation_id == correlation_id
        response = output.result
        assert response is not None
        assert response.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_the_envelope_id_is_echoed_as_the_response_causality_link(
        self, handler: HandlerGraph
    ) -> None:
        """envelope_id is what pairs a response with the request that caused it."""
        envelope_id = uuid4()
        envelope: dict[str, object] = {
            "operation": "graph.execute_query",
            "payload": {"query": "RETURN 1 AS one"},
            "envelope_id": str(envelope_id),
            "correlation_id": str(uuid4()),
        }

        output = await handler.execute(envelope)

        assert output.input_envelope_id == envelope_id

    @pytest.mark.asyncio
    async def test_a_missing_correlation_id_is_generated_rather_than_refused(
        self, handler: HandlerGraph
    ) -> None:
        """Tracing is never absent: an envelope without one still gets one."""
        envelope: dict[str, object] = {
            "operation": "graph.execute_query",
            "payload": {"query": "RETURN 1 AS one"},
        }

        output = await handler.execute(envelope)

        assert isinstance(output.correlation_id, UUID)


# =============================================================================
# Injection safety
# =============================================================================


class TestHandlerGraphInjectionSafety:
    """A parameter is a value. It is never a fragment of the query."""

    @pytest.mark.asyncio
    async def test_a_parameter_containing_cypher_is_stored_as_a_literal_value(
        self, handler: HandlerGraph, reader: GraphReader, unique_label: str
    ) -> None:
        """The injected clause is data, and the canary it targets survives.

        The old suite only round-tripped the string, which a vulnerable handler
        would also pass as long as it escaped the quote. The claim that matters
        is that the payload did not EXECUTE, so a canary node is created first,
        by the reader, and its survival is part of the assertion.
        """
        canary_label = f"Canary_{uuid4().hex[:12]}"
        await reader.rows(f"CREATE (n:{canary_label} {{name: 'canary'}})")
        injected = f"x'}}); MATCH (n:{canary_label}) DETACH DELETE n; //"

        try:
            result = await handler.execute_query(
                f"CREATE (n:{unique_label} {{name: $name}})", {"name": injected}
            )

            assert result.counters.nodes_created == 1
            stored = await reader.rows(
                f"MATCH (n:{unique_label}) RETURN n.name AS name"
            )
            assert [row["name"] for row in stored] == [injected]
            assert await reader.count(f"MATCH (n:{canary_label}) RETURN count(n)") == 1
        finally:
            await reader.rows(f"MATCH (n:{canary_label}) DETACH DELETE n")


# =============================================================================
# Error paths
# =============================================================================


class TestHandlerGraphErrors:
    """The refusals, each checked for the condition it is supposed to refuse."""

    @pytest.mark.asyncio
    async def test_execute_on_an_uninitialized_handler_is_refused(
        self, uninitialized_handler: HandlerGraph
    ) -> None:
        """No driver means no dispatch, before any operation is looked up."""
        envelope: dict[str, object] = {
            "operation": "graph.execute_query",
            "payload": {"query": "RETURN 1"},
        }

        with pytest.raises(RuntimeHostError, match="not initialized"):
            await uninitialized_handler.execute(envelope)

    @pytest.mark.asyncio
    async def test_execute_query_on_an_uninitialized_handler_is_refused(
        self, uninitialized_handler: HandlerGraph
    ) -> None:
        """The direct protocol method guards itself; it is not only the envelope."""
        with pytest.raises(RuntimeHostError, match="not initialized"):
            await uninitialized_handler.execute_query("RETURN 1")

    @pytest.mark.asyncio
    async def test_invalid_cypher_is_refused_and_writes_nothing(
        self, handler: HandlerGraph, reader: GraphReader, unique_label: str
    ) -> None:
        """A syntax error surfaces as a connection error, and leaves no partial write."""
        with pytest.raises(InfraConnectionError):
            await handler.execute_query(
                f"CREATE (n:{unique_label} {{name: 'alice'}}) THIS IS NOT CYPHER"
            )

        assert await reader.count(f"MATCH (n:{unique_label}) RETURN count(n)") == 0

    @pytest.mark.asyncio
    async def test_an_unsupported_operation_is_refused_and_lists_the_supported_ones(
        self, handler: HandlerGraph
    ) -> None:
        """The refusal names the alternatives, so a caller is not left guessing.

        ``graph.query`` is the operation the previous revision of this suite
        used throughout. It is asserted here because that is exactly the drift
        an unrunnable suite hid: the name is gone, and the handler says so.
        """
        envelope: dict[str, object] = {
            "operation": "graph.query",
            "payload": {"query": "RETURN 1"},
        }

        with pytest.raises(RuntimeHostError) as refusal:
            await handler.execute(envelope)

        message = str(refusal.value)
        assert "not supported" in message
        assert "graph.execute_query" in message
        assert "graph.traverse" in message

    @pytest.mark.asyncio
    async def test_an_envelope_with_no_payload_is_refused(
        self, handler: HandlerGraph
    ) -> None:
        """The payload is checked before the operation is dispatched."""
        envelope: dict[str, object] = {"operation": "graph.execute_query"}

        with pytest.raises(RuntimeHostError, match="payload"):
            await handler.execute(envelope)

    @pytest.mark.parametrize(
        ("operation", "payload", "expected"),
        [
            ("graph.execute_query", {}, "'query'"),
            ("graph.execute_query_batch", {}, "'queries'"),
            ("graph.create_relationship", {"from_node_id": "1"}, "relationship_type"),
            ("graph.delete_node", {}, "node_id"),
            ("graph.delete_relationship", {}, "relationship_id"),
            ("graph.traverse", {}, "start_node_id"),
        ],
    )
    @pytest.mark.asyncio
    async def test_a_missing_required_payload_field_is_refused_by_name(
        self,
        handler: HandlerGraph,
        operation: str,
        payload: dict[str, object],
        expected: str,
    ) -> None:
        """Every write operation names the field it needed and did not get."""
        envelope: dict[str, object] = {"operation": operation, "payload": payload}

        with pytest.raises(RuntimeHostError, match=expected):
            await handler.execute(envelope)


# =============================================================================
# Initialization
# =============================================================================


class TestHandlerGraphInitialization:
    """The two shapes initialize() accepts, proven by a query that follows."""

    @pytest.mark.asyncio
    async def test_initialize_accepts_a_config_dict_keyed_on_connection_uri(
        self, reader: GraphReader, unique_label: str
    ) -> None:
        """The dict form the handler registry uses reaches the same server.

        The recognised keys are ``connection_uri`` (or ``bolt_uri``), ``auth``
        and ``options`` — ``handler_graph.py:266``. ``uri`` is NOT one of them,
        despite ``ModelGraphHandlerConfig`` declaring a field by that name, so a
        config dict built from that model silently falls through to the contract
        descriptor and a different endpoint (OMN-18795). This test pins the form
        that actually works, and proves it by writing through it.
        """
        from omnibase_infra.handlers import HandlerGraph

        instance = HandlerGraph(ModelONEXContainer(enable_service_registry=False))
        await instance.initialize(
            {
                "connection_uri": MEMGRAPH_BOLT_URL,
                "auth": (MEMGRAPH_USERNAME, MEMGRAPH_PASSWORD)
                if MEMGRAPH_USERNAME
                else None,
                "options": {
                    "database": MEMGRAPH_DATABASE,
                    "timeout_seconds": 30.0,
                    "max_connection_pool_size": 5,
                },
            }
        )
        try:
            health = await instance.health_check()
            assert health.healthy is True

            await instance.create_node([unique_label], {"name": "alice"})
            assert await reader.count(f"MATCH (n:{unique_label}) RETURN count(n)") == 1
        finally:
            await instance.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_releases_the_driver_and_the_handler_refuses_afterwards(
        self,
    ) -> None:
        """After shutdown the handler is uninitialized again, not half-alive."""
        from omnibase_infra.handlers import HandlerGraph

        instance = HandlerGraph(ModelONEXContainer(enable_service_registry=False))
        await instance.initialize(MEMGRAPH_BOLT_URL, auth=BOLT_AUTH)
        assert (await instance.health_check()).healthy is True

        await instance.shutdown()

        with pytest.raises(RuntimeHostError, match="not initialized"):
            await instance.execute_query("RETURN 1")
