# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Correlation ID propagation across real HTTP, PostgreSQL and Kafka boundaries.

Selection in CI (OMN-18795). This suite does NOT skip gracefully in CI, because
a silent skip and a green run are indistinguishable in a junit summary:

    - the module carries ``pytest.mark.heavy``, which the PR test splits
      deselect, so it is never collected-and-skipped on a pull request;
    - a service-backed job in ``.github/workflows/ci.yml`` starts a real
      Redpanda and a real PostgreSQL, exports ``RUN_HEAVY_TESTS=1`` alongside
      the endpoint variables, and runs it;
    - a CI job that selects it without that provisioning FAILS, naming the
      variable, rather than skipping.

Outside CI it skips, so a laptop with neither a broker nor a database is not
forced to stand them up.

Why this file was rewritten, not merely ungated
-----------------------------------------------
``RUN_HEAVY_TESTS`` was set by no workflow in this repository, so these tests
had never executed once. Their first run against real services returned
``2 failed, 5 passed, 3 errors``, entirely from APIs that had moved underneath
a suite nothing was exercising:

    - the database tests requested a fixture named ``initialized_db_handler``.
      That fixture lives in ``tests/integration/handlers/conftest.py`` and has
      never been visible from this directory; only its ``POSTGRES_AVAILABLE``
      constant was imported, which made the dependency look satisfied. The
      handler is now built here, the way live code builds it.
    - ``ModelKafkaEventBusConfig(group=...)`` no longer validates. OMN-1602
      removed the bus-level consumer group: the group is derived per
      subscription from a ``ModelNodeIdentity`` passed to ``subscribe()``.
    - ``InfraTimeoutError`` takes a ``ModelTimeoutErrorContext``, not a
      ``ModelInfraErrorContext``. The old test passed the latter and the
      constructor reached for ``context.timeout_seconds``, a field that model
      does not have.
    - ``tests.helpers.kafka_utils`` does not exist; the helper is
      ``tests.helpers.util_kafka``. Nothing had ever imported it from here.

That is the cost of a suite that skips: it keeps its shape while the thing it
claims to cover moves out from under it, and nothing says so.

What every test here must do
----------------------------
A correlation-propagation test proves its claim by observing the value on the
FAR SIDE -- the request the HTTP server actually received, the row PostgreSQL
actually returned, the error context the failing infrastructure actually
produced, the message Kafka actually delivered. Re-reading the variable the
test itself set proves nothing, and five tests doing exactly that were deleted
rather than weakened (see the deletion notes on each section below).

Run with infrastructure::

    $ RUN_HEAVY_TESTS=1 KAFKA_BOOTSTRAP_SERVERS=<host:port> \\
        OMNIBASE_INFRA_DB_URL=postgresql://<user>:<pw>@<host>:<port>/<db> \\
        uv run pytest tests/integration/correlation/test_correlation_propagation_heavy.py -v

Related tickets: OMN-18781 (the precedent that introduced the fail-closed
helper), OMN-18795 (this rewrite).
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from pytest_httpserver import HTTPServer
from werkzeug import Request, Response

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraUnavailableError,
    RuntimeHostError,
)
from omnibase_infra.models import ModelNodeIdentity
from tests.helpers.service_env import require_service_env
from tests.helpers.util_kafka import wait_for_consumer_ready
from tests.helpers.util_postgres import PostgresConfig

if TYPE_CHECKING:
    from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
    from omnibase_infra.event_bus.models import ModelEventMessage
    from omnibase_infra.handlers import HandlerDb, HandlerHttpRest

# =============================================================================
# Module-level selection
# =============================================================================

pytestmark = [
    pytest.mark.integration,
    pytest.mark.heavy,
]


@pytest.fixture(autouse=True, scope="module")
def _require_services() -> None:
    """Refuse to skip this suite silently once CI has selected it.

    ``pytest-httpserver`` is deliberately imported at module scope above
    rather than behind a try/except, and ``httpx`` reaches the wire through
    ``HandlerHttpRest``. Both are hard dependencies of this project; the old
    ``HTTPSERVER_AVAILABLE`` fallback turned a genuinely missing one into a
    per-class skip, which is the exact class of false green OMN-18795 exists
    to remove. A missing dependency is now a collection error.
    """
    require_service_env(
        opt_in="RUN_HEAVY_TESTS",
        endpoint="KAFKA_BOOTSTRAP_SERVERS",
        workflow=".github/workflows/ci.yml",
        service="Kafka and Postgres",
    )


# Header name the HTTP boundary tests trace on. Defined once so the request
# assertion and the response assertion cannot drift apart.
CORRELATION_HEADER = "X-Correlation-ID"

# Kafka operation bounds. The delivery wait is generous because a freshly
# created topic's consumer has to join a group before the first record lands.
MESSAGE_DELIVERY_WAIT_SECONDS = 20.0
TEST_TIMEOUT_SECONDS = 30

# A broker address that cannot resolve, ever. ``.invalid`` is reserved by
# RFC 2606 precisely so a name is guaranteed NXDOMAIN, which makes the failure
# tests deterministic and fast instead of dependent on whatever a resolver does
# with an unqualified made-up hostname.
UNREACHABLE_BROKER = "kafka-unreachable.invalid:9092"


# =============================================================================
# HTTP boundary
# =============================================================================
#
# OMN-18795 deletion: ``test_correlation_through_http_boundary`` and
# ``test_correlation_echoed_in_response_header`` drove a raw ``httpx`` client
# against a mock server the test itself configured to return the correlation id
# it had just written. They exercised httpx and pytest-httpserver; no line of
# this repository was on the path, so neither could have caught a propagation
# defect in it. The two tests below put ``HandlerHttpRest`` -- the repository's
# own HTTP boundary -- in the middle, and read the far side back.


@pytest.fixture
async def http_handler(mock_container: MagicMock) -> AsyncGenerator[HandlerHttpRest]:
    """An initialized HandlerHttpRest, shut down on every exit path."""
    from omnibase_infra.handlers import HandlerHttpRest

    handler = HandlerHttpRest(container=mock_container)
    await handler.initialize(
        {
            "max_request_size": 1024 * 1024,
            "max_response_size": 10 * 1024 * 1024,
        }
    )
    try:
        yield handler
    finally:
        await handler.shutdown()


class TestCorrelationHttpBoundary:
    """Correlation ids cross the handler's HTTP boundary in both directions."""

    @pytest.mark.asyncio
    async def test_correlation_header_reaches_the_server_and_comes_back(
        self,
        httpserver: HTTPServer,
        http_handler: HandlerHttpRest,
        correlation_id: UUID,
    ) -> None:
        """The server sees the id on the wire, echoes it, and the handler binds it.

        Three observations, none of which re-reads the value the test set:

        1. ``httpserver.log`` is the request the server actually received --
           the id was on the wire, not merely in the envelope.
        2. the response body carries what the SERVER read out of that request,
           so the id survived serialization in both directions.
        3. the handler's own response envelope carries the same id, which is
           the property a downstream consumer of this handler depends on.
        """

        def echo_correlation(request: Request) -> Response:
            """Reflect the correlation header the server received into the body."""
            return Response(
                json.dumps({"seen": request.headers.get(CORRELATION_HEADER)}),
                status=200,
                content_type="application/json",
            )

        httpserver.expect_request("/trace").respond_with_handler(echo_correlation)

        envelope: dict[str, object] = {
            "operation": "http.get",
            "correlation_id": correlation_id,
            "payload": {
                "url": httpserver.url_for("/trace"),
                "headers": {CORRELATION_HEADER: str(correlation_id)},
            },
        }

        output = await http_handler.execute(envelope)
        result = output.result
        assert result is not None
        assert result["status"] == "success"

        # (1) what the server received
        assert len(httpserver.log) == 1, (
            f"expected exactly one request, got {len(httpserver.log)}"
        )
        received_request, _ = httpserver.log[0]
        assert received_request.headers.get(CORRELATION_HEADER) == str(correlation_id)

        # (2) what the server read back out and returned
        payload = result["payload"]
        assert isinstance(payload, dict)
        assert payload["body"] == {"seen": str(correlation_id)}

        # (3) what the handler bound to its own response envelope
        assert result["correlation_id"] == str(correlation_id)
        assert output.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_a_server_set_correlation_header_is_surfaced_to_the_caller(
        self,
        httpserver: HTTPServer,
        http_handler: HandlerHttpRest,
        correlation_id: UUID,
    ) -> None:
        """A correlation header the SERVER sets reaches the handler's caller.

        The inbound direction. The request carries no correlation header at
        all, so the id in the assertion can only have come from the server's
        response -- the handler has to have collected and returned it.

        Response header names are compared lower-cased: httpx normalizes them,
        and the handler passes that mapping through unchanged.
        """
        server_issued_id = str(correlation_id)
        httpserver.expect_request("/issues-correlation").respond_with_json(
            {"status": "ok"},
            headers={CORRELATION_HEADER: server_issued_id},
        )

        envelope: dict[str, object] = {
            "operation": "http.get",
            "payload": {"url": httpserver.url_for("/issues-correlation")},
        }

        output = await http_handler.execute(envelope)
        result = output.result
        assert result is not None
        assert result["status"] == "success"

        payload = result["payload"]
        assert isinstance(payload, dict)
        response_headers = payload["headers"]
        assert isinstance(response_headers, dict)
        assert response_headers[CORRELATION_HEADER.lower()] == server_issued_id

        # The request really did go out without one, so the value above is the
        # server's and not an echo of something this test put on the wire.
        received_request, _ = httpserver.log[0]
        assert received_request.headers.get(CORRELATION_HEADER) is None


# =============================================================================
# Error context, from failures that actually happened
# =============================================================================
#
# OMN-18795 deletion: four tests here constructed a ModelInfraErrorContext,
# constructed an error around it, and asserted the error carried the id they
# had just put in -- with no infrastructure involved at any point
# (``test_correlation_preserved_on_connection_error``,
# ``..._on_timeout_error``, ``..._on_unavailable_error``, and
# ``test_correlation_preserved_on_kafka_error``, which provoked a real
# connection failure, discarded the error it got, built a second error by hand
# and asserted on that one instead). Those are model round-trips, and
# ``tests/unit/errors/test_infra_errors.py`` already covers every one of them
# at unit level, including InfraTimeoutError with ModelTimeoutErrorContext.
#
# ``test_correlation_in_error_string_representation`` was deleted for a
# different reason: its docstring claimed the correlation id appears in the
# error's string representation, and it does not -- ``str(error)`` is
# ``"[ONEX_CORE_090_NETWORK_ERROR] <message>"``. The test quietly asserted on
# ``model_dump()`` instead, so the claim in the docstring had never been
# checked by anything.
#
# The two tests below replace all five with failures that really occur.


@pytest.fixture
def unreachable_bus_config() -> object:
    """Config for a bus pointed at a broker that cannot resolve.

    ``circuit_breaker_threshold=1`` so the circuit opens on the first failure:
    the second ``start()`` is then refused by the breaker rather than by DNS,
    which is what the second test needs to observe.
    """
    from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

    return ModelKafkaEventBusConfig(
        bootstrap_servers=UNREACHABLE_BROKER,
        environment="correlation-test",
        timeout_seconds=2,
        max_retry_attempts=0,
        retry_backoff_base=0.001,
        circuit_breaker_threshold=1,
        circuit_breaker_reset_timeout=60.0,
    )


@pytest.fixture
async def unreachable_bus(
    unreachable_bus_config: object,
) -> AsyncGenerator[EventBusKafka]:
    """A never-started bus aimed at an unresolvable broker, closed afterwards."""
    from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka

    bus = EventBusKafka(config=unreachable_bus_config)  # type: ignore[arg-type]
    try:
        yield bus
    finally:
        await bus.close()


class TestCorrelationErrorContextFromRealFailures:
    """Real infrastructure failures arrive carrying a traceable context."""

    @pytest.mark.asyncio
    async def test_a_real_connection_failure_is_traceable_and_names_the_broker(
        self,
        unreachable_bus: EventBusKafka,
    ) -> None:
        """A failed ``start()`` reports a correlation id and what it could not reach.

        The id is generated inside ``EventBusKafka.start()`` -- there is no
        caller-supplied correlation at that boundary -- so the assertion is
        that the error arrives traceable, and that its context identifies the
        operation and the broker rather than being empty.
        """
        with pytest.raises(InfraConnectionError) as exc_info:
            await unreachable_bus.start()

        error = exc_info.value

        assert error.correlation_id is not None
        assert error.model.correlation_id == error.correlation_id

        context = error.model.context
        assert context is not None
        assert context["operation"] == "start"
        assert context["target_name"] == "kafka.correlation-test"
        assert context["servers"] == UNREACHABLE_BROKER

    @pytest.mark.asyncio
    async def test_a_circuit_open_refusal_is_traceable_and_distinct(
        self,
        unreachable_bus: EventBusKafka,
    ) -> None:
        """Once the breaker opens, the refusal is its own traceable error.

        The first ``start()`` fails against DNS and trips the breaker
        (threshold 1). The second never reaches the network: the breaker
        refuses it with ``InfraUnavailableError``, carrying a correlation id of
        its own and a ``circuit_state`` the connection error above does not
        have. Two different failure modes, two distinguishable contexts -- the
        property an operator reading a trace depends on.
        """
        with pytest.raises(InfraConnectionError) as first:
            await unreachable_bus.start()

        with pytest.raises(InfraUnavailableError) as second:
            await unreachable_bus.start()

        refusal = second.value
        assert refusal.correlation_id is not None
        assert refusal.correlation_id != first.value.correlation_id, (
            "each failure needs its own correlation id, or two incidents "
            "collapse into one in a trace"
        )

        context = refusal.model.context
        assert context is not None
        assert context["operation"] == "start"
        assert context["circuit_state"] == "open"


# =============================================================================
# Database
# =============================================================================


@pytest.fixture
async def db_handler(mock_container: MagicMock) -> AsyncGenerator[HandlerDb]:
    """An initialized HandlerDb against the provisioned PostgreSQL.

    OMN-18795: the suite used to ask for ``initialized_db_handler``, a fixture
    defined in ``tests/integration/handlers/conftest.py`` and therefore never
    in scope here -- every database test errored at setup. The handler is
    constructed here the way live code constructs it: a container, then
    ``initialize()`` with a DSN, then ``shutdown()``.
    """
    from omnibase_infra.handlers import HandlerDb

    config = PostgresConfig.from_env()
    handler = HandlerDb(mock_container)
    await handler.initialize({"dsn": config.build_dsn(), "timeout": 30.0})
    try:
        yield handler
    finally:
        await handler.shutdown()


@pytest.fixture
def unique_table_name() -> str:
    """A table name unique to this test.

    Unique per test so a parallel run, or a leftover from a failed run, cannot
    make one test's assertions depend on another's rows.
    """
    return f"test_correlation_{uuid4().hex[:12]}"


class TestCorrelationDatabase:
    """Correlation ids survive a round trip through real PostgreSQL."""

    @pytest.mark.asyncio
    async def test_correlation_id_round_trips_through_a_real_row(
        self,
        db_handler: HandlerDb,
        correlation_id: UUID,
        unique_table_name: str,
    ) -> None:
        """The id is written to PostgreSQL and read back out of it.

        The assertion is on the value PostgreSQL returned, after a real INSERT
        and a real SELECT against a real table -- not on the handler's echo of
        its own input. The handler's response envelope is checked as well,
        because both halves have to agree for a trace to be followable.

        ``$1::uuid`` is used rather than string interpolation so the value is
        bound as a UUID by the driver and compared as one by the server.
        """
        # Identifiers are generated by the fixture from a UUID, never from
        # input, so they cannot carry an injection.
        create: dict[str, object] = {
            "operation": "db.execute",
            "correlation_id": str(correlation_id),
            "payload": {
                "sql": f'CREATE TABLE "{unique_table_name}" (corr uuid NOT NULL)',
                "parameters": [],
            },
        }
        await db_handler.execute(create)

        try:
            insert: dict[str, object] = {
                "operation": "db.execute",
                "correlation_id": str(correlation_id),
                "payload": {
                    "sql": f'INSERT INTO "{unique_table_name}" (corr) VALUES ($1)',  # noqa: S608
                    "parameters": [correlation_id],
                },
            }
            insert_output = await db_handler.execute(insert)
            insert_result = insert_output.result
            assert insert_result is not None
            assert insert_result.payload.row_count == 1

            select: dict[str, object] = {
                "operation": "db.query",
                "correlation_id": str(correlation_id),
                "payload": {
                    "sql": f'SELECT corr FROM "{unique_table_name}" WHERE corr = $1',  # noqa: S608
                    "parameters": [correlation_id],
                },
            }
            output = await db_handler.execute(select)
            selected = output.result
            assert selected is not None

            # The far side: the row PostgreSQL returned.
            assert selected.payload.row_count == 1
            assert UUID(str(selected.payload.rows[0]["corr"])) == correlation_id

            # The near side has to agree with it.
            assert selected.status == "success"
            assert output.correlation_id == correlation_id
            assert selected.correlation_id == correlation_id
        finally:
            drop: dict[str, object] = {
                "operation": "db.execute",
                "payload": {
                    "sql": f'DROP TABLE IF EXISTS "{unique_table_name}"',
                    "parameters": [],
                },
            }
            await db_handler.execute(drop)

    @pytest.mark.asyncio
    async def test_a_real_sql_error_carries_the_callers_correlation_id(
        self,
        db_handler: HandlerDb,
        correlation_id: UUID,
    ) -> None:
        """A statement PostgreSQL rejects comes back traceable to its caller.

        The failure is the server's, not the test's: ``SELECTT`` is a syntax
        error PostgreSQL raises (SQLSTATE 42601), which ``HandlerDb`` maps onto
        ``RuntimeHostError``. What is asserted is that the error the server
        produced arrived carrying the correlation id the CALLER supplied, so a
        failed query can be joined to the request that issued it.
        """
        envelope: dict[str, object] = {
            "operation": "db.query",
            "correlation_id": str(correlation_id),
            "payload": {
                # Deliberate typo: PostgreSQL, not the handler, rejects this.
                "sql": "SELECTT * FROM nonexistent_correlation_test_table",
                "parameters": [],
            },
        }

        with pytest.raises(RuntimeHostError) as exc_info:
            await db_handler.execute(envelope)

        error = exc_info.value

        # The message is the server's classification, carried through intact.
        assert "SQL syntax error" in str(error)

        assert error.correlation_id == correlation_id
        assert error.model.correlation_id == correlation_id

        context = error.model.context
        assert context is not None
        assert context["operation"] == "db.query"
        assert context["target_name"] == "db_handler"


# =============================================================================
# Kafka
# =============================================================================


@pytest.fixture
async def kafka_bus(kafka_bootstrap_servers: str) -> AsyncGenerator[EventBusKafka]:
    """A started EventBusKafka against the provisioned broker, closed afterwards.

    OMN-18795: the old fixture passed ``group=`` to ``ModelKafkaEventBusConfig``.
    OMN-1602 removed that field -- the consumer group is no longer a property
    of the bus, it is derived per subscription from the ``ModelNodeIdentity``
    handed to ``subscribe()`` -- and the model forbids extra fields, so every
    Kafka test in this file errored at setup.
    """
    from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
    from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

    config = ModelKafkaEventBusConfig(
        bootstrap_servers=kafka_bootstrap_servers,
        environment="correlation-test",
        timeout_seconds=TEST_TIMEOUT_SECONDS,
        max_retry_attempts=2,
        retry_backoff_base=0.5,
        circuit_breaker_threshold=5,
        circuit_breaker_reset_timeout=10.0,
        auto_offset_reset="earliest",
    )
    bus = EventBusKafka(config=config)
    await bus.start()
    try:
        yield bus
    finally:
        await bus.close()


@pytest.fixture
def kafka_bootstrap_servers() -> str:
    """The broker address the provisioning job exported.

    Indexed, not ``.get()``-with-a-default: a missing variable here is a
    misconfigured job, and a localhost fallback would quietly point the suite
    at a port nothing is listening on. ``_require_services`` has already failed
    the CI run by this point if the opt-in is absent.
    """
    return os.environ["KAFKA_BOOTSTRAP_SERVERS"]


@pytest.fixture
async def correlation_topic(kafka_bootstrap_servers: str) -> AsyncGenerator[str]:
    """A topic unique to this test, created up front and deleted afterwards.

    Created explicitly because the provisioned broker has topic auto-creation
    off; unique per test so a consumer cannot read another test's records.
    """
    from aiokafka.admin import AIOKafkaAdminClient, NewTopic
    from aiokafka.errors import TopicAlreadyExistsError

    topic_name = f"test.correlation.{uuid4().hex[:12]}"
    admin = AIOKafkaAdminClient(bootstrap_servers=kafka_bootstrap_servers)
    await admin.start()
    try:
        try:
            await admin.create_topics(
                [NewTopic(name=topic_name, num_partitions=1, replication_factor=1)]
            )
        except TopicAlreadyExistsError:
            pass
        # Metadata has to reach the broker's own view before a consumer can
        # join a group on the topic.
        await asyncio.sleep(0.5)

        yield topic_name
    finally:
        try:
            await admin.delete_topics([topic_name])
        finally:
            await admin.close()


@pytest.fixture
def consumer_identity() -> ModelNodeIdentity:
    """A node identity unique to this test, which the bus derives a group from."""
    return ModelNodeIdentity(
        env="correlation-test",
        service="test-service",
        node_name=f"correlation-node-{uuid4().hex[:8]}",
        version="v1",
    )


class TestCorrelationKafka:
    """Correlation ids survive a publish and consume through a real broker."""

    @pytest.mark.asyncio
    async def test_correlation_id_survives_a_real_publish_and_consume(
        self,
        kafka_bus: EventBusKafka,
        correlation_topic: str,
        consumer_identity: ModelNodeIdentity,
        correlation_id: UUID,
    ) -> None:
        """The id on the delivered message is the id that was published.

        The assertion is on ``received.headers.correlation_id`` -- a value that
        was serialized onto Kafka record headers, written to a partition, read
        back off it by a consumer in its own group, and deserialized. Nothing
        in the assertion path is the variable the test set; the only way it can
        match is if the id made the round trip.
        """
        from omnibase_infra.event_bus.models import ModelEventHeaders

        received: list[ModelEventMessage] = []
        delivered = asyncio.Event()

        async def collect(message: ModelEventMessage) -> None:
            received.append(message)
            delivered.set()

        unsubscribe = await kafka_bus.subscribe(
            correlation_topic,
            consumer_identity,
            collect,
        )
        try:
            await wait_for_consumer_ready(kafka_bus, correlation_topic)

            headers = ModelEventHeaders(
                source="correlation-test",
                event_type="test.correlation.propagation",
                correlation_id=correlation_id,
                timestamp=datetime.now(UTC),
            )
            await kafka_bus.publish(
                correlation_topic,
                b"correlation-key",
                b"correlation-test-payload",
                headers,
            )

            try:
                await asyncio.wait_for(
                    delivered.wait(), timeout=MESSAGE_DELIVERY_WAIT_SECONDS
                )
            except TimeoutError:
                pytest.fail(
                    f"no message on {correlation_topic} within "
                    f"{MESSAGE_DELIVERY_WAIT_SECONDS}s"
                )

            message = received[0]

            # The correlation id as the consumer read it off the wire.
            assert UUID(str(message.headers.correlation_id)) == correlation_id
            # Bound to the message this test published, not some other record.
            assert message.headers.event_type == "test.correlation.propagation"
            assert message.topic == correlation_topic
            assert message.value == b"correlation-test-payload"
        finally:
            await unsubscribe()
