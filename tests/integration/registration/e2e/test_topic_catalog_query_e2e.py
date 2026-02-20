# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""E2E tests for Topic Catalog query flows.

Proves Option B routing: multiple clients on a shared response topic with no
cross-talk via correlation_id filtering. All tests use component-level E2E
(handlers called in-process with real Consul, no runtime container required).

Test Suites:
    - Suite 1: Multi-client no cross-talk (2 clients, different consumer groups,
      correlation_id filtering)
    - Suite 2: Response determinism (identical results on repeated query)
    - Suite 3: Version-gap recovery simulation
    - Suite 5: Integration golden path (lightweight handler-level)

Note:
    Suite 4 (TestChangeNotificationFlow) tested the catalog-changed emission
    feature which was reverted and is not included in this file.

Infrastructure Requirements:
    All suites require ALL_INFRA_AVAILABLE (Consul + Kafka/Redpanda). The
    directory-level conftest applies a pytestmark skipif(not ALL_INFRA_AVAILABLE)
    to every test in this directory, so no suite can run independently of the
    shared infrastructure guard even if it does not use Kafka directly.

Related Tickets:
    - OMN-2317: Topic Catalog multi-client no-cross-talk E2E test
    - OMN-2313: Topic Catalog: query handler + dispatcher + contract wiring
    - OMN-2311: Topic Catalog: ServiceTopicCatalog + KV precedence + caching
    - OMN-2310: Topic Catalog model + suffix foundation
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from fnmatch import fnmatch
from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.models.catalog.model_topic_catalog_query import (
    ModelTopicCatalogQuery,
)
from omnibase_infra.models.catalog.model_topic_catalog_response import (
    ModelTopicCatalogResponse,
)
from omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_topic_catalog_query import (
    HandlerTopicCatalogQuery,
)
from omnibase_infra.services.service_topic_catalog import ServiceTopicCatalog
from omnibase_infra.topics.platform_topic_suffixes import (
    SUFFIX_TOPIC_CATALOG_RESPONSE,
)

# Note: ALL_INFRA_AVAILABLE skipif applied by conftest.py to all tests in this directory
from .conftest import (
    KAFKA_BOOTSTRAP_SERVERS,
    make_e2e_test_identity,
    wait_for_consumer_ready,
)

if TYPE_CHECKING:
    from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
    from omnibase_infra.handlers import HandlerConsul

logger = logging.getLogger(__name__)

# =============================================================================
# Timeout constants
# =============================================================================

# Maximum seconds to wait for a Kafka message to arrive
_KAFKA_RECEIVE_TIMEOUT_S = 15.0

# Maximum seconds to wait per subscription for readiness
_SUBSCRIPTION_READY_TIMEOUT_S = 10.0

# =============================================================================
# Run-scoped unique ID
# =============================================================================

# Module-level run ID: unique per test-runner process so concurrent workers and
# repeated runs on a shared Kafka cluster get distinct consumer group IDs, while
# all tests within a single run share the same suffix (important for cross-talk
# tests that need consistent group IDs within one run).
_RUN_ID = str(uuid4())[:8]

# =============================================================================
# Local fixtures
# =============================================================================


@pytest.fixture
def mock_container_for_catalog() -> MagicMock:
    """Minimal mock container for ServiceTopicCatalog construction."""
    from omnibase_core.container import ModelONEXContainer

    return MagicMock(spec=ModelONEXContainer)


@pytest.fixture
def catalog_service(
    real_consul_handler: HandlerConsul,
    mock_container_for_catalog: MagicMock,
) -> ServiceTopicCatalog:
    """ServiceTopicCatalog backed by real Consul.

    Args:
        real_consul_handler: Connected HandlerConsul instance.
        mock_container_for_catalog: Minimal mock container for DI.

    Returns:
        ServiceTopicCatalog with real Consul backend.
    """
    return ServiceTopicCatalog(
        container=mock_container_for_catalog,
        consul_handler=real_consul_handler,
    )


@pytest.fixture
def catalog_handler(
    catalog_service: ServiceTopicCatalog,
) -> HandlerTopicCatalogQuery:
    """HandlerTopicCatalogQuery wired to real ServiceTopicCatalog.

    Args:
        catalog_service: Service backed by real Consul.

    Returns:
        HandlerTopicCatalogQuery ready to process queries.
    """
    return HandlerTopicCatalogQuery(catalog_service=catalog_service)


# Function scope: each test needs a fresh consumer group offset
@pytest.fixture
async def second_kafka_bus() -> AsyncGenerator[EventBusKafka, None]:
    """Second independent EventBusKafka instance for multi-client tests.

    Uses a different environment string so its consumer group IDs do not
    overlap with those of the primary ``real_kafka_event_bus`` fixture.

    Yields:
        Started EventBusKafka with unique consumer group namespace.
    """
    from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
    from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

    if not KAFKA_BOOTSTRAP_SERVERS:
        pytest.skip("Kafka not available (KAFKA_BOOTSTRAP_SERVERS not set)")

    config = ModelKafkaEventBusConfig(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        environment=f"e2e-test-client-b-{_RUN_ID}",
        timeout_seconds=30,
        max_retry_attempts=3,
        circuit_breaker_threshold=5,
        circuit_breaker_reset_timeout=60.0,
        enable_auto_commit=False,
    )
    bus = EventBusKafka(config=config)
    await bus.start()

    yield bus

    await bus.close()


# =============================================================================
# Helper functions
# =============================================================================


def _make_query_envelope(
    correlation_id: UUID,
    client_id: str,
    include_inactive: bool = False,
    topic_pattern: str | None = None,
) -> ModelEventEnvelope[ModelTopicCatalogQuery]:
    """Build a ModelEventEnvelope[ModelTopicCatalogQuery] for handler calls.

    Args:
        correlation_id: Correlation ID for request-response pairing.
        client_id: Identifying label for the requesting client.
        include_inactive: Whether to include inactive topics.
        topic_pattern: Optional fnmatch filter pattern.

    Returns:
        Envelope ready for HandlerTopicCatalogQuery.handle().
    """
    query = ModelTopicCatalogQuery(
        correlation_id=correlation_id,
        client_id=client_id,
        include_inactive=include_inactive,
        topic_pattern=topic_pattern,
    )
    return ModelEventEnvelope(
        payload=query,
        correlation_id=correlation_id,
        envelope_timestamp=datetime.now(UTC),
    )


def _serialize_response(response: ModelTopicCatalogResponse) -> bytes:
    """Serialize ModelTopicCatalogResponse to JSON bytes for Kafka publishing.

    Args:
        response: Catalog response to serialize.

    Returns:
        UTF-8 encoded JSON bytes.
    """
    return response.model_dump_json().encode("utf-8")


def _deserialize_response(raw: bytes | str) -> ModelTopicCatalogResponse | None:
    """Deserialize bytes or string to ModelTopicCatalogResponse.

    Returns None if deserialization fails (message is not a catalog response).

    Args:
        raw: Raw bytes or string from Kafka message.

    Returns:
        Deserialized response or None on failure.
    """
    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        data = json.loads(raw)
        if isinstance(data, dict) and all(
            k in data
            for k in (
                "correlation_id",
                "topics",
                "catalog_version",
                "node_count",
                "generated_at",
                "schema_version",
            )
        ):
            return ModelTopicCatalogResponse.model_validate(data)
    except (
        json.JSONDecodeError,
        ValueError,
        KeyError,
        TypeError,
        ValidationError,
    ):
        # Expected noise: log at DEBUG only.
        #
        # json.JSONDecodeError is expected because both clients subscribe to the
        # shared SUFFIX_TOPIC_CATALOG_RESPONSE topic but the Kafka consumer may
        # deliver messages from other producers on the same topic (e.g., heartbeats,
        # control plane messages, or messages from concurrent integration tests) that
        # are not JSON or are not catalog response payloads. The topic-level consumer
        # group filter does not guarantee only catalog response messages arrive here;
        # correlation_id filtering is the correct isolation boundary (Option B).
        # ValueError and KeyError cover missing required fields or bad values in
        # otherwise-valid JSON that does not match the catalog response schema.
        # ValidationError (Pydantic v2, a subclass of ValueError) is expected noise
        # when a valid-JSON non-catalog message is passed to model_validate().
        # TypeError (e.g., None or non-iterable raw message) is also expected noise:
        # the shared-topic consumer naturally receives mixed payloads, so a None or
        # non-bytes/str value is a normal occurrence and not a genuine error.
        logger.debug(
            "_deserialize_response: failed to deserialize message",
            exc_info=True,
        )
    except Exception:
        # Genuinely unexpected errors that are not normal deserialization noise:
        # surface these at WARNING so they are visible during debugging.
        logger.warning(
            "_deserialize_response: unexpected payload structure",
            exc_info=True,
        )
    return None


# =============================================================================
# Suite 1: Multi-client no cross-talk
# =============================================================================


class TestMultiClientNoCrossTalk:
    """Prove Option B routing: shared response topic, correlation_id filtering.

    Two clients subscribe to the SAME response topic with DIFFERENT consumer
    groups. Both receive ALL messages but each filters by its own
    ``correlation_id`` so it only processes its own response.

    This is the core property that makes Option B work: topics are shared,
    isolation is per-correlation_id, not per-topic.
    """

    @pytest.mark.serial
    async def test_two_clients_no_cross_talk(
        self,
        real_kafka_event_bus: EventBusKafka,
        second_kafka_bus: EventBusKafka,
        catalog_handler: HandlerTopicCatalogQuery,
    ) -> None:
        """Two clients on shared response topic receive only their own responses.

        Steps:
        1. Client A and Client B subscribe to SUFFIX_TOPIC_CATALOG_RESPONSE
           with different consumer groups (derived from different node identities)
        2. Call handler twice to get two responses with different correlation_ids
        3. Publish both responses to the shared topic
        4. Both clients receive ALL messages (Option B property)
        5. Client A filters by its correlation_id -> only its response
        6. Client B filters by its correlation_id -> only its response
        7. Assert: A never holds B's response, B never holds A's response
        """
        correlation_a = uuid4()
        correlation_b = uuid4()

        # Collect all raw messages received on both sides
        received_a: list[ModelTopicCatalogResponse] = []
        received_b: list[ModelTopicCatalogResponse] = []

        got_a = asyncio.Event()
        got_b = asyncio.Event()

        async def on_message_a(message: object) -> None:
            """Client A receives all messages, filters by correlation_a.

            The message is an AIOKafkaConsumerRecord with a .value bytes attribute.
            """
            raw: bytes | None = None
            if hasattr(message, "value"):
                raw = message.value  # type: ignore[union-attr]
            elif isinstance(message, (bytes, str)):
                raw = message if isinstance(message, bytes) else message.encode()

            if raw is None:
                return

            response = _deserialize_response(raw)
            if response is not None:
                # All messages arrive here — Option B in action
                if response.correlation_id == correlation_a:
                    received_a.append(response)
                    got_a.set()

        async def on_message_b(message: object) -> None:
            """Client B receives all messages, filters by correlation_b.

            The message is an AIOKafkaConsumerRecord with a .value bytes attribute.
            """
            raw: bytes | None = None
            if hasattr(message, "value"):
                raw = message.value  # type: ignore[union-attr]
            elif isinstance(message, (bytes, str)):
                raw = message if isinstance(message, bytes) else message.encode()

            if raw is None:
                return

            response = _deserialize_response(raw)
            if response is not None:
                if response.correlation_id == correlation_b:
                    received_b.append(response)
                    got_b.set()

        # Subscribe both clients with different consumer groups (different node identities).
        # _RUN_ID suffix ensures group IDs are unique across concurrent workers and
        # repeated runs on a shared Kafka cluster while remaining consistent within
        # a single run (so the cross-talk test uses matching IDs on both sides).
        identity_a = make_e2e_test_identity(f"catalog_client_a_{_RUN_ID}")
        identity_b = make_e2e_test_identity(f"catalog_client_b_{_RUN_ID}")

        unsub_a = await real_kafka_event_bus.subscribe(
            topic=SUFFIX_TOPIC_CATALOG_RESPONSE,
            node_identity=identity_a,
            on_message=on_message_a,
        )
        unsub_b = await second_kafka_bus.subscribe(
            topic=SUFFIX_TOPIC_CATALOG_RESPONSE,
            node_identity=identity_b,
            on_message=on_message_b,
        )

        try:
            # Wait for both consumers to be ready before publishing
            await asyncio.gather(
                wait_for_consumer_ready(
                    real_kafka_event_bus,
                    SUFFIX_TOPIC_CATALOG_RESPONSE,
                    max_wait=_SUBSCRIPTION_READY_TIMEOUT_S,
                ),
                wait_for_consumer_ready(
                    second_kafka_bus,
                    SUFFIX_TOPIC_CATALOG_RESPONSE,
                    max_wait=_SUBSCRIPTION_READY_TIMEOUT_S,
                ),
            )

            # Build two catalog responses via the handler (in-process, real Consul)
            envelope_a = _make_query_envelope(correlation_a, client_id="client-a")
            envelope_b = _make_query_envelope(correlation_b, client_id="client-b")

            output_a = await catalog_handler.handle(envelope_a)
            output_b = await catalog_handler.handle(envelope_b)

            assert len(output_a.events) == 1, (
                "Handler must return exactly one event per query"
            )
            assert len(output_b.events) == 1, (
                "Handler must return exactly one event per query"
            )

            response_a = output_a.events[0]
            response_b = output_b.events[0]

            assert isinstance(response_a, ModelTopicCatalogResponse), (
                "Handler output must be ModelTopicCatalogResponse"
            )
            assert isinstance(response_b, ModelTopicCatalogResponse), (
                "Handler output must be ModelTopicCatalogResponse"
            )

            # Verify the handler wired the correct correlation_ids
            assert response_a.correlation_id == correlation_a, (
                "Response A must carry correlation_id from query A"
            )
            assert response_b.correlation_id == correlation_b, (
                "Response B must carry correlation_id from query B"
            )

            # Publish both responses to the SHARED topic
            # (simulating what the runtime dispatcher would do)
            await real_kafka_event_bus.publish(
                topic=SUFFIX_TOPIC_CATALOG_RESPONSE,
                key=str(correlation_a).encode("utf-8"),
                value=_serialize_response(response_a),
            )
            await real_kafka_event_bus.publish(
                topic=SUFFIX_TOPIC_CATALOG_RESPONSE,
                key=str(correlation_b).encode("utf-8"),
                value=_serialize_response(response_b),
            )

            # Wait for each client to receive its own message
            await asyncio.wait_for(got_a.wait(), timeout=_KAFKA_RECEIVE_TIMEOUT_S)
            await asyncio.wait_for(got_b.wait(), timeout=_KAFKA_RECEIVE_TIMEOUT_S)

            # Assert: no cross-talk
            # Client A must have received at least one response (its own).
            # correlation_id filtering is the primary cross-talk guard: even if
            # a message is redelivered, the correlation_id loop below ensures
            # only messages matching correlation_a are accepted. >= 1 is used
            # instead of == 1 to allow for Kafka's at-least-once redelivery
            # (e.g., consumer rebalance during CI can cause count > 1) without
            # producing spurious failures.
            assert len(received_a) >= 1, (
                f"Client A must have received at least one response, got {len(received_a)}"
            )
            for resp in received_a:
                assert resp.correlation_id == correlation_a, (
                    f"Client A received a response with wrong correlation_id: "
                    f"{resp.correlation_id} (expected {correlation_a})"
                )

            # Client B must have received at least one response (its own).
            # Same rationale as Client A above: correlation_id filtering is the
            # primary cross-talk guard, and >= 1 accommodates Kafka at-least-once
            # redelivery without spurious CI failures.
            assert len(received_b) >= 1, (
                f"Client B must have received at least one response, got {len(received_b)}"
            )
            for resp in received_b:
                assert resp.correlation_id == correlation_b, (
                    f"Client B received a response with wrong correlation_id: "
                    f"{resp.correlation_id} (expected {correlation_b})"
                )

            # The two sets of correlation_ids must not overlap
            a_ids = {r.correlation_id for r in received_a}
            b_ids = {r.correlation_id for r in received_b}
            assert a_ids.isdisjoint(b_ids), (
                f"Cross-talk detected! Client A and B share correlation IDs: "
                f"{a_ids & b_ids}"
            )

            logger.info(
                "Multi-client no-cross-talk test passed: "
                "Client A received %d messages, Client B received %d messages",
                len(received_a),
                len(received_b),
            )

        finally:
            await unsub_a()
            await unsub_b()


# =============================================================================
# Suite 2: Response determinism
# =============================================================================


class TestResponseDeterminism:
    """Two consecutive queries with no registry changes must return identical results.

    Response determinism is a contract of ServiceTopicCatalog: given the same
    catalog state in Consul, the topics tuple must be identical (same order,
    same entries, same count).
    """

    @pytest.mark.serial
    async def test_consecutive_queries_return_identical_results(
        self,
        catalog_handler: HandlerTopicCatalogQuery,
    ) -> None:
        """Two queries in a row with no catalog changes produce identical responses.

        Steps:
        1. Issue query 1 with correlation_id_1
        2. Issue query 2 with correlation_id_2 (different ID, same parameters)
        3. Assert: topics tuples are identical (same order, same entries)
        4. Assert: catalog_version is identical in both responses
        5. Assert: node_count is identical in both responses
        6. Assert: warnings are identical in both responses
        """
        correlation_1 = uuid4()
        correlation_2 = uuid4()

        envelope_1 = _make_query_envelope(
            correlation_1,
            client_id="determinism-test-1",
            include_inactive=True,
        )
        envelope_2 = _make_query_envelope(
            correlation_2,
            client_id="determinism-test-2",
            include_inactive=True,
        )

        output_1 = await catalog_handler.handle(envelope_1)
        output_2 = await catalog_handler.handle(envelope_2)

        assert len(output_1.events) == 1
        assert len(output_2.events) == 1

        response_1 = output_1.events[0]
        response_2 = output_2.events[0]

        assert isinstance(response_1, ModelTopicCatalogResponse)
        assert isinstance(response_2, ModelTopicCatalogResponse)

        # Verify correlation_id pairing (not checked for equality - they differ by design)
        assert response_1.correlation_id == correlation_1
        assert response_2.correlation_id == correlation_2

        # Core determinism assertions
        assert response_1.catalog_version == response_2.catalog_version, (
            f"catalog_version must be identical on consecutive queries: "
            f"{response_1.catalog_version} != {response_2.catalog_version}"
        )
        assert response_1.node_count == response_2.node_count, (
            f"node_count must be identical on consecutive queries: "
            f"{response_1.node_count} != {response_2.node_count}"
        )
        assert len(response_1.topics) == len(response_2.topics), (
            f"topic count must be identical: "
            f"{len(response_1.topics)} != {len(response_2.topics)}"
        )
        assert response_1.topics == response_2.topics, (
            "topics tuple must be identical on consecutive queries (order and content)"
        )
        assert response_1.warnings == response_2.warnings, (
            f"warnings must be identical on consecutive queries: "
            f"{response_1.warnings!r} != {response_2.warnings!r}"
        )

        logger.info(
            "Response determinism test passed: "
            "catalog_version=%d, topic_count=%d, node_count=%d",
            response_1.catalog_version,
            len(response_1.topics),
            response_1.node_count,
        )

    @pytest.mark.serial
    async def test_topic_ordering_is_alphabetical(
        self,
        catalog_handler: HandlerTopicCatalogQuery,
    ) -> None:
        """Topics in the response are sorted alphabetically by topic_suffix.

        ServiceTopicCatalog.build_catalog() sorts entries by topic_suffix.
        This test verifies that ordering is deterministic and alphabetical.
        """
        envelope = _make_query_envelope(
            uuid4(),
            client_id="ordering-test",
            include_inactive=True,
        )
        output = await catalog_handler.handle(envelope)

        assert len(output.events) == 1
        response = output.events[0]
        assert isinstance(response, ModelTopicCatalogResponse)

        if len(response.topics) >= 2:
            suffixes = [t.topic_suffix for t in response.topics]
            assert suffixes == sorted(suffixes), (
                f"Topics must be sorted alphabetically by topic_suffix. Got: {suffixes}"
            )
            logger.info(
                "Topic ordering test passed: %d topics in alphabetical order",
                len(response.topics),
            )
        else:
            pytest.skip("fewer than 2 topics available; cannot verify ordering")


# =============================================================================
# Suite 3: Version-gap recovery simulation
# =============================================================================


class TestVersionGapRecovery:
    """Simulate a dashboard detecting a version gap and triggering re-query.

    Option B's catalog_version enables clients to detect when they may have
    missed a change notification (version gap). This suite proves the pattern
    works end-to-end.
    """

    @pytest.mark.serial
    async def test_version_gap_detection_and_recovery(
        self,
        catalog_service: ServiceTopicCatalog,
        catalog_handler: HandlerTopicCatalogQuery,
    ) -> None:
        """Simulate version gap: bump twice, detect gap, re-query gets current state.

        Steps:
        1. Record initial catalog_version via get_catalog_version()
        2. Bump version twice via increment_version() (without emitting changed events)
        3. Check new_version >= initial_version + 2 (gap exists)
        4. Re-query via handler
        5. Assert: response catalog_version matches new_version (or higher)
        6. Assert: no error warnings in response (healthy state)
        """
        correlation_id = uuid4()

        # Step 1: record initial version
        initial_version = await catalog_service.get_catalog_version(correlation_id)
        # -1 means key absent; treat as 0 for gap arithmetic
        if initial_version == -1:
            initial_version = 0

        logger.info("Version gap test: initial catalog_version=%d", initial_version)

        # Step 2: bump version twice (simulating two registry changes without
        # emitting ModelTopicCatalogChanged events — i.e., a gap scenario)
        version_after_first = await catalog_service.increment_version(correlation_id)
        version_after_second = await catalog_service.increment_version(correlation_id)

        # increment_version returns -1 on CAS exhaustion; skip if Consul KV not writable
        if version_after_first == -1 or version_after_second == -1:
            pytest.skip(
                "increment_version returned -1 (Consul KV not writable or CAS failure). "
                "This test requires writable Consul KV."
            )

        logger.info(
            "Version gap test: bumped to %d then %d",
            version_after_first,
            version_after_second,
        )

        # Step 3: verify gap exists via delta-based checks.
        # Using per-increment assertions instead of a single
        # version_after_second >= initial_version + 2 check, which is
        # susceptible to false failures under parallel test execution: a
        # concurrent test could increment the shared Consul KV version between
        # reading initial_version and the first bump, making the absolute bound
        # incorrect.  Delta checks validate that each call actually advanced
        # the counter regardless of what concurrent tests do to the shared key.
        assert version_after_first >= initial_version + 1, (
            "first increment should advance version"
        )
        assert version_after_second >= version_after_first + 1, (
            "second increment should advance version"
        )

        # Step 4: dashboard detects gap, triggers re-query
        recovery_correlation_id = uuid4()
        recovery_envelope = _make_query_envelope(
            recovery_correlation_id,
            client_id="dashboard-recovery",
            include_inactive=True,
        )
        recovery_output = await catalog_handler.handle(recovery_envelope)

        assert len(recovery_output.events) == 1
        recovery_response = recovery_output.events[0]
        assert isinstance(recovery_response, ModelTopicCatalogResponse)

        # Step 5: response must reflect current version (>= what we bumped to)
        assert recovery_response.catalog_version >= version_after_second, (
            f"Recovery response catalog_version ({recovery_response.catalog_version}) "
            f"must be >= bumped version ({version_after_second})"
        )

        # Step 6: no error warnings (gap recovery is healthy)
        error_warnings = [
            w
            for w in recovery_response.warnings
            if w in ("internal_error", "invalid_query_payload", "no_consul_handler")
        ]
        assert error_warnings == [], (
            f"Recovery response must not contain error warnings: {error_warnings}"
        )

        logger.info(
            "Version gap recovery test passed: initial=%d, after_bumps=%d, "
            "recovery_version=%d, gap=%d",
            initial_version,
            version_after_second,
            recovery_response.catalog_version,
            version_after_second - initial_version,
        )


# =============================================================================
# Suite 5: Integration golden path
# =============================================================================


class TestIntegrationGoldenPath:
    """Golden path: query -> response -> verify expected properties.

    This suite exercises the complete query-response flow using in-process
    handlers against real Consul. It validates the core contracts that
    downstream clients depend on, including topic pattern filtering via
    fnmatch and the structural invariants of ModelTopicCatalogResponse
    (correlation_id pairing, catalog_version, topic suffix format).
    """

    @pytest.mark.serial
    async def test_golden_path_query_response(
        self,
        catalog_handler: HandlerTopicCatalogQuery,
    ) -> None:
        """Handler returns a valid response with correct structural properties.

        Steps:
        1. Publish ModelTopicCatalogQuery (via handler, not Kafka)
        2. Receive ModelTopicCatalogResponse
        3. Verify warnings is empty OR contains only non-error warnings
        4. Verify catalog_version >= 0
        5. Verify all topics have valid topic_suffix (non-empty)
        6. Verify response correlation_id matches query correlation_id
        """
        correlation_id = uuid4()
        envelope = _make_query_envelope(
            correlation_id,
            client_id="golden-path-test",
        )

        output = await catalog_handler.handle(envelope)

        assert len(output.events) == 1, "Handler must return exactly one event"
        response = output.events[0]
        assert isinstance(response, ModelTopicCatalogResponse)

        # Correlation pairing
        assert response.correlation_id == correlation_id, (
            "Response correlation_id must match query correlation_id"
        )

        # Catalog version must be non-negative
        assert response.catalog_version >= 0, (
            f"catalog_version must be >= 0, got {response.catalog_version}"
        )

        # No error-class warnings (infra-level errors would indicate unhealthy state)
        error_warnings = [
            w
            for w in response.warnings
            if w in ("internal_error", "invalid_query_payload", "no_consul_handler")
        ]
        assert error_warnings == [], (
            f"Response must not contain error-class warnings: {error_warnings}"
        )

        # All topic entries must have non-empty suffixes and names
        for entry in response.topics:
            assert entry.topic_suffix, (
                "Every topic entry must have a non-empty topic_suffix"
            )
            assert entry.topic_name, (
                "Every topic entry must have a non-empty topic_name"
            )
            assert entry.partitions >= 1, (
                f"Topic {entry.topic_suffix!r} must have partitions >= 1"
            )

        logger.info(
            "Golden path test passed: catalog_version=%d, topic_count=%d, warnings=%r",
            response.catalog_version,
            len(response.topics),
            response.warnings,
        )

    @pytest.mark.serial
    async def test_golden_path_with_topic_pattern_filter(
        self,
        catalog_handler: HandlerTopicCatalogQuery,
    ) -> None:
        """Query with topic_pattern filter returns only matching topics.

        Verifies that all returned topics match the provided fnmatch pattern.
        This validates the filter path in ServiceTopicCatalog._filter_response().
        """
        correlation_id = uuid4()
        # Pattern matching only platform event topics
        pattern = "onex.evt.platform.*"

        envelope = _make_query_envelope(
            correlation_id,
            client_id="pattern-filter-test",
            topic_pattern=pattern,
            include_inactive=True,
        )
        output = await catalog_handler.handle(envelope)

        assert len(output.events) == 1
        response = output.events[0]
        assert isinstance(response, ModelTopicCatalogResponse)
        assert response.correlation_id == correlation_id

        # Requires Consul catalog to be pre-populated with topics matching 'onex.evt.platform.*'
        if len(response.topics) == 0:
            pytest.skip(
                "Consul catalog has no topics matching 'onex.evt.platform.*'; "
                "ensure integration environment is seeded"
            )
        for entry in response.topics:
            assert fnmatch(entry.topic_suffix, pattern), (
                f"Topic {entry.topic_suffix!r} does not match pattern {pattern!r}. "
                "topic_pattern filter is not working correctly."
            )

        logger.info(
            "Pattern filter test passed: pattern=%r, matching_topics=%d",
            pattern,
            len(response.topics),
        )
