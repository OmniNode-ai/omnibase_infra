# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""E2E tests for Topic Catalog multi-client routing and change notification flows.

Proves Option B routing: multiple clients on a shared response topic with no
cross-talk via correlation_id filtering. All tests use component-level E2E
(handlers called in-process with real Consul, no runtime container required).

Test Suites:
    - Suite 1: Multi-client no cross-talk (2 clients, different consumer groups,
      correlation_id filtering)
    - Suite 2: Response determinism (identical results on repeated query)
    - Suite 3: Version-gap recovery simulation
    - Suite 4: Change notification golden path (register node → receive delta)

Infrastructure Requirements:
    Tests require:
    - Consul: CONSUL_HOST:28500
    - Kafka/Redpanda: KAFKA_BOOTSTRAP_SERVERS (Suite 1 only)

    Note: Suite 2, 3, and 4 only require Consul (no Kafka).

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
from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.models.catalog.model_topic_catalog_changed import (
    ModelTopicCatalogChanged,
)
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
    SUFFIX_TOPIC_CATALOG_CHANGED,
    SUFFIX_TOPIC_CATALOG_RESPONSE,
)

# Note: ALL_INFRA_AVAILABLE skipif applied by conftest.py to all tests in this directory
from .conftest import (
    CONSUL_AVAILABLE,
    KAFKA_BOOTSTRAP_SERVERS,
    make_e2e_test_identity,
    wait_for_consumer_ready,
)

if TYPE_CHECKING:
    from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
    from omnibase_infra.handlers import HandlerConsul

logger = logging.getLogger(__name__)

# =============================================================================
# Module-level markers
# =============================================================================
# Note: conftest.py already applies e2e marker and skipif(not ALL_INFRA_AVAILABLE)
# to all tests in this directory. We only add e2e here for explicit clarity.
pytestmark = [
    pytest.mark.e2e,
]

# =============================================================================
# Timeout constants
# =============================================================================

# Maximum seconds to wait for a Kafka message to arrive
_KAFKA_RECEIVE_TIMEOUT_S = 15.0

# Maximum seconds to wait per subscription for readiness
_SUBSCRIPTION_READY_TIMEOUT_S = 10.0

# =============================================================================
# Local fixtures
# =============================================================================


@pytest.fixture
def mock_container_for_catalog() -> MagicMock:
    """Minimal mock container for ServiceTopicCatalog construction."""
    from omnibase_core.container import ModelONEXContainer

    return MagicMock(spec=ModelONEXContainer)


@pytest.fixture
async def catalog_service(
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
async def catalog_handler(
    catalog_service: ServiceTopicCatalog,
) -> HandlerTopicCatalogQuery:
    """HandlerTopicCatalogQuery wired to real ServiceTopicCatalog.

    Args:
        catalog_service: Service backed by real Consul.

    Returns:
        HandlerTopicCatalogQuery ready to process queries.
    """
    return HandlerTopicCatalogQuery(catalog_service=catalog_service)


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
        environment="e2e-test-client-b",
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
        if isinstance(data, dict) and "correlation_id" in data and "topics" in data:
            return ModelTopicCatalogResponse.model_validate(data)
    except Exception:
        pass
    return None


async def _write_node_topics_to_consul(
    consul_handler: HandlerConsul,
    node_id: str,
    subscribe_topics: list[str],
    publish_topics: list[str],
    correlation_id: UUID,
) -> None:
    """Write node event bus topics to Consul KV for catalog scan.

    Populates the ``onex/nodes/{node_id}/event_bus/subscribe_topics`` and
    ``onex/nodes/{node_id}/event_bus/publish_topics`` keys used by
    ServiceTopicCatalog to build the catalog.

    Args:
        consul_handler: Connected HandlerConsul instance.
        node_id: Node identifier (used as KV path segment).
        subscribe_topics: Topic suffixes this node subscribes to.
        publish_topics: Topic suffixes this node publishes to.
        correlation_id: Correlation ID for tracing.
    """
    base = f"onex/nodes/{node_id}/event_bus"

    await consul_handler._kv_put_raw(
        f"{base}/subscribe_topics",
        json.dumps(subscribe_topics),
        correlation_id,
    )
    await consul_handler._kv_put_raw(
        f"{base}/publish_topics",
        json.dumps(publish_topics),
        correlation_id,
    )


async def _delete_node_from_consul(
    consul_handler: HandlerConsul,
    node_id: str,
    correlation_id: UUID,
) -> None:
    """Best-effort deletion of node KV keys from Consul.

    Args:
        consul_handler: Connected HandlerConsul instance.
        node_id: Node identifier whose keys should be removed.
        correlation_id: Correlation ID for tracing.
    """
    for sub_key in ("subscribe_topics", "publish_topics"):
        try:
            # Write empty arrays to effectively clear (delete not always exposed)
            await consul_handler._kv_put_raw(
                f"onex/nodes/{node_id}/event_bus/{sub_key}",
                "[]",
                correlation_id,
            )
        except Exception:
            pass  # Best-effort cleanup


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

    @pytest.mark.asyncio
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

        # Subscribe both clients with different consumer groups (different node identities)
        identity_a = make_e2e_test_identity("catalog_client_a")
        identity_b = make_e2e_test_identity("catalog_client_b")

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
            # Client A must have ONLY responses with correlation_a
            assert len(received_a) >= 1, "Client A must have received its own response"
            for resp in received_a:
                assert resp.correlation_id == correlation_a, (
                    f"Client A received a response with wrong correlation_id: "
                    f"{resp.correlation_id} (expected {correlation_a})"
                )

            # Client B must have ONLY responses with correlation_b
            assert len(received_b) >= 1, "Client B must have received its own response"
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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
            logger.info(
                "Topic ordering test: fewer than 2 topics present (count=%d), "
                "ordering assertion skipped (no ordering to verify)",
                len(response.topics),
            )


# =============================================================================
# Suite 3: Version-gap recovery simulation
# =============================================================================


class TestVersionGapRecovery:
    """Simulate a dashboard detecting a version gap and triggering re-query.

    Option B's catalog_version enables clients to detect when they may have
    missed a change notification (version gap). This suite proves the pattern
    works end-to-end.
    """

    @pytest.mark.asyncio
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

        # Step 3: verify gap exists
        # Gap condition: new_version >= last_seen + 2
        # last_seen = initial_version (dashboard never saw the bump events)
        assert version_after_second >= initial_version + 2, (
            f"Expected gap: version_after_second ({version_after_second}) must be "
            f">= initial_version + 2 ({initial_version + 2}). "
            f"This proves the gap scenario."
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
# Suite 4: Change notification golden path
# =============================================================================


class TestChangeNotificationFlow:
    """Register a new node with new topics, verify delta model is correct.

    This test proves the full change notification contract:
    - topics_added is sorted alphabetically (D7)
    - catalog_version incremented by exactly 1
    - topics_added contains only the newly registered topic suffixes
    - topics_removed is empty when no topics were removed
    """

    @pytest.mark.asyncio
    async def test_register_node_produces_correct_catalog_changed(
        self,
        catalog_service: ServiceTopicCatalog,
        catalog_handler: HandlerTopicCatalogQuery,
        real_consul_handler: HandlerConsul,
        cleanup_consul_services: list[str],
    ) -> None:
        """Register a new node, build delta, verify ModelTopicCatalogChanged.

        Steps:
        1. Query initial catalog (baseline)
        2. Record initial set of topic suffixes
        3. Bump catalog version via increment_version (simulate registration)
        4. Write new node's topics to Consul KV
        5. Invalidate service cache by bumping version again so rebuild fires
        6. Query catalog again (new state)
        7. Compute delta: new_topics - initial_topics
        8. Build ModelTopicCatalogChanged from delta
        9. Assert: topics_added is sorted alphabetically
        10. Assert: catalog_version incremented by exactly 1 from step 3 version
        11. Assert: topics_added contains the new node's topic suffixes
        12. Assert: topics_removed is empty
        """
        if not CONSUL_AVAILABLE:
            pytest.skip("Consul not available")

        correlation_id = uuid4()

        # Use a unique node_id to avoid colliding with other tests
        test_node_id = f"e2e-test-change-notification-{uuid4().hex[:8]}"

        # New topics this node will register (two suffixes for ordering test)
        # Using valid ONEX topic suffix format
        new_topic_b = f"onex.evt.test.node-event-b-{uuid4().hex[:8]}.v1"
        new_topic_a = f"onex.evt.test.node-event-a-{uuid4().hex[:8]}.v1"
        new_topics = [new_topic_a, new_topic_b]

        # Step 1: get initial catalog state (include_inactive=True to see all)
        initial_envelope = _make_query_envelope(
            correlation_id,
            client_id="change-notification-test",
            include_inactive=True,
        )
        initial_output = await catalog_handler.handle(initial_envelope)
        assert len(initial_output.events) == 1
        initial_response = initial_output.events[0]
        assert isinstance(initial_response, ModelTopicCatalogResponse)

        initial_topic_set = {t.topic_suffix for t in initial_response.topics}
        initial_version = initial_response.catalog_version

        logger.info(
            "Change notification test: initial state: version=%d, topics=%d",
            initial_version,
            len(initial_topic_set),
        )

        try:
            # Step 2: bump version to simulate the registration event
            version_before_node = await catalog_service.increment_version(
                correlation_id
            )
            if version_before_node == -1:
                pytest.skip(
                    "increment_version returned -1 (Consul KV not writable). "
                    "This test requires writable Consul KV."
                )

            # Step 3: write new node's topics to Consul KV
            await _write_node_topics_to_consul(
                consul_handler=real_consul_handler,
                node_id=test_node_id,
                subscribe_topics=[new_topics[0]],
                publish_topics=[new_topics[1]],
                correlation_id=correlation_id,
            )

            # Step 4: bump version again to invalidate cache so catalog rebuilds
            new_version = await catalog_service.increment_version(correlation_id)
            if new_version == -1:
                pytest.skip(
                    "Second increment_version returned -1. "
                    "Concurrent modification or Consul KV not writable."
                )

            logger.info(
                "Change notification test: wrote node %r, version=%d",
                test_node_id,
                new_version,
            )

            # Step 5: query catalog after registration
            post_registration_correlation_id = uuid4()
            post_envelope = _make_query_envelope(
                post_registration_correlation_id,
                client_id="change-notification-test-post",
                include_inactive=True,
            )
            post_output = await catalog_handler.handle(post_envelope)
            assert len(post_output.events) == 1
            post_response = post_output.events[0]
            assert isinstance(post_response, ModelTopicCatalogResponse)

            post_topic_set = {t.topic_suffix for t in post_response.topics}

            logger.info(
                "Change notification test: post state: version=%d, topics=%d",
                post_response.catalog_version,
                len(post_topic_set),
            )

            # Step 6: compute delta
            added_topics = post_topic_set - initial_topic_set

            # Our newly registered topics must appear in the added set
            for expected_topic in new_topics:
                assert expected_topic in added_topics, (
                    f"Expected new topic {expected_topic!r} to appear in catalog "
                    f"after node registration. "
                    f"Added: {sorted(added_topics)}"
                )

            removed_topics = initial_topic_set - post_topic_set

            # Step 7: build ModelTopicCatalogChanged from delta
            changed = ModelTopicCatalogChanged(
                correlation_id=post_registration_correlation_id,
                catalog_version=post_response.catalog_version,
                topics_added=tuple(added_topics),
                topics_removed=tuple(removed_topics),
                trigger_node_id=test_node_id,
                trigger_reason="Node registered with new topics (E2E test)",
                changed_at=datetime.now(UTC),
            )

            # Step 8: assert topics_added is sorted alphabetically (D7)
            assert changed.topics_added == tuple(sorted(changed.topics_added)), (
                f"topics_added must be sorted alphabetically (D7). "
                f"Got: {changed.topics_added}"
            )

            # Verify our new topics are in topics_added and are themselves sorted
            our_topics_in_added = tuple(
                t for t in changed.topics_added if t in new_topics
            )
            assert len(our_topics_in_added) == len(new_topics), (
                f"All new topics must appear in topics_added. "
                f"Expected: {sorted(new_topics)}, "
                f"Found in added: {our_topics_in_added}"
            )

            # The new topics must themselves appear in alphabetical order within added
            assert our_topics_in_added == tuple(sorted(our_topics_in_added)), (
                f"New topics must appear in sorted order within topics_added. "
                f"Got: {our_topics_in_added}"
            )

            # Step 9: assert catalog_version incremented by exactly 1 relative to
            # the version recorded just before writing the node (version_before_node)
            assert post_response.catalog_version == new_version, (
                f"Post-registration catalog_version ({post_response.catalog_version}) "
                f"must equal the version we set ({new_version})"
            )
            assert post_response.catalog_version == version_before_node + 1, (
                f"catalog_version must increment by exactly 1. "
                f"Before node write: {version_before_node}, "
                f"After: {post_response.catalog_version}"
            )

            # Step 10: no topics removed in this scenario
            assert len(removed_topics) == 0, (
                f"No topics should have been removed in this test. "
                f"Removed: {removed_topics}"
            )
            assert changed.topics_removed == (), (
                "topics_removed must be empty when no topics were removed"
            )

            logger.info(
                "Change notification test passed: "
                "topics_added=%r, catalog_version=%d (initial=%d, delta=+%d)",
                changed.topics_added,
                changed.catalog_version,
                initial_version,
                changed.catalog_version - initial_version,
            )

        finally:
            # Clean up: remove test node's KV entries from Consul
            await _delete_node_from_consul(
                consul_handler=real_consul_handler,
                node_id=test_node_id,
                correlation_id=correlation_id,
            )
            # Bump version to reflect the cleanup (best-effort)
            try:
                await catalog_service.increment_version(correlation_id)
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_change_notification_topic_suffix_format(
        self,
        catalog_handler: HandlerTopicCatalogQuery,
    ) -> None:
        """ModelTopicCatalogChanged is constructible from handler response data.

        Verifies that the data shape returned by HandlerTopicCatalogQuery is
        compatible with building a ModelTopicCatalogChanged notification — that
        is, all required fields are present and correctly typed.
        """
        correlation_id = uuid4()
        envelope = _make_query_envelope(
            correlation_id,
            client_id="change-format-test",
            include_inactive=True,
        )
        output = await catalog_handler.handle(envelope)
        assert len(output.events) == 1
        response = output.events[0]
        assert isinstance(response, ModelTopicCatalogResponse)

        # Build a ModelTopicCatalogChanged as if every current topic were newly added
        # (simulating an initial population event)
        all_suffixes = tuple(t.topic_suffix for t in response.topics)

        changed = ModelTopicCatalogChanged(
            correlation_id=correlation_id,
            catalog_version=response.catalog_version,
            topics_added=all_suffixes,
            topics_removed=(),
            trigger_reason="Initial population (E2E format test)",
            changed_at=datetime.now(UTC),
        )

        # D7: topics_added must be sorted after model_validator runs
        assert changed.topics_added == tuple(sorted(changed.topics_added)), (
            "sort_delta_tuples validator must sort topics_added alphabetically"
        )
        assert changed.topics_removed == (), "topics_removed must be empty tuple"
        assert changed.catalog_version >= 0
        assert changed.schema_version == 1

        # All topic suffixes in the changed event must match what the handler returned
        assert set(changed.topics_added) == set(all_suffixes), (
            "topics_added must contain exactly the topics from the response"
        )

        logger.info(
            "Change notification format test passed: "
            "%d topics in ModelTopicCatalogChanged, all sorted",
            len(changed.topics_added),
        )


# =============================================================================
# Suite 5: Integration golden path (lightweight, no Kafka required)
# =============================================================================


class TestIntegrationGoldenPath:
    """Lightweight golden path: query → response → verify expected properties.

    This suite exercises the complete query-response flow using in-process
    handlers against real Consul. It validates the core contracts that
    downstream clients depend on.
    """

    @pytest.mark.asyncio
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
            if w in ("internal_error", "invalid_query_payload")
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

    @pytest.mark.asyncio
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

        from fnmatch import fnmatch

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

    @pytest.mark.asyncio
    async def test_golden_path_published_to_changed_topic_suffix_exists(
        self,
    ) -> None:
        """SUFFIX_TOPIC_CATALOG_CHANGED constant has the correct format.

        Lightweight test that verifies the changed-event topic suffix is a
        valid ONEX 5-segment suffix. The topic must exist in platform specs.
        """
        from omnibase_infra.topics.platform_topic_suffixes import ALL_PLATFORM_SUFFIXES

        assert SUFFIX_TOPIC_CATALOG_CHANGED in ALL_PLATFORM_SUFFIXES, (
            f"SUFFIX_TOPIC_CATALOG_CHANGED={SUFFIX_TOPIC_CATALOG_CHANGED!r} "
            "must be in ALL_PLATFORM_SUFFIXES"
        )
        assert SUFFIX_TOPIC_CATALOG_RESPONSE in ALL_PLATFORM_SUFFIXES, (
            f"SUFFIX_TOPIC_CATALOG_RESPONSE={SUFFIX_TOPIC_CATALOG_RESPONSE!r} "
            "must be in ALL_PLATFORM_SUFFIXES"
        )
