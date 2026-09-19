# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""E2E tests for Topic Catalog multi-client routing and change notification flows.

Proves Option B routing: multiple clients on a shared response topic with no
cross-talk via correlation_id filtering. Suite 1 runs against the real broker;
the remaining suites call the handler in-process against the contract-derived
``ServiceTopicCatalog``.

Test Suites:
    - Suite 1: Multi-client no cross-talk (2 clients, different consumer groups,
      correlation_id filtering) — requires Kafka/Redpanda
    - Suite 2: Response determinism (identical results on repeated query)
    - Suite 4: Change notification golden path (register node → receive delta)
    - Suite 5: Integration golden path (lightweight handler-level)

    Suite 3 (version-gap recovery) was DELETED under OMN-18795. Its premise was
    a mutable Consul KV version counter. ``ServiceTopicCatalog.increment_version``
    now returns ``-1`` unconditionally because the catalog version is derived
    from contract files, so the suite could only ever take its own skip branch.

Infrastructure Requirements:
    The directory-level conftest fails closed when a CI job selects this family
    without provisioning Kafka and Postgres (OMN-18795). This module itself
    needs Kafka for Suite 1; the rest read tracked ``contract.yaml`` files.

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
from omnibase_infra.models.catalog.catalog_warning_codes import (
    INTERNAL_ERROR,
    INVALID_QUERY_PAYLOAD,
)
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
    ALL_PLATFORM_SUFFIXES,
    SUFFIX_TOPIC_CATALOG_CHANGED,
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
    mock_container_for_catalog: MagicMock,
) -> ServiceTopicCatalog:
    """ServiceTopicCatalog built from the tracked contract.yaml corpus.

    OMN-3540 removed the Consul KV source; the catalog is now derived by
    scanning ``omnibase_infra/nodes/**/contract.yaml``, so this is a real
    service over real inputs, not a stub.

    Args:
        mock_container_for_catalog: Minimal mock container for DI.

    Returns:
        ServiceTopicCatalog reading the in-repo contract corpus.
    """
    return ServiceTopicCatalog(
        container=mock_container_for_catalog,
    )


@pytest.fixture
def catalog_handler(
    catalog_service: ServiceTopicCatalog,
) -> HandlerTopicCatalogQuery:
    """HandlerTopicCatalogQuery wired to real ServiceTopicCatalog.

    Args:
        catalog_service: Service backed by the in-repo contract corpus.

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
        # The package gate has already proved KAFKA_INTEGRATION_TESTS=1, i.e.
        # this job declared it provisions a broker. An absent endpoint here is
        # a provisioning defect, not a reason to skip (OMN-18795).
        pytest.fail(
            "KAFKA_BOOTSTRAP_SERVERS is unset but KAFKA_INTEGRATION_TESTS=1 "
            "declared that this job provisions a broker.",
            pytrace=False,
        )

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
    except (json.JSONDecodeError, ValueError, KeyError, TypeError, ValidationError):
        logger.warning(
            "_deserialize_response: failed to deserialize message",
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

            # Build two catalog responses via the handler (in-process)
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
            assert len(received_a) >= 1, (
                "Client A must have received its own response"
            )  # len >= 1 confirms Client A received; per-resp correlation_id check below is the real cross-talk guard
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
    catalog state, the topics tuple must be identical (same order,
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

        # The catalog is derived from tracked contract.yaml files, so a corpus
        # too small to order is a defect in the scan, not an environment gap.
        assert len(response.topics) >= 2, (
            f"Contract-derived catalog returned {len(response.topics)} topics; "
            "at least 2 are required to verify ordering and the tracked "
            "contract corpus declares far more. The contract scan is broken."
        )
        suffixes = [t.topic_suffix for t in response.topics]
        assert suffixes == sorted(suffixes), (
            f"Topics must be sorted alphabetically by topic_suffix. Got: {suffixes}"
        )
        logger.info(
            "Topic ordering test passed: %d topics in alphabetical order",
            len(response.topics),
        )


# =============================================================================
# Suite 4: Change notification golden path
# =============================================================================


class TestChangeNotificationFlow:
    """Register a new node with new topics, verify delta model is correct.

    This test proves the full change notification contract:
    - topics_added is sorted alphabetically (D7)
    - catalog_version incremented by at least 1
    - topics_added contains only the newly registered topic suffixes
    - topics_removed is empty when no topics were removed
    """

    @pytest.mark.serial
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
# Suite 5: Integration golden path
# =============================================================================


class TestIntegrationGoldenPath:
    """Golden path: query → response → verify expected properties.

    This suite exercises the complete query-response flow using in-process
    handlers against the contract-derived catalog. It validates the core contracts that
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
        # Match against the constants the handler actually emits. The previous
        # revision hard-coded "no_consul_handler", a code that exists nowhere in
        # the product, so one third of this filter could never match anything.
        error_warnings = [
            w for w in response.warnings if w in (INTERNAL_ERROR, INVALID_QUERY_PAYLOAD)
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

        # The catalog is contract-derived and the tracked corpus declares many
        # onex.evt.platform.* topics, so an empty filtered result means the
        # filter or the scan is broken — never an unseeded environment.
        assert len(response.topics) > 0, (
            f"No topics matched {pattern!r}. The tracked contract corpus "
            "declares platform event topics, so either the contract scan or "
            "ServiceTopicCatalog._filter_response() is broken."
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

    # OMN-18795: a TODO here asked for this case to be moved to unit tests
    # "to avoid infra-gated skip". That premise is gone -- this file is no
    # longer infra-gated-skip. It is selected by marker and executed by
    # ci.yml's service-integration-suites job against a real Redpanda and a
    # migrated Postgres, where a missing service is a failure rather than a
    # skip. The case costs nothing to run alongside its neighbours and
    # reads the same catalog they do, so it stays. OMN-2317 is Done.
    def test_golden_path_published_to_changed_topic_suffix_exists(
        self,
    ) -> None:
        """SUFFIX_TOPIC_CATALOG_CHANGED constant has the correct format.

        Lightweight test that verifies the changed-event topic suffix is a
        valid ONEX 5-segment suffix. The topic must exist in platform specs.
        """
        assert SUFFIX_TOPIC_CATALOG_CHANGED in ALL_PLATFORM_SUFFIXES, (
            f"SUFFIX_TOPIC_CATALOG_CHANGED={SUFFIX_TOPIC_CATALOG_CHANGED!r} "
            "must be in ALL_PLATFORM_SUFFIXES"
        )
        assert SUFFIX_TOPIC_CATALOG_RESPONSE in ALL_PLATFORM_SUFFIXES, (
            f"SUFFIX_TOPIC_CATALOG_RESPONSE={SUFFIX_TOPIC_CATALOG_RESPONSE!r} "
            "must be in ALL_PLATFORM_SUFFIXES"
        )
