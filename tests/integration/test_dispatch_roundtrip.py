# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for dispatch round-trip (OMN-6998).

Proves the event -> handler -> side-effect chain works end-to-end by:
1. Publishing a node-introspection event to Kafka
2. Verifying consumer lag reaches 0 on the introspection topic (OMN-7236)
3. Verifying the side-effect appears in PostgreSQL (registration projection)

This is the single proof that Phase 1 (foundation) is solid: events flow
through Kafka, reach handlers via the dispatch engine, and produce
observable side-effects.

These tests require the full runtime stack (infra-up-runtime) with
Kafka/Redpanda and PostgreSQL running.

Related:
    - OMN-6995: Platform Subsystem Verification epic
    - OMN-6996: Runtime health integration test
    - OMN-6997: Node registration integration test
    - OMN-7236: Replace log-grep evidence with rpk consumer lag check
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid

import pytest

# All config from env vars — no hardcoded values.
KAFKA_BOOTSTRAP: str = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
POSTGRES_HOST: str = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PORT: str = os.environ.get("POSTGRES_PORT", "5436")
POSTGRES_DB: str = os.environ.get("POSTGRES_DB", "omnibase_infra")
POSTGRES_USER: str = os.environ.get("POSTGRES_USER", "postgres")

# Topic the registration orchestrator subscribes to.
INTROSPECTION_TOPIC: str = os.environ.get(
    "ONEX_INTROSPECTION_TOPIC",
    "onex.evt.platform.node-introspection.v1",
)


def _publish_kafka_event(topic: str, event: dict[str, object]) -> None:
    """Publish a JSON event to Kafka using kafka-python.

    Uses kafka-python (available as integration test dependency) for
    synchronous publishing. The runtime uses aiokafka for async consumption.
    """
    from kafka import KafkaProducer

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v, default=str).encode(),
        request_timeout_ms=10000,
        max_block_ms=10000,
    )
    try:
        future = producer.send(topic, event)
        future.get(timeout=10)
    finally:
        producer.close(timeout=5)


def _get_consumer_group_for_topic(topic: str) -> str | None:
    """Find the consumer group subscribed to a given topic via rpk group list."""
    result = subprocess.run(
        ["docker", "exec", "omnibase-infra-redpanda", "rpk", "group", "list"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if result.returncode != 0:
        return None
    groups = [
        line.strip()
        for line in result.stdout.strip().split("\n")
        if line.strip() and "GROUP" not in line.upper()
    ]
    # Return first runtime-looking group; caller can refine if needed.
    for g in groups:
        if "runtime" in g.lower() or "local." in g.lower():
            return g
    return groups[0] if groups else None


def _get_consumer_lag(group: str, topic: str) -> int | None:
    """Return total LAG for a consumer group on a specific topic, or None on error."""
    result = subprocess.run(
        [
            "docker",
            "exec",
            "omnibase-infra-redpanda",
            "rpk",
            "group",
            "describe",
            group,
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if result.returncode != 0:
        return None
    total_lag = 0
    found_topic = False
    for line in result.stdout.splitlines():
        # rpk output columns: TOPIC  PARTITION  CURRENT-OFFSET  LOG-END-OFFSET  LAG  ...
        parts = line.split()
        if len(parts) >= 5 and parts[0] == topic:
            found_topic = True
            lag_str = parts[4]
            if lag_str.isdigit():
                total_lag += int(lag_str)
    return total_lag if found_topic else None


def _query_postgres(query: str) -> str:
    """Run a psql query and return the output."""
    password = os.environ.get("POSTGRES_PASSWORD", "")
    env = {**os.environ, "PGPASSWORD": password} if password else dict(os.environ)
    result = subprocess.run(
        [
            "psql",
            "-h",
            POSTGRES_HOST,
            "-p",
            POSTGRES_PORT,
            "-U",
            POSTGRES_USER,
            "-d",
            POSTGRES_DB,
            "-t",  # tuples only
            "-A",  # unaligned
            "-c",
            query,
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
        env=env,
    )
    return result.stdout.strip()


@pytest.mark.slow
class TestDispatchRoundtrip:
    """Prove a Kafka event dispatches through a handler and produces a side-effect."""

    def test_registration_projections_exist(self) -> None:
        """The registration projection table must have >0 rows.

        This proves the full chain worked at some point:
        introspection event -> Kafka -> registration orchestrator ->
        intent executor -> PostgreSQL projection.
        """
        # Try registration_projections first (the active projection table),
        # fall back to node_registrations if not available.
        output = _query_postgres("SELECT count(*) FROM registration_projections;")
        if not output or not output.strip().isdigit():
            # Fall back to node_registrations
            output = _query_postgres("SELECT count(*) FROM node_registrations;")
        if not output or not output.strip().isdigit():
            pytest.skip(
                "Cannot query registration projection tables — "
                f"psql returned: {output!r}. "
                "Is PostgreSQL running and accessible?"
            )
        count = int(output.strip())
        assert count > 0, (
            "Registration projection tables have 0 rows. "
            "The dispatch chain (introspection -> registration orchestrator -> "
            "postgres projection) has never completed successfully."
        )

    def test_event_publish_produces_log_evidence(self) -> None:
        """Publishing a test event to Kafka must be consumed (consumer lag reaches 0).

        Publishes a node-introspection event then polls rpk group describe until
        the consumer group's LAG on the introspection topic returns to 0, proving
        the dispatch engine received and processed the event (OMN-7236).
        """
        group = _get_consumer_group_for_topic(INTROSPECTION_TOPIC)
        if group is None:
            pytest.skip(
                "No runtime consumer groups found in Redpanda. "
                "Is the runtime stack running?"
            )

        test_event: dict[str, object] = {
            "event_type": "node_introspection",
            "correlation_id": f"test-dispatch-{uuid.uuid4().hex[:12]}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "payload": {
                "source": "integration-test",
                "node_id": f"test-node-{uuid.uuid4().hex[:8]}",
                "node_type": "COMPUTE_GENERIC",
            },
        }

        try:
            _publish_kafka_event(INTROSPECTION_TOPIC, test_event)
        except ImportError:
            pytest.skip("kafka-python not installed — cannot publish test event")
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"Cannot publish to Kafka at {KAFKA_BOOTSTRAP}: {exc}")

        # Poll until consumer lag on the introspection topic returns to 0.
        deadline = time.time() + 30
        last_lag: int | None = None
        while time.time() < deadline:
            lag = _get_consumer_lag(group, INTROSPECTION_TOPIC)
            if lag is not None:
                last_lag = lag
                if lag == 0:
                    return  # All published messages consumed — dispatch chain live
            time.sleep(2)

        pytest.fail(
            f"Dispatch round-trip failed: consumer group {group!r} lag on "
            f"{INTROSPECTION_TOPIC!r} did not reach 0 within 30s "
            f"(last observed lag: {last_lag!r}). "
            f"Node dispatch layer may be dead or not subscribed to this topic."
        )

    def test_kafka_consumer_groups_active(self) -> None:
        """Runtime Kafka consumer groups must be active (not empty/dead).

        Verifies that the runtime has active consumer groups in Redpanda,
        proving it is connected to the event bus and consuming events.
        """
        result = subprocess.run(
            ["docker", "exec", "omnibase-infra-redpanda", "rpk", "group", "list"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if result.returncode != 0:
            pytest.skip(
                f"Cannot list consumer groups: {result.stderr}. Is Redpanda running?"
            )

        groups = [
            line.strip()
            for line in result.stdout.strip().split("\n")
            if line.strip() and "GROUP" not in line.upper()
        ]
        # Filter for runtime consumer groups (local.runtime_config.*)
        runtime_groups = [
            g for g in groups if "runtime" in g.lower() or "local." in g.lower()
        ]
        assert len(runtime_groups) > 0, (
            f"No runtime consumer groups found in Redpanda. "
            f"Total groups: {len(groups)}. "
            f"The runtime may not be connected to the event bus."
        )
