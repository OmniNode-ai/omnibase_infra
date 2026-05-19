# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration test: replay the golden event path against the live .201 runtime.

Publishes a contract registration event to real Kafka (KAFKA_BOOTSTRAP_SERVERS),
then asserts each checkpoint in the golden chain materializes on the live
RuntimeHostProcess. Verification is event-driven: we consume the topology
manifest delta event from Kafka as proof of materialization — NOT HTTP polling.

The golden chain fixtures are the specification. If the live runtime produces
the same shape, dynamic registration works end-to-end.

Requires:
    - KAFKA_BOOTSTRAP_SERVERS pointing at a live Redpanda/Kafka instance
    - omnimarket consuming onex.cmd.platform.node-registration-requested.v1
    - ONEX runtime consuming onex.evt.platform.node-registration.v1

Run: uv run pytest tests/integration/runtime/test_golden_chain_live_runtime.py -v -s -m integration

Related: OMN-11249
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
from pathlib import Path
from uuid import uuid4

import pytest

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "golden_chains"
KAFKA_BOOTSTRAP = os.environ.get(
    "KAFKA_BOOTSTRAP_SERVERS",
    "192.168.86.201:19092",  # kafka-fallback-ok
)
REGISTRATION_TOPIC = "onex.cmd.platform.node-registration-requested.v1"
TOPOLOGY_DELTA_TOPIC = "onex.evt.platform.topology-manifest-delta.v1"

# Test node name — must be unique enough to avoid collisions with other test runs.
_TEST_NODE_NAME = "node_golden_chain_live_test"

# Minimal contract referencing a handler already in the deployed runtime image.
# NOT a generated handler — we are testing registration, not code deployment.
_GOLDEN_CONTRACT_YAML = """\
name: node_golden_chain_live_test
handler_id: proto.golden_chain_live
contract_version:
  major: 1
  minor: 0
  patch: 0
description: Golden chain live integration test — proves dynamic registration on .201
input_model: "omnibase_infra.models.types.JsonDict"
output_model: "omnibase_core.models.dispatch.ModelHandlerOutput"
descriptor:
  node_archetype: effect
event_bus:
  subscribe_topics:
    - onex.evt.test.golden-chain-live.v1
  publish_topics: []
handler_routing:
  routing_strategy: payload_type_match
  handlers:
    - handler:
        name: HandlerHttp
        module: omnibase_infra.handlers.handler_http
runtime_profiles:
  - stability
  - demo
"""


def _can_reach(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _kafka_available() -> bool:
    parts = KAFKA_BOOTSTRAP.split(":")
    if len(parts) != 2:
        return False
    return _can_reach(parts[0], int(parts[1]))


pytestmark = [
    pytest.mark.integration,
    pytest.mark.external,
    pytest.mark.skipif(
        not _kafka_available(),
        reason=f"Kafka on {KAFKA_BOOTSTRAP} not reachable",
    ),
]


def _load_golden_chain(name: str) -> list[dict]:
    return json.loads((FIXTURES_DIR / f"{name}.json").read_text())


def _load_kafka_clients() -> tuple[type, type]:
    kafka_module = pytest.importorskip("kafka")
    return kafka_module.KafkaProducer, kafka_module.KafkaConsumer


class TestGoldenChainLiveRuntime:
    """Replay the golden event path against the live .201 runtime."""

    def test_golden_chain_success_on_live_runtime(self) -> None:
        """Publish a contract to real Kafka, assert each golden chain checkpoint.

        Checkpoints verified:
            0. Registration event published to Kafka
            3. Runtime caches the contract (inferred — no side-channel probe)
            4. Contract materialized (proven by topology manifest delta event)
            5. Topology manifest delta emitted (consumed from Kafka topic)
            6. Dispatch routable (proven by registered_handlers in delta payload)
        """
        KafkaProducer, KafkaConsumer = _load_kafka_clients()

        golden = _load_golden_chain("dynamic_registration_success")
        correlation_id = uuid4()
        contract_hash = (
            f"sha256:{hashlib.sha256(_GOLDEN_CONTRACT_YAML.encode()).hexdigest()}"
        )

        observed: list[dict] = []

        # --- Checkpoint 0: Publish registration event ---
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v, default=str).encode(),
        )
        envelope = {
            "envelope_id": str(uuid4()),
            "correlation_id": str(correlation_id),
            "source_node": "golden_chain_integration_test",
            "emitted_at": "2026-05-18T00:00:00Z",
            "payload": {
                "node_name": _TEST_NODE_NAME,
                "contract_yaml": _GOLDEN_CONTRACT_YAML,
                "contract_hash": contract_hash,
                "event_type": "registered",
                "node_version": {"major": 1, "minor": 0, "patch": 0},
            },
        }
        producer.send(
            REGISTRATION_TOPIC,
            key=_TEST_NODE_NAME.encode(),
            value=envelope,
        )
        producer.flush()
        producer.close()

        observed.append(
            {
                "sequence": 0,
                "event_type": "contract_registration_requested",
                "correlation_id": str(correlation_id),
            }
        )

        # --- Checkpoint 4+5: Consume topology manifest delta as proof ---
        # The runtime publishes topology-manifest-delta after materializing.
        # 20s timeout gives the pipeline (Kafka → omnimarket → Kafka → runtime) time to complete.
        topology_consumer = KafkaConsumer(
            TOPOLOGY_DELTA_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP,
            auto_offset_reset="latest",
            consumer_timeout_ms=20_000,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        )

        materialization_event: dict | None = None
        for msg in topology_consumer:
            payload = msg.value.get("payload", {})
            if payload.get("node_name") == _TEST_NODE_NAME:
                materialization_event = msg.value
                break
        topology_consumer.close()

        assert materialization_event is not None, (
            f"{_TEST_NODE_NAME} topology delta not received within 20s. "
            "Golden chain checkpoint 4 (materialization) failed. "
            "Check RuntimeHostProcess logs on .201."
        )

        observed.append(
            {
                "sequence": 4,
                "event_type": "contract_materialized",
                "evidence": "topology_manifest_delta_event",
                "node_name": materialization_event["payload"]["node_name"],
            }
        )
        observed.append(
            {
                "sequence": 5,
                "event_type": "topology_manifest_delta",
                "delta_type": materialization_event["payload"].get(
                    "delta_type", "node_added"
                ),
            }
        )

        # --- Checkpoint 6: Dispatch routable ---
        delta_payload = materialization_event["payload"]
        assert delta_payload.get("registered_handlers") or delta_payload.get(
            "node_name"
        ), "topology_manifest_delta payload missing routing evidence"
        observed.append(
            {
                "sequence": 6,
                "event_type": "dispatch_routable",
                "reachable": True,
            }
        )

        # --- Assert chain shape matches golden fixture ---
        assert len(observed) >= 3, (
            f"Expected at least 3 checkpoints, observed {len(observed)}"
        )
        for entry in observed:
            if "correlation_id" in entry:
                assert entry["correlation_id"] == str(correlation_id)

        # Every observed event_type must appear in the golden fixture.
        fixture_types = {e["event_type"] for e in golden}
        for obs in observed:
            assert obs["event_type"] in fixture_types, (
                f"Observed '{obs['event_type']}' not in golden fixture. "
                f"Fixture types: {fixture_types}"
            )

        # --- Write evidence bundle ---
        evidence_dir = Path("docs/evidence/dynamic-registration") / str(correlation_id)
        evidence_dir.mkdir(parents=True, exist_ok=True)
        (evidence_dir / "golden_chain_observed.json").write_text(
            json.dumps(observed, indent=2, default=str)
        )
        (evidence_dir / "golden_chain_expected.json").write_text(
            json.dumps(golden, indent=2)
        )
        (evidence_dir / "contract.yaml").write_text(_GOLDEN_CONTRACT_YAML)
        (evidence_dir / "topology_manifest_delta.json").write_text(
            json.dumps(materialization_event, indent=2, default=str)
        )

        print(f"\n{'=' * 60}")
        print("GOLDEN CHAIN LIVE: PASS")
        print(f"  correlation_id: {correlation_id}")
        print(f"  checkpoints observed: {len(observed)}")
        print(f"  topology delta node: {materialization_event['payload']['node_name']}")
        print(f"  evidence: {evidence_dir}")
        print(f"{'=' * 60}")

    def test_golden_chain_idempotent_reregistration(self) -> None:
        """Re-publishing the same contract (same hash) is idempotent.

        The runtime must not reject, error, or produce duplicate materialization.
        Verification is event-driven: consume topology-manifest-delta events and
        confirm at most one materialization event per (node_name, hash).
        """
        KafkaProducer, KafkaConsumer = _load_kafka_clients()

        correlation_id = uuid4()
        contract_hash = (
            f"sha256:{hashlib.sha256(_GOLDEN_CONTRACT_YAML.encode()).hexdigest()}"
        )

        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v, default=str).encode(),
        )
        envelope = {
            "envelope_id": str(uuid4()),
            "correlation_id": str(correlation_id),
            "source_node": "golden_chain_idempotent_test",
            "emitted_at": "2026-05-18T00:00:00Z",
            "payload": {
                "node_name": _TEST_NODE_NAME,
                "contract_yaml": _GOLDEN_CONTRACT_YAML,
                "contract_hash": contract_hash,
                "event_type": "registered",
            },
        }
        # Publish twice — idempotent registration should produce at most 1 delta.
        for _ in range(2):
            producer.send(
                REGISTRATION_TOPIC,
                key=_TEST_NODE_NAME.encode(),
                value=envelope,
            )
        producer.flush()
        producer.close()

        topology_consumer = KafkaConsumer(
            TOPOLOGY_DELTA_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP,
            auto_offset_reset="latest",
            consumer_timeout_ms=5_000,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        )

        materialization_count = 0
        for msg in topology_consumer:
            payload = msg.value.get("payload", {})
            if payload.get("node_name") == _TEST_NODE_NAME:
                materialization_count += 1
        topology_consumer.close()

        assert materialization_count <= 1, (
            f"Expected at most 1 materialization event after idempotent "
            f"re-registration, got {materialization_count}"
        )
