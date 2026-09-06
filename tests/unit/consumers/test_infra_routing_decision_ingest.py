# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16025 — the infra routing-decision projection reads the LIVE topic.

Every test here is a regression guard for one measured link of the break found
on the .201 dev lane on 2026-09-06:

  * ``onex.evt.omnibase-infra.routing-decision.v1`` HIGH-WATERMARK 860
  * ``infra_routing_decisions`` 0 rows
  * the consumer subscribed ``onex.evt.omnibase-infra.routing-decided.v1``,
    a topic that does not exist on the broker
  * and it read the record as a flat decision dict, while the live records are
    ``ModelEventEnvelope``-wrapped with the decision under ``payload``.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from omnibase_infra.services.observability.infra_routing_decisions.config import (
    ConfigInfraRoutingDecisionsConsumer,
)
from omnibase_infra.services.observability.infra_routing_decisions.consumer import (
    InfraRoutingDecisionsConsumer,
)
from omnibase_infra.services.observability.infra_routing_decisions.model_routing_decision_ingest import (
    ModelInfraRoutingDecisionIngest,
)
from omnibase_infra.topics import topic_keys
from omnibase_infra.topics.service_topic_registry import ServiceTopicRegistry

LIVE_TOPIC = "onex.evt.omnibase-infra.routing-decision.v1"
DEAD_TOPIC = "onex.evt.omnibase-infra.routing-decided.v1"


def make_live_envelope(correlation_id: str | None = None) -> dict[str, object]:
    """A verbatim-shaped record from the live topic (dev lane offset 858)."""
    cid = correlation_id or str(uuid4())
    return {
        "payload": {
            "correlation_id": cid,
            "task_type": "test",
            "selected_model": "Qwen3.6-35B-A3B",
            "selected_backend_id": str(uuid4()),
            "endpoint_url": "http://backend.invalid/v1/chat/completions",
            "cost_tier": "low",
            "max_context_tokens": 65536,
            "rationale": "Task 'test' routed to Qwen3.6-35B-A3B via tier 'local'.",
            "tier_name": "local",
            "selected_backend_ref": "local-coder",
        },
        "envelope_id": str(uuid4()),
        "correlation_id": cid,
        "event_type": "omnibase-infra.routing-decision",
    }


def make_record(value: object) -> MagicMock:
    record = MagicMock()
    record.topic = LIVE_TOPIC
    record.partition = 0
    record.offset = 858
    record.value = json.dumps(value).encode("utf-8")
    return record


@pytest.fixture
def consumer() -> InfraRoutingDecisionsConsumer:
    return InfraRoutingDecisionsConsumer(
        ConfigInfraRoutingDecisionsConsumer(
            kafka_bootstrap_servers="localhost:19092",
            postgres_dsn="postgresql://test:pw@localhost:5432/test",
        )
    )


@pytest.mark.unit
class TestSubscribedTopic:
    def test_default_topic_is_the_live_routing_decision_topic(self) -> None:
        config = ConfigInfraRoutingDecisionsConsumer(
            kafka_bootstrap_servers="localhost:19092",
            postgres_dsn="postgresql://test:pw@localhost:5432/test",
        )
        assert config.topics == [LIVE_TOPIC]

    def test_dead_routing_decided_topic_is_not_subscribed(self) -> None:
        config = ConfigInfraRoutingDecisionsConsumer(
            kafka_bootstrap_servers="localhost:19092",
            postgres_dsn="postgresql://test:pw@localhost:5432/test",
        )
        assert DEAD_TOPIC not in config.topics

    def test_default_topic_resolves_from_the_registry_not_a_literal(self) -> None:
        registry = ServiceTopicRegistry.from_defaults()
        config = ConfigInfraRoutingDecisionsConsumer(
            kafka_bootstrap_servers="localhost:19092",
            postgres_dsn="postgresql://test:pw@localhost:5432/test",
        )
        assert config.topics == [
            registry.resolve(topic_keys.DELEGATION_ROUTING_DECISION)
        ]


@pytest.mark.unit
class TestIngestModel:
    def test_maps_wire_names_to_column_names(self) -> None:
        cid = str(uuid4())
        row = ModelInfraRoutingDecisionIngest.from_envelope(make_live_envelope(cid))
        assert row is not None
        assert row.correlation_id == UUID(cid)
        assert row.selected_provider == "local-coder"
        assert row.selected_tier == "local"
        assert row.selected_model == "Qwen3.6-35B-A3B"
        assert row.reason.startswith("Task 'test' routed to")
        assert row.task_type == "test"

    def test_flat_pre_envelope_record_is_rejected(self) -> None:
        """The decision fields at the TOP level is the shape that wrote 0 rows."""
        flat = make_live_envelope()["payload"]
        assert ModelInfraRoutingDecisionIngest.from_envelope(dict(flat)) is None  # type: ignore[arg-type]

    def test_correlation_id_falls_back_to_the_envelope(self) -> None:
        cid = str(uuid4())
        envelope = make_live_envelope(cid)
        payload = envelope["payload"]
        assert isinstance(payload, dict)
        payload.pop("correlation_id")
        row = ModelInfraRoutingDecisionIngest.from_envelope(envelope)
        assert row is not None
        assert row.correlation_id == UUID(cid)

    def test_unparseable_correlation_id_is_rejected_not_nulled(self) -> None:
        envelope = make_live_envelope()
        payload = envelope["payload"]
        assert isinstance(payload, dict)
        payload["correlation_id"] = "not-a-uuid"
        envelope["correlation_id"] = "not-a-uuid"
        assert ModelInfraRoutingDecisionIngest.from_envelope(envelope) is None

    def test_missing_selected_model_is_rejected(self) -> None:
        envelope = make_live_envelope()
        payload = envelope["payload"]
        assert isinstance(payload, dict)
        payload["selected_model"] = ""
        assert ModelInfraRoutingDecisionIngest.from_envelope(envelope) is None

    def test_absent_optional_wire_strings_become_empty_not_none(self) -> None:
        envelope = make_live_envelope()
        payload = envelope["payload"]
        assert isinstance(payload, dict)
        payload.pop("tier_name")
        payload.pop("selected_backend_ref")
        row = ModelInfraRoutingDecisionIngest.from_envelope(envelope)
        assert row is not None
        assert row.selected_tier == ""
        assert row.selected_provider == ""


@pytest.mark.unit
class TestConsumerParsesLiveRecords:
    def test_parse_message_returns_a_typed_row(
        self, consumer: InfraRoutingDecisionsConsumer
    ) -> None:
        cid = str(uuid4())
        result = consumer._parse_message(make_record(make_live_envelope(cid)))
        assert isinstance(result, ModelInfraRoutingDecisionIngest)
        assert result.correlation_id == UUID(cid)
        assert result.selected_model == "Qwen3.6-35B-A3B"

    def test_parse_message_rejects_a_non_envelope_record(
        self, consumer: InfraRoutingDecisionsConsumer
    ) -> None:
        assert consumer._parse_message(make_record({"hello": "world"})) is None

    def test_parse_message_rejects_invalid_json(
        self, consumer: InfraRoutingDecisionsConsumer
    ) -> None:
        record = make_record({})
        record.value = b"not-json"
        assert consumer._parse_message(record) is None
