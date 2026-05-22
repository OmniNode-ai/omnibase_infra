# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Root integration coverage for infra routing decisions topic defaults."""

from __future__ import annotations

import pytest

from omnibase_infra.event_bus.topic_constants import get_dlq_topic_for_original
from omnibase_infra.services.observability.infra_routing_decisions.config import (
    ConfigInfraRoutingDecisionsConsumer,
)
from omnibase_infra.topics import topic_keys
from omnibase_infra.topics.service_topic_registry import ServiceTopicRegistry


@pytest.mark.integration
def test_routing_decisions_config_resolves_registered_topics() -> None:
    registry = ServiceTopicRegistry.from_defaults()
    routing_topic = registry.resolve(topic_keys.ROUTING_DECIDED)

    config = ConfigInfraRoutingDecisionsConsumer(
        kafka_bootstrap_servers="localhost:19092",
        postgres_dsn="postgresql://postgres:postgres@localhost:5432/postgres",
    )

    assert config.topics == [routing_topic]
    assert config.dlq_topic == get_dlq_topic_for_original(routing_topic)