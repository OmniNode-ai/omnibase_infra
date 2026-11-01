# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract topic names receive messages sent using the enum-bound registry."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra import topics
from omnibase_infra.enums import generated
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models import ModelEventMessage
from omnibase_infra.topics import platform_topic_suffixes as registry
from omnibase_infra.topics.contract_topic_extractor import ContractTopicExtractor

pytestmark = pytest.mark.integration

_BINDINGS = (
    (
        "SUFFIX_INTELLIGENCE_LLM_CALL_COMPLETED",
        "EnumOmniintelligenceTopic",
        "EVT_LLM_CALL_COMPLETED_V1",
    ),
    (
        "SUFFIX_OMNIMEMORY_POLICY_STATE_UPDATED",
        "EnumOmnimemoryTopic",
        "EVT_POLICY_STATE_UPDATED_V1",
    ),
    (
        "SUFFIX_VALIDATION_CROSS_REPO_RUN_STARTED",
        "EnumValidationTopic",
        "EVT_CROSS_REPO_RUN_STARTED_V1",
    ),
    (
        "SUFFIX_VALIDATION_CROSS_REPO_VIOLATIONS_BATCH",
        "EnumValidationTopic",
        "EVT_CROSS_REPO_VIOLATIONS_BATCH_V1",
    ),
    (
        "SUFFIX_VALIDATION_CROSS_REPO_RUN_COMPLETED",
        "EnumValidationTopic",
        "EVT_CROSS_REPO_RUN_COMPLETED_V1",
    ),
    (
        "SUFFIX_OMNICLAUDE_NOTIFICATION_BLOCKED",
        "EnumOmniclaudeTopic",
        "EVT_NOTIFICATION_BLOCKED_V1",
    ),
    (
        "SUFFIX_OMNICLAUDE_NOTIFICATION_COMPLETED",
        "EnumOmniclaudeTopic",
        "EVT_NOTIFICATION_COMPLETED_V1",
    ),
)

# Name the subscriber contract and direction only; the topic is found by
# membership so a reordered contract list does not move the binding.
_SUBSCRIBER_CONTRACTS = (
    ("node_waste_detection_compute", "subscribe_topics"),
    ("node_reward_binder_effect", "publish_topics"),
    ("node_validation_ledger_projection_compute", "subscribe_topics"),
    ("node_validation_ledger_projection_compute", "subscribe_topics"),
    ("node_validation_ledger_projection_compute", "subscribe_topics"),
    ("node_emit_daemon_runtime", "subscribe_topics"),
    ("node_emit_daemon_runtime", "subscribe_topics"),
)


@pytest.mark.asyncio
async def test_enum_suffixes_reach_contract_subscribers() -> None:
    """Each suffix reaches exactly one topic its contract declares, of all declared."""
    package_root = Path(registry.__file__).resolve().parents[1]
    manifest = ContractTopicExtractor().scan(package_root / "nodes")

    bus = EventBusInmemory()
    await bus.start()
    try:
        for (name, enum_name, member_name), (node, direction) in zip(
            _BINDINGS, _SUBSCRIBER_CONTRACTS, strict=True
        ):
            member = getattr(getattr(generated, enum_name), member_name)
            suffix = getattr(topics, name)
            assert type(suffix) is str
            assert suffix == member.value
            assert suffix in manifest.all_unique_topics
            document = yaml.safe_load(
                (package_root / "nodes" / node / "contract.yaml").read_text(
                    encoding="utf-8"
                )
            )
            contract_topics = document["event_bus"][direction]
            assert suffix in contract_topics, (node, direction, suffix)
            # Subscribe to every topic the contract declares, so the readback
            # shows the suffix routes to its own declaration and no sibling.
            received: list[tuple[str, bytes]] = []

            async def readback(
                message: ModelEventMessage,
                sink: list[tuple[str, bytes]] = received,
            ) -> None:
                sink.append((message.topic, message.value))

            unsubscribes = [
                await bus.subscribe_ungrouped(contract_topic, readback)
                for contract_topic in contract_topics
            ]
            try:
                payload = name.encode()
                await bus.publish(topic=suffix, key=None, value=payload)
                assert received == [(suffix, payload)], (node, received)
            finally:
                for unsubscribe in unsubscribes:
                    await unsubscribe()
    finally:
        await bus.close()
