# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract topic names receive messages sent using the enum-bound registry."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

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


@pytest.mark.asyncio
async def test_enum_suffixes_reach_contract_subscribers() -> None:
    """Read real contracts, then read back messages sent using all seven suffixes."""
    package_root = Path(registry.__file__).resolve().parents[1]
    manifest = ContractTopicExtractor().scan(package_root / "nodes")
    assignments = {
        node.target.id: ast.unparse(node.value)
        for node in ast.parse(Path(registry.__file__).read_text()).body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.value is not None
    }
    received: list[tuple[str, bytes]] = []

    async def readback(message: ModelEventMessage) -> None:
        received.append((message.topic, message.value))

    bus = EventBusInmemory()
    await bus.start()
    try:
        for name, enum_name, member_name in _BINDINGS:
            member = getattr(getattr(generated, enum_name), member_name)
            suffix = getattr(topics, name)
            assert type(suffix) is str
            assert suffix == member.value
            assert assignments[name] == f"topic_enums.{enum_name}.{member_name}.value"
            contract_topic = next(
                topic for topic in manifest.all_unique_topics if topic == member.value
            )
            unsubscribe = await bus.subscribe_ungrouped(contract_topic, readback)
            try:
                payload = name.encode()
                receipt = await bus.publish(topic=suffix, key=None, value=payload)
                assert receipt.offset == 0
                assert received[-1] == (contract_topic, payload)
            finally:
                await unsubscribe()
        assert len(received) == len(_BINDINGS)
    finally:
        await bus.close()
