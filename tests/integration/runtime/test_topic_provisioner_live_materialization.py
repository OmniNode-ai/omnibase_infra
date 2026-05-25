# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for live materialization topic provisioning."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = [pytest.mark.integration]

_PATCH_TARGET = "omnibase_infra.event_bus.service_topic_manager.TopicProvisioner"
_TEST_KAFKA_BOOTSTRAP = "192.168.86.201:19092"  # kafka-fallback-ok


def _make_kafka_bus_mock() -> MagicMock:
    from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka

    mock = MagicMock(spec=EventBusKafka)
    mock._bootstrap_servers = _TEST_KAFKA_BOOTSTRAP
    return mock


def _make_descriptor_mock(subscribe_topics: list[str]) -> MagicMock:
    descriptor = MagicMock()
    descriptor.contract_config = {
        "event_bus": {
            "version": {"major": 1, "minor": 0, "patch": 0},
            "subscribe_topics": subscribe_topics,
        }
    }
    return descriptor


@pytest.mark.asyncio
async def test_live_materialization_provisions_each_topic_best_effort(
    tmp_path: Path,
) -> None:
    """Live materialization provisions every subscribed topic before wiring."""
    from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess

    contracts_dir = tmp_path / "contracts"
    contracts_dir.mkdir()

    process = RuntimeHostProcess.__new__(RuntimeHostProcess)
    process._contract_paths = [contracts_dir]
    process._event_bus = _make_kafka_bus_mock()
    process._event_bus_wiring = MagicMock()
    process._event_bus_wiring.wire_subscriptions = AsyncMock()

    attempted_topics: list[str] = []

    async def _ensure_topic_exists(topic_name: str, **kwargs: object) -> bool:
        attempted_topics.append(topic_name)
        if topic_name == "onex.evt.test.topic-a.v1":
            raise RuntimeError("broker rejected topic-a")
        return True

    provisioner = AsyncMock()
    provisioner.ensure_topic_exists = AsyncMock(side_effect=_ensure_topic_exists)

    descriptor = _make_descriptor_mock(
        ["onex.evt.test.topic-a.v1", "onex.evt.test.topic-b.v1"]
    )

    with patch(_PATCH_TARGET, MagicMock(return_value=provisioner)):
        await process._wire_live_handler_subscriptions(
            node_name="test-handler",
            descriptor=descriptor,
        )

    assert attempted_topics == [
        "onex.evt.test.topic-a.v1",
        "onex.evt.test.topic-b.v1",
    ]
    process._event_bus_wiring.wire_subscriptions.assert_awaited_once()
