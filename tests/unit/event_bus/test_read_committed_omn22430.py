# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Consumers of the host state event topics read committed records only (OMN-22430).

The declaration is a packaged resource, validated on load; both consumer
constructors of the Kafka event bus (grouped ``_build_consumer`` and the
group-less ``subscribe_ungrouped``) take the setting from one helper.
"""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.models.event_bus.model_read_committed_topics import (
    ModelReadCommittedTopics,
)

_PREFIX = "onex.evt.omnimarket"
_EVENT_TOPICS = tuple(
    f"{_PREFIX}.{name}.v1"
    for name in ("host-layer-decided", "host-desired-state", "host-observed-state")
)
_UNLISTED_TOPIC = "onex.evt.omnimarket.something-else.v1"
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _bus() -> EventBusKafka:
    return EventBusKafka(
        config=ModelKafkaEventBusConfig(bootstrap_servers="localhost:9092")
    )


@pytest.mark.unit
class TestSubscribeReadCommitted:
    @pytest.mark.parametrize("topic", _EVENT_TOPICS)
    def test_subscribe_read_committed_grouped_constructor(self, topic: str) -> None:
        consumer_cls = MagicMock()
        with patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer", consumer_cls
        ):
            # first subscription, rebalance recreation and cold replay all
            # build through this one constructor with their own offset reset.
            for reset in ("latest", "earliest"):
                _bus()._build_consumer(topic, "grp", "inst", reset, group_id="grp")
        assert consumer_cls.call_count == 2
        for call in consumer_cls.call_args_list:
            assert call.kwargs["isolation_level"] == "read_committed"

    @pytest.mark.parametrize("topic", _EVENT_TOPICS)
    @pytest.mark.asyncio
    async def test_subscribe_read_committed_ungrouped_constructor(
        self, topic: str
    ) -> None:
        consumer_cls = MagicMock()
        reader = MagicMock()
        reader.start = AsyncMock()
        bus = _bus()
        bus._started = True
        with (
            patch(
                "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer",
                consumer_cls,
            ),
            patch(
                "omnibase_infra.event_bus.event_bus_kafka.KafkaUngroupedReader",
                return_value=reader,
            ),
        ):
            await bus.subscribe_ungrouped(topic, AsyncMock())
        assert consumer_cls.call_args.kwargs["isolation_level"] == "read_committed"

    def test_subscribe_read_committed_leaves_unlisted_topic_alone(self) -> None:
        consumer_cls = MagicMock()
        with patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer", consumer_cls
        ):
            _bus()._build_consumer(
                _UNLISTED_TOPIC, "grp", "inst", "latest", group_id="grp"
            )
        assert "isolation_level" not in consumer_cls.call_args.kwargs


@pytest.mark.unit
class TestReadCommittedDeclaration:
    def test_read_committed_declaration_validated_accepts_the_three_event_topics(
        self,
    ) -> None:
        declared = ModelReadCommittedTopics.load()
        assert set(declared.topics) == set(_EVENT_TOPICS)

    @pytest.mark.parametrize(
        "topics",
        [
            ["not-a-topic"],
            [_UNLISTED_TOPIC],
            [f"{_PREFIX}.host-desired-state.v1", f"{_PREFIX}.host-desired-state.v1"],
            [],
        ],
    )
    def test_read_committed_declaration_validated_refuses(
        self, topics: list[str]
    ) -> None:
        with pytest.raises(ValueError):
            ModelReadCommittedTopics.model_validate(
                {"schema_version": "1.0.0", "topics": topics}
            )

    def test_read_committed_missing_resource_fails_startup(self) -> None:
        with patch(
            "omnibase_infra.models.event_bus.model_read_committed_topics.resources.files",
            side_effect=ModuleNotFoundError("gone"),
        ):
            with pytest.raises(ProtocolConfigurationError):
                _bus()

    def test_read_committed_from_installed_wheel(self, tmp_path: Path) -> None:
        """The resource is inside the built wheel, which ships src/omnibase_infra only."""
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
        )
        wheel = next(tmp_path.glob("*.whl"))
        with zipfile.ZipFile(wheel) as archive:
            assert (
                "omnibase_infra/event_bus/read_committed_topics.yaml"
                in archive.namelist()
            )
