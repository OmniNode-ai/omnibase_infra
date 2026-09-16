# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration proof for the OMN-18418 delegate lane probe request shape."""

from __future__ import annotations

from typing import Any

import pytest

from omnibase_infra.backends.backend_probe import live_consumer_groups

pytestmark = pytest.mark.integration

_TOPIC = "onex.cmd.omnimarket.delegate-skill.v1"
_BROKER = "lane-broker.invalid:19092"


def _bound_group_id(version: str) -> str:
    from omnibase_core.event_bus.util_consumer_group import TOPIC_SCOPE_INFIX

    return (
        "onex-dev.omnimarket.node_delegate_skill_orchestrator.consume."
        f"{version}.__i.runtime-effects{TOPIC_SCOPE_INFIX}{_TOPIC}"
    )


class _DescribeResponse:
    def __init__(self, groups: list[tuple[Any, ...]]) -> None:
        self.groups = groups


class _RecordingAdminClient:
    listed_groups = [
        (_bound_group_id("1.1.0"), "consumer"),
        ("unrelated.__t.onex.cmd.other.v1", "consumer"),
        (_bound_group_id("1.2.0"), "consumer"),
    ]
    describe_calls: list[list[str]] = []
    closed = False

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        _RecordingAdminClient.closed = True

    async def describe_cluster(self) -> dict[str, Any]:
        return {"brokers": [{"node_id": 1}]}

    async def list_consumer_groups(self) -> list[tuple[str, str]]:
        return list(_RecordingAdminClient.listed_groups)

    async def describe_consumer_groups(
        self, group_ids: list[str]
    ) -> list[_DescribeResponse]:
        _RecordingAdminClient.describe_calls.append(list(group_ids))
        return [
            _DescribeResponse(
                [(0, group_id, "Stable", "consumer", "", []) for group_id in group_ids]
            )
        ]


def test_lane_probe_describes_each_candidate_group_individually(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The MSK-proven request shape is one DescribeGroups call per candidate."""
    import aiokafka.admin

    _RecordingAdminClient.describe_calls = []
    _RecordingAdminClient.closed = False
    monkeypatch.setattr(aiokafka.admin, "AIOKafkaAdminClient", _RecordingAdminClient)
    monkeypatch.delenv("KAFKA_SECURITY_PROTOCOL", raising=False)
    monkeypatch.delenv("KAFKA_SASL_MECHANISM", raising=False)

    assert live_consumer_groups(topic=_TOPIC, bootstrap_servers=_BROKER) == (
        _bound_group_id("1.1.0"),
        _bound_group_id("1.2.0"),
    )

    assert _RecordingAdminClient.describe_calls == [
        [_bound_group_id("1.1.0")],
        [_bound_group_id("1.2.0")],
    ]
    assert _RecordingAdminClient.closed is True
