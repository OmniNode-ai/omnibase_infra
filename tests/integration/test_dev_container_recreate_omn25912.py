# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Publish the recreate event from the checkout's real node contract (OMN-25912).

The topic map and the match tolerance come from the node's own ``contract.yaml``; only the
bus is absent, so the sink records what the contract-declared topic would carry.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.enums.enum_dev_recreate_cause import EnumDevRecreateCause
from omnibase_infra.handlers.dev_build_health.handler_dev_build_health import (
    HandlerDevBuildHealth,
)
from omnibase_infra.handlers.dev_container_recreate.handler_dev_container_recreate import (
    HandlerDevContainerRecreate,
)
from omnibase_infra.models.model_dev_build_health_request import (
    ModelDevBuildHealthRequest,
)
from omnibase_infra.models.model_dev_container_reading import ModelDevContainerReading
from omnibase_infra.models.model_dev_container_recreate_request import (
    ModelDevContainerRecreateRequest,
)
from omnibase_infra.nodes.node_dev_roll_events_effect.dev_roll_events import (
    DevRollEventsPublisher,
    audit_settings,
)

pytestmark = pytest.mark.integration

_CONTRACT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_dev_roll_events_effect"
    / "contract.yaml"
)
_BUILD = "sha256:" + "a" * 64


def _reading(container_id: str, started_at: datetime) -> ModelDevContainerReading:
    return ModelDevContainerReading(
        container="omninode-runtime",
        container_digest=container_id,
        started_at=started_at,
        restart_count=0,
        exit_code=0,
        image_digest=_BUILD,
        image_ref="omninode-runtime:dev",
    )


def test_same_build_recreate_is_published_on_the_contract_declared_topic() -> None:
    contract: dict[str, Any] = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
    topics = contract["topics"]
    event = (
        HandlerDevContainerRecreate()
        .handle(
            ModelDevContainerRecreateRequest(
                reading=_reading("c2", datetime(2026, 11, 23, 22, 53, 58, tzinfo=UTC)),
                previous=_reading("c1", datetime(2026, 11, 23, 21, 14, 35, tzinfo=UTC)),
                start_match_tolerance_seconds=audit_settings(_CONTRACT)[1],
            )
        )
        .event
    )
    assert event is not None
    assert event.cause is EnumDevRecreateCause.UNKNOWN
    assert event.same_build is True

    sent: list[tuple[str, str, dict[str, Any]]] = []
    publisher = DevRollEventsPublisher(
        lambda topic, key, payload: sent.append((topic, key, payload)) or True, topics
    )
    assert publisher.publish_observed("dev_container_recreated", event) is True
    assert sent[0][0] == topics["dev_container_recreated"]
    assert sent[0][2]["same_build"] is True


def test_build_health_known_good_and_known_bad_are_published_on_the_contract_topic() -> (
    None
):
    """OMN-26703: the contract's own limit judges a build, and the verdict rides its declared topic."""
    contract: dict[str, Any] = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
    topics = contract["topics"]
    limit = int(contract["settings"]["build_health_limit_seconds"])
    started = datetime(2026, 11, 26, 19, 24, 53, tzinfo=UTC)
    sent: list[tuple[str, str, dict[str, Any]]] = []
    publisher = DevRollEventsPublisher(
        lambda topic, key, payload: sent.append((topic, key, payload)) or True, topics
    )

    def judge(healthy_after_seconds: int, order: str) -> Any:
        reading = _reading("c3", started).model_copy(
            update={
                "image_revision": "9f77818ea0c96bc05ecffcbbfb621204bed0e6fa",
                "health_status": "healthy",
                "first_healthy_at": started + timedelta(seconds=healthy_after_seconds),
            }
        )
        return HandlerDevBuildHealth().handle(
            ModelDevBuildHealthRequest(
                reading=reading,
                previous_image_revision="2b0c718bb7a89f25e69b5adaaa0f3949cda86543",
                revision_order=order,
                previous_no_rollback_since=None,
                limit_seconds=limit,
            )
        )

    good = judge(14, "forward")
    bad = judge(limit + 60, "back")
    assert good is not None and bad is not None
    assert publisher.publish_observed("dev_build_health", good) is True
    assert publisher.publish_observed("dev_build_health", bad) is True

    assert [topic for topic, _, _ in sent] == [topics["dev_build_health"]] * 2
    assert (sent[0][2]["under_limit"], sent[0][2]["rolled_back"]) == (True, False)
    assert (sent[1][2]["under_limit"], sent[1][2]["rolled_back"]) == (False, True)
