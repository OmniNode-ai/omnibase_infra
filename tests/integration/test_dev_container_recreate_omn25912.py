# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Publish the recreate event from the checkout's real node contract (OMN-25912).

The topic map and the match tolerance come from the node's own ``contract.yaml``; only the
bus is absent, so the sink records what the contract-declared topic would carry.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.enums.enum_dev_recreate_cause import EnumDevRecreateCause
from omnibase_infra.handlers.handler_dev_container_recreate import (
    HandlerDevContainerRecreate,
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
    assert publisher.container_recreated(event) is True
    assert sent[0][0] == topics["dev_container_recreated"]
    assert sent[0][2]["same_build"] is True
