# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26512: the loop reading publisher's run line, against the real contract.

Publishes one reading through the real publisher and the in-memory event bus,
on the topic the on-disk node contract declares, and reads the run line.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import yaml

from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.handlers import handler_runtime_loop_lag_sampler as handler_module
from omnibase_infra.handlers.handler_runtime_loop_lag_sampler import (
    RuntimeLoopLagSampler,
    RuntimeLoopReadingPublisher,
)
from omnibase_infra.models import ModelNodeIdentity

pytestmark = pytest.mark.integration

_NODE_CONTRACT = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes/node_runtime_loop_reading_effect/contract.yaml"
)
_IDENTITY = ModelNodeIdentity(
    env="dev", service="omnibase-infra", node_name="runtime-main", version="v1"
)


async def test_publisher_on_inmemory_bus_logs_run_line_for_contract_topic(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(handler_module, "_PROCESS_SAMPLER", RuntimeLoopLagSampler())
    topic = yaml.safe_load(_NODE_CONTRACT.read_text())["event_bus"]["publish_topics"][0]
    bus = EventBusInmemory()
    await bus.start()
    publisher = RuntimeLoopReadingPublisher()
    try:
        await publisher.start(
            cast("Any", bus), _IDENTITY, MagicMock(), window_seconds=3600
        )
        with caplog.at_level(logging.INFO, logger=handler_module.__name__):
            await publisher.publish_reading(cast("Any", bus), _IDENTITY)
    finally:
        await publisher.stop()
        await bus.close()
    lines = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("Published runtime loop reading")
    ]
    assert len(lines) == 1
    assert f"topic={topic}" in lines[0]
