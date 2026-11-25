# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26512: a published loop reading leaves one run line in the runtime log.

The publisher of OMN-26249 logged nothing when a window's reading was
published, so a runtime that published every window read the same in its log
as one that never started the publisher. The contract's own topic names the
publish; this test reads the log line the publisher writes for it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from omnibase_infra.handlers import handler_runtime_loop_lag_sampler as handler_module
from omnibase_infra.handlers.handler_runtime_loop_lag_sampler import (
    RuntimeLoopLagSampler,
    RuntimeLoopReadingPublisher,
)
from omnibase_infra.models import ModelNodeIdentity

pytestmark = pytest.mark.unit

_NODE_CONTRACT = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes/node_runtime_loop_reading_effect/contract.yaml"
)
_IDENTITY = ModelNodeIdentity(
    env="dev", service="omnibase-infra", node_name="runtime-main", version="v1"
)


class _Bus:
    def __init__(self, failure: Exception | None = None) -> None:
        self.publish_envelope = AsyncMock(side_effect=failure)


@pytest.fixture
def sampler(monkeypatch: pytest.MonkeyPatch) -> RuntimeLoopLagSampler:
    fresh = RuntimeLoopLagSampler()
    monkeypatch.setattr(handler_module, "_PROCESS_SAMPLER", fresh)
    return fresh


async def _publish_one(bus: _Bus) -> None:
    publisher = RuntimeLoopReadingPublisher()
    await publisher.start(cast("Any", bus), _IDENTITY, MagicMock(), window_seconds=3600)
    try:
        await publisher.publish_reading(cast("Any", bus), _IDENTITY)
    finally:
        await publisher.stop()


async def test_a_published_window_logs_one_run_line_naming_the_contracted_topic(
    sampler: RuntimeLoopLagSampler, caplog: pytest.LogCaptureFixture
) -> None:
    contract = yaml.safe_load(_NODE_CONTRACT.read_text())
    topic = contract["event_bus"]["publish_topics"][0]
    bus = _Bus()
    with caplog.at_level(logging.INFO, logger=handler_module.__name__):
        await _publish_one(bus)
    lines = [
        r.getMessage()
        for r in caplog.records
        if r.name == handler_module.__name__
        and r.levelno == logging.INFO
        and r.getMessage().startswith("Published runtime loop reading")
    ]
    assert len(lines) == 1
    assert f"topic={topic}" in lines[0]
    assert "runtime_member=dev.omnibase-infra.runtime-main" in lines[0]
    assert "max_loop_gap_ms=" in lines[0]


async def test_a_refused_publish_logs_the_warning_and_no_run_line(
    sampler: RuntimeLoopLagSampler, caplog: pytest.LogCaptureFixture
) -> None:
    bus = _Bus(failure=OSError("broker down"))
    with caplog.at_level(logging.INFO, logger=handler_module.__name__):
        await _publish_one(bus)
    messages = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("Failed to publish runtime loop reading") for m in messages)
    assert not any(m.startswith("Published runtime loop reading") for m in messages)
