# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A principal-templated reply topic reaches a subscriber on the real in-memory bus (OMN-22103)."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import BaseModel

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.runtime.service_dispatch_result_applier import (
    build_contract_result_applier,
)

pytestmark = pytest.mark.integration

TERMINAL = "onex.cmd.omniintelligence.context-compaction-package.v1"
ALICE = "onex.evt.omnimarket.context-compaction-prepared-for-dev-alice-laptop.v1"
CONTRACT = """\
published_events:
  - event_type: Prepared
    topic_parts:
      kind: evt
      producer: omnimarket
      name_prefix: context-compaction-prepared-for-
      version: 1
      principal_pattern: "^[a-z0-9][a-z0-9-]{0,62}$"
"""


class Reply(BaseModel):
    value: str = "reply"


async def deliver(tmp_path: Path, emitted_topic: str) -> EventBusInmemory:
    contract = tmp_path / "contract.yaml"
    contract.write_text(CONTRACT)
    bus = EventBusInmemory()
    await bus.start()
    try:
        applier = build_contract_result_applier(
            event_bus=bus,
            contract_path=contract,
            publish_topics=[TERMINAL],
            terminal_event=TERMINAL,
        )
        await applier.apply(
            ModelDispatchResult(
                dispatch_id=uuid4(),
                status="success",
                topic="onex.cmd.omniintelligence.context-compaction-requested.v1",
                started_at="2026-01-01T00:00:00Z",
                output_events=[
                    ModelEventEnvelope(
                        payload=Reply(),
                        correlation_id=uuid4(),
                        event_type=emitted_topic,
                    )
                ],
            )
        )
    except BaseException:
        await bus.close()
        raise
    return bus


@pytest.mark.asyncio
async def test_the_requesters_reply_topic_is_delivered_on_the_real_bus(
    tmp_path: Path,
) -> None:
    bus = await deliver(tmp_path, ALICE)
    try:
        assert len(await bus.get_event_history(topic=ALICE)) == 1
        assert await bus.get_event_history(topic=TERMINAL) == []
    finally:
        await bus.close()
