# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""An orchestrator may answer on a topic of a declared `{principal}` template (OMN-22103).

A compaction command names the per-principal reply topics its answer goes to. The
topic is the requester's, so it cannot be a static contract entry: the contract
declares the template, and the applier accepts an emitted topic only when it is the
template with a well-formed principal in the slot.
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import BaseModel

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.service_dispatch_result_applier import (
    DispatchResultApplier,
)

pytestmark = pytest.mark.unit

TEMPLATE = "onex.evt.omnimarket.context-compaction-prepared-for-{principal}.v1"
TERMINAL = "onex.cmd.omniintelligence.context-compaction-package.v1"
ALICE = "onex.evt.omnimarket.context-compaction-prepared-for-dev-alice-laptop.v1"


class Reply(BaseModel):
    value: str = "reply"


def applier(bus: AsyncMock) -> DispatchResultApplier:
    return DispatchResultApplier(
        event_bus=bus,
        output_topic=TERMINAL,
        allowed_output_topics={TEMPLATE, TERMINAL},
    )


async def published_topic(emitted_topic: str) -> str:
    bus = AsyncMock(spec=ProtocolEventBusLike)
    result = ModelDispatchResult(
        dispatch_id=uuid4(),
        status="success",
        topic="onex.cmd.omniintelligence.context-compaction-requested.v1",
        started_at="2026-01-01T00:00:00Z",
        output_events=[
            ModelEventEnvelope(
                payload=Reply(), correlation_id=uuid4(), event_type=emitted_topic
            )
        ],
    )
    await applier(bus).apply(result)
    return str(bus.publish_envelope.call_args.kwargs["topic"])


@pytest.mark.asyncio
async def test_a_topic_of_the_declared_principal_template_is_published_to() -> None:
    assert await published_topic(ALICE) == ALICE


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "refused",
    [
        # the slot left unfilled
        TEMPLATE,
        # an empty principal
        "onex.evt.omnimarket.context-compaction-prepared-for-.v1",
        # characters outside the principal pattern: a dot starts another topic part,
        # an uppercase letter or slash is no principal the issuer issues
        "onex.evt.omnimarket.context-compaction-prepared-for-dev.alice.v1",
        "onex.evt.omnimarket.context-compaction-prepared-for-Dev-Alice.v1",
        "onex.evt.omnimarket.context-compaction-prepared-for-dev/alice.v1",
        # another topic the contract never declared
        "onex.evt.omnimarket.context-compaction-other-for-dev-alice-laptop.v1",
        # a longer tail than the template's own
        ALICE.replace(".v1", ".v1.extra"),
    ],
)
async def test_a_topic_outside_the_declared_template_falls_back_to_the_contract_topic(
    refused: str,
) -> None:
    assert await published_topic(refused) == TERMINAL
