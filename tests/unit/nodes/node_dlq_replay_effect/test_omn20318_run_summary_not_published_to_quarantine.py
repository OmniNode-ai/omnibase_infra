# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""OMN-20318 -- a DLQ replay run summary is not an event and is never published.

Measured read-only on the dev lane broker 2026-10-01 (dlq-depth-monitor runs
36874480284 and 36878227805): ``onex.dlq.omnibase-infra.quarantine.v1`` took
182-194 arrivals per 30-minute window and held ~190k records. The tail of the
topic was not one quarantined message. Every record was a
``ModelDlqReplayRunResult`` -- ``total_processed: 0``, ``quarantined: 0``,
``event_type: omnibase-infra.quarantine`` -- the summary of a drain that found
nothing to do.

Cause: ``HandlerDlqReplay.handle`` returned the summary as ``result``;
``_normalize_handler_result`` turns a ``BaseModel`` result into ``output_events``;
and the contract's only ``publish_topics`` entry is the quarantine sink (the
quarantine producer's own destination), which ``_select_dispatch_result_output_topic``
takes as the fallback output topic. So every DLQ record -- each one also triggers
a whole drain -- wrote one summary onto the terminal quarantine sink.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    ModelDlqReplayEngineConfig,
)
from omnibase_infra.nodes.node_dlq_replay_effect.handlers.handler_dlq_replay import (
    HandlerDlqReplay,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_dlq_message import (
    ModelDlqMessage,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import _make_dispatch_callback

pytestmark = pytest.mark.unit

_TOPIC = "onex.dlq.omnibase-infra.events.v1"  # onex-topic-allow: a declared subscribe topic of this node


class _EmptyConsumer:
    def __init__(self, config: ModelDlqReplayEngineConfig) -> None:
        self.config = config

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def consume_messages(self) -> AsyncIterator[ModelDlqMessage]:
        return
        yield  # pragma: no cover - makes this an async generator

    async def commit(self) -> None:
        return None


class _NoopEffect:
    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


def _handler() -> HandlerDlqReplay:
    config = ModelDlqReplayEngineConfig(
        bootstrap_servers="localhost:9092", dlq_topic=_TOPIC, max_replay_count=5
    )
    return HandlerDlqReplay(
        consumers={_TOPIC: _EmptyConsumer(config)},
        producer=_NoopEffect(),
        quarantine_producer=_NoopEffect(),
        tracking=None,
    )


async def test_a_drain_with_nothing_to_do_hands_the_runtime_no_output_event() -> None:
    callback = _make_dispatch_callback(_handler())

    result = await callback(
        {
            "payload": {},
            "__bindings": {},
            "__debug_trace": {"correlation_id": None, "topic": _TOPIC},
        }
    )

    assert result is not None
    assert result.output_events == [], (
        "the run summary reached the dispatch result as an output event; the "
        "contract's only publish topic is the quarantine sink, so the runtime "
        "would write it there on every DLQ record"
    )
