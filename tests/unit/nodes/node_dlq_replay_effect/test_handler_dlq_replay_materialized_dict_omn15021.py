# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""OMN-15021 -- HandlerDlqReplay.handle() crashes on the runtime's materialized
dispatch-dict shape, not a hydrated ModelEventEnvelope.

This is the RED-against-EXISTS-but-WRONG proof for the def-B / envelope-shaped
dispatch-boundary fix.

Live incident (2026-07-24, .201 stability-test, OMN-14551 G6 DLQ-boundary
flip): a message landing on ``onex.dlq.omnibase-infra.events.v1`` crashed
``node_dlq_replay_effect`` with::

    AttributeError: 'dict' object has no attribute 'correlation_id'

Root cause (traced via ``handler_wiring._make_dispatch_callback`` /
``_handler_accepts_event_envelope``): because ``HandlerDlqReplay.handle``'s
first parameter is literally named ``envelope``, the auto-wiring boundary
classifies it as envelope-accepting. For an ``operation_match`` handler (no
contract-declared ``event_model``) that classification skips the def-B
coercion path entirely and hands the handler the RAW materialized dispatch
dict unchanged -- ``ModelMaterializedDispatch``'s
``{"payload": ..., "__bindings": ..., "__debug_trace": ...}`` shape -- never a
hydrated ``ModelEventEnvelope`` instance. The pre-fix
``envelope.correlation_id`` attribute access on the FIRST line of ``handle()``
therefore crashed on EVERY dispatch delivered through this path, not only
"malformed" DLQ content; the topic simply carried zero prior traffic (HW
stayed 0) before the OMN-14551 G6 flip, so the bug was latent, not new.

This module drives the REAL production dispatch callback
(``_make_dispatch_callback``) over the REAL ``HandlerDlqReplay`` class (no
fake handler, no patched entrypoint), feeding it the exact materialized-dict
shape the engine produces in production (per
``MessageDispatchEngine._materialize_envelope_with_bindings`` /
``ModelMaterializedDispatch`` -- read from source, not independently re-read
from the live .201 topic in this session), so these tests FAIL against the
pre-fix handler and PASS only once the defensive extraction exists.

Self-feed coverage: the observed 2026-07-24 self-feed (DLQ topic HW 0 -> 14 in
seconds) was driven entirely by ``handle()`` raising on every dispatch of a
message landing on its own subscribe topic -- each raise fed
``_route_swallowed_exception`` (OMN-14507), which best-effort republishes the
raw message back onto ``get_dlq_topic_for_original(topic)``. For the DLQ's own
subscribe topic that resolves to the SAME topic (the helper is not
DLQ-topic-aware), so a crashing handler on this topic is a guaranteed feedback
loop by construction -- fixing the crash removes the loop's only trigger.
``test_self_feed_loop_cannot_start_repeated_malformed_dispatch`` proves
``handle()`` never raises across many consecutive deliveries of the same
malformed shape.

Ticket: OMN-15021
Evidence-Ticket: OMN-15021
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import UUID, uuid4

import pytest

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
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


# The exact correlation_id cited in the live 2026-07-24 .201 stability-test
# incident evidence (docker logs omninode-stability-test-runtime, OMN-15021).
_LIVE_INCIDENT_CORRELATION_ID = "481dc9f7-d06b-435b-bc50-102966f3315e"


def _config(**overrides: object) -> ModelDlqReplayEngineConfig:
    base: dict[str, object] = {
        "bootstrap_servers": "localhost:9092",
        "dlq_topic": "onex.dlq.omnibase-infra.events.v1",
        "max_replay_count": 5,
    }
    base.update(overrides)
    return ModelDlqReplayEngineConfig(**base)


class _EmptyConsumer:
    """Yields nothing -- isolates these tests to handle()'s dispatch-boundary
    id extraction. The DLQ-drain content path (replay/quarantine/tracking) is
    already covered by test_handler_dlq_replay.py and is untouched by this fix.
    """

    def __init__(self, config: ModelDlqReplayEngineConfig) -> None:
        self.config = config
        self._started = False
        self.commits = 0

    async def start(self) -> None:
        self._started = True

    async def stop(self) -> None:
        self._started = False

    async def consume_messages(self) -> AsyncIterator[ModelDlqMessage]:
        return
        yield  # pragma: no cover - makes this an async generator

    async def commit(self) -> None:
        self.commits += 1


class _NoopEffect:
    def __init__(self) -> None:
        self._started = False

    async def start(self) -> None:
        self._started = True

    async def stop(self) -> None:
        self._started = False


def _handler() -> HandlerDlqReplay:
    config = _config()
    return HandlerDlqReplay(
        consumer=_EmptyConsumer(config),
        producer=_NoopEffect(),
        quarantine_producer=_NoopEffect(),
        tracking=None,
    )


def _materialized_dispatch_dict(
    *, correlation_id: str | None = _LIVE_INCIDENT_CORRELATION_ID
) -> dict[str, object]:
    """The shape ``MessageDispatchEngine._execute_dispatcher`` hands to every
    dispatcher (``_materialize_envelope_with_bindings`` /
    ``ModelMaterializedDispatch``) -- never a ``ModelEventEnvelope`` instance.
    """
    return {
        "payload": {
            "original_topic": "onex.evt.omnibase-infra.runtime-booted.v1",
            "failure_reason": "unroutable operation",
            "correlation_id": correlation_id,
        },
        "__bindings": {},
        "__debug_trace": {
            "event_type": None,
            "correlation_id": correlation_id,
            "trace_id": None,
            "causation_id": None,
            "topic": "onex.dlq.omnibase-infra.events.v1",
            "timestamp": "2026-07-24T17:29:03Z",
            "partition_key": None,
        },
    }


async def test_real_dispatch_callback_does_not_crash_on_materialized_dict() -> None:
    """RED-against-EXISTS-but-WRONG: drives the REAL auto-wiring callback with
    the REAL materialized-dispatch-dict shape.

    Pre-fix, this raises ``AttributeError: 'dict' object has no attribute
    'correlation_id'`` on the first line of ``handle()`` -- the exact live
    2026-07-24 .201 crash. Post-fix it must return a SUCCESS
    ``ModelDispatchResult`` and never raise.
    """
    callback = _make_dispatch_callback(_handler())

    result = await callback(_materialized_dispatch_dict())

    assert result is not None, "handle() produced no result."
    assert isinstance(result, ModelDispatchResult)
    assert result.status is EnumDispatchStatus.SUCCESS, (
        f"handle() must never crash on the materialized dispatch dict; got "
        f"status={result.status!r} error={result.error_message!r}"
    )


async def test_handle_extracts_the_live_incident_correlation_id_from_dict() -> None:
    """The generated ModelHandlerOutput carries the REAL correlation_id read
    out of the materialized dict's __debug_trace -- proving the fix is a real
    extraction, not just crash-suppression via a throwaway generated id.
    """
    handler = _handler()

    output = await handler.handle(_materialized_dispatch_dict())

    assert output.correlation_id == UUID(_LIVE_INCIDENT_CORRELATION_ID)


async def test_handle_never_raises_on_a_correlation_id_free_dict() -> None:
    """A genuinely undecodable envelope shape (no correlation_id anywhere)
    still never raises -- fail-loud (a WARNING is logged), never a silent
    swallow. The message is not dropped: it still drives a full handle()/run()
    cycle; only the dispatch-boundary id is synthesized.
    """
    handler = _handler()
    bare_dict: dict[str, object] = {"payload": {}, "__bindings": {}}

    output = await handler.handle(bare_dict)

    assert output.correlation_id is not None
    assert output.input_envelope_id is not None


async def test_handle_still_works_with_a_real_envelope_instance() -> None:
    """Backward compatibility: a real ``ModelEventEnvelope`` (e.g. a direct
    test/standalone caller) still round-trips its own correlation_id /
    envelope_id exactly as before OMN-15021.
    """
    handler = _handler()
    correlation_id = uuid4()
    envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
        payload={"noop": True},
        correlation_id=correlation_id,
    )

    output = await handler.handle(envelope)

    assert output.correlation_id == correlation_id
    assert output.input_envelope_id == envelope.envelope_id


async def test_self_feed_loop_cannot_start_repeated_malformed_dispatch() -> None:
    """OMN-15021 self-feed coverage: repeated delivery of the same malformed
    shape must never raise. Bound (50) is well above the observed live burst
    (HW 0 -> 14) so the "never raises" claim is not a fluke of a short run.
    """
    handler = _handler()
    malformed = _materialized_dispatch_dict(correlation_id=None)

    for _ in range(50):
        output = await handler.handle(malformed)
        assert output.correlation_id is not None
