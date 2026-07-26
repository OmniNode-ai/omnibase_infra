# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Durable dispatch-seam regression test for node_ledger_write_effect [OMN-14391].

Ask (1) of OMN-14391: neither OMN-14134 (``HandlerLedgerAppend.handle()``) nor its
dict-envelope fix (OMN-14139/14140) had a durable, re-runnable regression test that
drives a REAL envelope through the ACTUAL dispatch seam:

    wire_from_manifest() -> _prepare_handler_wiring() -> _make_dispatch_callback()
        -> MessageDispatchEngine.dispatch() -> HandlerLedgerAppend.handle(envelope)

Every prior test (including OMN-14139/14140's own dict-envelope coverage) calls
``.handle()``/``.execute()`` directly with a hand-built ``MagicMock()`` or ``dict``
envelope -- never through the real dispatch engine. That is not hypothetical: the
OMN-14139/14140 fix exists *because* the OMN-14134 unit test's ``MagicMock()``
envelope faked attribute access and masked a bug that only manifested against the
real dispatch engine's dict-shaped envelope (``getattr(dict, "payload", default)``
silently no-ops on a dict). A ``MagicMock``-based unit test structurally cannot
catch this class of bug again.

RED-before / GREEN-now:
    Before OMN-14134, ``HandlerLedgerAppend`` had no ``handle()`` method.
    ``_make_dispatch_callback`` (handler_wiring.py) falls back to a
    ``_missing_handle`` sentinel whenever a handler exposes neither ``handle``
    nor ``handle_async`` -- that sentinel raises ``ModelOnexError`` on every
    invocation. Had this test existed before OMN-14134 landed, ``test_ledger_
    append_dispatches_through_real_engine_to_handle`` below would have failed
    with that ``ModelOnexError`` (dispatch would report a non-SUCCESS status
    with the sentinel's message, not the mocked ``ModelLedgerAppendResult``).
    It is GREEN now because OMN-14134 added ``handle()`` and OMN-14139/14140
    fixed its dict-envelope + routing-schema handling.

Scope: this is ask (1) ONLY -- the LOCAL/CI dispatch-seam test with no live
Kafka/Postgres/k8s dependency. ``HandlerLedgerAppend.append()`` (the Postgres I/O
boundary) is mocked so the test proves the WIRING + envelope-extraction seam
deterministically, in-process, without a live database. Ask (2) of OMN-14391 (a
live Kafka produce -> consume -> event_ledger row readback) remains explicitly
blocked on OMN-14136 (onex-dev/onex-prod omninode-runtime CrashLoopBackOff,
preventing a live consumer from attaching) and is NOT attempted here.

Two existing CI gates that touch dispatch do NOT close this gap (verified by
direct read, 2026-07-11 -- see OMN-14391 description for detail):

* ``dispatcher-route-coverage``: a static check that a route is *declared* in
  ``handler_routing`` -- passes regardless of whether ``.handle()`` exists or works.
* ``dispatch-parity-gate`` (``tests/integration/runtime/test_dispatch_selection_parity.py``
  + ``tests/fixtures/dispatch_parity/harness.py``): drives
  ``MessageDispatchEngine.dispatch()`` but substitutes an INERT stub dispatcher
  ("no errors, status=SUCCESS" per the harness's own comment) -- proves dispatcher
  *selection*, never handler *execution*.

This test drives the REAL handler (no stub) through the REAL dispatch engine,
wired from the REAL on-disk ``node_ledger_write_effect/contract.yaml`` via
``discover_contracts()`` -- the identical entry-point discovery path the kernel
uses at boot (mirrors ``test_auto_wiring_real_manifest.py``'s "real manifest"
discipline, scoped to one node).
"""

from __future__ import annotations

from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.nodes.node_ledger_write_effect.handlers.handler_ledger_append import (
    HandlerLedgerAppend,
)
from omnibase_infra.nodes.node_ledger_write_effect.models import (
    ModelLedgerAppendResult,
)
from omnibase_infra.nodes.node_registration_reducer.models import (
    ModelPayloadLedgerAppend,
)
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts
from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
from omnibase_infra.runtime.auto_wiring.models import ModelAutoWiringManifest
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

pytestmark = pytest.mark.integration

_LEDGER_APPEND_TOPIC = "onex.cmd.platform.ledger-append.v1"
_CONTRACT_NAME = "node_ledger_write_effect"


def _ledger_write_effect_manifest() -> ModelAutoWiringManifest:
    """Return a single-contract manifest for node_ledger_write_effect.

    Filters the REAL manifest produced by ``discover_contracts()`` (the same
    entry-point discovery the kernel uses at boot -- OMN-9119 "real manifest"
    discipline) down to just this node, so ``wire_from_manifest`` wires the
    actual on-disk ``contract.yaml`` and the actual handler modules, not a
    hand-built fixture contract.
    """
    manifest = discover_contracts()
    matches = [c for c in manifest.contracts if c.name == _CONTRACT_NAME]
    assert matches, (
        f"discover_contracts() did not find {_CONTRACT_NAME!r} -- entry point "
        "missing or contract.yaml broken; cannot exercise the dispatch seam."
    )
    return ModelAutoWiringManifest(contracts=(matches[0],), errors=())


@pytest.mark.asyncio
async def test_ledger_append_dispatches_through_real_engine_to_handle() -> None:
    """A real envelope on the real dispatch engine reaches HandlerLedgerAppend.handle().

    Drives: wire_from_manifest(real contract) -> engine.freeze() -> engine.dispatch()
    -> MessageDispatchEngine._materialize_envelope_with_bindings (ALWAYS produces a
    dict -- "ALWAYS materialize envelope to dict format for consistent dispatcher
    API", message_dispatch_engine.py) -> the registered dispatch callback ->
    HandlerLedgerAppend.handle(envelope=<dict>).

    Only ``HandlerLedgerAppend.append()`` (the Postgres I/O boundary) is mocked;
    handle()'s envelope/payload extraction, model_validate, and result-wrapping
    logic all run for real against the dict shape the live engine actually
    produces -- exactly the shape OMN-14139/14140 fixed extraction for.
    """
    manifest = _ledger_write_effect_manifest()
    engine = MessageDispatchEngine()
    container = MagicMock(name="container")

    report = await wire_from_manifest(
        manifest=manifest,
        dispatch_engine=engine,
        event_bus=None,
        container=container,
        subscribe_immediately=False,
    )

    assert report.total_failed == 0, (
        f"wire_from_manifest failed for {_CONTRACT_NAME}: "
        f"{[(r.contract_name, r.reason) for r in report.results if r.reason]}"
    )
    assert report.total_wired == 1, (
        f"Expected node_ledger_write_effect to wire cleanly; report={report.results}"
    )
    ledger_result = report.results[0]
    assert ledger_result.dispatchers_registered, (
        "node_ledger_write_effect registered zero dispatchers -- the "
        "ledger.append route never reached the dispatch engine."
    )

    engine.freeze()

    correlation_id = uuid4()
    mocked_result = ModelLedgerAppendResult(
        success=True,
        ledger_entry_id=uuid4(),
        duplicate=False,
        topic="onex.evt.platform.node-heartbeat.v1",
        partition=0,
        kafka_offset=42,
    )

    payload = ModelPayloadLedgerAppend(
        topic="onex.evt.platform.node-heartbeat.v1",
        partition=0,
        kafka_offset=42,
        event_value="dGVzdC1ldmVudA==",  # base64("test-event")
        correlation_id=correlation_id,
    )
    envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
        payload=payload,
        correlation_id=correlation_id,
        event_type=_LEDGER_APPEND_TOPIC,
    )

    with patch.object(
        HandlerLedgerAppend, "append", new_callable=AsyncMock
    ) as mock_append:
        mock_append.return_value = mocked_result
        result = await engine.dispatch(topic=_LEDGER_APPEND_TOPIC, envelope=envelope)

    # If the dispatcher were still bound to `_missing_handle` (the pre-OMN-14134
    # sentinel), this call would raise/report ModelOnexError, not SUCCESS.
    assert result.status == EnumDispatchStatus.SUCCESS, (
        f"Dispatch did not succeed -- status={result.status} "
        f"error={result.error_message!r}. If this fails with the sentinel's "
        '"does not expose a callable handle()" message, the dispatch callback '
        "is bound to _missing_handle, not HandlerLedgerAppend.handle()."
    )
    assert result.output_events == [mocked_result], (
        "Dispatch did not return the value produced by HandlerLedgerAppend.handle() "
        f"-> append(); output_events={result.output_events!r}"
    )

    # Prove handle()'s envelope-extraction logic actually parsed the REAL
    # dict-shaped materialized envelope into the correct typed payload --
    # the exact class of bug OMN-14139/14140 fixed (dict vs MagicMock shape).
    mock_append.assert_awaited_once()
    assert mock_append.await_args is not None
    (called_payload,) = mock_append.await_args.args
    assert isinstance(called_payload, ModelPayloadLedgerAppend)
    assert called_payload.topic == "onex.evt.platform.node-heartbeat.v1"
    assert called_payload.partition == 0
    assert called_payload.kafka_offset == 42
    assert called_payload.correlation_id == correlation_id


@pytest.mark.asyncio
async def test_ledger_append_missing_handle_sentinel_raises_model_onex_error() -> None:
    """Structural counterfactual: a handler with no handle()/handle_async() binds
    the pre-OMN-14134 `_missing_handle` sentinel, which raises ModelOnexError.

    This pins the RED-before behavior directly (rather than only asserting it in
    prose): if HandlerLedgerAppend ever lost its handle() adapter, the dispatch
    callback would revert to this sentinel and every ledger-append command would
    fail with exactly this error -- the failure mode
    ``test_ledger_append_dispatches_through_real_engine_to_handle`` above guards
    against end-to-end.
    """
    from omnibase_core.models.errors import ModelOnexError
    from omnibase_infra.runtime.auto_wiring.handler_wiring import (
        _make_dispatch_callback,
    )

    class _HandlerWithNoHandleMethod:
        """Stands in for a handler that predates OMN-14134's handle() adapter."""

    callback = _make_dispatch_callback(_HandlerWithNoHandleMethod())  # type: ignore[arg-type]

    with pytest.raises(ModelOnexError, match="does not expose a callable handle"):
        await callback(cast("ModelEventEnvelope[object]", {"payload": {}}))
