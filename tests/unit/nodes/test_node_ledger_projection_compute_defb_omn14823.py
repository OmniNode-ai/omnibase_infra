# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""RED->GREEN canonical def-B flip proof for HandlerLedgerProjection (OMN-14823).

Companion to ``test_node_ledger_projection_compute.py``. Proves the OMN-14355
Class-B Tier-1 hand-flip of ``node_ledger_projection_compute`` to canonical
definition B:

* **SHAPE (RED->GREEN).** The auto-wiring dispatch entrypoint ``handle`` is now
  the canonical definition-B shape — a single ``BaseModel``-typed positional
  ``request: ModelEventMessage``, resolvable by the runtime's own def-B
  input-model resolver (``_resolve_def_b_input_model_type``) and NOT an
  envelope-shaped signature. On the pre-flip tree ``handle(self, message:
  object)`` the resolver returns ``None`` (``nonadaptable``) and the handler
  module referenced ``ModelEventEnvelope`` in its core (``envelope_in_core``).
  Both non-canonical defects are closed here.

* **EQUIVALENCE.** Driving the REAL auto-wiring dispatch callback
  (``_make_dispatch_callback``) over a deterministic corpus yields a
  ``ModelDispatchResult`` whose ``ledger.append`` intent lands in
  ``output_intents`` (NOT ``output_events``) — the routing the contract's
  ``intent_routing_table`` (``ledger.append -> node_ledger_write_effect``)
  consumes. A bare-``ModelIntent`` def-B return would be reclassified into
  ``output_events`` and silently skip the ledger write; these tests are the seam
  guard against that regression, and assert the dispatched intent is byte-equal
  to the pure ``project()`` output (behavior preserved by the flip).

Related: OMN-14355 (canonical-shape ratchet), OMN-14781 (hand-flip proof path),
OMN-14809 (verify_flip_bundle seam gate), OMN-1648/OMN-1726 (original node).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock
from uuid import UUID

import pytest

pytestmark = [pytest.mark.unit]

from omnibase_core.models.reducer.model_intent import ModelIntent
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.nodes.node_ledger_projection_compute import HandlerLedgerProjection
from omnibase_infra.nodes.node_ledger_projection_compute.handlers import (
    handler_ledger_projection as handler_module,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _handler_accepts_event_envelope,
    _make_dispatch_callback,
    _resolve_def_b_input_model_type,
)
from omnibase_infra.runtime.auto_wiring.models import ModelHandlerRef

# The contract-declared event_model the runtime materializes for this node's
# every routing entry (OMN-14594). The dispatch callback validates the wire
# payload into this model and hands the handler the typed instance.
_EVENT_MODEL_REF = ModelHandlerRef(
    name="ModelEventMessage",
    module="omnibase_infra.event_bus.models.model_event_message",
)

# Deterministic identity/time so the corpus (and therefore the flip's
# selected_input_hashes) is reproducible across runs and machines.
_CORR = UUID(int=0x0ED6E12E0ED6E12E0ED6E12E0ED6E12E)
_MSG_ID = UUID(int=0x1ED6E12E1ED6E12E1ED6E12E1ED6E12E)
_TS = datetime(2026, 1, 29, 12, 0, 0, tzinfo=UTC)


def _headers() -> ModelEventHeaders:
    return ModelEventHeaders(
        source="ledger-projection-defb-test",
        event_type="test.event.v1",
        timestamp=_TS,
        correlation_id=_CORR,
        message_id=_MSG_ID,
    )


def build_corpus() -> list[ModelEventMessage]:
    """Deterministic ModelEventMessage corpus exercising the projection branches.

    Covers: key present/absent (``_b64`` bytes vs ``None``), JSON vs binary
    value bytes, and the mixed topic categories (event / command / intent) the
    node subscribes to. Shared by the parity tests and the flip's adequacy
    receipt so the selected input set is identical (verify_flip_bundle binds the
    two field-by-field)."""
    hdr = _headers()
    return [
        ModelEventMessage(
            topic="onex.evt.platform.node-registration.v1",
            key=b"partition-key-1",
            value=b'{"node_id": "n-1", "kind": "effect"}',
            headers=hdr,
            partition=3,
            offset="42",
        ),
        ModelEventMessage(
            topic="onex.evt.platform.node-heartbeat.v1",
            key=None,
            value=b'{"beat": 7}',
            headers=hdr,
            partition=0,
            offset="0",
        ),
        ModelEventMessage(
            topic="onex.evt.platform.registration-snapshots.v1",
            key=b"snap-key",
            value=b"\x00\x01\x02\xff\xfe binary-snapshot-payload",
            headers=hdr,
            partition=1,
            offset="1024",
        ),
        ModelEventMessage(
            topic="onex.cmd.platform.request-introspection.v1",
            key=b"cmd-key",
            value=b'{"request": "introspect"}',
            headers=hdr,
            partition=2,
            offset="99",
        ),
        ModelEventMessage(
            topic="onex.intent.platform.runtime-tick.v1",
            key=None,
            value=b'{"tick": 1}',
            headers=hdr,
            partition=4,
            offset="7",
        ),
    ]


_CORPUS = build_corpus()
_CORPUS_IDS = [m.topic.split(".")[3] for m in _CORPUS]


@pytest.fixture
def handler() -> HandlerLedgerProjection:
    return HandlerLedgerProjection(MagicMock())


# =============================================================================
# SHAPE — the canonical definition-B flip (RED on the pre-flip tree)
# =============================================================================


class TestCanonicalDefBShape:
    """The dispatch entrypoint is canonical definition B (OMN-14355)."""

    def test_handle_single_positional_is_typed_event_message(self) -> None:
        """RED->GREEN: the runtime's def-B resolver recovers ModelEventMessage.

        Pre-flip ``handle(self, message: object)`` -> resolver returns ``None``
        (``nonadaptable``). Post-flip ``handle(self, request: ModelEventMessage)``
        -> the resolver recovers the concrete input model.
        """
        resolved = _resolve_def_b_input_model_type(HandlerLedgerProjection.handle)
        assert resolved is ModelEventMessage

    def test_handle_is_not_envelope_shaped(self) -> None:
        """A definition-B core takes the domain model, never a transport envelope."""
        assert _handler_accepts_event_envelope(HandlerLedgerProjection.handle) is False

    def test_handler_core_has_no_event_envelope_reference(self) -> None:
        """C-core: the resolved handler module must not reference the envelope type.

        The pre-flip module carried a ``ModelEventEnvelope`` docstring reference in
        ``_coerce_event_message`` (``envelope_in_core``); the flip removed that dead
        legacy glue.
        """
        source = Path(handler_module.__file__).read_text(encoding="utf-8")
        assert "ModelEventEnvelope" not in source


# =============================================================================
# EQUIVALENCE — the real auto-wiring dispatch path routes the intent correctly
# =============================================================================


class TestDispatchEquivalence:
    """Driving the REAL dispatch callback preserves the ledger.append routing."""

    @pytest.mark.parametrize("message", _CORPUS, ids=_CORPUS_IDS)
    def test_dispatch_routes_ledger_append_intent_to_output_intents(
        self, handler: HandlerLedgerProjection, message: ModelEventMessage
    ) -> None:
        callback = _make_dispatch_callback(handler, event_model=_EVENT_MODEL_REF)
        result = asyncio.run(callback(message))

        assert result is not None
        assert result.status is EnumDispatchStatus.SUCCESS
        # The ledger.append intent MUST land in output_intents (not output_events)
        # so the intent_routing_table reaches node_ledger_write_effect. A bare
        # ModelIntent return would be misrouted to output_events -> ledger stays
        # empty. This assertion is the seam guard.
        assert len(result.output_intents) == 1
        assert len(result.output_events) == 0
        assert result.output_count == 1

        intent = result.output_intents[0]
        assert isinstance(intent, ModelIntent)
        assert intent.payload.intent_type == "ledger.append"

    @pytest.mark.parametrize("message", _CORPUS, ids=_CORPUS_IDS)
    def test_defb_handle_output_equals_pure_project(
        self, handler: HandlerLedgerProjection, message: ModelEventMessage
    ) -> None:
        """The def-B handle wraps the SAME intent the pure project() computes."""
        output = asyncio.run(handler.handle(message))
        expected = handler.project(message)

        assert isinstance(output.result, ModelIntent)
        assert output.result.payload == expected.payload
        assert output.result.target == expected.target
        assert output.result.intent_type == expected.intent_type


__all__ = ["TestCanonicalDefBShape", "TestDispatchEquivalence", "build_corpus"]
