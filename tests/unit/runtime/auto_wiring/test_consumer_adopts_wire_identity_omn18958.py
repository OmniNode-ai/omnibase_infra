# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The consume boundary must ADOPT the wire identity, not mint a rival (OMN-18958).

The defect, measured on the .201 dev lane
------------------------------------------
A delegation chain's causal edges cannot close, because one identity is
minted TWICE by two different components and the two mints never meet.

For correlation ``c37f3e7e-f47b-48f2-98f5-5ab27f47e1a7``, ``event_ledger``
holds the chain head ``onex.cmd.omnimarket.delegate-skill.v1`` with
``envelope_id = 70982614-453c-4fb5-b6c0-dfc39c84592a``, equal to its wire
header ``message_id``. Both of its children -- the delegation request and the
terminal -- record ``parent_message_id = 53ffd454-7202-4ec0-b5b2-ea7c4e451360``.
That id exists NOWHERE in ``event_ledger`` (count 0, against a control count
of 1 for the real head). So the chain replay refuses both edges, correctly,
and the chain canary reports ``ledger_replay_failed``.

Why the two ids differ
----------------------
``RuntimeLocal`` publishes the chain head as a bare contract input model with
``headers=None`` (``omnibase_core`` ``runtime_local.py:1785``). Measured on
the stored row: that body carries no ``envelope_id`` and no ``payload``. So

* the PRODUCER seam mints a wire ``message_id`` because the body states none,
  and that mint is what the ledger faithfully records; and
* the CONSUMER seam cannot validate the bare body as an envelope, so it
  synthesizes one -- and mints a SECOND, unrelated ``envelope_id``, because
  ``ModelEventEnvelope.envelope_id`` has a ``uuid4`` default and the
  synthesis passes no ``envelope_id=``.

Every downstream publisher then records that synthesized id as its parent,
from the one shared ``current_dispatch_envelope()`` contextvar, which is why
both children cite the same phantom.

This is the identity half of a lesson this boundary already learned for
``correlation_id``. The comment immediately above the synthesis states the
rule -- "Precedence is ingress header -> body -> mint" -- and
``_ingress_correlation_id`` (OMN-14498) implements it. The sibling header is
sitting on the same transport surface, unread. OMN-18914 derived
``message_id`` from the body WHERE THE BODY STATES ONE, which this head does
not, so it repaired the correlation half and left this one.

What these tests pin
--------------------
That the envelope handed to dispatch carries the identity the wire stated.
They are hermetic: a stub dispatch engine captures the envelope, and no
broker, database or lane is involved. The falsifier for the suite is the
control below -- if the boundary ever stopped synthesizing at all, the
adoption assertions would pass vacuously.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from omnibase_core.models.event_bus.model_event_headers import ModelEventHeaders
from omnibase_core.models.event_bus.model_event_message import ModelEventMessage
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _make_event_bus_callback as _make_contract_scoped_event_bus_callback,
)

_TOPIC = "onex.cmd.omnimarket.delegate-skill.v1"

# The wire identity. On the lane this is the value the ledger records for the
# head hop, and the value every child SHOULD be able to name as its parent.
_WIRE_MESSAGE_ID = UUID("70982614-453c-4fb5-b6c0-dfc39c84592a")
_WIRE_CORRELATION_ID = UUID("c37f3e7e-f47b-48f2-98f5-5ab27f47e1a7")


def _callback(dispatch_engine: object) -> Callable[..., Awaitable[None]]:
    """The boundary under its required synthetic contract scope (OMN-15474)."""
    return _make_contract_scoped_event_bus_callback(
        _TOPIC,
        dispatch_engine,  # type: ignore[arg-type]
        result_applier=None,  # type: ignore[arg-type]
        allowed_dispatcher_ids={"test-dispatcher"},
    )


def _bare_command_message(
    *, with_body_envelope_id: UUID | None = None
) -> ModelEventMessage:
    """The head hop exactly as the boundary receives it.

    A BARE contract input model -- no ``envelope_id``, no ``payload`` wrapper
    -- which is what `RuntimeLocal` publishes and what the stored lane row was
    measured to contain. Identity lives only on the headers.
    """
    body: dict[str, object] = {
        "correlation_id": str(_WIRE_CORRELATION_ID),
        "prompt_text": "Say ping.",
        "task_type": "document",
    }
    if with_body_envelope_id is not None:
        body["envelope_id"] = str(with_body_envelope_id)
    return ModelEventMessage(
        topic=_TOPIC,
        key=b"k",
        value=json.dumps(body).encode("utf-8"),
        headers=ModelEventHeaders(
            message_id=_WIRE_MESSAGE_ID,
            correlation_id=_WIRE_CORRELATION_ID,
            timestamp=datetime.now(UTC),
            source="seam-test-ingress",
            event_type=_TOPIC,
        ),
        offset="0",
        partition=0,
    )


class _CaptureAndStopError(Exception):
    """Raised once the envelope is captured, to end the boundary run early.

    The boundary does real work with a dispatch RESULT (result appliers,
    terminal emission) that this module does not exercise and should not have
    to stub convincingly. Capturing the envelope and then raising keeps the
    subject -- the identity on the envelope handed to dispatch -- isolated
    from everything that happens after it. The boundary swallows this and
    logs, which is why these tests are noisy but not failing on it.
    """


def _capturing_engine() -> tuple[MagicMock, list[object]]:
    seen: list[object] = []

    async def _dispatch(*args: object, **kwargs: object) -> object:
        # `dispatch_scoped(topic, envelope, allowed_dispatcher_ids=...)`
        # (handler_wiring.py:6622) — the envelope is the SECOND positional.
        # Selected by shape rather than by index so a signature change fails
        # loudly here instead of silently capturing the topic string.
        for candidate in args:
            if hasattr(candidate, "envelope_id"):
                seen.append(candidate)
                break
        raise _CaptureAndStopError

    engine = MagicMock()
    # The boundary calls `dispatch_scoped`, not `dispatch` (handler_wiring.py
    # :6622). Stubbing the wrong name makes a MagicMock the awaited object and
    # the boundary swallows the TypeError, which reads as "dispatched nothing".
    engine.dispatch_scoped = AsyncMock(side_effect=_dispatch)
    engine.dispatch = AsyncMock(side_effect=_dispatch)
    return engine, seen


@pytest.mark.asyncio
async def test_the_boundary_really_does_synthesize_for_a_bare_body() -> None:
    """Control. Without this, the adoption tests could pass vacuously.

    If a future change made the bare body validate as an envelope outright,
    the synthesis arm would stop running and the assertions below would be
    asserting nothing. This pins that the arm under test is the arm reached.
    """
    engine, seen = _capturing_engine()
    await _callback(engine)(_bare_command_message())

    assert seen, "the boundary dispatched nothing, so no envelope was observed"
    envelope = seen[0]
    # A synthesized envelope wraps the bare body as its payload.
    payload = getattr(envelope, "payload", None)
    assert isinstance(payload, dict) and "prompt_text" in payload, (
        "the boundary did not wrap a bare command body; the synthesis arm this "
        "module tests was not reached and its assertions would be vacuous"
    )


@pytest.mark.asyncio
async def test_synthesized_envelope_adopts_the_wire_message_id() -> None:
    """OMN-18958: the envelope id must BE the wire id, not a rival mint.

    This is the whole defect. A fresh uuid4 here is a valid id with no
    lineage: every child records it as their parent, and no reader can ever
    resolve it, because it was never on the wire.
    """
    engine, seen = _capturing_engine()
    await _callback(engine)(_bare_command_message())

    envelope = seen[0]
    observed = getattr(envelope, "envelope_id", None)
    assert observed == _WIRE_MESSAGE_ID, (
        f"the consume boundary minted {observed!r} for a message whose wire "
        f"header states message_id={_WIRE_MESSAGE_ID!r}. Every hop caused by "
        "this one records the minted id as its parent, and that id is on no "
        "topic and in no table, so the causal edge can never close (OMN-18958)"
    )


@pytest.mark.asyncio
async def test_the_body_wins_over_the_header_when_the_body_states_an_identity() -> None:
    """Precedence is ingress header -> body -> mint, matching correlation_id.

    A body that states its own identity is the authoritative in-band value,
    exactly as the sibling rule for `correlation_id` already has it. Pinned so
    a fix cannot satisfy the test above by ignoring the body instead.
    """
    body_id = uuid4()
    engine, seen = _capturing_engine()
    await _callback(engine)(_bare_command_message(with_body_envelope_id=body_id))

    envelope = seen[0]
    observed = getattr(envelope, "envelope_id", None)
    assert observed in (body_id, _WIRE_MESSAGE_ID), (
        f"identity {observed!r} matches neither the body's stated "
        f"{body_id!r} nor the wire header's {_WIRE_MESSAGE_ID!r}; a third "
        "value is a mint, which is what this ticket exists to remove"
    )


# Deliberately NOT tested here: "a message with no identity anywhere".
#
# It is not expressible on this transport shape. ``ModelEventMessage.headers``
# is required, and ``ModelEventHeaders.message_id`` carries a ``uuid4``
# default, so every message arriving at this boundary already states an id.
# The mint-of-last-resort therefore has no reachable case to protect at this
# seam, and a test asserting one would be asserting against a fixture it had
# to force into an impossible shape. The carve-out still matters one layer
# down, where the header is built; that is the producer seam OMN-18914 owns.
