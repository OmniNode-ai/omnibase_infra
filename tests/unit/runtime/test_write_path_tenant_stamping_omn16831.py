# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Write-path stamping of the tenant dimension (OMN-16831, ruled item 2).

The 2026-08-28 operator ruling on this ticket adopted option (D) and named five
mechanism-agnostic items. Item 2 is *"the runtime populates it -- write-path
stamping on the families that have a tenant in scope, and families with no
tenant concept record explicitly-none rather than nothing"*. Items 1, 3 and 4
shipped in 2026-08/09; item 2 was held back deliberately until the fleet
re-pinned onto the omnibase_core release carrying item 1, and then never
resumed as a whole. ``omnibase_infra#3573`` closed two of its sites (the
``tenant_scoped_ingress`` subscribe stamp and the ``DispatchResultApplier``
carriage); this closes the rest.

WHY EACH SITE BELOW IS A REAL HOLE, NOT A TIDINESS EXERCISE
-----------------------------------------------------------
``ModelEventEnvelope.tenant_id`` is the only field in the fleet that records
which tenant an event belongs to, and omnimarket's ``envelope_tenant_identity``
is its only reader. A tenant-classified projection write under ``FORCE ROW
LEVEL SECURITY`` cannot discover a row's tenant by reading, so an envelope that
reaches the writer without the dimension is refused, fail-closed, and the event
dead-letters. Every site here builds an envelope FROM a record that recorded a
tenant, and dropped it:

* ``_materialize_raw_event_envelope`` / ``_materialize_typed_event_envelope``
  re-hydrate a typed envelope when the runtime consumed a dict rather than a
  ``ModelEventEnvelope`` instance -- the shape every Kafka delivery takes. They
  carry ``correlation_id``, ``envelope_timestamp`` and ``event_type`` across and
  dropped the tenant, so the handler downstream of them saw an unattributed
  envelope even when the wire record was attributed.
* ``_emit_projection_terminal_event`` publishes a projection handler's own
  DECLARED output and, by construction (OMN-17214 Defect B), does NOT go through
  the result applier -- so the OMN-16831 carriage fix in ``#3573`` does not
  reach it. A tenant's projection terminal was published unattributed.

NEGATIVE CONTROLS ARE THE HALF THAT MATTERS
-------------------------------------------
OMN-16831 AC2 and OMN-16804 AC3 both forbid inventing or defaulting a tenant
inside the runtime. A propagation that also SOURCED one when none was consumed
would pass a one-sided test while destroying the property the projection
writer's refusal exists to protect, so each carriage assertion here is paired
with a control proving nothing is invented.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from pydantic import BaseModel

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _emit_projection_terminal_event,
    _materialize_raw_event_envelope,
    _materialize_typed_event_envelope,
)

_TENANT = "acme"


class _Payload(BaseModel):
    """Minimal typed payload for the typed-materialization path."""

    value: str


def _wire_envelope(*, tenant_id: str | None) -> dict[str, object]:
    """The dict shape a Kafka delivery hands the runtime.

    Deliberately a plain mapping, not a ``ModelEventEnvelope``: the
    materialization helpers short-circuit on an instance and only the dict
    branch rebuilds, which is the branch that dropped the dimension.
    """
    body: dict[str, object] = {
        "payload": {"value": "v"},
        "correlation_id": str(uuid4()),
        "envelope_timestamp": datetime.now(UTC).isoformat(),
        "event_type": "omnimarket.delegate-skill-completed",
    }
    if tenant_id is not None:
        body["tenant_id"] = tenant_id
    return body


class _RecordingBus:
    """Captures the raw bytes of every publish, like the real bus sees them."""

    def __init__(self) -> None:
        self.published: list[tuple[str, bytes]] = []

    async def publish(self, topic: str, key: object, value: bytes) -> None:
        self.published.append((topic, value))


@pytest.mark.unit
def test_rematerialized_raw_envelope_carries_the_consumed_tenant() -> None:
    """A rebuilt envelope records the tenant the wire record recorded."""
    consumed = _wire_envelope(tenant_id=_TENANT)

    rebuilt = _materialize_raw_event_envelope(
        consumed, consumed["payload"], "fallback.event"
    )

    assert rebuilt.tenant_id == _TENANT


@pytest.mark.unit
def test_rematerialized_raw_envelope_invents_no_tenant() -> None:
    """Negative control: nothing recorded means nothing recorded."""
    consumed = _wire_envelope(tenant_id=None)

    rebuilt = _materialize_raw_event_envelope(
        consumed, consumed["payload"], "fallback.event"
    )

    assert rebuilt.tenant_id is None


@pytest.mark.unit
def test_rematerialized_typed_envelope_carries_the_consumed_tenant() -> None:
    """The typed branch carries it too -- both branches feed the same handler."""
    consumed = _wire_envelope(tenant_id=_TENANT)

    rebuilt = _materialize_typed_event_envelope(
        consumed, _Payload(value="v"), "fallback.event"
    )

    assert rebuilt.tenant_id == _TENANT


@pytest.mark.unit
def test_rematerialized_typed_envelope_invents_no_tenant() -> None:
    """Negative control on the typed branch."""
    consumed = _wire_envelope(tenant_id=None)

    rebuilt = _materialize_typed_event_envelope(
        consumed, _Payload(value="v"), "fallback.event"
    )

    assert rebuilt.tenant_id is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_projection_terminal_records_the_source_tenant() -> None:
    """The projection terminal is a declared output and must be attributable.

    This publish bypasses ``DispatchResultApplier`` by design (OMN-17214
    Defect B), so the ``#3573`` carriage fix never reached it.
    """
    bus = _RecordingBus()
    source = _wire_envelope(tenant_id=_TENANT)

    await _emit_projection_terminal_event(
        bus,
        "onex.evt.omnimarket.projection-delegation-applied.v1",
        source,
        {"rows_upserted": 1},
    )

    assert len(bus.published) == 1
    published: Any = ModelEventEnvelope[object].model_validate_json(bus.published[0][1])
    assert published.tenant_id == _TENANT


@pytest.mark.unit
@pytest.mark.asyncio
async def test_projection_terminal_invents_no_tenant() -> None:
    """Negative control: an unattributed source stays unattributed."""
    bus = _RecordingBus()
    source = _wire_envelope(tenant_id=None)

    await _emit_projection_terminal_event(
        bus,
        "onex.evt.omnimarket.projection-delegation-applied.v1",
        source,
        {"rows_upserted": 1},
    )

    assert len(bus.published) == 1
    published: Any = ModelEventEnvelope[object].model_validate_json(bus.published[0][1])
    assert published.tenant_id is None
