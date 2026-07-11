# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cross-boundary seam tests for tenant_scoped_ingress stamping (OMN-14349).

OMN-14208 Path A: when a contract opts in (``event_bus.tenant_scoped_ingress:
true``), the auto-wiring engine derives a verified tenant_id from the
``tenant-<slug>.`` wire prefix a message structurally arrived on and stamps it
into the payload BEFORE dispatch -- overwriting any client-supplied value,
never falling back to one. These tests drive the real
``_make_event_bus_callback`` construction path (the same function every
ordinary node's topics already go through) and decode the payload the real
dispatch engine receives -- not a mock of the stamping logic itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.runtime.auto_wiring.handler_wiring import _make_event_bus_callback

pytestmark = pytest.mark.asyncio


@dataclass
class _FakeDispatchEngine:
    """Captures every envelope dispatched, keyed by topic."""

    calls: list[tuple[str, ModelEventEnvelope[object]]] = field(default_factory=list)
    is_frozen: bool = True

    async def dispatch(self, topic: str, envelope: ModelEventEnvelope[object]) -> None:
        self.calls.append((topic, envelope))


@dataclass(frozen=True)
class _Message:
    value: bytes


def _raw_message(
    payload: dict[str, object], correlation_id: object | None = None
) -> _Message:
    body: dict[str, object] = {
        "payload": payload,
        "correlation_id": str(correlation_id or uuid4()),
        "envelope_timestamp": datetime.now(UTC).isoformat(),
        "event_type": "omnimarket.delegate-skill",
        "source_tool": "test-adapter",
    }
    return _Message(value=json.dumps(body).encode("utf-8"))


async def test_tenant_scoped_stamps_verified_slug_overwriting_forged_payload() -> None:
    engine = _FakeDispatchEngine()
    callback = _make_event_bus_callback(
        "tenant-acme.onex.cmd.omnimarket.delegate-skill.v1",
        engine,
        tenant_scoped=True,
    )

    await callback(_raw_message({"prompt": "hi", "tenant_id": "evil-forged-tenant"}))

    assert len(engine.calls) == 1
    topic, envelope = engine.calls[0]
    assert topic == "tenant-acme.onex.cmd.omnimarket.delegate-skill.v1"
    assert envelope.payload["tenant_id"] == "acme"
    assert envelope.payload["tenant_id"] != "evil-forged-tenant"
    assert envelope.payload["prompt"] == "hi"


async def test_tenant_scoped_stamps_slug_when_payload_omits_tenant_id() -> None:
    engine = _FakeDispatchEngine()
    callback = _make_event_bus_callback(
        "tenant-beta.onex.cmd.omnimarket.delegate-skill.v1",
        engine,
        tenant_scoped=True,
    )

    await callback(_raw_message({"prompt": "hi"}))

    _, envelope = engine.calls[0]
    assert envelope.payload["tenant_id"] == "beta"


async def test_tenant_scoped_never_stamps_a_default_on_non_prefixed_topic() -> None:
    """Fail-closed: no tenant-<slug>. prefix -> no stamp, ever (Stage-1 warn).

    Never a defaulted/guessed tenant_id -- the OMN-14058 masking failure this
    whole design exists to prevent.
    """
    engine = _FakeDispatchEngine()
    callback = _make_event_bus_callback(
        "onex.cmd.omnimarket.delegate-skill.v1",  # bare, no tenant prefix
        engine,
        tenant_scoped=True,
    )

    await callback(_raw_message({"prompt": "hi", "tenant_id": "self-reported"}))

    _, envelope = engine.calls[0]
    # Unstamped: the self-reported value survives untouched (Stage-1 warn path
    # -- the existing OMN-14058 flow), never overwritten with a guess.
    assert envelope.payload["tenant_id"] == "self-reported"


async def test_tenant_scoped_off_by_default_is_a_true_no_op() -> None:
    """A contract that does NOT opt in sees zero behavior change."""
    engine = _FakeDispatchEngine()
    callback = _make_event_bus_callback(
        "tenant-acme.onex.cmd.omnimarket.delegate-skill.v1",
        engine,
        # tenant_scoped omitted -- defaults False
    )

    await callback(_raw_message({"prompt": "hi"}))

    _, envelope = engine.calls[0]
    assert "tenant_id" not in envelope.payload


async def test_tenant_scoped_rejects_malformed_slug_without_stamping() -> None:
    """A prefix that doesn't match the DNS-slug convention is treated as absent."""
    engine = _FakeDispatchEngine()
    callback = _make_event_bus_callback(
        "tenant-NOT_VALID!.onex.cmd.omnimarket.delegate-skill.v1",
        engine,
        tenant_scoped=True,
    )

    await callback(_raw_message({"prompt": "hi"}))

    _, envelope = engine.calls[0]
    assert "tenant_id" not in envelope.payload
