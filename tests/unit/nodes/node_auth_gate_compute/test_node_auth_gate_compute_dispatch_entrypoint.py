# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Canonical def-B dispatch-entrypoint proof for node_auth_gate_compute (OMN-14818).

RSD canonical-rewrite lane (parent OMN-14355; burns down 1 of the OMN-14510
entrypoint-less handlers). ``HandlerAuthGate`` was HAND-FLIPPED to canonical
definition B via a pure rename ``evaluate(request) -> handle(request)`` (the
10-step authorization cascade body is preserved byte-identically; the legacy
``execute(envelope: dict) -> ModelHandlerOutput`` wrapper was removed — it had
no production caller).

These tests drive the handler through the REAL auto-wiring dispatch entrypoint
(``_make_dispatch_callback``) — the exact bind the runtime uses — so they prove
the canonical ``handle(request) -> response`` shape is dispatch-REACHABLE, not
merely importable. Before the flip the class exposed no ``handle`` and this path
raised ``ModelOnexError`` (missing dispatch entrypoint); after the flip the same
path returns a ``ModelDispatchResult`` carrying the ``ModelAuthGateDecision``.
This is the RED->GREEN-on-broken-tree proof.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.enums.enum_auth_decision import EnumAuthDecision
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.nodes.node_auth_gate_compute.handlers import HandlerAuthGate
from omnibase_infra.nodes.node_auth_gate_compute.models import (
    ModelAuthGateDecision,
    ModelAuthGateRequest,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import _make_dispatch_callback

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 2, 10, 12, 0, 0, tzinfo=UTC)
_FAR_FUTURE = (_NOW + timedelta(hours=4)).isoformat()
_RUN_ID = str(uuid4())


def _wire_dispatch() -> object:
    """Bind the canonical handler through the real auto-wiring dispatch path.

    event_model=None mirrors an operation_match def-B handler: the engine hands
    the dispatcher the raw materialized wire dict, and the adapter coerces it
    into the handler's declared input model from the ``handle()`` signature.
    """
    return _make_dispatch_callback(HandlerAuthGate(MagicMock()))


def _wire_envelope(payload: dict[str, object]) -> dict[str, object]:
    """A raw materialized wire envelope; correlation_id triggers payload unwrap."""
    return {"payload": payload, "correlation_id": str(uuid4())}


# Deterministic corpus exercising distinct cascade branches (allow + deny paths).
_ALLOW_WHITELIST = {
    "tool_name": "Read",
    "target_path": "workspace/feature.plan.md",
    "now": _NOW.isoformat(),
}
_DENY_NO_RUN_ID = {
    "tool_name": "Bash",
    "target_path": "",
    "run_id": None,
    "now": _NOW.isoformat(),
}
_DENY_NO_AUTH = {
    "tool_name": "Edit",
    "target_path": "src/pkg/module.py",
    "run_id": _RUN_ID,
    "authorization": None,
    "now": _NOW.isoformat(),
}
_ALLOW_AUTHORIZED = {
    "tool_name": "Edit",
    "target_path": "src/pkg/module.py",
    "target_repo": "omnibase_infra",
    "run_id": _RUN_ID,
    "authorization": {
        "run_id": _RUN_ID,
        "allowed_tools": ["Edit", "Write"],
        "allowed_paths": ["src/**/*.py"],
        "repo_scopes": ["omnibase_infra"],
        "source": "explicit",
        "expires_at": _FAR_FUTURE,
        "reason": "test",
    },
    "now": _NOW.isoformat(),
}

_CASES = [
    ("E1_whitelist", _ALLOW_WHITELIST, EnumAuthDecision.ALLOW, 1),
    ("E2_no_run_id", _DENY_NO_RUN_ID, EnumAuthDecision.DENY, 3),
    ("E3_no_auth", _DENY_NO_AUTH, EnumAuthDecision.DENY, 4),
    ("E4_authorized", _ALLOW_AUTHORIZED, EnumAuthDecision.ALLOW, 10),
]


def test_handler_exposes_handle_entrypoint() -> None:
    """The canonical def-B dispatch entrypoint ``handle`` exists and is callable."""
    handler = HandlerAuthGate(MagicMock())
    assert callable(getattr(handler, "handle", None)), (
        "HandlerAuthGate must expose a callable handle() def-B dispatch entrypoint"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "expected_decision", "expected_step"),
    [(p, d, s) for _id, p, d, s in _CASES],
    ids=[c[0] for c in _CASES],
)
async def test_real_dispatch_callback_returns_success(
    payload: dict[str, object],
    expected_decision: EnumAuthDecision,
    expected_step: int,
) -> None:
    """Driving the REAL _make_dispatch_callback returns SUCCESS with the decision.

    Pre-flip (no ``handle``) this raised ``ModelOnexError`` (missing entrypoint);
    post-flip it returns a ``ModelDispatchResult`` carrying the decision — the
    canonical shape is dispatch-reachable.
    """
    callback = _wire_dispatch()
    result = await callback(_wire_envelope(payload))  # type: ignore[operator]

    assert isinstance(result, ModelDispatchResult)
    assert result.status is EnumDispatchStatus.SUCCESS
    assert result.output_count == 1
    (decision,) = result.output_events
    assert isinstance(decision, ModelAuthGateDecision)
    assert decision.decision is expected_decision
    assert decision.step == expected_step


def test_handle_owns_behavior_direct_call() -> None:
    """handle() itself owns the cascade behavior — no retained op-method shim."""
    handler = HandlerAuthGate(MagicMock())
    decision = handler.handle(ModelAuthGateRequest.model_validate(_ALLOW_AUTHORIZED))
    assert isinstance(decision, ModelAuthGateDecision)
    assert decision.decision is EnumAuthDecision.ALLOW
    assert decision.step == 10
