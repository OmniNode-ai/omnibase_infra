# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cross-boundary seam test for OMN-15546 — ingress owns correlation identity.

The local ingress assigns the correlation id when it ACCEPTS a `/skill` request.
Every command it dispatches, and every field inside that command's payload, must
carry that id BYTE-IDENTICAL. Nothing downstream may mint one.

The measured defect (onex-dev, 2026-07-30): one request entered with outer
correlation ``a4740001-…`` and the returned typed payload carried
``f34eea98-…``. ``RuntimeHostProcess._dispatch_local_ingress_request`` normalized
the outer UUID and stamped it on ``ModelDispatchBusCommand.correlation_id``, but
validated the route payload WITHOUT passing that authority in — so a typed input
model declaring ``correlation_id: UUID = Field(default_factory=uuid4)`` minted a
SECOND id for the absent key. Envelope identity and domain identity then diverge
for the rest of the request: the handler, the response payload and every
projection row use the minted id, while the caller awaits a terminal on the
outer one.

This drives the REAL host seam — ``_dispatch_local_ingress_request`` through to
the ``ModelDispatchBusCommand`` the Pattern B broker actually receives — and
asserts, per the ticket's acceptance criteria:

    command.correlation_id == UUID(command.payload["correlation_id"]) == outer

A unit test on ``validate_runtime_local_ingress_payload`` does NOT satisfy this:
the defect is that the host never handed the validator its authority, so a test
that calls the validator with the correlation already supplied cannot observe
the bug at all.

Shares ``tests/fixtures/seams/dispatch_correlation/accepted_command.json`` with
OMN-15474 (byte-identical file) — one accepted command, one correlation
authority, asserted at two different seams.

Merge-train order: OMN-15474 lands BEFORE this.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.models.dispatch.model_dispatch_bus_command import (
    ModelDispatchBusCommand,
)
from omnibase_core.models.dispatch.model_dispatch_bus_terminal_result import (
    ModelDispatchBusTerminalResult,
)
from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess
from omnibase_infra.runtime.runtime_local_ingress import ModelRuntimeLocalIngressRoute
from tests.helpers.runtime_helpers import make_runtime_config

pytestmark = pytest.mark.integration

SEAM_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "seams"
    / "dispatch_correlation"
    / "accepted_command.json"
)

_COMMAND_NAME = "delegate_skill"
_MODULE = __name__


class ModelSeamDelegateRequest(BaseModel):
    """Stands in for market's ``ModelDelegateSkillRequest``.

    The one property that matters is reproduced exactly: ``correlation_id`` has
    a ``default_factory=uuid4``, so an ABSENT key silently mints a new identity
    rather than failing. That is the whole defect surface.
    """

    model_config = ConfigDict(extra="forbid")

    correlation_id: UUID = Field(default_factory=uuid4)
    task_type: str = "test"
    prompt: str = ""


class ModelSeamNoCorrelationRequest(BaseModel):
    """Control: a route model with NO correlation field must be left alone."""

    model_config = ConfigDict(extra="forbid")

    task_type: str = "test"
    prompt: str = ""


def _route(input_model_name: str | None) -> ModelRuntimeLocalIngressRoute:
    return ModelRuntimeLocalIngressRoute(
        node_name="node_delegate_skill_orchestrator",
        contract_name=_COMMAND_NAME,
        command_topic="onex.cmd.omnibase-infra.delegation-request.v1",
        event_type="omnibase-infra.delegation-request",
        terminal_event="onex.evt.omnibase-infra.delegation-completed.v1",
        contract_path="/tmp/node_delegate_skill_orchestrator/contract.yaml",  # noqa: S108
        package_name="omnibase-infra",
        input_model_module=_MODULE if input_model_name else None,
        input_model_name=input_model_name,
    )


class _CapturingBroker:
    """Records the exact ModelDispatchBusCommand the host dispatches."""

    def __init__(self, route: ModelRuntimeLocalIngressRoute) -> None:
        self._route = route
        self.commands: list[ModelDispatchBusCommand] = []

    async def dispatch_request(
        self, command: ModelDispatchBusCommand
    ) -> tuple[ModelRuntimeLocalIngressRoute, ModelDispatchBusTerminalResult]:
        self.commands.append(command)
        return self._route, ModelDispatchBusTerminalResult(
            status="completed",
            payload={"status": "complete"},
            correlation_id=command.correlation_id,
        )


def _host(route: ModelRuntimeLocalIngressRoute) -> tuple[RuntimeHostProcess, Any]:
    process = RuntimeHostProcess(
        config=make_runtime_config(local_ingress={"enabled": True}),
        dispatch_engine=AsyncMock(),
    )
    process._is_running = True
    process._local_ingress_routes = {_COMMAND_NAME: route}
    broker = _CapturingBroker(route)
    process._pattern_b_broker = broker
    return process, broker


async def _dispatch(
    process: RuntimeHostProcess, payload: dict[str, object], correlation_id: UUID
) -> Any:
    from omnibase_infra.runtime.runtime_local_ingress import (
        ModelLocalRuntimeIngressRequest,
    )

    return await process._dispatch_local_ingress_request(
        ModelLocalRuntimeIngressRequest(
            command_name=_COMMAND_NAME,
            payload=payload,
            correlation_id=correlation_id,
        )
    )


@pytest.mark.asyncio
async def test_seam_ingress_correlation_is_authoritative_in_dispatched_command() -> (
    None
):
    """Omitted payload correlation_id must be the INGRESS id, never a minted one.

    This is the RED case: with a ``default_factory=uuid4`` input model and no
    ``correlation_id`` in the payload, the pre-fix host validated the payload
    without its authority and the model minted a second id. Envelope identity
    (``command.correlation_id``) and domain identity
    (``command.payload["correlation_id"]``) then disagree.
    """
    seam = json.loads(SEAM_FIXTURE.read_text(encoding="utf-8"))
    outer = UUID(seam["ingress_correlation_id"])
    payload = {
        k: v
        for k, v in seam["envelope"]["payload"].items()
        if k in {"task_type", "prompt"}
    }
    assert "correlation_id" not in payload, (
        "seam precondition: the payload must OMIT correlation_id — that is the "
        "case where default_factory mints a second identity"
    )

    process, broker = _host(_route("ModelSeamDelegateRequest"))
    response = await _dispatch(process, payload, outer)

    assert response.ok is True, f"dispatch failed: {response.error}"
    assert len(broker.commands) == 1
    command = broker.commands[0]

    # Envelope identity is the ingress id.
    assert command.correlation_id == outer, (
        f"command envelope correlation {command.correlation_id} != ingress {outer}"
    )

    # Domain identity is the SAME id, byte-identical — not a minted sibling.
    assert isinstance(command.payload, dict)
    payload_correlation = command.payload.get("correlation_id")
    assert payload_correlation is not None, (
        "the typed payload carries no correlation_id at all — ingress authority "
        "was not injected"
    )
    assert UUID(str(payload_correlation)) == outer, (
        "OMN-15546 split identity: the dispatched command's envelope carries "
        f"{command.correlation_id}, but its typed payload carries "
        f"{payload_correlation}. The input model's default_factory=uuid4 minted "
        "a SECOND correlation id because the host validated the payload without "
        "passing the authoritative ingress correlation in."
    )
    assert str(payload_correlation) == str(outer), (
        "byte-identical means byte-identical: the payload representation "
        f"{payload_correlation!s} differs textually from the ingress {outer!s}"
    )


@pytest.mark.asyncio
async def test_seam_conflicting_payload_correlation_is_rejected_before_dispatch() -> (
    None
):
    """A payload correlation that CONFLICTS with the ingress must not dispatch.

    Silently overwriting the caller's value would be as wrong as minting one:
    the caller believes it supplied an identity that the runtime discarded. The
    ticket's fix contract requires a typed ``validation_error`` and the broker
    NOT called.
    """
    seam = json.loads(SEAM_FIXTURE.read_text(encoding="utf-8"))
    outer = UUID(seam["ingress_correlation_id"])
    conflicting = uuid4()
    assert conflicting != outer

    process, broker = _host(_route("ModelSeamDelegateRequest"))
    response = await _dispatch(
        process,
        {"task_type": "test", "prompt": "x", "correlation_id": str(conflicting)},
        outer,
    )

    assert response.ok is False, (
        "a payload correlation_id conflicting with the ingress authority was "
        "ACCEPTED; one request now has two claimed identities"
    )
    assert response.error is not None
    assert response.error.code == "validation_error", (
        f"expected typed validation_error, got {response.error.code}"
    )
    assert broker.commands == [], (
        "fail-closed means the broker is never reached: a conflicting-identity "
        "command was dispatched anyway"
    )


@pytest.mark.asyncio
async def test_seam_matching_payload_correlation_is_accepted() -> None:
    """Supplying the SAME id is legal and must normalize, not conflict."""
    seam = json.loads(SEAM_FIXTURE.read_text(encoding="utf-8"))
    outer = UUID(seam["ingress_correlation_id"])

    process, broker = _host(_route("ModelSeamDelegateRequest"))
    response = await _dispatch(
        process,
        {"task_type": "test", "prompt": "x", "correlation_id": str(outer)},
        outer,
    )

    assert response.ok is True, f"matching correlation rejected: {response.error}"
    command = broker.commands[0]
    assert command.correlation_id == outer
    assert isinstance(command.payload, dict)
    assert UUID(str(command.payload["correlation_id"])) == outer


@pytest.mark.asyncio
async def test_seam_model_without_correlation_field_is_untouched() -> None:
    """Control: never inject an unknown key into an ``extra=forbid`` model.

    An over-broad fix that stamped ``correlation_id`` onto every route model
    would hard-fail every ``extra=forbid`` input model that does not declare it.
    """
    seam = json.loads(SEAM_FIXTURE.read_text(encoding="utf-8"))
    outer = UUID(seam["ingress_correlation_id"])

    process, broker = _host(_route("ModelSeamNoCorrelationRequest"))
    response = await _dispatch(process, {"task_type": "test", "prompt": "x"}, outer)

    assert response.ok is True, (
        "a route model without a correlation_id field was broken by the "
        f"authority injection: {response.error}"
    )
    command = broker.commands[0]
    assert command.correlation_id == outer
    assert isinstance(command.payload, dict)
    assert "correlation_id" not in command.payload, (
        "correlation_id was injected into a model that does not declare it"
    )


@pytest.mark.asyncio
async def test_seam_route_without_input_model_preserves_raw_payload() -> None:
    """Control: an untyped route must keep today's pass-through behavior."""
    seam = json.loads(SEAM_FIXTURE.read_text(encoding="utf-8"))
    outer = UUID(seam["ingress_correlation_id"])

    process, broker = _host(_route(None))
    response = await _dispatch(process, {"dry_run": True}, outer)

    assert response.ok is True, f"untyped route broke: {response.error}"
    command = broker.commands[0]
    assert command.correlation_id == outer
    assert command.payload == {"dry_run": True}
