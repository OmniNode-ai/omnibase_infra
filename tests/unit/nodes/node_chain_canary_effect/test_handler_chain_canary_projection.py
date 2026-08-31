# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-16963 — link 2 must read the PROJECTION, not infer it from the bus.

The defect these tests pin
--------------------------
OMN-16025 link 2 is *"Routing decision PUBLISHED and PROJECTED (readback from
projection, not logs)."* The canary had no instrument for it: it asserted a
terminal arrived on the bus and that quarantine was clean, and never read
``omnibase_infra.delegation_workflow_state`` at all. Link 2 reported
``no_leg``.

OMN-14843 is the standing proof that those are different layers. On
stability-test, 38 correlations were measured and **26 sat non-terminal** in
the projection — INFERENCE_COMPLETED=15, RECEIVED=9, ROUTED=2 — while the
topic layer was healthy at that same moment (``delegation-request.v1`` HW=100
against 49 completed + 53 failed = 102 terminals). Requests were terminalizing
on the bus while the projection left them mid-flight.

The OMN-16025 verdict comment asked whether the canary would have caught that,
and answered itself: *"No. And that is the finding, not a reassurance: the
canary is structurally blind to it."* A lane in exactly OMN-14843's condition
reported GREEN, because the layer the canary watched was the layer OMN-14843
says was fine.

The load-bearing test
---------------------
``test_stranded_projection_fails_link_two_while_link_four_passes`` is the
OMN-14843 signature reproduced: bus terminal present, projection row stopped
at INFERENCE_COMPLETED. Link 4 must still PASS — the bus really did carry it —
and link 2 must FAIL. If those two ever agree, the canary is blind again and
this whole ticket has regressed.

Every test drives the real handler with injected transport. No network, no
database.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_chain_canary_effect.handlers.handler_chain_canary import (
    HandlerChainCanary,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.enum_chain_link import (
    EnumChainLink,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.enum_chain_link_status import (
    EnumChainLinkStatus,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.model_chain_canary_request import (
    ModelChainCanaryRequest,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.model_chain_canary_result import (
    ModelChainCanaryResult,
)

_PROBE_URL = "http://127.0.0.1:8085"
_BOOTSTRAP = "127.0.0.1:19092"
_SUCCESS_TOPIC = "onex.evt.omnimarket.delegate-skill-completed.v1"
_PROJECTION_DSN = "postgresql://probe@127.0.0.1:5436/omnibase_infra"

# The three non-terminal states OMN-14843 actually measured, with the count it
# found for each. Parametrizing on the measured set rather than one invented
# value keeps the fixture tied to the evidence.
_OMN14843_STRANDED_STATES = ("INFERENCE_COMPLETED", "RECEIVED", "ROUTED")
_TERMINAL_STATES = ("COMPLETED", "FAILED")


def _request(**overrides: object) -> ModelChainCanaryRequest:
    fields: dict[str, object] = {
        "correlation_id": uuid4(),
        "probe_url": _PROBE_URL,
        "budget_ms": 5_000,
        "terminal_bootstrap_servers": _BOOTSTRAP,
        "quarantine_bootstrap_servers": _BOOTSTRAP,
        "projection_dsn": _PROJECTION_DSN,
        "settle_seconds": 0,
    }
    fields.update(overrides)
    return ModelChainCanaryRequest(**fields)  # type: ignore[arg-type]


class _Ingress:
    def __init__(self, response: dict[str, object] | None = None) -> None:
        self.response = response if response is not None else {"ok": True}

    async def __call__(
        self, url: str, body: dict[str, object], timeout_s: float
    ) -> tuple[dict[str, object] | None, str, int]:
        return self.response, "", 42


class _Quarantine:
    def __init__(self, found: bool | None = False) -> None:
        self.found = found

    async def __call__(
        self,
        bootstrap: str,
        topic: str,
        correlation_id: str,
        max_records: int,
        timeout_s: float,
    ) -> tuple[bool | None, int, str]:
        return self.found, 500, ""


class _TerminalReadback:
    def __init__(self, found: str | None = _SUCCESS_TOPIC) -> None:
        self.found = found
        self.calls: list[str] = []

    async def __call__(
        self,
        bootstrap: str,
        topics: tuple[str, ...],
        correlation_id: str,
        max_records: int,
        timeout_s: float,
    ) -> tuple[str | None, int, str]:
        self.calls.append(correlation_id)
        return self.found, 120, ""


class _ProjectionReadback:
    """Injected correlation-scoped read of ``delegation_workflow_state``.

    ``state`` is the FSM state the projection holds for the probe's own
    correlation id, following the same three-way convention the terminal
    readback already uses: a state name for "row found", ``""`` for "read, no
    row", and ``None`` for "the read could not be completed" — which must never
    be reported as either a pass or a clean miss.
    """

    def __init__(self, state: str | None = "COMPLETED", error: str = "") -> None:
        self.state = state
        self.error = error
        self.calls: list[tuple[str, str, float]] = []

    async def __call__(
        self, dsn: str, correlation_id: str, timeout_s: float
    ) -> tuple[str | None, str]:
        self.calls.append((dsn, correlation_id, timeout_s))
        return self.state, self.error


def _handler(
    projection: _ProjectionReadback | None = None,
    terminal_readback: _TerminalReadback | None = None,
    quarantine: _Quarantine | None = None,
) -> HandlerChainCanary:
    return HandlerChainCanary(
        ingress=_Ingress(),
        quarantine_scan=quarantine or _Quarantine(found=False),
        terminal_readback=terminal_readback or _TerminalReadback(),
        projection_readback=projection or _ProjectionReadback(),
        kill_switch_disabled=False,
    )


def _link(result: ModelChainCanaryResult, link: EnumChainLink) -> EnumChainLinkStatus:
    for verdict in result.link_verdicts:
        if verdict.link is link:
            return verdict.status
    raise AssertionError(f"receipt carries no verdict for {link}")


# -- AC2: the OMN-14843 signature — the load-bearing test -----------------


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("stranded_state", _OMN14843_STRANDED_STATES)
async def test_stranded_projection_fails_link_two_while_link_four_passes(
    stranded_state: str,
) -> None:
    """Bus terminal present, projection stranded mid-FSM. AC2 verbatim.

    This is the condition OMN-14843 measured and the canary was blind to. The
    two links must disagree: link 4 saw a real terminal on the bus and is
    entitled to PASS, and link 2 read a real row that never terminalized and
    must FAIL. If they ever agree here, link 2 is being inferred from the bus
    again rather than read from the projection.
    """
    handler = _handler(
        projection=_ProjectionReadback(state=stranded_state),
        terminal_readback=_TerminalReadback(found=_SUCCESS_TOPIC),
    )
    result = await handler.handle(_request())

    assert _link(result, EnumChainLink.ROUTING_PROJECTED) is EnumChainLinkStatus.FAIL
    assert _link(result, EnumChainLink.TERMINAL_ON_BUS) is EnumChainLinkStatus.PASS


# -- AC4: the four hermetic cases ----------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_state", _TERMINAL_STATES)
async def test_terminal_projection_passes_link_two(terminal_state: str) -> None:
    """A row that reached a terminal FSM state discharges link 2."""
    handler = _handler(projection=_ProjectionReadback(state=terminal_state))
    result = await handler.handle(_request())

    assert _link(result, EnumChainLink.ROUTING_PROJECTED) is EnumChainLinkStatus.PASS


@pytest.mark.unit
@pytest.mark.asyncio
async def test_absent_projection_row_fails_link_two() -> None:
    """No row for this correlation id is a failure, not a skip.

    Kept distinct from STRANDED in the status enum on purpose: a missing row
    may mean the routing decision was never published, whereas a stranded row
    means it was published and the projection stopped. Both are non-passing,
    but they send you to different layers.
    """
    handler = _handler(projection=_ProjectionReadback(state=""))
    result = await handler.handle(_request())

    assert _link(result, EnumChainLink.ROUTING_PROJECTED) is EnumChainLinkStatus.FAIL


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unreadable_projection_fails_closed_to_error() -> None:
    """A read that could not complete makes no claim — never PASS, never FAIL.

    ERROR rather than FAIL because the canary is not entitled to call the link
    bad on evidence it failed to collect.
    """
    handler = _handler(
        projection=_ProjectionReadback(state=None, error="relation does not exist")
    )
    result = await handler.handle(_request())

    assert _link(result, EnumChainLink.ROUTING_PROJECTED) is EnumChainLinkStatus.ERROR


# -- AC3: SKIP is not PASS ------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unconfigured_projection_is_not_configured_never_pass() -> None:
    """No projection configured reports NOT_CONFIGURED, and never falls back.

    The fallback is the defect this ticket family exists to remove: a green
    bus terminal must not be allowed to stand in for a projection nobody read.
    """
    handler = _handler(terminal_readback=_TerminalReadback(found=_SUCCESS_TOPIC))
    result = await handler.handle(_request(projection_dsn=""))

    assert (
        _link(result, EnumChainLink.ROUTING_PROJECTED)
        is EnumChainLinkStatus.NOT_CONFIGURED
    )
    assert _link(result, EnumChainLink.TERMINAL_ON_BUS) is EnumChainLinkStatus.PASS


# -- AC1: the status is falsifiable and correlation-scoped ----------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_link_two_status_changes_when_the_row_is_absent() -> None:
    """AC1's falsifier: the same run with and without the row must differ.

    AC1 is falsified by "a run whose link-2 status does not change when the
    projection row is absent" — so that comparison is the assertion.
    """
    present = await _handler(projection=_ProjectionReadback(state="COMPLETED")).handle(
        _request()
    )
    absent = await _handler(projection=_ProjectionReadback(state="")).handle(_request())

    assert _link(present, EnumChainLink.ROUTING_PROJECTED) is not _link(
        absent, EnumChainLink.ROUTING_PROJECTED
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_projection_is_read_for_the_probes_own_correlation_id() -> None:
    """Both legs must ask about the SAME id — the one this run minted.

    The handler mints a fresh ``uuid4()`` per ``handle()`` call rather than
    trusting the request, so the assertion is cross-leg agreement rather than
    a comparison against the request: link 2 and link 4 must be talking about
    one delegation. If they diverge, the two links describe different runs and
    the OMN-14843 comparison this ticket rests on is meaningless.

    Scoping also has to be per-correlation, not table-wide: a table-wide check
    would go green on somebody else's terminal row, the same class of error as
    reading the ingress response.
    """
    projection = _ProjectionReadback(state="COMPLETED")
    terminal = _TerminalReadback(found=_SUCCESS_TOPIC)
    await _handler(projection=projection, terminal_readback=terminal).handle(_request())

    projection_ids = [call[1] for call in projection.calls]
    assert len(projection_ids) == 1
    assert projection_ids == terminal.calls
