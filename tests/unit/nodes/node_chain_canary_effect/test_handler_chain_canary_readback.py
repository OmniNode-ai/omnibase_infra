# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-16931 — the terminal verdict must come off the BUS, not off the ingress.

The defect these tests pin
--------------------------
chain-canary run 33251822642 (2026-08-29T12:10:26Z, dev lane) reported
``terminal_missing`` at 4,369 ms of a 120,000 ms budget. The runtime log for
that run's own correlation id ``2e0e682f-ef70-4b14-9af6-ae2042e923fa`` shows
the terminal was published to
``onex.evt.omnimarket.delegate-skill-completed.v1`` at 12:10:23 — the chain
carried the request. The canary said otherwise because ``_decide`` derived
``terminal_landed`` from the synchronous ingress HTTP response, and that
response carried ``ok=false`` (a provider 429 on an escalation rung the local
model had already answered — OMN-16932).

Two directions of the same error, and both are tested here:

* **False RED.** Ingress lies (or says nothing) while the terminal IS on the
  bus → link 4 must be PASS on the readback.
* **False GREEN.** Ingress claims a terminal that was never published → link 4
  must be FAIL. OMN-15468 is the live proof that ``ok=true`` on this lane is
  not evidence of anything durable.

Why the per-link verdicts are in the same ticket
------------------------------------------------
OMN-16025 is a FIVE-link gate. This canary probes three of them and has never
had a leg for link 2 (projection readback, OMN-16963) or link 5 (ledger chain
+ replay, OMN-16964). Its single scalar verdict let a 3-link probe render as a
5-link proof — the 2h schedule's one GREEN was read exactly that way. So the
receipt now carries a status per link, and a link with no leg says ``no_leg``
and names the ticket that owes it.

Every test drives the real handler with injected transport. No network.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_infra.enums.generated.enum_omnimarket_topic import EnumOmnimarketTopic
from omnibase_infra.nodes.node_chain_canary_effect.handlers.handler_chain_canary import (
    HandlerChainCanary,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.enum_chain_canary_verdict import (
    EnumChainCanaryVerdict,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.enum_chain_link import (
    EnumChainLink,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.enum_chain_link_status import (
    EnumChainLinkStatus,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.enum_terminal_readback_status import (
    EnumTerminalReadbackStatus,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.model_chain_canary_request import (
    ModelChainCanaryRequest,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.model_chain_canary_result import (
    ModelChainCanaryResult,
)

_PROBE_URL = "http://runtime.invalid:8085"
_BOOTSTRAP = "broker.invalid:19092"
# Read off the generated topic enum rather than typed as literals — the same
# values the request model defaults to, so a topic rename cannot leave these
# fixtures asserting a topic nothing publishes to.
_SUCCESS_TOPIC = EnumOmnimarketTopic.EVT_DELEGATE_SKILL_COMPLETED_V1.value
_FAILURE_TOPIC = EnumOmnimarketTopic.EVT_DELEGATE_SKILL_FAILED_V1.value


def _request(**overrides: object) -> ModelChainCanaryRequest:
    fields: dict[str, object] = {
        "correlation_id": uuid4(),
        "probe_url": _PROBE_URL,
        "budget_ms": 5_000,
        "terminal_bootstrap_servers": _BOOTSTRAP,
        "quarantine_bootstrap_servers": _BOOTSTRAP,
        "settle_seconds": 0,
    }
    fields.update(overrides)
    return ModelChainCanaryRequest(**fields)  # type: ignore[arg-type]


class _Ingress:
    def __init__(
        self,
        response: dict[str, object] | None = None,
        error: str = "",
        elapsed_ms: int = 42,
    ) -> None:
        self.response = response
        self.error = error
        self.elapsed_ms = elapsed_ms
        self.calls: list[tuple[str, dict[str, object], float]] = []

    async def __call__(
        self, url: str, body: dict[str, object], timeout_s: float
    ) -> tuple[dict[str, object] | None, str, int]:
        self.calls.append((url, body, timeout_s))
        return self.response, self.error, self.elapsed_ms


class _Quarantine:
    def __init__(self, found: bool | None = False, error: str = "", scanned: int = 500):
        self.found = found
        self.error = error
        self.scanned = scanned
        self.calls: list[tuple[str, str, str, int, float]] = []

    async def __call__(
        self,
        bootstrap: str,
        topic: str,
        correlation_id: str,
        max_records: int,
        timeout_s: float,
    ) -> tuple[bool | None, int, str]:
        self.calls.append((bootstrap, topic, correlation_id, max_records, timeout_s))
        return self.found, self.scanned, self.error


class _TerminalReadback:
    """Injected broker readback of the declared terminal topics.

    ``found`` is the topic the correlation id was read back FROM: ``""`` for
    "scanned, not there", ``None`` for "the scan could not be completed" —
    which must never be reported as either a pass or a clean miss.
    """

    def __init__(self, found: str | None = "", error: str = "", scanned: int = 120):
        self.found = found
        self.error = error
        self.scanned = scanned
        self.calls: list[tuple[str, tuple[str, ...], str, int, float]] = []

    async def __call__(
        self,
        bootstrap: str,
        topics: tuple[str, ...],
        correlation_id: str,
        max_records: int,
        timeout_s: float,
    ) -> tuple[str | None, int, str]:
        self.calls.append((bootstrap, topics, correlation_id, max_records, timeout_s))
        return self.found, self.scanned, self.error


def _handler(
    ingress: _Ingress,
    terminal_readback: _TerminalReadback | None = None,
    quarantine: _Quarantine | None = None,
) -> HandlerChainCanary:
    return HandlerChainCanary(
        ingress=ingress,
        quarantine_scan=quarantine or _Quarantine(found=False),
        terminal_readback=terminal_readback or _TerminalReadback(found=_SUCCESS_TOPIC),
        kill_switch_disabled=False,
    )


def _link(result: ModelChainCanaryResult, link: EnumChainLink) -> EnumChainLinkStatus:
    for verdict in result.link_verdicts:
        if verdict.link is link:
            return verdict.status
    raise AssertionError(f"receipt carries no verdict for {link}")


# -- AC1/AC2: the 429 fixture — ingress ok=false, terminal on the bus ------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ingress_error_with_terminal_on_the_bus_is_not_terminal_missing() -> None:
    """The exact run-33251822642 shape. RED-first for OMN-16931 AC2.

    The ingress reports the provider 429 that killed an escalation rung; the
    terminal was published for this correlation id anyway. The old code called
    this ``terminal_missing`` and sent an operator hunting a dead chain.
    """
    readback = _TerminalReadback(found=_SUCCESS_TOPIC)
    handler = _handler(
        _Ingress(
            response={
                "ok": False,
                "error": {
                    "code": "dispatch_error",
                    "message": "RuntimeError: provider HTTP 429 quota exceeded",
                },
            },
            elapsed_ms=4_369,
        ),
        terminal_readback=readback,
    )

    result = await handler.handle(_request())

    assert result.verdict is EnumChainCanaryVerdict.INGRESS_ERROR_TERMINAL_PRESENT
    assert result.verdict is not EnumChainCanaryVerdict.TERMINAL_MISSING
    assert result.terminal_readback_status is EnumTerminalReadbackStatus.FOUND
    assert result.terminal_topic == _SUCCESS_TOPIC
    # Link 4 is discharged by the readback even though the ingress said no.
    assert _link(result, EnumChainLink.TERMINAL_ON_BUS) is EnumChainLinkStatus.PASS
    # The readback was asked about THIS run's correlation id, on the declared
    # terminal topics.
    bootstrap, topics, correlation, _, _ = readback.calls[0]
    assert bootstrap == _BOOTSTRAP
    assert correlation == str(result.probe_correlation_id)
    assert _SUCCESS_TOPIC in topics and _FAILURE_TOPIC in topics


@pytest.mark.unit
@pytest.mark.asyncio
async def test_link_four_passes_when_the_ingress_flag_is_absent_entirely() -> None:
    """A response with no ``ok`` and no ``terminal_event`` at all.

    AC1's falsification test: the verdict for link 4 must not move when only
    the ingress body changes. The bus is the evidence.
    """
    handler = _handler(
        _Ingress(response={"accepted": True}),
        terminal_readback=_TerminalReadback(found=_SUCCESS_TOPIC),
    )

    result = await handler.handle(_request())

    assert _link(result, EnumChainLink.TERMINAL_ON_BUS) is EnumChainLinkStatus.PASS
    assert result.terminal_readback_status is EnumTerminalReadbackStatus.FOUND


# -- the inverse error: ingress claims a terminal that never landed ---------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ingress_claimed_terminal_never_published_is_red() -> None:
    """OMN-15468's shape: ok=true + a terminal_event the bus never saw.

    RED-first for OMN-16931 AC3. Under the old code this was GREEN — the
    single most dangerous outcome, because it is the one that closes a gate.
    """
    handler = _handler(
        _Ingress(
            response={
                "ok": True,
                "terminal_event": "omnimarket.delegate-skill-completed",
            }
        ),
        terminal_readback=_TerminalReadback(found=""),
    )

    result = await handler.handle(_request())

    assert result.verdict is EnumChainCanaryVerdict.TERMINAL_MISSING
    assert result.success is False
    assert result.terminal_readback_status is EnumTerminalReadbackStatus.NOT_FOUND
    assert _link(result, EnumChainLink.TERMINAL_ON_BUS) is EnumChainLinkStatus.FAIL
    # The ingress CLAIM is still recorded — it is the discrepancy that names
    # the defect — but it is recorded as a claim, not as the terminal.
    assert result.ingress_terminal_event == "omnimarket.delegate-skill-completed"
    assert result.terminal_event == ""
    assert "ingress" in result.detail.lower()


# -- AC4 coverage: the remaining readback states ---------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_green_requires_both_a_clean_ingress_and_a_readback_terminal() -> None:
    handler = _handler(
        _Ingress(response={"ok": True, "terminal_event": "delegate-skill-completed"}),
        terminal_readback=_TerminalReadback(found=_SUCCESS_TOPIC),
    )

    result = await handler.handle(_request())

    assert result.verdict is EnumChainCanaryVerdict.GREEN
    assert result.success is True
    assert result.terminal_event == _SUCCESS_TOPIC


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ingress_error_and_no_terminal_is_terminal_missing() -> None:
    handler = _handler(
        _Ingress(response={"ok": False, "error": {"code": "dispatch_timeout"}}),
        terminal_readback=_TerminalReadback(found=""),
    )

    result = await handler.handle(_request())

    assert result.verdict is EnumChainCanaryVerdict.TERMINAL_MISSING
    assert result.success is False
    assert _link(result, EnumChainLink.TERMINAL_ON_BUS) is EnumChainLinkStatus.FAIL


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unrunnable_readback_fails_closed() -> None:
    """A configured-but-broken readback is RED, never a silent pass."""
    handler = _handler(
        _Ingress(response={"ok": True, "terminal_event": "delegate-skill-completed"}),
        terminal_readback=_TerminalReadback(found=None, error="broker unreachable"),
    )

    result = await handler.handle(_request())

    assert result.verdict is EnumChainCanaryVerdict.TERMINAL_READBACK_FAILED
    assert result.success is False
    assert result.terminal_readback_status is EnumTerminalReadbackStatus.ERROR
    assert _link(result, EnumChainLink.TERMINAL_ON_BUS) is EnumChainLinkStatus.ERROR


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unconfigured_readback_is_red_not_a_fallback_to_the_ingress() -> None:
    """No broker configured means NO claim about link 4 — so no green.

    Deliberately asymmetric with the quarantine leg, which may be skipped
    without blocking a green: quarantine is a corroborating negative check,
    the terminal readback IS the claim. Falling back to the ingress response
    here would restore the whole defect.
    """
    readback = _TerminalReadback(found=_SUCCESS_TOPIC)
    handler = _handler(
        _Ingress(response={"ok": True, "terminal_event": "delegate-skill-completed"}),
        terminal_readback=readback,
    )

    result = await handler.handle(_request(terminal_bootstrap_servers=""))

    assert result.verdict is EnumChainCanaryVerdict.TERMINAL_READBACK_NOT_CONFIGURED
    assert result.success is False
    assert readback.calls == []
    assert (
        result.terminal_readback_status
        is EnumTerminalReadbackStatus.SKIPPED_NOT_CONFIGURED
    )
    assert (
        _link(result, EnumChainLink.TERMINAL_ON_BUS)
        is EnumChainLinkStatus.NOT_CONFIGURED
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_quarantined_still_outranks_everything() -> None:
    """OMN-16773's ranking survives: the quarantine hit names the defect."""
    handler = _handler(
        _Ingress(response={"ok": False, "error": {"code": "dispatch_timeout"}}),
        terminal_readback=_TerminalReadback(found=""),
        quarantine=_Quarantine(found=True),
    )

    result = await handler.handle(_request())

    assert result.verdict is EnumChainCanaryVerdict.QUARANTINED
    assert _link(result, EnumChainLink.DELEGATED_EXECUTION) is EnumChainLinkStatus.FAIL


@pytest.mark.unit
@pytest.mark.asyncio
async def test_failure_topic_terminal_discharges_link_four_but_fails_link_three() -> (
    None
):
    """A terminal on the FAILURE topic is still emission-confirmed.

    Link 4 asks whether the emission landed on the bus; it did. Link 3 asks
    whether the delegated execution completed; it did not.
    """
    handler = _handler(
        _Ingress(response={"ok": False, "error": {"code": "dispatch_error"}}),
        terminal_readback=_TerminalReadback(found=_FAILURE_TOPIC),
    )

    result = await handler.handle(_request())

    assert _link(result, EnumChainLink.TERMINAL_ON_BUS) is EnumChainLinkStatus.PASS
    assert _link(result, EnumChainLink.DELEGATED_EXECUTION) is EnumChainLinkStatus.FAIL
    assert result.terminal_topic == _FAILURE_TOPIC


# -- per-link honesty: a 3-link probe may never render as a 5-link proof ----


@pytest.mark.unit
@pytest.mark.asyncio
async def test_receipt_carries_a_verdict_for_all_five_links() -> None:
    handler = _handler(
        _Ingress(response={"ok": True, "terminal_event": "delegate-skill-completed"}),
        terminal_readback=_TerminalReadback(found=_SUCCESS_TOPIC),
    )

    result = await handler.handle(_request())

    assert result.links_total == 5
    assert {v.link for v in result.link_verdicts} == set(EnumChainLink)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_best_possible_run_is_still_not_a_five_link_proof() -> None:
    """The load-bearing honesty test.

    Everything this probe CAN assert passes. It is still 3 of 5, and the
    receipt must say so — otherwise the 2h schedule keeps reporting a
    five-link gate as green off a three-link probe, which is exactly what
    happened on run 33215999994.
    """
    handler = _handler(
        _Ingress(response={"ok": True, "terminal_event": "delegate-skill-completed"}),
        terminal_readback=_TerminalReadback(found=_SUCCESS_TOPIC),
    )

    result = await handler.handle(_request())

    assert result.verdict is EnumChainCanaryVerdict.GREEN
    assert result.links_proven == 3
    assert result.chain_proof_complete is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_every_link_now_has_a_leg_and_owes_no_ticket() -> None:
    """Both debts are paid: no link reports ``NO_LEG`` any more.

    This test used to assert the opposite — links 2 and 5 named the tickets
    that owed them a leg. OMN-16963 paid link 2's and OMN-16964 paid link 5's,
    so the assertion inverts into a regression guard: if a future change ever
    drops a leg back to ``NO_LEG``, the probe has silently shrunk and this
    catches it.

    Both now report ``NOT_CONFIGURED`` on this fixture, which is a different
    fact from ``NO_LEG``: the instrument exists and was not pointed at
    anything, rather than not existing at all. Both are non-passing, and
    keeping them distinct is the point — see
    ``test_handler_chain_canary_projection.py`` and
    ``test_handler_chain_canary_ledger.py`` for their own coverage.
    """
    handler = _handler(
        _Ingress(response={"ok": True, "terminal_event": "delegate-skill-completed"}),
        terminal_readback=_TerminalReadback(found=_SUCCESS_TOPIC),
    )

    result = await handler.handle(_request())

    assert all(
        verdict.status is not EnumChainLinkStatus.NO_LEG
        for verdict in result.link_verdicts
    )
    assert all(verdict.owning_ticket == "" for verdict in result.link_verdicts)

    by_link = {v.link: v for v in result.link_verdicts}
    assert (
        by_link[EnumChainLink.ROUTING_PROJECTED].status
        is EnumChainLinkStatus.NOT_CONFIGURED
    )
    assert (
        by_link[EnumChainLink.LEDGER_REPLAY].status
        is EnumChainLinkStatus.NOT_CONFIGURED
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unreachable_ingress_does_not_claim_links_it_never_evaluated() -> None:
    handler = _handler(
        _Ingress(response=None, error="connection refused"),
        terminal_readback=_TerminalReadback(found=_SUCCESS_TOPIC),
    )

    result = await handler.handle(_request())

    assert result.verdict is EnumChainCanaryVerdict.INGRESS_UNREACHABLE
    assert _link(result, EnumChainLink.INGRESS_ACCEPTED) is EnumChainLinkStatus.FAIL
    assert (
        _link(result, EnumChainLink.TERMINAL_ON_BUS)
        is EnumChainLinkStatus.NOT_EVALUATED
    )
    assert result.links_proven == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_kill_switch_reports_every_link_unevaluated() -> None:
    ingress = _Ingress(response={"ok": True, "terminal_event": "x"})
    readback = _TerminalReadback(found=_SUCCESS_TOPIC)
    handler = HandlerChainCanary(
        ingress=ingress,
        quarantine_scan=_Quarantine(),
        terminal_readback=readback,
        kill_switch_disabled=True,
    )

    result = await handler.handle(_request())

    assert result.verdict is EnumChainCanaryVerdict.SKIPPED_DISABLED
    assert ingress.calls == [] and readback.calls == []
    assert result.links_proven == 0
    assert all(
        v.status is EnumChainLinkStatus.NOT_EVALUATED for v in result.link_verdicts
    )


# -- budget: the readback window must span the remaining budget ------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_readback_window_covers_the_rest_of_the_budget() -> None:
    """AC3 says 'inside the budget', so the readback must wait that long.

    Run 33251822642's ingress answered at 4,369 ms of 120,000 — giving up
    then is the bug. The readback window must be the remainder.
    """
    readback = _TerminalReadback(found=_SUCCESS_TOPIC)
    handler = _handler(
        _Ingress(response={"ok": False, "error": {"code": "x"}}, elapsed_ms=4_369),
        terminal_readback=readback,
    )

    await handler.handle(_request(budget_ms=120_000))

    _, _, _, _, timeout_s = readback.calls[0]
    assert timeout_s >= (120_000 - 4_369) / 1000.0
