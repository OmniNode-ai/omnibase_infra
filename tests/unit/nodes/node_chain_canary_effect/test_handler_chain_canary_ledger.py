# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-16964 — link 5 must assemble, replay, and verify HONESTLY.

The defect these tests pin
--------------------------
OMN-16025 link 5 reads *"Complete ledger chain + replay green through an
HONEST tier-2 verifier (SKIP != PASS)."* The OMN-16025 verdict comment records
it as flatly unexercised: *"5. Ledger chain + replay green, honest tier-2
verifier — unexercised"*, and concludes *"One of five links is proven. The
gate is nowhere near green, and the 2h schedule reporting GREEN once has been
over-reading a 3-link probe as a 5-link proof."*

The canary assembles no ledger chain, runs no replay, and invokes no verifier.
Nothing else on a schedule does either.

Why "honest" is load-bearing, not decoration
--------------------------------------------
The failure this link catches is a verifier that reports a pass **because it
never ran the check**. That is the same shape OMN-16773 named for its own
quarantine leg (``SKIPPED_NOT_CONFIGURED`` is not ``CLEAN``) and the same shape
OMN-16931 found in the terminal leg (a verdict derived from a claim rather
than from evidence). ``test_verifier_skip_is_never_green`` is the test that
holds that line: if a SKIP ever renders as PASS, this link is decorative.

Every test drives the real handler with injected transport. No network, no
ledger store, no replay engine.
"""

from __future__ import annotations

import asyncio
import sys
import time
from uuid import uuid4

import pytest

from omnibase_infra.enums.generated.enum_omnimarket_topic import EnumOmnimarketTopic
from omnibase_infra.nodes.node_chain_canary_effect.handlers.handler_chain_canary import (
    HandlerChainCanary,
    _replay_ledger_chain_via_asyncpg,
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
from omnibase_infra.nodes.node_chain_canary_effect.models.model_chain_canary_request import (
    ModelChainCanaryRequest,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.model_chain_canary_result import (
    ModelChainCanaryResult,
)

# Endpoints use the RFC 2606 reserved `.invalid` TLD, matching the sibling
# canary suites. This is isolation, not cosmetics: every leg here is stubbed,
# so a loopback literal would silently reach a real local Redpanda or Postgres
# if one happened to be up, and the suite would stop being hermetic without
# ever failing. `.invalid` cannot resolve, so it cannot reach anything.
_PROBE_URL = "http://runtime.invalid:8085"
_BOOTSTRAP = "broker.invalid:19092"
_SUCCESS_TOPIC = EnumOmnimarketTopic.EVT_DELEGATE_SKILL_COMPLETED_V1.value
_LEDGER_SOURCE = "postgresql://probe@db.invalid:5436/omnibase_infra"
_PROJECTION_DSN = "postgresql://probe@db.invalid:5436/omnibase_infra"

# The hops a complete chain must carry, end to end. A run missing any one of
# these is CHAIN_INCOMPLETE — "no gaps tolerated silently" is the scope's own
# wording.
_FULL_CHAIN = ("received", "routed", "inference_completed", "terminal")


def _request(**overrides: object) -> ModelChainCanaryRequest:
    fields: dict[str, object] = {
        "correlation_id": uuid4(),
        "probe_url": _PROBE_URL,
        "budget_ms": 5_000,
        "terminal_bootstrap_servers": _BOOTSTRAP,
        "quarantine_bootstrap_servers": _BOOTSTRAP,
        "ledger_source": _LEDGER_SOURCE,
        "expected_ledger_hops": _FULL_CHAIN,
        # OMN-16963: link 2 must be configured here too. A non-passing link 2
        # outranks every ledger verdict in `_decide`, so without this these
        # fixtures would all stop at PROJECTION_READBACK_NOT_CONFIGURED and
        # never reach the link-5 branch they exist to exercise.
        "projection_dsn": _PROJECTION_DSN,
        "settle_seconds": 0,
    }
    fields.update(overrides)
    return ModelChainCanaryRequest(**fields)  # type: ignore[arg-type]


class _Ingress:
    async def __call__(
        self, url: str, body: dict[str, object], timeout_s: float
    ) -> tuple[dict[str, object] | None, str, int]:
        return {"ok": True}, "", 42


class _Quarantine:
    async def __call__(
        self,
        bootstrap: str,
        topic: str,
        correlation_id: str,
        max_records: int,
        timeout_s: float,
    ) -> tuple[bool | None, int, str]:
        return False, 500, ""


class _TerminalReadback:
    async def __call__(
        self,
        bootstrap: str,
        topics: tuple[str, ...],
        correlation_id: str,
        max_records: int,
        timeout_s: float,
    ) -> tuple[str | None, int, str]:
        return _SUCCESS_TOPIC, 120, ""


class _ProjectionReadback:
    async def __call__(
        self, dsn: str, correlation_id: str, timeout_s: float
    ) -> tuple[str | None, str]:
        return "COMPLETED", ""


class _LedgerReplay:
    """Injected ledger assembly + replay + tier-2 verify.

    Returns raw facts rather than a verdict, so the classification — and in
    particular SKIP != PASS — lives in the handler where it can be pinned.

    ``hops`` is ``None`` when the ledger could not be read at all, which must
    never be reported as either a pass or a clean miss. ``verifier_verdict``
    carries the tier-2 verifier's own word: ``"pass"``, ``"fail"``, or
    ``"skip"``.
    """

    def __init__(
        self,
        hops: tuple[str, ...] | None = _FULL_CHAIN,
        replay_green: bool = True,
        verifier_verdict: str = "pass",
        error: str = "",
    ) -> None:
        self.hops = hops
        self.replay_green = replay_green
        self.verifier_verdict = verifier_verdict
        self.error = error
        self.calls: list[str] = []

    async def __call__(
        self, source: str, correlation_id: str, timeout_s: float
    ) -> tuple[tuple[str, ...] | None, bool, str, str]:
        self.calls.append(correlation_id)
        return self.hops, self.replay_green, self.verifier_verdict, self.error


def _handler(ledger: _LedgerReplay | None = None) -> HandlerChainCanary:
    return HandlerChainCanary(
        ingress=_Ingress(),
        quarantine_scan=_Quarantine(),
        terminal_readback=_TerminalReadback(),
        projection_readback=_ProjectionReadback(),
        ledger_replay=ledger or _LedgerReplay(),
        kill_switch_disabled=False,
    )


def _link(result: ModelChainCanaryResult, link: EnumChainLink) -> EnumChainLinkStatus:
    for verdict in result.link_verdicts:
        if verdict.link is link:
            return verdict.status
    raise AssertionError(f"receipt carries no verdict for {link}")


# -- AC2: the honesty test — the reason this link says "HONEST" -----------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_verifier_skip_is_never_green() -> None:
    """A tier-2 SKIP must not render as PASS. AC2 verbatim.

    This is the load-bearing test of the ticket. A verifier that reports a
    pass because it never ran the check is the exact defect OMN-16025's
    "SKIP != PASS" wording exists to prevent. If this ever goes green, link 5
    is decoration.
    """
    handler = _handler(_LedgerReplay(verifier_verdict="skip"))
    result = await handler.handle(_request())

    status = _link(result, EnumChainLink.LEDGER_REPLAY)
    assert status is not EnumChainLinkStatus.PASS
    assert status is EnumChainLinkStatus.FAIL


@pytest.mark.unit
@pytest.mark.asyncio
async def test_skip_does_not_count_toward_links_proven() -> None:
    """A SKIP must not inflate the proof count either.

    Asserting only the per-link status would leave the other half of the
    defect open: ``links_proven`` is what a reader scans, and a SKIP counted
    there re-creates the 3-link-probe-reads-as-5-link-proof error at the
    summary level.
    """
    skipped = await _handler(_LedgerReplay(verifier_verdict="skip")).handle(_request())
    verified = await _handler(_LedgerReplay()).handle(_request())

    assert skipped.links_proven == verified.links_proven - 1


# -- AC4: the four hermetic cases ----------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_complete_chain_replayed_green_passes_link_five() -> None:
    """Complete chain + green replay + a verifier that ran and passed."""
    result = await _handler(_LedgerReplay()).handle(_request())

    assert _link(result, EnumChainLink.LEDGER_REPLAY) is EnumChainLinkStatus.PASS


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("dropped", _FULL_CHAIN)
async def test_missing_hop_fails_link_five(dropped: str) -> None:
    """Any single missing hop fails the link — no gap tolerated silently.

    Parametrized across every hop rather than dropping one arbitrary link,
    so a chain check that only inspects the head or the tail is caught.
    """
    partial = tuple(hop for hop in _FULL_CHAIN if hop != dropped)
    handler = _handler(_LedgerReplay(hops=partial))
    result = await handler.handle(_request())

    assert _link(result, EnumChainLink.LEDGER_REPLAY) is EnumChainLinkStatus.FAIL


@pytest.mark.unit
@pytest.mark.asyncio
async def test_failed_replay_fails_link_five() -> None:
    """Complete evidence that did not reproduce is a failure, not an error."""
    handler = _handler(_LedgerReplay(replay_green=False))
    result = await handler.handle(_request())

    assert _link(result, EnumChainLink.LEDGER_REPLAY) is EnumChainLinkStatus.FAIL


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unavailable_verifier_fails_closed_to_error() -> None:
    """AC3: an unreadable ledger makes no claim — never a silent pass.

    ERROR rather than FAIL because the canary is not entitled to call the
    link bad on evidence it failed to collect.
    """
    handler = _handler(_LedgerReplay(hops=None, error="ledger store unreachable"))
    result = await handler.handle(_request())

    assert _link(result, EnumChainLink.LEDGER_REPLAY) is EnumChainLinkStatus.ERROR


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unconfigured_ledger_is_not_configured_never_pass() -> None:
    """No ledger source configured reports NOT_CONFIGURED, and never falls back."""
    result = await _handler(_LedgerReplay()).handle(_request(ledger_source=""))

    assert (
        _link(result, EnumChainLink.LEDGER_REPLAY) is EnumChainLinkStatus.NOT_CONFIGURED
    )


# -- AC1: the status is falsifiable and correlation-scoped ----------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_link_five_status_changes_when_a_hop_is_removed() -> None:
    """AC1's falsifier: same run with and without a hop must differ.

    AC1 is falsified by "a run whose link-5 status does not change when a
    ledger hop is removed", so that comparison is the assertion.
    """
    complete = await _handler(_LedgerReplay()).handle(_request())
    missing = await _handler(_LedgerReplay(hops=_FULL_CHAIN[:-1])).handle(_request())

    assert _link(complete, EnumChainLink.LEDGER_REPLAY) is not _link(
        missing, EnumChainLink.LEDGER_REPLAY
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ledger_is_assembled_for_the_probes_own_correlation_id() -> None:
    """Scoped to this run's own minted id, not to the ledger at large."""
    ledger = _LedgerReplay()
    await _handler(ledger).handle(_request())

    assert len(ledger.calls) == 1
    assert ledger.calls[0]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_link_five_no_longer_reports_no_leg() -> None:
    """The debt is paid: link 5 carries a real status and owes no ticket."""
    result = await _handler(_LedgerReplay()).handle(_request())

    verdict = next(
        v for v in result.link_verdicts if v.link is EnumChainLink.LEDGER_REPLAY
    )
    assert verdict.status is not EnumChainLinkStatus.NO_LEG
    assert verdict.owning_ticket == ""


# -- the SCALAR verdict, not just the link (OMN-16964) --------------------
#
# Link 5 gets the same treatment link 2 got: `_decide()` receives the ledger
# status, so a non-passing link 5 can no longer coexist with GREEN. Before
# this, the per-link status was honest -- `_link_five` has always mapped
# VERIFIER_SKIPPED to FAIL -- while the scalar verdict still said green, which
# is the over-reading one level up.


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ledger", "source", "expected"),
    [
        (
            _LedgerReplay(hops=("received", "terminal")),
            _LEDGER_SOURCE,
            EnumChainCanaryVerdict.LEDGER_CHAIN_INCOMPLETE,
        ),
        (
            _LedgerReplay(replay_green=False),
            _LEDGER_SOURCE,
            EnumChainCanaryVerdict.LEDGER_REPLAY_NOT_GREEN,
        ),
        (
            _LedgerReplay(verifier_verdict="skip"),
            _LEDGER_SOURCE,
            EnumChainCanaryVerdict.LEDGER_VERIFIER_SKIPPED,
        ),
        (
            _LedgerReplay(hops=None, error="connection refused"),
            _LEDGER_SOURCE,
            EnumChainCanaryVerdict.LEDGER_REPLAY_UNREADABLE,
        ),
        (
            _LedgerReplay(),
            "",
            EnumChainCanaryVerdict.LEDGER_REPLAY_NOT_CONFIGURED,
        ),
    ],
    ids=["incomplete", "replay_red", "verifier_skip", "unreadable", "not_configured"],
)
async def test_non_passing_link_five_is_never_green(
    ledger: _LedgerReplay,
    source: str,
    expected: EnumChainCanaryVerdict,
) -> None:
    """Each non-passing ledger outcome gets its own scalar verdict.

    Every case here has a healthy ingress, a real terminal on the bus AND a
    terminalized projection -- so links 1, 2 and 4 all pass and the ledger is
    the only thing left that can disprove the run. That is exactly the shape
    that used to return GREEN.
    """
    result = await _handler(ledger).handle(_request(ledger_source=source))

    assert result.verdict is expected
    assert result.success is False
    assert result.chain_proof_complete is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_verifier_skip_reaches_the_scalar_verdict_not_just_the_link() -> None:
    """The SKIP != PASS property, asserted where a reader actually looks.

    ``test_verifier_skip_is_never_green`` above holds the line at the link.
    This holds it at the field people quote. A receipt whose link 5 says FAIL
    while its verdict says GREEN is still a receipt that gets read as green.
    """
    result = await _handler(_LedgerReplay(verifier_verdict="skip")).handle(_request())

    assert _link(result, EnumChainLink.LEDGER_REPLAY) is EnumChainLinkStatus.FAIL
    assert result.verdict is EnumChainCanaryVerdict.LEDGER_VERIFIER_SKIPPED
    assert result.success is False
    assert "checked nothing" in result.detail


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ledger", "source"),
    [
        (_LedgerReplay(hops=("received",)), _LEDGER_SOURCE),
        (_LedgerReplay(replay_green=False), _LEDGER_SOURCE),
        (_LedgerReplay(verifier_verdict="skip"), _LEDGER_SOURCE),
        (_LedgerReplay(verifier_verdict="fail"), _LEDGER_SOURCE),
        (_LedgerReplay(hops=None, error="boom"), _LEDGER_SOURCE),
        (_LedgerReplay(), ""),
    ],
)
async def test_success_and_link_five_can_never_disagree(
    ledger: _LedgerReplay, source: str
) -> None:
    """The invariant, stated directly rather than per-case.

    If link 5 is anything other than PASS, ``success`` must be False. Guards
    the direction the per-case tests cannot: a new ledger status, or a new
    branch in ``_decide``, that quietly reintroduces the defect.
    """
    result = await _handler(ledger).handle(_request(ledger_source=source))

    link_five_passed = (
        _link(result, EnumChainLink.LEDGER_REPLAY) is EnumChainLinkStatus.PASS
    )
    assert link_five_passed is False
    assert result.success is False


# -- the replay's own resilience (same two defects as link 2's readback) --


class _FakeConnection:
    """Minimal asyncpg connection stand-in."""

    def __init__(
        self,
        *,
        rows: list[dict[str, object]] | None = None,
        fetch_delay_s: float = 0.0,
        close_raises: bool = False,
    ) -> None:
        self._rows = rows if rows is not None else []
        self._fetch_delay_s = fetch_delay_s
        self._close_raises = close_raises
        self.closed = False

    async def fetch(self, query: str, correlation_id: str) -> list[dict[str, object]]:
        if self._fetch_delay_s:
            await asyncio.sleep(self._fetch_delay_s)
        return self._rows

    async def close(self) -> None:
        self.closed = True
        if self._close_raises:
            raise OSError("connection reset while closing")


class _FakeAsyncpg:
    """Stands in for the lazily-imported ``asyncpg`` module."""

    def __init__(
        self, connection: _FakeConnection, *, connect_delay_s: float = 0.0
    ) -> None:
        self._connection = connection
        self._connect_delay_s = connect_delay_s

    async def connect(self, source: str) -> _FakeConnection:
        if self._connect_delay_s:
            await asyncio.sleep(self._connect_delay_s)
        return self._connection


@pytest.mark.unit
@pytest.mark.asyncio
async def test_close_failure_does_not_replace_the_replay_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure to close must not escape and abort the sibling legs.

    ``close()`` ran in a bare ``finally`` outside the ``except``, so an error
    there propagated past the fail-closed return, out of ``asyncio.gather()``,
    and took the terminal, quarantine and projection legs down with it.
    """
    rows: list[dict[str, object]] = [
        {"hop": hop, "replay_green": True, "verifier_verdict": "pass"}
        for hop in _FULL_CHAIN
    ]
    connection = _FakeConnection(rows=rows, close_raises=True)
    monkeypatch.setitem(sys.modules, "asyncpg", _FakeAsyncpg(connection))

    hops, replay_green, verdict, error = await _replay_ledger_chain_via_asyncpg(
        _LEDGER_SOURCE, str(uuid4()), 5.0
    )

    assert hops == _FULL_CHAIN
    assert replay_green is True
    assert verdict == "pass"
    assert error == ""
    assert connection.closed is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_connect_and_query_share_one_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``timeout_s`` is the budget for the leg, not for each call in it.

    Connect and fetch each fit inside the window individually and do not fit
    together, so this passes only if one deadline spans both.
    """
    connection = _FakeConnection(rows=[], fetch_delay_s=0.25)
    monkeypatch.setitem(
        sys.modules, "asyncpg", _FakeAsyncpg(connection, connect_delay_s=0.25)
    )

    started = time.monotonic()
    hops, _replay_green, _verdict, error = await _replay_ledger_chain_via_asyncpg(
        _LEDGER_SOURCE, str(uuid4()), 0.3
    )
    elapsed = time.monotonic() - started

    assert hops is None
    assert error
    assert elapsed < 0.6
