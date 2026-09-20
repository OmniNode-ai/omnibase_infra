# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18872 — link 2 must WAIT for the projection, not sample it once.

The defect these tests pin
--------------------------
``_readback_projection_via_asyncpg`` fired a single ``SELECT`` the instant the
ingress answered. That was sound only while the ingress BLOCKED until the
chain finished. The OMN-18421 move to the tenant-bearing gateway made the
submission asynchronous — the route answers ``202`` in a few hundred
milliseconds — so the one sample started landing tens of seconds before the
projection wrote anything.

Measured on run 35482636275, correlation ``756bae29``: submitted
``01:56:17.26``, projection row created ``01:56:50.60``, ``COMPLETED`` at
``01:57:14.11``, and the canary published ``projection_row_absent`` at
``01:58:17.98`` — 87 seconds after the row it called absent reached a terminal
state. Run 35037024216 is the control from the other side: on the OLD blocking
``/skill`` route the same leg reached 4/5 with link 2 PASSING.

The load-bearing test
---------------------
``test_poll_does_not_stop_at_a_non_terminal_first_sight``. A row is CREATED
non-terminal — ``RECEIVED`` and ``ROUTED`` both occur live on the dev lane —
and reaches ``COMPLETED`` a second or two later. A poll that returned on first
sight of the row would have replaced a wrong ``ROW_ABSENT`` with a wrong
``STRANDED`` and left the canary exactly as red for exactly as wrong a reason.
If that test ever passes while the leg stops at first sight, this fix has
regressed into a different false red.

Every test drives the real transport with an injected ``asyncpg``. No network,
no database.
"""

from __future__ import annotations

import asyncio
import sys
import time
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_chain_canary_effect.handlers import (
    handler_chain_canary as handler_module,
)
from omnibase_infra.nodes.node_chain_canary_effect.handlers.handler_chain_canary import (
    _readback_projection_via_asyncpg,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.enum_projection_readback_status import (
    EnumProjectionReadbackStatus,
)

# RFC 2606 reserved TLD, matching the sibling canary suites: every leg here is
# stubbed, and a loopback literal would silently reach a real local Postgres
# if one happened to be up. `.invalid` cannot resolve, so it cannot reach one.
_PROJECTION_DSN = "postgresql://reader@projection.invalid:5432/omnibase_infra"

# A poll interval short enough that the suite stays fast and long enough that
# "did it poll more than once" is a real question. The shipped default is
# asserted separately by `test_shipped_poll_interval_is_sane`, so speeding the
# tests up here cannot quietly ship a zero-second busy loop.
_FAST_POLL_SECONDS = 0.02


def _completed() -> dict[str, object]:
    return {"state": "COMPLETED", "traffic_class": "unclassified"}


def _non_terminal(state: str) -> dict[str, object]:
    return {"state": state, "traffic_class": "unclassified"}


class _ScriptedConnection:
    """asyncpg connection stand-in whose DATA reads follow a script.

    The role probe is answered separately and is not consumed from the script:
    the OMN-18060 identity check runs once, before any data read, and counting
    it as a poll would make every assertion about read counts off by one.
    """

    def __init__(
        self,
        *,
        rows: list[dict[str, object] | None],
        tail: dict[str, object] | None = None,
        data_raises: BaseException | None = None,
    ) -> None:
        self._rows = list(rows)
        # What every read after the script returns. `None` keeps a row absent
        # for the whole window; a dict keeps it parked in one state.
        self._tail = tail
        self._data_raises = data_raises
        self.data_reads = 0
        self.closed = False

    async def fetchrow(self, query: str, *args: object) -> object:
        if "pg_roles" in query:
            return {
                "rolname": "chain_canary_reader",
                "rolsuper": False,
                "rolbypassrls": False,
            }
        self.data_reads += 1
        if self._data_raises is not None:
            raise self._data_raises
        if self._rows:
            return self._rows.pop(0)
        return self._tail

    async def close(self) -> None:
        self.closed = True


class _FakeAsyncpg:
    def __init__(self, connection: _ScriptedConnection) -> None:
        self._connection = connection

    async def connect(self, dsn: str) -> _ScriptedConnection:
        return self._connection


@pytest.fixture
def fast_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        handler_module, "_PROJECTION_POLL_INTERVAL_SECONDS", _FAST_POLL_SECONDS
    )


def _install(
    monkeypatch: pytest.MonkeyPatch, connection: _ScriptedConnection
) -> _ScriptedConnection:
    monkeypatch.setitem(sys.modules, "asyncpg", _FakeAsyncpg(connection))
    return connection


# -- AC1: a row that appears after the ingress answers is found ------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_row_that_appears_after_the_ingress_answers_is_found(
    monkeypatch: pytest.MonkeyPatch, fast_poll: None
) -> None:
    """The live shape: absent at submission, COMPLETED ~30s later.

    This is run 35482636275 in miniature. Under the single-sample leg the
    first read decided the verdict and the answer was ROW_ABSENT; the row it
    called absent was written moments later and the canary was red for it.
    """
    connection = _install(
        monkeypatch,
        _ScriptedConnection(rows=[None, None, None], tail=_completed()),
    )

    outcome = await _readback_projection_via_asyncpg(_PROJECTION_DSN, str(uuid4()), 5.0)

    assert outcome.status is EnumProjectionReadbackStatus.TERMINAL
    assert outcome.state == "COMPLETED"
    assert outcome.error == ""
    # The point of the fix: it did not decide on the first read.
    assert connection.data_reads >= 4
    assert connection.closed is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_terminal_row_already_present_returns_without_waiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A poll is not a sleep. An answer already there is returned at once.

    Runs against the SHIPPED interval deliberately: if the leg ever waits a
    poll interval before its first read, or sleeps once before returning a
    terminal answer, this is the test that catches it.
    """
    connection = _install(monkeypatch, _ScriptedConnection(rows=[_completed()]))

    started = time.monotonic()
    outcome = await _readback_projection_via_asyncpg(
        _PROJECTION_DSN, str(uuid4()), 30.0
    )
    elapsed = time.monotonic() - started

    assert outcome.status is EnumProjectionReadbackStatus.TERMINAL
    assert connection.data_reads == 1
    assert elapsed < handler_module._PROJECTION_POLL_INTERVAL_SECONDS


# -- AC1, the load-bearing case: first sight is not the answer -------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_poll_does_not_stop_at_a_non_terminal_first_sight(
    monkeypatch: pytest.MonkeyPatch, fast_poll: None
) -> None:
    """RECEIVED then ROUTED then COMPLETED must read TERMINAL, never STRANDED.

    The projection creates the row non-terminal and terminalises it a beat
    later — live counts on the dev lane at the time of writing were RECEIVED
    22 and ROUTED 1 against COMPLETED 303 and FAILED 170. Stopping on first
    sight would report the chain stuck for a run that finished normally.
    """
    connection = _install(
        monkeypatch,
        _ScriptedConnection(
            rows=[None, _non_terminal("RECEIVED"), _non_terminal("ROUTED")],
            tail=_completed(),
        ),
    )

    outcome = await _readback_projection_via_asyncpg(_PROJECTION_DSN, str(uuid4()), 5.0)

    assert outcome.status is EnumProjectionReadbackStatus.TERMINAL
    assert outcome.state == "COMPLETED"
    assert connection.data_reads >= 4


# -- AC4: a genuine absence is still caught, at the deadline ---------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_absent_row_fails_at_the_deadline_with_elapsed_evidence(
    monkeypatch: pytest.MonkeyPatch, fast_poll: None
) -> None:
    """A row that never appears is still ROW_ABSENT — and says what it waited.

    The evidence half is not decoration. An absence found after one read and
    an absence found after a full window are different findings, and reporting
    the first as though it were the second is precisely the defect this ticket
    exists to remove.
    """
    window_s = 0.4
    connection = _install(monkeypatch, _ScriptedConnection(rows=[], tail=None))

    started = time.monotonic()
    outcome = await _readback_projection_via_asyncpg(
        _PROJECTION_DSN, str(uuid4()), window_s
    )
    elapsed = time.monotonic() - started

    assert outcome.status is EnumProjectionReadbackStatus.ROW_ABSENT
    assert "polled for" in outcome.error
    assert "read(s)" in outcome.error
    # It used the window it was given, and did not overrun it.
    assert connection.data_reads > 1
    assert window_s * 0.5 <= elapsed < window_s * 3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_row_still_non_terminal_at_the_deadline_is_stranded(
    monkeypatch: pytest.MonkeyPatch, fast_poll: None
) -> None:
    """OMN-14843's signature survives the fix, with the last state reported.

    A row parked mid-FSM for the whole window is the condition the STRANDED
    status was written for. The poll must not turn it into a pass, and must
    not report it as absent either.
    """
    connection = _install(
        monkeypatch,
        _ScriptedConnection(rows=[], tail=_non_terminal("INFERENCE_COMPLETED")),
    )

    outcome = await _readback_projection_via_asyncpg(_PROJECTION_DSN, str(uuid4()), 0.3)

    assert outcome.status is EnumProjectionReadbackStatus.STRANDED
    assert outcome.state == "INFERENCE_COMPLETED"
    assert "polled for" in outcome.error
    assert connection.data_reads > 1


# -- AC4: the error branch stays distinct from absence ---------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_permission_error_stays_distinct_from_absence(
    monkeypatch: pytest.MonkeyPatch, fast_poll: None
) -> None:
    """A denied SELECT is ERROR, never ROW_ABSENT, and never retried into one.

    This is the branch that keeps the live `chain_canary_reader` grant gap
    (OMN-18872 AC5) diagnosable: the role holds no SELECT on
    `delegation_workflow_state`, and if a refusal were ever folded into the
    absence arm the canary would blame the projection for a permission
    problem. The poll must surface it on the first read rather than spending
    the window re-asking a question already answered.
    """
    connection = _install(
        monkeypatch,
        _ScriptedConnection(
            rows=[],
            data_raises=PermissionError(
                "permission denied for table delegation_workflow_state"
            ),
        ),
    )

    outcome = await _readback_projection_via_asyncpg(_PROJECTION_DSN, str(uuid4()), 5.0)

    assert outcome.status is EnumProjectionReadbackStatus.ERROR
    assert outcome.status is not EnumProjectionReadbackStatus.ROW_ABSENT
    assert "permission denied" in outcome.error
    assert connection.data_reads == 1
    assert connection.closed is True


# -- AC2: the knob is declared, and there is only one ----------------------


@pytest.mark.unit
def test_shipped_poll_interval_is_sane() -> None:
    """The default cannot be a busy loop, and cannot be longer than the window.

    Asserted separately because every other test in this module speeds the
    interval up, so without this one a zero could ship green.
    """
    interval = handler_module._PROJECTION_POLL_INTERVAL_SECONDS
    assert 0.1 <= interval <= 5.0


@pytest.mark.unit
def test_no_second_deadline_knob_was_added_to_the_request() -> None:
    """Link 2 is bounded by the readback window, not by a knob of its own.

    A separate projection deadline would let a run narrow the poll without
    narrowing the window it claims to have waited — a run could report "no row
    in 120s" having looked for two. The window every other leg is measured
    against is the only deadline.
    """
    from omnibase_infra.nodes.node_chain_canary_effect.models.model_chain_canary_request import (
        ModelChainCanaryRequest,
    )

    offending = [
        name
        for name in ModelChainCanaryRequest.model_fields
        if "projection" in name and ("timeout" in name or "poll" in name)
    ]
    assert offending == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_privileged_dsn_is_refused_before_any_data_read(
    monkeypatch: pytest.MonkeyPatch, fast_poll: None
) -> None:
    """The OMN-18060 identity gate still runs first, and the poll never starts.

    Regression guard for the fix itself: wrapping the read in a loop must not
    move it ahead of the refusal, or a privileged DSN would read the row
    repeatedly before being declined.
    """

    class _PrivilegedConnection(_ScriptedConnection):
        async def fetchrow(self, query: str, *args: object) -> object:
            if "pg_roles" in query:
                return {
                    "rolname": "postgres",
                    "rolsuper": True,
                    "rolbypassrls": True,
                }
            return await super().fetchrow(query, *args)

    connection = _PrivilegedConnection(rows=[], tail=_completed())
    _install(monkeypatch, connection)

    outcome = await _readback_projection_via_asyncpg(_PROJECTION_DSN, str(uuid4()), 5.0)

    assert outcome.status is EnumProjectionReadbackStatus.REFUSED
    assert connection.data_reads == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_poll_is_cancelled_by_the_shared_deadline_not_its_own(
    monkeypatch: pytest.MonkeyPatch, fast_poll: None
) -> None:
    """The leg still honours ONE deadline across connect, probe and every poll.

    The OMN-18060 property this must not break: the leg cannot hold
    ``asyncio.gather()`` past the window it was budgeted, whatever the poll
    interval is.
    """
    window_s = 0.3
    _install(monkeypatch, _ScriptedConnection(rows=[], tail=None))

    started = time.monotonic()
    outcome = await _readback_projection_via_asyncpg(
        _PROJECTION_DSN, str(uuid4()), window_s
    )
    elapsed = time.monotonic() - started

    assert outcome.status is EnumProjectionReadbackStatus.ROW_ABSENT
    assert elapsed < window_s * 3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_concurrent_legs_are_not_blocked_by_the_poll(
    monkeypatch: pytest.MonkeyPatch, fast_poll: None
) -> None:
    """The poll yields, so the sibling gather legs keep running.

    A poll implemented with a blocking sleep would stall the terminal bus read
    it runs beside, turning a link-2 fix into a link-4 regression.
    """
    _install(monkeypatch, _ScriptedConnection(rows=[], tail=None))
    ticks = 0

    async def _sibling() -> None:
        nonlocal ticks
        for _ in range(5):
            await asyncio.sleep(_FAST_POLL_SECONDS)
            ticks += 1

    outcome, _ = await asyncio.gather(
        _readback_projection_via_asyncpg(_PROJECTION_DSN, str(uuid4()), 0.3),
        _sibling(),
    )

    assert outcome.status is EnumProjectionReadbackStatus.ROW_ABSENT
    assert ticks == 5
