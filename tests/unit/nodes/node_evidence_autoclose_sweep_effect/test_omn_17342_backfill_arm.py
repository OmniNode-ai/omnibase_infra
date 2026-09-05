# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17342 — the sweep's enumeration is forward-only, so most of the board is never looked at.

``handle()`` computed one window, ``now - lookback_hours``, and called
``_fetch_merged_companions`` once. No cursor, no watermark, no ticket-side arm.
A companion that merges, sits inside ~12 consecutive ``*/30`` windows, reaches
no verdict in any of them, and then ages out is never re-examined — not
"correctly withheld", never *looked at*. Measured 2026-09-05: 118 of 238 open
sprint tickets carry a merged OCC companion and only 5 of those merged inside
the live 6h window; 113 are permanently out of reach, 74 of them carrying the
behaviour-proof receipt that is the hardest flip conjunct to satisfy.

This suite pins the second arm and, just as importantly, its bounds:

* AC1 — a companion merged OUTSIDE ``lookback_hours`` reaches a real decision.
* AC2 — the candidate set is bounded per run and does not grow with the board.
* AC3 — no per-PR ``gh api .../files`` call for a candidate the Linear state
  short-circuit discards, asserted by call count rather than by inspection.
* AC4 — every flip conjunct is unchanged: a backfilled candidate proved only by
  merge state is still a gap, and a title/file binding disagreement on an open
  ticket still refuses rather than guessing.

The arm is OFF unless asked for, and that is pinned too: an omitted
``backfill_lookback_hours`` must reproduce the single-arm run exactly.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    HandlerEvidenceAutocloseSweep,
    _rotating_slice,
    _rotation_tick,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_evidence_autoclose_decision import (
    EnumEvidenceAutocloseDecision,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_evidence_autoclose_sweep_request import (
    ModelEvidenceAutocloseSweepRequest,
)

pytestmark = pytest.mark.unit

_OCC_REPO = "OmniNode-ai/onex_change_control"
_DOD_VERIFY_STATE_MODEL = (
    "omnimarket.nodes.node_dod_verify.models.model_dod_verify_state.ModelDodVerifyState"
)


def _iso(hours_ago: float) -> str:
    return (datetime.now(tz=UTC) - timedelta(hours=hours_ago)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _pr(number: int, ticket: str, hours_ago: float) -> dict[str, object]:
    stamp = _iso(hours_ago)
    return {
        "number": number,
        "html_url": f"https://github.com/{_OCC_REPO}/pull/{number}",
        "title": f"evidence({ticket}): OCC companion",
        "updated_at": stamp,
        "merged_at": stamp,
    }


def _verdict(
    *, behavior_proving_count: int, ticket: str = "OMN-0000"
) -> dict[str, object]:
    """A receipt-arm ModelSkillResult shaped exactly as `onex skill dod_verify` prints."""
    checks = [
        {
            "evidence_id": "dod-pr-state",
            "description": "merge state",
            "status": "verified",
            "message": "OK (1ms)",
            "proof_class": "merge-state",
        }
    ]
    if behavior_proving_count > 0:
        checks.append(
            {
                "evidence_id": "dod-tests",
                "description": "behaviour",
                "status": "verified",
                "message": "OK (1ms)",
                "proof_class": "behavior",
            }
        )
    return {
        "skill_name": "dod_verify",
        "node_name": "node_dod_verify",
        "status": "success",
        "exit_code": 0,
        "result": {
            "correlation_id": str(uuid4()),
            "ticket_id": ticket,
            "status": "verified",
            "dry_run": False,
            "checks": checks,
            "total_checks": len(checks),
            "verified_count": len(checks),
            "failed_count": 0,
            "skipped_count": 0,
            "superseded_count": 0,
            "behavior_proving_count": behavior_proving_count,
            "error_message": None,
        },
        "result_model": _DOD_VERIFY_STATE_MODEL,
    }


class _FakeLinear:
    """Linear double keyed per ticket, recording every read and every write."""

    def __init__(self, terminal: set[str] | None = None) -> None:
        self.terminal = terminal or set()
        self.state_updates: list[tuple[str, str]] = []
        self.comments: list[tuple[str, str]] = []
        self.issues_fetched: list[str] = []

    async def fetch_issue(self, ticket_id: str) -> dict[str, object]:
        self.issues_fetched.append(ticket_id)
        done = ticket_id in self.terminal
        return {
            "id": f"issue-{ticket_id}",
            "identifier": ticket_id,
            "state": (
                {"id": "s-done", "name": "Done", "type": "completed"}
                if done
                else {"id": "s1", "name": "In Progress", "type": "started"}
            ),
            "labels": {"nodes": []},
            "team": {"id": "team-1"},
            "description": None,
        }

    async def fetch_done_state_id(self, team_id: str) -> str:
        return "state-done-id"

    async def update_issue_state(self, issue_id: str, state_id: str) -> bool:
        self.state_updates.append((issue_id, state_id))
        return True

    async def create_comment(self, issue_id: str, body: str) -> bool:
        self.comments.append((issue_id, body))
        return True

    async def fetch_comment_bodies(self, issue_id: str) -> tuple[str, ...] | None:
        return tuple(body for target, body in self.comments if target == issue_id)


class _Calls:
    def __init__(self) -> None:
        self.file_fetches: list[int] = []
        self.dod_verify: list[str] = []


def _handler(
    pool: list[dict[str, object]],
    linear: _FakeLinear,
    calls: _Calls,
    *,
    behavior_proving_count: int = 1,
    files_for: dict[int, list[str]] | None = None,
) -> HandlerEvidenceAutocloseSweep:
    """Handler wired to a fixed OCC page-1 pool.

    ``_fetch_merged_companions`` filters the pool by ``merged_at >= since_iso``
    itself, so the SAME double serves both enumeration arms — which is the
    point: the arms differ only in the window they ask for, never in what the
    transport returns.
    """
    by_number = {int(str(pr["number"])): pr for pr in pool}

    async def fake_gh(args: list[str], timeout: float):
        path = args[2]
        if "/files" in path:
            number = int(path.rsplit("/pulls/", 1)[1].split("/", 1)[0])
            calls.file_fetches.append(number)
            if files_for is not None and number in files_for:
                return [{"filename": name} for name in files_for[number]], ""
            title = str(by_number[number]["title"])
            ticket = title.split("(", 1)[1].split(")", 1)[0]
            return [{"filename": f"contracts/{ticket}.yaml"}], ""
        page = int(path.rsplit("page=", 1)[1])
        return (list(pool), "") if page == 1 else ([], "")

    async def fake_dod_verify(ticket_id: str, cwd: str, timeout: float):
        calls.dod_verify.append(ticket_id)
        return (
            _verdict(behavior_proving_count=behavior_proving_count, ticket=ticket_id),
            0,
            "",
        )

    return HandlerEvidenceAutocloseSweep(
        linear_client=linear,  # type: ignore[arg-type]
        autoclose_disabled=False,
        run_gh_command=fake_gh,
        run_dod_verify_command=fake_dod_verify,
    )


def _request(**overrides: object) -> ModelEvidenceAutocloseSweepRequest:
    defaults: dict[str, object] = {
        "correlation_id": uuid4(),
        "occ_repo": _OCC_REPO,
        "lookback_hours": 6,
        "apply": False,
    }
    defaults.update(overrides)
    return ModelEvidenceAutocloseSweepRequest(**defaults)


def _tickets(result) -> set[str]:
    return {o.ticket_id for o in result.outcomes if o.ticket_id}


# -- AC1 -----------------------------------------------------------------


async def test_a_companion_merged_outside_the_lookback_reaches_a_decision() -> None:
    """The blocker, executed.

    OMN-16961's own companion merged 2026-09-04T08:15:44Z with a behaviour-proof
    receipt on the ticket and the ticket still open — 18h old against a 6h
    window, so on the single-arm enumeration it is not scanned, not skipped, and
    not refused. It is absent, which reads on the board exactly like a ticket
    with no evidence at all.
    """
    fresh = _pr(8300, "OMN-17872", hours_ago=1)
    stale = _pr(8179, "OMN-16961", hours_ago=18)
    linear = _FakeLinear()
    calls = _Calls()
    handler = _handler([fresh, stale], linear, calls)

    single_arm = await handler.handle(_request())
    assert _tickets(single_arm) == {"OMN-17872"}, (
        "positive control for the zero below: the forward arm does see the "
        "fresh companion, so an absent OMN-16961 is the window and not the fake"
    )
    assert "OMN-16961" not in _tickets(single_arm)

    backfilled = await handler.handle(_request(backfill_lookback_hours=168))

    assert "OMN-16961" in _tickets(backfilled)
    outcome = next(o for o in backfilled.outcomes if o.ticket_id == "OMN-16961")
    assert outcome.decision is not EnumEvidenceAutocloseDecision.SKIPPED_NO_BINDING
    assert outcome.enumeration_arm == "backfill"
    assert outcome.companion_pr_number == 8179


async def test_the_backfill_arm_is_off_unless_asked_for() -> None:
    """Omitting the field reproduces the single-arm run exactly — zero extra I/O."""
    fresh = _pr(8300, "OMN-17872", hours_ago=1)
    stale = _pr(8179, "OMN-16961", hours_ago=18)
    linear = _FakeLinear()
    calls = _Calls()
    handler = _handler([fresh, stale], linear, calls)

    result = await handler.handle(_request())

    assert _tickets(result) == {"OMN-17872"}
    assert calls.dod_verify == ["OMN-17872"]
    assert linear.issues_fetched == ["OMN-17872"]
    assert result.backfill_candidates_selected == 0


# -- AC2 -----------------------------------------------------------------


async def test_the_candidate_set_is_bounded_per_run_and_does_not_grow_with_the_board() -> (
    None
):
    """The bound is the whole point: `dod_verify` dominates the run budget.

    ~15s/ticket under sweep concurrency (OMN-16961's measurement) against a
    30-minute cadence. An arm that verified its whole candidate set every tick
    would overrun the cadence the moment the backlog grew, which is precisely
    the population this arm exists to drain.
    """
    small = [_pr(7000 + i, f"OMN-1{i:04d}", hours_ago=40 + i) for i in range(25)]
    large = [_pr(7000 + i, f"OMN-1{i:04d}", hours_ago=40 + i) for i in range(60)]

    for pool in (small, large):
        linear = _FakeLinear()
        calls = _Calls()
        handler = _handler(pool, linear, calls)

        result = await handler.handle(
            _request(backfill_lookback_hours=336, backfill_max_candidates=5)
        )

        assert len(calls.dod_verify) == 5, (
            f"pool of {len(pool)} produced {len(calls.dod_verify)} verifier runs — "
            "the bound must not track the size of the board"
        )
        assert result.backfill_candidates_selected == 5
        assert result.backfill_pool_size == len(pool)


def test_the_rotating_slice_advances_and_eventually_covers_the_whole_pool() -> None:
    """Bounded per tick, exhaustive across ticks — otherwise the tail never drains."""
    pool = list(range(23))

    first = _rotating_slice(pool, 5, 0)
    second = _rotating_slice(pool, 5, 1)
    assert first == [0, 1, 2, 3, 4]
    assert second == [5, 6, 7, 8, 9]
    assert not set(first) & set(second)

    covered: set[int] = set()
    for tick in range(5):
        covered |= set(_rotating_slice(pool, 5, tick))
    assert covered == set(pool), "five ticks of five must cover a pool of 23"

    # Wrap is a rotation, not a truncation: the slice stays full width.
    assert len(_rotating_slice(pool, 5, 4)) == 5
    assert _rotating_slice(pool, 5, 4) == [20, 21, 22, 0, 1]

    # A pool no larger than the slice is taken whole, with no wrap duplication.
    assert _rotating_slice([1, 2], 5, 7) == [1, 2]
    assert _rotating_slice([], 5, 3) == []


def test_the_rotation_tick_advances_once_per_cadence_period() -> None:
    """Derived from the run's own clock, so consecutive scheduled runs differ."""
    base = datetime(2026, 9, 5, 0, 0, 0, tzinfo=UTC)
    assert _rotation_tick(base, 30) == _rotation_tick(base + timedelta(minutes=29), 30)
    assert _rotation_tick(base + timedelta(minutes=30), 30) == (
        _rotation_tick(base, 30) + 1
    )
    assert _rotation_tick(base + timedelta(minutes=60), 30) == (
        _rotation_tick(base, 30) + 2
    )


# -- AC3 -----------------------------------------------------------------


async def test_no_pr_files_fetch_for_a_candidate_the_state_short_circuit_discards() -> (
    None
):
    """The cheap filter has to run before the expensive one.

    ``handle()`` used to call ``_fetch_pr_files`` for EVERY companion before
    anything could discard an already-completed ticket, so widening the window
    multiplied `gh api` calls immediately. A title binding is sufficient to look
    up state; the file listing is only needed to disambiguate or confirm before
    a flip.
    """
    open_pr = _pr(8300, "OMN-17872", hours_ago=1)
    done_pr = _pr(8301, "OMN-15060", hours_ago=1)
    linear = _FakeLinear(terminal={"OMN-15060"})
    calls = _Calls()
    handler = _handler([open_pr, done_pr], linear, calls)

    result = await handler.handle(_request())

    assert calls.file_fetches == [8300], (
        "the already-completed ticket must cost zero file listings; the open "
        "one is the positive control proving the fetch still happens"
    )
    done = next(o for o in result.outcomes if o.ticket_id == "OMN-15060")
    assert done.decision is EnumEvidenceAutocloseDecision.SKIPPED_ALREADY_DONE
    assert calls.dod_verify == ["OMN-17872"]


async def test_an_excluded_candidate_costs_no_files_fetch_and_no_linear_read() -> None:
    """The OMN-17891 fence moves ahead of the files fetch too.

    The fence's whole claim is that a refusal costs zero I/O and is therefore
    terminal. That was true of the Linear read and false of the `gh api` file
    listing, which ran first.
    """
    fenced = _pr(8300, "OMN-17857", hours_ago=1)
    ordinary = _pr(8301, "OMN-17872", hours_ago=1)
    linear = _FakeLinear()
    calls = _Calls()
    handler = _handler([fenced, ordinary], linear, calls)

    result = await handler.handle(_request(exclude_tickets=("omn-17857 ",)))

    assert calls.file_fetches == [8301]
    assert linear.issues_fetched == ["OMN-17872"]
    fence = next(o for o in result.outcomes if o.ticket_id == "OMN-17857")
    assert fence.decision is EnumEvidenceAutocloseDecision.SKIPPED_EXCLUDED


# -- AC4 -----------------------------------------------------------------


async def test_a_backfilled_candidate_proved_only_by_merge_state_is_still_a_gap() -> (
    None
):
    """Strictness is unchanged in every direction — the arm widens WHO is asked, never WHAT counts."""
    stale = _pr(8179, "OMN-16961", hours_ago=18)
    linear = _FakeLinear()
    calls = _Calls()
    handler = _handler([stale], linear, calls, behavior_proving_count=0)

    result = await handler.handle(_request(apply=True, backfill_lookback_hours=168))

    assert result.tickets_flipped == 0
    outcome = result.outcomes[0]
    assert outcome.decision is EnumEvidenceAutocloseDecision.GAP_NO_BEHAVIOR_PROOF
    assert linear.state_updates == []


async def test_a_backfilled_candidate_with_a_behaviour_proof_does_flip() -> None:
    """The negative control for the gap above: the arm can reach a flip, on the same predicate."""
    stale = _pr(8179, "OMN-16961", hours_ago=18)
    linear = _FakeLinear()
    calls = _Calls()
    handler = _handler([stale], linear, calls, behavior_proving_count=1)

    result = await handler.handle(_request(apply=True, backfill_lookback_hours=168))

    assert result.tickets_flipped == 1
    assert result.outcomes[0].decision is EnumEvidenceAutocloseDecision.FLIPPED
    assert result.outcomes[0].enumeration_arm == "backfill"
    assert linear.state_updates == [("issue-OMN-16961", "state-done-id")]


async def test_a_title_file_binding_disagreement_on_an_open_ticket_still_refuses() -> (
    None
):
    """The state short-circuit must not become a way past the ambiguity guard.

    It fires only on a TERMINAL state, where every path is a zero-write skip. On
    an open ticket the file listing is still fetched and a disagreement between
    the title and the contract file still refuses rather than guessing.
    """
    conflicted = _pr(8300, "OMN-17872", hours_ago=1)
    linear = _FakeLinear()
    calls = _Calls()
    handler = _handler(
        [conflicted],
        linear,
        calls,
        files_for={8300: ["contracts/OMN-15060.yaml"]},
    )

    result = await handler.handle(_request())

    assert calls.file_fetches == [8300]
    assert calls.dod_verify == []
    assert (
        result.outcomes[0].decision
        is EnumEvidenceAutocloseDecision.SKIPPED_AMBIGUOUS_BINDING
    )


async def test_the_same_ticket_bound_by_both_arms_is_processed_once() -> None:
    """A companion inside the forward window is not re-verified by the backfill pool."""
    fresh = _pr(8300, "OMN-17872", hours_ago=1)
    older_for_same = _pr(8100, "OMN-17872", hours_ago=50)
    linear = _FakeLinear()
    calls = _Calls()
    handler = _handler([fresh, older_for_same], linear, calls)

    result = await handler.handle(_request(backfill_lookback_hours=168))

    assert calls.dod_verify == ["OMN-17872"]
    assert [o.ticket_id for o in result.outcomes] == ["OMN-17872"]
    assert result.outcomes[0].enumeration_arm == "forward"
