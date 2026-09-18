# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the rule-4 cost-sentence window resolution in scripts/ledger_lock.py
(OMN-18554, closing the inert-gate defect left by OMN-15649's window source;
re-sourced onto the goal file by OMN-18751).

Context (fact, measured 2026-09-17): ``DEFAULT_PLAN_PATH`` pointed at the
rolling seven-day plan under ``docs/plans/`` in this repo. That file was
deleted outright by omni_home#341 (``17bbbac6``) when the plans corpus
migrated to knowledge-base-internal, and no successor was created at that
path. So ``read_declared_window()`` returned ``None`` on every invocation, and
``resolve_enforcement_window()`` treated "no window could be resolved" as
identical to "we are outside an open window" -- it printed an INERT warning
and returned False, and every CLAIM row on the fleet's primary coordination
surface landed with zero rule-4 enforcement. 143 of the 146 CLAIM rows in the
24h before the ticket was filed carry no cost sentence at all.

Second context (OMN-18751, operator ruling 2026-09-18, verbatim "retire it").
OMN-18554 re-pointed the gate at the same hand-maintained plan in the
knowledge-base-internal clone, where it promptly went 43 days stale and put a
staleness banner on every append the fleet wrote. That plan is now RETIRED
rather than re-cut, and the freshness source is ``beta/GOAL.md``'s
``state_as_of:`` stamp, which the morning ground-state workflow rewrites. The
window is ``(state_as_of, state_as_of + FRESHNESS_HORIZON_DAYS)``.

These tests pin five separable behaviours:

1. The freshness source is ``beta/GOAL.md`` in the knowledge-base-internal
   clone, fail-fast on an unset ``KNOWLEDGE_BASE_INTERNAL_PATH`` (rule 8: no
   silent default, and in particular no omni_home fallback -- the omni_home
   path is the one that evaporated), and the retired plan is named nowhere in
   the script.
2. A goal file re-measured longer ago than the horizon is a loud,
   still-enforcing state, not an inert one, and the banner asks for a
   re-measurement rather than a re-cut of a document that no longer exists.
   The rule-4 requirement is that a claim row carries its price; the goal file
   supplies the denominator that price is read against, and a stale
   denominator is a reason to shout, never a reason to stop asking.
3. A goal file that cannot be read at all refuses CLAIM rows with a named
   cause, instead of accepting them silently. The explicit
   ``LEDGER_LOCK_ALLOW_INERT`` override is the only way to land a claim row
   with the gate unresolved, and taking it is visible on stderr.
4. "Unresolvable" and "outside an open window" are distinguishable outcomes
   (OMN-18554 AC4), which is exactly what the pre-fix code conflated.
5. The horizon is seven days, pinned at its own boundary rather than only
   well inside and well outside it.

Non-claim rows (TERMINAL, NOTE, RULING, PROGRESS) are never touched by any of
this -- they were not touched before and a gate that blocked them would take
the fleet's whole coordination surface down rather than price it.

The subject module is untracked local tooling (omni_home gitignores scripts/
wholesale and the no-functional-code hook forbids committing .py outside docs/
and tests/). On a machine without the script this module skips loudly rather
than passing vacuously -- same convention as tests/test_ledger_lock_clock_guard.py.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "ledger_lock.py"

assert SCRIPT.is_file(), (
    f"scripts/ledger_lock.py must exist at {SCRIPT}. In omni_home this module skipped "
    "when the script was absent, because there it was untracked local tooling. Here the "
    "script is COMMITTED (OMN-18554), so absence is a failure, never a skip."
)

_spec = importlib.util.spec_from_file_location(
    "ledger_lock_cost_sentence_infra", SCRIPT
)
assert _spec is not None and _spec.loader is not None
ll = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ll
_spec.loader.exec_module(ll)


# --------------------------------------------------------------------------
# Fixtures: a ledger, and goal files in each of the four window states
# --------------------------------------------------------------------------


def _stamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def claim_row(*, priced: bool, lane: str = "omn18554-test") -> str:
    """A CLAIM row in a shape ``is_claim_row`` actually recognizes (the
    bracketed-handle family, CLAIM_SESSION_PATTERN), stamped from the real
    clock so the OMN-17427 clock guard -- which runs ahead of this gate and is
    not window-scoped -- passes and leaves the cost-sentence gate as the only
    thing under test.

    Deliberately NOT the dominant live ``<ts> | CLAIM | lane=...`` pipe shape:
    that shape is invisible to the matcher today, which is a separate defect
    from the one under test here and is pinned as a measured residual by
    test_dominant_live_pipe_claim_shape_is_not_recognized_residual below. A
    fixture in the unrecognized shape would make every assertion in this file
    vacuous.
    """
    head = (
        f"- {_stamp(datetime.now(UTC))} [{lane}] CLAIM OMN-18554 — "
        "exercise the rule-4 window gate"
    )
    if not priced:
        return head
    return head + "; est ~2 lane-hours; displaces nothing; (OMN-18554)"


def terminal_row(lane: str = "omn18554-test") -> str:
    return (
        f"{_stamp(datetime.now(UTC))} | TERMINAL | lane={lane} | "
        "actor=test | RESULT: nothing mutated | friction=none"
    )


@pytest.fixture
def ledger(tmp_path: Path) -> Path:
    path = tmp_path / "ROLLING_WORK_LEDGER.md"
    path.write_text("## §5 ACTION LOG (append-only)\n", encoding="utf-8")
    return path


def _goal(tmp_path: Path, state_as_of: date, name: str) -> Path:
    """A goal file in the real shape: ``state_as_of:`` on its own first line,
    with the trailing parenthetical the live file carries, so the parser is
    tested against the format it will actually meet rather than a bare date."""
    path = tmp_path / name
    path.write_text(
        f"state_as_of: {state_as_of.isoformat()} (fixture, written by the test suite)\n"
        "L1 | a row | R2-UNMET | ticket:OMN-18554 | fixture row\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def open_goal(tmp_path: Path) -> Path:
    """Re-measured two days ago: inside the seven-day freshness horizon."""
    today = datetime.now(UTC).date()
    return _goal(tmp_path, today - timedelta(days=2), "open-goal.md")


@pytest.fixture
def stale_goal(tmp_path: Path) -> Path:
    """Re-measured 48 days ago -- the real condition this replaces, where the
    retired rolling plan's window had lapsed 43 days and every append said so."""
    today = datetime.now(UTC).date()
    return _goal(tmp_path, today - timedelta(days=48), "stale-goal.md")


@pytest.fixture
def pending_goal(tmp_path: Path) -> Path:
    """A stamp dated in the FUTURE. The file describes a state that has not
    happened, so it is not yet in force and the row lands unenforced -- the
    state that keeps "not current" observably different from "unresolvable"
    (OMN-18554 AC4)."""
    today = datetime.now(UTC).date()
    return _goal(tmp_path, today + timedelta(days=10), "pending-goal.md")


def _append(ledger: Path, line: str) -> int:
    return ll.main([str(ledger), "--append", line])


# --------------------------------------------------------------------------
# 1. Window source: beta/GOAL.md in knowledge-base-internal, fail-fast
# --------------------------------------------------------------------------


def test_window_source_resolves_to_the_goal_file_in_the_clone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OMN-18751 AC1. RED before this change: the resolver returned
    ``beta/plans/ROLLING_SEVEN_DAY_PLAN.md``, a hand-maintained document that
    was 43 days stale when the operator retired it."""
    monkeypatch.delenv(ll.GOAL_PATH_ENV, raising=False)
    monkeypatch.setenv(ll.KB_INTERNAL_ROOT_ENV, str(tmp_path))
    assert ll.goal_path_for_window() == tmp_path / "beta" / "GOAL.md"


def test_the_retired_rolling_plan_is_named_nowhere_in_the_script(
    tmp_path: Path,
) -> None:
    """OMN-18751 AC1, second half. A dormant mention is how a gate gets
    re-pointed at an unmaintained document the next time someone reaches for a
    default, which is the OMN-18554 defect exactly.

    The positive control is the point of the second half: an empty grep and a
    broken grep are the same two characters of output (CLAUDE.md rule 16), so
    the same search is run against a fixture that DOES carry the name.
    """
    retired = "ROLLING_SEVEN_DAY_PLAN"
    source = SCRIPT.read_text(encoding="utf-8")
    assert retired not in source, (
        f"{SCRIPT} still names the retired rolling seven-day plan. It was retired by "
        "the operator on 2026-09-18 (OMN-18751); the window source is beta/GOAL.md."
    )
    control = tmp_path / "control.md"
    control.write_text(f"beta/plans/{retired}.md\n", encoding="utf-8")
    assert retired in control.read_text(encoding="utf-8"), (
        "positive control failed: the search itself is broken, so the assertion "
        "above proved nothing"
    )


def test_explicit_goal_path_override_still_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ll.KB_INTERNAL_ROOT_ENV, str(tmp_path))
    monkeypatch.setenv(ll.GOAL_PATH_ENV, str(tmp_path / "elsewhere.md"))
    assert ll.goal_path_for_window() == tmp_path / "elsewhere.md"


def test_unset_clone_root_fails_fast_and_never_falls_back_to_omni_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rule 8: an unset required env var raises, naming itself. The failure
    mode this replaces is a silent default that resolves to a path nobody
    maintains -- which is how the gate went inert in the first place."""
    monkeypatch.delenv(ll.GOAL_PATH_ENV, raising=False)
    monkeypatch.delenv(ll.KB_INTERNAL_ROOT_ENV, raising=False)
    with pytest.raises(ll.GoalSourceUnresolvedError) as excinfo:
        ll.goal_path_for_window()
    message = str(excinfo.value)
    assert ll.KB_INTERNAL_ROOT_ENV in message
    assert "omni_home" not in message.lower().replace("omni_home/docs", ""), (
        "the refusal must not advertise an omni_home fallback path"
    )


def test_no_omni_home_default_path_constant_survives() -> None:
    """The constant that pointed at the deleted file is gone, not merely
    unreferenced -- a dormant default is what silently re-points the gate at a
    path nobody maintains the next time someone reaches for one."""
    assert not hasattr(ll, "DEFAULT_PLAN_PATH")
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'OMNI_HOME / "docs" / "plans"' not in source


# --------------------------------------------------------------------------
# 2. Window states, and the causes an unresolved one reports
# --------------------------------------------------------------------------


def test_window_states_are_four_way_not_boolean(
    open_goal: Path,
    stale_goal: Path,
    pending_goal: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(UTC)
    for plan, expected in (
        (open_goal, ll.WINDOW_OPEN),
        (stale_goal, ll.WINDOW_STALE),
        (pending_goal, ll.WINDOW_PENDING),
        (tmp_path / "does-not-exist.md", ll.WINDOW_UNRESOLVED),
    ):
        monkeypatch.setenv(ll.GOAL_PATH_ENV, str(plan))
        assert ll.resolve_window_state(now).state == expected, plan


@pytest.mark.parametrize(
    ("body", "expected_fragment"),
    [
        (None, "file missing"),
        ("# Goal\n\nno front matter here\n", "no 'state_as_of:' line"),
        ("state_as_of: 2026-13-45\n", "malformed date"),
        ("state_as_of: 2026-02-30\n", "malformed date"),
    ],
)
def test_unresolved_window_reports_a_distinguishing_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    body: str | None,
    expected_fragment: str,
) -> None:
    """All four unresolvable causes collapsed to a bare None before the fix,
    so a refusal could not say which one it hit."""
    plan = tmp_path / "plan.md"
    if body is not None:
        plan.write_text(body, encoding="utf-8")
    monkeypatch.setenv(ll.GOAL_PATH_ENV, str(plan))
    resolution = ll.resolve_window_state(datetime.now(UTC))
    assert resolution.state == ll.WINDOW_UNRESOLVED
    assert resolution.cause is not None
    assert expected_fragment in resolution.cause


def test_unset_clone_root_is_a_no_registry_window_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ll.GOAL_PATH_ENV, raising=False)
    monkeypatch.delenv(ll.KB_INTERNAL_ROOT_ENV, raising=False)
    resolution = ll.resolve_window_state(datetime.now(UTC))
    assert resolution.state == ll.WINDOW_NO_REGISTRY
    assert resolution.cause is not None and ll.KB_INTERNAL_ROOT_ENV in resolution.cause


def test_an_environment_with_no_registry_lands_claim_rows_and_says_so(
    ledger: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A CI runner has no knowledge-base-internal clone. Refusing there prices
    nothing -- the rows this gate exists to price are written on machines that DO
    have the clone -- and would only stop CI appending at all. Caught by CI itself:
    the first push of this port failed six claim-token tests for exactly this
    reason, on a runner, where a developer machine with the variable exported saw
    nothing wrong."""
    monkeypatch.delenv(ll.GOAL_PATH_ENV, raising=False)
    monkeypatch.delenv(ll.KB_INTERNAL_ROOT_ENV, raising=False)
    monkeypatch.delenv(ll.ALLOW_INERT_ENV, raising=False)
    assert ll.main([str(ledger), "--append", claim_row(priced=False)]) == 0
    err = capsys.readouterr().err
    assert "cannot run in this environment" in err
    assert ll.KB_INTERNAL_ROOT_ENV in err, "the announcement must name what is missing"


def test_a_pointed_goal_file_that_does_not_resolve_still_refuses_without_a_registry(
    ledger: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The split that keeps the no-registry state from becoming a loophole:
    pointing LEDGER_LOCK_GOAL_PATH at a goal file that does not resolve is a
    MISCONFIGURED lane machine and still refuses, registry or no registry."""
    monkeypatch.delenv(ll.KB_INTERNAL_ROOT_ENV, raising=False)
    monkeypatch.delenv(ll.ALLOW_INERT_ENV, raising=False)
    monkeypatch.setenv(ll.GOAL_PATH_ENV, str(tmp_path / "no-such-goal.md"))
    assert ll.main([str(ledger), "--append", claim_row(priced=False)]) == 65


# --------------------------------------------------------------------------
# 3. A stale window still enforces, loudly
# --------------------------------------------------------------------------


def test_stale_goal_refuses_an_unpriced_claim_row_and_names_its_age(
    ledger: Path,
    stale_goal: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """RED before OMN-18554: a lapsed window returned False from
    resolve_enforcement_window and the row landed unpriced and unremarked.
    OMN-18751 AC2: the banner now names the goal file's own staleness, in
    days, so the remedy it asks for is a re-measurement and not a re-cut of a
    document that no longer exists."""
    monkeypatch.setenv(ll.GOAL_PATH_ENV, str(stale_goal))
    before = ledger.read_text(encoding="utf-8")
    rc = _append(ledger, claim_row(priced=False))
    err = capsys.readouterr().err
    assert rc == 65
    assert ledger.read_text(encoding="utf-8") == before, (
        "a refused claim row must not land"
    )
    assert "STALE GOAL" in err
    start, _end = ll.read_declared_window(stale_goal)  # type: ignore[misc]
    assert start.isoformat() in err, "the banner must name the stamp it read"
    assert "48 day(s)" in err, (
        "the banner must name the goal file's age in days, so a reader can tell "
        "a two-day lapse from a six-week one"
    )


def test_a_fresh_goal_enforces_silently(
    ledger: Path,
    open_goal: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """OMN-18751 AC2, the other half. A freshness banner that fires while the
    goal file IS fresh is noise on every append, which is how a real signal
    stops being read."""
    monkeypatch.setenv(ll.GOAL_PATH_ENV, str(open_goal))
    assert _append(ledger, claim_row(priced=True)) == 0
    err = capsys.readouterr().err
    assert "STALE GOAL" not in err
    assert err.strip() == "" or "CLAIM-TOKEN" in err, (
        f"a fresh goal file must produce no freshness chatter; got {err!r}"
    )


@pytest.mark.parametrize(
    ("age_days", "expected"),
    [
        (0, "open"),
        (7, "open"),
        (8, "stale"),
    ],
)
def test_the_freshness_horizon_is_seven_days_inclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, age_days: int, expected: str
) -> None:
    """OMN-18751 AC2: the boundary itself, pinned. Seven days old is still
    fresh; the eighth day is stale. A horizon nobody tested at its edge is a
    horizon that drifts by one on the next refactor."""
    today = datetime.now(UTC).date()
    goal = _goal(tmp_path, today - timedelta(days=age_days), f"age-{age_days}.md")
    monkeypatch.setenv(ll.GOAL_PATH_ENV, str(goal))
    state = ll.resolve_window_state(datetime.now(UTC)).state
    assert state == (ll.WINDOW_OPEN if expected == "open" else ll.WINDOW_STALE)
    assert ll.FRESHNESS_HORIZON_DAYS == 7


def test_stale_goal_lands_a_priced_claim_row(
    ledger: Path,
    stale_goal: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Enforcement, not blockade: the priced row is exactly what the gate
    wants, and a stale goal file must not refuse it."""
    monkeypatch.setenv(ll.GOAL_PATH_ENV, str(stale_goal))
    rc = _append(ledger, claim_row(priced=True))
    assert rc == 0
    assert "] CLAIM OMN-18554" in ledger.read_text(encoding="utf-8")
    assert "STALE GOAL" in capsys.readouterr().err, (
        "a stale goal file is announced even on success"
    )


def test_open_window_enforces_exactly_as_before(
    ledger: Path, open_goal: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ll.GOAL_PATH_ENV, str(open_goal))
    assert _append(ledger, claim_row(priced=False)) == 65
    assert _append(ledger, claim_row(priced=True)) == 0


def test_a_non_claim_payload_gets_no_window_announcement(
    ledger: Path,
    stale_goal: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The stale banner is scoped to payloads that actually carry a claim row.
    A banner on every TERMINAL / NOTE / PROGRESS append is noise about a gate
    that was never going to inspect the row, and noise on every write is how a
    real signal stops being read -- the rolling plan has been un-re-cut for
    weeks, so this fires on the whole fleet's writes otherwise."""
    monkeypatch.setenv(ll.GOAL_PATH_ENV, str(stale_goal))
    assert _append(ledger, terminal_row()) == 0
    assert "STALE GOAL" not in capsys.readouterr().err


# --------------------------------------------------------------------------
# 4. An unresolvable goal file refuses; AC4 distinguishability
# --------------------------------------------------------------------------


def test_unresolvable_goal_refuses_an_unpriced_claim_row_with_a_named_cause(
    ledger: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """OMN-18554 AC1 -- the headline defect. Before the fix this exited 0 and
    wrote the row."""
    missing = tmp_path / "no-such-goal.md"
    monkeypatch.setenv(ll.GOAL_PATH_ENV, str(missing))
    monkeypatch.delenv(ll.ALLOW_INERT_ENV, raising=False)
    before = ledger.read_text(encoding="utf-8")
    rc = _append(ledger, claim_row(priced=False))
    err = capsys.readouterr().err
    assert rc == 65
    assert ledger.read_text(encoding="utf-8") == before
    assert "cannot be enforced" in err
    assert str(missing) in err
    assert "file missing" in err


def test_unresolvable_goal_refuses_even_a_priced_claim_row(
    ledger: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal is about the gate being unable to run, not about the row.
    A gate that cannot read its own input has not passed; it has not run
    (CLAUDE.md rule 16). Accepting the well-formed subset would make the
    refusal a style check."""
    monkeypatch.setenv(ll.GOAL_PATH_ENV, str(tmp_path / "no-such-goal.md"))
    monkeypatch.delenv(ll.ALLOW_INERT_ENV, raising=False)
    assert _append(ledger, claim_row(priced=True)) == 65


def test_unresolvable_goal_never_blocks_a_non_claim_row(
    ledger: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TERMINAL / NOTE / RULING rows were never in this gate's scope and stay
    out of it -- the coordination surface must keep working while the window
    source is being repaired."""
    monkeypatch.setenv(ll.GOAL_PATH_ENV, str(tmp_path / "no-such-goal.md"))
    monkeypatch.delenv(ll.ALLOW_INERT_ENV, raising=False)
    assert _append(ledger, terminal_row()) == 0
    assert "| TERMINAL |" in ledger.read_text(encoding="utf-8")


def test_allow_inert_override_lands_the_row_and_says_so(
    ledger: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(ll.GOAL_PATH_ENV, str(tmp_path / "no-such-goal.md"))
    monkeypatch.setenv(ll.ALLOW_INERT_ENV, "1")
    rc = _append(ledger, claim_row(priced=False))
    err = capsys.readouterr().err
    assert rc == 0
    assert "] CLAIM OMN-18554" in ledger.read_text(encoding="utf-8")
    assert ll.ALLOW_INERT_ENV in err, "taking the override is never silent"


def test_pending_window_lands_an_unpriced_claim_row(
    ledger: Path, pending_goal: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OMN-18554 AC4: a window that RESOLVES but is not current still lands
    the row unenforced, which is what makes it distinguishable from an
    unresolvable one. (The lapsed half of "not current" enforces instead --
    see test_stale_window_refuses_...; the two are deliberately different and
    the divergence from AC4's parenthetical is recorded on the ticket.)"""
    monkeypatch.setenv(ll.GOAL_PATH_ENV, str(pending_goal))
    assert _append(ledger, claim_row(priced=False)) == 0
    assert "] CLAIM OMN-18554" in ledger.read_text(encoding="utf-8")


def test_unresolvable_and_out_of_window_are_not_the_same_outcome(
    ledger: Path, pending_goal: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC4's first clause, stated as one assertion: the two states the pre-fix
    code collapsed into a single `return False` now produce different rcs for
    the identical row."""
    monkeypatch.delenv(ll.ALLOW_INERT_ENV, raising=False)
    row = claim_row(priced=False)

    monkeypatch.setenv(ll.GOAL_PATH_ENV, str(pending_goal))
    out_of_window_rc = _append(ledger, row)

    monkeypatch.setenv(ll.GOAL_PATH_ENV, str(tmp_path / "no-such-goal.md"))
    unresolvable_rc = _append(ledger, row + " ")

    assert out_of_window_rc == 0
    assert unresolvable_rc == 65


# --------------------------------------------------------------------------
# 5. The `-- COMMAND` write verb enforces exactly what --append enforces
# --------------------------------------------------------------------------


def _write_via_command(ledger: Path, line: str) -> int:
    script = "import sys,pathlib;p=pathlib.Path(sys.argv[1]);p.write_text(p.read_text()+sys.argv[2]+'\\n')"
    return ll.main([str(ledger), "--", sys.executable, "-c", script, str(ledger), line])


def test_command_verb_reverts_an_unpriced_claim_row_when_the_window_is_unresolvable(
    ledger: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """There is no editor bypass: the same row refused by --append is refused
    and reverted here."""
    monkeypatch.setenv(ll.GOAL_PATH_ENV, str(tmp_path / "no-such-goal.md"))
    monkeypatch.delenv(ll.ALLOW_INERT_ENV, raising=False)
    before = ledger.read_text(encoding="utf-8")
    rc = _write_via_command(ledger, claim_row(priced=False))
    assert rc == 65
    assert ledger.read_text(encoding="utf-8") == before


def test_command_verb_reverts_an_unpriced_claim_row_when_the_window_is_stale(
    ledger: Path, stale_goal: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ll.GOAL_PATH_ENV, str(stale_goal))
    before = ledger.read_text(encoding="utf-8")
    rc = _write_via_command(ledger, claim_row(priced=False))
    assert rc == 65
    assert ledger.read_text(encoding="utf-8") == before


# --------------------------------------------------------------------------
# 6. Precision: `claim=` is a citation field, not a claim verb
#
# This class had to be closed in the same change as the window repair, because
# it was INVISIBLE while the gate was inert and turning the gate back on is
# exactly what exposes it. Measured over the live ledger on 2026-09-17: of the
# 194 rows is_claim_row then recognized, 151 were TERMINAL / PROGRESS / NOTE /
# CORRECTION rows carrying a `claim=<path>` citation, a `closes-CLAIM=<path>`
# citation, or a `claim-token` field. 52 of them were written in the three days
# to 2026-09-17, and every one would have been refused as an unpriced claim row
# -- i.e. the fleet's whole closeout path -- the moment enforcement resumed.
# After the fix: 42 recognized, 3 in that three-day window, and the one
# remaining prose false positive (a CORRECTION row quoting a peer's claim
# verbatim) is dropped by the 120-character field bound.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("row", "recognized", "why"),
    [
        (
            "2026-08-08T10:00:00Z | ticket-dedupe-0808 | OMN-15725 | claimed the occ-preflight pin bump",
            True,
            "a genuine act of claiming: the verb leads a pipe field and takes an object",
        ),
        (
            "2026-09-16T21:28:47Z | CLAIM | lane=x | taking over the stamp-rerun tail: watch OCC#99",
            True,
            "the 'taking' synonym, same structural position",
        ),
        (
            "2026-09-17T11:01:18Z | TERMINAL | lane=x | claim=docs/tracking/ROLLING_WORK_LEDGER.md:12",
            False,
            "claim= is a key=value citation pointing AT a claim, not an act of claiming",
        ),
        (
            "2026-09-17T09:33:21Z | TERMINAL | lane=x | OMN-17195 | claim-token LCT1-4137128-4064",
            False,
            "claim-token is a compound field name, not the verb",
        ),
        (
            "2026-09-16T00:43:27Z | CORRECTION | lane=x | ticket=OMN-1 | the row above cites "
            "closes-CLAIM=docs/tracking/ROLLING_WORK_LEDGER.md:8699, which is WRONG -- 8699 is "
            "another lane's row. This lane's CLAIM is at docs/tracking/ROLLING_WORK_LEDGER.md:8428 "
            "(2026-09-15T22:57:27Z | CLAIM | lane=x | ticket=OMN-1).",
            False,
            "a CORRECTION row quoting a peer's claim row verbatim deep in its prose",
        ),
    ],
)
def test_claim_verb_is_distinguished_from_a_claim_citation(
    row: str, recognized: bool, why: str
) -> None:
    assert ll.is_rule4_claim_row(row) is recognized, why


def test_the_precision_guards_are_present_in_the_pipe_field_pattern() -> None:
    """Pins the three guards by behaviour-adjacent inspection, so removing one
    while leaving the others is a red test rather than a silent regression of a
    class that only shows up once enforcement is live."""
    pattern = ll.CLAIM_PIPE_FIELD_PATTERN.pattern
    assert "(?<![\\w-])" in pattern, (
        "lost the leading compound-token guard (closes-CLAIM)"
    )
    assert "(?![\\s]*[=\\-])" in pattern, (
        "lost the trailing assignment guard (claim=, claim-token)"
    )
    assert "{0,120}" in pattern, "lost the field-length bound (quoted-claim-in-prose)"


# --------------------------------------------------------------------------
# 7. Measured residual: the dominant live CLAIM shape is invisible to the
#    matcher, so repairing the window does not by itself make the gate bite
# --------------------------------------------------------------------------


LIVE_PIPE_CLAIM_ROW = (
    "2026-09-17T12:14:43Z | CLAIM | lane=lab-receipt-settle-budget-build-1214 | "
    "actor=claude:opus5:subagent | scope=lab-pass receipt deployed_revision settle"
)


def test_the_dominant_live_pipe_claim_shape_is_now_recognized() -> None:
    """In omni_home this same row was pinned as an OPEN residual: 714 ledger rows
    lead with a timestamp whose first pipe field is the claim verb, and the
    matcher saw 3 of them. Porting the gate here closed it."""
    assert ll.is_rule4_claim_row(LIVE_PIPE_CLAIM_ROW)
    assert ll.is_pipe_lead_claim_row(LIVE_PIPE_CLAIM_ROW)


def test_the_pipe_lead_cutover_date_exists_and_is_a_date_not_a_flag() -> None:
    """The date is the ONLY switch. A flag is a thing lanes set and forget, and a
    gate that is off while everyone assumes it is on is the whole defect this
    ticket closes -- so the absence of an env opt-in is itself pinned."""
    assert isinstance(ll.RULE4_PIPE_LEAD_CUTOVER_UTC, datetime)
    assert datetime(2026, 9, 24, tzinfo=UTC) == ll.RULE4_PIPE_LEAD_CUTOVER_UTC
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in (
        "RULE4_PIPE_LEAD_GRACE",
        "LEDGER_LOCK_PIPE_LEAD",
        "PIPE_LEAD_OPT_IN",
    ):
        assert forbidden not in source, (
            f"{forbidden} is an opt-in switch; the date is the only one"
        )


def _pipe_row(moment: datetime, *, priced: bool, lane: str = "grace-lane") -> str:
    head = f"{_stamp(moment)} | CLAIM | lane={lane} | ticket=OMN-18554 | scope=x"
    return head + (
        "; est ~2 lane-hours; displaces nothing; (OMN-18554)" if priced else ""
    )


# The cutover is exercised through validate_claim_payload with an explicit `now`
# rather than through main(). main() runs the OMN-17427 clock guard first, and
# that guard deliberately refuses BOTH a row stamped days from the wall clock and
# a LEDGER_LOCK_NOW override that disagrees with the real clock -- which is the
# anti-forgery property OMN-17427 exists for. Driving a future date through main()
# would therefore be testing the clock guard, not the cutover, and the only way to
# make it pass would be to weaken the guard. The date lives in
# validate_claim_payload, so that is where it is pinned.


def test_before_the_cutover_an_unpriced_pipe_lead_row_lands_and_is_reported(
    capsys: pytest.CaptureFixture[str],
    stale_goal: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ll.GOAL_PATH_ENV, str(stale_goal))
    before_cutover = ll.RULE4_PIPE_LEAD_CUTOVER_UTC - timedelta(days=1)
    row = _pipe_row(before_cutover, priced=False)
    written, rejection = ll.validate_claim_payload(
        row, cost_unknown=[], now=before_cutover
    )
    err = capsys.readouterr().err
    assert rejection is None, "an unpriced pipe-lead row still LANDS before the cutover"
    assert written == row
    assert "RULE-4 UNPRICED CLAIM" in err
    assert "lane=grace-lane" in err
    assert "2026-09-24T00:00:00Z" in err, (
        "the report must name the date it becomes a refusal"
    )


def test_at_the_cutover_the_same_row_is_refused(
    stale_goal: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The falsifier for the whole dated-grace design: same row, same plan, one
    second past the cutover, opposite outcome."""
    monkeypatch.setenv(ll.GOAL_PATH_ENV, str(stale_goal))
    at_cutover = ll.RULE4_PIPE_LEAD_CUTOVER_UTC + timedelta(seconds=1)
    _written, rejection = ll.validate_claim_payload(
        _pipe_row(at_cutover, priced=False), cost_unknown=[], now=at_cutover
    )
    assert rejection is not None
    assert "cost-sentence check failed" in rejection


def test_a_priced_pipe_lead_row_is_accepted_on_both_sides_of_the_cutover(
    stale_goal: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cutover changes what happens to an UNPRICED row. A priced one was
    always fine and stays fine -- otherwise the date would read as a deadline for
    the row SHAPE rather than for the price."""
    monkeypatch.setenv(ll.GOAL_PATH_ENV, str(stale_goal))
    for moment in (
        ll.RULE4_PIPE_LEAD_CUTOVER_UTC - timedelta(days=1),
        ll.RULE4_PIPE_LEAD_CUTOVER_UTC + timedelta(days=1),
    ):
        _written, rejection = ll.validate_claim_payload(
            _pipe_row(moment, priced=True), cost_unknown=[], now=moment
        )
        assert rejection is None, moment


def test_an_unpriced_pipe_lead_row_lands_through_main_today(
    ledger: Path,
    stale_goal: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End-to-end on the REAL clock, which is what every lane hits today: the
    dominant shape is now inspected, reported, and still lands."""
    monkeypatch.setenv(ll.GOAL_PATH_ENV, str(stale_goal))
    assert datetime.now(UTC) < ll.RULE4_PIPE_LEAD_CUTOVER_UTC, (
        "this test is only meaningful before the cutover; after it, the companion "
        "test_at_the_cutover_the_same_row_is_refused is the live one"
    )
    row = _pipe_row(datetime.now(UTC), priced=False)
    assert ll.main([str(ledger), "--append", row]) == 0
    assert "RULE-4 UNPRICED CLAIM" in capsys.readouterr().err
    assert "| CLAIM |" in ledger.read_text(encoding="utf-8")


def test_the_claim_token_notion_is_a_separate_predicate() -> None:
    """is_claim_row here answers "should this row mint a claim token" and is
    deliberately broad; is_rule4_claim_row answers "does this row owe a price"
    and is deliberately narrow. Merging them would either refuse rows that only
    needed a token, or mint tokens for rows that only needed a price."""
    citation = "- 2026-09-17T11:01:18Z NOTE: the row above cites CLAIM: ledger.md:12"
    assert ll.is_claim_row(citation), "the token notion is broad by design"
    assert not ll.is_rule4_claim_row(citation), "the pricing notion is narrow by design"


def test_the_recognized_claim_shape_this_file_tests_with_is_really_recognized() -> None:
    """Positive control for the test above: a zero-recognition result would
    make every enforcement assertion in this file vacuous, and an empty result
    is not evidence of absence (CLAUDE.md rule 16)."""
    assert ll.is_rule4_claim_row(claim_row(priced=False))
    assert ll.is_rule4_claim_row(claim_row(priced=True))
    assert not ll.is_rule4_claim_row(terminal_row())


# --------------------------------------------------------------------------
# 8. AC3 positive control: the live declared source resolves a window
# --------------------------------------------------------------------------


def test_live_goal_source_declares_a_resolvable_stamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skips rather than fails where the clone is absent -- a machine without
    the private clone is not evidence about the goal file's contents. Where the
    clone IS present this is the AC3 falsifier: read_declared_window against
    the real resolved path must return a (start, end) pair with no override."""
    monkeypatch.delenv(ll.GOAL_PATH_ENV, raising=False)
    try:
        goal_path = ll.goal_path_for_window()
    except ll.GoalSourceUnresolvedError as exc:
        pytest.skip(f"knowledge-base-internal clone not resolvable here: {exc}")
    if not goal_path.is_file():
        pytest.skip(f"goal file not present in this clone: {goal_path}")
    window = ll.read_declared_window(goal_path)
    assert window is not None, f"{goal_path} carries no parseable 'state_as_of:' line"
    start, end = window
    assert end - start == timedelta(days=ll.FRESHNESS_HORIZON_DAYS)
