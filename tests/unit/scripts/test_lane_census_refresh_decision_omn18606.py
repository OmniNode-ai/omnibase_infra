# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The refresh leg must open a PR when the census moved, and stay quiet otherwise (OMN-18606).

`.github/workflows/lane-census-refresh.yml` collects a census on the lab host
four times a day. The interesting half is not the collection — it is deciding
whether the result is worth a pull request. Two failure directions, both real:

  Too eager   Four no-op PRs a day forever. That is how an automation earns
              itself an exemption, and then the leg is gone and the census is
              back to being hand-healed.

  Too quiet   The committed census ages past the gate's 7-day limit anyway and
              a person opens the PR — the exact loop OMN-18606 exists to end.

There is also a direction that is worse than either: committing a snapshot that
makes things WORSE. A malformed candidate, or one whose `emitted_at` runs
backwards, would replace a good committed census with one that fails the
staleness gate. Those are guards, not preferences, and they are pinned first.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO / "scripts"
_MODULE = _SCRIPTS / "lane_census_refresh_decision.py"
_WORKFLOW = _REPO / ".github" / "workflows" / "lane-census-refresh.yml"

sys.path.insert(0, str(_SCRIPTS))

from lane_census_refresh_decision import (
    REASON_AGING,
    REASON_CANDIDATE_NOT_NEWER,
    REASON_COMMITTED_UNREADABLE,
    REASON_FRESH_AND_UNCHANGED,
    REASON_TOPOLOGY_CHANGED,
    MalformedCandidateError,
    decide_refresh,
)

_NOW = datetime(2026, 9, 17, 18, 0, 0, tzinfo=UTC)


def _snapshot(
    *,
    emitted: datetime,
    alert_key: str = "lane-census-drift:omninode-pc:aaaaaaaaaaaaaaaa",
) -> dict[str, Any]:
    """A minimally valid census carrying the fields the decision reads."""
    return {
        "schema_version": "1.0.0",
        "emitted_at": emitted.isoformat(),
        "alert_key": alert_key,
        "lanes_checked": ["stability-test", "judge", "dev", "lakshman"],
        "findings": [],
        "drift_count": 0,
    }


# --------------------------------------------------------------------------
# Guards. These protect against making the committed state worse.
# --------------------------------------------------------------------------


def test_a_backwards_clock_never_moves_the_census() -> None:
    """A candidate older than the committed one is a NO-OP, not a refresh.

    The staleness gate reads `emitted_at`. If a host with a skewed clock could
    write an older timestamp over a newer one, this leg would MANUFACTURE the
    staleness it exists to prevent. The guard is unconditional — it is checked
    before the topology and aging rules, so a backwards candidate cannot smuggle
    itself in by also happening to differ.
    """
    committed = _snapshot(emitted=_NOW - timedelta(hours=1))
    candidate = _snapshot(
        emitted=_NOW - timedelta(days=2), alert_key="totally:different:key"
    )

    decision = decide_refresh(committed, candidate, now=_NOW)

    assert decision.refresh is False
    assert decision.reason == REASON_CANDIDATE_NOT_NEWER


def test_an_equal_timestamp_is_also_a_no_op() -> None:
    """Boundary: `<=`, not `<`. Re-committing an identical census is pointless."""
    emitted = _NOW - timedelta(days=5)
    decision = decide_refresh(
        _snapshot(emitted=emitted), _snapshot(emitted=emitted), now=_NOW
    )
    assert decision.refresh is False
    assert decision.reason == REASON_CANDIDATE_NOT_NEWER


@pytest.mark.parametrize(
    ("candidate", "why"),
    [
        ({"schema_version": "9.9.9", "emitted_at": _NOW.isoformat()}, "unknown schema"),
        ({"schema_version": "1.0.0"}, "no emitted_at at all"),
        ({"schema_version": "1.0.0", "emitted_at": ""}, "empty emitted_at"),
        ({"schema_version": "1.0.0", "emitted_at": "not-a-timestamp"}, "unparseable"),
        ({"schema_version": "1.0.0", "emitted_at": 1758132000}, "non-string"),
    ],
)
def test_a_malformed_candidate_is_an_error_never_a_refresh(
    candidate: dict[str, Any], why: str
) -> None:
    """Committing an ungateable snapshot is worse than committing nothing.

    Each of these would pass the *file exists* check and then fail the staleness
    gate as malformed on the next PR, turning an automation into a repo-wide
    outage. The decision refuses rather than returning `refresh=True`.
    """
    with pytest.raises(MalformedCandidateError):
        decide_refresh(_snapshot(emitted=_NOW - timedelta(days=9)), candidate, now=_NOW)


# --------------------------------------------------------------------------
# The refresh cases.
# --------------------------------------------------------------------------


def test_missing_committed_census_refreshes() -> None:
    """Bootstrap and corruption both want a good file written."""
    decision = decide_refresh(None, _snapshot(emitted=_NOW), now=_NOW)
    assert decision.refresh is True
    assert decision.reason == REASON_COMMITTED_UNREADABLE


def test_committed_census_with_unusable_timestamp_refreshes() -> None:
    """A committed file the gate cannot read is as good as absent."""
    committed = _snapshot(emitted=_NOW - timedelta(days=1))
    committed["emitted_at"] = "garbage"
    decision = decide_refresh(committed, _snapshot(emitted=_NOW), now=_NOW)
    assert decision.refresh is True
    assert decision.reason == REASON_COMMITTED_UNREADABLE


def test_topology_change_refreshes_even_when_recent() -> None:
    """A moved fleet is worth a PR on its own merits, aging or not.

    `alert_key` is a content hash of the census findings, so a difference means
    the committed census now describes lanes that are not what is running. That
    is the documentation failure OMN-13034 exists to catch, and waiting three
    days to report it would be the phantom-lane problem again.
    """
    decision = decide_refresh(
        _snapshot(emitted=_NOW - timedelta(minutes=30), alert_key="key:before"),
        _snapshot(emitted=_NOW, alert_key="key:after"),
        now=_NOW,
    )
    assert decision.refresh is True
    assert decision.reason == REASON_TOPOLOGY_CHANGED


def test_aging_census_refreshes_at_the_threshold() -> None:
    """At the threshold exactly, not one day after it."""
    decision = decide_refresh(
        _snapshot(emitted=_NOW - timedelta(days=3)),
        _snapshot(emitted=_NOW),
        refresh_after_days=3,
        now=_NOW,
    )
    assert decision.refresh is True
    assert decision.reason == REASON_AGING


def test_the_threshold_leaves_real_margin_under_the_gate() -> None:
    """The refresh threshold must be well under the gate's own limit.

    A threshold at or near 7 days would open every PR as a race against the
    deadline. This asserts the margin exists rather than trusting the comment.
    """
    from check_lane_census_age import _DEFAULT_MAX_AGE_DAYS
    from lane_census_refresh_decision import DEFAULT_REFRESH_AFTER_DAYS

    assert DEFAULT_REFRESH_AFTER_DAYS < _DEFAULT_MAX_AGE_DAYS, (
        "the refresh threshold must be strictly under the staleness limit"
    )
    assert _DEFAULT_MAX_AGE_DAYS - DEFAULT_REFRESH_AFTER_DAYS >= 3, (
        "a refresh PR needs days to land, not hours; keep at least a 3-day margin"
    )


# --------------------------------------------------------------------------
# THE NO-OP. The case that keeps this automation alive.
# --------------------------------------------------------------------------


def test_recent_and_unchanged_is_a_no_op() -> None:
    """Nothing moved and nothing is aging: say nothing.

    Without this the leg opens four PRs a day forever.
    """
    decision = decide_refresh(
        _snapshot(emitted=_NOW - timedelta(hours=6)),
        _snapshot(emitted=_NOW),
        refresh_after_days=3,
        now=_NOW,
    )
    assert decision.refresh is False
    assert decision.reason == REASON_FRESH_AND_UNCHANGED


def test_a_days_worth_of_scheduled_runs_opens_no_pr_when_nothing_moves() -> None:
    """The cadence itself must not generate PRs — the real anti-churn assertion.

    Four runs a day against an unchanged fleet, walked forward on the clock. The
    single-case test above could pass while the cadence still churned; this one
    fails if the rule is ever loosened to "newer means refresh".
    """
    committed = _snapshot(emitted=_NOW)
    for hours in (6, 12, 18, 24, 30, 36, 42, 48):
        later = _NOW + timedelta(hours=hours)
        decision = decide_refresh(
            committed, _snapshot(emitted=later), refresh_after_days=3, now=later
        )
        assert decision.refresh is False, (
            f"the leg would have opened a PR {hours}h in with nothing changed "
            f"(reason {decision.reason}); that is four no-op PRs a day"
        )

    # ... and on day 3 it speaks up, so the quiet above is restraint, not silence.
    day_three = _NOW + timedelta(days=3)
    assert (
        decide_refresh(
            committed, _snapshot(emitted=day_three), refresh_after_days=3, now=day_three
        ).refresh
        is True
    )


# --------------------------------------------------------------------------
# CLI contract — the workflow shells this, so its exit codes are load-bearing.
# --------------------------------------------------------------------------


def _run_cli(committed: Path, candidate: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(_MODULE),
            "--committed",
            str(committed),
            "--candidate",
            str(candidate),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_emits_a_decision_document(tmp_path: Path) -> None:
    committed = tmp_path / "committed.json"
    candidate = tmp_path / "candidate.json"
    committed.write_text(
        json.dumps(_snapshot(emitted=datetime.now(UTC) - timedelta(days=9)))
    )
    candidate.write_text(json.dumps(_snapshot(emitted=datetime.now(UTC))))

    result = _run_cli(committed, candidate)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["refresh"] is True


def test_cli_refuses_an_absent_candidate(tmp_path: Path) -> None:
    """No census collected means no PR — never a silent success."""
    committed = tmp_path / "committed.json"
    committed.write_text(json.dumps(_snapshot(emitted=datetime.now(UTC))))
    result = _run_cli(committed, tmp_path / "does-not-exist.json")
    assert result.returncode == 1
    assert "absent or not valid JSON" in result.stderr


# --------------------------------------------------------------------------
# Workflow wiring. The decision is useless if the leg does not consult it.
# --------------------------------------------------------------------------


def test_the_workflow_opens_prs_only_behind_the_decision() -> None:
    """Both the PR step and the no-op step must be gated on the decision output."""
    body = _WORKFLOW.read_text(encoding="utf-8")
    assert "scripts/lane_census_refresh_decision.py" in body
    assert "if: steps.decide.outputs.refresh == 'true'" in body, (
        "the PR-opening step must be conditional on the decision, or the leg "
        "opens a PR on every scheduled run"
    )
    assert "gh pr create" in body


def test_the_workflow_runs_where_the_census_is_meaningful() -> None:
    """The census describes the lab host's lanes; collecting it elsewhere is a lie."""
    body = _WORKFLOW.read_text(encoding="utf-8")
    assert "runs-on: [self-hosted, omnibase-verify, host-201]" in body


def test_the_workflow_fails_rather_than_opening_a_pr_on_a_bad_collection() -> None:
    """Collector exit codes other than 0 (clean) and 30 (drift) must fail the job.

    3 (missing deps), 4 (inventory unobservable) and 2 (bad args) all produce no
    census. A leg that treated them as "nothing changed" would report a healthy
    fleet it never observed.
    """
    body = _WORKFLOW.read_text(encoding="utf-8")
    assert "$rc -ne 0 && $rc -ne 30" in body
    assert "LANE-CENSUS-UNCOLLECTABLE" in body


# --------------------------------------------------------------------------
# The writer-App PR path. The leg's PR-opening half cannot be exercised
# without the `host-201` runner, so these pin the properties a reviewer would
# otherwise have to take on trust — and the end-to-end below drives the real
# decision CLI from a FIXTURE census rather than the lab host.
# --------------------------------------------------------------------------


def test_the_pr_is_authored_by_the_writer_app_not_the_default_token() -> None:
    """A default-GITHUB_TOKEN push does not trigger downstream CI.

    The bump PR is worthless if the staleness gate never runs on it, so the leg
    must mint and use an App installation token. `omninode_infra` kept a PAT for
    years on the mistaken belief that App pushes were also suppressed; the
    correction (OMN-18273) is that an App token DOES trigger CI provided the
    checkout does not leave a `GITHUB_TOKEN` extraheader overriding it. Here
    that is satisfied by checking out WITH the app token rather than clearing
    credentials.
    """
    body = _WORKFLOW.read_text(encoding="utf-8")
    assert "actions/create-github-app-token" in body
    assert "secrets.ONEXBOT_OCC_APP_ID" in body
    assert "secrets.ONEXBOT_OCC_PRIVATE_KEY" in body
    assert "token: ${{ steps.app-token.outputs.token }}" in body, (
        "the checkout must carry the app token, or the push authenticates as "
        "GITHUB_TOKEN and the bump PR gets no CI"
    )
    assert "secrets.GITHUB_TOKEN" not in body, (
        "no fallback to GITHUB_TOKEN: a silent substitution when the mint fails "
        "is exactly the confound that made the PAT look necessary (OMN-18273)"
    )


def test_the_commit_is_attributed_to_the_app_it_authenticates_as() -> None:
    """Commit identity must match the pushing identity (OMN-18273)."""
    body = _WORKFLOW.read_text(encoding="utf-8")
    assert 'git config user.name "onexbot-occ-writer[bot]"' in body
    assert "onexbot-occ-writer[bot]@users.noreply.github.com" in body


def test_the_pr_title_clears_the_ticket_gate_and_arms_auto_merge() -> None:
    """A bot PR nobody merges is a queue of stale branches.

    `chore(deps,` is the prefix the pr-title ticket gate exempts, matching the
    sibling-lock-refresh and publish-downstream-pin-bump conventions; without it
    every bump PR fails `pr-title / check-title` on a bot-authored title. And
    the PR must arm auto-merge, or the leg replaces a hand-opened PR with a
    hand-merged one and has removed half a step.
    """
    body = _WORKFLOW.read_text(encoding="utf-8")
    assert (
        '--title "chore(deps, OMN-18606): refresh the lane census snapshot [bot]"'
        in body
    )
    assert "gh pr merge" in body and "--squash --auto" in body


def test_the_leg_writes_only_the_census_file() -> None:
    """Blast radius. A refresh PR that touched anything else would be unreviewable."""
    body = _WORKFLOW.read_text(encoding="utf-8")
    added = [
        line.strip()
        for line in body.splitlines()
        if line.strip().startswith("git add ")
    ]
    assert added == ["git add deploy/lane-census/census-snapshot.json"], (
        f"the leg stages something other than the census: {added}"
    )


def test_end_to_end_from_a_fixture_census_rather_than_the_lab_host(
    tmp_path: Path,
) -> None:
    """Drive the real decision CLI over a fixture pair, on a synthetic clock.

    This is the collect-decide-report chain the workflow runs, with the collect
    step replaced by a fixture so it needs no docker socket, no runner and no
    access to the lab host. Both directions are asserted from the same pair of
    files, which is what makes it a test of the RULE and not of the clock:

      * `--refresh-after-days 0` forces the aging branch  -> refresh, PR opens
      * the default 3-day threshold on a recent census    -> no-op, no PR

    The second is the one that matters operationally. It is also exactly how the
    leg is meant to be dispatched for a live proof once the host is reachable:
    `workflow_dispatch` with `refresh_after_days: 0`.

    OMN-18980. Both documents are fixtures with SYNTHETIC timestamps, and the
    committed one is no longer the live file in the repository. It was, and the
    shape of the committed census is still taken from it so the fixture cannot
    drift from the real document -- but its `emitted_at` was the real one, so
    the steady case asserted "recent" against a file whose age nobody in this
    test controls. It expired on 2026-09-21 when the committed census turned
    three days old, and it reddened `Tests (Split 12/15)` on EVERY open pull
    request in the repository, cascading to the tests gate, the ratchet and the
    summary.

    That is worse than a stale fixture, because of WHEN it fires. The threshold
    is three days precisely so a refresh opened at day three has four days to be
    reviewed and land before the seven-day age gate goes red. This test made
    that window unusable: the repository goes red at day three, the same moment
    the bot opens its refresh, so nothing can merge during the margin --
    including the refresh pull request itself. The race the threshold exists to
    avoid was being reintroduced by the test that proves the leg.
    """
    committed_path = _REPO / "deploy" / "lane-census" / "census-snapshot.json"
    committed = dict(json.loads(committed_path.read_text(encoding="utf-8")))

    # A synthetic committed census: the real document's shape, a controlled age.
    # One day old, so it is unambiguously inside a 3-day threshold and
    # unambiguously outside a 0-day one, whatever day this test runs on.
    now = datetime.now(UTC)
    committed["emitted_at"] = (now - timedelta(days=1)).isoformat()
    fixture_committed_path = tmp_path / "census-committed.json"
    fixture_committed_path.write_text(json.dumps(committed), encoding="utf-8")
    committed_path = fixture_committed_path

    # A fixture candidate: same fleet (same alert_key), collected "now".
    candidate = dict(committed)
    candidate["emitted_at"] = now.isoformat()
    candidate_path = tmp_path / "census-candidate.json"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    def _decide(days: str) -> dict[str, Any]:
        result = subprocess.run(
            [
                sys.executable,
                str(_MODULE),
                "--committed",
                str(committed_path),
                "--candidate",
                str(candidate_path),
                "--refresh-after-days",
                days,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        decision: dict[str, Any] = json.loads(result.stdout)
        return decision

    forced = _decide("0")
    assert forced["refresh"] is True, (
        "a zero-day threshold must force the aging branch; this is the dispatch "
        "shape used to prove the leg end to end"
    )
    assert forced["reason"] == REASON_AGING

    steady = _decide("3")
    assert steady["refresh"] is False, (
        "the committed census is recent and the fixture describes the same "
        f"fleet, so the leg must stay quiet; got {steady}"
    )
    assert steady["reason"] == REASON_FRESH_AND_UNCHANGED


def test_a_census_aged_past_the_threshold_still_asks_for_a_refresh(
    tmp_path: Path,
) -> None:
    """The other direction, on the same synthetic clock (OMN-18980).

    Making the steady case immune to wall-clock time would, on its own, let a
    module that NEVER ages out pass every assertion here -- the `_decide("0")`
    arm forces the branch through the threshold rather than through elapsed
    time, so it cannot tell the two apart. This case ages the fixture instead
    of forcing the threshold, so the aging branch is still proven by a real
    elapsed interval and the rule stays pinned in both directions.
    """
    committed_path = _REPO / "deploy" / "lane-census" / "census-snapshot.json"
    committed = dict(json.loads(committed_path.read_text(encoding="utf-8")))

    now = datetime.now(UTC)
    committed["emitted_at"] = (now - timedelta(days=9)).isoformat()
    aged_path = tmp_path / "census-committed-aged.json"
    aged_path.write_text(json.dumps(committed), encoding="utf-8")

    candidate = dict(committed)
    candidate["emitted_at"] = now.isoformat()
    candidate_path = tmp_path / "census-candidate.json"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(_MODULE),
            "--committed",
            str(aged_path),
            "--candidate",
            str(candidate_path),
            "--refresh-after-days",
            "3",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    decision: dict[str, Any] = json.loads(result.stdout)

    assert decision["refresh"] is True, (
        "a census nine days old against a three-day threshold must ask for a "
        f"refresh; got {decision}"
    )
    assert decision["reason"] == REASON_AGING


def test_the_bot_pr_binds_to_the_ticket_that_owns_the_leg() -> None:
    """The title decides which contract's DoD tree the bot PR must satisfy.

    The autobind resolves a ticket from this title and mints an OCC companion
    bound to it. While the title cited the lane-census ratchet's ticket, every
    refresh PR inherited that contract's accumulated tree — 17 checks, two of
    them BLOCK. Those two fetch the census at a PINNED sha and then assert the
    frozen artifact is under seven days old against wall-clock now, so each was
    true the day it was written and false forever about a week later.
    `omnibase_infra#3726` died on exactly that and so would every successor.

    Binding to the leg's own ticket is not cosmetic: it is the difference
    between a PR that can land unattended and one that structurally cannot.
    """
    body = _WORKFLOW.read_text(encoding="utf-8")

    titles = [line for line in body.splitlines() if "--title" in line]
    assert len(titles) == 1, titles
    assert "OMN-18606" in titles[0], (
        f"the bot PR must cite the ticket that owns this leg: {titles[0].strip()}"
    )

    commits = [line for line in body.splitlines() if 'git commit -m "' in line]
    assert len(commits) == 1, commits
    assert "OMN-18606" in commits[0], (
        "the commit subject and the PR title must name the same ticket, or the "
        f"receipt gate and the autobind can resolve different ones: {commits[0].strip()}"
    )


def test_the_bot_pr_body_resolves_exactly_one_ticket() -> None:
    """A second resolvable reference is a second thing for an extractor to pick.

    The defect this file's sibling test records was the WRONG ticket being
    resolved. Naming two in the body reintroduces the ambiguity by a different
    route, so the ratchet is referred to in prose rather than by token.
    """
    body = _WORKFLOW.read_text(encoding="utf-8")
    # The body is built by a printf block redirected into census_refresh_body.md.
    # Anchor on the redirect and walk BACK to the `{` that opens it: the marker
    # string's first occurrence in the file IS that redirect, so searching
    # forward from it runs off the end of the block.
    close = body.index('} > "${RUNNER_TEMP}/census_refresh_body.md"')
    open_brace = body.rindex("\n          {\n", 0, close)
    emitted = body[open_brace:close]
    assert "printf" in emitted, "did not locate the PR-body block"

    tickets = set(re.findall(r"OMN-\d+", emitted))
    assert tickets == {"OMN-18606"}, (
        f"the bot PR body must resolve exactly one ticket, its own; found {sorted(tickets)}"
    )
