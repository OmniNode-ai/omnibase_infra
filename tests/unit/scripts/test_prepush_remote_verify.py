# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit proof for remote full-suite verification of the pre-push gate (OMN-16688).

This helper lets the governed pre-push hook accept a GitHub-hosted CI run in
place of running the heavy suite on an over-loaded local host. That is exactly
the shape of change that historically turns into a bypass, so the properties
under test are the ones that keep it from becoming one:

  * **sha-pinned**    -- a run against any other tree is not evidence, and a
                         prefix or branch name is refused outright
  * **green-only**    -- in-flight, failed, and cancelled runs are not evidence
  * **full-suite**    -- a selector-NARROWED run is refused with the same force
                         as a failing one; this is the binding that stops the
                         helper from quietly downgrading the gate
  * **complete**      -- a missing or still-running shard voids the run
  * **unresolvable != pass** -- an API/`gh` failure is "no evidence" (exit 2),
                         never a silent success

The full-suite binding rests on one structural fact, pinned by
``test_narrowed_ceiling_cannot_reach_full_suite_count`` below: the selector
emits ``split_count == _FULL_SUITE_SPLIT_COUNT`` **only** for
``is_full_suite=True``, because narrowed selections are capped far below it. If
someone ever raises that cap, the gap closes and the shard denominator stops
being a faithful witness of "full suite" -- so that test fails loudly rather
than letting the guarantee erode silently.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

import pytest

from scripts.ci.detect_test_paths import _FULL_SUITE_SPLIT_COUNT, _split_count_for
from scripts.hooks.prepush_remote_verify import (
    FULL_SUITE_SPLIT_COUNT,
    RemoteVerifyError,
    check,
    main,
    run_is_full_suite_shaped,
)

pytestmark = pytest.mark.unit

_SHA = "a" * 40
_OTHER_SHA = "b" * 40
_SLUG = "OmniNode-ai/omnibase_infra"


def _shard_jobs(
    count: int,
    denominator: int | None = None,
    conclusion: str = "success",
) -> list[dict[str, Any]]:
    """Build ``count`` shard jobs named ``Tests (Split i/denominator)``."""
    denom = FULL_SUITE_SPLIT_COUNT if denominator is None else denominator
    return [
        {"name": f"Tests (Split {i}/{denom})", "conclusion": conclusion}
        for i in range(1, count + 1)
    ]


def _run(
    *,
    run_id: int = 101,
    head_sha: str = _SHA,
    status: str = "completed",
    conclusion: str = "success",
) -> dict[str, Any]:
    return {
        "id": run_id,
        "head_sha": head_sha,
        "status": status,
        "conclusion": conclusion,
        "html_url": f"https://github.com/{_SLUG}/actions/runs/{run_id}",
    }


@pytest.fixture
def fake_gh(monkeypatch: pytest.MonkeyPatch):
    """Stub the `gh` subprocess boundary; no network, no real credentials."""

    state: dict[str, Any] = {"runs": [], "jobs": [], "fail": None, "calls": []}

    def _fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        state["calls"].append(cmd)
        if state["fail"] is not None:
            return subprocess.CompletedProcess(cmd, 1, "", str(state["fail"]))
        joined = " ".join(cmd)
        if "repo view" in joined:
            payload: Any = {"nameWithOwner": _SLUG}
        elif "/jobs" in joined:
            payload = {"jobs": state["jobs"]}
        else:
            payload = {"workflow_runs": state["runs"]}
        return subprocess.CompletedProcess(cmd, 0, json.dumps(payload), "")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    return state


# ---------------------------------------------------------------------------
# Binding 3 -- full-suite shape (the anti-downgrade binding)
# ---------------------------------------------------------------------------


def test_full_matrix_all_green_is_full_suite_shaped() -> None:
    ok, why = run_is_full_suite_shaped(_shard_jobs(FULL_SUITE_SPLIT_COUNT))
    assert ok, why


def test_narrowed_run_is_rejected_even_when_every_shard_is_green() -> None:
    """A selector-narrowed run is green but is NOT full-suite evidence."""
    narrowed = _split_count_for(["tests/unit/a", "tests/unit/b", "tests/unit/c"])
    ok, why = run_is_full_suite_shaped(_shard_jobs(narrowed, denominator=narrowed))
    assert not ok
    assert "NARROWED" in why


def test_missing_shard_voids_the_run() -> None:
    ok, why = run_is_full_suite_shaped(_shard_jobs(FULL_SUITE_SPLIT_COUNT - 1))
    assert not ok
    assert "missing" in why


def test_failed_shard_voids_the_run() -> None:
    jobs = _shard_jobs(FULL_SUITE_SPLIT_COUNT)
    jobs[3]["conclusion"] = "failure"
    ok, why = run_is_full_suite_shaped(jobs)
    assert not ok
    assert "did not all succeed" in why


def test_incomplete_shard_voids_the_run() -> None:
    jobs = _shard_jobs(FULL_SUITE_SPLIT_COUNT)
    jobs[0]["conclusion"] = None
    ok, why = run_is_full_suite_shaped(jobs)
    assert not ok
    assert "incomplete" in why


def test_run_with_no_shard_jobs_is_not_evidence() -> None:
    ok, why = run_is_full_suite_shaped(
        [{"name": "Code Quality", "conclusion": "success"}]
    )
    assert not ok
    assert "no 'Tests (Split i/N)' shard jobs" in why


def test_narrowed_ceiling_cannot_reach_full_suite_count() -> None:
    """The structural gap the shard-denominator binding depends on.

    ``_split_count_for`` is the ONLY producer of a narrowed ``split_count``. If
    its ceiling ever reaches ``_FULL_SUITE_SPLIT_COUNT``, a narrowed run becomes
    indistinguishable from a full one by job name alone and the pre-push gate
    silently starts accepting less work than it demands.
    """
    ceiling = max(_split_count_for(["p"] * n) for n in range(200))
    assert ceiling < _FULL_SUITE_SPLIT_COUNT, (
        f"narrowed split ceiling {ceiling} has reached the full-suite count "
        f"{_FULL_SUITE_SPLIT_COUNT}; the shard-denominator binding in "
        "prepush_remote_verify.py is no longer sound"
    )


def test_module_constant_tracks_the_selector() -> None:
    assert FULL_SUITE_SPLIT_COUNT == _FULL_SUITE_SPLIT_COUNT


# ---------------------------------------------------------------------------
# Bindings 1 and 2 -- sha pinning and greenness
# ---------------------------------------------------------------------------


def test_accepts_sha_pinned_green_full_suite_run(fake_gh: dict[str, Any]) -> None:
    fake_gh["runs"] = [_run()]
    fake_gh["jobs"] = _shard_jobs(FULL_SUITE_SPLIT_COUNT)
    outcome = check(_SHA, slug=_SLUG)
    assert outcome.ok
    assert outcome.run_id == 101


def test_rejects_run_for_a_different_sha(fake_gh: dict[str, Any]) -> None:
    fake_gh["runs"] = [_run(head_sha=_OTHER_SHA)]
    fake_gh["jobs"] = _shard_jobs(FULL_SUITE_SPLIT_COUNT)
    outcome = check(_SHA, slug=_SLUG)
    assert not outcome.ok
    assert "pinned to the exact sha" in outcome.reason


def test_rejects_in_flight_run(fake_gh: dict[str, Any]) -> None:
    fake_gh["runs"] = [_run(status="in_progress", conclusion="")]
    outcome = check(_SHA, slug=_SLUG)
    assert not outcome.ok
    assert "not completed yet" in outcome.reason


@pytest.mark.parametrize("conclusion", ["failure", "cancelled", "timed_out"])
def test_rejects_non_success_conclusion(
    fake_gh: dict[str, Any], conclusion: str
) -> None:
    fake_gh["runs"] = [_run(conclusion=conclusion)]
    outcome = check(_SHA, slug=_SLUG)
    assert not outcome.ok
    assert "not success" in outcome.reason


def test_no_run_at_all_is_not_evidence(fake_gh: dict[str, Any]) -> None:
    fake_gh["runs"] = []
    outcome = check(_SHA, slug=_SLUG)
    assert not outcome.ok


def test_green_but_narrowed_run_is_refused_end_to_end(fake_gh: dict[str, Any]) -> None:
    """The whole point: green + sha-pinned is still not enough."""
    fake_gh["runs"] = [_run()]
    fake_gh["jobs"] = _shard_jobs(2, denominator=2)
    outcome = check(_SHA, slug=_SLUG)
    assert not outcome.ok
    assert "not full-suite shaped" in outcome.reason


def test_newest_qualifying_run_wins(fake_gh: dict[str, Any]) -> None:
    """A later green full-suite re-run supersedes an earlier failed attempt."""
    fake_gh["runs"] = [_run(run_id=1, conclusion="failure"), _run(run_id=2)]
    fake_gh["jobs"] = _shard_jobs(FULL_SUITE_SPLIT_COUNT)
    outcome = check(_SHA, slug=_SLUG)
    assert outcome.ok
    assert outcome.run_id == 2


@pytest.mark.parametrize(
    "bad_sha",
    ["a" * 39, "a" * 41, "A" * 40, "HEAD", "dev", "", "a" * 7],
)
def test_refuses_anything_that_is_not_a_full_sha(bad_sha: str) -> None:
    with pytest.raises(RemoteVerifyError, match="40-character"):
        check(bad_sha, slug=_SLUG)


# ---------------------------------------------------------------------------
# Unresolvable != pass
# ---------------------------------------------------------------------------


def test_gh_failure_raises_rather_than_passing(fake_gh: dict[str, Any]) -> None:
    fake_gh["fail"] = "gh: could not authenticate"
    with pytest.raises(RemoteVerifyError):
        check(_SHA, slug=_SLUG)


def test_cli_exit_codes_distinguish_no_evidence_from_unresolvable(
    fake_gh: dict[str, Any],
) -> None:
    """0 = pass, 1 = resolved-but-no-evidence, 2 = could not resolve."""
    fake_gh["runs"] = [_run()]
    fake_gh["jobs"] = _shard_jobs(FULL_SUITE_SPLIT_COUNT)
    assert main(["check", "--head-sha", _SHA, "--repo", _SLUG]) == 0

    fake_gh["runs"] = []
    assert main(["check", "--head-sha", _SHA, "--repo", _SLUG]) == 1

    fake_gh["fail"] = "boom"
    assert main(["check", "--head-sha", _SHA, "--repo", _SLUG]) == 2
