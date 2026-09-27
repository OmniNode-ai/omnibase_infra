# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Incident replay for the companion-merge heal (OMN-15547 case, OMN-18812).

THE INCIDENT, IN COMMITTED BYTES
--------------------------------
``omnibase_infra#3999``, head ``c8de38dc6326``, body stamped
``Evidence-Source: OCC#10983``. Its ``occ-preflight / eligibility`` and
``OCC Companion Merged Gate (OMN-15214)`` check runs failed between 18:02Z and
18:46Z on 2026-09-23, after the bounded 1500 s wait for the companion ran out.
The companion ``onex_change_control#10983`` merged at 20:35:00Z the same day.
Nothing in this repository listens for that event: the OMN-18352
``occ-preflight-heal`` fires on ``pull_request: edited`` and on
``workflow_run: completed``, and both had already fired. So the head still
carried the pre-merge failures three days later, when these bytes were captured
on 2026-09-26, and only a push or a hand-typed ``gh run rerun --failed`` would
have cleared them.

That is the state the merge-throughput measurement of 2026-09-26 counted across
the registry: 212 extra PR-hours in one day on companion-bound merges, and the
heal that clears it (omniclaude ``occ-companion-merge-heal``, proven live there
by run 36230786411) existed in omniclaude only.

WHY THE FIXTURES ARE CAPTURES AND NOT A RECONSTRUCTION
------------------------------------------------------
Every file under ``tests/fixtures/omn18812/`` is the verbatim output of one
``gh api`` read, gzip'd with ``gzip -n -9`` for size, re-fetchable from the
locator in ``tests/incident_replays/registry.yaml``. The check-run, run and job
lists were read with ``--paginate --slurp``, which is exactly the shape the
guard's own ``GhCli`` reads, so the real payload parsers run over them unchanged.
The one adaptation is the companion state: the guard reads it through
``gh pr view --json state`` (``MERGED``/``OPEN``/``CLOSED``), which has no
re-fetchable locator, so the replay reads the REST pull payload and maps its
``merged`` flag onto that one field before handing it to the guard's own parser.

WHAT THE REPLAY PROVES
----------------------
Driven over the real bytes, the guard selects exactly the three runs whose OWN
preflight job failed (Reject skip-gate bypass tokens ``35898373010``, CI
``35898372640``, Hostile Reviewer ``35898371776``) out of six failed runs on the
head. The two Receipt Gate runs and the later CI run failed for reasons the
companion merge does not touch, and re-running them would spend compute to
reproduce a failure that is already correct.

THE DISCRIMINATOR IS MANDATORY. A heal hardwired to re-run would replay this
incident perfectly and then spend a second 1500 s budget on every PR whose
companion is still open. The same guard is driven over ``omnibase_infra#4191``,
whose failed preflight cites ``OCC#11538``, open at capture time, and must refuse
it without reading a single run.
"""

from __future__ import annotations

import gzip
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from scripts.ci.occ_companion_merge_heal import (
    PREFLIGHT_JOB_MARKERS,
    EnumCompanionHealOutcome,
    EnumCompanionState,
    RunSnapshot,
    collect_decisions,
    companion_state_from_payload,
    failed_preflight_check_count_in_payload,
    failed_runs_in_payload,
    main,
    run_failed_on_preflight,
)

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "omn18812"
REPO = "OmniNode-ai/omnibase_infra"
OCC_REPO = "OmniNode-ai/onex_change_control"

INCIDENT_PR = 3999
INCIDENT_COMPANION = 10983
CONTROL_PR = 4191
CONTROL_COMPANION = 11538

#: The cited artifact in tests/incident_replays/registry.yaml: the incident
#: head's check-run list, the surface that stayed red after the merge.
INCIDENT_CHECK_RUNS = FIXTURES / "check-runs-3999.json.gz.captured"

#: The runs on the incident head whose own preflight-family job failed, in the
#: order the captured runs listing returns them.
EXPECTED_HEALED_RUNS = (35898373010, 35898372640, 35898371776)
#: Failed on the same head for reasons the companion merge does not touch.
EXPECTED_LEFT_ALONE = (35915880328, 35899231397, 35898372411)


def _ts(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def _load(name: str) -> Any:
    raw = gzip.decompress((FIXTURES / f"{name}.json.gz.captured").read_bytes())
    return json.loads(raw)


class ReplayGh:
    """:class:`GhPort` served from the committed captures, recording writes."""

    def __init__(self, pr_numbers: tuple[int, ...]) -> None:
        self.pr_numbers = pr_numbers
        self.reruns: list[int] = []
        self.runs_read: list[str] = []

    def open_pull_requests(self, *, repo: str) -> tuple[tuple[int, str, str], ...]:
        assert repo == REPO
        out: list[tuple[int, str, str]] = []
        for number in self.pr_numbers:
            pull = _load(f"pull-{number}")
            out.append((pull["number"], pull["head"]["sha"], pull["body"] or ""))
        return tuple(out)

    def _pr_for_head(self, head_sha: str) -> int:
        for number in self.pr_numbers:
            if _load(f"pull-{number}")["head"]["sha"] == head_sha:
                return number
        raise AssertionError(f"no capture for head {head_sha}")

    def failed_preflight_check_count(self, *, repo: str, head_sha: str) -> int:
        assert repo == REPO
        pages = _load(f"check-runs-{self._pr_for_head(head_sha)}")
        return sum(
            failed_preflight_check_count_in_payload(page, markers=PREFLIGHT_JOB_MARKERS)
            for page in pages
        )

    def companion_state(self, *, occ_repo: str, number: int) -> EnumCompanionState:
        assert occ_repo == OCC_REPO
        rest = _load(f"occ-pull-{number}")
        view_state = "MERGED" if rest["merged"] else str(rest["state"]).upper()
        return companion_state_from_payload({"state": view_state})

    def failed_runs(self, *, repo: str, head_sha: str) -> tuple[RunSnapshot, ...]:
        assert repo == REPO
        self.runs_read.append(head_sha)
        pages = _load(f"runs-{self._pr_for_head(head_sha)}")
        out: list[RunSnapshot] = []
        for page in pages:
            out.extend(failed_runs_in_payload(page))
        return tuple(out)

    def run_failed_on_preflight(self, *, repo: str, run_id: int) -> bool:
        assert repo == REPO
        pages = _load(f"jobs-{run_id}")
        return any(
            run_failed_on_preflight(page, markers=PREFLIGHT_JOB_MARKERS)
            for page in pages
        )

    def rerun_failed(self, *, repo: str, run_id: int) -> None:
        assert repo == REPO
        self.reruns.append(run_id)


# ---------------------------------------------------------------------------
# The incident's facts, read out of the bytes before the guard is involved.
# ---------------------------------------------------------------------------


def test_the_captures_show_a_preflight_that_failed_before_its_companion_merged() -> (
    None
):
    pull = _load(f"pull-{INCIDENT_PR}")
    stamp = re.search(r"^Evidence-Source:\s*(\S+)\s*$", pull["body"], re.MULTILINE)
    assert stamp is not None
    assert stamp.group(1) == f"OCC#{INCIDENT_COMPANION}"

    companion = _load(f"occ-pull-{INCIDENT_COMPANION}")
    assert companion["merged"] is True
    merged_at = _ts(companion["merged_at"])

    head = pull["head"]["sha"]
    check_pages = json.loads(gzip.decompress(INCIDENT_CHECK_RUNS.read_bytes()))
    failed_completions = [
        _ts(check["completed_at"])
        for page in check_pages
        for check in page["check_runs"]
        if check["head_sha"] == head
        and check["conclusion"] == "failure"
        and "occ" in check["name"].lower()
    ]
    assert failed_completions
    # Every failed change-control verdict on the head predates the merge, so
    # it describes a companion state that no longer holds.
    assert all(done < merged_at for done in failed_completions)


# ---------------------------------------------------------------------------
# The real guard over the real incident.
# ---------------------------------------------------------------------------


def test_the_real_guard_reruns_exactly_the_runs_the_merged_companion_unblocks() -> None:
    gh = ReplayGh((INCIDENT_PR,))
    [decision] = collect_decisions(gh, repo=REPO, occ_repo=OCC_REPO)

    assert decision.outcome is EnumCompanionHealOutcome.RERUN_REQUIRED
    assert decision.pr_number == INCIDENT_PR
    assert decision.run_ids == EXPECTED_HEALED_RUNS
    assert not set(decision.run_ids) & set(EXPECTED_LEFT_ALONE)
    # Deciding is read-only; the write happens only in main().
    assert gh.reruns == []


def test_main_issues_the_rerun_the_incident_was_waiting_for() -> None:
    gh = ReplayGh((INCIDENT_PR,))
    assert main(["--repo", REPO], gh=gh) == 0
    assert gh.reruns == list(EXPECTED_HEALED_RUNS)


def test_dry_run_over_the_incident_issues_nothing() -> None:
    gh = ReplayGh((INCIDENT_PR,))
    assert main(["--repo", REPO, "--dry-run"], gh=gh) == 0
    assert gh.reruns == []


# ---------------------------------------------------------------------------
# Discriminator: the same guard leaves an open companion alone.
# ---------------------------------------------------------------------------


def test_the_same_guard_refuses_a_real_pr_whose_companion_is_still_open() -> None:
    companion = _load(f"occ-pull-{CONTROL_COMPANION}")
    assert companion["state"] == "open"
    assert companion["merged"] is False

    gh = ReplayGh((CONTROL_PR,))
    [decision] = collect_decisions(gh, repo=REPO, occ_repo=OCC_REPO)

    assert decision.outcome is EnumCompanionHealOutcome.COMPANION_UNMERGED
    assert decision.run_ids == ()
    # The companion state is read BEFORE any run, so an open companion never
    # costs a runs listing, let alone a rerun.
    assert gh.runs_read == []
    assert main(["--repo", REPO], gh=gh) == 0
    assert gh.reruns == []


def test_both_prs_in_one_pass_heal_only_the_merged_one() -> None:
    gh = ReplayGh((INCIDENT_PR, CONTROL_PR))
    assert main(["--repo", REPO], gh=gh) == 0
    assert gh.reruns == list(EXPECTED_HEALED_RUNS)
