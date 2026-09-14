# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Incident replay for the occ-preflight heal race (OMN-15547 case, OMN-18352).

THE INCIDENT, IN THE BUGGY GUARD'S OWN WORDS
--------------------------------------------
``omnibase_infra#3503``, head ``fe5db385``. The OCC autobind stamped the PR body
at 20:53:00Z, which fired ``OCC Preflight Heal`` run ``34782126673``. Its one job
(``103790926965``) logged, at 20:53:06Z:

    Run 34782083407 is queued; a rerun is only valid on a completed run.
    Nothing to do.

and concluded ``success``. That is the whole defect in two lines: the heal
required its target CI run to already read ``completed``, the ~48-job matrix was
still queued a minute after the push, and the job that did nothing reported the
same colour as a job that had healed something. Nothing re-invoked it — a rerun
emits no ``edited`` event, and the run reaching ``completed`` at 21:12:08Z was
not an event the workflow listened to. ``occ-preflight / eligibility`` stayed
FAILURE from 20:52:21Z and ``CI Summary`` fail-closed behind it until a human ran
``gh run rerun --failed`` at 21:21:21Z.

WHY THE FIXTURES ARE CAPTURES AND NOT A RECONSTRUCTION
------------------------------------------------------
Every fact this replay decides on is asserted out of committed bytes BEFORE the
guard is driven, and every byte is re-fetchable:

* ``heal-job103790926965.log.gz.captured`` — the heal job's verbatim log
  (``gh api repos/OmniNode-ai/omnibase_infra/actions/jobs/103790926965/logs``,
  reproducible through ``gzip -nc``). This is the cited artifact: the surface
  that failed, saying what it did.
* ``heal-run34782126673.json.captured`` — the heal run. Its ``conclusion`` is
  the false green, and its ``created_at`` is the body edit that fired it, which
  is why this replay does not need to capture the PR body at all: the edit's own
  event timestamps the edit.
* ``ci-run34782083407-attempt1.json.captured`` and
  ``...-attempt1-jobs.json.captured`` — the CI run as it stood when it finished,
  carrying the failed ``occ-preflight / eligibility`` and its ``started_at``.
* ``...-attempt2-jobs.json.captured`` — the human rerun, which is the
  discriminator below.

WHAT THE REPLAY PROVES
----------------------
The guard is driven twice over the same incident, at the two moments that
matter, and the pair is the proof:

1. At 20:53:06Z, with the run in the state the log itself names, the guard says
   ``RUN_NOT_COMPLETED``. It still cannot heal there, and it never could — a
   single failed job cannot be re-run inside an unfinished run.
2. At 21:12:08Z, when the same run finished, the guard says ``RERUN_REQUIRED``
   on the same PR, the same stale verdict, the same everything.

So the fix is not a smarter guard, it is a second invocation point: the CI run's
own ``workflow_run: completed``. Step 2 is the rerun the human had to issue nine
minutes later, and the replay pins that the guard would have issued it.

THE DISCRIMINATOR IS MANDATORY, NOT A FORMALITY. A guard hardwired to
``RERUN_REQUIRED`` would replay this incident perfectly and then re-queue jobs on
every completion of every CI run in the repo forever — including the completion
of its own reruns, which is an unbounded loop, not a gate. The same guard is
therefore driven over attempt 2's real job list, where the preflight concluded
``success``, and is required to leave it alone.
"""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from scripts.ci.occ_preflight_heal import (
    EnumHealOutcome,
    decide_heal,
    earliest_failed_preflight_started_at_in_jobs,
    parse_github_timestamp,
    snapshot_from_run_payload,
)

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "omn18352"

HEAL_JOB_LOG = FIXTURES / "heal-job103790926965.log.gz.captured"
HEAL_RUN = FIXTURES / "heal-run34782126673.json.captured"
CI_RUN_ATTEMPT1 = FIXTURES / "ci-run34782083407-attempt1.json.captured"
CI_JOBS_ATTEMPT1 = FIXTURES / "ci-run34782083407-attempt1-jobs.json.captured"
CI_JOBS_ATTEMPT2 = FIXTURES / "ci-run34782083407-attempt2-jobs.json.captured"

CI_RUN_ID = 34782083407
PREFLIGHT_JOB_PREFIX = "occ-preflight"

# The line the shipped heal printed. Its exact wording is the incident.
NOTHING_TO_DO = (
    "Run 34782083407 is queued; a rerun is only valid on a completed run. "
    "Nothing to do."
)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _log_text() -> str:
    return gzip.decompress(HEAL_JOB_LOG.read_bytes()).decode("utf-8", "replace")


def _ts(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)


def test_the_real_guard_reruns_the_stale_verdict_the_shipped_heal_skipped() -> None:
    """Drive the real guard over the captured incident at both moments."""
    # --- facts, out of the captured bytes, before the guard is involved ---
    log = _log_text()
    assert NOTHING_TO_DO in log, (
        "the cited capture must contain the shipped heal's own refusal; without "
        "that line this fixture is not the incident"
    )

    heal_run = _json(HEAL_RUN)
    assert heal_run["conclusion"] == "success", (
        "the false green IS the incident: the heal that healed nothing reported "
        "the same colour as one that did"
    )
    assert heal_run["event"] == "pull_request"
    stamp_edit_at = parse_github_timestamp(heal_run["created_at"])
    assert stamp_edit_at == _ts("2026-09-13T20:53:00Z")

    ci_run_payload = _json(CI_RUN_ATTEMPT1)
    completed_run = snapshot_from_run_payload(ci_run_payload)
    assert completed_run is not None
    assert completed_run.run_id == CI_RUN_ID
    assert completed_run.status == "completed"
    assert completed_run.run_attempt == 1
    run_finished_at = parse_github_timestamp(ci_run_payload["updated_at"])
    assert run_finished_at == _ts("2026-09-13T21:12:08Z")

    preflight_started_at = earliest_failed_preflight_started_at_in_jobs(
        _json(CI_JOBS_ATTEMPT1), job_prefix=PREFLIGHT_JOB_PREFIX
    )
    assert preflight_started_at == _ts("2026-09-13T20:52:09Z"), (
        "attempt 1 must carry a FAILED occ-preflight job; that failing verdict "
        "is what CI Summary stayed fail-closed on"
    )

    # The ordering that makes this a race and not a missing trigger.
    assert preflight_started_at < stamp_edit_at < run_finished_at

    # --- moment 1: 20:53:06Z, the only moment the shipped heal was invoked ---
    at_edit_time = decide_heal(
        run=snapshot_from_run_payload({**ci_run_payload, "status": "queued"}),
        earliest_failed_preflight_started_at=preflight_started_at,
        pr_updated_at=stamp_edit_at,
    )
    assert at_edit_time.outcome is EnumHealOutcome.RUN_NOT_COMPLETED
    assert at_edit_time.rerun is False
    assert "queued" in at_edit_time.detail

    # --- moment 2: 21:12:08Z, the invocation point OMN-18352 adds ---
    at_completion = decide_heal(
        run=completed_run,
        earliest_failed_preflight_started_at=preflight_started_at,
        pr_updated_at=stamp_edit_at,
    )
    assert at_completion.outcome is EnumHealOutcome.RERUN_REQUIRED, (
        "on the real completed run with the real stale verdict the guard must "
        "issue the rerun a human issued at 21:21:21Z"
    )
    assert at_completion.rerun is True
    assert at_completion.run_id == CI_RUN_ID


def test_the_same_guard_leaves_the_healed_attempt_alone() -> None:
    """Discriminator: attempt 2 is real, and the guard must not touch it.

    ``gh run rerun --failed`` at 21:21:21Z produced attempt 2, whose
    ``occ-preflight / eligibility`` concluded ``success``. A guard that acted
    here would act on the completion of its own reruns, forever.
    """
    jobs_payload = _json(CI_JOBS_ATTEMPT2)
    preflight = [
        job
        for job in jobs_payload["jobs"]
        if str(job["name"]).startswith(PREFLIGHT_JOB_PREFIX)
    ]
    assert preflight, "attempt 2 must contain the preflight job"
    assert [job["conclusion"] for job in preflight] == ["success"], (
        "the discriminator is only a discriminator if attempt 2 really passed"
    )

    healed_start = earliest_failed_preflight_started_at_in_jobs(
        jobs_payload, job_prefix=PREFLIGHT_JOB_PREFIX
    )
    assert healed_start is None

    decision = decide_heal(
        run=snapshot_from_run_payload(
            {**_json(CI_RUN_ATTEMPT1), "status": "completed", "run_attempt": 2}
        ),
        earliest_failed_preflight_started_at=healed_start,
        pr_updated_at=_ts("2026-09-13T20:53:00Z"),
    )
    assert decision.outcome is EnumHealOutcome.NO_FAILED_PREFLIGHT
    assert decision.rerun is False


def test_a_second_completion_after_a_heal_that_still_fails_stops() -> None:
    """The other half of the loop bound, on the same real run.

    Suppose attempt 2's preflight had failed again. Its job started at
    21:21:26Z — after the body edit — so the verdict describes the current PR
    and the guard stops rather than re-running a third time. The start time is
    read from the captured attempt-2 job list, not invented.
    """
    jobs_payload = _json(CI_JOBS_ATTEMPT2)
    attempt2_start = parse_github_timestamp(
        next(
            job["started_at"]
            for job in jobs_payload["jobs"]
            if str(job["name"]).startswith(PREFLIGHT_JOB_PREFIX)
        )
    )
    assert attempt2_start == _ts("2026-09-13T21:21:26Z")

    decision = decide_heal(
        run=snapshot_from_run_payload(
            {**_json(CI_RUN_ATTEMPT1), "status": "completed", "run_attempt": 2}
        ),
        earliest_failed_preflight_started_at=attempt2_start,
        pr_updated_at=_ts("2026-09-13T20:53:00Z"),
    )
    assert decision.outcome is EnumHealOutcome.VERDICT_NOT_STALE
    assert decision.rerun is False
