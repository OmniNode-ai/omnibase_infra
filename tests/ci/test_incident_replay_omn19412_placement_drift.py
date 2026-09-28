# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Incident replay: C13's placement moved live while the committed record still named the old pool (OMN-19412).

THE INCIDENT. omninode_infra#1725 (merged 2026-09-28T05:28:12Z) placed C13, C14
and C29 with ``runs-on: ${{ fromJSON(vars.CUSTOMER_MACHINE_RUNS_ON_JSON) }}``
and the live variable put them on the ``omnibase-verify`` pool, while the
committed record in omnibase_infra still said ``omnipc2-customer``. Nothing
compared the live placement with the committed one before it mattered, and the
disagreement surfaced as a fleet-wide red on the required Lab Probe Windows
check (about 55 minutes, FRICTION 2026-09-28T07:03:41Z lane
lab-probe-windows-fix-83). This is the class the drift probe exists for: a live
placement and its committed value disagreeing with no surface saying so
(false_green of every surface in place at the time).

THE GUARD is ``scripts/ci/check_probe_placement_drift.py``. Since the OMN-19412
follow-up the Lab Probe Windows check resolves placements from the committed
``probe_placement_variables`` map instead of reading live variables with the
operator's CROSS_REPO_PAT, and this probe keeps that map equal to live.

THE ARTIFACT is the C13 job of the first run after #1725 (run 36386308491, job
108812316933), exactly as the Actions API returned it. Its ``labels`` are the
evaluated ``fromJSON(vars.CUSTOMER_MACHINE_RUNS_ON_JSON)`` (the expression has
no fallback), i.e. the variable's live value as the runner resolved it. Driven
with the committed placement recorded before the fix (the customer machine),
the guard must refuse, naming the variable and both values. THE DISCRIMINATOR
drives the same bytes with the committed value equal to the live one and
requires acceptance, so a guard stuck red cannot pass.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts.ci import check_probe_placement_drift as probe_drift

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests/fixtures/omn19412/c13-job-108812316933.json.captured"
FIXTURE_SHA256 = "293108ebaa450a4c61ac562d7adc2e5766ef8f971539c7b884c620d992e8997e"
NAME = "CUSTOMER_MACHINE_RUNS_ON_JSON"
# The placement the committed record named before omnibase_infra#4241.
COMMITTED_BEFORE_THE_FIX = '["self-hosted","omnipc2-customer"]'


def _job() -> dict[str, object]:
    job = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(job, dict)
    return job


def _live_vars(tmp_path: Path) -> Path:
    """The run's vars context for this variable: the labels the runner resolved."""
    path = tmp_path / "vars.json"
    path.write_text(json.dumps({NAME: json.dumps(_job()["labels"])}), encoding="utf-8")
    return path


def _run(tmp_path: Path, committed: str | None) -> int:
    policy = tmp_path / "runner_routing_policy.yaml"
    policy.write_text(
        yaml.safe_dump(
            {"probe_placement_variables": {"omninode_infra": {NAME: committed}}}
        ),
        encoding="utf-8",
    )
    return probe_drift.main(
        [
            "--repo",
            "omninode_infra",
            "--vars-json-file",
            str(_live_vars(tmp_path)),
            "--policy",
            str(policy),
        ]
    )


@pytest.mark.unit
def test_the_fixture_is_the_captured_bytes() -> None:
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256


@pytest.mark.unit
def test_the_captured_job_is_the_c13_run_placed_on_the_verify_pool() -> None:
    """Not vacuous: the captured job is C13 and it ran on the moved placement."""
    job = _job()
    assert job["run_id"] == 36386308491
    assert str(job["workflow_name"]).startswith("C13 ")
    assert job["labels"] == ["self-hosted", "omnibase-verify"]
    assert job["conclusion"] == "success"


@pytest.mark.unit
def test_the_real_guard_refuses_the_moved_placement_against_the_old_committed_value(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(tmp_path, COMMITTED_BEFORE_THE_FIX) == probe_drift.EXIT_DRIFT
    err = capsys.readouterr().err
    assert f"vars.{NAME}" in err
    assert "omnipc2-customer" in err
    assert "omnibase-verify" in err


@pytest.mark.unit
def test_the_same_guard_accepts_the_moved_placement_once_it_is_committed(
    tmp_path: Path,
) -> None:
    assert _run(tmp_path, '["self-hosted","omnibase-verify"]') == probe_drift.EXIT_OK
