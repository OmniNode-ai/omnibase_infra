# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the OMN-15776 broker-dispatch-wedge detection + targeted rerun.

Proven mechanism (2026-08-09, ``omn15776-wedge`` ledger entry): GitHub's Actions
broker dispatches a job to a self-hosted runner within 2-7s of that same runner
finishing its previous job, exactly while the runner's ``Runner.Listener`` is
mid-reconnect on its broker long-poll (every job completion triggers a
retry/backoff storm on that connection). The new dispatch lands in the
reconnect gap and is never delivered to the runner's local message loop — no
``Runner.Worker`` process is ever spawned (``steps: []``, not a crashed step 1)
— while GitHub's server side records the assignment, sets ``started_at``, and
independently times the orphaned assignment out at a fixed ~10m0-1s.

No local entrypoint.sh/watchdog fix applies: the drop happens in the GitHub
Actions client/broker protocol path, before any local process (Worker, or the
existing OMN-14564 heartbeat watchdog, which only inspects a process that never
spawned) can observe it. The bounded remediation is a targeted, signature-keyed
rerun: detect jobs matching the exact structural fingerprint (self-hosted
runner assigned, conclusion in {failure, cancelled}, zero steps recorded,
duration within a tight band around the fixed ~10m0-1s server-side timeout) and
reissue only that job — never a blanket retry-on-red policy, which would
launder genuine content failures.

This test proves the DETECTION GAP first (RED): feeding the exact wedge
signature through the script with no detection logic present must not yet
distinguish it from other failures. Then verifies the fix distinguishes:
  - a wedge-signature job -> flagged as a rerun candidate (and, when not
    dry-run, the job-rerun endpoint is called exactly once for it)
  - a genuine content failure on a self-hosted runner (has steps, short
    duration) -> NEVER a candidate
  - a GitHub-hosted job (no runner_name) with the same duration/steps shape
    -> NEVER a candidate (this fleet's remediation must not touch hosted jobs)
  - a successful job -> NEVER a candidate
"""

from __future__ import annotations

import http.server
import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from urllib.parse import urlparse

import pytest

pytestmark = pytest.mark.ci

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "runner_broker_dispatch_wedge_rerun.sh"

ORG = "OmniNode-ai"
REPO = "omnibase_infra"
RUN_ID = 90000000001

# The exact proven wedge signature (2026-08-09 ledger evidence): runner
# assigned, conclusion failure, zero steps, duration exactly 600s (10m0s).
WEDGE_JOB = {
    "id": 93294479341,
    "run_id": RUN_ID,
    "name": "Dep Provenance Gate",
    "status": "completed",
    "conclusion": "failure",
    "runner_name": "omninode-runner-17",
    "started_at": "2026-08-08T20:06:49Z",
    "completed_at": "2026-08-08T20:16:49Z",
    "steps": [],
}

# A genuine content failure on the same fleet: has real steps, short duration.
# Must NEVER be treated as a rerun candidate.
REAL_FAILURE_JOB = {
    "id": 93294479999,
    "run_id": RUN_ID,
    "name": "Unit Tests",
    "status": "completed",
    "conclusion": "failure",
    "runner_name": "omninode-runner-9",
    "started_at": "2026-08-08T20:00:00Z",
    "completed_at": "2026-08-08T20:00:45Z",
    "steps": [{"name": "Run pytest", "conclusion": "failure"}],
}

# Same duration/steps shape as the wedge signature, but a GitHub-hosted job
# (no runner_name). Must NEVER be treated as a rerun candidate.
HOSTED_JOB_SAME_SHAPE = {
    "id": 93294480000,
    "run_id": RUN_ID,
    "name": "hosted-timeout-job",
    "status": "completed",
    "conclusion": "cancelled",
    "runner_name": None,
    "started_at": "2026-08-08T20:06:49Z",
    "completed_at": "2026-08-08T20:16:49Z",
    "steps": [],
}

SUCCESS_JOB = {
    "id": 93294480001,
    "run_id": RUN_ID,
    "name": "Lint",
    "status": "completed",
    "conclusion": "success",
    "runner_name": "omninode-runner-3",
    "started_at": "2026-08-08T20:00:00Z",
    "completed_at": "2026-08-08T20:02:00Z",
    "steps": [{"name": "ruff", "conclusion": "success"}],
}


class _StubState:
    """Records what the stub server observed (thread-shared)."""

    def __init__(self, jobs: list[dict[str, object]]) -> None:
        self.jobs = jobs
        self.rerun_job_ids: list[int] = []
        self.lock = threading.Lock()


def _make_handler(state: _StubState) -> type[http.server.BaseHTTPRequestHandler]:
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:  # silence request logging
            pass

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            if path == f"/repos/{ORG}/{REPO}/actions/runs":
                body = {
                    "workflow_runs": [
                        {
                            "id": RUN_ID,
                            "status": "completed",
                            "created_at": "2026-08-08T20:00:00Z",
                        }
                    ]
                }
            elif path == f"/repos/{ORG}/{REPO}/actions/runs/{RUN_ID}/jobs":
                body = {"jobs": state.jobs}
            else:
                self.send_response(404)
                self.end_headers()
                return
            payload = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            prefix = f"/repos/{ORG}/{REPO}/actions/jobs/"
            suffix = "/rerun"
            if parsed.path.startswith(prefix) and parsed.path.endswith(suffix):
                job_id = int(parsed.path[len(prefix) : -len(suffix)])
                with state.lock:
                    state.rerun_job_ids.append(job_id)
                self.send_response(201)
                self.end_headers()
                return
            self.send_response(404)
            self.end_headers()

    return Handler


def _run_script(
    jobs: list[dict[str, object]], *, dry_run: bool
) -> tuple[subprocess.CompletedProcess[str], _StubState]:
    state = _StubState(jobs)
    handler = _make_handler(state)
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        env = dict(os.environ)
        env.update(
            {
                "GITHUB_API_URL": f"http://127.0.0.1:{server.server_port}",
                "RUNNER_GITHUB_TOKEN": "test-token",
                "REPOS_CSV": REPO,
                "GITHUB_ORG": ORG,
                # Fixture timestamps are fixed historical values (2026-08-08);
                # use a huge lookback so the test's pass/fail is independent
                # of wall-clock time at execution — the lookback window itself
                # is not what this test suite is verifying.
                "LOOKBACK_HOURS": "87600",
                "ONEX_STATE_DIR": "",  # forced per-invocation via --state-dir below
            }
        )
        with tempfile.TemporaryDirectory(prefix="omn15776-test-state-") as state_dir:
            args = ["bash", str(SCRIPT), "--state-dir", state_dir]
            if dry_run:
                args.append("--dry-run")
            result = subprocess.run(
                args,
                check=False,
                capture_output=True,
                text=True,
                env=env,
                cwd=REPO_ROOT,
                timeout=60,
            )
        return result, state
    finally:
        server.shutdown()
        thread.join(timeout=5)


class TestBrokerDispatchWedgeSignatureDetection:
    """DoD: the exact proven signature is distinguished from every neighbor."""

    def test_script_exists_and_is_executable_shape(self) -> None:
        assert SCRIPT.exists(), (
            f"missing {SCRIPT} — OMN-15776 remediation not implemented"
        )

    def test_wedge_signature_flagged_as_candidate_dry_run(self, tmp_path: Path) -> None:
        jobs = [WEDGE_JOB, REAL_FAILURE_JOB, HOSTED_JOB_SAME_SHAPE, SUCCESS_JOB]
        result, state = _run_script(jobs, dry_run=True)
        assert result.returncode == 0, (
            f"dry-run scan must exit 0: out={result.stdout} err={result.stderr}"
        )
        assert str(WEDGE_JOB["id"]) in result.stdout, (
            f"wedge-signature job {WEDGE_JOB['id']} must be flagged as a "
            f"candidate: out={result.stdout}"
        )
        assert str(REAL_FAILURE_JOB["id"]) not in result.stdout, (
            "a genuine content failure (has steps) must never be a candidate"
        )
        assert str(HOSTED_JOB_SAME_SHAPE["id"]) not in result.stdout, (
            "a GitHub-hosted job must never be a candidate regardless of shape"
        )
        assert str(SUCCESS_JOB["id"]) not in result.stdout
        # dry-run must never call the rerun endpoint
        assert state.rerun_job_ids == []

    def test_wedge_signature_reruns_exactly_the_matched_job(self) -> None:
        jobs = [WEDGE_JOB, REAL_FAILURE_JOB, HOSTED_JOB_SAME_SHAPE, SUCCESS_JOB]
        result, state = _run_script(jobs, dry_run=False)
        assert result.returncode == 0, (
            f"live scan must exit 0: out={result.stdout} err={result.stderr}"
        )
        assert state.rerun_job_ids == [WEDGE_JOB["id"]], (
            f"exactly the wedge-signature job must be rerun, got "
            f"{state.rerun_job_ids}: out={result.stdout}"
        )

    def test_no_candidates_is_a_clean_noop(self) -> None:
        jobs = [REAL_FAILURE_JOB, HOSTED_JOB_SAME_SHAPE, SUCCESS_JOB]
        result, state = _run_script(jobs, dry_run=False)
        assert result.returncode == 0
        assert state.rerun_job_ids == []

    def test_off_by_one_duration_below_band_is_not_a_candidate(self) -> None:
        """duration=540s (9m) must not match — it is not the proven ~10m0-1s
        fixed server-side timeout, so treating it as the same signature would
        risk masking a genuine early failure."""
        near_miss = dict(WEDGE_JOB)
        near_miss["id"] = 93294481111
        near_miss["completed_at"] = "2026-08-08T20:15:49Z"  # 540s after started_at
        result, state = _run_script([near_miss], dry_run=True)
        assert result.returncode == 0
        assert str(near_miss["id"]) not in result.stdout
        assert state.rerun_job_ids == []

    def test_cancelled_conclusion_also_matches_the_signature(self) -> None:
        """Ledger evidence records both failure and cancelled conclusions for
        this class (occ#6122/#6161 runs)."""
        cancelled = dict(WEDGE_JOB)
        cancelled["id"] = 93294482222
        cancelled["conclusion"] = "cancelled"
        result, state = _run_script([cancelled], dry_run=False)
        assert result.returncode == 0
        assert state.rerun_job_ids == [cancelled["id"]]


# ---------------------------------------------------------------------------
# Incident replay (OMN-15547): the guard driven against REAL captured bytes,
# not a hand-typed synthetic fixture. Fetched verbatim via
#   gh api repos/OmniNode-ai/onex_change_control/actions/jobs/93294479341
# on 2026-08-09 (the exact job cited in the omn15776-wedge ledger proof:
# runner=omninode-runner-17, conclusion=failure, steps=[],
# started_at=20:06:56Z, completed_at=20:16:56Z — exactly 600s). Committed
# byte-identical at tests/fixtures/omn15776/job-93294479341.json.captured;
# see tests/incident_replays/registry.yaml (id: omn15776-broker-dispatch-wedge)
# for the sha256 pin and capture provenance.
# ---------------------------------------------------------------------------

REAL_ORG = "OmniNode-ai"
REAL_REPO = "onex_change_control"
CAPTURED_JOB_FIXTURE = (
    REPO_ROOT / "tests" / "fixtures" / "omn15776" / "job-93294479341.json.captured"
)


def _run_script_against_real_org_repo(
    jobs: list[dict[str, object]], *, dry_run: bool, run_id: int
) -> tuple[subprocess.CompletedProcess[str], _StubState]:
    """Same driver as ``_run_script``, but keyed to the REAL org/repo/run_id
    the captured fixture came from — required so the guard's endpoint
    construction matches what actually served the real incident."""
    state = _StubState(jobs)
    base_handler = _make_handler(state)

    class RealRepoHandler(base_handler):  # type: ignore[misc, valid-type]
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            if path == f"/repos/{REAL_ORG}/{REAL_REPO}/actions/runs":
                body = {
                    "workflow_runs": [
                        {
                            "id": run_id,
                            "status": "completed",
                            "created_at": "2026-08-09T20:00:00Z",
                        }
                    ]
                }
            elif path == f"/repos/{REAL_ORG}/{REAL_REPO}/actions/runs/{run_id}/jobs":
                body = {"jobs": state.jobs}
            else:
                self.send_response(404)
                self.end_headers()
                return
            payload = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            prefix = f"/repos/{REAL_ORG}/{REAL_REPO}/actions/jobs/"
            suffix = "/rerun"
            if parsed.path.startswith(prefix) and parsed.path.endswith(suffix):
                job_id = int(parsed.path[len(prefix) : -len(suffix)])
                with state.lock:
                    state.rerun_job_ids.append(job_id)
                self.send_response(201)
                self.end_headers()
                return
            self.send_response(404)
            self.end_headers()

    server = http.server.HTTPServer(("127.0.0.1", 0), RealRepoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        env = dict(os.environ)
        env.update(
            {
                "GITHUB_API_URL": f"http://127.0.0.1:{server.server_port}",
                "RUNNER_GITHUB_TOKEN": "test-token",
                "REPOS_CSV": REAL_REPO,
                "GITHUB_ORG": REAL_ORG,
                "LOOKBACK_HOURS": "87600",
            }
        )
        with tempfile.TemporaryDirectory(prefix="omn15776-replay-state-") as state_dir:
            args = ["bash", str(SCRIPT), "--state-dir", state_dir]
            if dry_run:
                args.append("--dry-run")
            result = subprocess.run(
                args,
                check=False,
                capture_output=True,
                text=True,
                env=env,
                cwd=REPO_ROOT,
                timeout=60,
            )
        return result, state
    finally:
        server.shutdown()
        thread.join(timeout=5)


class TestOmn15776IncidentReplay:
    """R1-R5 (OMN-15547): the guard driven against the REAL captured job that
    motivated it, not a hand-typed reconstruction."""

    def test_fixture_is_the_real_captured_bytes(self) -> None:
        import hashlib

        assert CAPTURED_JOB_FIXTURE.exists(), (
            f"missing captured fixture {CAPTURED_JOB_FIXTURE}"
        )
        digest = hashlib.sha256(CAPTURED_JOB_FIXTURE.read_bytes()).hexdigest()
        assert digest == (
            "5f4305bd74e2d7ee2ceb9d09e9861d1f8d04cdc198150c64aa2befff8ff0c0bf"
        ), "captured fixture bytes drifted from the sha256 pinned in registry.yaml"

    def test_real_wedge_job_is_rejected_ie_flagged_for_rerun(self) -> None:
        """The buggy prior state (no guard at all) silently accepted this real
        orphaned dispatch as an unremediated red — a human had to notice and
        manually rerun it (see the multiple 2026-08-09 ledger passes that
        diagnosed but could not act on this exact job). The guard must REJECT
        it: flag it as a rerun candidate and reissue it."""
        captured_job = json.loads(CAPTURED_JOB_FIXTURE.read_text(encoding="utf-8"))
        run_id = captured_job["run_id"]
        result, state = _run_script_against_real_org_repo(
            [captured_job], dry_run=False, run_id=run_id
        )
        assert result.returncode == 0, (
            f"replay scan must exit 0: out={result.stdout} err={result.stderr}"
        )
        assert state.rerun_job_ids == [captured_job["id"]], (
            f"the real captured wedge job must be rejected (flagged + rerun): "
            f"out={result.stdout} err={result.stderr}"
        )
