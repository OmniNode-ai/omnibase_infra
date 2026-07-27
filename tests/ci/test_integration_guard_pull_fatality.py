# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-15249 — the Integration Silent-Skip Guard must die AT the image pull.

Live defect (omnibase_infra#2492, heads ``1729c7c3`` / ``dbfc0c98``, job
``89995662337``). The ``integration-guard`` job provisioned Postgres through a
GitHub-managed ``services:`` block. ``registry-1.docker.io`` timed out, and the
run looked like this::

    ##[command]/usr/bin/docker pull postgres:16-alpine
    Error response from daemon: Get "https://registry-1.docker.io/v2/": context deadline exceeded
    ##[warning]Docker pull failed with exit code 1, back off 8.077 seconds before retry.
    ...
    ##[error]Docker pull failed with exit code 1
    ... (checkout, setup-python-uv, migrations, curated proofs: ALL skipped) ...
    /home/runner/work/_temp/....sh: line 1: uv: command not found
    ##[error]Process completed with exit code 127.

Two defects, both fixed here:

1. **The guard emitted a verdict for a run whose container never materialized.**
   ``Enforce no missing-service silent-skips`` carried a bare ``if: always()``,
   so it fired past a failed container init. Its toolchain had been skipped, so
   it died ``exit 127``; had ``uv`` survived, ``check_integration_skips.py``
   would have exited 2 on the missing JUnit report — either way a verdict from a
   run that never provisioned Postgres.
2. **The terminal signal was misattributed.** ``exit 127`` is the LAST error in
   the log, several steps removed from the registry timeout that caused it, so
   triage lands on a phantom missing-binary problem and reaches for "transient".

The fix moves the pull out of the GitHub-managed ``services:`` block and into an
explicit, bounded-retry, fail-closed step this repo owns, so the pull failure is
fatal *at the pull*, names registry/image/timeout, and every downstream step
(including the verdict) is unreachable.

Test posture: the pull script is executed for real as a bash subprocess against
a stubbed ``docker`` on ``PATH`` (a simulated registry timeout, not a source-text
grep), and the step graph is replayed through a GitHub-``if``-semantics
simulator that fails closed on any condition form it does not understand.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

GUARD_JOB_ID = "integration-guard"
PULL_STEP_ID = "pull_postgres"
START_STEP_ID = "start_postgres"
PROOFS_STEP_ID = "run_curated_proofs"
ENFORCE_STEP_ID = "enforce_no_silent_skips"
UPLOAD_STEP_ID = "upload_guard_results"
CLEANUP_STEP_ID = "stop_postgres"

# Tokens whose presence in an executed step's shell body means that step depends
# on a toolchain the pull/setup steps install. Reaching one of these after a
# failed pull is exactly how the live run produced ``exit 127``.
_TOOLCHAIN_TOKENS = ("uv ", "uv\n", "pytest", "python ", "python\n")


def _load_ci_workflow() -> dict[str, Any]:
    loaded = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _guard_job() -> dict[str, Any]:
    workflow = _load_ci_workflow()
    jobs = workflow["jobs"]
    assert GUARD_JOB_ID in jobs, (
        f"ci.yml no longer defines the `{GUARD_JOB_ID}` job — OMN-14172's guard "
        "is the surface OMN-15249 hardens."
    )
    job = jobs[GUARD_JOB_ID]
    assert isinstance(job, dict)
    return job


def _steps() -> list[dict[str, Any]]:
    steps = _guard_job()["steps"]
    assert isinstance(steps, list)
    return [step for step in steps if isinstance(step, dict)]


def _step_by_id(step_id: str) -> dict[str, Any]:
    for step in _steps():
        if step.get("id") == step_id:
            return step
    raise AssertionError(
        f"`{GUARD_JOB_ID}` has no step with id `{step_id}`. OMN-15249 requires "
        "the Postgres pull/start/proof/verdict steps to be individually "
        "addressable so the verdict can be gated on the proof having run."
    )


# ---------------------------------------------------------------------------
# 1. The pull must be a step this repo owns.
# ---------------------------------------------------------------------------


def test_guard_does_not_delegate_the_pull_to_a_services_block() -> None:
    """A ``services:`` image pull happens in GitHub's ``Initialize containers``.

    That step is not addressable from the workflow: its failure cannot be given
    a named message, its retry budget cannot be bounded by this repo, and — as
    #2492 proved — steps carrying ``if: always()`` still fire past it. Owning the
    pull is what makes every other assertion in this module enforceable.
    """
    job = _guard_job()
    assert "services" not in job, (
        "`integration-guard` still provisions Postgres via a GitHub-managed "
        "`services:` block. OMN-15249 requires an explicit, fail-closed pull "
        "step so a registry timeout terminates the job at the pull with a named "
        "error instead of surfacing as a downstream `exit 127`."
    )


def test_pull_is_the_first_step_and_cannot_be_softened() -> None:
    steps = _steps()
    assert steps[0].get("id") == PULL_STEP_ID, (
        "The Postgres pull must be the FIRST step: nothing after a failed pull "
        f"can be trusted. Got first step id={steps[0].get('id')!r}."
    )
    for step_id in (PULL_STEP_ID, START_STEP_ID):
        step = _step_by_id(step_id)
        assert "continue-on-error" not in step, (
            f"step `{step_id}` sets continue-on-error — the OMN-15249 defect "
            "class is precisely 'container failure downgraded to a warning'."
        )


def test_pull_retry_budget_is_bounded_and_declared() -> None:
    job_env = _guard_job()["env"]
    attempts = int(str(job_env["GUARD_PG_PULL_ATTEMPTS"]))
    timeout_seconds = int(str(job_env["GUARD_PG_PULL_TIMEOUT_SECONDS"]))
    assert 1 <= attempts <= 5, (
        f"pull attempts must be bounded and small; got {attempts}. An unbounded "
        "retry re-creates the warn-and-continue failure mode as a hang."
    )
    assert 0 < timeout_seconds <= 600, (
        f"per-attempt pull timeout must be bounded; got {timeout_seconds}."
    )
    assert str(job_env["GUARD_PG_IMAGE"]) == "postgres:16-alpine"
    assert str(job_env["GUARD_PG_REGISTRY"]) == "registry-1.docker.io"


# ---------------------------------------------------------------------------
# 2. Behavioral: run the real pull script against a simulated registry timeout.
# ---------------------------------------------------------------------------


def _write_docker_stub(bin_dir: Path, *, exit_code: int, message: str) -> Path:
    """Install a ``docker`` stub on PATH that records every invocation."""
    attempt_log = bin_dir / "attempts.log"
    stub = bin_dir / "docker"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$@" >> "{attempt_log}"\n'
        f'echo "{message}" >&2\n'
        f"exit {exit_code}\n",
        encoding="utf-8",
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return attempt_log


def _run_pull_script(tmp_path: Path, *, docker_exit_code: int) -> tuple[int, str, int]:
    """Execute the workflow's real pull ``run:`` body; return (rc, output, attempts)."""
    job_env = _guard_job()["env"]
    script = tmp_path / "pull_step.sh"
    script.write_text(_step_by_id(PULL_STEP_ID)["run"], encoding="utf-8")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    attempt_log = _write_docker_stub(
        bin_dir,
        exit_code=docker_exit_code,
        message=(
            'Error response from daemon: Get "https://registry-1.docker.io/v2/": '
            "context deadline exceeded (Client.Timeout exceeded while awaiting headers)"
        ),
    )

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    # Take the real configured values from the workflow, except the inter-attempt
    # backoff: boundedness is the property under test, wall-clock sleep is not.
    for key in (
        "GUARD_PG_IMAGE",
        "GUARD_PG_REGISTRY",
        "GUARD_PG_PULL_ATTEMPTS",
        "GUARD_PG_PULL_TIMEOUT_SECONDS",
    ):
        env[key] = str(job_env[key])
    env["GUARD_PG_PULL_BACKOFF_SECONDS"] = "0"
    env["GITHUB_OUTPUT"] = str(tmp_path / "github_output")

    completed = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    attempts = (
        len(
            [
                line
                for line in attempt_log.read_text(encoding="utf-8").splitlines()
                if line
            ]
        )
        if attempt_log.exists()
        else 0
    )
    return completed.returncode, completed.stdout + completed.stderr, attempts


def test_simulated_registry_timeout_fails_the_pull_step_with_a_named_error(
    tmp_path: Path,
) -> None:
    """DoD-1: pull failure terminates AT the pull, naming registry/image/timeout."""
    job_env = _guard_job()["env"]
    rc, output, attempts = _run_pull_script(tmp_path, docker_exit_code=1)

    assert rc != 0, (
        "A docker pull that never succeeds must fail the step. Warning-and-"
        "continuing is the OMN-15249 defect."
    )
    assert attempts == int(str(job_env["GUARD_PG_PULL_ATTEMPTS"])), (
        f"expected exactly {job_env['GUARD_PG_PULL_ATTEMPTS']} bounded pull "
        f"attempts, observed {attempts} — retry must fail closed on exhaustion."
    )
    assert "::error" in output, "the terminal failure must be a GitHub error annotation"
    for token in (
        str(job_env["GUARD_PG_IMAGE"]),
        str(job_env["GUARD_PG_REGISTRY"]),
        str(job_env["GUARD_PG_PULL_TIMEOUT_SECONDS"]),
    ):
        assert token in output, (
            f"the failure message must name {token!r} so triage lands on the "
            "registry pull instead of a phantom missing-binary problem"
        )


def test_pull_step_succeeds_and_stops_retrying_when_the_registry_answers(
    tmp_path: Path,
) -> None:
    """The fail-closed loop must not be a permanently-red check."""
    rc, output, attempts = _run_pull_script(tmp_path, docker_exit_code=0)
    assert rc == 0, f"a successful pull must exit 0; output:\n{output}"
    assert attempts == 1, f"a successful pull must not retry; observed {attempts}"


# ---------------------------------------------------------------------------
# 3. Step-graph replay: nothing toolchain-dependent is reachable after a bad pull.
# ---------------------------------------------------------------------------

_ALWAYS_ONLY = re.compile(r"^always\(\)$")
_ALWAYS_AND_NOT_SKIPPED = re.compile(
    r"^always\(\)\s*&&\s*steps\.(?P<step>[A-Za-z0-9_\-]+)\.conclusion\s*!=\s*'skipped'$"
)


def _evaluate_step_if(condition: str, conclusions: dict[str, str]) -> bool:
    """Evaluate the ``if:`` forms this job is allowed to use. Fail closed."""
    condition = condition.strip()
    if _ALWAYS_ONLY.match(condition):
        return True
    match = _ALWAYS_AND_NOT_SKIPPED.match(condition)
    if match:
        return conclusions.get(match.group("step"), "skipped") != "skipped"
    pytest.fail(
        f"`{GUARD_JOB_ID}` uses an unrecognised step `if:` form: {condition!r}. "
        "OMN-15249 gates the guard verdict on step conclusions; an unmodelled "
        "condition could silently re-open the warn-and-continue path, so this "
        "simulator fails closed rather than guessing."
    )


def _simulate(failing_step_id: str | None) -> tuple[list[str], dict[str, str]]:
    """Replay GitHub's step-execution semantics; return (executed ids, conclusions)."""
    executed: list[str] = []
    conclusions: dict[str, str] = {}
    job_failed = False
    for index, step in enumerate(_steps()):
        step_id = step.get("id") or f"__unnamed_{index}"
        condition = step.get("if")
        runs = (
            (not job_failed)
            if condition is None
            else _evaluate_step_if(str(condition), conclusions)
        )
        if not runs:
            conclusions[step_id] = "skipped"
            continue
        executed.append(step_id)
        if step_id == failing_step_id:
            conclusions[step_id] = "failure"
            job_failed = True
        else:
            conclusions[step_id] = "success"
    return executed, conclusions


def test_no_toolchain_step_is_reachable_after_a_failed_pull() -> None:
    """DoD-2: no ``exit 127`` downstream is reachable from a failed pull."""
    executed, conclusions = _simulate(failing_step_id=PULL_STEP_ID)

    assert executed[0] == PULL_STEP_ID
    assert conclusions[ENFORCE_STEP_ID] == "skipped", (
        "the silent-skip verdict still runs after a failed pull — this is the "
        "exact `if: always()` path that produced `uv: command not found` / "
        "exit 127 on omnibase_infra#2492."
    )
    assert conclusions[UPLOAD_STEP_ID] == "skipped", (
        "artifact upload still runs after a failed pull and warns 'No files were "
        "found', adding noise to an already-misattributed failure."
    )
    assert CLEANUP_STEP_ID in executed, (
        "container cleanup must still run after a failed pull so a partially "
        "started container is never leaked."
    )

    by_id = {step.get("id"): step for step in _steps()}
    for step_id in executed:
        if step_id in (PULL_STEP_ID, CLEANUP_STEP_ID):
            continue
        body = str(by_id.get(step_id, {}).get("run", ""))
        for token in _TOOLCHAIN_TOKENS:
            assert token not in body, (
                f"step `{step_id}` runs after a failed pull and invokes "
                f"{token.strip()!r}; the toolchain that provides it is installed "
                "by a step that a failed pull skips, so this is a latent "
                "exit-127 misattribution."
            )


def test_verdict_still_runs_when_the_curated_proofs_actually_fail() -> None:
    """The OMN-14172 guard must keep firing on real integration FAILURES."""
    _, conclusions = _simulate(failing_step_id=PROOFS_STEP_ID)
    assert conclusions[ENFORCE_STEP_ID] == "success", (
        "gating the verdict on container materialization must not disable it on "
        "a genuine test failure — that would trade one false-green for another."
    )
    assert conclusions[UPLOAD_STEP_ID] == "success"


def test_verdict_runs_on_the_fully_healthy_path() -> None:
    executed, conclusions = _simulate(failing_step_id=None)
    assert conclusions[ENFORCE_STEP_ID] == "success"
    for step_id in (PULL_STEP_ID, START_STEP_ID, PROOFS_STEP_ID, CLEANUP_STEP_ID):
        assert step_id in executed
