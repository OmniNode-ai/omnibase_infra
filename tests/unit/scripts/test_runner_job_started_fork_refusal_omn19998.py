# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The job-started hook refuses a fork's code (OMN-19998).

The function is run from the hook file itself, in bash, against the event
payloads GitHub writes. The failing controls are the fork payloads: each must end
with a non-zero exit before any job step would run.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
HOOK = REPO_ROOT / "docker" / "runners" / "runner-job-started.sh"
BASE = "OmniNode-ai/omnimarket"
FORK = "outsider/omnimarket"


def _function() -> str:
    text = HOOK.read_text(encoding="utf-8")
    found = re.search(
        r"^# >>> fork refusal.*?^# <<< fork refusal", text, re.DOTALL | re.MULTILINE
    )
    assert found is not None
    return found.group(0)


def _run(
    tmp_path: Path,
    payload: dict[str, object] | None,
    *,
    ref: str = "refs/heads/dev",
    path: str = "",
) -> subprocess.CompletedProcess[str]:
    event = tmp_path / "event.json"
    if payload is not None:
        event.write_text(json.dumps(payload), encoding="utf-8")
    script = f"set -euo pipefail\n{_function()}\n"
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
            "GITHUB_EVENT_PATH": path or str(event),
            "GITHUB_REF": ref,
        },
    )


def _pull_request(head: str | None, base: str = BASE) -> dict[str, object]:
    head_repo = {"full_name": head} if head else None
    return {
        "pull_request": {
            "head": {"repo": head_repo},
            "base": {"repo": {"full_name": base}},
        },
        "repository": {"full_name": base},
    }


needs_jq = pytest.mark.skipif(
    shutil.which("jq") is None, reason="jq is on every runner"
)


@needs_jq
def test_failing_control_a_fork_pull_request_is_refused(tmp_path: Path) -> None:
    done = _run(tmp_path, _pull_request(FORK), ref="refs/pull/7/merge")
    assert done.returncode == 1
    assert "::error::OMN-19998" in done.stdout
    assert FORK in done.stdout


@needs_jq
def test_failing_control_a_deleted_fork_is_refused(tmp_path: Path) -> None:
    done = _run(tmp_path, _pull_request(None), ref="refs/pull/7/merge")
    assert done.returncode == 1
    assert "deleted" in done.stdout


@needs_jq
def test_failing_control_a_run_from_a_fork_workflow_is_refused(tmp_path: Path) -> None:
    payload = {
        "workflow_run": {"head_repository": {"full_name": FORK}},
        "repository": {"full_name": BASE},
    }
    assert _run(tmp_path, payload).returncode == 1


def test_failing_control_an_unreadable_payload_on_a_pull_ref_is_refused(
    tmp_path: Path,
) -> None:
    done = _run(tmp_path, None, ref="refs/pull/7/merge")
    assert done.returncode == 1
    assert "unproven" in done.stdout


@needs_jq
def test_a_same_repository_pull_request_passes(tmp_path: Path) -> None:
    assert _run(tmp_path, _pull_request(BASE), ref="refs/pull/7/merge").returncode == 0


@needs_jq
@pytest.mark.parametrize(
    "payload",
    [
        {"ref": "refs/heads/dev", "repository": {"full_name": BASE}},
        {"schedule": "0 * * * *", "repository": {"full_name": BASE}},
        {"merge_group": {"head_sha": "a"}, "repository": {"full_name": BASE}},
        {
            "workflow_run": {"head_repository": {"full_name": BASE}},
            "repository": {"full_name": BASE},
        },
    ],
)
def test_jobs_that_are_not_from_a_fork_pass(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    assert _run(tmp_path, payload).returncode == 0


def test_an_unreadable_payload_off_a_pull_ref_does_not_fail_the_job(
    tmp_path: Path,
) -> None:
    assert _run(tmp_path, None).returncode == 0


def test_the_refusal_runs_first_in_the_hook() -> None:
    text = HOOK.read_text(encoding="utf-8")
    assert text.count("\nfork_verdict=") == 1
    refusal = text.index("\nfork_verdict=")
    gate = text.index("\ndisk_admission_gate || exit 1\n")
    reset = text.index("wire_pypi_cache || true")
    assert refusal < gate < reset
