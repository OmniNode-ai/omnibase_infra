# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The release train's green-CI premise read against recorded GitHub check runs.

OMN-20298 changed ``scripts/ci/release_train.py``. The train reaches GitHub
through ``gh api`` for the check runs its CI premise judges, and until now no
test replayed what GitHub actually returned for that call. These tests do: the
recorded check-run rows of a real merged omnibase_infra head (captured
2026-09-21, ``tests/ci/fixtures/omn18960_external_sweep_check_runs.json``) are
served by a ``gh`` on PATH and read through the train's real subprocess adapter
``default_check_runs`` and its real classifier ``classify_ci_green``. Only the
GitHub response is replayed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HEAD_KEY = "3894"
_REPO = "omnibase_infra"

_CHILD = """
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("rt", sys.argv[1])
rt = importlib.util.module_from_spec(spec)
sys.modules["rt"] = rt
spec.loader.exec_module(rt)
repo, sha, required = sys.argv[2], sys.argv[3], json.loads(sys.argv[4])
runs = rt.default_check_runs(repo, sha)
reason, detail = rt.classify_ci_green(
    repo,
    sha,
    "dev",
    required_contexts=lambda r, b: required,
    check_runs=lambda r, s: runs,
    gating_sha=lambda r, s: s,
)
print(json.dumps({"rows": len(runs), "reason": reason.value if reason else None}))
"""


def _read_through_the_train(
    tmp_path: Path, head: dict[str, Any], required: list[str]
) -> dict[str, Any]:
    """Serve the recorded rows from a ``gh`` on PATH and classify them."""
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps(head["check_runs_all"]), encoding="utf-8")
    calls = tmp_path / "calls.jsonl"
    gh = tmp_path / "gh"
    sha = head["head_sha"]
    gh.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\nfrom pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "assert args[:2] == ['api', '--paginate'], args\n"
        f"assert args[2] == 'repos/OmniNode-ai/{_REPO}/commits/{sha}/check-runs', args\n"
        f"with Path({str(calls)!r}).open('a') as stream:\n"
        "    stream.write(json.dumps(args) + '\\n')\n"
        f"for row in json.loads(Path({str(rows)!r}).read_text()):\n"
        "    print(json.dumps({k: row.get(k) for k in "
        "('name', 'status', 'conclusion', 'started_at', 'completed_at')}))\n",
        encoding="utf-8",
    )
    gh.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(tmp_path) + os.pathsep + env["PATH"]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _CHILD,
            str(_REPO_ROOT / "scripts" / "ci" / "release_train.py"),
            _REPO,
            sha,
            json.dumps(required),
        ],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(calls.read_text(encoding="utf-8").splitlines()) == 1
    verdict: dict[str, Any] = json.loads(result.stdout.splitlines()[-1])
    return verdict


@pytest.mark.live_contact("tests/ci/fixtures/omn18960_external_sweep_check_runs.json")
def test_recorded_green_head_satisfies_the_trains_ci_premise(
    recorded_response: dict[str, Any], tmp_path: Path
) -> None:
    head = recorded_response["pull_requests"][_HEAD_KEY]
    verdict = _read_through_the_train(
        tmp_path, head, ["CI Summary", "Detect Changes", "Fingerprint Check"]
    )
    assert verdict["rows"] == len(head["check_runs_all"])
    assert verdict["reason"] is None


@pytest.mark.live_contact("tests/ci/fixtures/omn18960_external_sweep_check_runs.json")
def test_recorded_failed_check_run_is_not_green_for_the_trains_ci_premise(
    recorded_response: dict[str, Any], tmp_path: Path
) -> None:
    """Discrimination: the same head with a recorded failing context required."""
    head = recorded_response["pull_requests"][_HEAD_KEY]
    names = [row["name"] for row in head["check_runs_all"]]
    failed = [
        row["name"]
        for row in head["check_runs_all"]
        if row["conclusion"] == "failure" and names.count(row["name"]) == 1
    ]
    assert failed, "the recording carries no single-run failing context"
    verdict = _read_through_the_train(tmp_path, head, [failed[0]])
    assert verdict["reason"] == "ci_not_green"
