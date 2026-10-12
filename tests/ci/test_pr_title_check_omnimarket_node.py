# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21009: preserve the title verdict and context through the published node.

The bash oracle below is the check-title step at the caller's former pin
326ebee0561ab42b3d970e533ae9f447abed2292. It is self-contained so CI needs
neither a sibling checkout nor access to the former reusable repository.
The fixture runs the workflow's actual installation and invocation script,
then probes that installed release with isolated Python, including refusals.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/pr-title-check.yml"
MANIFEST = ROOT / ".github/required-checks.yaml"
CONTEXT = "pr-title / check-title"
NODE_MODULE = "omnimarket.nodes.node_pr_title_check_compute"
RUNS_ON = """${{
  (github.event_name == 'pull_request' &&
   github.event.pull_request.head.repo.full_name != github.repository)
  && fromJSON(vars.OMNI_PUBLIC_PR_RUNS_ON_JSON || '["ubuntu-latest"]')
  || fromJSON(vars.OMNI_TRUSTED_CI_RUNS_ON_JSON || '["self-hosted","omnibase-ci"]')
}}"""
LEGACY_BASH = r"""# Guard: workflow_call must be triggered by a pull_request event
if [[ -z "$PR_TITLE" ]]; then
  echo "[FAIL] PR_TITLE is empty — this workflow must be called from a pull_request event."
  exit 1
fi

# Exempt categories (case-insensitive matching)
# SYNC NOTE: exempt prefixes must match _EXEMPT_TITLE_PREFIXES in:
#   - omniclaude/src/omniclaude/nodes/node_git_effect/models/model_git_request.py
#   - omnibase_infra/scripts/generate_deep_dive.py
TITLE_LOWER=$(echo "$PR_TITLE" | tr '[:upper:]' '[:lower:]')

# 1. Any GitHub App bot (catches dependabot, renovate, coderabbit, snyk, etc.)
if [[ "$PR_AUTHOR" == *"[bot]" ]]; then
  echo "[PASS] Exempt: GitHub App bot ($PR_AUTHOR)"
  exit 0
fi

# 2. Dep bump titles (manual or non-bot)
if [[ "$TITLE_LOWER" =~ ^(chore\(deps|build\(deps|bump ) ]]; then
  echo "[PASS] Exempt: dependency bump"
  exit 0
fi

# 3. Release PRs
if [[ "$TITLE_LOWER" =~ ^(chore:\ release|chore\(release\)|release:) ]]; then
  echo "[PASS] Exempt: release PR"
  exit 0
fi

# 4. Check for OMN-XXXX pattern
if echo "$PR_TITLE" | grep -qE 'OMN-[0-9]+'; then
  echo "[PASS] PR title contains ticket reference: $(echo "$PR_TITLE" | grep -oE 'OMN-[0-9]+')"
  exit 0
fi

# Fail
echo "[FAIL] PR title must contain an OMN-XXXX ticket reference."
echo ""
echo "Title: $PR_TITLE"
echo ""
echo "Exempt categories (no ticket needed):"
echo "  - Any GitHub App bot (author ends with [bot])"
echo "  - Titles starting with: chore(deps, build(deps, Bump"
echo "  - Titles starting with: chore: release, chore(release), release:"
echo ""
echo "To fix: add [OMN-XXXX] or OMN-XXXX to the PR title."
exit 1
"""


def _workflow() -> dict[Any, Any]:
    loaded: dict[Any, Any] = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return loaded


def _job() -> dict[str, Any]:
    job: dict[str, Any] = _workflow()["jobs"]["pr-title"]
    return job


def _node_step() -> dict[str, Any]:
    return next(
        step for step in _job()["steps"] if "OMNIMARKET_VERSION" in step.get("env", {})
    )


def test_workflow_no_longer_calls_change_control() -> None:
    assert "onex_change_control" not in WORKFLOW.read_text(encoding="utf-8")
    assert set(_workflow()["jobs"]) == {"pr-title"}
    assert "uses" not in _job()


def test_job_keeps_context_triggers_routing_and_fail_closed_execution() -> None:
    job = _job()
    assert job["name"] == CONTEXT
    assert job["runs-on"] == RUNS_ON
    assert job["timeout-minutes"] == 10
    assert job["permissions"] == {"contents": "read"}
    assert "if" not in job
    assert "needs" not in job
    assert "continue-on-error" not in job
    assert _workflow()[True] == {
        "pull_request": {"types": ["opened", "edited", "synchronize", "reopened"]}
    }
    for step in job["steps"]:
        assert "continue-on-error" not in step
        assert "if" not in step
        if "uses" in step:
            assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", step["uses"])


def test_job_runs_the_isolated_node_from_a_pinned_published_release() -> None:
    step = _node_step()
    script = step["run"]
    env = step["env"]
    assert re.fullmatch(r"\d+\.\d+\.\d+", str(env["OMNIMARKET_VERSION"]))
    assert tuple(map(int, env["OMNIMARKET_VERSION"].split("."))) >= (0, 4, 310)
    assert '"omnimarket==${OMNIMARKET_VERSION}"' in script
    assert "--no-deps" in script
    assert f"-I -m {NODE_MODULE}" in script
    assert "set -euo pipefail" in script
    assert env["PR_TITLE"] == "${{ github.event.pull_request.title }}"
    assert env["PR_AUTHOR"] == "${{ github.event.pull_request.user.login }}"
    assert "${{ github.event.pull_request.title }}" not in script
    assert "${{ github.event.pull_request.user.login }}" not in script


def test_manifest_row_names_the_local_producer() -> None:
    rows = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["gates"]
    matches = [row for row in rows if row["name"] == CONTEXT]
    assert len(matches) == 1
    row = matches[0]
    assert row["mode"] == "REQUIRED"
    assert row["producer_kind"] == "local"
    assert row["workflow"] == "pr-title-check.yml"
    assert row["job_path"] == ["pr-title"]
    assert row["skip_semantics"] == "never"
    assert "cross_repo_ref" not in row
    assert "onex_change_control" not in row["rationale"]


@pytest.fixture(scope="module")
def installed_node(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Install using precisely the workflow script, without project imports."""
    step = _node_step()
    runner_temp = tmp_path_factory.mktemp("pr-title-runner")
    env = {
        **os.environ,
        "RUNNER_TEMP": str(runner_temp),
        "OMNIMARKET_VERSION": step["env"]["OMNIMARKET_VERSION"],
        "PR_TITLE": "ci(OMN-21009): port title check",
        "PR_AUTHOR": "contributor",
    }
    installed = subprocess.run(
        ["bash", "-c", step["run"]],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert installed.returncode == 0, installed.stdout + installed.stderr
    python = runner_temp / "pr-title-check/bin/python"
    version = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            "from importlib.metadata import version; print(version('omnimarket'))",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    assert version.stdout.strip() == step["env"]["OMNIMARKET_VERSION"]
    return python


# All six exempt prefixes, mixed case, bot suffix and its boundaries, ticket
# case and location, empty-title precedence, and literal shell metacharacters.
@pytest.mark.parametrize(
    ("title", "author", "exit_code"),
    [
        ("ci(OMN-21009): port title check", "contributor", 0),
        ("fix: use [OMN-1]", "contributor", 0),
        ("OMN-1 and OMN-2", "contributor", 0),
        ("bare title", "contributor", 1),
        ("fix: omn-21009", "contributor", 1),
        ("fix: OMN-", "contributor", 1),
        ("fix: OMN-X", "contributor", 1),
        ("chore(deps): bump library", "contributor", 0),
        ("ChOrE(DePs-dev): bump library", "contributor", 0),
        ("build(deps): bump library", "contributor", 0),
        ("BUILD(DEPS-dev): bump library", "contributor", 0),
        ("bump library", "contributor", 0),
        ("BUMP library", "contributor", 0),
        ("chore: release library", "contributor", 0),
        ("CHORE: RELEASE library", "contributor", 0),
        ("chore(release): library", "contributor", 0),
        ("CHORE(RELEASE): library", "contributor", 0),
        ("release: library", "contributor", 0),
        ("RELEASE: library", "contributor", 0),
        ("bare title", "automation[bot]", 0),
        ("bare title", "automation[bot]-suffix", 1),
        ("bare title", "automation[BOT]", 1),
        ("bump", "contributor", 1),
        (" chore(deps): library", "contributor", 1),
        ("chore:release library", "contributor", 1),
        ("chore(release-candidate): library", "contributor", 1),
        ("", "contributor", 1),
        ("", "automation[bot]", 1),
        ('fix(OMN-21009): literal $(exit 17) `exit 19` "quote"', "contributor", 0),
    ],
)
def test_installed_release_verdict_equals_pinned_bash(
    installed_node: Path,
    title: str,
    author: str,
    exit_code: int,
) -> None:
    bash = subprocess.run(
        ["bash", "-c", LEGACY_BASH],
        env={**os.environ, "PR_TITLE": title, "PR_AUTHOR": author},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    node = subprocess.run(
        [str(installed_node), "-I", "-m", NODE_MODULE],
        input=json.dumps({"title": title, "author": author}),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert bash.returncode == exit_code, bash.stdout + bash.stderr
    assert node.returncode == bash.returncode, node.stdout + node.stderr
    assert node.stdout == bash.stdout


@pytest.mark.parametrize(
    "payload", ["not json", "{}", '{"title": 7, "author": "contributor"}']
)
def test_installed_release_refuses_malformed_requests(
    installed_node: Path, payload: str
) -> None:
    node = subprocess.run(
        [str(installed_node), "-I", "-m", NODE_MODULE],
        input=payload,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert node.returncode == 2
    assert "PR_TITLE_CHECK_BAD_REQUEST:" in node.stderr


@pytest.mark.live_contact("tests/ci/fixtures/omn21009_pr_title_node_responses.json")
def test_installed_release_replays_recorded_lab_responses(
    installed_node: Path, recorded_response: dict[str, Any]
) -> None:
    """Replay captured published-node responses through the real CLI boundary."""
    assert (
        recorded_response["_provenance"]["omnimarket_version"]
        == _node_step()["env"]["OMNIMARKET_VERSION"]
    )
    responses = recorded_response["responses"]
    assert [case["response"]["exit_code"] for case in responses] == [0, 1, 0]
    for case in responses:
        node = subprocess.run(
            [str(installed_node), "-I", "-m", NODE_MODULE],
            input=json.dumps(case["request"]),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert node.returncode == case["response"]["exit_code"]
        assert node.stdout == case["response"]["stdout"]
        assert node.stderr == case["response"]["stderr"]
