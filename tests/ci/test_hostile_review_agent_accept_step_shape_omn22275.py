# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22275: the workflow adds no decision logic -- one node call and one ``if:`` on the enforcement step.

The gate-accept step runs the omnimarket node's ``python -m`` entry from a clone of
omnimarket's newest ``dev-<UTC>-<sha12>`` tag, inside the omniintelligence environment the
job already built, never from the pull request's checkout, so a pull request cannot
supply the code that accepts itself. Its shell holds only the node call and the
``decision=unchanged`` fallbacks for a clone without the node. The "Enforce hostile-review
verdict" step keeps its ``run:`` text and ``env`` byte for byte (pinned by digest, taken
from the default branch) and gains only the condition that skips it on ``pass_agent``.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)
from tests.ci._hostile_review_job_omn22275 import (
    NODE_DIR,
    REVIEW_JOB,
    job,
    load_workflow,
    step_named,
)

pytestmark = pytest.mark.unit

#: sha256 of the enforcement step's ``run:`` text on the default branch before OMN-22275.
ENFORCE_RUN_SHA256 = "78da11758989faeb5c86c27aae1e1e1ce5eb08cffe55e27489750d988075046a"
ENFORCE_ENV = {
    "REVIEW_VERDICT": "${{ steps.review.outputs.verdict || steps.preflight.outputs.verdict }}",
    "REVIEW_ERROR": "${{ steps.review.outputs.error || steps.preflight.outputs.error }}",
}
ENFORCE_IF = "always() && steps.accept.outputs.decision != 'pass_agent'"
ACCEPT_CONTEXTS = {
    "github.repository",
    "github.event.pull_request.number",
    "github.event.pull_request.head.sha",
    "github.workspace",
    "secrets.GITHUB_TOKEN",
}
NODE_MODULE = (
    "omnimarket.nodes.node_agent_review_gate_accept_effect.runtime_gate_accept"
)
OVERLAY_ID = "agent_review_gate_accept"
CLONE_STEP = "Clone omnimarket"
_TAGS = (
    "dev-20261103015640-79e4562b8010",
    "dev-20261103020838-7b42a4c63edc",
    "dev-20261103015329-caf44805f6c1",
    "dev-latest",
    "v0.99.0",
)
_NEWEST = "dev-20261103020838-7b42a4c63edc"


def _steps() -> list[dict[str, Any]]:
    return list(job(REVIEW_JOB)["steps"])


def _accept() -> dict[str, Any]:
    found = [s for s in _steps() if s.get("id") == "accept"]
    assert len(found) == 1, "the job must hold exactly one gate-accept step"
    return found[0]


def test_the_accept_step_runs_the_node_from_the_omnimarket_clone_under_if_always() -> (
    None
):
    step = _accept()
    assert set(step) == {"name", "id", "if", "working-directory", "env", "run"}, sorted(
        step
    )
    assert step["if"] == "always()"
    assert step["working-directory"] == "../omniintelligence"
    run = str(step["run"])
    assert "onex run-node" not in run
    calls = re.findall(rf"python -m {re.escape(NODE_MODULE)}\b", run)
    assert len(calls) == 1, calls
    assert ('PYTHONPATH="$ACCEPT_SRC" uv run --no-sync python -m ' + NODE_MODULE) in run
    assert step["env"]["ACCEPT_SRC"] == "${{ github.workspace }}/../omnimarket/src"
    assert "PYTHONPATH" not in run.replace('PYTHONPATH="$ACCEPT_SRC"', "")
    assert "$GITHUB_WORKSPACE/src" not in run
    assert "git " not in run, "the clone is its own step; the accept step holds none"
    assert "::notice::" in run, "a clone without the node must skip with a notice"
    assert run.count("decision=unchanged") == 2, "one fallback per skipped path"
    for decided in ("pass_agent", "decision=refuse", "decision=pass"):
        assert decided not in run, f"the step shell decides: {decided}"


def test_the_accept_step_reads_the_reviewers_result_and_nothing_of_the_pull_request() -> (
    None
):
    run = str(_accept()["run"])
    assert (
        '--voter-result-path "${{ github.workspace }}/hostile-review-result.json"'
        in run
    )
    review = step_named(REVIEW_JOB, "Run adversarial review")
    assert review["env"]["REVIEW_JSON_PATH"] == (
        "${{ github.workspace }}/hostile-review-result.json"
    )
    clean = str(step_named(REVIEW_JOB, "Clean stale dependency clones")["run"])
    assert "hostile-review-result.json" in clean, (
        "a result file committed to the pull request must not stand in for the "
        "reviewer's own output"
    )
    assert "../omnimarket" in clean


def test_the_accept_step_names_only_the_contexts_it_needs() -> None:
    step = _accept()
    run_contexts = set(re.findall(r"\$\{\{\s*(.+?)\s*\}\}", str(step["run"])))
    env_contexts = {
        c
        for value in step["env"].values()
        for c in re.findall(r"\$\{\{\s*(.+?)\s*\}\}", str(value))
    }
    assert run_contexts | env_contexts == ACCEPT_CONTEXTS, sorted(
        run_contexts | env_contexts
    )
    assert step["env"]["GITHUB_TOKEN"] == "${{ secrets.GITHUB_TOKEN }}"
    assert "LOCAL_LLM_SHARED_SECRET" not in str(step["env"])


def test_the_ci_overlay_that_holds_the_nodes_deployment_facts_is_fetched_and_rooted() -> (
    None
):
    roots = str(_accept()["env"]["ONEX_SKILL_OVERLAY_ROOTS"])
    assert roots == "${{ github.workspace }}/../ci-overlay/config/onex/overlays"
    fetch = step_named(REVIEW_JOB, "Fetch the CI overlay")
    sparse = [
        line.strip() for line in str(fetch["with"]["sparse-checkout"]).splitlines()
    ]
    assert f"config/onex/overlays/{OVERLAY_ID}" in sparse, sparse
    assert "config/onex/overlays/node_review_voters_overlay_compute" in sparse, sparse


def test_the_accept_step_sits_between_the_summary_comment_and_the_enforcement_step() -> (
    None
):
    names = [str(s.get("name", "")) for s in _steps()]
    review = names.index(step_named(REVIEW_JOB, "Run adversarial review")["name"])
    posts = names.index(step_named(REVIEW_JOB, "Post review summary")["name"])
    accept = names.index(str(_accept()["name"]))
    enforce = names.index(
        step_named(REVIEW_JOB, "Enforce hostile-review verdict")["name"]
    )
    assert review < posts < accept < enforce
    assert enforce == len(names) - 1


def test_the_enforcement_step_changes_only_its_condition() -> None:
    step = step_named(REVIEW_JOB, "Enforce hostile-review verdict")
    assert set(step) == {"name", "if", "env", "run"}, sorted(step)
    assert step["if"] == ENFORCE_IF
    assert step["env"] == ENFORCE_ENV
    digest = hashlib.sha256(str(step["run"]).encode()).hexdigest()
    assert digest == ENFORCE_RUN_SHA256, "the run: text of the enforcement step moved"


def test_the_shape_check_fails_on_a_step_that_holds_more_than_a_node_call() -> None:
    """POSITIVE CONTROL: the checks above reject a decision made in the step's shell."""
    run = str(_accept()["run"])
    assert "pass_agent" not in run
    planted = run + '\necho "decision=pass_agent" >> "$GITHUB_OUTPUT"\n'
    assert "pass_agent" in planted
    drifted = str(step_named(REVIEW_JOB, "Enforce hostile-review verdict")["run"]) + " "
    assert hashlib.sha256(drifted.encode()).hexdigest() != ENFORCE_RUN_SHA256


def test_the_gate_job_is_unchanged_in_what_it_reads() -> None:
    gate = job("hostile-review-gate")
    assert gate["needs"] == "hostile-review"
    assert gate["if"] == "${{ !cancelled() }}"
    assert "needs.hostile-review.result" in str(gate["steps"][0]["run"])
    assert load_workflow()["jobs"]["hostile-review-thread-gate"]["needs"] == [
        "hostile-review"
    ]


def _git(cwd: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false", *args],
        cwd=cwd,
        env=scrub_git_location_env(
            {
                **os.environ,
                "GIT_AUTHOR_NAME": "t",
                "GIT_AUTHOR_EMAIL": "t@example.invalid",
                "GIT_COMMITTER_NAME": "t",
                "GIT_COMMITTER_EMAIL": "t@example.invalid",
            }
        ),
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


def _origin(tmp_path: Path, tags: tuple[str, ...]) -> tuple[Path, dict[str, str]]:
    """A real local bare origin with one commit per tag; returns it and tag -> commit."""
    work = tmp_path / "seed"
    work.mkdir(parents=True)
    _git(work, "init", "-q", "-b", "dev")
    commits: dict[str, str] = {}
    for tag in tags:
        (work / "marker.txt").write_text(tag, encoding="utf-8")
        _git(work, "add", "marker.txt")
        _git(work, "commit", "-q", "-m", tag)
        commits[tag] = _git(work, "rev-parse", "HEAD")
        _git(work, "tag", tag)
    origin = tmp_path / "origin.git"
    _git(tmp_path, "clone", "-q", "--bare", str(work), str(origin))
    return origin, commits


def _run_clone_step(
    tmp_path: Path, url: str
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run the clone step as written, against a real git origin; only ``sleep`` is faked."""
    bindir = tmp_path / "bin"
    bindir.mkdir(parents=True)
    sleep = bindir / "sleep"
    sleep.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    sleep.chmod(sleep.stat().st_mode | stat.S_IEXEC)
    work = tmp_path / "ws"
    work.mkdir()
    env = scrub_git_location_env(
        {
            **os.environ,
            "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}",
            "OMNIMARKET_URL": url,
        }
    )
    done = subprocess.run(
        ["bash", "-e", "-c", str(step_named(REVIEW_JOB, CLONE_STEP)["run"])],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return done, tmp_path / "omnimarket"


def test_the_clone_step_names_no_repository_url_of_its_own() -> None:
    step = step_named(REVIEW_JOB, CLONE_STEP)
    assert "github.com" not in str(step["run"])
    assert step["env"]["OMNIMARKET_URL"] == (
        "${{ github.server_url }}/${{ github.repository_owner }}/omnimarket.git"
    )


def test_the_clone_step_reads_the_newest_dev_tag_of_omnimarket_and_logs_it(
    tmp_path: Path,
) -> None:
    origin, commits = _origin(tmp_path / "o", _TAGS)
    done, clone = _run_clone_step(tmp_path / "r", origin.as_uri())
    assert done.returncode == 0, done.stderr
    assert _git(clone, "rev-parse", "HEAD") == commits[_NEWEST]
    assert (clone / "marker.txt").read_text(encoding="utf-8") == _NEWEST
    assert f"omnimarket newest dev tag: {_NEWEST} -> {commits[_NEWEST]}" in done.stdout


def test_the_clone_step_never_fails_the_job_when_no_tag_or_clone_resolves(
    tmp_path: Path,
) -> None:
    """A failed clone must not turn a passed review red: the voters' verdict stands."""
    origin, _commits = _origin(tmp_path / "o", ("v0.99.0", "dev-latest"))
    done, clone = _run_clone_step(tmp_path / "notag", origin.as_uri())
    assert done.returncode == 0, done.stderr
    assert not clone.exists()
    assert "::warning::" in done.stdout
    gone = tmp_path / "gone.git"
    refused, clone = _run_clone_step(tmp_path / "nofetch", gone.as_uri())
    assert refused.returncode == 0, refused.stderr
    assert not clone.exists()
    assert "::warning::" in refused.stdout


def test_the_clone_step_never_reads_a_branch_tip_or_a_pin() -> None:
    run = str(step_named(REVIEW_JOB, CLONE_STEP)["run"])
    assert "git ls-remote --tags --refs" in run
    assert "'refs/tags/dev-*'" in run
    assert not re.search(r"--branch\s+(dev|main)\b", run)
    assert not re.search(r"checkout\s+(--quiet\s+)?(origin/)?(dev|main)\b", run)
    assert not re.search(r"\b[0-9a-f]{40}\b", run)
    assert "sibling-pins.yaml" not in run
    assert NODE_DIR  # the directory the accept step looks for in this clone
