# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The failure-rate alerter covers the three repositories it missed (OMN-20794).

MEASURED 2026-10-09, before this change: ``nonrequired_check_alert.fleet_repos``
named thirteen repositories and not ``omnibase_internal``, ``omniclaude-internal``
or ``omnicursor``. A scheduled workflow that failed in any of the three raised
nothing anywhere, because no evaluator was pointed at them.

Each added repository belongs to the GitHub-hosted share
(``scheduled_actions_repos``): the ``nonrequired-check-failure-rate`` job in
``pr-ci-zombie-detector.yml`` runs on a GitHub-hosted runner under the onexbot
App token, so the dead-man's own workflow and the three newly covered
repositories are evaluated there and never by the ``.201`` reporter, whose
``--repos-from-policy`` selection is ``fleet_repos`` minus that share.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = REPO_ROOT / "scripts" / "ci" / "nonrequired_check_failure_rate.py"
POLICY = REPO_ROOT / "config" / "runner_routing_policy.yaml"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "pr-ci-zombie-detector.yml"
ALERT_JOB = "nonrequired-check-failure-rate"
OWNER = "OmniNode-ai"
ADDED = ("omnibase_internal", "omniclaude-internal", "omnicursor")


def _load_evaluator() -> Any:
    spec = importlib.util.spec_from_file_location("nrcfr_internal", EVALUATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _alert_block() -> dict[str, Any]:
    block = yaml.safe_load(POLICY.read_text())["route"]["nonrequired_check_alert"]
    assert isinstance(block, dict)
    return block


def _workflow_step(name_fragment: str) -> dict[str, Any]:
    doc: dict[Any, Any] = yaml.safe_load(WORKFLOW.read_text())
    steps = doc["jobs"][ALERT_JOB]["steps"]
    matches = [s for s in steps if name_fragment in str(s.get("name", "")).lower()]
    assert len(matches) == 1, name_fragment
    return matches[0]


def _flag_values(run: str, flag: str) -> set[str]:
    return {
        line.strip().removeprefix(f"{flag} ").strip().rstrip("\\").strip()
        for line in run.splitlines()
        if line.strip().startswith(f"{flag} ")
    }


class _Fake:
    """GitHub faked at the evaluator's own ``_gh_json`` seam, with a call log."""

    def __init__(self) -> None:
        self.paths: list[str] = []

    def __call__(self, path: str, token: str | None) -> Any:
        self.paths.append(path)
        if "/branches/" in path and path.endswith("/required_status_checks"):
            return {"contexts": ["CI Summary"]}
        if "/pulls?" in path:
            return [{"head": {"sha": "abc"}}]
        if "/check-runs" in path:
            return {"check_runs": []}
        if path.endswith("/actions/workflows?per_page=100"):
            return {
                "workflows": [
                    {
                        "id": 11,
                        "path": ".github/workflows/scheduled-gap-detect.yml",
                        "state": "active",
                    }
                ]
            }
        if path.endswith("/actions/workflows/runner-routing-audit.yml"):
            return {
                "id": 12,
                "path": ".github/workflows/runner-routing-audit.yml",
                "state": "active",
            }
        if "/actions/workflows/12/runs" in path:
            return {"workflow_runs": []}
        if "/actions/workflows/11/runs" in path:
            return {
                "workflow_runs": [
                    {
                        "conclusion": "success",
                        "created_at": f"2026-09-2{i}T00:00:00Z",
                        "html_url": f"https://github.com/x/runs/{i}",
                    }
                    for i in range(8)
                ]
            }
        raise AssertionError(f"unexpected gh api path {path}")


def _evaluate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, argv: list[str]
) -> tuple[_Fake, dict[str, Any]]:
    module = _load_evaluator()
    fake = _Fake()
    monkeypatch.setattr(module, "_gh_json", fake)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    report = tmp_path / "report.json"
    code = module.main(
        [*argv, "--policy", str(POLICY), "--report", str(report), "--dry-run"]
    )
    assert code == 0
    return fake, json.loads(report.read_text())


def test_fleet_repos_include_internal_and_cursor() -> None:
    block = _alert_block()
    for repo in ADDED:
        assert repo in block["fleet_repos"], f"{repo} is not a fleet repository"


def test_fleet_repos_include_internal_in_the_github_hosted_share() -> None:
    share = _alert_block()["scheduled_actions_repos"]
    for repo in ADDED:
        assert repo in share, (
            f"{repo} is not in scheduled_actions_repos, so the .201 reporter "
            "would be the one evaluating it"
        )


def test_fleet_repos_include_internal_in_the_workflow_scheduled_share() -> None:
    run = str(_workflow_step("evaluate non-required check failure rates").get("run"))
    scheduled = _flag_values(run, "--scheduled-repo")
    for repo in ADDED:
        assert repo in scheduled, (
            f"pr-ci-zombie-detector.yml omits --scheduled-repo {repo}"
        )


def test_fleet_repos_include_internal_in_the_minted_token_scope() -> None:
    with_block = _workflow_step("mint the onexbot app token").get("with")
    assert isinstance(with_block, dict)
    scope = {
        line.strip()
        for line in str(with_block.get("repositories", "")).splitlines()
        if line.strip()
    }
    for repo in ADDED:
        assert repo in scope, f"the App token is not scoped to read {repo}"


def test_fleet_repos_include_internal_and_the_host_reporter_skips_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The ``.201`` selection is the real policy's fleet minus the share."""
    fake, report = _evaluate(
        monkeypatch, tmp_path, ["--repos-from-policy", "--scheduled-only"]
    )
    for repo in ADDED:
        assert f"{OWNER}/{repo}" not in report["repos"], f"the host evaluates {repo}"
        assert not any(f"/{repo}/" in p for p in fake.paths), fake.paths


def test_fleet_repos_include_internal_and_the_actions_job_evaluates_their_scheduled_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The Actions job's own argv reads each added repository's scheduled runs."""
    run = str(_workflow_step("evaluate non-required check failure rates").get("run"))
    scheduled = sorted(_flag_values(run, "--scheduled-repo"))
    argv: list[str] = ["--repo", "omnibase_infra", "--no-scheduled"]
    for repo in scheduled:
        argv += ["--scheduled-repo", repo]
    fake, report = _evaluate(monkeypatch, tmp_path, argv)
    for repo in ADDED:
        assert f"{OWNER}/{repo}" in report["repos"], f"{repo} was not evaluated"
        assert any(f"/{repo}/actions/" in p for p in fake.paths), fake.paths
        assert not any(f"/{repo}/branches/" in p for p in fake.paths), (
            f"{repo} must be scheduled-only, no branch-protection read"
        )
