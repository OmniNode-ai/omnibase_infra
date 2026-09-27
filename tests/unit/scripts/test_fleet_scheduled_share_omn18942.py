# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The registry repository's scheduled workflows reach a destination (OMN-18942, AC-2).

MEASURED 2026-09-23, before this change. The scheduled-workflow sweep had one
caller, the `.201` system reporter, and that host's token returns HTTP 404 on
`omni_home`. Its tick at 17:45Z rendered `sched/omni_home/source-unreadable`
and evaluated nothing there. The GitHub Actions alerter job passed
`--no-scheduled` over three repositories. So the registry repository, the one
repository the ticket names, was watched by nobody. A dry run of this evaluator
under a token that can read it found two of its three scheduled workflows red
on 8 of 8 runs (`scheduled-gap-detect.yml`, `topic-parity.yml`).

THE FIX IS A DECLARED PARTITION, NOT A SECOND SWEEP. The policy names the fleet
repositories whose scheduled half the Actions job evaluates, because its onexbot
App installation reads every repository, and the host sweep takes the rest. Each
fleet repository's scheduled runs have exactly one evaluator and one poster, so
the "two surfaces, one run, two destinations" divergence stays impossible.

No credential is created, widened or rotated. The App token the Actions job
already mints now requests one more repository out of an installation whose
selection is already `all`.
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

REPO_ROOT = Path(__file__).resolve().parents[3]
EVALUATOR = REPO_ROOT / "scripts" / "ci" / "nonrequired_check_failure_rate.py"
POLICY = REPO_ROOT / "config" / "runner_routing_policy.yaml"
ZOMBIE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "pr-ci-zombie-detector.yml"
ALERT_JOB = "nonrequired-check-failure-rate"
OWNER = "OmniNode-ai"


def _load_evaluator() -> Any:
    spec = importlib.util.spec_from_file_location("nrcfr_share", EVALUATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _alert_block() -> dict[str, Any]:
    block = yaml.safe_load(POLICY.read_text())["route"]["nonrequired_check_alert"]
    assert isinstance(block, dict)
    return block


def _policy_with(tmp_path: Path, **overrides: Any) -> Path:
    doc = yaml.safe_load(POLICY.read_text())
    block = doc["route"]["nonrequired_check_alert"]
    for key, value in overrides.items():
        if value is None:
            block.pop(key, None)
        else:
            block[key] = value
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


# ---------------------------------------------------------------------------
# The policy declares the partition, and refuses a malformed one.
# ---------------------------------------------------------------------------


def test_policy_hands_the_registry_repository_to_the_actions_job() -> None:
    block = _alert_block()
    share = block["scheduled_actions_repos"]
    assert isinstance(share, list) and share, "the Actions share is empty"
    assert "omni_home" in share, (
        "the registry repository is the one the host token cannot read; it must "
        "be evaluated where a token that can read it runs"
    )
    assert set(share) <= set(block["fleet_repos"])


def test_load_policy_requires_the_actions_share(tmp_path: Path) -> None:
    """Rule 8: a defaulted-empty share silently hands the registry back to nobody."""
    module = _load_evaluator()
    with pytest.raises(KeyError, match="scheduled_actions_repos"):
        module.load_policy(_policy_with(tmp_path, scheduled_actions_repos=None))


def test_load_policy_refuses_a_share_outside_the_fleet(tmp_path: Path) -> None:
    """A repository outside the fleet list would be evaluated by an ungoverned caller."""
    module = _load_evaluator()
    with pytest.raises(ValueError, match="not in fleet_repos"):
        module.load_policy(
            _policy_with(tmp_path, scheduled_actions_repos=["not_a_fleet_repo"])
        )


def test_load_policy_refuses_a_duplicate_in_the_share(tmp_path: Path) -> None:
    module = _load_evaluator()
    with pytest.raises(ValueError, match="duplicate"):
        module.load_policy(
            _policy_with(tmp_path, scheduled_actions_repos=["omni_home", "omni_home"])
        )


# ---------------------------------------------------------------------------
# The two callers split the fleet with no overlap and no gap.
# ---------------------------------------------------------------------------


class _Fake:
    """GitHub faked at the evaluator's own `_gh_json` seam, with a call log."""

    def __init__(self, red_workflow_repo: str) -> None:
        self.paths: list[str] = []
        self.red_repo = red_workflow_repo

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
        if "/actions/workflows/11/runs" in path:
            conclusion = "failure" if f"/{self.red_repo}/" in path else "success"
            return {
                "workflow_runs": [
                    {
                        "conclusion": conclusion,
                        "created_at": f"2026-09-2{i}T00:00:00Z",
                        "html_url": f"https://github.com/x/runs/{i}",
                    }
                    for i in range(8)
                ]
            }
        raise AssertionError(f"unexpected gh api path {path}")


def _run(
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake: _Fake,
    argv: list[str],
) -> tuple[int, dict[str, Any]]:
    monkeypatch.setattr(module, "_gh_json", fake)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    report = tmp_path / "report.json"
    code = module.main([*argv, "--report", str(report), "--dry-run"])
    return int(code), json.loads(report.read_text())


def test_the_host_selection_excludes_the_actions_share(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`--repos-from-policy` is the host's call; it must not sweep omni_home too."""
    module = _load_evaluator()
    policy = _policy_with(
        tmp_path,
        fleet_repos=["omni_home", "omnibase_infra"],
        scheduled_actions_repos=["omni_home"],
    )
    fake = _Fake(red_workflow_repo="omni_home")
    code, report = _run(
        module,
        monkeypatch,
        tmp_path,
        fake,
        ["--repos-from-policy", "--scheduled-only", "--policy", str(policy)],
    )
    assert code == 0
    assert sorted(report["repos"]) == [f"{OWNER}/omnibase_infra"]
    assert not any("/omni_home/" in p for p in fake.paths), fake.paths


def test_the_actions_share_is_evaluated_scheduled_only_beside_the_check_run_half(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One run, one report, one dedup state: the check-run half for `--repo`,
    the scheduled half for `--scheduled-repo`, and neither half leaks."""
    module = _load_evaluator()
    policy = _policy_with(
        tmp_path,
        fleet_repos=["omni_home", "omnibase_infra"],
        scheduled_actions_repos=["omni_home"],
    )
    fake = _Fake(red_workflow_repo="omni_home")
    code, report = _run(
        module,
        monkeypatch,
        tmp_path,
        fake,
        [
            "--repo",
            "omnibase_infra",
            "--no-scheduled",
            "--scheduled-repo",
            "omni_home",
            "--policy",
            str(policy),
        ],
    )
    assert code == 0
    # omni_home: scheduled only. No branch-protection read: its default branch
    # is `main` and it has no `dev`, which is why the check-run half 404s there.
    assert not any("/omni_home/branches/" in p for p in fake.paths), fake.paths
    assert not any("/omni_home/pulls" in p for p in fake.paths), fake.paths
    # omnibase_infra: check-run half only; the host owns its scheduled half.
    assert not any("/omnibase_infra/actions/" in p for p in fake.paths), fake.paths
    alerts = report["repos"][f"{OWNER}/omni_home"]["scheduled"]["alerts"]
    assert alerts == [
        {
            "workflow": ".github/workflows/scheduled-gap-detect.yml",
            "failures": 8,
            "observed": 8,
            "rate_pct": 100.0,
            "last_failure_url": "https://github.com/x/runs/7",
        }
    ], "the red registry workflow must raise exactly one finding with its rate"
    due = [entry["key"] for entry in report["dedup"]["due"]]
    assert due == [
        f"scheduled|{OWNER}/omni_home|.github/workflows/scheduled-gap-detect.yml"
    ], "the registry finding must go through the same dedup as every other"


def test_a_scheduled_repo_outside_the_policy_share_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Otherwise any caller could re-sweep a host-owned repository and double-post."""
    module = _load_evaluator()
    monkeypatch.setattr(module, "_gh_json", _Fake("omni_home"))
    with pytest.raises(SystemExit):
        module.main(
            [
                "--scheduled-repo",
                "omnibase_infra",
                "--policy",
                str(POLICY),
                "--report",
                str(tmp_path / "r.json"),
                "--dry-run",
            ]
        )
    assert "not in the policy's scheduled_actions_repos" in capsys.readouterr().err


def test_a_repo_given_both_ways_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_evaluator()
    monkeypatch.setattr(module, "_gh_json", _Fake("omni_home"))
    with pytest.raises(SystemExit):
        module.main(
            [
                "--repo",
                "omni_home",
                "--scheduled-repo",
                "omni_home",
                "--policy",
                str(POLICY),
                "--report",
                str(tmp_path / "r.json"),
                "--dry-run",
            ]
        )
    assert "given both as --repo and as --scheduled-repo" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The Actions job is wired to the policy's share, and its token to its sweep.
# ---------------------------------------------------------------------------


def _evaluate_step() -> dict[str, Any]:
    doc: dict[Any, Any] = yaml.safe_load(ZOMBIE_WORKFLOW.read_text())
    steps = doc["jobs"][ALERT_JOB]["steps"]
    matches = [
        s
        for s in steps
        if "evaluate non-required check failure rates" in str(s.get("name", "")).lower()
    ]
    assert len(matches) == 1
    return matches[0]


def _flag_values(run: str, flag: str) -> set[str]:
    return {
        line.strip().removeprefix(f"{flag} ").strip().rstrip("\\").strip()
        for line in run.splitlines()
        if line.strip().startswith(f"{flag} ")
    }


def test_the_actions_job_sweeps_exactly_the_policy_share() -> None:
    run = str(_evaluate_step().get("run", ""))
    assert _flag_values(run, "--scheduled-repo") == set(
        _alert_block()["scheduled_actions_repos"]
    ), (
        "the Actions job's scheduled share and the policy's share differ; a "
        "repository in the policy share but not the run step is watched by nobody"
    )


def test_every_fleet_repository_has_exactly_one_scheduled_evaluator() -> None:
    block = _alert_block()
    fleet = set(block["fleet_repos"])
    actions = set(block["scheduled_actions_repos"])
    host = fleet - actions
    assert host | actions == fleet
    assert not host & actions
    run = str(_evaluate_step().get("run", ""))
    assert "--no-scheduled" in run, (
        "the Actions job's --repo list must stay check-run only; the host owns "
        "those repositories' scheduled half"
    )
