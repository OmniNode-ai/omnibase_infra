# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Public-repository runner placement in omnibase_infra (OMN-19998).

RULING 2026-10-16T07:07:09Z: "Ci runners should be for private repos only."
omnibase_infra is public, so:

* no routing-seam fallback in a workflow names the self-hosted fleet, except
  the two lab-broker carve-outs whose jobs publish to the tailnet-only dev-lane
  broker (runtime rebuild, hostile review), which stay until they move off
  GitHub Actions;
* omnibase_core's check-public-runner-placement runs as a CI job that CI
  Summary registers and as a pre-commit hook, at the same core pin.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
CI_WORKFLOW = WORKFLOWS / "ci.yml"
PRECOMMIT = REPO_ROOT / ".pre-commit-config.yaml"
CI_SUMMARY_GATE = REPO_ROOT / "scripts" / "ci" / "ci_summary_gate.py"

FLEET_FALLBACK = """'["self-hosted","omnibase-ci"]'"""
LAB_BROKER_CARVE_OUTS = (
    "OMNI_RUNTIME_REBUILD_RUNS_ON_JSON",
    "OMNI_HOSTILE_REVIEW_RUNS_ON_JSON",
)
JOB_NAME = "Public Runner Placement (OMN-19998)"
HOOK_ID = "check-public-runner-placement"
_CORE_PIN = re.compile(r"omnibase_core@([0-9a-f]{40})")


def _load(path: Path) -> dict[object, object]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def fleet_fallbacks(text: str) -> list[str]:
    """Code lines whose routing fallback is the self-hosted fleet, carve-outs excluded."""
    return [
        line.strip()
        for line in text.splitlines()
        if FLEET_FALLBACK in line
        and not line.lstrip().startswith("#")
        and not any(knob in line for knob in LAB_BROKER_CARVE_OUTS)
    ]


def test_fleet_fallback_detector_positive_control() -> None:
    planted = (
        "    runs-on: ${{ fromJSON(vars.OMNI_TRUSTED_CI_RUNS_ON_JSON || "
        """'["self-hosted","omnibase-ci"]') }}\n"""
    )
    carve_out = (
        "        || fromJSON(vars.OMNI_RUNTIME_REBUILD_RUNS_ON_JSON || "
        """'["self-hosted","omnibase-ci"]')\n"""
    )
    hosted = (
        "    runs-on: ${{ fromJSON(vars.OMNI_TRUSTED_CI_RUNS_ON_JSON || "
        """'["ubuntu-latest"]') }}\n"""
    )
    assert fleet_fallbacks(planted) == [planted.strip()]
    assert fleet_fallbacks(carve_out) == []
    assert fleet_fallbacks(hosted) == []


def test_no_workflow_falls_back_to_the_self_hosted_fleet() -> None:
    offenders = {
        path.name: found
        for path in sorted(WORKFLOWS.glob("*.y*ml"))
        if (found := fleet_fallbacks(path.read_text(encoding="utf-8")))
    }
    assert offenders == {}, (
        "A public repository's routing fallback must be GitHub-hosted "
        "(RULING 2026-10-16T07:07:09Z): " + repr(offenders)
    )


def test_nightly_performance_and_dev_merge_tag_are_hosted() -> None:
    nightly = _load(WORKFLOWS / "nightly-tests.yml")["jobs"]
    assert isinstance(nightly, dict)
    assert nightly["performance-tests"]["runs-on"] == "ubuntu-latest"
    tag = _load(WORKFLOWS / "dev-merge-tag.yml")["jobs"]
    assert isinstance(tag, dict)
    assert tag["dev-merge-tag"]["runs-on"] == "ubuntu-latest"


def _ci_job() -> dict[str, object]:
    jobs = _load(CI_WORKFLOW)["jobs"]
    assert isinstance(jobs, dict)
    job = jobs["public-runner-placement"]
    assert isinstance(job, dict)
    return job


def test_ci_job_runs_the_core_check_unconditionally_on_a_hosted_runner() -> None:
    job = _ci_job()
    assert job["name"] == JOB_NAME
    assert job["runs-on"] == "ubuntu-latest"
    assert "needs" not in job
    assert "if" not in job
    steps = job["steps"]
    assert isinstance(steps, list)
    runs = "\n".join(str(step.get("run", "")) for step in steps)
    assert "omnibase_core.handlers.handler_public_runner_placement_cli" in runs
    assert "POSITIVE CONTROL FAILED" in runs
    assert "--base-ref" in runs


def test_ci_job_and_hook_share_one_core_pin() -> None:
    """The job has no pin of its own: it reads the hook's rev from the config."""
    job = _ci_job()
    assert "env" not in job
    steps = job["steps"]
    assert isinstance(steps, list)
    reads = [
        str(step.get("run", ""))
        for step in steps
        if step.get("name") == "Read the core pin from .pre-commit-config.yaml"
    ]
    assert len(reads) == 1
    assert f'"id"] == "{HOOK_ID}"' in reads[0]
    assert 'RUNNER_PLACEMENT_CORE=${spec}" >> "$GITHUB_ENV"' in reads[0]
    assert not _CORE_PIN.search(yaml.safe_dump(job))


def test_ci_summary_registers_the_job() -> None:
    assert f'"{JOB_NAME}",' in CI_SUMMARY_GATE.read_text(encoding="utf-8")


def test_precommit_hook_names_this_public_repository() -> None:
    repos = _load(PRECOMMIT)["repos"]
    assert isinstance(repos, list)
    hooks = [
        hook
        for repo in repos
        if isinstance(repo, dict)
        for hook in repo.get("hooks", [])
        if isinstance(hook, dict) and hook.get("id") == HOOK_ID
    ]
    assert hooks and hooks[0]["args"] == [
        "--repo",
        "omnibase_infra",
        "--visibility",
        "public",
    ]


def test_the_lab_selection_refusal_is_in_the_required_job() -> None:
    """The refusal runs in the job CI Summary sweeps, with its failing control first."""
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["public-runner-placement"]["steps"]
    runs = [str(step.get("run", "")) for step in steps]
    control = next(i for i, run in enumerate(runs) if "FAILING CONTROL FAILED" in run)
    refusal = next(
        i
        for i, run in enumerate(runs)
        if "validator_public_lab_selection" in run and "--variables-json" in run
    )
    assert control < refusal
