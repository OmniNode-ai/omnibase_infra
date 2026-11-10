# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The retired k3s lab-overlay job is gone and nothing still waits on it.

OMN-23508 (child of OMN-23422). The post-merge job ``verify-lab-overlay-converged``
probed the dead lab k3s pod and emitted that pod's lab-pass receipt. It was
removed from ``.github/workflows/runtime-rebuild-trigger.yml``, its entry was
removed from ``config/ci_evidence_policy.yaml``, and the staging gate step no
longer selects the lane that receipt belonged to.

These pin the workflow graph, not prose:

* the retired job is absent and the evidence policy names no producer for it;
* no job ``needs`` a job its own workflow does not define;
* every producer job the evidence policy names exists;
* a lab-pass gate never selects a lane that no workflow step emits, which is the
  shape of "waiting on a receipt nobody writes any more".
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
EVIDENCE_POLICY = REPO_ROOT / "config" / "ci_evidence_policy.yaml"
REBUILD_TRIGGER = WORKFLOWS_DIR / "runtime-rebuild-trigger.yml"
RETIRED_JOB = "verify-lab-overlay-converged"

# `--lane X` as its own flag, never the tail of `--require-lane X`.
_LANE_FLAG = re.compile(r"(?<![\w-])--lane\s+([A-Za-z0-9][\w-]*)")


def _workflow_files() -> list[Path]:
    return sorted(WORKFLOWS_DIR.glob("*.yml"))


def _load(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def dangling_needs(workflow: dict[str, Any]) -> list[tuple[str, str]]:
    """Return ``(job, missing_target)`` for every ``needs`` target with no job."""
    jobs = workflow.get("jobs") or {}
    out: list[tuple[str, str]] = []
    for job_name, job in jobs.items():
        needs = (job or {}).get("needs")
        if needs is None:
            continue
        targets = [needs] if isinstance(needs, str) else list(needs)
        out.extend((job_name, t) for t in targets if t not in jobs)
    return out


def run_scripts(workflow: dict[str, Any]) -> list[str]:
    """Every step ``run`` script in the workflow."""
    return [
        step["run"]
        for job in (workflow.get("jobs") or {}).values()
        for step in (job or {}).get("steps") or []
        if isinstance(step, dict) and isinstance(step.get("run"), str)
    ]


def lanes_selected_but_never_emitted(scripts: list[str]) -> set[str]:
    """Lanes a lab-pass ``gate`` reads that no lab-pass ``emit`` step writes."""
    emitted: set[str] = set()
    selected: set[str] = set()
    for script in scripts:
        if "lab_pass_receipt.py emit" in script:
            emitted.update(_LANE_FLAG.findall(script))
        if "lab_pass_receipt.py gate" in script:
            selected.update(_LANE_FLAG.findall(script))
    return selected - emitted


@pytest.mark.parametrize("path", _workflow_files(), ids=lambda p: p.name)
def test_every_needs_target_names_a_job_in_the_same_workflow(path: Path) -> None:
    dangling = dangling_needs(_load(path))
    assert not dangling, (
        f"{path.name}: needs targets that name no job in the same workflow: "
        + ", ".join(f"{job} -> {target}" for job, target in dangling)
    )


def test_the_retired_job_is_gone() -> None:
    assert RETIRED_JOB not in (_load(REBUILD_TRIGGER).get("jobs") or {}), (
        f"{REBUILD_TRIGGER.name} still defines the retired job {RETIRED_JOB}"
    )


def test_the_delivery_reread_waits_only_on_jobs_that_exist() -> None:
    jobs = _load(REBUILD_TRIGGER)["jobs"]
    needs = jobs["rerun-refused-deliveries"]["needs"]
    assert RETIRED_JOB not in needs
    assert set(needs) <= set(jobs)


def test_the_evidence_policy_names_no_producer_for_the_retired_job() -> None:
    policy = yaml.safe_load(EVIDENCE_POLICY.read_text(encoding="utf-8"))
    producers = {
        (e.get("workflow"), e.get("job")) for e in policy["evidence_artifacts"]
    }
    assert (
        ".github/workflows/runtime-rebuild-trigger.yml",
        RETIRED_JOB,
    ) not in producers


def test_every_evidence_policy_producer_job_exists() -> None:
    policy = yaml.safe_load(EVIDENCE_POLICY.read_text(encoding="utf-8"))
    problems: list[str] = []
    for entry in policy["evidence_artifacts"]:
        if "workflow" not in entry or "job" not in entry:
            continue
        workflow_path = REPO_ROOT / entry["workflow"]
        if not workflow_path.is_file():
            problems.append(f"{entry['id']}: workflow {entry['workflow']} is missing")
        elif entry["job"] not in (_load(workflow_path).get("jobs") or {}):
            problems.append(
                f"{entry['id']}: job {entry['job']} not in {entry['workflow']}"
            )
    assert not problems, "; ".join(problems)


def test_no_gate_selects_a_lane_that_no_step_emits() -> None:
    scripts: list[str] = []
    for path in _workflow_files():
        scripts.extend(run_scripts(_load(path)))
    assert lanes_selected_but_never_emitted(scripts) == set()


def test_the_dangling_needs_check_fails_on_a_removed_job() -> None:
    removed = {"jobs": {"a": {}, "b": {"needs": ["a", "gone"]}}}
    assert dangling_needs(removed) == [("b", "gone")]
    assert dangling_needs({"jobs": {"a": {}, "b": {"needs": "a"}}}) == []


def test_the_selected_but_never_emitted_check_fails_on_an_orphaned_gate() -> None:
    emit = "uv run python scripts/ci/lab_pass_receipt.py emit --lane alpha --sha x"
    gate = (
        "python3 scripts/ci/lab_pass_receipt.py gate --lane beta "
        "--require-lane alpha --sha x"
    )
    assert lanes_selected_but_never_emitted([emit, gate]) == {"beta"}
    assert (
        lanes_selected_but_never_emitted([emit, gate.replace("beta", "alpha")]) == set()
    )
