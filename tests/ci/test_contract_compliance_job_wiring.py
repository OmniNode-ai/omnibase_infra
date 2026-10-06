# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20135: the Contract Compliance Check job's wiring in ``ci.yml``.

The job used to read ``contracts/<ticket>.yaml`` from the change-control
checkout that also pins the checker code (7352cb2d, 2026-07-12). A contract
created after the pin was therefore absent, and the pinned runner's
missing-contract branch passed the PR with nothing executed (omnibase_infra#4325,
job 109722692581). These tests pin the replacement, mirrored from omnibase_core
(OMN-18157, OMN-20130):

* the checker code stays at one immutable pin, in its own checkout;
* the contracts come from a second checkout at the commit the PR's evidence
  reference resolves to, never from the checker checkout;
* the run goes through the fail-closed wrapper and the PR-scoping and
  ``test_passes``-deferring driver;
* the deferral record is a declared evidence artifact, and CI Summary judges the
  deferred items after its own verdict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_YML = REPO_ROOT / ".github/workflows/ci.yml"
POLICY = REPO_ROOT / "config/ci_evidence_policy.yaml"

# onex_change_control 91f5b691 (2026-07-24): the checker omnibase_core pins. It
# is the first pin that ships the importable
# ``onex_change_control.scripts.contract_compliance_check`` module the wrapper
# and the driver load; the old 7352cb2d pin ships only the script.
CHECKER_PIN = "91f5b69104366c2d86e4bb21de67c3c497702111"  # pragma: allowlist secret
OLD_PIN = "7352cb2dc634a7223877bfbbf3498303c9662700"  # pragma: allowlist secret
RUN_STEP = "Run DoD contract compliance against the product tree (OMN-20135)"
EXEMPT_STEP_OUTPUT = "steps.dep-scope.outputs.exempt"
RECORD = "contract-compliance-deferred/record.json"


def _jobs() -> dict[str, Any]:
    return yaml.safe_load(CI_YML.read_text(encoding="utf-8"))["jobs"]


def _steps(job: str) -> list[dict[str, Any]]:
    return list(_jobs()[job]["steps"])


def _step(job: str, name: str) -> dict[str, Any]:
    matches = [s for s in _steps(job) if s.get("name") == name]
    assert len(matches) == 1, f"{job}: expected one step named {name!r}"
    return matches[0]


def _occ_checkouts() -> dict[str, dict[str, Any]]:
    return {
        str(s["with"].get("path")): s
        for s in _steps("contract-compliance")
        if str(s.get("uses", "")).startswith("actions/checkout@")
        and (s.get("with") or {}).get("repository") == "OmniNode-ai/onex_change_control"
    }


def test_checker_and_evidence_are_separate_checkouts() -> None:
    checkouts = _occ_checkouts()
    assert set(checkouts) == {
        "onex_change_control_checker",
        "onex_change_control_evidence",
    }, "the contracts must never be read from the checker checkout"
    assert checkouts["onex_change_control_checker"]["with"]["ref"] == CHECKER_PIN
    assert (
        checkouts["onex_change_control_evidence"]["with"]["ref"]
        == "${{ steps.resolve_contract_compliance_evidence.outputs.occ_sha }}"
    )


def test_the_old_pin_that_lacked_every_new_contract_is_gone() -> None:
    assert OLD_PIN not in CI_YML.read_text(encoding="utf-8")


def test_evidence_resolution_is_scoped_to_the_resolved_pr() -> None:
    names = [s.get("name") for s in _steps("contract-compliance")]
    resolve_pr = names.index(
        "Resolve the PR whose DoD check_values govern this run (OMN-16508)"
    )
    resolve_evidence = names.index(
        "Resolve contract-compliance evidence data (OMN-20135)"
    )
    assert resolve_pr < resolve_evidence
    step = _step(
        "contract-compliance", "Resolve contract-compliance evidence data (OMN-20135)"
    )
    assert step["id"] == "resolve_contract_compliance_evidence"
    assert "scripts/ci/resolve_contract_compliance_evidence.py" in step["run"]
    assert step["env"]["PR_NUMBER"] == "${{ steps.dod-pr.outputs.pr_number }}"
    assert '--pr-number "${PR_NUMBER}"' in step["run"]
    assert "--github-output" in step["run"]


def test_the_run_goes_through_the_fail_closed_wrapper() -> None:
    step = _step("contract-compliance", RUN_STEP)
    run = step["run"]
    assert "scripts/ci/run_contract_compliance_with_evidence.py" in run
    assert "run_contract_compliance_check.py" not in run
    assert step["env"]["PR_NUMBER"] == (
        "${{ steps.resolve_contract_compliance_evidence.outputs.pr_number }}"
    )
    assert '--pr "${PR_NUMBER}"' in run
    assert "$GITHUB_WORKSPACE/onex_change_control_evidence/contracts" in run
    assert '--workspace "$GITHUB_WORKSPACE"' in run
    assert (
        "$GITHUB_WORKSPACE/onex_change_control_checker/scripts/ci/"
        "dod_runner_legacy_allowlist.txt"
    ) in run
    assert f'--deferred-record "$GITHUB_WORKSPACE/{RECORD}"' in run
    # The step has no `if:`: it is the declared producer of the deferral
    # record on both branches (evaluation, and the declared exemption).
    assert "if" not in step
    assert EXEMPT_STEP_OUTPUT in str(step.get("env", {}).get("EXEMPT", ""))


def test_the_exemption_branch_writes_an_empty_record_and_runs_nothing() -> None:
    run = _step("contract-compliance", RUN_STEP)["run"]
    exempt_branch, _, evaluated_branch = run.partition("else")
    assert '"deferred": []' in exempt_branch
    assert "run_contract_compliance_with_evidence.py" not in exempt_branch
    assert "run_contract_compliance_with_evidence.py" in evaluated_branch


def test_the_exemption_is_classified_by_the_declared_rule() -> None:
    step = _step(
        "contract-compliance",
        "Classify the declared ticketless dependency-bot exemption (OMN-20135)",
    )
    assert step["id"] == "dep-scope"
    assert "scripts.ci.classify_contract_compliance_scope" in step["run"]
    env = step["env"]
    assert env["PR_AUTHOR"] == "${{ github.event.pull_request.user.login }}"
    assert env["PR_TITLE"] == "${{ github.event.pull_request.title }}"
    assert env["PR_HEAD_REF"] == "${{ github.event.pull_request.head.ref }}"
    assert env["MERGE_GROUP_REF"] == "${{ github.event.merge_group.head_ref }}"
    assert "--merge-group-ref" in step["run"]


@pytest.mark.parametrize(
    "name",
    [
        "Mint onexbot-occ-writer app token",
        "Checkout onex_change_control checker (pinned code)",
        "Install onex_change_control checker",
        "Resolve contract-compliance evidence data (OMN-20135)",
        "Checkout onex_change_control evidence data (OMN-20135)",
    ],
)
def test_evidence_steps_are_skipped_only_on_the_declared_exemption(name: str) -> None:
    assert _step("contract-compliance", name)["if"] == (
        f"{EXEMPT_STEP_OUTPUT} != 'true'"
    )


def test_the_deferral_record_is_a_declared_evidence_artifact() -> None:
    steps = _steps("contract-compliance")
    names = [s.get("name") for s in steps]
    assertion = names.index("Assert the deferral record is present and non-empty")
    upload = names.index("Upload deferred test_passes record (OMN-20135)")
    assert names.index(RUN_STEP) < assertion < upload
    assert steps[assertion]["if"] == "always()"
    assert f"--require {RECORD}" in steps[assertion]["run"]
    uploader = steps[upload]
    assert str(uploader["uses"]).startswith("actions/upload-artifact@")
    assert uploader["with"]["name"] == "contract-compliance-deferred"
    assert uploader["with"]["path"] == RECORD
    assert uploader["with"]["if-no-files-found"] == "error"
    assert uploader["with"]["overwrite"] is True
    policy = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    entry = next(
        e
        for e in policy["evidence_artifacts"]
        if e["id"] == "contract-compliance-deferred"
    )
    assert entry["job"] == "contract-compliance"
    assert entry["producer_step"] == RUN_STEP
    assert entry["required_paths"] == [RECORD]


def test_ci_summary_judges_the_deferred_items_after_its_verdict() -> None:
    job = _jobs()["ci-summary"]
    names = [s.get("name") for s in job["steps"]]
    poll = names.index("Poll run jobs and compute fail-closed CI Summary verdict")
    deferred = names.index("Evaluate deferred test_passes DoD items (OMN-20135)")
    assert deferred == poll + 1
    step = job["steps"][deferred]
    assert "if" not in step, "must run only when the poll step succeeded"
    assert "scripts.ci.deferred_test_passes_gate" in step["run"]
    assert "--name contract-compliance-deferred" in step["run"]
    assert "--jobs-file jobs.json" in step["run"]
    assert job["permissions"]["pull-requests"] == "read"
    assert job["permissions"]["actions"] == "read"
    # The poll deadline (90m) plus the deferred deadline (40m) must fit inside
    # the job's hard backstop, or the backstop kills the required context.
    assert "--deadline-seconds 2400" in step["run"]
    assert int(job["timeout-minutes"]) >= 90 + 40 + 10
