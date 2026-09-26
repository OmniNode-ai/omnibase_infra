# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pin whole-tree CI counterparts for OMN-19612's staged-file hooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci.ci_summary_gate import SOFT_ALLOWLIST, STRICT_GATE_JOBS
from scripts.ci.detect_test_paths import (
    PRE_COMMIT_CONFIG_PATH,
    compute_selection,
)

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
PRECOMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"
REQUIRED_CHECKS = REPO_ROOT / ".github" / "required-checks.yaml"
ADJ = REPO_ROOT / "scripts" / "ci" / "test_selection_adjacency.yaml"
THIS_TEST = "tests/ci/test_staged_precommit_backstops_omn19612.py"

# These hooks were already staged-file scoped before OMN-19612. Keeping the
# baseline explicit makes a newly staged-scoped hook fail this test until its
# whole-tree counterpart is added below; the baseline may shrink, not grow.
PRE_EXISTING_STAGED_HOOKS = frozenset(
    {
        "check-lockfile-registry-allowlist",
        "check-pin-reachability",
        "handler-routing-schema-gate",
        "lint-pricing-manifest",
        "onex-imperative-orchestrator-ratchet",
        "onex-orchestration-monolith-ratchet",
        "onex-orchestrator-reducer-state-invariant",
        "onex-validate-backend-secret-discipline",
        "onex-validate-declarative-nodes",
        "operation-match-requires-operation",
        "tenant-scoped-ingress-gate",
        # These were already pass_filenames: true before commit c30aa2504dd9
        # ("fix: scope local validators to staged files (OMN-19612)") -- not
        # moved to staged scope by this PR, so a whole-tree backstop for them
        # is out of scope here. Discovered once _staged_scoped_hook_ids()
        # stopped requiring an explicit files: regex (a types_or: filter is
        # just as much a staged-scope signal as files: is).
        "check-ai-slop",
        "exposed-identifier-gate",
        "no-env-fallbacks",
        "onex-validate-markdown-links",
        "reject-deploy-gate-skip-token",
        "shell-hygiene",
    }
)

BACKSTOPS = {
    "check-no-credential-in-log": (
        "onex-validation",
        "scripts/ci/check_no_credential_in_log.py",
        "--root src/omnibase_infra",
    ),
    "handler-any-signature": (
        "lint",
        "omnibase_infra.validators.handler_any_signature",
        "src/omnibase_infra",
    ),
    "envelope-tenant-dimension": (
        "lint",
        "omnibase_infra.validators.envelope_tenant_dimension",
        "src/omnibase_infra",
    ),
    "kafka-no-hardcoded-fallback": (
        "lint",
        "scripts/validation/check_kafka_no_hardcoded_fallback.sh",
    ),
    "no-infra-inmemory-import": (
        "lint",
        "scripts/validation/check_no_infra_inmemory_import.sh",
    ),
    "validate-spdx-headers": (
        "onex-validation",
        "onex spdx validate src tests scripts",
    ),
}

# Steps in a backstop's job that are legitimately advisory (report-only
# ratchets unrelated to any staged-scoped hook) and so are allowed to carry
# continue-on-error: true without failing the fail-closed check below. Only
# the step(s) that actually implement a BACKSTOPS whole-tree run are checked.
ADVISORY_STEP_NAMES = frozenset(
    {
        "Run imperative-orchestrator ratchet report (ARCH-004)",
        "Run orchestration-monolith ratchet report (ARCH-004 Signal B)",
    }
)
ALLOWED_BACKSTOP_STEP_CONDITIONS = frozenset({"always()", "success()", "!cancelled()"})


def _load_yaml(path: Path) -> dict[object, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict), f"{path.name} did not parse to a mapping"
    return loaded


def _staged_scoped_hook_ids() -> set[str]:
    """Hooks pre-commit hands only the staged diff at commit time.

    pass_filenames: true is the actual staged-scope signal -- it is what
    limits the hook's file arguments to the staged diff. A files: regex or
    a types_or: filter narrows WHICH staged files qualify; neither is
    required for the hook to be staged-scoped, and a hook that filters by
    types_or alone (no files:) is exactly as staged-scoped as one that uses
    files:.
    """
    config = _load_yaml(PRECOMMIT_CONFIG)
    return {
        str(hook["id"])
        for repo in config["repos"]
        for hook in repo.get("hooks", [])
        if hook.get("pass_filenames") is True
        and "pre-commit" in hook.get("stages", ["pre-commit"])
    }


def _workflow() -> dict[object, Any]:
    return _load_yaml(CI_WORKFLOW)


def _job(job_id: str) -> dict[str, Any]:
    job = _workflow()["jobs"][job_id]
    assert isinstance(job, dict)
    return job


def _commands(job_id: str) -> str:
    return "\n".join(str(step.get("run", "")) for step in _job(job_id)["steps"])


def _normalise_condition(value: object) -> str:
    condition = str(value).strip()
    return condition.removeprefix("${{").removesuffix("}}").strip()


def test_every_new_staged_hook_has_a_declared_whole_tree_backstop() -> None:
    staged = _staged_scoped_hook_ids()
    assert staged >= PRE_EXISTING_STAGED_HOOKS
    assert staged - PRE_EXISTING_STAGED_HOOKS == set(BACKSTOPS)


@pytest.mark.parametrize(("hook_id", "backstop"), BACKSTOPS.items())
def test_whole_tree_counterpart_is_in_its_required_job(
    hook_id: str, backstop: tuple[str, ...]
) -> None:
    job_id, *required_fragments = backstop
    job = _job(job_id)
    commands = _commands(job_id)
    for fragment in required_fragments:
        assert fragment in commands, f"{hook_id} lost whole-tree fragment {fragment!r}"

    backstop_steps = [
        step
        for step in job["steps"]
        if all(fragment in str(step.get("run", "")) for fragment in required_fragments)
    ]
    assert backstop_steps, (
        f"{hook_id} has no single step in {job_id} containing all whole-tree "
        f"fragments: {required_fragments!r}"
    )
    for step in backstop_steps:
        if "if" not in step:
            continue
        condition = _normalise_condition(step["if"])
        step_name = str(step.get("name", "<unnamed>"))
        assert condition in ALLOWED_BACKSTOP_STEP_CONDITIONS, (
            f"{hook_id} whole-tree step {step_name!r} has conditional `if`: "
            f"{step['if']!r}; allowed unconditional forms are "
            f"{sorted(ALLOWED_BACKSTOP_STEP_CONDITIONS)!r}"
        )

    assert "if" not in job, f"{job_id} acquired a job-level condition"
    assert job.get("continue-on-error") is not True
    checked_steps = [
        step
        for step in job["steps"]
        if str(step.get("name", "")) not in ADVISORY_STEP_NAMES
    ]
    assert checked_steps, f"{job_id} has no non-advisory steps to check"
    assert all(step.get("continue-on-error") is not True for step in checked_steps), (
        f"a non-advisory step of {job_id} is continue-on-error, so {hook_id}'s "
        "whole-tree run can fail silently"
    )

    job_name = str(job["name"])
    assert job_name in STRICT_GATE_JOBS
    assert job_name not in SOFT_ALLOWLIST


def test_this_pin_is_selected_for_changes_to_its_inputs() -> None:
    for changed_files in (
        [PRE_COMMIT_CONFIG_PATH],
        [".github/workflows/ci.yml"],
    ):
        selection = compute_selection(
            changed_files=changed_files,
            adjacency_path=ADJ,
            ref_name="dev",
            event_name="pull_request",
            feature_flag_enabled=True,
        )
        assert selection.is_full_suite or any(
            THIS_TEST.startswith(selected_path)
            for selected_path in selection.selected_paths
        ), (
            f"{changed_files!r} does not select {THIS_TEST}: "
            f"{selection.selected_paths!r}"
        )


def test_pytest_job_does_not_path_gate_this_pin() -> None:
    condition = str(_job("test-parallel").get("if", "")).lower()
    for path_signal in ("path", ".pre-commit-config", "ci.yml"):
        assert path_signal not in condition, (
            "test-parallel path-gates the OMN-19612 pin via its job-level "
            f"condition: {condition!r}"
        )


def test_required_summary_and_workflow_trigger_cannot_drop_the_backstops() -> None:
    workflow = _workflow()
    triggers = workflow[True] if True in workflow else workflow["on"]
    pull_request = triggers.get("pull_request") or {}
    assert "paths" not in pull_request
    assert "paths-ignore" not in pull_request

    summary = _job("ci-summary")
    assert summary["name"] == "CI Summary"
    assert "needs" not in summary
    required = _load_yaml(REQUIRED_CHECKS)
    required_names = {
        gate["name"] for gate in required["gates"] if gate.get("mode") == "REQUIRED"
    }
    assert "CI Summary" in required_names
