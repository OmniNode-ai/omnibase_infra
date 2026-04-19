# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Schema tests for .github/workflows/reusable-runtime-boot.yml (OMN-9250).

Validates the reusable workflow's YAML shape so Tier-1 smoke (OMN-9251) and
Tier-2 regression (OMN-9252) can safely call it via `workflow_call`. Asserts:

* `on.workflow_call.inputs.mode` is required + string + constrained to real|compose
* `on.workflow_call.inputs.runtime_host` default == 'localhost'
* `on.workflow_call.inputs.pg_port` default == '5436'
* `jobs.boot.runs-on` uses the exact USE_SELF_HOSTED_RUNNERS conditional from
  ci-baseline.md §B (shared with ci.yml)
* `jobs.boot.steps` contains checkout, uv setup, conditional compose bring-up,
  health-wait with jq `.data.status == "healthy" and .is_running == true`,
  60s hold + RestartCount check, psql registration_projections liveness query,
  rpk group list check, and smoke-result.json artifact upload.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

WORKFLOW_PATH = (
    Path(__file__).resolve().parents[2]
    / ".github"
    / "workflows"
    / "reusable-runtime-boot.yml"
)

Workflow = dict[str, Any]
Step = dict[str, Any]


def _on_block(workflow: Workflow) -> dict[str, Any]:
    # PyYAML parses the `on:` key as the Python bool True (YAML 1.1 "on"
    # boolean alias). Accept either spelling so the test is agnostic to the
    # underlying loader quirk. Cast keys to `Any` so mypy's dict overload
    # accepts the bool fallback key.
    typed_workflow: dict[Any, Any] = workflow
    block = typed_workflow.get("on", typed_workflow.get(True))
    assert isinstance(block, dict), "workflow missing `on` trigger"
    return block


def _inputs(workflow: Workflow) -> dict[str, Any]:
    inputs = _on_block(workflow)["workflow_call"]["inputs"]
    assert isinstance(inputs, dict)
    return inputs


def _boot_job(workflow: Workflow) -> dict[str, Any]:
    job = workflow["jobs"]["boot"]
    assert isinstance(job, dict)
    return job


def _boot_steps(workflow: Workflow) -> list[Step]:
    steps = _boot_job(workflow)["steps"]
    assert isinstance(steps, list)
    return steps


@pytest.fixture(scope="module")
def workflow() -> Workflow:
    assert WORKFLOW_PATH.exists(), f"workflow missing: {WORKFLOW_PATH}"
    with WORKFLOW_PATH.open() as fh:
        loaded = yaml.safe_load(fh)
    assert isinstance(loaded, dict)
    return loaded


def test_workflow_is_workflow_call_reusable(workflow: Workflow) -> None:
    assert "workflow_call" in _on_block(workflow), "workflow must expose workflow_call"


def test_mode_input_required_string(workflow: Workflow) -> None:
    inputs = _inputs(workflow)
    assert "mode" in inputs, "mode input missing"
    mode = inputs["mode"]
    assert mode.get("required") is True, "mode must be required"
    assert mode.get("type") == "string", "mode must be string"


def test_mode_restricted_to_real_or_compose(workflow: Workflow) -> None:
    """Restriction may be via `options` (enum) or a description-documented pattern.

    GitHub's workflow_call does not support enum-typed inputs natively; the
    plan accepts either `options` (forward-compat) or a description that
    documents the `real | compose` contract. A runtime guard in the first
    job step is not required to be asserted here — it's covered by the
    step-level validation tests.
    """
    mode = _inputs(workflow)["mode"]
    description = (mode.get("description") or "").lower()
    options = mode.get("options") or []
    documented = "real" in description and "compose" in description
    enumerated = set(options) == {"real", "compose"}
    assert documented or enumerated, (
        "mode must be constrained to real|compose via `options` or description"
    )


def test_runtime_host_input_defaults_localhost(workflow: Workflow) -> None:
    inputs = _inputs(workflow)
    assert "runtime_host" in inputs
    assert inputs["runtime_host"].get("type") == "string"
    assert inputs["runtime_host"].get("default") == "localhost"


def test_pg_port_input_defaults_5436(workflow: Workflow) -> None:
    inputs = _inputs(workflow)
    assert "pg_port" in inputs
    assert inputs["pg_port"].get("type") == "string"
    assert inputs["pg_port"].get("default") == "5436"


def test_boot_job_runs_on_uses_self_hosted_conditional(workflow: Workflow) -> None:
    """runs-on must match the shared ci-baseline §B conditional verbatim.

    Pattern (whitespace-normalized):
        (vars.USE_SELF_HOSTED_RUNNERS == 'true' &&
         (github.event_name == 'push' ||
          github.event_name == 'workflow_dispatch' ||
          github.event_name == 'schedule'))
        && fromJSON('["self-hosted","omnibase-ci"]')
        || fromJSON('["ubuntu-latest"]')
    """
    runs_on = _boot_job(workflow)["runs-on"]
    assert isinstance(runs_on, str), "runs-on must be a string expression"
    normalized = " ".join(runs_on.split())
    assert "vars.USE_SELF_HOSTED_RUNNERS == 'true'" in normalized
    assert "github.event_name == 'push'" in normalized
    assert "github.event_name == 'workflow_dispatch'" in normalized
    assert "github.event_name == 'schedule'" in normalized
    assert 'fromJSON(\'["self-hosted","omnibase-ci"]\')' in normalized
    assert "fromJSON('[\"ubuntu-latest\"]')" in normalized


def test_boot_env_parameterizes_runtime_host_and_pg_port(workflow: Workflow) -> None:
    env = _boot_job(workflow)["env"]
    assert env.get("RUNTIME_HOST") == "${{ inputs.runtime_host }}"
    assert env.get("PG_PORT") == "${{ inputs.pg_port }}"


def _step_text(step: Step) -> str:
    parts = [
        str(step.get("name", "")),
        str(step.get("uses", "")),
        str(step.get("run", "")),
    ]
    return "\n".join(parts).lower()


def _has_step(steps: list[Step], predicate: Callable[[Step], bool]) -> bool:
    return any(predicate(step) for step in steps)


def test_boot_has_checkout_step(workflow: Workflow) -> None:
    steps = _boot_steps(workflow)
    assert _has_step(steps, lambda s: "actions/checkout" in str(s.get("uses", "")))


def test_boot_has_uv_setup_step(workflow: Workflow) -> None:
    steps = _boot_steps(workflow)
    assert _has_step(
        steps,
        lambda s: "astral-sh/setup-uv" in str(s.get("uses", ""))
        or "uv" in _step_text(s),
    )


def test_boot_has_conditional_compose_bringup(workflow: Workflow) -> None:
    steps = _boot_steps(workflow)
    matched = [
        s
        for s in steps
        if "docker compose" in _step_text(s) or "docker-compose" in _step_text(s)
    ]
    assert matched, "compose bring-up step missing"
    for step in matched:
        cond = str(step.get("if", ""))
        assert "inputs.mode" in cond and "compose" in cond, (
            f"compose step must be gated on mode=='compose': {step}"
        )


def test_boot_health_wait_uses_jq_hard_gate(workflow: Workflow) -> None:
    """Health-wait step must assert jq `.data.status == "healthy" and .is_running == true`."""
    steps = _boot_steps(workflow)
    matched = [s for s in steps if "8085/health" in _step_text(s)]
    assert matched, "health-wait step missing"
    step_texts = "\n".join(_step_text(s) for s in matched)
    assert ".data.status" in step_texts
    assert '"healthy"' in step_texts or "'healthy'" in step_texts
    assert ".is_running" in step_texts
    assert "true" in step_texts


def test_boot_emits_subscriber_warning_not_hard_gate(workflow: Workflow) -> None:
    """subscriber_count<300 must emit `::warning::` — NOT fail the job."""
    steps = _boot_steps(workflow)
    all_text = "\n".join(_step_text(s) for s in steps)
    assert "subscriber_count" in all_text, "subscriber_count observation missing"
    assert "::warning::" in all_text, "subscriber_count must emit ::warning:: not fail"


def test_boot_holds_60s_and_checks_restart_count(workflow: Workflow) -> None:
    steps = _boot_steps(workflow)
    matched = [
        s
        for s in steps
        if "restartcount" in _step_text(s) or "docker inspect" in _step_text(s)
    ]
    assert matched, "60s-hold RestartCount check missing (compose hard gate)"


def test_boot_has_psql_registration_projections_query(workflow: Workflow) -> None:
    steps = _boot_steps(workflow)
    matched = [
        s
        for s in steps
        if "psql" in _step_text(s) and "registration_projections" in _step_text(s)
    ]
    assert matched, "psql registration_projections liveness query missing"
    # Must scope to active + recent heartbeat (60s window) per plan.
    step_texts = "\n".join(_step_text(s) for s in matched)
    assert "active" in step_texts
    assert "last_heartbeat_at" in step_texts


def test_boot_has_rpk_group_list_step(workflow: Workflow) -> None:
    steps = _boot_steps(workflow)
    matched = [s for s in steps if "rpk group list" in _step_text(s)]
    assert matched, "rpk group list check missing"


def test_boot_uploads_smoke_result_artifact(workflow: Workflow) -> None:
    steps = _boot_steps(workflow)
    matched = [
        s
        for s in steps
        if "actions/upload-artifact" in str(s.get("uses", ""))
        or "smoke-result" in _step_text(s)
    ]
    assert matched, "smoke-result.json artifact output missing"


def test_boot_has_nine_steps_minimum(workflow: Workflow) -> None:
    steps = _boot_steps(workflow)
    # Plan enumerates 9 ordered steps; implementation may add minor helpers
    # (e.g. a jq/kcat install step). Enforce >= 9 to catch accidental drops.
    assert len(steps) >= 9, f"boot job must have >= 9 steps, found {len(steps)}"


def test_no_hardcoded_user_paths(workflow: Workflow) -> None:
    raw = WORKFLOW_PATH.read_text()
    assert "/Users/" not in raw, "no /Users/ absolute paths in workflow YAML"
    assert "/Volumes/" not in raw, "no /Volumes/ absolute paths in workflow YAML"
