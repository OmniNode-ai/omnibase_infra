# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime Boot Smoke (compose) is strict on merge_group (OMN-20147).

The fail-closed verdict suite for the whole poller lives in
``tests/ci/test_ci_summary_gate.py``. This file pins one tier added on top of
it: on a ``merge_group`` build, CI Summary requires the real-runtime boot
(ONEX runtime and runtime-effects over compose Postgres and Redpanda, plus the
two runtime-backed suites) to be present, completed and ``success``. On
``pull_request`` the job does not run and the verdict is what it was before.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci.ci_summary_gate import (
    EXIT_FAILURE,
    EXIT_PENDING,
    EXIT_SUCCESS,
    MERGE_GROUP_STRICT_GATE_JOBS,
    RUNTIME_BOOT_SMOKE_COMPOSE_GATE,
    SKIPPABLE_GATE_JOBS,
    SOFT_ALLOWLIST,
    STRICT_GATE_JOBS,
    evaluate,
    main,
    strict_gates_for_event,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
REUSABLE_BOOT = REPO_ROOT / ".github" / "workflows" / "reusable-runtime-boot.yml"
CALLER = "Runtime Boot Smoke (compose)"
CATALOG_LOCAL_INNER = f"{CALLER} / runtime-boot (mode=catalog-local)"


def _job(
    name: str, conclusion: str | None, status: str = "completed"
) -> dict[str, Any]:
    return {"name": name, "status": status, "conclusion": conclusion, "run_attempt": 1}


def _gates_green() -> list[dict[str, Any]]:
    return [_job(g, "success") for g in (*STRICT_GATE_JOBS, *SKIPPABLE_GATE_JOBS)]


def _merge_group(jobs: list[dict[str, Any]]) -> tuple[int, str]:
    return evaluate(
        jobs, strict_gates=strict_gates_for_event("merge_group"), sweep_external=False
    )


def _pull_request(jobs: list[dict[str, Any]]) -> tuple[int, str]:
    return evaluate(
        jobs, strict_gates=strict_gates_for_event("pull_request"), sweep_external=False
    )


def test_runtime_boot_gate_name_is_the_compose_inner_job() -> None:
    assert f"{CALLER} / runtime-boot (mode=compose)" == RUNTIME_BOOT_SMOKE_COMPOSE_GATE
    assert MERGE_GROUP_STRICT_GATE_JOBS == (RUNTIME_BOOT_SMOKE_COMPOSE_GATE,)


def test_runtime_boot_strict_only_on_merge_group() -> None:
    assert RUNTIME_BOOT_SMOKE_COMPOSE_GATE in strict_gates_for_event("merge_group")
    for event in ("pull_request", "push", "workflow_dispatch"):
        assert strict_gates_for_event(event) == STRICT_GATE_JOBS


def test_runtime_boot_merge_group_success_passes() -> None:
    jobs = _gates_green() + [
        _job(RUNTIME_BOOT_SMOKE_COMPOSE_GATE, "success"),
        _job(CATALOG_LOCAL_INNER, "skipped"),
    ]
    code, report = _merge_group(jobs)
    assert code == EXIT_SUCCESS, report
    assert f"- {RUNTIME_BOOT_SMOKE_COMPOSE_GATE}: completed/success" in report


@pytest.mark.parametrize("conclusion", ["failure", "skipped", "cancelled", "timed_out"])
def test_runtime_boot_merge_group_non_success_fails(conclusion: str) -> None:
    jobs = _gates_green() + [_job(RUNTIME_BOOT_SMOKE_COMPOSE_GATE, conclusion)]
    code, report = _merge_group(jobs)
    assert code == EXIT_FAILURE, report
    assert RUNTIME_BOOT_SMOKE_COMPOSE_GATE in report.split("strict-gate failures:")[1]


def test_runtime_boot_merge_group_absent_never_passes() -> None:
    """Absent is PENDING, which the poller turns into FAILURE at its deadline."""
    code, report = _merge_group(_gates_green())
    assert code == EXIT_PENDING, report
    assert RUNTIME_BOOT_SMOKE_COMPOSE_GATE in report


def test_runtime_boot_merge_group_caller_skipped_never_passes() -> None:
    """A skipped caller leaves only the bare caller row, so the gate reads absent.

    The caller skips on merge_group only when occ-preflight or tests-gate did
    not succeed, and both are STRICT, so that run already fails; alone, the
    skip holds CI Summary PENDING into the deadline, never SUCCESS.
    """
    code, report = _merge_group(_gates_green() + [_job(CALLER, "skipped")])
    assert code == EXIT_PENDING, report


def test_runtime_boot_merge_group_running_is_pending() -> None:
    jobs = _gates_green() + [_job(RUNTIME_BOOT_SMOKE_COMPOSE_GATE, None, "in_progress")]
    code, _ = _merge_group(jobs)
    assert code == EXIT_PENDING


def test_runtime_boot_pull_request_skipped_caller_passes() -> None:
    """On pull_request the job is skipped by its caller `if:`; unchanged verdict."""
    code, report = _pull_request(_gates_green() + [_job(CALLER, "skipped")])
    assert code == EXIT_SUCCESS, report
    assert RUNTIME_BOOT_SMOKE_COMPOSE_GATE not in report


def test_runtime_boot_pull_request_absent_passes() -> None:
    code, report = _pull_request(_gates_green())
    assert code == EXIT_SUCCESS, report


def test_runtime_boot_stays_advisory_off_merge_group() -> None:
    """workflow_dispatch and a main push keep the advisory reading."""
    assert CALLER in SOFT_ALLOWLIST
    jobs = _gates_green() + [_job(RUNTIME_BOOT_SMOKE_COMPOSE_GATE, "failure")]
    code, report = _pull_request(jobs)
    assert code == EXIT_SUCCESS, report


def test_runtime_boot_cli_merge_group_fails_on_failure(tmp_path: Path) -> None:
    """The CLI entry point the poller calls wires --event-name to the tier."""
    import json

    jobs = _gates_green() + [_job(RUNTIME_BOOT_SMOKE_COMPOSE_GATE, "failure")]
    jobs_file = tmp_path / "jobs.json"
    jobs_file.write_text(json.dumps(jobs), encoding="utf-8")
    assert main(["--jobs-file", str(jobs_file), "--event-name", "merge_group"]) == (
        EXIT_FAILURE
    )


def _load(path: Path) -> dict[Any, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_runtime_boot_gate_name_matches_the_workflows() -> None:
    """A rename in ci.yml or the reusable would otherwise wedge every queue entry."""
    ci = _load(CI_WORKFLOW)
    caller = ci["jobs"]["runtime-boot-smoke"]
    assert caller["name"] == CALLER
    assert caller["uses"] == "./.github/workflows/reusable-runtime-boot.yml"
    assert caller["with"]["mode"] == "compose"
    assert "github.event_name == 'merge_group'" in caller["if"]
    triggers = ci.get("on", ci.get(True))
    assert isinstance(triggers, dict)
    assert "merge_group" in triggers

    boot = _load(REUSABLE_BOOT)["jobs"]["boot"]
    inner = boot["name"].replace("${{ inputs.mode }}", "compose")
    assert f"{CALLER} / {inner}" == RUNTIME_BOOT_SMOKE_COMPOSE_GATE
