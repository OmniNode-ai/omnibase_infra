# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22364: the commit stage stays inside its time budget, and no check is lost.

A commit that touched a few files ran 218 hook entries here, 105 of them on a
probe commit of five files, and took 300 s on an idle lab host and 10 to 25
minutes on a loaded one. One-runtime lanes ended at their runtime limit with
nothing pushed. The cost was whole-repo scans and suites, each its own
``uv run`` process, that already run as required CI jobs.

Those hooks now carry ``stages: [manual]``. A manual hook never runs at commit
and is never auto-installed (``pre-commit run --hook-stage manual --all-files``
runs them by hand). This file is the registry that keeps the move honest. A
manual hook must name the CI job that enforces it, and that job must

* exist in the workflow it is registered under,
* be gated by CI Summary (``STRICT_GATE_JOBS``, ``SKIPPABLE_GATE_JOBS`` or an
  expected external context in ``scripts/ci/ci_summary_gate.py``),
* have no condition that can skip it other than ``always()``, and not be
  ``continue-on-error``,
* run the hook's command (every needle appears in the job's ``run`` steps).

Falsifier: set ``stages: [manual]`` on any hook that is not registered, drop or
rename a registered job, remove its command from the job, or remove the job
from CI Summary's gate list, and a named test here fails.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci.ci_summary_gate import (
    EXPECTED_EXTERNAL_CONTEXTS,
    SKIPPABLE_GATE_JOBS,
    STRICT_GATE_JOBS,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
PRECOMMIT = REPO_ROOT / ".pre-commit-config.yaml"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
VALIDATE_PY = REPO_ROOT / "scripts" / "validate.py"

GATED_CONTEXTS: frozenset[str] = frozenset(
    (*STRICT_GATE_JOBS, *SKIPPABLE_GATE_JOBS, *EXPECTED_EXTERNAL_CONTEXTS)
)


@dataclass(frozen=True)
class Home:
    """The CI job that enforces a hook that left the commit stage."""

    workflow: str
    job: str
    needles: tuple[str, ...]


LINT = ("ci.yml", "lint")
VALIDATORS = ("ci.yml", "onex-validation")

# Commit-stage hooks that take no filenames and always run. The ceiling is
# today's count; lower it when one is scoped or moved, never raise it.
UNSCOPED_COMMIT_STAGE_CEILING = 21

MOVED: Mapping[str, Home] = {
    "topic-naming-lint": Home(
        "ci.yml",
        "topic-naming-lint",
        (
            "lint_topic_names.py --scan-contracts src/omnibase_infra/nodes",
            "lint_topic_names.py --scan-python src/omnibase_infra",
        ),
    ),
    "no-omnimarket-dependency": Home(
        *LINT, ("tests/ci/test_no_omnimarket_dependency_omn19998.py",)
    ),
    "linear-transport-consolidation": Home(
        *LINT, ("tests/ci/test_linear_transport_consolidation.py",)
    ),
    "no-coding-plan-endpoint": Home(
        *LINT, ("python scripts/check_no_coding_plan_endpoint.py",)
    ),
    "no-literal-event-type-in-tests": Home(
        *LINT, ("omnibase_infra.validators.no_literal_event_type_in_tests tests",)
    ),
    "handler-dispatch-entrypoint": Home(
        *LINT,
        ("omnibase_infra.validators.handler_dispatch_entrypoint src/omnibase_infra",),
    ),
    "handler-event-type-source": Home(
        *LINT,
        ("omnibase_infra.validators.handler_event_type_source --scope omnibase_infra",),
    ),
    "contract-result-applier-construction": Home(
        *LINT,
        (
            "omnibase_infra.validators.contract_result_applier_construction "
            "src/omnibase_infra",
        ),
    ),
    "operation-match-fanout": Home(
        *LINT,
        ("omnibase_infra.validators.operation_match_fanout src/omnibase_infra",),
    ),
    "subscriber-dispatcher-resolution": Home(
        *LINT,
        (
            "omnibase_infra.validators.subscriber_dispatcher_resolution "
            "src/omnibase_infra",
        ),
    ),
    "contract-topic-category": Home(
        *LINT,
        ("omnibase_infra.validators.contract_topic_category --scope omnibase_infra",),
    ),
    "onex-validate-architecture": Home(*VALIDATORS, ("scripts/validate.py all",)),
    "onex-validate-clean-root": Home(*VALIDATORS, ("scripts/validate.py all",)),
    "onex-validate-contracts": Home(*VALIDATORS, ("scripts/validate.py all",)),
    "onex-validate-patterns": Home(*VALIDATORS, ("scripts/validate.py all",)),
    "onex-validate-unions": Home(*VALIDATORS, ("scripts/validate.py all",)),
    "onex-validate-any-types": Home(*VALIDATORS, ("scripts/validate.py all",)),
    "onex-validate-imports": Home(*VALIDATORS, ("scripts/validate.py all",)),
    "onex-validate-published-events": Home(
        *VALIDATORS, ("scripts/validation/check_published_events_consistency.py",)
    ),
    "onex-delegation-dispatch-consumer-kwarg-parity": Home(
        "delegation-consumer-kwarg-parity.yml",
        "consumer-kwarg-parity",
        (
            "tests/integration/runtime/test_delegation_dispatch_port_consumer_kwarg_parity.py",
        ),
    ),
    "subscribe-wiring-health": Home(
        "ci.yml",
        "subscribe-wiring-health",
        ("scripts/check_subscribe_wiring_health.py",),
    ),
    "governed-helper-primitive": Home(
        "governed-helper-primitive-gate.yml",
        "governed-helper-primitive-gate",
        ("scripts/ci/governed_helper_gate.py",),
    ),
    "onex-validate-runtime-profiles": Home(
        "ci.yml",
        "runtime-profiles-validator",
        ('ValidatorRuntimeProfiles().validate(Path("src"))',),
    ),
    "no-noncanonical-lifecycle-classes": Home(
        "ci.yml",
        "no-noncanonical-lifecycle-classes",
        (
            "omnibase_core.validators.no_noncanonical_lifecycle_classes",
            "src/omnibase_infra",
        ),
    ),
}

# The validate.py modes behind the ``scripts/validate.py all`` home. ``all``
# must run each of them, so CI enforces what the hook's mode enforced.
VALIDATE_MODES_BEHIND_ALL: Mapping[str, str] = {
    "onex-validate-architecture": "run_architecture",
    "onex-validate-clean-root": "run_clean_root",
    "onex-validate-contracts": "run_contracts",
    "onex-validate-patterns": "run_patterns",
    "onex-validate-unions": "run_unions",
    "onex-validate-any-types": "run_any_types",
    "onex-validate-imports": "run_imports",
}


def _hook_entries() -> list[dict[str, Any]]:
    """Every hook entry in the config; an id may be declared more than once."""
    config = yaml.safe_load(PRECOMMIT.read_text(encoding="utf-8"))
    return [hook for repo in config["repos"] for hook in repo.get("hooks", [])]


def _job(workflow: str, job_id: str) -> dict[str, Any]:
    document = yaml.safe_load((WORKFLOWS / workflow).read_text(encoding="utf-8"))
    jobs: dict[str, Any] = document["jobs"]
    assert job_id in jobs, f"{workflow} has no job {job_id}"
    job: dict[str, Any] = jobs[job_id]
    return job


def _run_text(job: Mapping[str, Any]) -> str:
    runs = [str(step["run"]) for step in job["steps"] if step.get("run")]
    flat = re.sub(r"\\\s*\n\s*", " ", "\n".join(runs))
    return " ".join(flat.split())


def test_every_manual_hook_is_registered_with_a_ci_home() -> None:
    manual = {
        hook["id"] for hook in _hook_entries() if "manual" in (hook.get("stages") or [])
    }
    assert manual == set(MOVED), (
        "a hook left the commit stage without a registered CI home, "
        f"or a registered hook is back at commit: {sorted(manual ^ (set(MOVED)))}"
    )


@pytest.mark.parametrize("hook_id", sorted(MOVED))
def test_moved_hook_is_manual_only(hook_id: str) -> None:
    entries = [hook for hook in _hook_entries() if hook["id"] == hook_id]
    assert entries, f"{hook_id} is not declared"
    assert all(hook.get("stages") == ["manual"] for hook in entries), (
        f"every declaration of {hook_id} must be manual-only"
    )


@pytest.mark.parametrize("hook_id", sorted(MOVED))
def test_moved_hook_has_a_gated_ci_job_that_runs_its_command(hook_id: str) -> None:
    home = MOVED[hook_id]
    job = _job(home.workflow, home.job)
    name = job.get("name", home.job)
    assert name in GATED_CONTEXTS, (
        f"{home.workflow}:{home.job} ({name!r}) is not a CI Summary gate, so "
        f"moving {hook_id} out of the commit stage would leave it unenforced"
    )
    assert job.get("if") in (None, "always()"), (
        f"{home.workflow}:{home.job} can be skipped by {job.get('if')!r}"
    )
    assert job.get("continue-on-error") is not True
    text = _run_text(job)
    for needle in home.needles:
        assert needle in text, (
            f"{home.workflow}:{home.job} no longer runs {needle!r} for {hook_id}"
        )


@pytest.mark.parametrize("hook_id", sorted(VALIDATE_MODES_BEHIND_ALL))
def test_validate_all_runs_the_mode_a_moved_hook_ran(hook_id: str) -> None:
    source = VALIDATE_PY.read_text(encoding="utf-8")
    body = source.split("def run_all(", 1)[1].split("\ndef ", 1)[0]
    assert VALIDATE_MODES_BEHIND_ALL[hook_id] in body
    assert "--quick" not in _run_text(_job(*VALIDATORS)).split()


def test_commit_stage_keeps_a_ceiling_on_unscoped_whole_repo_hooks() -> None:
    """A new whole-repo hook that fires on every commit needs a CI home first."""
    unscoped = sorted(
        hook["id"]
        for hook in _hook_entries()
        if (hook.get("stages") or ["pre-commit"]) == ["pre-commit"]
        and hook.get("pass_filenames") is False
        and hook.get("always_run")
    )
    assert len(unscoped) <= UNSCOPED_COMMIT_STAGE_CEILING, unscoped
