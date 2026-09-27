# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Committed lab-proof registry through planning, verdicts, and its CI gate."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from omnibase_infra.lab_proof.enum_lab_proof_outcome import EnumLabProofOutcome
from omnibase_infra.lab_proof.enum_lab_proof_step_id import EnumLabProofStepId
from omnibase_infra.lab_proof.lab_proof_profile_registry import (
    load_lab_proof_profile_registry,
)
from omnibase_infra.lab_proof.model_lab_proof_plan_request import (
    ModelLabProofPlanRequest,
)
from omnibase_infra.lab_proof.model_lab_proof_profile import ModelLabProofProfile
from omnibase_infra.lab_proof.model_lab_proof_profile_registry import (
    ModelLabProofProfileRegistry,
)
from omnibase_infra.lab_proof.model_lab_proof_result import ModelLabProofResult
from omnibase_infra.lab_proof.model_lab_proof_subject import ModelLabProofSubject
from omnibase_infra.lab_proof.model_lab_proof_verdict_request import (
    ModelLabProofVerdictRequest,
)
from omnibase_infra.nodes.node_lab_proof_plan_compute.handlers.handler_lab_proof_plan import (
    HandlerLabProofPlan,
)
from omnibase_infra.nodes.node_lab_proof_plan_compute.handlers.handler_lab_proof_verdict import (
    HandlerLabProofVerdict,
)
from tests.unit.nodes.node_lab_proof_plan_compute.lab_proof_fixtures import (
    BASE,
    CHANGED,
    HEAD,
    INFRA,
    LANE,
    RUN,
    bundle,
    report_for,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
REGISTRY = REPO_ROOT / "config" / "lab_proof_profiles.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "ci" / "validate_lab_proof_profiles.py"
CORE_REPO = "OmniNode-ai/omnibase_core"


def _request(
    profile: ModelLabProofProfile, *, negative_control: bool
) -> ModelLabProofPlanRequest:
    subject = ModelLabProofSubject(
        repo=CORE_REPO,
        pr_number=1766,
        head_sha=HEAD,
        base_sha=BASE,
        proved_sha=HEAD,
        fetch_ref="refs/pull/1766/head",
        changed_files=CHANGED,
    )
    return ModelLabProofPlanRequest.model_validate(
        {
            "profile_key": profile.profile_key,
            "profile_version": profile.profile_version,
            "profile_repo": profile.repo,
            "variant": profile.variants[0],
            "subject": subject,
            "run_key": RUN + ("-negative" if negative_control else ""),
            "host": "192.168.86.202",
            "lane_root": LANE,
            "harness_root": f"{LANE}/harness",
            "infra_sha": INFRA,
            "model_endpoint_url": ("http://192.168.86.202:8000/v1/chat/completions"),
            "positive_control_project": "omnibase-infra-dev-202",
            "negative_control": negative_control,
            "bundle": bundle(),
        }
    )


def _run_validator(registry: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), "--registry", str(registry)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_committed_registry_drives_discriminating_plan_verdicts() -> None:
    loaded_registry = load_lab_proof_profile_registry(REGISTRY)
    registry = ModelLabProofProfileRegistry.model_validate(loaded_registry.model_dump())
    assert {profile.repo for profile in registry.profiles} == set(
        registry.registry_repos
    )
    assert len(registry.profiles) == len(registry.registry_repos)

    profile = registry.profile_for(CORE_REPO)
    assert profile.profile_key == "omnibase_core.foundation_override"
    mandatory_checks = profile.variants[0].mandatory_checks
    planner = HandlerLabProofPlan()
    judge = HandlerLabProofVerdict()

    head_plan = planner.handle(_request(profile, negative_control=False))
    head = judge.handle(
        ModelLabProofVerdictRequest(
            plan=head_plan,
            report=report_for(head_plan),
            mandatory_checks=mandatory_checks,
        )
    )

    negative_plan = planner.handle(_request(profile, negative_control=True))
    negative_failure = judge.handle(
        ModelLabProofVerdictRequest(
            plan=negative_plan,
            report=report_for(
                negative_plan,
                failing=frozenset(
                    {
                        EnumLabProofStepId.HEALTH_RUNTIME_MAIN,
                        EnumLabProofStepId.HEALTH_RUNTIME_EFFECTS,
                        EnumLabProofStepId.IMPORT_SMOKE_RUNTIME_MAIN,
                        EnumLabProofStepId.IMPORT_SMOKE_RUNTIME_EFFECTS,
                        EnumLabProofStepId.GOLDEN_CHAIN_DELEGATION,
                    }
                ),
            ),
            mandatory_checks=mandatory_checks,
        )
    )
    assert head.outcome is EnumLabProofOutcome.PASS
    assert not head.negative_control
    # The sabotaged run is judged by the same handler and must come back FAIL,
    # marked as a negative control: the proof is able to fail.
    assert negative_failure.outcome is EnumLabProofOutcome.FAIL
    assert negative_failure.negative_control

    negative_did_not_fail = judge.handle(
        ModelLabProofVerdictRequest(
            plan=negative_plan,
            report=report_for(negative_plan),
            mandatory_checks=mandatory_checks,
        )
    )
    # A negative control whose checks all pass is reported PASS, flagged as a
    # negative control, with the reason saying PASS means the checks cannot
    # fail; run_lab_proof.py turns exactly that into its exit 4.
    assert negative_did_not_fail.outcome is EnumLabProofOutcome.PASS
    assert negative_did_not_fail.negative_control
    assert any("cannot fail" in reason for reason in negative_did_not_fail.reasons)


def test_validator_accepts_registry_and_names_a_removed_repository(
    tmp_path: Path,
) -> None:
    committed = _run_validator(REGISTRY)
    assert committed.returncode == 0, committed.stdout + committed.stderr

    raw = cast("dict[str, Any]", yaml.safe_load(REGISTRY.read_text(encoding="utf-8")))
    profiles = cast("list[dict[str, Any]]", raw["profiles"])
    removed_repo = "OmniNode-ai/omnidash"
    raw["profiles"] = [
        profile for profile in profiles if profile["repo"] != removed_repo
    ]
    incomplete_registry = tmp_path / "lab_proof_profiles.yaml"
    incomplete_registry.write_text(
        yaml.safe_dump(raw, sort_keys=False), encoding="utf-8"
    )

    incomplete = _run_validator(incomplete_registry)
    assert incomplete.returncode != 0
    assert removed_repo in incomplete.stdout + incomplete.stderr
