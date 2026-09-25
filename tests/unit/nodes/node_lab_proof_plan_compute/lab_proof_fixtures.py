# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Shared builders for the lab proof node tests: a real registry row, a plan, a report.

Ticket: OMN-19572
"""

from __future__ import annotations

import json
from pathlib import Path

from omnibase_infra.lab_proof.enum_lab_proof_step_id import EnumLabProofStepId
from omnibase_infra.lab_proof.enum_lab_proof_step_phase import EnumLabProofStepPhase
from omnibase_infra.lab_proof.lab_proof_profile_registry import (
    load_lab_proof_profile_registry,
)
from omnibase_infra.lab_proof.model_lab_proof_bundle_policy import (
    ModelLabProofBundlePolicy,
)
from omnibase_infra.lab_proof.model_lab_proof_observation import (
    ModelLabProofObservation,
)
from omnibase_infra.lab_proof.model_lab_proof_plan import ModelLabProofPlan
from omnibase_infra.lab_proof.model_lab_proof_plan_request import (
    ModelLabProofPlanRequest,
)
from omnibase_infra.lab_proof.model_lab_proof_run_report import (
    ModelLabProofRunReport,
)
from omnibase_infra.lab_proof.model_lab_proof_subject import ModelLabProofSubject
from omnibase_infra.nodes.node_lab_proof_plan_compute.handlers.handler_lab_proof_plan import (
    HandlerLabProofPlan,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
REGISTRY = REPO_ROOT / "config" / "lab_proof_profiles.yaml"
CONTRACT = (
    REPO_ROOT / "src/omnibase_infra/nodes/node_lab_proof_plan_compute/contract.yaml"
)
HEAD = "bc32773871ae066939d69116271361710cb2e6d0"
BASE = "0123456789abcdef0123456789abcdef01234567"
INFRA = "c7150a15a94d5c4cc929de5f07bab2e7b622b497"
LANE = "/home/prover/prove-lab-proof-build-1"
RUN = "core-1766-r1"
CHANGED = (
    "src/omnibase_core/topics.py",
    "tests/unit/test_topic_base_dod_verify_duplicate_retired_omn19153.py",
)
TREE = json.dumps({"files": 1200, "sha256": "ab" * 32})


def bundle() -> ModelLabProofBundlePolicy:
    import yaml

    loaded = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    return ModelLabProofBundlePolicy.model_validate(loaded["config"]["local_bundle"])


def core_request(
    *,
    negative_control: bool = False,
    base_control: bool = False,
    changed: tuple[str, ...] = CHANGED,
    **overrides: object,
) -> ModelLabProofPlanRequest:
    registry = load_lab_proof_profile_registry(REGISTRY)
    profile = registry.profile_for("OmniNode-ai/omnibase_core")
    proved = BASE if base_control else HEAD
    subject = ModelLabProofSubject(
        repo="OmniNode-ai/omnibase_core",
        pr_number=1766,
        head_sha=HEAD,
        base_sha=BASE,
        proved_sha=proved,
        fetch_ref=BASE if base_control else "refs/pull/1766/head",
        changed_files=changed,
    )
    data: dict[str, object] = {
        "profile_key": profile.profile_key,
        "profile_version": profile.profile_version,
        "profile_repo": profile.repo,
        "variant": profile.variants[0],
        "subject": subject,
        "run_key": RUN + ("-base" if base_control else ""),
        "host": "192.168.86.202",
        "lane_root": LANE,
        "harness_root": f"{LANE}/harness",
        "infra_sha": INFRA,
        "model_endpoint_url": "http://192.168.86.202:8000/v1/chat/completions",
        "positive_control_project": "omnibase-infra-dev-202",
        "negative_control": negative_control,
        "bundle": bundle(),
    }
    data.update(overrides)
    return ModelLabProofPlanRequest.model_validate(data)


def core_plan(
    *,
    negative_control: bool = False,
    base_control: bool = False,
    changed: tuple[str, ...] = CHANGED,
) -> ModelLabProofPlan:
    return HandlerLabProofPlan().handle(
        core_request(
            negative_control=negative_control,
            base_control=base_control,
            changed=changed,
        )
    )


def _stdout_for(plan: ModelLabProofPlan, step_id: EnumLabProofStepId) -> str:
    step = next(s for s in plan.steps if s.step_id is step_id)
    if step.expect_stdout_equals is not None:
        return step.expect_stdout_equals + "\n"
    if step.expect_stdout_nonempty:
        return "c0ffee\n"
    if step_id in (
        EnumLabProofStepId.SUBJECT_HASH,
        EnumLabProofStepId.IDENTITY_RUNTIME_MAIN,
        EnumLabProofStepId.IDENTITY_RUNTIME_EFFECTS,
    ):
        return TREE + "\n"
    return ""


def report_for(
    plan: ModelLabProofPlan,
    *,
    failing: frozenset[EnumLabProofStepId] = frozenset(),
    stdout: dict[EnumLabProofStepId, str] | None = None,
    patterns: dict[EnumLabProofStepId, dict[str, int]] | None = None,
    extracted: dict[EnumLabProofStepId, tuple[str, ...]] | None = None,
) -> ModelLabProofRunReport:
    """A report as the run effect would write it, every step ok unless listed."""
    stdout = stdout or {}
    patterns = patterns or {}
    extracted = extracted or {}
    observations: list[ModelLabProofObservation] = []
    aborted: EnumLabProofStepId | None = None
    always = {EnumLabProofStepPhase.TEARDOWN, EnumLabProofStepPhase.RESIDUE}
    for step in plan.steps:
        if aborted is not None and step.phase not in always:
            observations.append(
                ModelLabProofObservation(
                    step_id=step.step_id,
                    phase=step.phase,
                    attribution=step.attribution,
                    ran=False,
                    skip_reason=f"not run: {aborted} failed",
                )
            )
            continue
        ok = step.step_id not in failing
        observations.append(
            ModelLabProofObservation(
                step_id=step.step_id,
                phase=step.phase,
                attribution=step.attribution,
                ran=True,
                exit_code=0 if ok else 1,
                attempts=1,
                stdout_tail=stdout.get(step.step_id, _stdout_for(plan, step.step_id)),
                expectation_met=ok,
                ok=ok,
                pattern_counts=patterns.get(
                    step.step_id, dict.fromkeys(step.grep_patterns, 0)
                ),
                extracted=extracted.get(step.step_id, ()),
            )
        )
        if (
            not ok
            and step.must_succeed
            and step.phase not in always
            and aborted is None
        ):
            aborted = step.step_id
    return ModelLabProofRunReport(
        run_key=plan.run_key,
        host=plan.host,
        started_at="2026-09-25T14:00:00Z",
        finished_at="2026-09-25T14:20:00Z",
        aborted_at=aborted,
        observations=tuple(observations),
    )
