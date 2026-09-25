#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Run one lab proof on this host: the I/O boundary around the lab proof nodes.

WHAT THIS IS, AND WHAT IT IS NOT. A prover runs this on a lab host, in a clone of
omnibase_infra, for one pull request. It reads the profile row from
config/lab_proof_profiles.yaml and the bundle facts from the plan node's
contract, and then asks the nodes, in order:

  node_lab_proof_plan_compute  lab_proof.plan     render the argv steps
  node_lab_proof_run_effect    lab_proof.run      run them on this host
  node_lab_proof_plan_compute  lab_proof.verdict  decide the outcome

When the head FAILs, the same steps are run once more at the merge base (a base
control) and the head's verdict is taken again with it, so a defect dev already
had is reported DEV_INHERITED instead of blamed on the PR (interim recipes,
common frame 8). None of the deciding is here: this file only wires inputs.

It reads nothing from GitHub. The PR head, merge base and changed files are
handed in by the prover, which already read them to select the PR; a head that
moved since is caught by the plan's own ``subject_rev`` step.

It does not take a HOLD or write the ledger: the prover lane does both around
this call, as the interim recipes say. The scheduler (OMN-19569) replaces this
file with a contract-declared orchestrator.

Exit status: 0 PASS (or, for a negative control, the expected FAIL), 1 FAIL,
2 INCONCLUSIVE, 3 DEV_INHERITED, 4 a negative control that did not fail,
5 refused before anything ran.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from omnibase_infra.lab_proof.enum_lab_proof_check import (
    EnumLabProofCheck,
)
from omnibase_infra.lab_proof.enum_lab_proof_execution import (
    EnumLabProofExecution,
)
from omnibase_infra.lab_proof.enum_lab_proof_outcome import (
    EnumLabProofOutcome,
)
from omnibase_infra.lab_proof.lab_proof_profile_registry import (
    load_lab_proof_profile_registry,
)
from omnibase_infra.lab_proof.model_lab_proof_bundle_policy import (
    ModelLabProofBundlePolicy,
)
from omnibase_infra.lab_proof.model_lab_proof_plan import (
    ModelLabProofPlan,
)
from omnibase_infra.lab_proof.model_lab_proof_plan_request import (
    ModelLabProofPlanRequest,
)
from omnibase_infra.lab_proof.model_lab_proof_result import (
    ModelLabProofResult,
)
from omnibase_infra.lab_proof.model_lab_proof_run_report import (
    ModelLabProofRunReport,
)
from omnibase_infra.lab_proof.model_lab_proof_subject import (
    ModelLabProofSubject,
)
from omnibase_infra.lab_proof.model_lab_proof_verdict_request import (
    ModelLabProofVerdictRequest,
)
from omnibase_infra.nodes.node_lab_proof_plan_compute.handlers.handler_lab_proof_plan import (
    HandlerLabProofPlan,
)
from omnibase_infra.nodes.node_lab_proof_plan_compute.handlers.handler_lab_proof_verdict import (
    HandlerLabProofVerdict,
)
from omnibase_infra.nodes.node_lab_proof_run_effect.handlers.handler_lab_proof_run import (
    HandlerLabProofRun,
)

REGISTRY = REPO_ROOT / "config" / "lab_proof_profiles.yaml"
PLAN_CONTRACT = (
    REPO_ROOT / "src/omnibase_infra/nodes/node_lab_proof_plan_compute/contract.yaml"
)
EXIT = {
    EnumLabProofOutcome.PASS: 0,
    EnumLabProofOutcome.FAIL: 1,
    EnumLabProofOutcome.INCONCLUSIVE: 2,
    EnumLabProofOutcome.DEV_INHERITED: 3,
}


def load_bundle_policy(contract: Path = PLAN_CONTRACT) -> ModelLabProofBundlePolicy:
    """Parse the plan node contract's ``config.local_bundle`` block."""
    loaded = yaml.safe_load(contract.read_text(encoding="utf-8"))
    return ModelLabProofBundlePolicy.model_validate(loaded["config"]["local_bundle"])


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one lab proof on this host.")
    parser.add_argument("--repo", required=True, help="OmniNode-ai/<name>")
    parser.add_argument("--pr", required=True, type=int)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--base-sha", required=True, help="the merge base")
    parser.add_argument(
        "--changed-files",
        type=Path,
        required=True,
        help="file: one changed path per line",
    )
    parser.add_argument("--host", required=True)
    parser.add_argument("--lane-root", required=True, help="absolute ~/prove-<lane>")
    parser.add_argument("--run-key", required=True)
    parser.add_argument("--infra-sha", required=True)
    parser.add_argument("--model-endpoint", required=True)
    parser.add_argument("--positive-control-project", required=True)
    parser.add_argument("--negative-control", action="store_true")
    parser.add_argument("--no-base-control", action="store_true")
    parser.add_argument(
        "--out", type=Path, required=True, help="directory for JSON results"
    )
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    return parser.parse_args(argv)


def _run(
    out: Path, request: ModelLabProofPlanRequest, label: str
) -> tuple[ModelLabProofPlan, ModelLabProofRunReport]:
    plan = HandlerLabProofPlan().handle(request)
    (out / f"{label}-plan.json").write_text(
        plan.model_dump_json(indent=2), encoding="utf-8"
    )
    report = HandlerLabProofRun().handle(plan)
    (out / f"{label}-report.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    return plan, report


def _judge(
    out: Path,
    plan: ModelLabProofPlan,
    report: ModelLabProofRunReport,
    mandatory: tuple[EnumLabProofCheck, ...],
    label: str,
    base_result: ModelLabProofResult | None = None,
) -> ModelLabProofResult:
    result = HandlerLabProofVerdict().handle(
        ModelLabProofVerdictRequest(
            plan=plan,
            report=report,
            mandatory_checks=mandatory,
            base_result=base_result,
        )
    )
    (out / f"{label}-result.json").write_text(
        result.model_dump_json(indent=2), encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    registry = load_lab_proof_profile_registry(args.registry)
    profile = registry.profile_for(args.repo)
    variants = [
        v for v in profile.variants if v.execution is EnumLabProofExecution.NODE
    ]
    if len(variants) != 1:
        print(
            f"refused: {args.repo} has {len(variants)} node-executed variants, need 1"
        )
        return 5
    variant = variants[0]
    changed = tuple(
        line.strip()
        for line in args.changed_files.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    common = {
        "profile_key": profile.profile_key,
        "profile_version": profile.profile_version,
        "profile_repo": profile.repo,
        "variant": variant,
        "host": args.host,
        "lane_root": args.lane_root,
        "harness_root": str(REPO_ROOT),
        "infra_sha": args.infra_sha,
        "model_endpoint_url": args.model_endpoint,
        "positive_control_project": args.positive_control_project,
        "bundle": load_bundle_policy(),
    }
    head_subject = ModelLabProofSubject(
        repo=args.repo,
        pr_number=args.pr,
        head_sha=args.head_sha,
        base_sha=args.base_sha,
        proved_sha=args.head_sha,
        fetch_ref=f"refs/pull/{args.pr}/head",
        changed_files=changed,
    )
    mandatory = tuple(variant.mandatory_checks)
    head_request = ModelLabProofPlanRequest.model_validate(
        {
            **common,
            "subject": head_subject,
            "run_key": args.run_key,
            "negative_control": args.negative_control,
        }
    )
    head_plan, head_report = _run(args.out, head_request, "head")
    result = _judge(args.out, head_plan, head_report, mandatory, "head")
    if (
        result.outcome is EnumLabProofOutcome.FAIL
        and not args.negative_control
        and not args.no_base_control
    ):
        base_request = ModelLabProofPlanRequest.model_validate(
            {
                **common,
                "subject": head_subject.model_copy(
                    update={"proved_sha": args.base_sha, "fetch_ref": args.base_sha}
                ),
                "run_key": f"{args.run_key}-base",
                "negative_control": False,
            }
        )
        base_plan, base_report = _run(args.out, base_request, "base")
        base_result = _judge(args.out, base_plan, base_report, mandatory, "base")
        result = _judge(
            args.out, head_plan, head_report, mandatory, "head-final", base_result
        )
    print(json.dumps(json.loads(result.model_dump_json()), indent=2))
    if args.negative_control:
        return 0 if result.outcome is EnumLabProofOutcome.FAIL else 4
    return EXIT[result.outcome]


if __name__ == "__main__":
    sys.exit(main())
