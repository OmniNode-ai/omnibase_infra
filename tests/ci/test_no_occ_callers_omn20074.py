# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20074 S6 part 2: omnibase_infra has no OCC caller left.

Pull-request admission here requires repo-owned evidence and nothing from
onex_change_control: no workflow calls an OCC reusable workflow, no job waits on
``occ-preflight``, and neither CI Summary nor the required-checks manifest
expects an OCC companion context. The assertions read the parsed workflow
files, so a comment that names a retired workflow cannot trip or satisfy them.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from scripts.ci import ci_summary_gate as gate

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
REQUIRED_CHECKS = REPO_ROOT / ".github" / "required-checks.yaml"

# A reusable whose file name starts with ``occ-`` or ``call-occ-`` is an OCC
# preflight, autobind or companion-effect path.
_OCC_REUSABLE = re.compile(r"^(?:call-)?occ-[a-z0-9-]+\.ya?ml$")
_PREFLIGHT_WAIT = re.compile(r"occ-preflight")

# The pinned skip-token reusable is the squash commit of omniclaude#2540, which
# removed the nested change-control preflight job from it.
_SKIP_TOKEN_REUSABLE = "reject-deploy-gate-skip.yml"
_PREFLIGHT_FREE_SKIP_TOKEN_SHA = "4358450ccbba0cee11e390208dd0b8b1728e94ab"

# The deploy-gate reusable reads the caller's own contracts/OMN-<n>.yaml at the
# pull request head when the caller passes contract-source: caller. The pin is
# the omniclaude commit that taught the reusable that input (omniclaude#2650);
# the pin before it read onex_change_control only.
_DEPLOY_GATE_REUSABLE = (
    "OmniNode-ai/omniclaude/.github/workflows/deploy-gate-reusable.yml"
)
_CALLER_CONTRACT_SOURCE_SHA = "0790179fc52759bd7354984fae45bb0e0486ef8a"
_OCC_ONLY_DEPLOY_GATE_SHA = "0c0d91e5e10904db67d43ad537fdf6e65e219f21"

_OCC_CONTEXTS = frozenset(
    {
        "occ-preflight / eligibility",
        "call-reject-skip-token / occ-preflight / eligibility",
        "verify / verify",
        "OCC Companion Merged Gate (OMN-15214)",
        "occ-companion-effect / Publish occ-companion-effect command",
        "occ-autobind / outcome",
    }
)


def _workflows() -> dict[str, dict[str, Any]]:
    parsed: dict[str, dict[str, Any]] = {}
    for path in sorted(WORKFLOWS.glob("*.y*ml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict), path.name
        parsed[path.name] = data
    return parsed


def _jobs() -> list[tuple[str, str, dict[str, Any]]]:
    rows: list[tuple[str, str, dict[str, Any]]] = []
    for name, data in _workflows().items():
        for job_id, job in (data.get("jobs") or {}).items():
            assert isinstance(job, dict), f"{name}:{job_id}"
            rows.append((name, job_id, job))
    return rows


def _needs(job: dict[str, Any]) -> list[str]:
    needs = job.get("needs") or []
    return [needs] if isinstance(needs, str) else list(needs)


def test_no_occ_callers_workflow_calls_an_occ_reusable() -> None:
    offenders = []
    for name, job_id, job in _jobs():
        uses = job.get("uses")
        if not isinstance(uses, str):
            continue
        reusable = uses.split("@", 1)[0].rsplit("/", 1)[-1]
        if _OCC_REUSABLE.match(reusable):
            offenders.append(f"{name}:{job_id} -> {reusable}")
    assert not offenders, offenders


def test_no_occ_callers_receipt_gate_runs_only_in_caller_mode() -> None:
    offenders = []
    for name, job_id, job in _jobs():
        uses = job.get("uses")
        if not isinstance(uses, str) or not uses.split("@", 1)[0].endswith(
            "/receipt-gate.yml"
        ):
            continue
        if (job.get("with") or {}).get("evidence-source") != "caller":
            offenders.append(f"{name}:{job_id}")
    assert not offenders, offenders


def test_no_occ_callers_skip_token_reusable_has_no_nested_preflight() -> None:
    pins = []
    for _name, _job_id, job in _jobs():
        uses = job.get("uses")
        if isinstance(uses, str) and uses.split("@", 1)[0].endswith(
            f"/{_SKIP_TOKEN_REUSABLE}"
        ):
            pins.append(uses.split("@", 1)[1].split()[0])
    assert pins == [_PREFLIGHT_FREE_SKIP_TOKEN_SHA]


def test_no_occ_callers_job_waits_on_occ_preflight() -> None:
    offenders = []
    for name, job_id, job in _jobs():
        if _PREFLIGHT_WAIT.search(job_id):
            offenders.append(f"{name}:{job_id} is a preflight job")
        if "occ-preflight" in _needs(job):
            offenders.append(f"{name}:{job_id} needs occ-preflight")
        if _PREFLIGHT_WAIT.search(str(job.get("if", ""))):
            offenders.append(f"{name}:{job_id} conditions on occ-preflight")
    assert not offenders, offenders


def test_no_occ_callers_workflow_runs_the_companion_gate_or_heals() -> None:
    workflows = _workflows()
    offenders = [
        name
        for name in (
            "call-occ-autobind.yml",
            "call-occ-companion-effect.yml",
            "occ-companion-merge-heal.yml",
            "occ-preflight-heal.yml",
            "call-receipt-gate.yml",
        )
        if name in workflows
    ]
    assert not offenders, offenders
    ci_jobs = workflows["ci.yml"]["jobs"]
    assert "occ-companion-merged" not in ci_jobs
    assert "occ-born-path-trigger-coverage" not in ci_jobs
    assert not [
        job_id
        for job_id, job in ci_jobs.items()
        if "OCC" in str(job.get("name", "")) and "Companion" in str(job.get("name", ""))
    ]


def test_no_occ_callers_ci_resolves_no_evidence_source() -> None:
    """No ci.yml job resolves a PR's companion or checks out OCC evidence data."""
    ci_jobs = _workflows()["ci.yml"]["jobs"]
    assert "contract-compliance" not in ci_jobs
    assert "Contract Compliance Check" not in gate.GATE_JOBS
    offenders = []
    for job_id, job in ci_jobs.items():
        for step in job.get("steps") or []:
            repository = (step.get("with") or {}).get("repository")
            run = str(step.get("run", ""))
            if repository == "OmniNode-ai/onex_change_control" and "steps." in str(
                (step.get("with") or {}).get("ref", "")
            ):
                offenders.append(f"{job_id}: {step.get('name')}")
            if "resolve_contract_compliance_evidence" in run:
                offenders.append(f"{job_id}: {step.get('name')}")
    assert not offenders, offenders


def test_no_occ_callers_ci_summary_expects_no_occ_context() -> None:
    expected = set(gate.EXPECTED_EXTERNAL_CONTEXTS)
    strict = set(gate.STRICT_GATE_JOBS) | set(gate.GATE_JOBS)
    assert not _OCC_CONTEXTS & expected
    assert not _OCC_CONTEXTS & strict
    assert not _OCC_CONTEXTS & set(gate.MEASURED_NOT_ENFORCED_CONTEXTS)
    assert not _OCC_CONTEXTS & set(gate.EXTERNAL_SWEEP_EXCLUSIONS)


def test_no_occ_callers_required_checks_manifest_names_no_occ_context() -> None:
    manifest = yaml.safe_load(REQUIRED_CHECKS.read_text(encoding="utf-8"))
    names = {row["name"] for row in manifest["gates"] if isinstance(row, dict)}
    assert not _OCC_CONTEXTS & names


def test_no_occ_callers_deploy_gate_reads_caller_contracts() -> None:
    """The deploy gate reads this repository's contracts, not change control."""
    calls = [
        (name, job_id, job)
        for name, job_id, job in _jobs()
        if isinstance(job.get("uses"), str)
        and job["uses"].split("@", 1)[0] == _DEPLOY_GATE_REUSABLE
    ]
    assert [(name, job_id) for name, job_id, _ in calls] == [
        ("deploy-gate.yml", "deploy-gate")
    ]
    job = calls[0][2]
    pin = job["uses"].split("@", 1)[1].split()[0]
    assert pin != _OCC_ONLY_DEPLOY_GATE_SHA
    assert pin == _CALLER_CONTRACT_SOURCE_SHA
    options = job.get("with") or {}
    assert options.get("contract-source") == "caller", options
    assert "contracts-dir" not in options, "deprecated input; the source decides"


@pytest.mark.live_contact("tests/ci/fixtures/deploy_gate_caller_source_omn20074.json")
def test_no_occ_callers_deploy_gate_reads_caller_contracts_recorded_verdicts(
    recorded_response: dict[str, object],
) -> None:
    """The pinned validator, run on real PRs, still refuses what lacks evidence."""
    assert recorded_response["omniclaude_sha"] == _CALLER_CONTRACT_SOURCE_SHA
    runs = cast("dict[str, dict[str, Any]]", recorded_response["response"])
    assert runs["admit"]["exit_code"] == 0
    assert "DEPLOY GATE PASSED" in runs["admit"]["output_first_line"]
    for name in (
        "refuse_no_contracts",
        "refuse_no_repo_contract",
        "refuse_no_falsifiable_probe",
    ):
        assert runs[name]["exit_code"] == 1, name
        assert "DEPLOY GATE FAILED" in runs[name]["output_first_line"], name
    assert (
        "no contract file in this repository's contracts/"
        in runs["refuse_no_contracts"]["output_first_line"]
    )
    assert (
        "declaring no falsifiable deploy probe"
        in runs["refuse_no_falsifiable_probe"]["output_first_line"]
    )
