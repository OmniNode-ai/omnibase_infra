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
from typing import Any

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
