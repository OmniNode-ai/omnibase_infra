# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20074: omnibase_infra's remaining reads of onex_change_control.

Three paths read the change-control repository and no longer need to:

* ``branch-protection-audit.yml`` fetched ``audit_branch_protection.sh`` from it at
  a pinned sha. The shared audit now runs from the workflow of the repository that owns it
  ``branch-protection-audit.yml`` over an overlay that names this repository, so
  the fetch and the canonical-audit step here are gone and the orphaned-context
  audit this repository owns is the workflow's one hard check.
* ``check_deploy_scope_dod.py`` read each cited ticket's contract from an
  ``$OMNI_HOME/onex_change_control/contracts`` clone. The hosted deploy gate reads
  ``contracts/<ticket>.yaml`` at the pull request head
  (``contract-source: caller``), so the hook reads the same directory of this
  repository.
* ``run_contract_sweep.py`` and ``verify_error_triage_e2e.py`` ran scripts out of a
  change-control clone and had no caller, so they are deleted.

Falsifier: restore any one read and the matching test names it.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import yaml

from scripts.ci import check_deploy_scope_dod as hook

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]

# A read of the change-control repository, in either spelling.
OCC_READ = re.compile(r"onex[_-]change[_-]control|\bOCC\b", re.IGNORECASE)


def _code_lines(relative: str) -> list[str]:
    """The file's lines that are not YAML comment lines."""
    text = (ROOT / relative).read_text(encoding="utf-8")
    return [line for line in text.splitlines() if not line.lstrip().startswith("#")]


def test_branch_protection_audit_reads_no_onex_change_control() -> None:
    offenders = [
        line
        for line in _code_lines(".github/workflows/branch-protection-audit.yml")
        if OCC_READ.search(line)
    ]
    assert not offenders, f"still reads onex_change_control: {offenders}"


def test_branch_protection_audit_keeps_its_orphan_and_parity_checks() -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/branch-protection-audit.yml").read_text()
    )
    runs = "\n".join(
        str(step.get("run", "")) for step in workflow["jobs"]["audit"]["steps"]
    )
    assert "scripts/audit_orphan_required_contexts.py" in runs
    assert "scripts/audit_required_context_parity_cli.py" in runs
    assert "ONEX_CHANGE_CONTROL_REF" not in (workflow.get("env") or {})


def test_deploy_scope_hook_code_reads_no_onex_change_control() -> None:
    source = (ROOT / "scripts/ci/check_deploy_scope_dod.py").read_text(encoding="utf-8")
    assert not hasattr(hook, "_OCC_CONTRACTS_RELPATH")
    assert not OCC_READ.search(source)


class _RecordedValidator(ModuleType):
    """Replay the canonical validator's recorded verdicts on the two contracts.

    The hosted runner has no omniclaude clone, so the real validator cannot be
    imported there and a test that needs it never executes (OMN-18648,
    OMN-18776). The recording's provenance holds the canonical
    ``has_deploy_evidence`` verdict on each contract; this stand-in returns that
    verdict for the exact bytes the hook hands it and refuses any other file, so
    the test still fails if the hook reads a contract from anywhere else.
    """

    TICKET_PATTERN = re.compile(r"OMN-(\d+)")

    def __init__(self, recording: dict[str, Any]) -> None:
        super().__init__("_recorded_deploy_gate_validator")
        self._verdicts = {
            recording["with_deploy_evidence"]["contract"]: True,
            recording["without_deploy_evidence"]["contract"]: False,
        }

    def find_runtime_paths(self, files: list[str]) -> list[str]:
        return [f for f in files if f.startswith("src/omnibase_infra/runtime/")]

    def parse_evidence_metadata(self, body: str) -> SimpleNamespace:
        return SimpleNamespace(source="", ticket="")

    def has_deploy_evidence(self, path: Path) -> bool:
        return self._verdicts[path.read_text()]


@pytest.mark.live_contact(
    "tests/ci/fixtures/ci_live_contact/deploy_scope_hook_repo_contracts_omn20074.json"
)
def test_deploy_scope_hook_judges_recorded_repo_contracts(
    recorded_response: dict[str, Any], tmp_path: Path
) -> None:
    """Replay two of this repository's own committed contracts through the hook.

    The recording holds the bytes of ``contracts/OMN-9877.yaml`` (declares
    deploy-scope evidence) and ``contracts/OMN-9050.yaml`` (declares none). Laid
    out under ``contracts/`` they pass and fail respectively, and a ticket with
    no file there fails, which is the gap the hosted gate refuses.
    """
    recording = recorded_response["response"]
    texts = "".join(
        recording[key]["contract"]
        for key in ("with_deploy_evidence", "without_deploy_evidence")
    )
    assert (
        hashlib.sha256(texts.encode("utf-8")).hexdigest()
        == recorded_response["_provenance"]["source_sha256"]
    )
    validator = _RecordedValidator(recording)
    contracts = hook.contracts_dir_for(tmp_path)
    contracts.mkdir()
    for key in ("with_deploy_evidence", "without_deploy_evidence"):
        row = recording[key]
        (contracts / f"{row['ticket']}.yaml").write_text(row["contract"])
    runtime_file = ["src/omnibase_infra/runtime/service_kernel.py"]
    with_ticket = recording["with_deploy_evidence"]["ticket"]
    without_ticket = recording["without_deploy_evidence"]["ticket"]

    def classify(ticket: str) -> hook.DeployScopeDecision:
        return hook.classify_deploy_scope(validator, runtime_file, ticket, contracts)

    assert classify(with_ticket).outcome == hook.PASS_EVIDENCE
    assert classify(without_ticket).outcome == hook.FAIL_NO_EVIDENCE
    missing = classify("OMN-1")
    assert missing.outcome == hook.FAIL_NO_EVIDENCE
    assert "contracts/OMN-1.yaml" in missing.detail


def test_deploy_scope_hook_default_contracts_dir_is_this_repository() -> None:
    assert hook.contracts_dir_for(ROOT) == ROOT / "contracts"


@pytest.mark.parametrize(
    "relative",
    [
        "scripts/ci/run_contract_sweep.py",
        "scripts/verify_error_triage_e2e.py",
    ],
)
def test_script_that_ran_out_of_a_change_control_clone_is_gone(relative: str) -> None:
    assert not (ROOT / relative).exists()
    baseline = (ROOT / ".onex_ratchets/canonical_file_shape_baseline.txt").read_text()
    assert relative not in baseline.splitlines()
