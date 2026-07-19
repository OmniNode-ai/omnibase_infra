# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the pre-push deploy-scope DoD parity gate (OMN-14681).

These tests drive the pure ``classify_deploy_scope`` core against the REAL
canonical deploy-gate validator (imported the same way the hook imports it),
proving both the DRY import wiring and every tri-state outcome -- including the
omnibase_infra#2319 gap (deploy-scoped surface + cited ticket whose OCC contract
declares no deploy-scope DoD evidence).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ci.check_deploy_scope_dod import (
    FAIL_NO_EVIDENCE,
    FAIL_NO_TICKET,
    NOTICE_COMPANION_UNMERGED,
    PASS_EVIDENCE,
    SKIP_NO_RUNTIME,
    DeployScopeHookError,
    classify_deploy_scope,
    load_canonical_validator,
    resolve_omni_home,
)

# A runtime path that matches the canonical RUNTIME_PATH_PATTERNS
# (src/omnibase_infra/runtime/**/*.py) -- deploy-scoped by construction.
_RUNTIME_FILE = "src/omnibase_infra/runtime/service_kernel.py"
# A non-deploy-scoped path (docs) -- matches no runtime pattern.
_NON_RUNTIME_FILE = "docs/patterns/error_handling_patterns.md"


@pytest.fixture(scope="module")
def validator():
    """The REAL canonical validator, imported exactly as the hook imports it."""
    repo_root = Path(__file__).resolve().parents[2]
    try:
        omni_home = resolve_omni_home(repo_root)
    except DeployScopeHookError:
        pytest.skip("omniclaude sibling clone not present under OMNI_HOME")
    return load_canonical_validator(omni_home)


@pytest.fixture(autouse=True)
def _report_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    # Pin the evidence rule to CI's default rollout mode so the test asserts
    # against the same behaviour the hosted gate uses today.
    monkeypatch.setenv("DEPLOY_GATE_FALSIFIABILITY", "report")


def _write_contract(contracts_dir: Path, ticket: str, check_value: str) -> None:
    contracts_dir.mkdir(parents=True, exist_ok=True)
    (contracts_dir / f"{ticket}.yaml").write_text(
        "dod_evidence:\n"
        "  - id: dod-deploy\n"
        "    checks:\n"
        f"      - check_value: {check_value!r}\n",
        encoding="utf-8",
    )


def test_no_runtime_paths_skips(validator, tmp_path: Path) -> None:
    decision = classify_deploy_scope(
        validator=validator,
        changed_files=[_NON_RUNTIME_FILE],
        pr_body="Closes OMN-1234",
        contracts_dir=tmp_path,
    )
    assert decision.outcome == SKIP_NO_RUNTIME
    assert decision.exit_code == 0


def test_runtime_path_without_ticket_fails(validator, tmp_path: Path) -> None:
    decision = classify_deploy_scope(
        validator=validator,
        changed_files=[_RUNTIME_FILE],
        pr_body="a push that cites no ticket at all",
        contracts_dir=tmp_path,
    )
    assert decision.outcome == FAIL_NO_TICKET
    assert decision.exit_code == 1
    assert _RUNTIME_FILE in decision.runtime_hits


def test_runtime_path_ticket_with_deploy_evidence_passes(
    validator, tmp_path: Path
) -> None:
    _write_contract(
        tmp_path,
        "OMN-1234",
        "docker exec ${RUNTIME_CONTAINER:-omninode-runtime} python -c 'import x'",
    )
    decision = classify_deploy_scope(
        validator=validator,
        changed_files=[_RUNTIME_FILE],
        pr_body="Closes OMN-1234",
        contracts_dir=tmp_path,
    )
    assert decision.outcome == PASS_EVIDENCE
    assert decision.exit_code == 0
    assert "OMN-1234" in decision.tickets


def test_runtime_path_ticket_without_deploy_evidence_fails(
    validator, tmp_path: Path
) -> None:
    # The omnibase_infra#2319 gap: contract present, but no deploy-scope probe.
    # Value carries none of the legacy keywords (docker exec / rpk topic
    # produce / deploy) nor a live-surface probe, so it fails in both modes.
    _write_contract(tmp_path, "OMN-1234", "pytest tests/unit -k service_kernel")
    decision = classify_deploy_scope(
        validator=validator,
        changed_files=[_RUNTIME_FILE],
        pr_body="Closes OMN-1234",
        contracts_dir=tmp_path,
    )
    assert decision.outcome == FAIL_NO_EVIDENCE
    assert decision.exit_code == 1
    assert "OMN-1234" in decision.tickets


def test_runtime_path_ticket_without_local_contract_notices(
    validator, tmp_path: Path
) -> None:
    # Companion OCC contract not merged/authored locally -> NOTICE, never red.
    decision = classify_deploy_scope(
        validator=validator,
        changed_files=[_RUNTIME_FILE],
        pr_body="Closes OMN-9999",
        contracts_dir=tmp_path,  # empty -> OMN-9999.yaml absent
    )
    assert decision.outcome == NOTICE_COMPANION_UNMERGED
    assert decision.exit_code == 0
    assert "OMN-9999" in decision.tickets


def test_runtime_detection_is_dry_with_canonical_validator(validator) -> None:
    # Prove the hook's detection is the canonical validator's own function,
    # not a local re-implementation (OMN-14655 DRIFT-3 guard).
    assert validator.find_runtime_paths([_RUNTIME_FILE]) == [_RUNTIME_FILE]
    assert validator.find_runtime_paths([_NON_RUNTIME_FILE]) == []
