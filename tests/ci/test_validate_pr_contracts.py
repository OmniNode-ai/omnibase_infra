# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for validate_pr_contracts.py — Layer 1 contract gate.

Ticket: OMN-8909
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "validate_pr_contracts.py"

RUNTIME_FILE = "src/omnibase_infra/nodes/node_foo/handler.py"
DOCS_ONLY_FILE = "docs/README.md"


def _run(
    *,
    diff_files: str = "",
    branch: str = "jonah/omn-1234-test",
    pr_title: str = "fix: thing (OMN-1234)",
    pr_body: str = "",
    contracts_path: str = "",
    env_extra: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = {
        "PR_DIFF_FILES": diff_files,
        "PR_BRANCH": branch,
        "PR_TITLE": pr_title,
        "CONTRACTS_PATH": contracts_path,
        "PATH": "/usr/bin:/bin:/usr/local/bin",
    }
    if pr_body:
        env["PR_BODY"] = pr_body
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=False,
    )


def _write_contract(tmp_path: Path, ticket_id: str, *, with_dod: bool = True) -> Path:
    contracts_dir = tmp_path / "contracts"
    contracts_dir.mkdir(exist_ok=True)
    contract_file = contracts_dir / f"{ticket_id}.yaml"

    base = {
        "schema_version": "1.0.0",
        "ticket_id": ticket_id,
        "summary": "test contract",
        "is_seam_ticket": False,
        "interface_change": False,
        "interfaces_touched": [],
        "evidence_requirements": [],
        "emergency_bypass": {
            "enabled": False,
            "justification": "",
            "follow_up_ticket_id": "",
        },
    }

    if with_dod:
        base["dod_evidence"] = [
            {
                "id": "dod-001",
                "description": "CI passes",
                "source": "generated",
                "checks": [{"check_type": "command", "check_value": "echo ok"}],
            }
        ]
    else:
        base["dod_evidence"] = []

    contract_file.write_text(
        yaml.dump(base, default_flow_style=False), encoding="utf-8"
    )
    return contracts_dir


class TestMissingContractFails:
    """dod-001: PRs with OMN ticket touching runtime code but no contract file must fail."""

    def test_missing_contract_fails(self, tmp_path: Path) -> None:
        contracts_dir = tmp_path / "contracts"
        contracts_dir.mkdir()

        result = _run(
            diff_files=RUNTIME_FILE,
            branch="jonah/omn-1234-test",
            contracts_path=str(contracts_dir),
        )
        assert result.returncode != 0, (
            f"Expected non-zero exit, got stdout: {result.stdout}"
        )
        assert "OMN-1234" in result.stdout or "OMN-1234" in result.stderr

    def test_missing_contract_with_runtime_file(self, tmp_path: Path) -> None:
        contracts_dir = tmp_path / "contracts"
        contracts_dir.mkdir()

        result = _run(
            diff_files="src/omnimarket/nodes/node_bar/handler_main.py",
            branch="jonah/omn-5678-new-node",
            pr_title="feat: add node_bar (OMN-5678)",
            contracts_path=str(contracts_dir),
        )
        assert result.returncode != 0


class TestEmptyDodEvidenceFails:
    """dod-002: Contract with empty dod_evidence on runtime-touching PR must fail."""

    def test_empty_dod_evidence_fails(self, tmp_path: Path) -> None:
        contracts_dir = _write_contract(tmp_path, "OMN-1234", with_dod=False)

        result = _run(
            diff_files=RUNTIME_FILE,
            branch="jonah/omn-1234-test",
            contracts_path=str(contracts_dir),
        )
        assert result.returncode != 0, (
            f"Expected non-zero exit, got stdout: {result.stdout}"
        )
        assert (
            "dod_evidence" in result.stdout.lower()
            or "dod_evidence" in result.stderr.lower()
        )

    def test_populated_dod_evidence_passes(self, tmp_path: Path) -> None:
        contracts_dir = _write_contract(tmp_path, "OMN-1234", with_dod=True)

        result = _run(
            diff_files=RUNTIME_FILE,
            branch="jonah/omn-1234-test",
            contracts_path=str(contracts_dir),
        )
        assert result.returncode == 0, f"Expected exit 0, got stderr: {result.stderr}"


class TestNoTicketSkips:
    """dod-003: PRs with no OMN ticket reference must skip (exit 0)."""

    def test_no_ticket_skips(self, tmp_path: Path) -> None:
        result = _run(
            diff_files=RUNTIME_FILE,
            branch="dependabot/npm_and_yarn/next-15.0.0",
            pr_title="chore(deps): bump next from 14 to 15",
            contracts_path=str(tmp_path),
        )
        assert result.returncode == 0

    def test_release_branch_skips(self, tmp_path: Path) -> None:
        result = _run(
            diff_files=RUNTIME_FILE,
            branch="release/v2.0.0",
            pr_title="release: v2.0.0",
            contracts_path=str(tmp_path),
        )
        assert result.returncode == 0

    def test_docs_only_pr_skips(self, tmp_path: Path) -> None:
        contracts_dir = tmp_path / "contracts"
        contracts_dir.mkdir()

        result = _run(
            diff_files=DOCS_ONLY_FILE,
            branch="jonah/omn-1234-docs",
            contracts_path=str(contracts_dir),
        )
        assert result.returncode == 0


class TestTicketExtraction:
    """Ticket IDs are extracted from branch, title, and body."""

    def test_ticket_from_title_only(self, tmp_path: Path) -> None:
        contracts_dir = _write_contract(tmp_path, "OMN-9999", with_dod=True)

        result = _run(
            diff_files=RUNTIME_FILE,
            branch="jonah/fix-thing",
            pr_title="fix: resolve crash (OMN-9999)",
            contracts_path=str(contracts_dir),
        )
        assert result.returncode == 0

    def test_ticket_from_body(self, tmp_path: Path) -> None:
        contracts_dir = _write_contract(tmp_path, "OMN-4444", with_dod=True)

        result = _run(
            diff_files=RUNTIME_FILE,
            branch="jonah/fix-thing",
            pr_title="fix: resolve crash",
            pr_body="Closes OMN-4444",
            contracts_path=str(contracts_dir),
        )
        assert result.returncode == 0

    def test_multiple_tickets_all_need_contracts(self, tmp_path: Path) -> None:
        contracts_dir = _write_contract(tmp_path, "OMN-1111", with_dod=True)

        result = _run(
            diff_files=RUNTIME_FILE,
            branch="jonah/omn-1111-and-omn-2222",
            pr_title="feat: multi-ticket",
            contracts_path=str(contracts_dir),
        )
        assert result.returncode != 0, (
            "Should fail because OMN-2222 contract is missing"
        )
