# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Wave D: CI gate — runtime-change PRs must cite a ticket with deploy evidence.

Ticket: OMN-8912
Root cause: PRs touching runtime code (Dockerfiles, node handlers, compose)
could merge without any deploy DoD evidence, leaving the .201 runtime stale.
Today's incident: Dockerfile.runtime changed but no deploy step in any ticket.

Gate rules:
  - If PR touches a runtime path → ticket contract must have a dod_evidence
    item whose check_value contains "docker exec", "rpk topic produce", or "deploy"
  - Override: [skip-deploy-gate: <reason>] in PR description → PASS + friction log
  - Non-runtime paths (docs, tests, scripts) → PASS unconditionally
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.validation.validate_pr_deploy_required import (
    find_runtime_paths,
    has_deploy_evidence,
    validate_pr_deploy_gate,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RUNTIME_PATHS = [
    "docker/Dockerfile.runtime",
    "docker/docker-compose.generated.yml",
    "src/omnibase_infra/nodes/handlers/db/handler_db.py",
    "src/omnibase_infra/nodes/handlers/db/contract.yaml",
    "src/omnibase_infra/runtime/service_kernel.py",
    "scripts/monitor_logs.py",
]

NON_RUNTIME_PATHS = [
    "docs/plans/2026-04-15-some-plan.md",
    "tests/ci/test_something.py",
    "README.md",
    ".github/workflows/contract-validation.yml",
]


def _make_contract(tmp_path: Path, dod_items: list[dict]) -> Path:
    """Write a minimal ticket contract YAML with given dod_evidence items."""
    contract = {
        "schema_version": "1.0.0",
        "ticket_id": "OMN-9999",
        "summary": "Test ticket",
        "is_seam_ticket": False,
        "interface_change": False,
        "interfaces_touched": [],
        "evidence_requirements": [],
        "emergency_bypass": {
            "enabled": False,
            "justification": "",
            "follow_up_ticket_id": "",
        },
        "dod_evidence": dod_items,
    }
    p = tmp_path / "OMN-9999.yaml"
    p.write_text(yaml.dump(contract), encoding="utf-8")
    return p


def _deploy_dod_item(check_value: str) -> dict:
    return {
        "id": "dod-deploy-001",
        "description": "Deploy to .201 runtime",
        "source": "manual",
        "checks": [{"check_type": "command", "check_value": check_value}],
        "status": "pending",
    }


def _non_deploy_dod_item() -> dict:
    return {
        "id": "dod-001",
        "description": "Tests pass",
        "source": "generated",
        "checks": [{"check_type": "test_passes", "check_value": "tests/ci/"}],
        "status": "pending",
    }


# ---------------------------------------------------------------------------
# Unit: find_runtime_paths
# ---------------------------------------------------------------------------


class TestFindRuntimePaths:
    def test_dockerfile_runtime_is_runtime_path(self) -> None:
        changed = ["docker/Dockerfile.runtime", "docs/README.md"]
        assert find_runtime_paths(changed) == ["docker/Dockerfile.runtime"]

    def test_compose_file_is_runtime_path(self) -> None:
        changed = ["docker/docker-compose.generated.yml"]
        assert find_runtime_paths(changed) == ["docker/docker-compose.generated.yml"]

    def test_node_handler_py_is_runtime_path(self) -> None:
        changed = ["src/omnibase_infra/nodes/handlers/db/handler_db.py"]
        assert find_runtime_paths(changed) != []

    def test_node_contract_yaml_is_runtime_path(self) -> None:
        changed = ["src/omnibase_infra/nodes/handlers/db/contract.yaml"]
        assert find_runtime_paths(changed) != []

    def test_runtime_kernel_is_runtime_path(self) -> None:
        changed = ["src/omnibase_infra/runtime/service_kernel.py"]
        assert find_runtime_paths(changed) != []

    def test_monitor_logs_is_runtime_path(self) -> None:
        changed = ["scripts/monitor_logs.py"]
        assert find_runtime_paths(changed) != []

    def test_omnimarket_node_is_runtime_path(self) -> None:
        changed = ["src/omnimarket/nodes/my_node/handlers/handler_foo.py"]
        assert find_runtime_paths(changed) != []

    def test_src_package_is_runtime_path(self) -> None:
        changed = ["src/omnibase_infra/some_module.py"]
        assert find_runtime_paths(changed) != []

    def test_docs_only_not_runtime(self) -> None:
        changed = ["docs/plans/foo.md", "README.md"]
        assert find_runtime_paths(changed) == []

    def test_test_files_not_runtime(self) -> None:
        changed = ["tests/ci/test_something.py", "tests/unit/test_other.py"]
        assert find_runtime_paths(changed) == []

    def test_github_workflow_not_runtime(self) -> None:
        changed = [".github/workflows/contract-validation.yml"]
        assert find_runtime_paths(changed) == []

    def test_empty_list_returns_empty(self) -> None:
        assert find_runtime_paths([]) == []


# ---------------------------------------------------------------------------
# Unit: has_deploy_evidence
# ---------------------------------------------------------------------------


class TestHasDeployEvidence:
    def test_docker_exec_command_is_deploy_evidence(self, tmp_path: Path) -> None:
        contract_path = _make_contract(
            tmp_path, [_deploy_dod_item("docker exec omninode-runtime echo ok")]
        )
        assert has_deploy_evidence(contract_path) is True

    def test_rpk_topic_produce_is_deploy_evidence(self, tmp_path: Path) -> None:
        contract_path = _make_contract(
            tmp_path,
            [
                _deploy_dod_item(
                    "rpk topic produce onex.cmd.deploy.rebuild-requested.v1"
                )
            ],
        )
        assert has_deploy_evidence(contract_path) is True

    def test_deploy_keyword_is_deploy_evidence(self, tmp_path: Path) -> None:
        contract_path = _make_contract(
            tmp_path, [_deploy_dod_item("onex deploy trigger rebuild")]
        )
        assert has_deploy_evidence(contract_path) is True

    def test_non_deploy_check_is_not_evidence(self, tmp_path: Path) -> None:
        contract_path = _make_contract(tmp_path, [_non_deploy_dod_item()])
        assert has_deploy_evidence(contract_path) is False

    def test_empty_dod_evidence_is_not_evidence(self, tmp_path: Path) -> None:
        contract_path = _make_contract(tmp_path, [])
        assert has_deploy_evidence(contract_path) is False

    def test_missing_contract_returns_false(self, tmp_path: Path) -> None:
        assert has_deploy_evidence(tmp_path / "nonexistent.yaml") is False


# ---------------------------------------------------------------------------
# Integration: validate_pr_deploy_gate — the 4 canonical cases
# ---------------------------------------------------------------------------


class TestValidatePrDeployGate:
    def test_runtime_change_with_deploy_evidence_passes(self, tmp_path: Path) -> None:
        """DoD case 1: Dockerfile.runtime + ticket with deploy_step → PASS."""
        _make_contract(
            tmp_path,
            [_deploy_dod_item("docker exec omninode-runtime echo deployed")],
        )
        result = validate_pr_deploy_gate(
            changed_files=["docker/Dockerfile.runtime"],
            pr_body="Fixes OMN-9999 deployment issue.",
            contracts_dir=tmp_path,
        )
        assert result.passed, f"Expected PASS but got: {result.message}"
        assert result.skipped is False

    def test_runtime_change_without_deploy_evidence_fails(self, tmp_path: Path) -> None:
        """DoD case 2: Dockerfile.runtime + ticket without deploy_step → FAIL with incident reference."""
        _make_contract(tmp_path, [_non_deploy_dod_item()])
        result = validate_pr_deploy_gate(
            changed_files=["docker/Dockerfile.runtime"],
            pr_body="Refs OMN-9999: update runtime Dockerfile",
            contracts_dir=tmp_path,
        )
        assert not result.passed, "Expected FAIL but got PASS"
        assert "deploy" in result.message.lower() or "OMN-8912" in result.message

    def test_docs_only_pr_passes_unconditionally(self, tmp_path: Path) -> None:
        """DoD case 3: docs-only → PASS (no runtime path touched)."""
        result = validate_pr_deploy_gate(
            changed_files=["docs/plans/2026-04-15-my-plan.md", "README.md"],
            pr_body="Add documentation for new feature.",
            contracts_dir=tmp_path,
        )
        assert result.passed
        assert result.skipped is True

    def test_override_token_passes_and_marks_friction(self, tmp_path: Path) -> None:
        """DoD case 4: [skip-deploy-gate: <reason>] → PASS but friction=True."""
        result = validate_pr_deploy_gate(
            changed_files=["docker/Dockerfile.runtime"],
            pr_body="[skip-deploy-gate: hotfix, deploy tracked in OMN-9998]\nFixes crash.",
            contracts_dir=tmp_path,
        )
        assert result.passed
        assert result.friction_logged is True

    def test_runtime_change_no_ticket_cited_fails(self, tmp_path: Path) -> None:
        """No OMN-XXXX in PR body and runtime path touched → FAIL."""
        result = validate_pr_deploy_gate(
            changed_files=["src/omnibase_infra/runtime/service_kernel.py"],
            pr_body="Minor cleanup, no ticket.",
            contracts_dir=tmp_path,
        )
        assert not result.passed

    def test_runtime_change_ticket_contract_missing_fails(self, tmp_path: Path) -> None:
        """OMN-XXXX cited but no contract file exists → FAIL (can't verify deploy step)."""
        result = validate_pr_deploy_gate(
            changed_files=["src/omnibase_infra/runtime/service_kernel.py"],
            pr_body="Refs OMN-9999: update kernel.",
            contracts_dir=tmp_path,  # empty tmp_path — no contract file
        )
        assert not result.passed

    def test_multiple_runtime_paths_any_ticket_with_deploy_passes(
        self, tmp_path: Path
    ) -> None:
        """Multiple runtime paths changed, ticket cites deploy → PASS."""
        _make_contract(
            tmp_path,
            [
                _non_deploy_dod_item(),
                _deploy_dod_item(
                    "rpk topic produce onex.cmd.deploy.rebuild-requested.v1"
                ),
            ],
        )
        result = validate_pr_deploy_gate(
            changed_files=[
                "docker/Dockerfile.runtime",
                "src/omnibase_infra/runtime/service_kernel.py",
            ],
            pr_body="OMN-9999: update runtime and Dockerfile",
            contracts_dir=tmp_path,
        )
        assert result.passed
