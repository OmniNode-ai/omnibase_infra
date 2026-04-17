# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for validate_pr_deploy_required.py deploy gate (OMN-8912).

Verifies end-to-end behavior of the Wave D deploy gate:
  - Runtime path pattern matching against 16 real path patterns
  - Ticket contract YAML parsing for deploy evidence
  - Override token handling with friction logging
  - Realistic PR workflows (no runtime change, override, missing evidence, valid evidence)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from scripts.validation.validate_pr_deploy_required import (
    find_runtime_paths,
    has_deploy_evidence,
    validate_pr_deploy_gate,
)


@pytest.mark.integration
def test_runtime_path_patterns_match_docker_files() -> None:
    """Dockerfiles and compose files must trigger the deploy gate."""
    changed_files = [
        "docker/Dockerfile.runtime",
        "docker/Dockerfile.core",
        "docker/docker-compose.infra.yml",
        "docker/docker-compose.generated.yaml",
        "docker/services/postgres.Dockerfile",
    ]
    hits = find_runtime_paths(changed_files)
    assert len(hits) == 5, "All Docker-related files should match runtime patterns"
    assert "docker/Dockerfile.runtime" in hits
    assert "docker/docker-compose.infra.yml" in hits


@pytest.mark.integration
def test_runtime_path_patterns_match_node_handlers() -> None:
    """Node handler changes must trigger the deploy gate."""
    changed_files = [
        "src/omnibase_infra/nodes/node_registration_orchestrator/handlers/handler_node_introspected.py",
        "src/omnibase_infra/nodes/node_registry_effect/handlers/handler_consul_register.py",
        "src/omnibase_infra/nodes/node_session_orchestrator/contract.yaml",
    ]
    hits = find_runtime_paths(changed_files)
    assert len(hits) == 3, "All node handler and contract files should match"


@pytest.mark.integration
def test_runtime_path_patterns_match_runtime_kernel() -> None:
    """Changes to runtime kernel must trigger the deploy gate."""
    changed_files = [
        "src/omnibase_infra/runtime/service_kernel.py",
        "src/omnibase_infra/runtime/auto_wiring/handler_wiring.py",
        "src/omnibase_infra/runtime/models/model_kafka_producer_config.py",
    ]
    hits = find_runtime_paths(changed_files)
    assert len(hits) == 3, "Runtime kernel files should match"


@pytest.mark.integration
def test_runtime_path_patterns_skip_tests_and_docs() -> None:
    """Test files and documentation should NOT trigger the deploy gate."""
    changed_files = [
        "tests/unit/runtime/test_kernel.py",
        "tests/integration/test_catalog_extra_networks.py",
        "docs/patterns/container_dependency_injection.md",
        "README.md",
        ".github/workflows/ci.yml",
    ]
    hits = find_runtime_paths(changed_files)
    assert len(hits) == 0, "Non-runtime files should not trigger gate"


@pytest.mark.integration
def test_has_deploy_evidence_with_docker_exec() -> None:
    """Contract with 'docker exec' deploy check should pass."""
    with tempfile.TemporaryDirectory() as tmpdir:
        contract_path = Path(tmpdir) / "OMN-1234.yaml"
        contract_path.write_text(
            """
dod_evidence:
  - name: "Runtime verification"
    checks:
      - check_type: "runtime_verification"
        check_value: "docker exec omninode-runtime python -c 'print(1)'"
        status: "pending"
"""
        )
        assert has_deploy_evidence(contract_path), (
            "Should detect 'docker exec' evidence"
        )


@pytest.mark.integration
def test_has_deploy_evidence_with_rpk_topic_produce() -> None:
    """Contract with 'rpk topic produce' deploy check should pass."""
    with tempfile.TemporaryDirectory() as tmpdir:
        contract_path = Path(tmpdir) / "OMN-5678.yaml"
        contract_path.write_text(
            """
dod_evidence:
  - name: "Event verification"
    checks:
      - check_type: "event_emission"
        check_value: "rpk topic produce onex.cmd.deploy.rebuild-requested.v1"
        status: "pending"
"""
        )
        assert has_deploy_evidence(contract_path), (
            "Should detect 'rpk topic produce' evidence"
        )


@pytest.mark.integration
def test_has_deploy_evidence_with_deploy_keyword() -> None:
    """Contract with generic 'deploy' keyword should pass."""
    with tempfile.TemporaryDirectory() as tmpdir:
        contract_path = Path(tmpdir) / "OMN-9999.yaml"
        contract_path.write_text(
            """
dod_evidence:
  - name: "Deployment verification"
    checks:
      - check_type: "manual_verification"
        check_value: "Deploy to .201 and verify service starts"
        status: "pending"
"""
        )
        assert has_deploy_evidence(contract_path), (
            "Should detect generic 'deploy' keyword"
        )


@pytest.mark.integration
def test_has_deploy_evidence_false_when_no_deploy_keywords() -> None:
    """Contract without deploy keywords should fail evidence check."""
    with tempfile.TemporaryDirectory() as tmpdir:
        contract_path = Path(tmpdir) / "OMN-0000.yaml"
        contract_path.write_text(
            """
dod_evidence:
  - name: "Unit test coverage"
    checks:
      - check_type: "test_coverage"
        check_value: "pytest tests/unit/nodes/test_foo.py --cov"
        status: "pending"
"""
        )
        assert not has_deploy_evidence(contract_path), (
            "Should reject non-deploy evidence"
        )


@pytest.mark.integration
def test_validate_pr_deploy_gate_passes_when_no_runtime_paths() -> None:
    """Gate should skip when PR touches only docs/tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        contracts_dir = Path(tmpdir)
        changed_files = ["README.md", "tests/unit/test_foo.py"]
        pr_body = "Ticket: OMN-1234"

        result = validate_pr_deploy_gate(
            changed_files=changed_files,
            pr_body=pr_body,
            contracts_dir=contracts_dir,
        )

        assert result.passed
        assert result.skipped
        assert "No runtime paths touched" in result.message


@pytest.mark.integration
def test_validate_pr_deploy_gate_passes_with_override_token() -> None:
    """Gate should pass with [skip-deploy-gate: reason] and log friction."""
    with tempfile.TemporaryDirectory() as tmpdir:
        contracts_dir = Path(tmpdir)
        changed_files = ["docker/Dockerfile.runtime"]
        pr_body = "[skip-deploy-gate: emergency hotfix] Fixing critical bug"

        result = validate_pr_deploy_gate(
            changed_files=changed_files,
            pr_body=pr_body,
            contracts_dir=contracts_dir,
        )

        assert result.passed
        assert result.friction_logged
        assert "emergency hotfix" in result.message
        assert "FRICTION: deploy gate bypassed" in result.message


@pytest.mark.integration
def test_validate_pr_deploy_gate_fails_when_no_ticket_cited() -> None:
    """Gate should fail when runtime paths changed but no OMN-XXXX ticket in PR body."""
    with tempfile.TemporaryDirectory() as tmpdir:
        contracts_dir = Path(tmpdir)
        changed_files = ["src/omnibase_infra/runtime/service_kernel.py"]
        pr_body = "This is a PR description with no ticket ID"

        result = validate_pr_deploy_gate(
            changed_files=changed_files,
            pr_body=pr_body,
            contracts_dir=contracts_dir,
        )

        assert not result.passed
        assert "cites no OMN-XXXX ticket" in result.message
        assert (
            "src/omnibase_infra/runtime/service_kernel.py" in result.runtime_paths_hit
        )


@pytest.mark.integration
def test_validate_pr_deploy_gate_fails_when_ticket_has_no_deploy_evidence() -> None:
    """Gate should fail when cited ticket exists but has no deploy DoD evidence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        contracts_dir = Path(tmpdir)
        changed_files = ["docker/Dockerfile.runtime"]
        pr_body = "Ticket: OMN-1234"

        # Create ticket contract without deploy evidence
        (contracts_dir / "OMN-1234.yaml").write_text(
            """
dod_evidence:
  - name: "Unit tests"
    checks:
      - check_type: "test"
        check_value: "pytest tests/unit/"
        status: "pending"
"""
        )

        result = validate_pr_deploy_gate(
            changed_files=changed_files,
            pr_body=pr_body,
            contracts_dir=contracts_dir,
        )

        assert not result.passed
        assert "missing deploy evidence" in result.message
        assert "OMN-1234" in result.tickets_checked


@pytest.mark.integration
def test_validate_pr_deploy_gate_passes_when_ticket_has_deploy_evidence() -> None:
    """Gate should pass when cited ticket has valid deploy DoD evidence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        contracts_dir = Path(tmpdir)
        changed_files = [
            "docker/Dockerfile.runtime",
            "src/omnibase_infra/runtime/service_kernel.py",
        ]
        pr_body = "Fixes OMN-5678\n\nThis PR updates the runtime kernel."

        # Create ticket contract with deploy evidence
        (contracts_dir / "OMN-5678.yaml").write_text(
            """
dod_evidence:
  - name: "Deployment verification"
    checks:
      - check_type: "runtime_verification"
        check_value: "docker exec omninode-runtime python -c 'import sys; print(sys.version)'"
        status: "pending"
      - check_type: "test_coverage"
        check_value: "pytest tests/unit/ --cov"
        status: "pending"
"""
        )

        result = validate_pr_deploy_gate(
            changed_files=changed_files,
            pr_body=pr_body,
            contracts_dir=contracts_dir,
        )

        assert result.passed
        assert "OMN-5678 has deploy evidence" in result.message
        assert len(result.runtime_paths_hit) == 2
        assert "OMN-5678" in result.tickets_checked


@pytest.mark.integration
def test_validate_pr_deploy_gate_checks_multiple_tickets() -> None:
    """Gate should check all cited tickets until one with deploy evidence is found."""
    with tempfile.TemporaryDirectory() as tmpdir:
        contracts_dir = Path(tmpdir)
        changed_files = ["docker/Dockerfile.runtime"]
        pr_body = "Related: OMN-1111, OMN-2222, OMN-3333"

        # First ticket: no contract file
        # Second ticket: contract exists but no deploy evidence
        (contracts_dir / "OMN-2222.yaml").write_text(
            """
dod_evidence:
  - name: "Tests only"
    checks:
      - check_type: "test"
        check_value: "pytest tests/"
        status: "pending"
"""
        )

        # Third ticket: has deploy evidence (should pass)
        (contracts_dir / "OMN-3333.yaml").write_text(
            """
dod_evidence:
  - name: "Deploy check"
    checks:
      - check_type: "deployment"
        check_value: "deploy to .201 and verify"
        status: "pending"
"""
        )

        result = validate_pr_deploy_gate(
            changed_files=changed_files,
            pr_body=pr_body,
            contracts_dir=contracts_dir,
        )

        assert result.passed
        assert "OMN-3333 has deploy evidence" in result.message
        # Should have checked OMN-1111, OMN-2222, and stopped at OMN-3333
        assert "OMN-3333" in result.tickets_checked
