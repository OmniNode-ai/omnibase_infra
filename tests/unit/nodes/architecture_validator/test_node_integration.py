# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for NodeArchitectureValidator.compute().

These tests verify end-to-end behavior of the architecture validator node,
including file counting, rule filtering, and violation aggregation.

Ticket: OMN-1099
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_infra.nodes.architecture_validator.models.model_validation_request import (
    ModelArchitectureValidationRequest,
)
from omnibase_infra.nodes.architecture_validator.node import NodeArchitectureValidator


@pytest.fixture
def container() -> ModelONEXContainer:
    """Create ONEX container for tests."""
    return ModelONEXContainer()


@pytest.fixture
def validator(container: ModelONEXContainer) -> NodeArchitectureValidator:
    """Create validator instance."""
    return NodeArchitectureValidator(container)


@pytest.fixture
def project_temp_dir() -> Path:
    """Create temporary directory within project for testing.

    The NodeArchitectureValidator has security validation that rejects
    absolute paths outside the working directory, so we need to create
    temp files within the project directory.

    IMPORTANT: The temp directory must NOT be within /tests/ because
    ARCH-001 validator exempts files in test directories.

    Yields:
        Path to temporary directory within project (not in tests/).
    """
    # Create temp dir at project root (NOT in tests/) to avoid ARCH-001 exemption
    # Find project root (directory containing pyproject.toml)
    current = Path(__file__).resolve()
    project_root = current.parent
    while project_root != project_root.parent:
        if (project_root / "pyproject.toml").exists():
            break
        project_root = project_root.parent

    temp_dir = project_root / f"_test_tmp_{uuid.uuid4().hex[:8]}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestNodeArchitectureValidatorIntegration:
    """Integration tests for compute() method."""

    def test_compute_single_file_no_violations(
        self, validator: NodeArchitectureValidator, project_temp_dir: Path
    ) -> None:
        """Validate clean file reports correct file count.

        A simple Python file with no architecture violations should:
        - Return valid=True
        - Report files_checked=1
        - Have empty violations list
        """
        # Create clean Python file with no violations
        clean_file = project_temp_dir / "clean.py"
        clean_file.write_text("x = 1\n")

        request = ModelArchitectureValidationRequest(paths=[str(clean_file)])
        result = validator.compute(request)

        assert result.valid is True
        assert result.files_checked == 1
        assert len(result.violations) == 0

    def test_compute_directory_counts_files_correctly(
        self, validator: NodeArchitectureValidator, project_temp_dir: Path
    ) -> None:
        """Validate directory with multiple files reports correct count.

        FILE COUNTING BUG TEST: With N files and 3 validators,
        files_checked should be N (the actual number of files),
        NOT N*3 (files multiplied by validator count).

        This test creates 5 clean Python files and verifies the file count
        matches the actual file count regardless of how many validators run.
        """
        # Create 5 clean Python files in the directory
        file_count = 5
        for i in range(file_count):
            clean_file = project_temp_dir / f"file_{i}.py"
            clean_file.write_text(f"x_{i} = {i}\n")

        request = ModelArchitectureValidationRequest(paths=[str(project_temp_dir)])
        result = validator.compute(request)

        assert result.valid is True
        # CRITICAL: files_checked should be N, not N*validator_count
        assert result.files_checked == file_count, (
            f"Expected files_checked={file_count}, got {result.files_checked}. "
            "This may indicate the file counting bug where files are counted "
            "once per validator instead of once per file."
        )
        assert len(result.violations) == 0

    def test_compute_with_rule_filter(
        self, validator: NodeArchitectureValidator, project_temp_dir: Path
    ) -> None:
        """Pass specific rule_ids, verify only that rule is checked.

        When rule_ids=["ARCH-001"] is specified, only the ARCH-001 validator
        should run. The rules_checked list should contain only ARCH-001.
        """
        # Create clean file
        clean_file = project_temp_dir / "clean.py"
        clean_file.write_text("x = 1\n")

        # Request validation with only ARCH-001 rule
        request = ModelArchitectureValidationRequest(
            paths=[str(clean_file)],
            rule_ids=["ARCH-001"],
        )
        result = validator.compute(request)

        assert result.valid is True
        assert result.rules_checked == ["ARCH-001"]
        assert "ARCH-002" not in result.rules_checked
        assert "ARCH-003" not in result.rules_checked

    def test_compute_fail_fast_stops_on_first_violation(
        self, validator: NodeArchitectureValidator, project_temp_dir: Path
    ) -> None:
        """Verify fail_fast=True stops on first violation found.

        When fail_fast is enabled and a violation is found:
        - Validation should stop immediately
        - Only violations up to the first error should be reported
        - The result should be invalid
        """
        # Create file with ARCH-001 violation (direct handler dispatch)
        bad_file = project_temp_dir / "bad_service.py"
        bad_code = """
class SomeService:
    def process(self):
        handler = HandlerSomething()
        handler.handle(event)  # ARCH-001 violation
"""
        bad_file.write_text(bad_code)

        request = ModelArchitectureValidationRequest(
            paths=[str(bad_file)],
            fail_fast=True,
        )
        result = validator.compute(request)

        assert result.valid is False
        # With fail_fast, should have at least 1 violation and stop
        assert len(result.violations) >= 1
        # The first violation should be from ARCH-001
        assert result.violations[0].rule_id == "ARCH-001"

    def test_compute_aggregates_violations_from_all_validators(
        self, validator: NodeArchitectureValidator, project_temp_dir: Path
    ) -> None:
        """Verify violations are collected from all validators.

        Create a file with violations from multiple rules:
        - ARCH-001: Direct handler dispatch
        - ARCH-002: Handler with event bus access
        - ARCH-003: Orchestrator with FSM

        All violations should be aggregated in the result.
        """
        # Create file with violations from ARCH-001 and ARCH-002
        # (These can both exist in one file - handler with direct dispatch AND publishing)
        multi_violation_file = project_temp_dir / "multi_violation.py"
        bad_code = """
class HandlerBad:
    def __init__(self, container, event_bus):
        self._bus = event_bus  # ARCH-002 violation: handler has bus attribute

    def handle(self, event):
        other_handler = HandlerOther()
        other_handler.handle(event)  # ARCH-001 violation: direct handler dispatch
        self._bus.publish(event)  # ARCH-002 violation: handler publishing


class OrchestratorBad:
    STATES = ["pending", "active"]  # ARCH-003 violation: FSM in orchestrator
    TRANSITIONS = {"pending": ["active"]}  # ARCH-003 violation: FSM transitions

    def __init__(self, container):
        self._state = "pending"  # ARCH-003 violation: FSM state tracking
"""
        multi_violation_file.write_text(bad_code)

        request = ModelArchitectureValidationRequest(
            paths=[str(multi_violation_file)],
            fail_fast=False,  # Must be False to collect all violations
        )
        result = validator.compute(request)

        assert result.valid is False

        # Collect rule IDs from violations
        violation_rule_ids = {v.rule_id for v in result.violations}

        # Should have violations from multiple rules
        assert "ARCH-001" in violation_rule_ids, "Expected ARCH-001 violation"
        assert "ARCH-002" in violation_rule_ids, "Expected ARCH-002 violation"
        assert "ARCH-003" in violation_rule_ids, "Expected ARCH-003 violation"

        # Verify we have multiple violations total
        assert len(result.violations) >= 3, (
            f"Expected at least 3 violations, got {len(result.violations)}"
        )


class TestNodeArchitectureValidatorEdgeCases:
    """Edge case tests for NodeArchitectureValidator integration."""

    def test_compute_empty_paths_list(
        self, validator: NodeArchitectureValidator
    ) -> None:
        """Empty paths list should return valid result with zero files checked."""
        request = ModelArchitectureValidationRequest(paths=[])
        result = validator.compute(request)

        assert result.valid is True
        assert result.files_checked == 0
        assert len(result.violations) == 0

    def test_compute_nonexistent_path(
        self, validator: NodeArchitectureValidator, project_temp_dir: Path
    ) -> None:
        """Nonexistent path should be handled gracefully."""
        nonexistent = project_temp_dir / "does_not_exist.py"

        request = ModelArchitectureValidationRequest(paths=[str(nonexistent)])
        result = validator.compute(request)

        assert result.valid is True
        assert result.files_checked == 0

    def test_compute_mixed_valid_and_invalid_files(
        self, validator: NodeArchitectureValidator, project_temp_dir: Path
    ) -> None:
        """Mix of clean and violating files reports correct totals."""
        # Create 2 clean files
        clean1 = project_temp_dir / "clean1.py"
        clean1.write_text("x = 1\n")
        clean2 = project_temp_dir / "clean2.py"
        clean2.write_text("y = 2\n")

        # Create 1 file with violation
        bad_file = project_temp_dir / "bad.py"
        bad_file.write_text("""
class HandlerBad:
    def __init__(self, container, event_bus):
        self._bus = event_bus
""")

        request = ModelArchitectureValidationRequest(paths=[str(project_temp_dir)])
        result = validator.compute(request)

        # Should find the violation
        assert result.valid is False
        assert len(result.violations) >= 1
        # All 3 files should be checked
        assert result.files_checked >= 3

    def test_compute_with_multiple_rule_filters(
        self, validator: NodeArchitectureValidator, project_temp_dir: Path
    ) -> None:
        """Multiple rule_ids filters to subset of rules."""
        clean_file = project_temp_dir / "clean.py"
        clean_file.write_text("x = 1\n")

        # Request only ARCH-001 and ARCH-003
        request = ModelArchitectureValidationRequest(
            paths=[str(clean_file)],
            rule_ids=["ARCH-001", "ARCH-003"],
        )
        result = validator.compute(request)

        assert result.valid is True
        assert set(result.rules_checked) == {"ARCH-001", "ARCH-003"}
        assert "ARCH-002" not in result.rules_checked


__all__ = [
    "TestNodeArchitectureValidatorIntegration",
    "TestNodeArchitectureValidatorEdgeCases",
]
