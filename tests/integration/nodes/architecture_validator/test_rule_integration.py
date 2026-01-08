# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for Rule classes with NodeArchitectureValidator.

This module provides comprehensive integration testing for the three Rule classes
introduced in PR #124:
    - RuleNoDirectDispatch (ARCH-001)
    - RuleNoHandlerPublishing (ARCH-002)
    - RuleNoOrchestratorFSM (ARCH-003)

Test Categories:
    1. Multi-Rule Validation: Testing all rules together on mixed files
    2. Result Aggregation: Verifying violations are properly collected
    3. Statelessness: Confirming rules produce consistent results across calls
    4. Thread Safety: Verifying concurrent execution produces correct results
    5. Pipeline Integration: Testing full validation from rules through results

Related:
    - Ticket: OMN-1099 (Architecture Validator)
    - PR: #124 (Protocol-compliant Rule classes)
"""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from omnibase_core.models.container.model_onex_container import ModelONEXContainer

from omnibase_infra.nodes.architecture_validator.models.model_validation_request import (
    ModelArchitectureValidationRequest,
)
from omnibase_infra.nodes.architecture_validator.node import NodeArchitectureValidator
from omnibase_infra.nodes.architecture_validator.validators import (
    RuleNoDirectDispatch,
    RuleNoHandlerPublishing,
    RuleNoOrchestratorFSM,
)

if TYPE_CHECKING:
    from omnibase_infra.nodes.architecture_validator.models import ModelRuleCheckResult
    from omnibase_infra.nodes.architecture_validator.models.model_validation_result import (
        ModelFileValidationResult,
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def container() -> ModelONEXContainer:
    """Create ONEX container for tests."""
    return ModelONEXContainer()


@pytest.fixture
def validator(container: ModelONEXContainer) -> NodeArchitectureValidator:
    """Create NodeArchitectureValidator instance."""
    return NodeArchitectureValidator(container)


@pytest.fixture
def all_rules() -> tuple[
    RuleNoDirectDispatch, RuleNoHandlerPublishing, RuleNoOrchestratorFSM
]:
    """Create instances of all three rule classes."""
    return (
        RuleNoDirectDispatch(),
        RuleNoHandlerPublishing(),
        RuleNoOrchestratorFSM(),
    )


@pytest.fixture
def project_temp_dir() -> Path:
    """Create temporary directory within project for testing.

    The NodeArchitectureValidator has security validation that rejects
    absolute paths outside the working directory. This fixture creates
    temp files within the project directory (but NOT in tests/ to avoid
    exemptions).

    Falls back to test file directory if pyproject.toml not found.

    Yields:
        Path to temporary directory within project.
    """
    current = Path(__file__).resolve()
    project_root = current.parent
    original_parent = project_root  # Save for fallback

    while project_root != project_root.parent:
        if (project_root / "pyproject.toml").exists():
            break
        project_root = project_root.parent

    # Fallback if pyproject.toml not found (use test file's directory)
    if not (project_root / "pyproject.toml").exists():
        project_root = original_parent

    temp_dir = project_root / f"_test_tmp_{uuid.uuid4().hex[:8]}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def create_temp_file(project_temp_dir: Path) -> Callable[[str, str], Path]:
    """Factory fixture for creating temporary Python files.

    Args:
        project_temp_dir: Temporary directory within project.

    Returns:
        A callable that takes (filename, content) and returns the Path.
    """

    def _create(filename: str, content: str) -> Path:
        file_path = project_temp_dir / filename
        file_path.write_text(content, encoding="utf-8")
        return file_path

    return _create


# =============================================================================
# Test Class: Multi-Rule Integration
# =============================================================================


class TestMultiRuleValidation:
    """Tests for validating files with multiple rules together."""

    def test_all_rules_on_clean_file(
        self,
        all_rules: tuple[
            RuleNoDirectDispatch, RuleNoHandlerPublishing, RuleNoOrchestratorFSM
        ],
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """All rules should pass on a clean file.

        A file with no architecture violations should pass all three rules
        independently.
        """
        clean_code = '''
class CleanService:
    """A service that follows all architecture rules."""

    def __init__(self, container):
        self._container = container

    def process(self, event):
        # Correct: dispatch through runtime
        return self.runtime.dispatch(event)
'''
        file_path = create_temp_file("clean_service.py", clean_code)

        for rule in all_rules:
            result = rule.check(str(file_path))
            assert result.passed is True, f"{rule.rule_id} should pass on clean file"

    def test_all_rules_on_mixed_violation_file(
        self,
        all_rules: tuple[
            RuleNoDirectDispatch, RuleNoHandlerPublishing, RuleNoOrchestratorFSM
        ],
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """File with violations from all three rules should fail all checks.

        This tests that each rule independently detects its specific violation
        pattern even when all violations exist in the same file.
        """
        mixed_violation_code = '''
class HandlerBad:
    """Violates ARCH-002: Handler with event bus access."""

    def __init__(self, container, event_bus):
        self._bus = event_bus  # ARCH-002: handler has bus attribute

    def handle(self, event):
        # Violation: handler publishing
        self._bus.publish(SomeEvent())
        return event


class ServiceBad:
    """Violates ARCH-001: Direct handler dispatch."""

    def process(self, event):
        handler = HandlerBad(self.container, self.bus)
        return handler.handle(event)  # ARCH-001: direct dispatch


class OrchestratorBad:
    """Violates ARCH-003: FSM in orchestrator."""

    STATES = ["pending", "processing", "done"]  # ARCH-003: FSM constants
    TRANSITIONS = {"pending": ["processing"]}  # ARCH-003: FSM transitions

    def __init__(self, container):
        self._state = "pending"  # ARCH-003: FSM state attribute

    def orchestrate(self, event):
        pass
'''
        file_path = create_temp_file("mixed_violations.py", mixed_violation_code)

        # Check each rule independently
        (
            rule_no_direct_dispatch,
            rule_no_handler_publishing,
            rule_no_orchestrator_fsm,
        ) = all_rules

        result_001 = rule_no_direct_dispatch.check(str(file_path))
        result_002 = rule_no_handler_publishing.check(str(file_path))
        result_003 = rule_no_orchestrator_fsm.check(str(file_path))

        # Each rule should detect its violation
        assert result_001.passed is False, "ARCH-001 should fail on direct dispatch"
        assert result_002.passed is False, "ARCH-002 should fail on handler publishing"
        assert result_003.passed is False, "ARCH-003 should fail on FSM in orchestrator"

    def test_rules_detect_only_their_violations(
        self,
        all_rules: tuple[
            RuleNoDirectDispatch, RuleNoHandlerPublishing, RuleNoOrchestratorFSM
        ],
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Each rule should only detect violations within its scope.

        Create files with violations from only one rule at a time and verify
        other rules pass.
        """
        (
            rule_no_direct_dispatch,
            rule_no_handler_publishing,
            rule_no_orchestrator_fsm,
        ) = all_rules

        # File with only ARCH-001 violation
        arch001_only = """
class ServiceWithDirectDispatch:
    def process(self, event):
        handler = MyHandler(self.container)
        return handler.handle(event)  # Only ARCH-001 violation
"""
        file_001 = create_temp_file("arch001_only.py", arch001_only)

        assert rule_no_direct_dispatch.check(str(file_001)).passed is False
        assert rule_no_handler_publishing.check(str(file_001)).passed is True
        assert rule_no_orchestrator_fsm.check(str(file_001)).passed is True

        # File with only ARCH-002 violation
        arch002_only = """
class HandlerWithBus:
    def __init__(self, container, event_bus):
        self._bus = event_bus  # Only ARCH-002 violation

    def handle(self, event):
        return event
"""
        file_002 = create_temp_file("arch002_only.py", arch002_only)

        assert rule_no_direct_dispatch.check(str(file_002)).passed is True
        assert rule_no_handler_publishing.check(str(file_002)).passed is False
        assert rule_no_orchestrator_fsm.check(str(file_002)).passed is True

        # File with only ARCH-003 violation
        arch003_only = """
class OrchestratorWithFSM:
    STATES = ["created", "running", "done"]  # Only ARCH-003 violation

    def orchestrate(self, event):
        pass
"""
        file_003 = create_temp_file("arch003_only.py", arch003_only)

        assert rule_no_direct_dispatch.check(str(file_003)).passed is True
        assert rule_no_handler_publishing.check(str(file_003)).passed is True
        assert rule_no_orchestrator_fsm.check(str(file_003)).passed is False


# =============================================================================
# Test Class: Result Aggregation via NodeArchitectureValidator
# =============================================================================


class TestResultAggregation:
    """Tests for proper violation aggregation through the validator node."""

    def test_validator_aggregates_all_violations(
        self,
        validator: NodeArchitectureValidator,
        project_temp_dir: Path,
    ) -> None:
        """NodeArchitectureValidator should collect violations from all rules.

        Create a file with violations from all three rules and verify the
        validator aggregates them correctly.
        """
        multi_violation_file = project_temp_dir / "multi_violation.py"
        multi_violation_code = """
class HandlerBad:
    def __init__(self, container, event_bus):
        self._bus = event_bus  # ARCH-002

    def handle(self, event):
        other_handler = HandlerOther()
        other_handler.handle(event)  # ARCH-001
        self._bus.publish(event)  # ARCH-002


class OrchestratorBad:
    STATES = ["pending", "active"]  # ARCH-003
    TRANSITIONS = {"pending": ["active"]}  # ARCH-003

    def __init__(self, container):
        self._state = "pending"  # ARCH-003
"""
        multi_violation_file.write_text(multi_violation_code)

        request = ModelArchitectureValidationRequest(
            paths=[str(multi_violation_file)],
            fail_fast=False,
        )
        result = validator.compute(request)

        assert result.valid is False
        violation_rule_ids = {v.rule_id for v in result.violations}
        assert "ARCH-001" in violation_rule_ids
        assert "ARCH-002" in violation_rule_ids
        assert "ARCH-003" in violation_rule_ids

    def test_validator_tracks_rules_checked(
        self,
        validator: NodeArchitectureValidator,
        project_temp_dir: Path,
    ) -> None:
        """Validator should correctly report which rules were checked."""
        clean_file = project_temp_dir / "clean.py"
        clean_file.write_text("x = 1\n")

        # Check all rules
        request_all = ModelArchitectureValidationRequest(paths=[str(clean_file)])
        result_all = validator.compute(request_all)

        assert set(result_all.rules_checked) == {"ARCH-001", "ARCH-002", "ARCH-003"}

        # Check specific rules only
        request_subset = ModelArchitectureValidationRequest(
            paths=[str(clean_file)],
            rule_ids=["ARCH-001", "ARCH-003"],
        )
        result_subset = validator.compute(request_subset)

        assert set(result_subset.rules_checked) == {"ARCH-001", "ARCH-003"}
        assert "ARCH-002" not in result_subset.rules_checked

    def test_validator_aggregates_multiple_files(
        self,
        validator: NodeArchitectureValidator,
        project_temp_dir: Path,
    ) -> None:
        """Validator should aggregate violations across multiple files."""
        # File 1: ARCH-001 violation
        file1 = project_temp_dir / "service1.py"
        file1.write_text("""
class Service1:
    def process(self):
        handler = MyHandler()
        handler.handle(event)  # ARCH-001
""")

        # File 2: ARCH-002 violation
        file2 = project_temp_dir / "service2.py"
        file2.write_text("""
class HandlerService2:
    def __init__(self, event_bus):
        self._bus = event_bus  # ARCH-002
""")

        # File 3: Clean
        file3 = project_temp_dir / "service3.py"
        file3.write_text("x = 1\n")

        request = ModelArchitectureValidationRequest(
            paths=[str(project_temp_dir)],
            fail_fast=False,
        )
        result = validator.compute(request)

        assert result.valid is False
        assert result.files_checked == 3
        violation_rule_ids = {v.rule_id for v in result.violations}
        assert "ARCH-001" in violation_rule_ids
        assert "ARCH-002" in violation_rule_ids


# =============================================================================
# Test Class: Statelessness
# =============================================================================


class TestRuleStatelessness:
    """Tests verifying rules are stateless and produce consistent results."""

    def test_multiple_checks_same_result(
        self,
        all_rules: tuple[
            RuleNoDirectDispatch, RuleNoHandlerPublishing, RuleNoOrchestratorFSM
        ],
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Multiple checks on same file should produce identical results.

        This verifies rules don't maintain internal state that affects results.
        """
        violation_code = """
class HandlerBad:
    def __init__(self, container, event_bus):
        self._bus = event_bus

    def handle(self, event):
        other = HandlerOther()
        other.handle(event)
        self._bus.publish(event)


class OrchestratorBad:
    STATES = ["pending"]
"""
        file_path = create_temp_file("stateless_test.py", violation_code)

        for rule in all_rules:
            # Run check multiple times
            results = [rule.check(str(file_path)) for _ in range(5)]

            # All results should be identical
            first_result = results[0]
            for i, result in enumerate(results[1:], start=2):
                assert result.passed == first_result.passed, (
                    f"{rule.rule_id}: Result #{i} differs from first result"
                )
                assert result.rule_id == first_result.rule_id
                # Message should be consistent (if present)
                if first_result.message:
                    assert result.message == first_result.message

    def test_interleaved_checks_independent(
        self,
        all_rules: tuple[
            RuleNoDirectDispatch, RuleNoHandlerPublishing, RuleNoOrchestratorFSM
        ],
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Checking different files interleaved should not affect results.

        This tests that checking one file doesn't affect the result of
        checking a different file.
        """
        clean_code = "x = 1\n"
        violation_code = """
class Service:
    def process(self):
        handler = MyHandler()
        handler.handle(event)
"""
        clean_file = create_temp_file("clean_interleaved.py", clean_code)
        violation_file = create_temp_file("violation_interleaved.py", violation_code)

        rule = all_rules[0]  # RuleNoDirectDispatch

        # Interleave checks
        result_violation_1 = rule.check(str(violation_file))
        result_clean_1 = rule.check(str(clean_file))
        result_violation_2 = rule.check(str(violation_file))
        result_clean_2 = rule.check(str(clean_file))

        # Results should be consistent
        assert result_violation_1.passed is False
        assert result_violation_2.passed is False
        assert result_clean_1.passed is True
        assert result_clean_2.passed is True

    def test_different_instances_same_result(
        self,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Different rule instances should produce identical results.

        This verifies no class-level state affects results.
        """
        violation_code = """
class HandlerWithBus:
    def __init__(self, event_bus):
        self._bus = event_bus
"""
        file_path = create_temp_file("instance_test.py", violation_code)

        # Create multiple instances
        instances = [RuleNoHandlerPublishing() for _ in range(5)]

        results = [instance.check(str(file_path)) for instance in instances]

        # All should produce same result
        for i, result in enumerate(results):
            assert result.passed is False, f"Instance #{i} should detect violation"
            assert result.rule_id == "ARCH-002"


# =============================================================================
# Test Class: Thread Safety
# =============================================================================


class TestThreadSafety:
    """Tests verifying rules are thread-safe for concurrent execution."""

    def test_concurrent_rule_checks(
        self,
        all_rules: tuple[
            RuleNoDirectDispatch, RuleNoHandlerPublishing, RuleNoOrchestratorFSM
        ],
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Rules should be safe for concurrent execution.

        Run multiple checks in parallel threads and verify all results
        are correct and consistent.
        """
        # Create separate files for each rule to ensure reliable detection
        # ARCH-001: Direct handler dispatch (variable name must contain "handler")
        violation_001 = """
class ServiceBad:
    def process(self, event):
        my_handler = MyHandler(self.container)
        my_handler.handle(event)  # ARCH-001 violation
"""
        # ARCH-002: Handler with event bus
        violation_002 = """
class HandlerBad:
    def __init__(self, event_bus):
        self._bus = event_bus  # ARCH-002 violation
"""
        # ARCH-003: Orchestrator with FSM
        violation_003 = """
class OrchestratorBad:
    STATES = ["pending", "done"]  # ARCH-003 violation
"""
        file_001 = create_temp_file("concurrent_001.py", violation_001)
        file_002 = create_temp_file("concurrent_002.py", violation_002)
        file_003 = create_temp_file("concurrent_003.py", violation_003)

        rule_to_file = {
            "ARCH-001": str(file_001),
            "ARCH-002": str(file_002),
            "ARCH-003": str(file_003),
        }

        def check_rule(
            rule: RuleNoDirectDispatch
            | RuleNoHandlerPublishing
            | RuleNoOrchestratorFSM,
        ) -> tuple[str, bool]:
            """Check a rule against its specific violation file."""
            file_path = rule_to_file[rule.rule_id]
            result = rule.check(file_path)
            return (result.rule_id, result.passed)

        # Run all rules concurrently multiple times
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for _ in range(20):  # 20 iterations
                for rule in all_rules:
                    futures.append(executor.submit(check_rule, rule))

            results = [f.result() for f in futures]

        # Verify all results are correct and consistent
        results_by_rule: dict[str, list[bool]] = {}
        for rule_id, passed in results:
            if rule_id not in results_by_rule:
                results_by_rule[rule_id] = []
            results_by_rule[rule_id].append(passed)

        # All results should be False (violation detected) and consistent
        for rule_id in ["ARCH-001", "ARCH-002", "ARCH-003"]:
            rule_results = results_by_rule.get(rule_id, [])
            assert len(rule_results) == 20, f"{rule_id} should have 20 results"
            assert all(not p for p in rule_results), (
                f"{rule_id} should consistently detect violation across all concurrent runs"
            )

    def test_concurrent_different_files(
        self,
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Concurrent checks on different files should not interfere.

        Create multiple files and check them concurrently to verify
        no cross-contamination of results.
        """
        # Create files with different violation patterns
        files: list[tuple[Path, str, bool]] = []  # (path, rule_id, expected_pass)

        clean_code = "x = 1\n"
        violation_001 = """
class Service:
    def process(self):
        h = MyHandler()
        h.handle(event)
"""
        violation_002 = """
class HandlerX:
    def __init__(self, bus):
        self._bus = bus
"""
        violation_003 = """
class OrchestratorX:
    STATES = ["a", "b"]
"""

        files.append(
            (create_temp_file("clean_concurrent.py", clean_code), "ARCH-001", True)
        )
        files.append(
            (create_temp_file("v001_concurrent.py", violation_001), "ARCH-001", False)
        )
        files.append(
            (create_temp_file("v002_concurrent.py", violation_002), "ARCH-002", False)
        )
        files.append(
            (create_temp_file("v003_concurrent.py", violation_003), "ARCH-003", False)
        )

        rule_map = {
            "ARCH-001": RuleNoDirectDispatch(),
            "ARCH-002": RuleNoHandlerPublishing(),
            "ARCH-003": RuleNoOrchestratorFSM(),
        }

        def check_file(
            path: Path, rule_id: str, expected: bool
        ) -> tuple[str, bool, bool]:
            """Check file and return (file_name, expected, actual)."""
            rule = rule_map[rule_id]
            result = rule.check(str(path))
            return (path.name, expected, result.passed)

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for _ in range(10):  # 10 iterations of all files
                for path, rule_id, expected in files:
                    futures.append(executor.submit(check_file, path, rule_id, expected))

            results = [f.result() for f in futures]

        # Verify all results match expected
        for file_name, expected, actual in results:
            assert expected == actual, (
                f"File {file_name}: expected passed={expected}, got passed={actual}"
            )


# =============================================================================
# Test Class: Full Pipeline Integration
# =============================================================================


class TestFullPipelineIntegration:
    """Tests for the complete validation pipeline from rules through results."""

    def test_rule_results_match_validator_results(
        self,
        validator: NodeArchitectureValidator,
        all_rules: tuple[
            RuleNoDirectDispatch, RuleNoHandlerPublishing, RuleNoOrchestratorFSM
        ],
        project_temp_dir: Path,
    ) -> None:
        """Rule class results should be consistent with validator results.

        Verify that checking a file directly with Rule classes produces
        consistent pass/fail outcomes as the NodeArchitectureValidator.
        """
        violation_file = project_temp_dir / "pipeline_test.py"
        violation_file.write_text("""
class HandlerPipeline:
    def __init__(self, bus):
        self._bus = bus  # ARCH-002
""")

        # Check with validator
        request = ModelArchitectureValidationRequest(
            paths=[str(violation_file)],
            rule_ids=["ARCH-002"],
        )
        validator_result = validator.compute(request)

        # Check with rule directly
        rule = all_rules[1]  # RuleNoHandlerPublishing
        rule_result = rule.check(str(violation_file))

        # Both should detect the violation
        assert validator_result.valid is False
        assert rule_result.passed is False

    def test_empty_directory_validation(
        self,
        validator: NodeArchitectureValidator,
        project_temp_dir: Path,
    ) -> None:
        """Validation of empty directory should pass with zero files checked."""
        empty_dir = project_temp_dir / "empty"
        empty_dir.mkdir()

        request = ModelArchitectureValidationRequest(paths=[str(empty_dir)])
        result = validator.compute(request)

        assert result.valid is True
        assert result.files_checked == 0

    def test_fail_fast_mode(
        self,
        validator: NodeArchitectureValidator,
        project_temp_dir: Path,
    ) -> None:
        """Fail-fast mode should stop after first file with violations.

        Create multiple files with violations and verify fail_fast=True
        stops processing additional files after finding violations in one file.

        Note: A single file can have multiple violations (e.g., ARCH-002 can
        flag both a forbidden parameter AND a forbidden attribute). Fail-fast
        stops iteration across files/rules, not within a single validator call.
        """
        # Create multiple files with different violations
        for i in range(5):
            file = project_temp_dir / f"violation_{i}.py"
            file.write_text(f"""
class Handler{i}:
    def __init__(self, bus):
        self._bus = bus  # ARCH-002 violation
""")

        request = ModelArchitectureValidationRequest(
            paths=[str(project_temp_dir)],
            fail_fast=True,
        )
        result = validator.compute(request)

        assert result.valid is False
        # With fail_fast, should have at least 1 violation and stop
        # (may have multiple from the same file/validator call)
        assert len(result.violations) >= 1
        # Should NOT have violations from all 5 files
        # If it processed all files, we'd have at least 5+ violations (one per file)
        # With fail_fast, we should have fewer than that
        assert len(result.violations) < 10, (
            f"Fail-fast should stop early, got {len(result.violations)} violations"
        )

    def test_non_fail_fast_collects_all(
        self,
        validator: NodeArchitectureValidator,
        project_temp_dir: Path,
    ) -> None:
        """Non-fail-fast mode should collect all violations.

        Create multiple files with violations and verify fail_fast=False
        collects all violations.
        """
        file_count = 3
        for i in range(file_count):
            file = project_temp_dir / f"violation_all_{i}.py"
            file.write_text(f"""
class Handler{i}:
    def __init__(self, bus):
        self._bus = bus  # ARCH-002 violation
""")

        request = ModelArchitectureValidationRequest(
            paths=[str(project_temp_dir)],
            fail_fast=False,
        )
        result = validator.compute(request)

        assert result.valid is False
        assert result.files_checked == file_count
        # Should have at least one violation per file
        assert len(result.violations) >= file_count

    def test_correlation_id_not_yet_supported(
        self,
        validator: NodeArchitectureValidator,
        project_temp_dir: Path,
    ) -> None:
        """Document that correlation_id is not currently supported in the request model.

        The ModelArchitectureValidationRequest does not currently accept correlation_id.
        ModelArchitectureValidationResult (a different model for OMN-1138) does support
        correlation_id, but ModelFileValidationResult (used by this validator) does not.

        TODO(OMN-XXXX): Add correlation_id support for distributed tracing:
            1. Add correlation_id field to ModelArchitectureValidationRequest
            2. Add correlation_id field to ModelFileValidationResult
            3. Propagate correlation_id through compute() method
            4. Update this test to verify propagation
        """
        clean_file = project_temp_dir / "correlation_test.py"
        clean_file.write_text("x = 1\n")

        request = ModelArchitectureValidationRequest(
            paths=[str(clean_file)],
        )

        # Verify the request model does NOT have correlation_id
        assert "correlation_id" not in ModelArchitectureValidationRequest.model_fields

        result = validator.compute(request)

        # Verify the result model (ModelFileValidationResult) does NOT have correlation_id
        assert "correlation_id" not in type(result).model_fields

        # Validation should still work correctly
        assert result.valid is True
        assert result.files_checked == 1


# =============================================================================
# Test Class: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_rule_on_non_existent_file(
        self,
        all_rules: tuple[
            RuleNoDirectDispatch, RuleNoHandlerPublishing, RuleNoOrchestratorFSM
        ],
    ) -> None:
        """Rules should handle non-existent files gracefully."""
        fake_path = "/nonexistent/path/to/file.py"

        for rule in all_rules:
            result = rule.check(fake_path)
            # Should pass (graceful handling - can't validate what doesn't exist)
            assert result.passed is True
            assert result.rule_id == rule.rule_id

    def test_rule_on_non_python_file(
        self,
        all_rules: tuple[
            RuleNoDirectDispatch, RuleNoHandlerPublishing, RuleNoOrchestratorFSM
        ],
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Rules should handle non-Python files gracefully."""
        # Create a non-.py file
        txt_file = create_temp_file("readme.txt", "This is not Python code.")

        for rule in all_rules:
            result = rule.check(str(txt_file))
            # Should pass (not applicable)
            assert result.passed is True

    def test_rule_on_syntax_error_file(
        self,
        all_rules: tuple[
            RuleNoDirectDispatch, RuleNoHandlerPublishing, RuleNoOrchestratorFSM
        ],
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Rules should handle files with syntax errors gracefully."""
        syntax_error_code = """
class Broken:
    def method(self)  # Missing colon - syntax error
        pass
"""
        error_file = create_temp_file("syntax_error.py", syntax_error_code)

        for rule in all_rules:
            result = rule.check(str(error_file))
            # Should not crash - graceful handling expected
            # May pass (can't validate) or fail (file-level warning)
            assert result.rule_id == rule.rule_id

    def test_rule_on_empty_file(
        self,
        all_rules: tuple[
            RuleNoDirectDispatch, RuleNoHandlerPublishing, RuleNoOrchestratorFSM
        ],
        create_temp_file: Callable[[str, str], Path],
    ) -> None:
        """Rules should handle empty files gracefully."""
        empty_file = create_temp_file("empty.py", "")

        for rule in all_rules:
            result = rule.check(str(empty_file))
            assert result.passed is True
            assert result.rule_id == rule.rule_id


__all__ = [
    "TestMultiRuleValidation",
    "TestResultAggregation",
    "TestRuleStatelessness",
    "TestThreadSafety",
    "TestFullPipelineIntegration",
    "TestEdgeCases",
]
