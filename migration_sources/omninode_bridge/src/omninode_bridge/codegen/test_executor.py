#!/usr/bin/env python3
"""
Test Executor for Generated Code.

Executes pytest on generated tests and analyzes results, providing
comprehensive quality verification for code generation output.

Features:
- Pytest execution via subprocess (clean isolation)
- JSON output parsing for detailed results
- Test classification (unit, integration, performance)
- Coverage measurement integration
- Timeout handling and error recovery
- Actionable failure reports

This closes the critical production readiness gap by validating that
generated code actually works, not just that it was generated.
"""

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ModelTestResults(BaseModel):
    """
    Test execution results with detailed metrics.

    Contains comprehensive information about test execution including
    pass/fail counts, errors, coverage, and detailed failure information.
    """

    # Test counts
    passed: int = Field(..., ge=0, description="Number of tests that passed")
    failed: int = Field(..., ge=0, description="Number of tests that failed")
    skipped: int = Field(..., ge=0, description="Number of tests skipped")
    total: int = Field(..., ge=0, description="Total number of tests run")

    # Execution metrics
    duration_seconds: float = Field(..., ge=0.0, description="Total execution time")
    coverage_percent: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Code coverage percentage"
    )

    # Error tracking
    errors: list[str] = Field(default_factory=list, description="Test execution errors")
    failed_tests: list[dict[str, Any]] = Field(
        default_factory=list, description="Detailed information about failed tests"
    )

    # Test classification
    test_types_run: list[str] = Field(
        default_factory=list, description="Types of tests executed (unit, integration)"
    )

    # Output
    stdout: str = Field(default="", description="Test execution stdout")
    stderr: str = Field(default="", description="Test execution stderr")
    exit_code: int = Field(..., description="Pytest exit code")

    # Metadata
    pytest_version: Optional[str] = Field(None, description="Pytest version used")
    python_version: Optional[str] = Field(None, description="Python version used")

    @property
    def success_rate(self) -> float:
        """Calculate test success rate (0.0-1.0)."""
        if self.total == 0:
            return 0.0
        return self.passed / self.total

    @property
    def is_passing(self) -> bool:
        """Check if all tests passed."""
        return self.failed == 0 and self.total > 0

    def get_summary(self) -> str:
        """Get human-readable test summary."""
        status = "✅ PASSED" if self.is_passing else "❌ FAILED"
        return (
            f"{status}\n"
            f"  Tests: {self.passed}/{self.total} passed "
            f"({self.success_rate:.1%} success rate)\n"
            f"  Failed: {self.failed}\n"
            f"  Skipped: {self.skipped}\n"
            f"  Duration: {self.duration_seconds:.2f}s"
        )


class TestExecutionConfig(BaseModel):
    """Configuration for test execution."""

    # Test selection
    test_types: list[str] = Field(
        default=["unit", "integration"],
        description="Types of tests to run",
    )

    # Execution settings
    timeout_seconds: int = Field(
        default=300, ge=10, le=3600, description="Maximum execution time"
    )
    verbose: bool = Field(default=True, description="Enable verbose output")
    capture_output: bool = Field(default=True, description="Capture stdout/stderr")

    # Coverage
    enable_coverage: bool = Field(default=True, description="Measure code coverage")
    coverage_threshold: float = Field(
        default=80.0, ge=0.0, le=100.0, description="Minimum coverage percentage"
    )

    # Retry
    retry_on_failure: bool = Field(default=False, description="Retry failed tests once")

    # Output
    output_format: str = Field(
        default="json", description="Pytest output format (json, junit, html)"
    )


class TestExecutor:
    """
    Execute generated tests and analyze results.

    Uses subprocess to run pytest in clean isolation, parses JSON output
    for detailed results, and provides comprehensive failure analysis.

    Example:
        >>> executor = TestExecutor()
        >>> results = await executor.run_tests(
        ...     output_directory=Path("./generated_nodes/postgres_crud"),
        ...     test_types=["unit", "integration"]
        ... )
        >>> assert results.is_passing
        >>> assert results.coverage_percent >= 80.0
    """

    def __init__(self, config: Optional[TestExecutionConfig] = None):
        """
        Initialize test executor.

        Args:
            config: Test execution configuration (uses defaults if None)
        """
        self.config = config or TestExecutionConfig()

    async def run_tests(
        self,
        output_directory: Path,
        test_types: Optional[list[str]] = None,
        timeout_seconds: Optional[int] = None,
    ) -> ModelTestResults:
        """
        Run pytest on generated tests.

        Executes tests in subprocess for clean isolation, parses JSON output,
        and returns comprehensive results including failures and coverage.

        Args:
            output_directory: Directory containing generated node and tests
            test_types: Override default test types to run (unit, integration, etc.)
            timeout_seconds: Override default timeout

        Returns:
            ModelTestResults with detailed execution results

        Raises:
            FileNotFoundError: If test directory doesn't exist
            TimeoutError: If tests exceed timeout
            RuntimeError: If pytest execution fails critically

        Example:
            >>> results = await executor.run_tests(
            ...     output_directory=Path("./generated_nodes/vault_secrets"),
            ...     test_types=["unit"],
            ...     timeout_seconds=60
            ... )
            >>> print(results.get_summary())
        """
        test_dir = output_directory / "tests"

        # Validation
        if not test_dir.exists():
            logger.error(f"Test directory not found: {test_dir}")
            raise FileNotFoundError(f"Test directory not found: {test_dir}")

        # Use provided values or config defaults
        test_types = test_types or self.config.test_types
        timeout = timeout_seconds or self.config.timeout_seconds

        logger.info(
            f"Running tests in {test_dir} (types: {test_types}, timeout: {timeout}s)"
        )

        # Build pytest command
        pytest_args = self._build_pytest_args(
            test_dir=test_dir,
            test_types=test_types,
            enable_coverage=self.config.enable_coverage,
        )

        # Execute pytest
        start_time = time.time()
        try:
            # Run pytest from the generated node directory (parent of tests/)
            # This allows relative imports like `from ..node import NodeXxx` to work
            result = await self._run_pytest_subprocess(
                pytest_args, timeout, cwd=output_directory
            )
            duration = time.time() - start_time

            # Parse results
            test_results = self._parse_pytest_output(
                stdout=result["stdout"],
                stderr=result["stderr"],
                exit_code=result["exit_code"],
                duration=duration,
                test_types=test_types,
            )

            logger.info(
                f"Test execution complete: {test_results.passed}/{test_results.total} passed"
            )
            return test_results

        except subprocess.TimeoutExpired as e:
            duration = time.time() - start_time
            logger.error(f"Test execution timed out after {timeout}s")
            return ModelTestResults(
                passed=0,
                failed=0,
                skipped=0,
                total=0,
                duration_seconds=duration,
                coverage_percent=None,
                errors=[f"Test execution timed out after {timeout}s"],
                exit_code=-1,
                stdout="",
                stderr=str(e),
                test_types_run=test_types,
                pytest_version=None,
                python_version=None,
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Test execution failed: {e}")
            return ModelTestResults(
                passed=0,
                failed=0,
                skipped=0,
                total=0,
                duration_seconds=duration,
                coverage_percent=None,
                errors=[f"Test execution failed: {e!s}"],
                exit_code=-1,
                stdout="",
                stderr=str(e),
                test_types_run=test_types,
                pytest_version=None,
                python_version=None,
            )

    def _build_pytest_args(
        self,
        test_dir: Path,
        test_types: list[str],
        enable_coverage: bool,
    ) -> list[str]:
        """
        Build pytest command line arguments.

        Args:
            test_dir: Directory containing tests
            test_types: Types of tests to run (unit, integration)
            enable_coverage: Enable coverage measurement

        Returns:
            List of pytest arguments
        """
        args = [
            "pytest",
            "tests/",  # Use relative path since we run from node directory
        ]

        # Try to use JSON report if available
        try:
            import pytest_jsonreport  # noqa: F401

            args.extend(
                [
                    "--json-report",
                    "--json-report-file=pytest_report.json",
                ]
            )
        except ImportError:
            logger.warning(
                "pytest-json-report not installed, falling back to text parsing"
            )

        # Verbosity
        if self.config.verbose:
            args.append("-v")

        # Coverage
        if enable_coverage:
            args.extend(
                [
                    "--cov",
                    str(test_dir.parent),  # Cover the node directory
                    "--cov-report=term-missing",
                    "--cov-report=json",
                ]
            )

        # Test markers (filter by type)
        if test_types:
            # Build marker expression (e.g., "unit or integration")
            marker_expr = " or ".join(test_types)
            args.extend(["-m", marker_expr])

        # Output format
        if self.config.capture_output:
            args.append("-s")  # Don't capture output

        logger.debug(f"Pytest args: {' '.join(args)}")
        return args

    async def _run_pytest_subprocess(
        self, pytest_args: list[str], timeout: int, cwd: Optional[Path] = None
    ) -> dict[str, Any]:
        """
        Run pytest in subprocess.

        Args:
            pytest_args: Pytest command arguments
            timeout: Timeout in seconds
            cwd: Working directory for pytest execution (default: project root)

        Returns:
            Dict with stdout, stderr, exit_code

        Raises:
            subprocess.TimeoutExpired: If execution times out
        """
        working_dir = cwd or Path.cwd()
        logger.debug(f"Executing: {' '.join(pytest_args)} (cwd={working_dir})")

        # Run pytest
        process = subprocess.run(
            pytest_args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=working_dir,
        )

        return {
            "stdout": process.stdout,
            "stderr": process.stderr,
            "exit_code": process.returncode,
        }

    def _parse_pytest_output(
        self,
        stdout: str,
        stderr: str,
        exit_code: int,
        duration: float,
        test_types: list[str],
    ) -> ModelTestResults:
        """
        Parse pytest output (JSON format).

        Args:
            stdout: Standard output from pytest
            stderr: Standard error from pytest
            exit_code: Pytest exit code
            duration: Execution duration
            test_types: Test types run

        Returns:
            ModelTestResults with parsed data
        """
        # Try to find JSON report (pytest saves it in the working directory)
        # Since we run pytest from the generated node directory, report is there
        default_path = Path.cwd() / "pytest_report.json"
        json_report_path: Optional[Path] = None

        # Check if report exists
        if default_path.exists():
            json_report_path = default_path
        else:
            # Also check in the last executed directory (fallback)
            # This handles the case where test output directory was changed
            for possible_path in [Path.cwd() / "pytest_report.json"]:
                if possible_path.exists():
                    json_report_path = possible_path
                    break

        if json_report_path and json_report_path.exists():
            try:
                with open(json_report_path) as f:
                    report_data = json.load(f)

                # Extract test counts
                summary = report_data.get("summary", {})
                passed = summary.get("passed", 0)
                failed = summary.get("failed", 0)
                skipped = summary.get("skipped", 0)
                total = summary.get("total", 0)

                # Extract failed test details
                failed_tests = []
                tests = report_data.get("tests", [])
                for test in tests:
                    if test.get("outcome") == "failed":
                        failed_tests.append(
                            {
                                "name": test.get("nodeid", "unknown"),
                                "error": test.get("call", {})
                                .get("longrepr", "No error info")
                                .split("\n")[0],  # First line of error
                                "traceback": test.get("call", {})
                                .get("longrepr", "")
                                .split("\n")[:10],  # First 10 lines
                                "file": test.get("nodeid", "").split("::")[0],
                                "line": None,  # Not easily available in JSON
                            }
                        )

                # Extract coverage (if available)
                coverage_percent = None
                coverage_path = Path.cwd() / "coverage.json"
                if coverage_path.exists():
                    try:
                        with open(coverage_path) as f:
                            cov_data = json.load(f)
                            coverage_percent = cov_data.get("totals", {}).get(
                                "percent_covered"
                            )
                    except Exception as e:
                        logger.warning(f"Failed to parse coverage data: {e}")

                # Extract versions
                environment = report_data.get("environment", {})
                pytest_version = environment.get("pytest_version")
                python_version = environment.get("Python")

                # Clean up report files
                json_report_path.unlink(missing_ok=True)
                coverage_path.unlink(missing_ok=True)

                return ModelTestResults(
                    passed=passed,
                    failed=failed,
                    skipped=skipped,
                    total=total,
                    duration_seconds=duration,
                    coverage_percent=coverage_percent,
                    errors=[],
                    failed_tests=failed_tests,
                    test_types_run=test_types,
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=exit_code,
                    pytest_version=pytest_version,
                    python_version=python_version,
                )

            except Exception as e:
                logger.error(f"Failed to parse pytest JSON report: {e}")

        # Fallback: Parse text output
        logger.warning("JSON report not found, falling back to text parsing")
        return self._parse_text_output(stdout, stderr, exit_code, duration, test_types)

    def _parse_text_output(
        self,
        stdout: str,
        stderr: str,
        exit_code: int,
        duration: float,
        test_types: list[str],
    ) -> ModelTestResults:
        """
        Parse pytest text output (fallback when JSON unavailable).

        Args:
            stdout: Standard output
            stderr: Standard error
            exit_code: Exit code
            duration: Execution duration
            test_types: Test types run

        Returns:
            ModelTestResults with basic counts
        """
        # Try to extract counts from output
        passed = 0
        failed = 0
        skipped = 0

        # Look for pattern: "X passed, Y failed, Z skipped"
        import re

        if match := re.search(r"(\d+) passed", stdout):
            passed = int(match.group(1))
        if match := re.search(r"(\d+) failed", stdout):
            failed = int(match.group(1))
        if match := re.search(r"(\d+) skipped", stdout):
            skipped = int(match.group(1))

        total = passed + failed + skipped

        errors = []
        if exit_code != 0:
            errors.append(f"Pytest exited with code {exit_code}")

        return ModelTestResults(
            passed=passed,
            failed=failed,
            skipped=skipped,
            total=total,
            duration_seconds=duration,
            coverage_percent=None,
            errors=errors,
            failed_tests=[],
            test_types_run=test_types,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            pytest_version=None,
            python_version=None,
        )


# Export
__all__ = ["TestExecutor", "ModelTestResults", "TestExecutionConfig"]
