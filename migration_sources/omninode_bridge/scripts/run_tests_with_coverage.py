#!/usr/bin/env python3
"""
Comprehensive test runner with coverage reporting for OmniNode Bridge.

This script provides a unified interface for running all tests with comprehensive
coverage reporting. It can be used for local development, CI/CD pipelines, and
quality assurance processes.

Features:
- Run specific test categories (unit, integration, security, performance, error)
- Generate multiple coverage report formats (HTML, XML, JSON, terminal)
- Configurable coverage thresholds
- Test result aggregation and reporting
- Performance metrics and timing
- Parallel test execution options
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table


@dataclass
class TestResult:
    """Test execution result."""

    category: str
    success: bool
    duration: float
    coverage_percentage: float | None = None
    test_count: int = 0
    failed_count: int = 0
    error_message: str | None = None


@dataclass
class CoverageConfig:
    """Coverage configuration."""

    min_coverage: float = 70.0
    fail_under: float = 70.0
    include_branches: bool = True
    parallel: bool = True
    source_dir: str = "src/omninode_bridge"
    config_file: str = ".coveragerc"


class TestRunner:
    """Comprehensive test runner with coverage reporting."""

    def __init__(self, console: Console):
        """Initialize test runner."""
        self.console = console
        self.project_root = Path(__file__).parent.parent
        self.results: list[TestResult] = []

    def run_command(
        self,
        command: list[str],
        category: str,
        timeout: int = 300,
    ) -> TestResult:
        """Run a test command and capture results."""
        start_time = time.time()

        try:
            self.console.print(f"[blue]Running {category} tests...[/blue]")

            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )

            duration = time.time() - start_time

            # Parse test results from output
            test_count, failed_count = self._parse_test_counts(result.stdout)
            coverage_percentage = self._parse_coverage_percentage(result.stdout)

            success = result.returncode == 0
            error_message = result.stderr if not success else None

            test_result = TestResult(
                category=category,
                success=success,
                duration=duration,
                coverage_percentage=coverage_percentage,
                test_count=test_count,
                failed_count=failed_count,
                error_message=error_message,
            )

            self.results.append(test_result)

            # Print immediate result
            status = "✅ PASSED" if success else "❌ FAILED"
            self.console.print(f"{category}: {status} ({duration:.2f}s)")

            if not success and error_message:
                self.console.print(f"[red]Error: {error_message[:200]}...[/red]")

            return test_result

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            test_result = TestResult(
                category=category,
                success=False,
                duration=duration,
                error_message=f"Test timeout after {timeout}s",
            )
            self.results.append(test_result)
            self.console.print(f"{category}: ⏰ TIMEOUT ({duration:.2f}s)")
            return test_result

        except Exception as e:
            duration = time.time() - start_time
            test_result = TestResult(
                category=category,
                success=False,
                duration=duration,
                error_message=str(e),
            )
            self.results.append(test_result)
            self.console.print(f"{category}: ❌ ERROR ({duration:.2f}s) - {e}")
            return test_result

    def _parse_test_counts(self, output: str) -> tuple[int, int]:
        """Parse test counts from pytest output."""
        try:
            # Look for patterns like "123 passed" or "45 failed"
            import re

            passed_match = re.search(r"(\d+) passed", output)
            failed_match = re.search(r"(\d+) failed", output)

            passed = int(passed_match.group(1)) if passed_match else 0
            failed = int(failed_match.group(1)) if failed_match else 0

            return passed + failed, failed

        except Exception:
            return 0, 0

    def _parse_coverage_percentage(self, output: str) -> float | None:
        """Parse coverage percentage from output."""
        try:
            import re

            # Look for patterns like "TOTAL ... 85%"
            coverage_match = re.search(r"TOTAL.*?(\d+)%", output)
            if coverage_match:
                return float(coverage_match.group(1))

            return None

        except Exception:
            return None

    def run_unit_tests(self, coverage_config: CoverageConfig) -> TestResult:
        """Run unit tests with coverage."""
        command = [
            "poetry",
            "run",
            "pytest",
            "tests/unit/",
            f"--cov={coverage_config.source_dir}",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov/unit",
            "--cov-report=xml:coverage-unit.xml",
            f"--cov-fail-under={coverage_config.fail_under}",
            f"--cov-config={coverage_config.config_file}",
            "--verbose",
            "--tb=short",
        ]

        if coverage_config.include_branches:
            command.append("--cov-branch")

        return self.run_command(command, "Unit Tests", timeout=600)

    def run_integration_tests(self) -> TestResult:
        """Run integration tests."""
        command = [
            "poetry",
            "run",
            "pytest",
            "tests/integration/",
            "--verbose",
            "--tb=short",
            "--timeout=300",
        ]

        return self.run_command(command, "Integration Tests", timeout=900)

    def run_security_tests(self) -> TestResult:
        """Run security tests."""
        command = [
            "poetry",
            "run",
            "pytest",
            "tests/test_security*.py",
            "--verbose",
            "--tb=short",
            "--timeout=300",
        ]

        return self.run_command(command, "Security Tests", timeout=600)

    def run_performance_tests(self) -> TestResult:
        """Run performance tests."""
        command = [
            "poetry",
            "run",
            "pytest",
            "tests/performance/",
            "--verbose",
            "--tb=short",
            "--timeout=600",
        ]

        return self.run_command(command, "Performance Tests", timeout=1200)

    def run_error_scenario_tests(self) -> TestResult:
        """Run error scenario tests."""
        command = [
            "poetry",
            "run",
            "pytest",
            "tests/test_error_scenarios.py",
            "--verbose",
            "--tb=short",
            "--timeout=300",
        ]

        return self.run_command(command, "Error Scenario Tests", timeout=600)

    def run_comprehensive_coverage(self, coverage_config: CoverageConfig) -> TestResult:
        """Run all tests with comprehensive coverage."""
        command = [
            "poetry",
            "run",
            "pytest",
            "tests/",
            f"--cov={coverage_config.source_dir}",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov/comprehensive",
            "--cov-report=xml:coverage-comprehensive.xml",
            "--cov-report=json:coverage-comprehensive.json",
            f"--cov-fail-under={coverage_config.fail_under}",
            f"--cov-config={coverage_config.config_file}",
            "--verbose",
        ]

        if coverage_config.include_branches:
            command.append("--cov-branch")

        return self.run_command(command, "Comprehensive Coverage", timeout=1800)

    def generate_coverage_badge(self) -> bool:
        """Generate coverage badge."""
        try:
            command = ["poetry", "run", "coverage-badge", "-o", "coverage-badge.svg"]
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                check=False,
            )
            return result.returncode == 0
        except Exception:
            return False

    def generate_summary_report(self) -> dict:
        """Generate summary report of all test results."""
        total_tests = sum(r.test_count for r in self.results)
        total_failed = sum(r.failed_count for r in self.results)
        total_duration = sum(r.duration for r in self.results)

        successful_categories = [r.category for r in self.results if r.success]
        failed_categories = [r.category for r in self.results if not r.success]

        # Calculate overall coverage (from comprehensive test if available)
        overall_coverage = None
        for result in self.results:
            if (
                result.category == "Comprehensive Coverage"
                and result.coverage_percentage
            ):
                overall_coverage = result.coverage_percentage
                break

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_categories": len(self.results),
                "successful_categories": len(successful_categories),
                "failed_categories": len(failed_categories),
                "total_tests": total_tests,
                "total_failed": total_failed,
                "total_duration": total_duration,
                "overall_coverage": overall_coverage,
            },
            "results": [
                {
                    "category": r.category,
                    "success": r.success,
                    "duration": r.duration,
                    "coverage_percentage": r.coverage_percentage,
                    "test_count": r.test_count,
                    "failed_count": r.failed_count,
                    "error_message": r.error_message,
                }
                for r in self.results
            ],
            "successful_categories": successful_categories,
            "failed_categories": failed_categories,
        }

        return report

    def print_summary_table(self):
        """Print summary table of all test results."""
        table = Table(title="Test Execution Summary")

        table.add_column("Category", style="cyan", no_wrap=True)
        table.add_column("Status", style="bold")
        table.add_column("Duration", justify="right")
        table.add_column("Tests", justify="right")
        table.add_column("Failed", justify="right")
        table.add_column("Coverage", justify="right")

        for result in self.results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            duration = f"{result.duration:.2f}s"
            tests = str(result.test_count) if result.test_count > 0 else "-"
            failed = str(result.failed_count) if result.failed_count > 0 else "-"
            coverage = (
                f"{result.coverage_percentage:.1f}%"
                if result.coverage_percentage
                else "-"
            )

            table.add_row(result.category, status, duration, tests, failed, coverage)

        self.console.print(table)

        # Print overall summary
        successful = sum(1 for r in self.results if r.success)
        total = len(self.results)
        total_duration = sum(r.duration for r in self.results)

        summary_text = (
            f"Overall: {successful}/{total} categories passed in {total_duration:.2f}s"
        )

        if successful == total:
            self.console.print(Panel(f"🎉 {summary_text}", style="green"))
        else:
            self.console.print(Panel(f"⚠️  {summary_text}", style="red"))


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description="Comprehensive test runner with coverage",
    )

    parser.add_argument(
        "--categories",
        nargs="+",
        choices=[
            "unit",
            "integration",
            "security",
            "performance",
            "error",
            "comprehensive",
            "all",
        ],
        default=["all"],
        help="Test categories to run",
    )

    parser.add_argument(
        "--min-coverage",
        type=float,
        default=70.0,
        help="Minimum coverage percentage required",
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel test execution",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="test-reports",
        help="Output directory for test reports",
    )

    parser.add_argument(
        "--json-report",
        type=str,
        help="Path to save JSON summary report",
    )

    parser.add_argument("--badge", action="store_true", help="Generate coverage badge")

    args = parser.parse_args()

    console = Console()
    console.print("[bold blue]OmniNode Bridge - Comprehensive Test Runner[/bold blue]")
    console.print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Configure coverage
    coverage_config = CoverageConfig(
        min_coverage=args.min_coverage,
        fail_under=args.min_coverage,
        parallel=args.parallel,
    )

    runner = TestRunner(console)

    # Determine which tests to run
    categories = args.categories
    if "all" in categories:
        categories = [
            "unit",
            "integration",
            "security",
            "performance",
            "error",
            "comprehensive",
        ]

    console.print(f"Running test categories: {', '.join(categories)}")
    console.print()

    # Run tests
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        total_categories = len(categories)
        task = progress.add_task("Running tests...", total=total_categories)

        for i, category in enumerate(categories):
            progress.update(task, description=f"Running {category} tests...")

            if category == "unit":
                runner.run_unit_tests(coverage_config)
            elif category == "integration":
                runner.run_integration_tests()
            elif category == "security":
                runner.run_security_tests()
            elif category == "performance":
                runner.run_performance_tests()
            elif category == "error":
                runner.run_error_scenario_tests()
            elif category == "comprehensive":
                runner.run_comprehensive_coverage(coverage_config)

            progress.update(task, advance=1)

    console.print()

    # Generate coverage badge if requested
    if args.badge:
        console.print("[blue]Generating coverage badge...[/blue]")
        if runner.generate_coverage_badge():
            console.print("✅ Coverage badge generated successfully")
        else:
            console.print("❌ Failed to generate coverage badge")
        console.print()

    # Print summary
    runner.print_summary_table()

    # Generate and save JSON report
    summary_report = runner.generate_summary_report()

    if args.json_report:
        with open(args.json_report, "w") as f:
            json.dump(summary_report, f, indent=2)
        console.print(f"📄 JSON report saved to {args.json_report}")

    # Save to default location in output directory
    default_json_path = output_dir / "test-summary.json"
    with open(default_json_path, "w") as f:
        json.dump(summary_report, f, indent=2)

    console.print(f"📁 Reports saved to {output_dir}")

    # Exit with appropriate code
    failed_categories = [r for r in runner.results if not r.success]
    if failed_categories:
        console.print(f"\n❌ {len(failed_categories)} test categories failed")
        sys.exit(1)
    else:
        console.print("\n🎉 All test categories passed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
