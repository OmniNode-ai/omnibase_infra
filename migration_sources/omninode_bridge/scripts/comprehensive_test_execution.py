#!/usr/bin/env python3
"""
Comprehensive Test Execution and Coverage Analysis Script

This script provides a systematic approach to test execution, coverage analysis,
and test infrastructure improvements for the OmniNode Bridge project.

Features:
- Automated test execution with proper error handling
- Coverage analysis with detailed reporting
- Test infrastructure validation
- Performance baseline measurement
- Security test execution
- CI/CD preparation

Usage:
    python scripts/comprehensive_test_execution.py [options]

Options:
    --quick         Run only fast unit tests
    --full          Run complete test suite including slow tests
    --coverage      Generate comprehensive coverage reports
    --fix-only      Only fix test infrastructure issues
    --report        Generate HTML and JSON reports
    --baseline      Establish performance baselines
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Test execution configuration
TEST_CONFIG = {
    "unit_tests": {
        "path": "tests/unit/",
        "timeout": 120,
        "parallel": True,
        "coverage_target": 85,
    },
    "integration_tests": {
        "path": "tests/integration/",
        "timeout": 300,
        "parallel": False,
        "coverage_target": 80,
    },
    "security_tests": {
        "path": "tests/test_security*.py",
        "timeout": 180,
        "parallel": True,
        "coverage_target": 95,
    },
    "performance_tests": {
        "path": "tests/performance/",
        "timeout": 600,
        "parallel": False,
        "coverage_target": 70,
    },
}


class TestExecutor:
    """Comprehensive test execution and analysis."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results = {}
        self.start_time = datetime.now()

    def run_command(self, cmd: list[str], timeout: int = 120) -> tuple[bool, str, str]:
        """Execute command with timeout and error handling."""
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            return False, "", f"Command failed: {e!s}"

    def validate_test_infrastructure(self) -> dict[str, bool]:
        """Validate test infrastructure components."""
        print("🔍 Validating test infrastructure...")

        validations = {}

        # Check test dependencies
        success, stdout, stderr = self.run_command(
            [
                "poetry",
                "run",
                "python",
                "-c",
                "import pytest, coverage, testcontainers; print('Dependencies OK')",
            ],
        )
        validations["dependencies"] = success

        # Check fixture availability
        success, stdout, stderr = self.run_command(
            ["poetry", "run", "pytest", "--collect-only", "-q"],
            timeout=60,
        )
        validations["fixtures"] = success and "error" not in stderr.lower()

        # Check Docker availability for testcontainers
        success, stdout, stderr = self.run_command(["docker", "--version"])
        validations["docker"] = success

        # Check test file syntax
        test_files_valid = True
        for test_file in self.project_root.glob("tests/**/*.py"):
            try:
                with open(test_file) as f:
                    compile(f.read(), test_file, "exec")
            except SyntaxError:
                test_files_valid = False
                break
        validations["syntax"] = test_files_valid

        return validations

    def run_unit_tests(self, coverage: bool = True) -> dict[str, any]:
        """Execute unit tests with coverage analysis."""
        print("🧪 Running unit tests...")

        cmd = ["poetry", "run", "pytest", "tests/unit/", "-v", "--tb=short"]

        if coverage:
            cmd.extend(
                [
                    "--cov=omninode_bridge",
                    "--cov-report=term-missing",
                    "--cov-report=json:coverage-unit.json",
                    "--cov-report=html:htmlcov-unit",
                ],
            )

        # Run with timeout
        success, stdout, stderr = self.run_command(
            cmd,
            timeout=TEST_CONFIG["unit_tests"]["timeout"],
        )

        # Parse results
        results = {
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
            "coverage": self.parse_coverage("coverage-unit.json") if coverage else None,
        }

        return results

    def run_integration_tests(self) -> dict[str, any]:
        """Execute integration tests with proper container setup."""
        print("🔗 Running integration tests...")

        # Check if testcontainers are available
        validations = self.validate_test_infrastructure()
        if not validations.get("docker", False):
            return {
                "success": False,
                "stdout": "",
                "stderr": "Docker not available - skipping integration tests",
                "skipped": True,
            }

        cmd = [
            "poetry",
            "run",
            "pytest",
            "tests/integration/",
            "-v",
            "--tb=short",
            "-m",
            "not slow",
        ]

        success, stdout, stderr = self.run_command(
            cmd,
            timeout=TEST_CONFIG["integration_tests"]["timeout"],
        )

        return {
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
            "skipped": False,
        }

    def run_security_tests(self) -> dict[str, any]:
        """Execute security tests."""
        print("🔒 Running security tests...")

        cmd = [
            "poetry",
            "run",
            "pytest",
            "tests/test_security*.py",
            "tests/test_security_integration.py",
            "-v",
            "--tb=short",
        ]

        success, stdout, stderr = self.run_command(
            cmd,
            timeout=TEST_CONFIG["security_tests"]["timeout"],
        )

        return {"success": success, "stdout": stdout, "stderr": stderr}

    def run_performance_tests(self, baseline: bool = False) -> dict[str, any]:
        """Execute performance tests and establish baselines."""
        print("⚡ Running performance tests...")

        cmd = [
            "poetry",
            "run",
            "pytest",
            "tests/performance/",
            "-v",
            "--tb=short",
            "-m",
            "not slow",
        ]

        if baseline:
            cmd.extend(["--benchmark-json=benchmark-results.json"])

        success, stdout, stderr = self.run_command(
            cmd,
            timeout=TEST_CONFIG["performance_tests"]["timeout"],
        )

        results = {"success": success, "stdout": stdout, "stderr": stderr}

        benchmark_path = self.project_root / "benchmark-results.json"
        if baseline:
            if benchmark_path.exists() and benchmark_path.stat().st_size > 0:
                try:
                    with open(benchmark_path) as f:
                        results["benchmarks"] = json.load(f)
                except json.JSONDecodeError:
                    results["benchmarks"] = None
                    results["benchmark_error"] = "Invalid JSON in benchmark results"
            else:
                results["benchmarks"] = None
                results["benchmark_error"] = "No benchmark results file generated"

        return results

    def run_comprehensive_coverage(self) -> dict[str, any]:
        """Run comprehensive coverage analysis across all test categories."""
        print("📊 Running comprehensive coverage analysis...")

        cmd = [
            "poetry",
            "run",
            "pytest",
            "tests/unit/",
            "tests/test_security*.py",
            "tests/test_error_scenarios.py",
            "--cov=omninode_bridge",
            "--cov-report=term-missing",
            "--cov-report=json:coverage-comprehensive.json",
            "--cov-report=html:htmlcov-comprehensive",
            "--cov-fail-under=75",  # Minimum acceptable coverage
            "-v",
            "--tb=short",
        ]

        success, stdout, stderr = self.run_command(cmd, timeout=300)

        coverage_data = self.parse_coverage("coverage-comprehensive.json")

        return {
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
            "coverage": coverage_data,
        }

    def parse_coverage(self, coverage_file: str) -> dict | None:
        """Parse coverage JSON file and extract metrics."""
        coverage_path = self.project_root / coverage_file
        if not coverage_path.exists():
            return None

        try:
            with open(coverage_path) as f:
                data = json.load(f)

            return {
                "total_coverage": data["totals"]["percent_covered"],
                "lines_covered": data["totals"]["covered_lines"],
                "lines_missing": data["totals"]["missing_lines"],
                "files": {
                    filename: {
                        "coverage": file_data["summary"]["percent_covered"],
                        "missing_lines": file_data["summary"]["missing_lines"],
                    }
                    for filename, file_data in data["files"].items()
                    if filename.startswith("src/omninode_bridge/")
                },
            }
        except Exception as e:
            print(f"Warning: Could not parse coverage file {coverage_file}: {e}")
            return None

    def generate_report(self) -> dict[str, any]:
        """Generate comprehensive test execution report."""
        end_time = datetime.now()
        execution_time = (end_time - self.start_time).total_seconds()

        report = {
            "execution_summary": {
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_execution_time_seconds": execution_time,
                "infrastructure_validated": True,
            },
            "test_results": self.results,
            "recommendations": self.generate_recommendations(),
            "next_steps": self.generate_next_steps(),
        }

        return report

    def generate_recommendations(self) -> list[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        # Check unit test coverage
        if "unit_tests" in self.results and self.results["unit_tests"].get("coverage"):
            coverage = self.results["unit_tests"]["coverage"]["total_coverage"]
            if coverage < 85:
                recommendations.append(
                    f"Increase unit test coverage from {coverage:.1f}% to >85%",
                )

        # Check for failing tests
        for test_type, results in self.results.items():
            if not results.get("success", True) and not results.get("skipped", False):
                recommendations.append(f"Fix failing {test_type.replace('_', ' ')}")

        # Infrastructure recommendations
        if not self.validate_test_infrastructure().get("docker", False):
            recommendations.append(
                "Install Docker for integration testing with testcontainers",
            )

        return recommendations

    def generate_next_steps(self) -> list[str]:
        """Generate next steps based on current state."""
        next_steps = [
            "Review test execution report and address failing tests",
            "Implement missing test fixtures identified in failing tests",
            "Enhance integration tests with more comprehensive scenarios",
            "Set up automated test execution in CI/CD pipeline",
            "Establish performance baselines for critical operations",
        ]

        return next_steps


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive Test Execution")
    parser.add_argument("--quick", action="store_true", help="Run only fast unit tests")
    parser.add_argument("--full", action="store_true", help="Run complete test suite")
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage reports",
    )
    parser.add_argument(
        "--fix-only",
        action="store_true",
        help="Only validate infrastructure",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate detailed reports",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Establish performance baselines",
    )

    args = parser.parse_args()

    # Default to quick if no options specified
    if not any([args.quick, args.full, args.coverage, args.fix_only, args.baseline]):
        args.quick = True

    project_root = Path(__file__).parent.parent
    executor = TestExecutor(project_root)

    print("🚀 Starting comprehensive test execution...")
    print(f"📁 Project root: {project_root}")
    print(f"⏰ Start time: {executor.start_time}")
    print()

    # Validate infrastructure first
    validations = executor.validate_test_infrastructure()

    print("Infrastructure validation results:")
    for component, status in validations.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {component}")
    print()

    if args.fix_only:
        print("Infrastructure validation complete. Use results to fix issues.")
        return

    # Execute tests based on options
    if args.quick or args.full:
        # Unit tests
        executor.results["unit_tests"] = executor.run_unit_tests(
            coverage=args.coverage or args.full,
        )

        # Security tests (always run these as they're important)
        executor.results["security_tests"] = executor.run_security_tests()

    if args.full:
        # Integration tests
        executor.results["integration_tests"] = executor.run_integration_tests()

        # Performance tests
        executor.results["performance_tests"] = executor.run_performance_tests(
            baseline=args.baseline,
        )

    if args.coverage:
        # Comprehensive coverage analysis
        executor.results["comprehensive_coverage"] = (
            executor.run_comprehensive_coverage()
        )

    # Generate final report
    report = executor.generate_report()

    # Display summary
    print("\n" + "=" * 60)
    print("📋 TEST EXECUTION SUMMARY")
    print("=" * 60)

    total_success = True
    for test_type, results in executor.results.items():
        if results.get("skipped", False):
            print(f"⏭️  {test_type.replace('_', ' ').title()}: SKIPPED")
        elif results.get("success", False):
            print(f"✅ {test_type.replace('_', ' ').title()}: PASSED")
        else:
            print(f"❌ {test_type.replace('_', ' ').title()}: FAILED")
            total_success = False

    # Coverage summary
    if args.coverage and "comprehensive_coverage" in executor.results:
        coverage_data = executor.results["comprehensive_coverage"].get("coverage")
        if coverage_data:
            print(f"📊 Overall Coverage: {coverage_data['total_coverage']:.1f}%")

    print(
        f"\n⏱️  Total execution time: {report['execution_summary']['total_execution_time_seconds']:.1f}s",
    )

    # Recommendations
    if report["recommendations"]:
        print("\n💡 RECOMMENDATIONS:")
        for rec in report["recommendations"]:
            print(f"   • {rec}")

    # Save detailed report if requested
    if args.report:
        report_file = (
            project_root
            / f"test_execution_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n📄 Detailed report saved to: {report_file}")

    # Exit code based on success
    sys.exit(0 if total_success else 1)


if __name__ == "__main__":
    main()
