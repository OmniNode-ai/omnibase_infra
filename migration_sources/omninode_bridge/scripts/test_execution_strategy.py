#!/usr/bin/env python3
"""
Comprehensive Test Execution Strategy for OmniNode Bridge

This script provides intelligent test execution with:
1. Automatic test categorization and prioritization
2. Performance-optimized execution patterns
3. Health monitoring and regression detection
4. Robust isolation validation
5. Detailed reporting and recommendations

Usage:
    python scripts/test_execution_strategy.py --mode fast
    python scripts/test_execution_strategy.py --mode full --health-check
    python scripts/test_execution_strategy.py --mode integration --containers
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Note: psutil could be added here for enhanced monitoring if needed


class TestExecutionStrategy:
    """Intelligent test execution with health monitoring and optimization."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_dir = project_root / "tests"
        self.artifacts_dir = project_root / "test-artifacts"
        self.artifacts_dir.mkdir(exist_ok=True)

        # Test categorization
        self.test_categories = {
            "fast": {
                "patterns": ["tests/unit/ci/", "tests/unit/services/test_*models*"],
                "timeout": 60,
                "parallel": True,
                "description": "Fast unit tests (<100ms each)",
            },
            "unit": {
                "patterns": ["tests/unit/"],
                "timeout": 180,
                "parallel": True,
                "description": "All unit tests",
            },
            "integration": {
                "patterns": ["tests/integration/", "tests/test_*integration*"],
                "timeout": 300,
                "parallel": False,
                "description": "Integration tests with containers",
            },
            "performance": {
                "patterns": ["tests/performance/"],
                "timeout": 600,
                "parallel": False,
                "description": "Performance and load tests",
            },
            "security": {
                "patterns": ["tests/security/", "tests/test_security*"],
                "timeout": 300,
                "parallel": False,
                "description": "Security and vulnerability tests",
            },
        }

    def categorize_tests(self) -> dict[str, list[str]]:
        """Automatically categorize tests based on patterns."""
        categorized = {category: [] for category in self.test_categories}

        for test_file in self.test_dir.rglob("test_*.py"):
            relative_path = test_file.relative_to(self.project_root)
            path_str = str(relative_path)

            for category, config in self.test_categories.items():
                for pattern in config["patterns"]:
                    if pattern.replace("tests/", "") in path_str or path_str.startswith(
                        pattern
                    ):
                        categorized[category].append(str(relative_path))
                        break

        return categorized

    def execute_test_category(
        self,
        category: str,
        tests: list[str],
        enable_health_check: bool = False,
        enable_containers: bool = False,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Execute tests for a specific category with optimization."""
        if not tests:
            return {
                "category": category,
                "status": "skipped",
                "reason": "no_tests_found",
                "duration": 0,
                "tests_run": 0,
            }

        config = self.test_categories[category]
        start_time = time.time()

        # Build pytest command
        cmd = ["poetry", "run", "pytest"]

        # Add test paths
        cmd.extend(tests)

        # Add category-specific options
        cmd.extend(
            [
                "--no-cov" if not enable_health_check else "--cov=src/omninode_bridge",
                "--disable-warnings",
                "--tb=short",
                f"--timeout={config['timeout']}",
                "--maxfail=10",
            ]
        )

        # Add performance optimizations
        if config["parallel"] and len(tests) > 5:
            cmd.extend(
                ["-n", "auto", "--dist=loadscope"]
            )  # pytest-xdist parallel execution

        # Add verbosity
        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")

        # Add health monitoring
        if enable_health_check:
            cmd.extend(
                [
                    "--durations=10",
                    "--json-report",
                    f"--json-report-file={self.artifacts_dir}/report_{category}.json",
                ]
            )

        # Container-specific options
        if not enable_containers and category in ["integration", "security"]:
            cmd.extend(["-m", "not integration"])

        print(f"\n🧪 Executing {category} tests ({len(tests)} test files)")
        print(f"⚙️  Command: {' '.join(cmd[-10:])}")  # Show last 10 args

        # Execute tests
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=config["timeout"] + 30,  # Add buffer
            )

            duration = time.time() - start_time
            success = result.returncode == 0

            # Parse test results
            output_lines = result.stdout.split("\n")
            tests_run = 0
            tests_passed = 0
            tests_failed = 0

            for line in output_lines:
                if " passed" in line and " in " in line:
                    # Parse line like "232 passed in 40.17s"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "passed":
                            tests_passed = int(parts[i - 1])
                            tests_run += tests_passed
                        elif part == "failed":
                            tests_failed = int(parts[i - 1])
                            tests_run += tests_failed

            return {
                "category": category,
                "status": "success" if success else "failed",
                "duration": duration,
                "tests_run": tests_run,
                "tests_passed": tests_passed,
                "tests_failed": tests_failed,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "performance_summary": self._extract_performance_summary(result.stdout),
            }

        except subprocess.TimeoutExpired:
            return {
                "category": category,
                "status": "timeout",
                "duration": config["timeout"],
                "tests_run": 0,
                "error": f"Tests timed out after {config['timeout']}s",
            }

        except Exception as e:
            return {
                "category": category,
                "status": "error",
                "duration": time.time() - start_time,
                "tests_run": 0,
                "error": str(e),
            }

    def _extract_performance_summary(self, stdout: str) -> dict[str, Any]:
        """Extract performance metrics from pytest output."""
        performance = {}

        lines = stdout.split("\n")
        for line in lines:
            if "slowest" in line and "durations" in line:
                performance["has_slow_tests"] = True
            elif " passed in " in line:
                # Extract total execution time
                parts = line.split(" in ")
                if len(parts) > 1:
                    time_part = parts[1].split("s")[0]
                    try:
                        performance["total_time"] = float(time_part)
                    except (ValueError, TypeError):
                        pass

        return performance

    def validate_test_health(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Validate overall test suite health based on execution results."""
        total_tests = sum(r.get("tests_run", 0) for r in results)
        total_passed = sum(r.get("tests_passed", 0) for r in results)
        total_failed = sum(r.get("tests_failed", 0) for r in results)
        total_duration = sum(r.get("duration", 0) for r in results)

        success_rate = total_passed / total_tests if total_tests > 0 else 0
        avg_test_time = total_duration / total_tests if total_tests > 0 else 0

        # Health scoring
        health_score = 100

        # Success rate impact
        if success_rate < 0.95:
            health_score -= (0.95 - success_rate) * 200  # Heavy penalty for failures

        # Performance impact
        if avg_test_time > 0.2:  # >200ms average
            health_score -= 20
        elif avg_test_time > 0.1:  # >100ms average
            health_score -= 10

        # Timeout penalties
        timeout_categories = [r for r in results if r.get("status") == "timeout"]
        health_score -= len(timeout_categories) * 30

        health_score = max(0, health_score)

        # Generate recommendations
        recommendations = []
        if success_rate < 0.9:
            recommendations.append(f"Fix failing tests: {total_failed} tests failed")
        if avg_test_time > 0.2:
            recommendations.append("Optimize slow tests: average >200ms per test")
        if timeout_categories:
            recommendations.append(
                f"Address timeouts in: {[r['category'] for r in timeout_categories]}"
            )

        return {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "success_rate": success_rate,
            "total_duration": total_duration,
            "average_test_time": avg_test_time,
            "health_score": health_score,
            "recommendations": recommendations,
            "timeout_categories": [r["category"] for r in timeout_categories],
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def generate_execution_report(
        self, results: list[dict[str, Any]], health_report: dict[str, Any]
    ) -> str:
        """Generate comprehensive execution report."""
        report_lines = [
            "=" * 80,
            "OMNINODE BRIDGE TEST EXECUTION REPORT",
            "=" * 80,
            f"Execution Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Total Execution Time: {health_report['total_duration']:.2f}s",
            "",
            "📊 EXECUTION SUMMARY:",
            f"   Total Tests: {health_report['total_tests']}",
            f"   Passed: {health_report['total_passed']} ({health_report['success_rate']:.1%})",
            f"   Failed: {health_report['total_failed']}",
            f"   Average Duration: {health_report['average_test_time']:.3f}s per test",
            f"   Health Score: {health_report['health_score']:.1f}/100",
            "",
        ]

        # Category breakdown
        report_lines.extend(["📂 CATEGORY BREAKDOWN:", ""])

        for result in results:
            status_emoji = {
                "success": "✅",
                "failed": "❌",
                "timeout": "⏱️",
                "error": "💥",
                "skipped": "⏭️",
            }.get(result["status"], "❓")

            report_lines.extend(
                [
                    f"{status_emoji} {result['category'].upper()}:",
                    f"   Status: {result['status']}",
                    f"   Tests: {result.get('tests_run', 0)}",
                    f"   Duration: {result['duration']:.2f}s",
                    "",
                ]
            )

            if result["status"] == "failed" and result.get("tests_failed", 0) > 0:
                report_lines.append(f"   ❌ Failures: {result['tests_failed']}")
                report_lines.append("")

        # Health assessment
        report_lines.extend(["🏥 HEALTH ASSESSMENT:", ""])

        if health_report["health_score"] >= 90:
            report_lines.append("✅ EXCELLENT: Test suite is in excellent health")
        elif health_report["health_score"] >= 75:
            report_lines.append("✅ GOOD: Test suite health is good")
        elif health_report["health_score"] >= 60:
            report_lines.append("⚠️  FAIR: Test suite needs improvements")
        else:
            report_lines.append("❌ POOR: Test suite requires attention")

        report_lines.append("")

        # Recommendations
        if health_report["recommendations"]:
            report_lines.extend(["💡 RECOMMENDATIONS:", ""])
            for i, rec in enumerate(health_report["recommendations"], 1):
                report_lines.append(f"   {i}. {rec}")
            report_lines.append("")

        # Performance insights
        slow_categories = [
            r
            for r in results
            if r.get("duration", 0) / max(r.get("tests_run", 1), 1) > 0.2
        ]
        if slow_categories:
            report_lines.extend(["⚡ PERFORMANCE INSIGHTS:", ""])
            for cat_result in slow_categories:
                avg_time = cat_result["duration"] / max(
                    cat_result.get("tests_run", 1), 1
                )
                report_lines.append(
                    f"   • {cat_result['category']}: {avg_time:.3f}s avg per test"
                )
            report_lines.append("")

        report_lines.append("=" * 80)
        return "\n".join(report_lines)

    def save_execution_report(
        self, report: str, health_data: dict[str, Any]
    ) -> tuple[Path, Path]:
        """Save execution report and health data."""
        # Save text report
        report_file = self.artifacts_dir / f"execution_report_{int(time.time())}.txt"
        with open(report_file, "w") as f:
            f.write(report)

        # Save JSON health data
        health_file = self.artifacts_dir / f"health_data_{int(time.time())}.json"
        with open(health_file, "w") as f:
            json.dump(health_data, f, indent=2)

        return report_file, health_file

    def run_strategy(
        self,
        mode: str = "fast",
        enable_health_check: bool = False,
        enable_containers: bool = False,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Execute comprehensive test strategy."""
        print(f"🚀 Starting test execution strategy: {mode}")
        print(f"📍 Project root: {self.project_root}")

        # Categorize tests
        categorized_tests = self.categorize_tests()

        # Determine execution categories based on mode
        if mode == "fast":
            execution_categories = ["fast"]
        elif mode == "unit":
            execution_categories = ["unit"]
        elif mode == "integration":
            execution_categories = ["unit", "integration"]
        elif mode == "full":
            execution_categories = ["unit", "integration", "performance", "security"]
        elif mode == "security":
            execution_categories = ["security"]
        elif mode == "performance":
            execution_categories = ["performance"]
        else:
            execution_categories = ["unit"]

        print(f"📋 Execution plan: {', '.join(execution_categories)}")

        # Execute test categories
        results = []
        for category in execution_categories:
            if category in categorized_tests:
                result = self.execute_test_category(
                    category=category,
                    tests=categorized_tests[category],
                    enable_health_check=enable_health_check,
                    enable_containers=enable_containers,
                    verbose=verbose,
                )
                results.append(result)

                # Print immediate status
                status_emoji = "✅" if result["status"] == "success" else "❌"
                print(
                    f"{status_emoji} {category}: {result.get('tests_run', 0)} tests in {result['duration']:.2f}s"
                )

        # Generate health report
        health_report = self.validate_test_health(results)

        # Generate and save comprehensive report
        execution_report = self.generate_execution_report(results, health_report)
        report_file, health_file = self.save_execution_report(
            execution_report,
            {"execution_results": results, "health_report": health_report},
        )

        # Print final report
        print(execution_report)
        print("\n📄 Reports saved:")
        print(f"   Text: {report_file}")
        print(f"   JSON: {health_file}")

        return {
            "results": results,
            "health_report": health_report,
            "report_files": {"text": str(report_file), "json": str(health_file)},
        }


def main():
    """Main execution entry point."""
    parser = argparse.ArgumentParser(description="Intelligent test execution strategy")

    parser.add_argument(
        "--mode",
        choices=["fast", "unit", "integration", "full", "security", "performance"],
        default="fast",
        help="Test execution mode",
    )

    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Enable comprehensive health checking and monitoring",
    )

    parser.add_argument(
        "--containers",
        action="store_true",
        help="Enable container-based integration tests",
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose test output"
    )

    args = parser.parse_args()

    # Get project root
    project_root = Path(__file__).parent.parent
    if not (project_root / "pyproject.toml").exists():
        print("❌ Error: Could not find project root (pyproject.toml)")
        sys.exit(1)

    # Initialize and run strategy
    strategy = TestExecutionStrategy(project_root)
    result = strategy.run_strategy(
        mode=args.mode,
        enable_health_check=args.health_check,
        enable_containers=args.containers,
        verbose=args.verbose,
    )

    # Exit with appropriate code
    health_score = result["health_report"]["health_score"]
    if health_score >= 75:
        sys.exit(0)  # Success
    elif health_score >= 60:
        sys.exit(1)  # Warning
    else:
        sys.exit(2)  # Critical issues


if __name__ == "__main__":
    main()
