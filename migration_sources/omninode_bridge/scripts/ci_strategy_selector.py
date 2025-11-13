#!/usr/bin/env python3
"""
CI Strategy Selector

Analyzes test execution patterns and recommends the appropriate CI workflow
based on test failure patterns and isolation requirements.
"""

import argparse
import subprocess
import sys
from pathlib import Path


class CIStrategySelector:
    """Selects optimal CI strategy based on test execution patterns."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.workflows_dir = project_root / ".github" / "workflows"

    def analyze_test_patterns(self) -> dict:
        """Analyze current test execution patterns."""
        try:
            # Run tests with collection-only to get test inventory
            result = subprocess.run(
                ["poetry", "run", "pytest", "--collect-only", "-q", "tests/"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )

            if result.returncode != 0:
                print(f"Warning: Test collection failed: {result.stderr}")
                return {"status": "collection_failed", "tests": []}

            # Parse collected tests
            test_lines = [
                line
                for line in result.stdout.split("\n")
                if "::test_" in line and not line.startswith(" ")
            ]

            # Categorize tests
            categories = {
                "unit_services": [],
                "unit_ci": [],
                "integration": [],
                "security": [],
                "performance": [],
                "root_scenarios": [],
            }

            for test_line in test_lines:
                test_path = test_line.split()[0] if test_line.split() else ""

                if "unit/services/" in test_path:
                    categories["unit_services"].append(test_path)
                elif "unit/ci/" in test_path:
                    categories["unit_ci"].append(test_path)
                elif "integration/" in test_path:
                    categories["integration"].append(test_path)
                elif "security" in test_path:
                    categories["security"].append(test_path)
                elif "performance/" in test_path:
                    categories["performance"].append(test_path)
                else:
                    categories["root_scenarios"].append(test_path)

            return {
                "status": "success",
                "total_tests": len(test_lines),
                "categories": categories,
                "category_counts": {k: len(v) for k, v in categories.items()},
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def run_test_execution_analysis(self) -> dict:
        """Run tests and analyze failure patterns."""
        print("🔍 Running test execution analysis...")

        try:
            # Run all tests once to see overall success rate
            result = subprocess.run(
                [
                    "poetry",
                    "run",
                    "pytest",
                    "tests/",
                    "-v",
                    "--tb=short",
                    "--maxfail=10",  # Stop after 10 failures to get quick feedback
                ],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )

            # Parse test results
            output_lines = result.stdout.split("\n") + result.stderr.split("\n")

            failed_tests = []
            passed_tests = []
            error_patterns = []

            for line in output_lines:
                if "FAILED" in line and "::" in line:
                    failed_tests.append(line.split()[0])
                elif "PASSED" in line and "::" in line:
                    passed_tests.append(line.split()[0])
                elif any(
                    error in line
                    for error in ["TypeError", "AttributeError", "ConnectionError"]
                ):
                    error_patterns.append(line.strip())

            success_rate = (
                len(passed_tests) / (len(passed_tests) + len(failed_tests))
                if (passed_tests or failed_tests)
                else 0
            )

            return {
                "success_rate": success_rate,
                "total_passed": len(passed_tests),
                "total_failed": len(failed_tests),
                "failed_tests": failed_tests[:10],  # Limit for readability
                "error_patterns": error_patterns[:5],
                "return_code": result.returncode,
            }

        except Exception as e:
            return {"error": str(e), "success_rate": 0}

    def recommend_ci_strategy(
        self, test_analysis: dict, execution_analysis: dict
    ) -> str:
        """Recommend CI strategy based on analysis results."""

        if test_analysis.get("status") != "success":
            return "standard"  # Fallback to standard CI

        total_tests = test_analysis.get("total_tests", 0)
        success_rate = execution_analysis.get("success_rate", 0)
        failed_count = execution_analysis.get("total_failed", 0)

        print("📊 Analysis Results:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Success Rate: {success_rate:.1%}")
        print(f"   Failed Tests: {failed_count}")

        # Decision logic
        if success_rate >= 0.98:
            print("✅ High success rate - standard CI sufficient")
            return "standard"
        elif success_rate >= 0.90:
            print("⚠️  Moderate success rate - recommend isolated CI")
            return "isolated"
        else:
            print("🚨 Low success rate - recommend ultra-isolated CI")
            return "ultra-isolated"

    def activate_ci_workflow(self, strategy: str) -> bool:
        """Activate the appropriate CI workflow."""
        workflows = {
            "standard": "ci.yml",
            "isolated": "ci-isolated.yml",
            "ultra-isolated": "ci-ultra-isolated.yml",
        }

        target_workflow = workflows.get(strategy)
        if not target_workflow:
            print(f"❌ Unknown strategy: {strategy}")
            return False

        source_path = self.workflows_dir / target_workflow
        active_path = self.workflows_dir / "ci-active.yml"

        if not source_path.exists():
            print(f"❌ Workflow file not found: {source_path}")
            return False

        try:
            # Copy the selected workflow to active
            import shutil

            shutil.copy2(source_path, active_path)
            print(f"✅ Activated {strategy} CI strategy ({target_workflow})")

            # Also disable other workflows by renaming them
            for other_strategy, other_workflow in workflows.items():
                if other_strategy != strategy:
                    other_path = self.workflows_dir / other_workflow
                    disabled_path = self.workflows_dir / f"{other_workflow}.disabled"
                    if other_path.exists() and not disabled_path.exists():
                        shutil.move(other_path, disabled_path)
                        print(f"   Disabled {other_workflow}")

            return True

        except Exception as e:
            print(f"❌ Failed to activate workflow: {e}")
            return False

    def generate_test_summary(
        self, test_analysis: dict, execution_analysis: dict
    ) -> str:
        """Generate a summary report."""
        summary = []
        summary.append("# Test Execution Analysis Summary\n")

        if test_analysis.get("status") == "success":
            summary.append(
                f"**Total Tests Discovered:** {test_analysis['total_tests']}"
            )
            summary.append("\n**Test Categories:**")
            for category, count in test_analysis["category_counts"].items():
                summary.append(f"- {category.replace('_', ' ').title()}: {count} tests")

        summary.append("\n**Execution Results:**")
        success_rate = execution_analysis.get("success_rate", 0)
        summary.append(f"- Success Rate: {success_rate:.1%}")
        summary.append(f"- Passed: {execution_analysis.get('total_passed', 0)}")
        summary.append(f"- Failed: {execution_analysis.get('total_failed', 0)}")

        if execution_analysis.get("failed_tests"):
            summary.append("\n**Recently Failed Tests:**")
            for test in execution_analysis["failed_tests"][:5]:
                summary.append(f"- {test}")

        if execution_analysis.get("error_patterns"):
            summary.append("\n**Common Error Patterns:**")
            for pattern in execution_analysis["error_patterns"][:3]:
                summary.append(f"- {pattern[:100]}...")

        return "\n".join(summary)


def main():
    parser = argparse.ArgumentParser(description="CI Strategy Selector")
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze test patterns and recommend strategy",
    )
    parser.add_argument(
        "--activate",
        choices=["standard", "isolated", "ultra-isolated"],
        help="Activate specific CI strategy",
    )
    parser.add_argument(
        "--report", action="store_true", help="Generate test analysis report"
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path.cwd(), help="Project root directory"
    )

    args = parser.parse_args()

    selector = CIStrategySelector(args.project_root)

    if args.analyze:
        print("🔍 Analyzing test patterns...")
        test_analysis = selector.analyze_test_patterns()
        execution_analysis = selector.run_test_execution_analysis()

        strategy = selector.recommend_ci_strategy(test_analysis, execution_analysis)
        print(f"\n💡 Recommended Strategy: {strategy}")

        if args.report:
            report = selector.generate_test_summary(test_analysis, execution_analysis)
            report_path = args.project_root / "test-analysis-report.md"
            with open(report_path, "w") as f:
                f.write(report)
            print(f"📋 Report saved to: {report_path}")

        # Ask if user wants to activate the recommended strategy
        response = input(f"\nActivate {strategy} strategy? (y/N): ").lower().strip()
        if response in ["y", "yes"]:
            selector.activate_ci_workflow(strategy)

    elif args.activate:
        success = selector.activate_ci_workflow(args.activate)
        sys.exit(0 if success else 1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
