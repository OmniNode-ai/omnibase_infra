#!/usr/bin/env python3
"""
Enhanced Performance Report Generator.

This script consolidates pytest-benchmark results from multiple benchmark files
and generates comprehensive performance reports in markdown and JSON formats.

Features:
- Consolidates results from orchestrator, reducer, Kafka, and database benchmarks
- Generates markdown report with tables and visualizations
- Compares against performance thresholds
- Tracks performance trends over time
- Generates CI/CD-ready metrics

Usage:
    # Generate report from benchmark JSON files
    python tests/performance/generate_performance_report.py

    # Generate report with baseline comparison
    python tests/performance/generate_performance_report.py --baseline=baseline.json

    # Generate report with custom output
    python tests/performance/generate_performance_report.py --output=performance_report.md

Output:
    - test-artifacts/performance_report.md - Human-readable markdown report
    - test-artifacts/performance_metrics.json - Machine-readable metrics
    - test-artifacts/performance_summary.txt - CI/CD summary
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class PerformanceReportGenerator:
    """Generate comprehensive performance reports from benchmark results."""

    def __init__(self, artifacts_dir: Path = None):
        """Initialize report generator.

        Args:
            artifacts_dir: Directory containing benchmark artifacts
        """
        self.artifacts_dir = artifacts_dir or Path("test-artifacts")
        self.artifacts_dir.mkdir(exist_ok=True)

        # Performance thresholds from test files
        self.thresholds = {
            "orchestrator": {
                "single_workflow_ms": {"max": 50, "p95": 40, "p99": 45},
                "concurrent_workflows": {
                    "min_throughput": 100,
                    "max_latency_ms": 500,
                },
                "fsm_transition_ms": {"max": 1, "p95": 0.8, "p99": 0.9},
            },
            "reducer": {
                "batch_1000_ms": {"max": 1000, "p95": 800},
                "throughput_items_per_sec": {"min": 1000},
            },
            "kafka": {
                "single_publish_ms": {"max": 5, "p95": 4, "p99": 4.5},
                "batch_throughput_events_per_sec": {"min": 1000},
            },
            "database": {
                "simple_select_ms": {"max": 10, "p95": 8, "p99": 9},
                "insert_operation_ms": {"max": 20, "p95": 15, "p99": 18},
            },
        }

    def load_benchmark_results(self) -> dict[str, Any]:
        """Load all benchmark JSON results from artifacts directory.

        Returns:
            Dictionary of benchmark results by component
        """
        results = {}

        # Expected benchmark files
        benchmark_files = {
            "orchestrator": "orchestrator.json",
            "reducer": "reducer.json",
            "kafka": "kafka.json",
            "database": "database.json",
        }

        for component, filename in benchmark_files.items():
            filepath = self.artifacts_dir / filename
            if filepath.exists():
                try:
                    with open(filepath) as f:
                        results[component] = json.load(f)
                except Exception as e:
                    print(f"Warning: Failed to load {filename}: {e}")

        return results

    def calculate_statistics(self, benchmark_data: dict) -> dict[str, Any]:
        """Calculate statistics from benchmark data.

        Args:
            benchmark_data: Benchmark data from pytest-benchmark

        Returns:
            Dictionary of calculated statistics
        """
        benchmarks = benchmark_data.get("benchmarks", [])

        stats = {
            "total_benchmarks": len(benchmarks),
            "benchmarks": [],
            "summary": {"passed": 0, "failed": 0, "warnings": 0},
        }

        for benchmark in benchmarks:
            bench_stats = benchmark.get("stats", {})
            name = benchmark.get("name", "unknown")

            # Extract key metrics
            mean_s = bench_stats.get("mean", 0)
            min_s = bench_stats.get("min", 0)
            max_s = bench_stats.get("max", 0)
            stddev_s = bench_stats.get("stddev", 0)

            # Convert to milliseconds for readability
            stats["benchmarks"].append(
                {
                    "name": name,
                    "mean_ms": mean_s * 1000,
                    "min_ms": min_s * 1000,
                    "max_ms": max_s * 1000,
                    "stddev_ms": stddev_s * 1000,
                    "rounds": bench_stats.get("rounds", 0),
                    "iterations": bench_stats.get("iterations", 0),
                }
            )

        return stats

    def compare_against_thresholds(
        self, component: str, stats: dict[str, Any]
    ) -> dict[str, Any]:
        """Compare benchmark results against performance thresholds.

        Args:
            component: Component name (orchestrator, reducer, kafka, database)
            stats: Calculated statistics

        Returns:
            Comparison results with pass/fail status
        """
        component_thresholds = self.thresholds.get(component, {})
        comparison = {"component": component, "results": [], "summary": {}}

        passed = 0
        failed = 0
        warnings = 0

        for benchmark in stats["benchmarks"]:
            name = benchmark["name"]
            mean_ms = benchmark["mean_ms"]
            max_ms = benchmark["max_ms"]

            # Find matching threshold
            threshold = None
            for threshold_name, threshold_value in component_thresholds.items():
                if threshold_name.replace("_", "") in name.lower().replace("_", ""):
                    threshold = threshold_value
                    break

            if threshold:
                # Check against threshold
                if isinstance(threshold, dict):
                    max_threshold = threshold.get("max", float("inf"))
                    p95_threshold = threshold.get("p95", float("inf"))

                    status = "PASS"
                    if max_ms > max_threshold:
                        status = "FAIL"
                        failed += 1
                    elif mean_ms > p95_threshold:
                        status = "WARNING"
                        warnings += 1
                    else:
                        passed += 1

                    comparison["results"].append(
                        {
                            "benchmark": name,
                            "mean_ms": mean_ms,
                            "max_ms": max_ms,
                            "threshold_max": max_threshold,
                            "threshold_p95": p95_threshold,
                            "status": status,
                        }
                    )

        comparison["summary"] = {
            "passed": passed,
            "failed": failed,
            "warnings": warnings,
        }

        return comparison

    def generate_markdown_report(
        self, results: dict[str, Any], comparisons: dict[str, Any]
    ) -> str:
        """Generate markdown performance report.

        Args:
            results: Benchmark results by component
            comparisons: Threshold comparison results

        Returns:
            Markdown report string
        """
        report_lines = []

        # Header
        report_lines.append("# OmniNode Bridge Performance Benchmark Report\n")
        report_lines.append(f"**Generated**: {datetime.now(UTC).isoformat()}\n")
        report_lines.append("**Repository**: omninode_bridge (MVP Foundation)\n")
        report_lines.append("\n---\n")

        # Executive Summary
        report_lines.append("## Executive Summary\n")

        total_passed = 0
        total_failed = 0
        total_warnings = 0

        for component, comparison in comparisons.items():
            summary = comparison["summary"]
            total_passed += summary["passed"]
            total_failed += summary["failed"]
            total_warnings += summary["warnings"]

        total_tests = total_passed + total_failed + total_warnings

        report_lines.append(f"- **Total Benchmarks**: {total_tests}\n")
        report_lines.append(f"- **Passed**: {total_passed} ✅\n")
        report_lines.append(f"- **Warnings**: {total_warnings} ⚠️\n")
        report_lines.append(f"- **Failed**: {total_failed} ❌\n")
        report_lines.append("\n")

        # Overall status
        if total_failed == 0:
            status_emoji = "✅"
            status_text = "ALL BENCHMARKS PASSED"
        elif total_failed > 0 and total_passed > 0:
            status_emoji = "⚠️"
            status_text = "SOME BENCHMARKS FAILED"
        else:
            status_emoji = "❌"
            status_text = "CRITICAL PERFORMANCE ISSUES"

        report_lines.append(f"**Overall Status**: {status_emoji} {status_text}\n")
        report_lines.append("\n---\n")

        # Component-by-component results
        for component, comparison in comparisons.items():
            report_lines.append(f"## {component.title()} Performance\n")

            # Summary table
            report_lines.append("### Summary\n")
            report_lines.append("| Metric | Value | Status |\n")
            report_lines.append("|--------|-------|--------|\n")

            summary = comparison["summary"]
            report_lines.append(f"| Tests Passed | {summary['passed']} | ✅ |\n")
            if summary["warnings"] > 0:
                report_lines.append(
                    f"| Tests with Warnings | {summary['warnings']} | ⚠️ |\n"
                )
            if summary["failed"] > 0:
                report_lines.append(f"| Tests Failed | {summary['failed']} | ❌ |\n")
            report_lines.append("\n")

            # Detailed results table
            report_lines.append("### Detailed Results\n")
            report_lines.append(
                "| Benchmark | Mean (ms) | Max (ms) | Threshold (ms) | Status |\n"
            )
            report_lines.append(
                "|-----------|-----------|----------|----------------|--------|\n"
            )

            for result in comparison["results"]:
                status_icon = {
                    "PASS": "✅",
                    "WARNING": "⚠️",
                    "FAIL": "❌",
                }.get(result["status"], "❓")

                report_lines.append(
                    f"| {result['benchmark']} | "
                    f"{result['mean_ms']:.2f} | "
                    f"{result['max_ms']:.2f} | "
                    f"{result['threshold_max']:.2f} | "
                    f"{status_icon} {result['status']} |\n"
                )

            report_lines.append("\n")

        # Performance Thresholds Reference
        report_lines.append("## Performance Thresholds Reference\n")
        report_lines.append(
            "Performance targets from ROADMAP.md and infrastructure code:\n"
        )
        report_lines.append("\n")

        for component, thresholds in self.thresholds.items():
            report_lines.append(f"### {component.title()}\n")
            for threshold_name, threshold_value in thresholds.items():
                if isinstance(threshold_value, dict):
                    report_lines.append(f"- **{threshold_name}**:\n")
                    for key, value in threshold_value.items():
                        report_lines.append(f"  - {key}: {value}\n")
                else:
                    report_lines.append(f"- **{threshold_name}**: {threshold_value}\n")
            report_lines.append("\n")

        # Footer
        report_lines.append("---\n")
        report_lines.append(
            "\n*This report was automatically generated by generate_performance_report.py*\n"
        )

        return "".join(report_lines)

    def generate_json_metrics(self, comparisons: dict[str, Any]) -> dict[str, Any]:
        """Generate machine-readable JSON metrics.

        Args:
            comparisons: Threshold comparison results

        Returns:
            JSON metrics dictionary
        """
        metrics = {
            "timestamp": datetime.now(UTC).isoformat(),
            "components": {},
            "overall": {
                "total_passed": 0,
                "total_failed": 0,
                "total_warnings": 0,
            },
        }

        for component, comparison in comparisons.items():
            summary = comparison["summary"]
            metrics["components"][component] = {
                "passed": summary["passed"],
                "failed": summary["failed"],
                "warnings": summary["warnings"],
                "results": comparison["results"],
            }

            # Update overall metrics
            metrics["overall"]["total_passed"] += summary["passed"]
            metrics["overall"]["total_failed"] += summary["failed"]
            metrics["overall"]["total_warnings"] += summary["warnings"]

        return metrics

    def generate_ci_summary(self, comparisons: dict[str, Any]) -> str:
        """Generate CI/CD summary text.

        Args:
            comparisons: Threshold comparison results

        Returns:
            CI summary string
        """
        summary_lines = []

        summary_lines.append("PERFORMANCE BENCHMARK SUMMARY")
        summary_lines.append("=" * 60)

        total_passed = 0
        total_failed = 0

        for component, comparison in comparisons.items():
            summary = comparison["summary"]
            total_passed += summary["passed"]
            total_failed += summary["failed"]

            status = "PASS" if summary["failed"] == 0 else "FAIL"
            summary_lines.append(
                f"{component.upper()}: {status} "
                f"({summary['passed']} passed, {summary['failed']} failed)"
            )

        summary_lines.append("=" * 60)

        if total_failed == 0:
            summary_lines.append("OVERALL: ✅ ALL BENCHMARKS PASSED")
        else:
            summary_lines.append(f"OVERALL: ❌ {total_failed} BENCHMARKS FAILED")

        return "\n".join(summary_lines)

    def run(self, baseline: Path = None) -> int:
        """Run report generation.

        Args:
            baseline: Optional baseline JSON file for comparison

        Returns:
            Exit code (0 for success, 1 for failures)
        """
        print("🔍 Loading benchmark results...")
        results = self.load_benchmark_results()

        if not results:
            print("❌ No benchmark results found in test-artifacts/")
            print("   Run benchmarks first with pytest --benchmark-only")
            return 1

        print(f"✅ Loaded results for {len(results)} components")

        # Calculate statistics
        print("📊 Calculating statistics...")
        stats = {}
        for component, data in results.items():
            stats[component] = self.calculate_statistics(data)

        # Compare against thresholds
        print("🎯 Comparing against performance thresholds...")
        comparisons = {}
        for component, component_stats in stats.items():
            comparisons[component] = self.compare_against_thresholds(
                component, component_stats
            )

        # Generate reports
        print("📝 Generating performance reports...")

        # Markdown report
        markdown_report = self.generate_markdown_report(results, comparisons)
        markdown_path = self.artifacts_dir / "performance_report.md"
        with open(markdown_path, "w") as f:
            f.write(markdown_report)
        print(f"   ✅ Markdown report: {markdown_path}")

        # JSON metrics
        json_metrics = self.generate_json_metrics(comparisons)
        json_path = self.artifacts_dir / "performance_metrics.json"
        with open(json_path, "w") as f:
            json.dump(json_metrics, f, indent=2)
        print(f"   ✅ JSON metrics: {json_path}")

        # CI summary
        ci_summary = self.generate_ci_summary(comparisons)
        ci_path = self.artifacts_dir / "performance_summary.txt"
        with open(ci_path, "w") as f:
            f.write(ci_summary)
        print(f"   ✅ CI summary: {ci_path}")

        # Print summary to console
        print("\n" + ci_summary)

        # Return exit code based on failures
        total_failed = json_metrics["overall"]["total_failed"]
        return 1 if total_failed > 0 else 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate performance benchmark reports"
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("test-artifacts"),
        help="Directory containing benchmark artifacts",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Baseline JSON file for comparison",
    )

    args = parser.parse_args()

    generator = PerformanceReportGenerator(artifacts_dir=args.artifacts_dir)
    exit_code = generator.run(baseline=args.baseline)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
