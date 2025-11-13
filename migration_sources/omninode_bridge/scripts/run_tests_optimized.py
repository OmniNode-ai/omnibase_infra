#!/usr/bin/env python3
"""
Optimized test runner that uses different configurations for different test types.

This script dramatically improves CI performance by:
- Using fast config for unit tests (1s vs 69s per test)
- Using full config only for integration tests that need it
- Running tests in optimal order (fast first, slow last)
"""

import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd: list[str], description: str) -> tuple[int, float]:
    """Run a command and return (exit_code, duration)."""
    print(f"\n🚀 {description}")
    print(f"Command: {' '.join(cmd)}")
    start_time = time.time()

    try:
        result = subprocess.run(cmd, check=False, capture_output=False)
        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ {description} completed in {duration:.2f}s")
        else:
            print(f"❌ {description} failed in {duration:.2f}s")

        return result.returncode, duration
    except Exception as e:
        duration = time.time() - start_time
        print(f"💥 {description} crashed: {e}")
        return 1, duration


def main():
    """Run optimized test suite."""
    total_start = time.time()
    results = []

    # Test configurations
    configs = {
        "unit": {
            "config": "pytest-unit.ini",
            "paths": ["tests/unit"],
            "description": "Fast unit tests (no containers, no coverage)",
        },
        "integration": {
            "config": "pyproject.toml",
            "paths": ["tests/integration"],
            "description": "Integration tests (with containers)",
        },
        "security": {
            "config": "pytest-unit.ini",  # Security tests are usually fast
            "paths": ["tests/security"],
            "description": "Security tests (fast config)",
        },
        "performance": {
            "config": "pyproject.toml",
            "paths": ["tests/performance"],
            "description": "Performance tests (full config)",
        },
    }

    # Get test type from command line or run all
    test_type = sys.argv[1] if len(sys.argv) > 1 else "all"

    if test_type == "all":
        run_configs = configs
    elif test_type in configs:
        run_configs = {test_type: configs[test_type]}
    else:
        print(f"❌ Unknown test type: {test_type}")
        print(f"Available types: {', '.join(configs.keys())}, all")
        return 1

    print(f"🎯 Running test type: {test_type}")

    # Run tests in optimal order (fast first)
    for config_name, config in run_configs.items():
        # Check if test paths exist
        test_paths = [path for path in config["paths"] if Path(path).exists()]
        if not test_paths:
            print(f"⏭️  Skipping {config_name} - no test files found")
            continue

        # Build command
        cmd = ["poetry", "run", "pytest", "-c", config["config"], *test_paths]

        # Run tests
        exit_code, duration = run_command(cmd, config["description"])
        results.append(
            {"type": config_name, "exit_code": exit_code, "duration": duration}
        )

        # Stop on first failure for CI
        if exit_code != 0 and "--continue-on-failure" not in sys.argv:
            break

    # Summary
    total_duration = time.time() - total_start
    print("\n📊 Test Summary")
    print(f"Total duration: {total_duration:.2f}s")

    failed_tests = []
    for result in results:
        status = "✅" if result["exit_code"] == 0 else "❌"
        print(f"{status} {result['type']}: {result['duration']:.2f}s")
        if result["exit_code"] != 0:
            failed_tests.append(result["type"])

    if failed_tests:
        print(f"\n❌ Failed test types: {', '.join(failed_tests)}")
        return 1
    else:
        print("\n🎉 All tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
