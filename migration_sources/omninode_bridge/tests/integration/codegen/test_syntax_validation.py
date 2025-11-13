#!/usr/bin/env python3
"""
Test script to verify AST-based syntax validation catches all error types.

Tests:
1. Valid Python code → should pass
2. IndentationError → should fail with line number
3. TabError → should fail with line number
4. SyntaxError → should fail with error details
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from omninode_bridge.codegen.quality_gates import QualityGatePipeline


async def test_valid_code():
    """Test validation passes for valid Python code."""
    print("\n" + "=" * 70)
    print("TEST 1: Valid Python Code")
    print("=" * 70)

    code = """
def hello_world():
    \"\"\"Simple valid function.\"\"\"
    message = "Hello, World!"
    return message

class ExampleClass:
    \"\"\"Simple valid class.\"\"\"

    def __init__(self, value: int):
        self.value = value

    def get_value(self) -> int:
        return self.value
"""

    pipeline = QualityGatePipeline(validation_level="development")
    result = await pipeline.validate(code)

    print(f"Result: {'✅ PASSED' if result.passed else '❌ FAILED'}")
    print(f"Quality Score: {result.quality_score:.2f}")
    print(f"Execution Time: {result.total_execution_time_ms:.1f}ms")

    if result.all_issues:
        print("\nIssues Found:")
        for issue in result.all_issues:
            print(f"  - {issue}")

    return result.passed


async def test_indentation_error():
    """Test validation catches IndentationError."""
    print("\n" + "=" * 70)
    print("TEST 2: IndentationError")
    print("=" * 70)

    code = """
def broken_function():
    \"\"\"Function with indentation error.\"\"\"
    message = "This line is fine"
  return message  # Wrong indentation - should be 4 spaces
"""

    pipeline = QualityGatePipeline(validation_level="development")
    result = await pipeline.validate(code)

    print(
        f"Result: {'❌ FAILED (Expected)' if not result.passed else '✅ PASSED (Unexpected!)'}"
    )
    print(f"Quality Score: {result.quality_score:.2f}")
    print(f"Execution Time: {result.total_execution_time_ms:.1f}ms")

    if result.all_issues:
        print("\nIssues Found:")
        for issue in result.all_issues:
            print(f"  - {issue}")

    # Should fail for this test
    return not result.passed


async def test_tab_error():
    """Test validation catches TabError."""
    print("\n" + "=" * 70)
    print("TEST 3: TabError")
    print("=" * 70)

    # Mix tabs and spaces (tab = \t, spaces = regular spaces)
    code = """
def mixed_indentation():
    \"\"\"Function mixing tabs and spaces.\"\"\"
\tmessage = "This uses a tab"
    return message  # This uses spaces
"""

    pipeline = QualityGatePipeline(validation_level="development")
    result = await pipeline.validate(code)

    print(
        f"Result: {'❌ FAILED (Expected)' if not result.passed else '✅ PASSED (Unexpected!)'}"
    )
    print(f"Quality Score: {result.quality_score:.2f}")
    print(f"Execution Time: {result.total_execution_time_ms:.1f}ms")

    if result.all_issues:
        print("\nIssues Found:")
        for issue in result.all_issues:
            print(f"  - {issue}")

    # Should fail for this test
    return not result.passed


async def test_syntax_error():
    """Test validation catches SyntaxError."""
    print("\n" + "=" * 70)
    print("TEST 4: SyntaxError")
    print("=" * 70)

    code = """
def invalid_syntax():
    \"\"\"Function with syntax error.\"\"\"
    message = "Missing closing quote
    return message
"""

    pipeline = QualityGatePipeline(validation_level="development")
    result = await pipeline.validate(code)

    print(
        f"Result: {'❌ FAILED (Expected)' if not result.passed else '✅ PASSED (Unexpected!)'}"
    )
    print(f"Quality Score: {result.quality_score:.2f}")
    print(f"Execution Time: {result.total_execution_time_ms:.1f}ms")

    if result.all_issues:
        print("\nIssues Found:")
        for issue in result.all_issues:
            print(f"  - {issue}")

    # Should fail for this test
    return not result.passed


async def test_fail_fast_behavior():
    """Test that pipeline stops after syntax error (fail-fast)."""
    print("\n" + "=" * 70)
    print("TEST 5: Fail-Fast Behavior")
    print("=" * 70)

    # Use code with actual syntax error
    code = """
def broken():
    return "missing closing quote
"""

    pipeline = QualityGatePipeline(validation_level="strict", enable_mypy=True)
    result = await pipeline.validate(code)

    print(
        f"Result: {'❌ FAILED (Expected)' if not result.passed else '✅ PASSED (Unexpected!)'}"
    )
    print(f"Failed Stages: {result.failed_stages}")
    print(f"Passed Stages: {result.passed_stages}")
    print(f"Skipped Stages: {result.skipped_stages}")
    print(f"Total Stages Run: {len(result.stage_results)}")

    # Verify syntax failed and other stages were not run
    syntax_failed = "syntax" in result.failed_stages
    # In strict mode with mypy, there are 5 stages total
    # If fail-fast works, only syntax stage should have results
    only_syntax_ran = len(result.stage_results) == 1

    if syntax_failed and only_syntax_ran:
        print("✅ Fail-fast working: Only syntax stage ran, others skipped")
        return True
    else:
        print(
            f"⚠️  Checking fail-fast: syntax_failed={syntax_failed}, stages_run={len(result.stage_results)}"
        )
        return False


async def main():
    """Run all tests and report results."""
    print("\n" + "=" * 70)
    print("AST-BASED SYNTAX VALIDATION TEST SUITE")
    print("=" * 70)

    results = {
        "Valid Code": await test_valid_code(),
        "IndentationError Detection": await test_indentation_error(),
        "TabError Detection": await test_tab_error(),
        "SyntaxError Detection": await test_syntax_error(),
        "Fail-Fast Behavior": await test_fail_fast_behavior(),
    }

    print("\n" + "=" * 70)
    print("TEST RESULTS SUMMARY")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")

    total = len(results)
    passed_count = sum(1 for p in results.values() if p)

    print(f"\nTotal: {passed_count}/{total} tests passed")

    if passed_count == total:
        print("\n🎉 All tests passed! Syntax validation is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed_count} test(s) failed.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
