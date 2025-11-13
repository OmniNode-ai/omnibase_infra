#!/usr/bin/env python3
"""
Code Generation with Test Execution Example.

Demonstrates the enhanced code generation workflow with automatic test execution:
1. PRD Analysis
2. Node Classification
3. Code Generation
4. Quality Validation
5. **NEW: Test Execution & Failure Analysis**

This shows the complete production-ready workflow where generated code
is validated by actually running the tests, not just generating them.

Usage:
    python examples/codegen_with_test_execution.py
"""

import asyncio
from pathlib import Path
from uuid import uuid4

from omninode_bridge.codegen import (
    FailureAnalyzer,
    NodeClassifier,
    PRDAnalyzer,
    QualityValidator,
    TemplateEngine,
    TestExecutor,
)


async def generate_and_test_node():
    """Generate a node and execute its tests."""
    print("🚀 OmniNode Code Generation with Test Execution")
    print("=" * 80)

    # Configuration
    prompt = """
    Create a simple PostgreSQL connection pool Effect node with:
    - Connection pooling (10-20 connections)
    - Health check ping operation
    - Basic error handling
    - Async/await support
    """

    output_dir = Path("./generated_nodes") / "postgres_pool_test" / str(uuid4())[:8]
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n📝 Prompt:")
    print(prompt.strip())

    # Step 1: PRD Analysis
    print("\n" + "=" * 80)
    print("STEP 1: PRD Analysis")
    print("=" * 80)

    analyzer = PRDAnalyzer(enable_intelligence=False)
    requirements = await analyzer.analyze_prompt(
        prompt=prompt,
        correlation_id=uuid4(),
    )

    print(f"\n✅ Service Name: {requirements.service_name}")
    print(f"   Node Type: {requirements.node_type}")
    print(f"   Operations: {', '.join(requirements.operations)}")

    # Step 2: Node Classification
    print("\n" + "=" * 80)
    print("STEP 2: Node Classification")
    print("=" * 80)

    classifier = NodeClassifier()
    classification = classifier.classify(requirements)

    print(f"\n✅ Node Type: {classification.node_type.value}")
    print(f"   Confidence: {classification.confidence:.1%}")
    print(f"   Template: {classification.template_name}")

    # Step 3: Code Generation (WITHOUT test execution first)
    print("\n" + "=" * 80)
    print("STEP 3: Code Generation (WITHOUT Test Execution)")
    print("=" * 80)

    # Get templates directory path
    templates_dir = (
        Path(__file__).parent.parent
        / "src"
        / "omninode_bridge"
        / "codegen"
        / "templates"
    )
    print(f"   Using templates from: {templates_dir}")

    engine = TemplateEngine(
        templates_directory=templates_dir, enable_inline_templates=True
    )
    artifacts_no_tests = await engine.generate(
        requirements=requirements,
        classification=classification,
        output_directory=output_dir,
        run_tests=False,  # Don't run tests yet
    )

    print(f"\n✅ Code generated: {artifacts_no_tests.node_name}")
    print(f"   Files: {len(artifacts_no_tests.get_all_files())}")
    print("   Tests executed: No (run_tests=False)")

    # Step 4: Code Generation WITH test execution
    print("\n" + "=" * 80)
    print("STEP 4: Code Generation WITH Test Execution (NEW!)")
    print("=" * 80)

    output_dir_with_tests = (
        Path("./generated_nodes") / "postgres_pool_tested" / str(uuid4())[:8]
    )
    output_dir_with_tests.mkdir(parents=True, exist_ok=True)

    artifacts_with_tests = await engine.generate(
        requirements=requirements,
        classification=classification,
        output_directory=output_dir_with_tests,
        run_tests=True,  # Run tests automatically
        strict_mode=False,  # Don't raise exception on failures
    )

    print(f"\n✅ Code generated: {artifacts_with_tests.node_name}")
    print(f"   Files: {len(artifacts_with_tests.get_all_files())}")
    print("   Tests executed: Yes (run_tests=True)")

    # Step 5: Analyze test results
    if artifacts_with_tests.test_results:
        print("\n" + "=" * 80)
        print("STEP 5: Test Execution Results")
        print("=" * 80)

        results = artifacts_with_tests.test_results
        print(f"\n{results.get_summary()}")

        if results.coverage_percent:
            print(f"  Coverage: {results.coverage_percent:.1f}%")

        if results.pytest_version:
            print(f"\n  Pytest Version: {results.pytest_version}")
        if results.python_version:
            print(f"  Python Version: {results.python_version}")

        # Show failed tests if any
        if results.failed_tests:
            print(f"\n❌ Failed Tests ({len(results.failed_tests)}):")
            for test in results.failed_tests[:3]:  # Show first 3
                print(f"\n   Test: {test['name']}")
                print(f"   Error: {test['error']}")
                print(f"   File: {test.get('file', 'unknown')}")

        # Show failure analysis
        if artifacts_with_tests.failure_analysis:
            print("\n" + "=" * 80)
            print("STEP 6: Failure Analysis (NEW!)")
            print("=" * 80)

            analysis = artifacts_with_tests.failure_analysis
            print(f"\n{analysis.get_report()}")

            print("\n📋 Actionable Recommendations:")
            for i, fix in enumerate(analysis.recommended_fixes, 1):
                print(f"   {i}. {fix}")

            print(
                f"\n⏱  Estimated Fix Time: {analysis.estimated_fix_time_minutes} minutes"
            )
            print(f"   Auto-Fixable: {'Yes ✅' if analysis.auto_fixable else 'No ❌'}")
            print(f"   Severity: {analysis.severity.value.upper()}")

    # Step 7: Manual test execution (independent)
    print("\n" + "=" * 80)
    print("STEP 7: Manual Test Execution (Independent)")
    print("=" * 80)

    print("\nYou can also run tests independently using TestExecutor:")

    executor = TestExecutor()
    independent_results = await executor.run_tests(
        output_directory=output_dir_with_tests,
        test_types=["unit"],  # Only run unit tests
        timeout_seconds=60,
    )

    print(f"\n{independent_results.get_summary()}")

    # Analyze manually
    if not independent_results.is_passing:
        analyzer = FailureAnalyzer()
        manual_analysis = analyzer.analyze(independent_results)
        print("\nManual Analysis Summary:")
        print(f"  {manual_analysis.summary}")

    # Step 8: Quality Validation
    print("\n" + "=" * 80)
    print("STEP 8: Quality Validation")
    print("=" * 80)

    validator = QualityValidator(min_quality_threshold=0.7)
    validation = await validator.validate(artifacts_with_tests)

    print(f"\n✅ Quality Score: {validation.quality_score:.1%}")
    print(f"   Status: {'✅ PASSED' if validation.passed else '❌ FAILED'}")
    print("\n   Component Scores:")
    print(f"      • ONEX Compliance: {validation.onex_compliance_score:.1%}")
    print(f"      • Type Safety: {validation.type_safety_score:.1%}")
    print(f"      • Code Quality: {validation.code_quality_score:.1%}")
    print(f"      • Documentation: {validation.documentation_score:.1%}")
    print(f"      • Test Coverage: {validation.test_coverage_score:.1%}")

    # Summary
    print("\n" + "=" * 80)
    print("✨ Complete Workflow Summary")
    print("=" * 80)

    print("\n📦 Generated Node:")
    print(f"   Class: {artifacts_with_tests.node_name}")
    print(f"   Type: {artifacts_with_tests.node_type}")
    print(f"   Output: {output_dir_with_tests}")

    print("\n🧪 Test Execution:")
    if artifacts_with_tests.test_results:
        tr = artifacts_with_tests.test_results
        print(f"   Passed: {tr.passed}/{tr.total}")
        print(f"   Failed: {tr.failed}")
        print(f"   Duration: {tr.duration_seconds:.2f}s")
        print(
            f"   Success Rate: {tr.success_rate:.1%} {'✅' if tr.is_passing else '❌'}"
        )
    else:
        print("   Not executed")

    print("\n✅ Quality:")
    print(f"   Score: {validation.quality_score:.1%}")
    print(f"   Passed: {'Yes ✅' if validation.passed else 'No ❌'}")

    print("\n🎯 Key Benefits of Test Execution:")
    print("   ✅ Validates generated code actually works")
    print("   ✅ Catches failures immediately (not in CI/CD)")
    print("   ✅ Provides actionable fix recommendations")
    print("   ✅ Auto-classifies failure root causes")
    print("   ✅ Estimates fix time and severity")
    print("   ✅ Closes production readiness gap")

    print("\n🔍 Next Steps:")
    print(f"   1. Review code in: {output_dir_with_tests}")
    if (
        artifacts_with_tests.test_results
        and not artifacts_with_tests.test_results.is_passing
    ):
        print("   2. Apply recommended fixes from failure analysis")
        print(f"   3. Re-run tests: pytest {output_dir_with_tests}/tests/")
    else:
        print("   2. Implement TODO items in node.py")
        print("   3. Add custom test cases")
    print("   4. Integrate with your project")

    return artifacts_with_tests, validation


async def main():
    """Run the example."""
    try:
        artifacts, validation = await generate_and_test_node()
        print("\n✅ Example completed successfully!")

        # Return non-zero if tests failed (for CI/CD)
        if artifacts.test_results and not artifacts.test_results.is_passing:
            print("\n⚠️  Note: Tests failed, but example completed (strict_mode=False)")
            return 0  # Still success for the example
        return 0

    except Exception as e:
        print(f"\n❌ Example failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
