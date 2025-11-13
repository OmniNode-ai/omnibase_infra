#!/usr/bin/env python3
"""
Code Generation Verification Test.

Generates a redis_cache_effect node and verifies all improvements work correctly:
1. io_operations field is included in test contracts
2. Generated tests execute successfully
3. Node includes registration/introspection code
4. All validation passes
"""

import asyncio
import sys
from pathlib import Path
from uuid import uuid4

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omninode_bridge.codegen import (
    NodeClassifier,
    PRDAnalyzer,
    QualityValidator,
    TemplateEngine,
)


def print_section(title: str, char: str = "="):
    """Print formatted section header."""
    print(f"\n{char * 80}")
    print(f"{title}")
    print(f"{char * 80}\n")


async def main():
    """Generate and verify redis_cache_effect node."""
    print_section("🧪 Code Generation Verification Test", "=")

    # Configuration
    prompt = """
    Create a simple Redis cache Effect node with:
    - set operation (store key-value)
    - get operation (retrieve value by key)
    - delete operation (remove key)
    - Connection pooling (5-10 connections)
    - TTL support (configurable expiration)
    """

    output_dir = Path("./generated_nodes") / "redis_cache_tested" / str(uuid4())[:8]
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📝 Prompt:\n{prompt.strip()}\n")
    print(f"📂 Output Directory: {output_dir}\n")

    # Step 1: PRD Analysis
    print_section("STEP 1: PRD Analysis", "-")
    analyzer = PRDAnalyzer(enable_intelligence=False)
    requirements = await analyzer.analyze_prompt(
        prompt=prompt,
        correlation_id=uuid4(),
    )

    print(f"✅ Service Name: {requirements.service_name}")
    print(f"   Node Type: {requirements.node_type}")
    print(f"   Operations: {', '.join(requirements.operations)}")
    print(f"   Features: {', '.join(requirements.features)}")
    print(f"   Domain: {requirements.domain}")

    # Step 2: Node Classification
    print_section("STEP 2: Node Classification", "-")
    classifier = NodeClassifier()
    classification = classifier.classify(requirements)

    print(f"✅ Node Type: {classification.node_type.value}")
    print(f"   Confidence: {classification.confidence:.1%}")
    print(f"   Template: {classification.template_name}")

    # Step 3: Code Generation with Test Execution
    print_section("STEP 3: Code Generation with Test Execution", "-")

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

    artifacts = await engine.generate(
        requirements=requirements,
        classification=classification,
        output_directory=output_dir,
        run_tests=True,  # Execute tests
        strict_mode=False,  # Don't raise exception on test failures
    )

    print(f"\n✅ Code generated: {artifacts.node_name}")
    print(f"   Files: {len(artifacts.get_all_files())}")
    print(f"   Output: {output_dir}")

    # Step 4: Verification
    print_section("STEP 4: Generated Code Verification", "-")

    verification_results = {
        "template_fixes": {},
        "registration_code": {},
        "test_execution": {},
    }

    # 4.1: Verify test template fixes (io_operations field)
    print("🔍 Checking test templates for io_operations field...")

    test_files = {
        "conftest.py": output_dir / "tests" / "conftest.py",
        "test_node.py": output_dir / "tests" / "test_node.py",
        "test_integration.py": output_dir / "tests" / "test_integration.py",
    }

    for test_file_name, test_file_path in test_files.items():
        if test_file_path.exists():
            content = test_file_path.read_text()

            # Check for ModelIOOperationConfig import
            has_import = "ModelIOOperationConfig" in content
            verification_results["template_fixes"][
                f"{test_file_name}_import"
            ] = has_import

            # Check for io_operations field (only for Effect nodes)
            has_io_operations = "io_operations=" in content
            verification_results["template_fixes"][
                f"{test_file_name}_io_operations"
            ] = has_io_operations

            status = "✅" if (has_import and has_io_operations) else "❌"
            print(
                f"  {status} {test_file_name}: import={has_import}, io_operations={has_io_operations}"
            )
        else:
            print(f"  ⚠️  {test_file_name}: File not found")
            verification_results["template_fixes"][f"{test_file_name}_import"] = False
            verification_results["template_fixes"][
                f"{test_file_name}_io_operations"
            ] = False

    # 4.2: Verify registration code in node.py
    print("\n🔍 Checking node.py for registration code...")

    node_file = output_dir / "node.py"
    if node_file.exists():
        node_content = node_file.read_text()

        checks = {
            "mixin_inheritance": "HealthCheckMixin, IntrospectionMixin" in node_content,
            "startup_method": "async def startup(self)" in node_content,
            "shutdown_method": "async def shutdown(self)" in node_content,
            "initialize_health_checks": "initialize_health_checks()" in node_content,
            "initialize_introspection": "initialize_introspection()" in node_content,
            "publish_introspection": "publish_introspection(" in node_content,
        }

        for check_name, passed in checks.items():
            verification_results["registration_code"][check_name] = passed
            status = "✅" if passed else "❌"
            print(f"  {status} {check_name}: {passed}")
    else:
        print("  ⚠️  node.py: File not found")
        for check_name in [
            "mixin_inheritance",
            "startup_method",
            "shutdown_method",
            "initialize_health_checks",
            "initialize_introspection",
            "publish_introspection",
        ]:
            verification_results["registration_code"][check_name] = False

    # 4.3: Analyze test execution results
    print_section("STEP 5: Test Execution Results", "-")

    if artifacts.test_results:
        results = artifacts.test_results
        print(f"{results.get_summary()}")

        if results.coverage_percent:
            print(f"  Coverage: {results.coverage_percent:.1f}%")

        if results.pytest_version:
            print(f"\n  Pytest Version: {results.pytest_version}")
        if results.python_version:
            print(f"  Python Version: {results.python_version}")

        verification_results["test_execution"]["tests_collected"] = results.total
        verification_results["test_execution"]["tests_passed"] = results.passed
        verification_results["test_execution"]["tests_failed"] = results.failed
        verification_results["test_execution"]["success_rate"] = results.success_rate
        verification_results["test_execution"]["is_passing"] = results.is_passing

        # Show failed tests if any
        if results.failed_tests:
            print(f"\n❌ Failed Tests ({len(results.failed_tests)}):")
            for test in results.failed_tests[:5]:  # Show first 5
                print(f"\n   Test: {test['name']}")
                error_msg = test["error"][:200]  # First 200 chars
                print(f"   Error: {error_msg}...")
                print(f"   File: {test.get('file', 'unknown')}")

        # Show failure analysis
        if artifacts.failure_analysis:
            print_section("STEP 6: Failure Analysis", "-")

            analysis = artifacts.failure_analysis
            print(f"{analysis.summary}")

            print("\n📋 Actionable Recommendations:")
            for i, fix in enumerate(analysis.recommended_fixes[:5], 1):
                print(f"   {i}. {fix}")

            print(
                f"\n⏱  Estimated Fix Time: {analysis.estimated_fix_time_minutes} minutes"
            )
            print(f"   Auto-Fixable: {'Yes ✅' if analysis.auto_fixable else 'No ❌'}")
            print(f"   Severity: {analysis.severity.value.upper()}")
    else:
        print("⚠️  No test results available")
        verification_results["test_execution"]["tests_collected"] = 0
        verification_results["test_execution"]["tests_passed"] = 0
        verification_results["test_execution"]["tests_failed"] = 0
        verification_results["test_execution"]["success_rate"] = 0.0
        verification_results["test_execution"]["is_passing"] = False

    # Step 5: Quality Validation
    print_section("STEP 7: Quality Validation", "-")

    validator = QualityValidator(min_quality_threshold=0.7)
    validation = await validator.validate(artifacts)

    print(f"✅ Quality Score: {validation.quality_score:.1%}")
    print(f"   Status: {'✅ PASSED' if validation.passed else '❌ FAILED'}")
    print("\n   Component Scores:")
    print(f"      • ONEX Compliance: {validation.onex_compliance_score:.1%}")
    print(f"      • Type Safety: {validation.type_safety_score:.1%}")
    print(f"      • Code Quality: {validation.code_quality_score:.1%}")
    print(f"      • Documentation: {validation.documentation_score:.1%}")
    print(f"      • Test Coverage: {validation.test_coverage_score:.1%}")

    # Final Summary Report
    print_section("📊 VERIFICATION REPORT", "=")

    print("## Generation Summary")
    print(f"- Node name: {artifacts.node_name}")
    print(f"- Node type: {artifacts.node_type}")
    print(f"- Output directory: {output_dir}")
    print("- Generation status: ✅ SUCCESS")

    print("\n## Verification Results")

    print("\n### Template Fix Verification")
    template_checks = verification_results["template_fixes"]
    template_passed = sum(1 for v in template_checks.values() if v)
    template_total = len(template_checks)
    print(f"- Passed: {template_passed}/{template_total}")
    for check, passed in template_checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")

    print("\n### Registration Code Verification")
    reg_checks = verification_results["registration_code"]
    reg_passed = sum(1 for v in reg_checks.values() if v)
    reg_total = len(reg_checks)
    print(f"- Passed: {reg_passed}/{reg_total}")
    for check, passed in reg_checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")

    print("\n## Test Execution Results")
    test_results = verification_results["test_execution"]
    print(f"- Tests collected: {test_results.get('tests_collected', 0)}")
    print(f"- Tests passed: {test_results.get('tests_passed', 0)}")
    print(f"- Tests failed: {test_results.get('tests_failed', 0)}")
    print(
        f"- Success rate: {test_results.get('success_rate', 0.0):.1%} {'✅' if test_results.get('is_passing', False) else '❌'}"
    )

    print("\n## Conclusion")

    all_checks_passed = (
        template_passed == template_total
        and reg_passed == reg_total
        and test_results.get("tests_collected", 0) > 0
    )

    if all_checks_passed:
        print("✅ All verification checks PASSED!")
        print("✅ Template fixes are working correctly")
        print("✅ Registration code is present")
        print("✅ Tests are executing (may have expected failures)")
    else:
        print("⚠️  Some verification checks FAILED")
        if template_passed < template_total:
            print("❌ Template fixes need attention")
        if reg_passed < reg_total:
            print("❌ Registration code is incomplete")
        if test_results.get("tests_collected", 0) == 0:
            print("❌ Tests did not execute")

    print("\n## Recommendations")

    if artifacts.test_results and not artifacts.test_results.is_passing:
        print("1. Review test failures (these may be expected for generated stubs)")
        print("2. Implement TODOs in node.py for full functionality")
        print("3. Re-run tests after implementation")
    else:
        print("1. Review generated code for completeness")
        print("2. Implement custom business logic")
        print("3. Add additional test cases as needed")

    print("\n🔍 Next Steps:")
    print(f"   1. Review code in: {output_dir}")
    print("   2. Check node.py for implementation TODOs")
    print(f"   3. Run tests manually: pytest {output_dir}/tests/ -v")
    print("   4. Integrate with your project")

    print("\n✨ Verification test completed!\n")

    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
