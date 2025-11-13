#!/usr/bin/env python3
"""
Standalone test for MixinInjector.

Tests mixin code generation without requiring full environment setup.
This script can run independently to validate the MixinInjector component.
"""

import ast
import sys
from pathlib import Path

# Direct import of mixin_injector module
injector_path = (
    Path(__file__).parent.parent
    / "src"
    / "omninode_bridge"
    / "codegen"
    / "mixin_injector.py"
)
spec = __import__("importlib.util").util.spec_from_file_location(
    "mixin_injector", injector_path
)
mixin_injector = __import__("importlib.util").util.module_from_spec(spec)
spec.loader.exec_module(mixin_injector)

MixinInjector = mixin_injector.MixinInjector


def test_basic_generation():
    """Test basic code generation."""
    print("=" * 80)
    print("TEST: Basic Node Generation")
    print("=" * 80)

    contract = {
        "name": "test_effect",
        "node_type": "EFFECT",
        "description": "Test effect node",
        "mixins": [],
    }

    injector = MixinInjector()
    node_code = injector.generate_node_file(contract)

    print("\nGenerated code (first 40 lines):")
    print("-" * 80)
    lines = node_code.split("\n")
    for i, line in enumerate(lines[:40], 1):
        print(f"{i:3d}: {line}")
    print("-" * 80)
    print(f"Total lines: {len(lines)}")

    # Validate syntax
    print("\n✓ Checking Python syntax...")
    try:
        ast.parse(node_code)
        print("  ✓ Syntax is valid")
        return True
    except SyntaxError as e:
        print(f"  ✗ Syntax error: {e}")
        return False


def test_health_check_mixin():
    """Test generation with MixinHealthCheck."""
    print("\n" + "=" * 80)
    print("TEST: Node with MixinHealthCheck")
    print("=" * 80)

    contract = {
        "name": "postgres_adapter_effect",
        "node_type": "EFFECT",
        "description": "PostgreSQL adapter with health monitoring",
        "mixins": [
            {
                "name": "MixinHealthCheck",
                "enabled": True,
                "config": {
                    "check_interval_ms": 30000,
                    "timeout_seconds": 5.0,
                },
            }
        ],
    }

    injector = MixinInjector()
    node_code = injector.generate_node_file(contract)

    # Show key sections
    print("\nGenerated code sections:")
    print("-" * 80)

    lines = node_code.split("\n")

    # Find and show imports
    print("\n[IMPORTS]")
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            print(f"  {line}")
        elif line.startswith("logger ="):
            break

    # Find and show class definition
    print("\n[CLASS DEFINITION]")
    for i, line in enumerate(lines):
        if line.startswith("class "):
            for j in range(10):
                if i + j < len(lines):
                    print(f"  {lines[i + j]}")
            break

    # Find and show __init__
    print("\n[__INIT__ METHOD]")
    for i, line in enumerate(lines):
        if "def __init__" in line:
            for j in range(15):
                if i + j < len(lines):
                    print(f"  {lines[i + j]}")
            break

    print("-" * 80)

    # Validation checks
    print("\n✓ Validation checks:")
    checks = [
        ("Python syntax", lambda: ast.parse(node_code)),
        (
            "MixinHealthCheck import",
            lambda: "from omnibase_core.mixins.mixin_health_check import MixinHealthCheck"
            in node_code,
        ),
        (
            "Class inheritance",
            lambda: "class NodePostgresAdapterEffect(NodeEffect, MixinHealthCheck):"
            in node_code,
        ),
        ("Health check config", lambda: '"check_interval_ms": 30000' in node_code),
        (
            "get_health_checks method",
            lambda: "def get_health_checks(self)" in node_code,
        ),
        (
            "_check_self_health method",
            lambda: "async def _check_self_health(self)" in node_code,
        ),
    ]

    all_passed = True
    for check_name, check_func in checks:
        try:
            result = check_func()
            if result is False:
                print(f"  ✗ {check_name}: condition returned False")
                all_passed = False
            else:
                print(f"  ✓ {check_name}")
        except Exception as e:
            print(f"  ✗ {check_name}: {e}")
            all_passed = False

    return all_passed


def test_multiple_mixins():
    """Test generation with multiple mixins."""
    print("\n" + "=" * 80)
    print("TEST: Node with Multiple Mixins")
    print("=" * 80)

    contract = {
        "name": "event_processor_effect",
        "node_type": "EFFECT",
        "description": "Event processor with monitoring",
        "mixins": [
            {"name": "MixinHealthCheck", "enabled": True, "config": {}},
            {"name": "MixinMetrics", "enabled": True, "config": {}},
            {"name": "MixinEventDrivenNode", "enabled": True, "config": {}},
        ],
    }

    injector = MixinInjector()
    node_code = injector.generate_node_file(contract)

    # Show imports
    print("\nMixin imports:")
    print("-" * 80)
    lines = node_code.split("\n")
    for line in lines:
        if "omnibase_core.mixins" in line:
            print(f"  {line}")
    print("-" * 80)

    # Validation
    print("\n✓ Validation checks:")
    checks = [
        ("Python syntax", lambda: ast.parse(node_code)),
        ("MixinHealthCheck", lambda: "MixinHealthCheck" in node_code),
        ("MixinMetrics", lambda: "MixinMetrics" in node_code),
        ("MixinEventDrivenNode", lambda: "MixinEventDrivenNode" in node_code),
        ("get_capabilities method", lambda: "def get_capabilities(self)" in node_code),
        (
            "supports_introspection method",
            lambda: "def supports_introspection(self)" in node_code,
        ),
    ]

    all_passed = True
    for check_name, check_func in checks:
        try:
            result = check_func()
            if result is False:
                print(f"  ✗ {check_name}: condition returned False")
                all_passed = False
            else:
                print(f"  ✓ {check_name}")
        except Exception as e:
            print(f"  ✗ {check_name}: {e}")
            all_passed = False

    return all_passed


def test_different_node_types():
    """Test generation for different node types."""
    print("\n" + "=" * 80)
    print("TEST: Different Node Types")
    print("=" * 80)

    node_types = [
        ("EFFECT", "NodeEffect"),
        ("COMPUTE", "NodeCompute"),
        ("REDUCER", "NodeReducer"),
        ("ORCHESTRATOR", "NodeOrchestrator"),
    ]

    injector = MixinInjector()
    all_passed = True

    for node_type, expected_base in node_types:
        contract = {
            "name": f"test_{node_type.lower()}",
            "node_type": node_type,
            "description": f"Test {node_type} node",
            "mixins": [],
        }

        node_code = injector.generate_node_file(contract)

        print(f"\n  Testing {node_type}...")

        # Check base class import
        expected_import = (
            f"from omnibase_core.nodes.node_{node_type.lower()} import {expected_base}"
        )
        if expected_import in node_code:
            print(f"    ✓ {expected_base} import present")
        else:
            print(f"    ✗ {expected_base} import missing")
            all_passed = False

        # Check syntax
        try:
            ast.parse(node_code)
            print("    ✓ Syntax valid")
        except SyntaxError as e:
            print(f"    ✗ Syntax error: {e}")
            all_passed = False

    return all_passed


def test_code_compilation():
    """Test that generated code compiles."""
    print("\n" + "=" * 80)
    print("TEST: Code Compilation")
    print("=" * 80)

    contract = {
        "name": "comprehensive_effect",
        "node_type": "EFFECT",
        "description": "Comprehensive test node",
        "mixins": [
            {
                "name": "MixinHealthCheck",
                "enabled": True,
                "config": {"check_interval_ms": 30000},
            },
            {"name": "MixinMetrics", "enabled": True, "config": {}},
        ],
    }

    injector = MixinInjector()
    node_code = injector.generate_node_file(contract)

    print("\nAttempting to compile generated code...")
    try:
        compile(node_code, "<generated>", "exec")
        print("  ✓ Code compiles successfully")
        return True
    except Exception as e:
        print(f"  ✗ Compilation error: {e}")
        return False


def save_sample_output():
    """Save a sample generated file for manual inspection."""
    print("\n" + "=" * 80)
    print("SAVING SAMPLE OUTPUT")
    print("=" * 80)

    contract = {
        "name": "sample_database_effect",
        "node_type": "EFFECT",
        "description": "Sample database effect node with health monitoring and metrics",
        "mixins": [
            {
                "name": "MixinHealthCheck",
                "enabled": True,
                "config": {
                    "check_interval_ms": 30000,
                    "timeout_seconds": 5.0,
                },
            },
            {"name": "MixinMetrics", "enabled": True, "config": {}},
        ],
    }

    injector = MixinInjector()
    node_code = injector.generate_node_file(contract)

    output_dir = (
        Path(__file__).parent.parent / "generated_nodes" / "sample_mixin_output"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "node.py"
    output_file.write_text(node_code)

    lines_count = len(node_code.split("\n"))
    print(f"\n✓ Sample output saved to: {output_file}")
    print(f"  Lines: {lines_count}")
    print(f"  Size: {len(node_code)} bytes")

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("MIXIN INJECTOR STANDALONE TESTS")
    print("=" * 80)

    tests = [
        ("Basic Generation", test_basic_generation),
        ("HealthCheck Mixin", test_health_check_mixin),
        ("Multiple Mixins", test_multiple_mixins),
        ("Different Node Types", test_different_node_types),
        ("Code Compilation", test_code_compilation),
        ("Save Sample Output", save_sample_output),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' failed with exception:")
            print(f"  {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")

    print(f"\n  {passed}/{total} tests passed")
    print("=" * 80)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
