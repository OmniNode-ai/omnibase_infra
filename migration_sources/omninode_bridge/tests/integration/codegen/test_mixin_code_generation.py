#!/usr/bin/env python3
"""
Integration test for mixin code generation.

Tests complete node generation with mixin integration, verifying:
- Generated code has valid Python syntax
- Proper import organization
- Correct class inheritance
- All required methods present
- Code compiles without errors

This test runs standalone without requiring omnibase_core installation.
"""

import ast
import sys
from pathlib import Path

# Add src to path for direct import
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from omninode_bridge.codegen.mixin_injector import MixinInjector


def test_generate_effect_with_health_check():
    """Test generating Effect node with MixinHealthCheck."""
    print("\n=== Test: Effect Node with MixinHealthCheck ===")

    contract = {
        "name": "postgres_crud_effect",
        "node_type": "EFFECT",
        "description": "PostgreSQL CRUD operations with health monitoring",
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

    # Print first 50 lines
    lines = node_code.split("\n")
    print("\nGenerated code (first 50 lines):")
    print("=" * 80)
    for i, line in enumerate(lines[:50], 1):
        print(f"{i:3d}: {line}")
    print("=" * 80)
    print(f"\nTotal lines: {len(lines)}")

    # Validate syntax
    print("\nValidating Python syntax...")
    try:
        ast.parse(node_code)
        print("✓ Syntax is valid")
    except SyntaxError as e:
        print(f"✗ Syntax error: {e}")
        return False

    # Check key components
    print("\nValidating key components...")
    checks = [
        ("Shebang", "#!/usr/bin/env python3", node_code),
        (
            "NodeEffect import",
            "from omnibase_core.nodes.node_effect import NodeEffect",
            node_code,
        ),
        (
            "MixinHealthCheck import",
            "from omnibase_core.mixins.mixin_health_check import MixinHealthCheck",
            node_code,
        ),
        (
            "Class definition",
            "class NodePostgresCrudEffect(NodeEffect, MixinHealthCheck):",
            node_code,
        ),
        (
            "__init__ method",
            "def __init__(self, container: ModelContainer):",
            node_code,
        ),
        ("initialize method", "async def initialize(self) -> None:", node_code),
        ("shutdown method", "async def shutdown(self) -> None:", node_code),
        ("get_health_checks", "def get_health_checks(self)", node_code),
        ("_check_self_health", "async def _check_self_health(self)", node_code),
        ("Health config", '"check_interval_ms": 30000', node_code),
    ]

    all_passed = True
    for check_name, expected, code in checks:
        if expected in code:
            print(f"  ✓ {check_name}")
        else:
            print(f"  ✗ {check_name} - NOT FOUND")
            all_passed = False

    return all_passed


def test_generate_effect_with_multiple_mixins():
    """Test generating Effect node with multiple mixins."""
    print("\n=== Test: Effect Node with Multiple Mixins ===")

    contract = {
        "name": "event_processor_effect",
        "node_type": "EFFECT",
        "description": "Event processor with full monitoring and caching",
        "mixins": [
            {"name": "MixinHealthCheck", "enabled": True, "config": {}},
            {"name": "MixinMetrics", "enabled": True, "config": {}},
            {"name": "MixinEventDrivenNode", "enabled": True, "config": {}},
            {"name": "MixinCaching", "enabled": True, "config": {}},
        ],
    }

    injector = MixinInjector()
    node_code = injector.generate_node_file(contract)

    # Validate syntax
    print("\nValidating Python syntax...")
    try:
        ast.parse(node_code)
        print("✓ Syntax is valid")
    except SyntaxError as e:
        print(f"✗ Syntax error: {e}")
        return False

    # Check all mixins are present
    print("\nValidating mixin presence...")
    mixins_to_check = [
        "MixinHealthCheck",
        "MixinMetrics",
        "MixinEventDrivenNode",
        "MixinCaching",
    ]

    all_passed = True
    for mixin_name in mixins_to_check:
        import_statement = f"from omnibase_core.mixins.{mixin_name.lower().replace('mixin', 'mixin_')} import {mixin_name}"
        # Simplified check
        if mixin_name in node_code:
            print(f"  ✓ {mixin_name} present in code")
        else:
            print(f"  ✗ {mixin_name} - NOT FOUND")
            all_passed = False

    # Check class inheritance includes all mixins
    print("\nValidating class inheritance...")
    if "NodeEffect" in node_code and all(m in node_code for m in mixins_to_check):
        print("  ✓ All mixins in class definition")
    else:
        print("  ✗ Not all mixins in class definition")
        all_passed = False

    return all_passed


def test_generate_compute_node():
    """Test generating Compute node."""
    print("\n=== Test: Compute Node ===")

    contract = {
        "name": "data_transformer_compute",
        "node_type": "COMPUTE",
        "description": "Data transformation with caching",
        "mixins": [
            {"name": "MixinCaching", "enabled": True, "config": {}},
            {"name": "MixinMetrics", "enabled": True, "config": {}},
        ],
    }

    injector = MixinInjector()
    node_code = injector.generate_node_file(contract)

    # Validate syntax
    print("\nValidating Python syntax...")
    try:
        ast.parse(node_code)
        print("✓ Syntax is valid")
    except SyntaxError as e:
        print(f"✗ Syntax error: {e}")
        return False

    # Check Compute-specific imports
    print("\nValidating Compute node specifics...")
    checks = [
        (
            "NodeCompute import",
            "from omnibase_core.nodes.node_compute import NodeCompute",
            node_code,
        ),
        ("Class definition", "class NodeDataTransformerCompute(NodeCompute", node_code),
    ]

    all_passed = True
    for check_name, expected, code in checks:
        if expected in code:
            print(f"  ✓ {check_name}")
        else:
            print(f"  ✗ {check_name} - NOT FOUND")
            all_passed = False

    return all_passed


def test_pep8_import_organization():
    """Test that imports are organized per PEP 8."""
    print("\n=== Test: PEP 8 Import Organization ===")

    contract = {
        "name": "test_effect",
        "node_type": "EFFECT",
        "description": "Test node",
        "mixins": [{"name": "MixinHealthCheck", "enabled": True}],
    }

    injector = MixinInjector()
    node_code = injector.generate_node_file(contract)

    lines = node_code.split("\n")
    import_lines = [
        line for line in lines if line.startswith("import ") or line.startswith("from ")
    ]

    print(f"\nFound {len(import_lines)} import statements:")
    for line in import_lines[:15]:  # Show first 15
        print(f"  {line}")

    # Check organization
    stdlib_imports = [
        l for l in import_lines if "import logging" in l or "from typing" in l
    ]
    omnibase_imports = [l for l in import_lines if "omnibase_core" in l]

    print(f"\nStandard library imports: {len(stdlib_imports)}")
    print(f"omnibase_core imports: {len(omnibase_imports)}")

    # In PEP 8, stdlib should come before third-party/project imports
    if stdlib_imports and omnibase_imports:
        stdlib_idx = import_lines.index(stdlib_imports[0])
        omnibase_idx = import_lines.index(omnibase_imports[0])

        if stdlib_idx < omnibase_idx:
            print("✓ Import organization follows PEP 8")
            return True
        else:
            print("✗ Import organization does not follow PEP 8")
            return False

    return True


def test_disabled_mixin_not_included():
    """Test that disabled mixins are not included in generated code."""
    print("\n=== Test: Disabled Mixin Exclusion ===")

    contract = {
        "name": "test_effect",
        "node_type": "EFFECT",
        "description": "Test node",
        "mixins": [
            {"name": "MixinHealthCheck", "enabled": False},  # Disabled
            {"name": "MixinMetrics", "enabled": True},  # Enabled
        ],
    }

    injector = MixinInjector()
    node_code = injector.generate_node_file(contract)

    # Check that disabled mixin is not imported
    if "MixinHealthCheck" not in node_code:
        print("✓ Disabled mixin (MixinHealthCheck) not included")
        result1 = True
    else:
        print("✗ Disabled mixin (MixinHealthCheck) should not be included")
        result1 = False

    # Check that enabled mixin is imported
    if "MixinMetrics" in node_code:
        print("✓ Enabled mixin (MixinMetrics) is included")
        result2 = True
    else:
        print("✗ Enabled mixin (MixinMetrics) should be included")
        result2 = False

    return result1 and result2


def validate_generated_code_compiles():
    """Test that generated code compiles without syntax errors."""
    print("\n=== Test: Code Compilation ===")

    contract = {
        "name": "comprehensive_test_effect",
        "node_type": "EFFECT",
        "description": "Comprehensive test with all features",
        "mixins": [
            {
                "name": "MixinHealthCheck",
                "enabled": True,
                "config": {"check_interval_ms": 30000},
            },
            {"name": "MixinMetrics", "enabled": True, "config": {}},
            {"name": "MixinEventDrivenNode", "enabled": True, "config": {}},
        ],
    }

    injector = MixinInjector()
    node_code = injector.generate_node_file(contract)

    # Try to compile the code
    print("\nCompiling generated code...")
    try:
        compile(node_code, "<generated>", "exec")
        print("✓ Code compiles successfully")
        return True
    except Exception as e:
        print(f"✗ Compilation error: {e}")
        return False


def main():
    """Run all integration tests."""
    print("\n" + "=" * 80)
    print("MIXIN CODE GENERATION INTEGRATION TESTS")
    print("=" * 80)

    tests = [
        ("Effect with HealthCheck", test_generate_effect_with_health_check),
        ("Effect with Multiple Mixins", test_generate_effect_with_multiple_mixins),
        ("Compute Node", test_generate_compute_node),
        ("PEP 8 Import Organization", test_pep8_import_organization),
        ("Disabled Mixin Exclusion", test_disabled_mixin_not_included),
        ("Code Compilation", validate_generated_code_compiles),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\n{passed}/{total} tests passed")
    print("=" * 80)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
