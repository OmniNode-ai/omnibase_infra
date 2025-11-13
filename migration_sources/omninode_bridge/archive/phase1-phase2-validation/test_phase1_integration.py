#!/usr/bin/env python3
"""
Integration test for Phase 1: MixinSelector + TemplateEngine integration.

Validates that the template engine correctly uses mixin_selector to:
1. Detect when to use convenience wrappers (80% path)
2. Fall back to custom mixin composition (20% path)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from omninode_bridge.codegen.mixin_selector import MixinSelector
from omninode_bridge.codegen.template_engine import TemplateEngine


def test_mixin_selector_integration():
    """Test MixinSelector component directly."""
    print("=" * 80)
    print("TEST 1: MixinSelector Component")
    print("=" * 80)

    selector = MixinSelector()

    # Test 1: Orchestrator with standard requirements (should use wrapper)
    print("\n1️⃣ Test Orchestrator with standard requirements:")
    requirements = {
        "node_type": "orchestrator",
        "features": ["health_check", "metrics", "event_bus"],
    }

    result = selector.select_base_class("orchestrator", requirements)
    print(f"   Input: {requirements}")
    print(f"   Result: {result}")

    if result.use_convenience_wrapper:
        print(f"   ✅ PASS: Uses convenience wrapper '{result.base_class_name}'")
        assert result.base_class_name == "ModelServiceOrchestrator"
    else:
        print("   ❌ FAIL: Should use convenience wrapper")
        return False

    # Test 2: Reducer with standard requirements (should use wrapper)
    print("\n2️⃣ Test Reducer with standard requirements:")
    requirements = {
        "node_type": "reducer",
        "features": ["health_check", "metrics"],
    }

    result = selector.select_base_class("reducer", requirements)
    print(f"   Input: {requirements}")
    print(f"   Result: {result}")

    if result.use_convenience_wrapper:
        print(f"   ✅ PASS: Uses convenience wrapper '{result.base_class_name}'")
        assert result.base_class_name == "ModelServiceReducer"
    else:
        print("   ❌ FAIL: Should use convenience wrapper")
        return False

    # Test 3: Orchestrator with custom mixins (should NOT use wrapper)
    print("\n3️⃣ Test Orchestrator with custom mixins:")
    requirements = {
        "node_type": "orchestrator",
        "features": ["custom_auth", "custom_validation"],
        "mixins": ["MixinCustomAuth", "MixinCustomValidation"],
    }

    result = selector.select_base_class("orchestrator", requirements)
    print(f"   Input: {requirements}")
    print(f"   Result: {result}")

    if not result.use_convenience_wrapper:
        print(
            f"   ✅ PASS: Uses custom composition with base '{result.base_class_name}'"
        )
        assert result.base_class_name == "NodeOrchestrator"
        assert len(result.mixin_list) > 0
    else:
        print("   ❌ FAIL: Should use custom composition")
        return False

    print("\n" + "=" * 80)
    print("✅ All MixinSelector tests PASSED!")
    print("=" * 80)
    return True


def test_template_engine_integration():
    """Test TemplateEngine integration with MixinSelector."""
    print("\n" + "=" * 80)
    print("TEST 2: TemplateEngine Integration")
    print("=" * 80)

    template_engine = TemplateEngine(enable_inline_templates=True)

    # Test that _select_base_class method exists
    print("\n1️⃣ Check TemplateEngine has _select_base_class method:")
    if hasattr(template_engine, "_select_base_class"):
        print("   ✅ PASS: Method exists")
    else:
        print("   ❌ FAIL: Method not found")
        return False

    # Test convenience wrapper selection
    print("\n2️⃣ Test convenience wrapper selection:")
    requirements = {
        "node_type": "orchestrator",
        "features": ["health_check", "metrics"],
    }

    result = template_engine._select_base_class("orchestrator", requirements)
    print(f"   Input: {requirements}")
    print(f"   Result: {result}")

    if result["use_convenience_wrapper"]:
        print(f"   ✅ PASS: Selected wrapper '{result['base_class_name']}'")
        assert result["base_class_name"] == "ModelServiceOrchestrator"
    else:
        print("   ❌ FAIL: Should select convenience wrapper")
        return False

    # Test custom composition selection
    print("\n3️⃣ Test custom composition selection:")
    requirements = {
        "node_type": "orchestrator",
        "mixins": ["MixinRetry", "MixinCircuitBreaker"],
    }

    result = template_engine._select_base_class("orchestrator", requirements)
    print(f"   Input: {requirements}")
    print(f"   Result: {result}")

    if not result["use_convenience_wrapper"]:
        print("   ✅ PASS: Selected custom composition")
        assert result["base_class_name"] == "NodeOrchestrator"
        assert len(result["mixin_list"]) > 0
    else:
        print("   ❌ FAIL: Should select custom composition")
        return False

    print("\n" + "=" * 80)
    print("✅ All TemplateEngine integration tests PASSED!")
    print("=" * 80)
    return True


def test_convenience_wrapper_imports():
    """Test that convenience wrapper classes are importable."""
    print("\n" + "=" * 80)
    print("TEST 3: Convenience Wrapper Imports")
    print("=" * 80)

    try:
        print("\n1️⃣ Import ModelServiceOrchestrator:")
        from omninode_bridge.utils.node_services import ModelServiceOrchestrator

        print(f"   ✅ PASS: Imported {ModelServiceOrchestrator.__name__}")

        print("\n2️⃣ Import ModelServiceReducer:")
        from omninode_bridge.utils.node_services import ModelServiceReducer

        print(f"   ✅ PASS: Imported {ModelServiceReducer.__name__}")

        print("\n3️⃣ Check base classes:")
        print(
            f"   ModelServiceOrchestrator bases: {ModelServiceOrchestrator.__bases__}"
        )
        print(f"   ModelServiceReducer bases: {ModelServiceReducer.__bases__}")

        print("\n" + "=" * 80)
        print("✅ All import tests PASSED!")
        print("=" * 80)
        return True

    except ImportError as e:
        print(f"   ❌ FAIL: Import error - {e}")
        return False


def main():
    """Run all Phase 1 integration tests."""
    print("\n" + "=" * 80)
    print("🚀 PHASE 1 INTEGRATION TESTS")
    print("=" * 80)
    print("\nValidating:")
    print("  1. MixinSelector component (80/20 split)")
    print("  2. TemplateEngine integration")
    print("  3. Convenience wrapper imports")
    print()

    results = []

    # Test 1: MixinSelector
    try:
        results.append(("MixinSelector", test_mixin_selector_integration()))
    except Exception as e:
        print(f"\n❌ MixinSelector test failed with error: {e}")
        import traceback

        traceback.print_exc()
        results.append(("MixinSelector", False))

    # Test 2: TemplateEngine
    try:
        results.append(("TemplateEngine", test_template_engine_integration()))
    except Exception as e:
        print(f"\n❌ TemplateEngine test failed with error: {e}")
        import traceback

        traceback.print_exc()
        results.append(("TemplateEngine", False))

    # Test 3: Imports
    try:
        results.append(("Imports", test_convenience_wrapper_imports()))
    except Exception as e:
        print(f"\n❌ Import test failed with error: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Imports", False))

    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n" + "=" * 80)
        print("🎉 ALL PHASE 1 INTEGRATION TESTS PASSED!")
        print("=" * 80)
        print("\n✅ Phase 1 Implementation Validated:")
        print("  • MixinSelector: 80% convenience wrapper, 20% custom composition")
        print("  • TemplateEngine: Integrated with intelligent base class selection")
        print(
            "  • ConvenienceWrappers: ModelServiceOrchestrator & ModelServiceReducer available"
        )
        print("\n🚀 Ready to proceed with real node generation!")
        return 0
    else:
        print("\n" + "=" * 80)
        print("❌ SOME TESTS FAILED")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    exit(main())
