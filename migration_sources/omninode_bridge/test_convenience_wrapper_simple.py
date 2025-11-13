#!/usr/bin/env python3
"""
Simple standalone test for convenience wrapper logic.

Tests the core logic without needing full module imports.
"""

# Mock the catalog data
CONVENIENCE_WRAPPER_CATALOG = {
    "orchestrator": {
        "class_name": "ModelServiceOrchestrator",
        "import_path": "omninode_bridge.utils.node_services",
        "standard_mixins": [
            "MixinNodeService",
            "MixinHealthCheck",
            "MixinEventBus",
            "MixinMetrics",
        ],
        "description": "Pre-composed orchestrator with standard mixins",
    },
    "reducer": {
        "class_name": "ModelServiceReducer",
        "import_path": "omninode_bridge.utils.node_services",
        "standard_mixins": [
            "MixinNodeService",
            "MixinHealthCheck",
            "MixinCaching",
            "MixinMetrics",
        ],
        "description": "Pre-composed reducer with standard mixins",
    },
}


def should_use_convenience_wrapper(contract, catalog):
    """Test the convenience wrapper detection logic."""
    node_type = contract.get("node_type", "").lower()

    # Check if convenience wrapper exists for this node type
    if node_type not in catalog:
        return False

    wrapper_info = catalog[node_type]
    standard_mixins = set(wrapper_info["standard_mixins"])

    # Get declared mixins from contract
    declared_mixins = contract.get("mixins", [])
    enabled_mixin_names = {
        m.get("name", "") for m in declared_mixins if m.get("enabled", True)
    }

    # If no mixins declared, use convenience wrapper with defaults
    if not enabled_mixin_names:
        return True

    # Check if declared mixins match standard mixins exactly
    if enabled_mixin_names == standard_mixins:
        # Also check that no custom configurations are specified
        has_custom_config = any(
            m.get("config") for m in declared_mixins if m.get("enabled", True)
        )
        if not has_custom_config:
            return True

    return False


def test_orchestrator_no_mixins():
    """Test orchestrator with no mixins specified."""
    print("\nTest 1: Orchestrator with no mixins (should use wrapper)")
    contract = {
        "name": "workflow_orchestrator",
        "node_type": "ORCHESTRATOR",
        "description": "Workflow orchestration",
        "mixins": [],
    }

    result = should_use_convenience_wrapper(contract, CONVENIENCE_WRAPPER_CATALOG)
    print(f"  Result: {result}")
    assert result is True, "Should use convenience wrapper when no mixins specified"
    print("  ✅ Passed")


def test_reducer_no_mixins():
    """Test reducer with no mixins specified."""
    print("\nTest 2: Reducer with no mixins (should use wrapper)")
    contract = {
        "name": "metrics_reducer",
        "node_type": "REDUCER",
        "description": "Metrics aggregation",
        "mixins": [],
    }

    result = should_use_convenience_wrapper(contract, CONVENIENCE_WRAPPER_CATALOG)
    print(f"  Result: {result}")
    assert result is True, "Should use convenience wrapper when no mixins specified"
    print("  ✅ Passed")


def test_orchestrator_standard_mixins():
    """Test orchestrator with standard mixins."""
    print("\nTest 3: Orchestrator with standard mixins (should use wrapper)")
    contract = {
        "name": "workflow_orchestrator",
        "node_type": "ORCHESTRATOR",
        "description": "Workflow orchestration",
        "mixins": [
            {"name": "MixinNodeService", "enabled": True},
            {"name": "MixinHealthCheck", "enabled": True},
            {"name": "MixinEventBus", "enabled": True},
            {"name": "MixinMetrics", "enabled": True},
        ],
    }

    result = should_use_convenience_wrapper(contract, CONVENIENCE_WRAPPER_CATALOG)
    print(f"  Result: {result}")
    assert result is True, "Should use convenience wrapper with standard mixins"
    print("  ✅ Passed")


def test_effect_node():
    """Test effect node (no wrapper available)."""
    print("\nTest 4: Effect node (no wrapper available)")
    contract = {
        "name": "database_effect",
        "node_type": "EFFECT",
        "description": "Database operations",
        "mixins": [],
    }

    result = should_use_convenience_wrapper(contract, CONVENIENCE_WRAPPER_CATALOG)
    print(f"  Result: {result}")
    assert result is False, "Should not use wrapper for EFFECT (not available)"
    print("  ✅ Passed")


def test_orchestrator_custom_mixins():
    """Test orchestrator with custom mixins."""
    print("\nTest 5: Orchestrator with custom mixins (should NOT use wrapper)")
    contract = {
        "name": "custom_orchestrator",
        "node_type": "ORCHESTRATOR",
        "description": "Custom orchestration",
        "mixins": [
            {"name": "MixinHealthCheck", "enabled": True},
            {"name": "MixinCaching", "enabled": True},  # Not standard for orchestrator
        ],
    }

    result = should_use_convenience_wrapper(contract, CONVENIENCE_WRAPPER_CATALOG)
    print(f"  Result: {result}")
    assert result is False, "Should not use wrapper with custom mixins"
    print("  ✅ Passed")


def test_orchestrator_with_custom_config():
    """Test orchestrator with custom config (should NOT use wrapper)."""
    print("\nTest 6: Orchestrator with custom config (should NOT use wrapper)")
    contract = {
        "name": "custom_orchestrator",
        "node_type": "ORCHESTRATOR",
        "description": "Custom orchestration",
        "mixins": [
            {"name": "MixinNodeService", "enabled": True},
            {
                "name": "MixinHealthCheck",
                "enabled": True,
                "config": {"check_interval_ms": 5000},  # Custom config
            },
            {"name": "MixinEventBus", "enabled": True},
            {"name": "MixinMetrics", "enabled": True},
        ],
    }

    result = should_use_convenience_wrapper(contract, CONVENIENCE_WRAPPER_CATALOG)
    print(f"  Result: {result}")
    assert result is False, "Should not use wrapper with custom config"
    print("  ✅ Passed")


def main():
    """Run all tests."""
    print("=" * 80)
    print("CONVENIENCE WRAPPER DETECTION LOGIC TESTS")
    print("=" * 80)

    try:
        test_orchestrator_no_mixins()
        test_reducer_no_mixins()
        test_orchestrator_standard_mixins()
        test_effect_node()
        test_orchestrator_custom_mixins()
        test_orchestrator_with_custom_config()

        print("\n" + "=" * 80)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 80)
        print("\nConvenience wrapper detection logic is working correctly:")
        print("  ✅ Uses wrapper for standard configurations")
        print("  ✅ Falls back to custom composition when needed")
        print("  ✅ Handles missing wrappers gracefully")
        print("  ✅ Respects custom configurations")
        print()

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    main()
