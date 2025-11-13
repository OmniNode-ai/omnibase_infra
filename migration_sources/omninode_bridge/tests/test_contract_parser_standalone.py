#!/usr/bin/env python3
"""
Standalone test for YAML contract parser (no dependencies on full package).

Run this directly to test the parser functionality without pytest.
"""

import sys
from pathlib import Path

# Add codegen directory directly to path to avoid package initialization
codegen_path = Path(__file__).parent.parent / "src" / "omninode_bridge" / "codegen"
sys.path.insert(0, str(codegen_path))

# Import modules directly (they'll be found in sys.path)
from yaml_contract_parser import YAMLContractParser


def test_basic_parsing():
    """Test basic parsing functionality."""
    print("\n=== Test 1: Basic Parsing ===")

    parser = YAMLContractParser()

    # Test minimal v1.0 contract
    contract_data = {
        "name": "NodeMinimalEffect",
        "version": {"major": 1, "minor": 0, "patch": 0},
        "node_type": "effect",
        "description": "Minimal effect node",
    }

    contract = parser.parse_contract(contract_data)

    assert contract.name == "NodeMinimalEffect"
    assert contract.version.major == 1
    assert contract.schema_version == "v1.0.0"
    assert len(contract.mixins) == 0
    assert contract.is_valid

    print("✅ Basic parsing: PASSED")


def test_mixin_parsing():
    """Test mixin parsing."""
    print("\n=== Test 2: Mixin Parsing ===")

    parser = YAMLContractParser()

    contract_data = {
        "schema_version": "v2.0.0",
        "name": "NodeWithMixinsEffect",  # Fixed: must end with Effect/Compute/etc
        "version": {"major": 1, "minor": 0, "patch": 0},
        "node_type": "effect",
        "description": "Node with mixins",
        "mixins": [
            {
                "name": "MixinHealthCheck",
                "enabled": True,
                "config": {"check_interval_ms": 30000},
            },
            {"name": "MixinMetrics", "enabled": True},
        ],
    }

    contract = parser.parse_contract(contract_data)

    assert len(contract.mixins) == 2
    assert contract.has_mixin("MixinHealthCheck")
    assert contract.has_mixin("MixinMetrics")
    assert contract.is_v2

    print(f"  - Parsed {len(contract.mixins)} mixins")
    print(f"  - Mixin names: {contract.get_mixin_names()}")
    print(f"  - Valid: {contract.is_valid}")

    if contract.is_valid:
        print("✅ Mixin parsing: PASSED")
    else:
        print(f"❌ Mixin parsing: FAILED - {contract.validation_errors}")


def test_mixin_registry():
    """Test mixin registry."""
    print("\n=== Test 3: Mixin Registry ===")

    parser = YAMLContractParser()

    available_mixins = parser.get_available_mixins()
    print(f"  - Available mixins: {len(available_mixins)}")

    # Check for key mixins
    key_mixins = [
        "MixinHealthCheck",
        "MixinMetrics",
        "MixinEventDrivenNode",
        "MixinCaching",
    ]

    for mixin in key_mixins:
        assert mixin in available_mixins, f"Missing mixin: {mixin}"
        info = parser.get_mixin_info(mixin)
        print(f"  - {mixin}: {info['import_path']}")

    print("✅ Mixin registry: PASSED")


def test_mixin_dependencies():
    """Test mixin dependency checking."""
    print("\n=== Test 4: Mixin Dependencies ===")

    parser = YAMLContractParser()

    # Test satisfied dependencies
    contract_data = {
        "schema_version": "v2.0.0",
        "name": "NodeSatisfiedDepsEffect",  # Fixed: must end with Effect/Compute/etc
        "version": {"major": 1, "minor": 0, "patch": 0},
        "node_type": "effect",
        "description": "Node with satisfied dependencies",
        "mixins": [
            {"name": "MixinEventHandler", "enabled": True},
            {"name": "MixinNodeLifecycle", "enabled": True},
            {"name": "MixinIntrospectionPublisher", "enabled": True},
            {
                "name": "MixinEventDrivenNode",
                "enabled": True,
                # Add required config
                "config": {
                    "kafka_bootstrap_servers": "localhost:9092",
                    "consumer_group": "test-group",
                    "topics": ["test-topic"],
                },
            },
        ],
    }

    contract = parser.parse_contract(contract_data)
    assert contract.is_valid, f"Should be valid but got: {contract.validation_errors}"

    print("  - Satisfied dependencies: ✓")

    # Test unsatisfied dependencies
    contract_data_bad = {
        "schema_version": "v2.0.0",
        "name": "NodeUnsatisfiedDepsEffect",  # Fixed: must end with Effect/Compute/etc
        "version": {"major": 1, "minor": 0, "patch": 0},
        "node_type": "effect",
        "description": "Node with unsatisfied dependencies",
        "mixins": [
            {
                "name": "MixinEventDrivenNode",
                "enabled": True,
                "config": {  # Add required config
                    "kafka_bootstrap_servers": "localhost:9092",
                    "consumer_group": "test-group",
                    "topics": ["test-topic"],
                },
            },
        ],
    }

    contract = parser.parse_contract(contract_data_bad)
    assert not contract.is_valid, "Should have dependency errors"
    print(
        f"  - Unsatisfied dependencies detected: {len(contract.validation_errors)} errors"
    )

    print("✅ Mixin dependencies: PASSED")


def test_advanced_features():
    """Test advanced features parsing."""
    print("\n=== Test 5: Advanced Features ===")

    parser = YAMLContractParser()

    contract_data = {
        "schema_version": "v2.0.0",
        "name": "NodeWithFeaturesEffect",  # Fixed: must end with Effect/Compute/etc
        "version": {"major": 1, "minor": 0, "patch": 0},
        "node_type": "effect",
        "description": "Node with advanced features",
        "advanced_features": {
            "circuit_breaker": {"enabled": True, "failure_threshold": 5},
            "retry_policy": {"enabled": True, "max_attempts": 3},
        },
    }

    contract = parser.parse_contract(contract_data)

    assert contract.advanced_features is not None
    assert contract.advanced_features.circuit_breaker is not None
    assert contract.advanced_features.retry_policy is not None
    assert contract.advanced_features.circuit_breaker.failure_threshold == 5
    assert contract.is_valid

    print("  - Circuit breaker: ✓")
    print("  - Retry policy: ✓")
    print("✅ Advanced features: PASSED")


def test_example_files():
    """Test parsing example files."""
    print("\n=== Test 6: Example Files ===")

    parser = YAMLContractParser()

    examples_dir = (
        Path(__file__).parent.parent
        / "src"
        / "omninode_bridge"
        / "codegen"
        / "schemas"
        / "examples"
    )

    # Test valid_minimal_v1.yaml
    minimal_v1 = examples_dir / "valid_minimal_v1.yaml"
    if minimal_v1.exists():
        contract = parser.parse_contract_file(minimal_v1)
        assert (
            contract.is_valid
        ), f"valid_minimal_v1.yaml failed: {contract.validation_errors}"
        print(f"  - valid_minimal_v1.yaml: ✓ ({contract.name})")

    # Test valid_with_mixins_v2.yaml
    with_mixins = examples_dir / "valid_with_mixins_v2.yaml"
    if with_mixins.exists():
        contract = parser.parse_contract_file(with_mixins)
        assert (
            contract.is_valid
        ), f"valid_with_mixins_v2.yaml failed: {contract.validation_errors}"
        print(
            f"  - valid_with_mixins_v2.yaml: ✓ ({contract.name}, {len(contract.mixins)} mixins)"
        )

    # Test valid_complete_v2.yaml
    complete_v2 = examples_dir / "valid_complete_v2.yaml"
    if complete_v2.exists():
        contract = parser.parse_contract_file(complete_v2)
        assert (
            contract.is_valid
        ), f"valid_complete_v2.yaml failed: {contract.validation_errors}"
        print(f"  - valid_complete_v2.yaml: ✓ ({contract.name})")

    # Test invalid files (should fail validation)
    invalid_mixin = examples_dir / "invalid_mixin_name.yaml"
    if invalid_mixin.exists():
        contract = parser.parse_contract_file(invalid_mixin)
        assert not contract.is_valid, "invalid_mixin_name.yaml should have errors"
        print("  - invalid_mixin_name.yaml: ✓ (correctly detected errors)")

    print("✅ Example files: PASSED")


def test_validation_errors():
    """Test validation error handling."""
    print("\n=== Test 7: Validation Errors ===")

    parser = YAMLContractParser()

    # Invalid mixin name
    contract_data = {
        "schema_version": "v2.0.0",
        "name": "NodeInvalidEffect",  # Fixed: must end with Effect/Compute/etc
        "version": {"major": 1, "minor": 0, "patch": 0},
        "node_type": "effect",
        "description": "Invalid node",
        "mixins": [
            {"name": "InvalidName", "enabled": True},
        ],
    }

    contract = parser.parse_contract(contract_data)
    assert not contract.is_valid
    assert len(contract.validation_errors) > 0
    print(f"  - Invalid mixin name: ✓ ({len(contract.validation_errors)} errors)")

    # Unknown mixin
    contract_data["mixins"] = [{"name": "MixinNonExistent", "enabled": True}]
    contract = parser.parse_contract(contract_data)
    assert not contract.is_valid
    print(f"  - Unknown mixin: ✓ ({len(contract.validation_errors)} errors)")

    print("✅ Validation errors: PASSED")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("YAML Contract Parser Standalone Tests")
    print("=" * 60)

    try:
        test_basic_parsing()
        test_mixin_parsing()
        test_mixin_registry()
        test_mixin_dependencies()
        test_advanced_features()
        test_example_files()
        test_validation_errors()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
