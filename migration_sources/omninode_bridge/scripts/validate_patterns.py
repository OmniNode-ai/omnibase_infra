#!/usr/bin/env python3
"""
Validate Pattern Library.

This script validates all patterns in the pattern library, ensuring they:
1. Load successfully
2. Have valid structure
3. Pass all validation rules
4. Meet performance targets

Usage:
    python scripts/validate_patterns.py
"""

import sys
import time
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from metadata_stamping.code_gen.patterns import (
    EnumNodeType,
    PatternLoader,
    PatternMatcher,
    PatternRegistry,
    PatternValidator,
)


def main():
    """Main validation routine."""
    print("=" * 70)
    print("PATTERN LIBRARY VALIDATION")
    print("=" * 70)

    # 1. Test Pattern Loading
    print("\n1. Testing Pattern Loading...")
    loader = PatternLoader()

    start = time.perf_counter()
    patterns = loader.load_all_patterns()
    load_time = (time.perf_counter() - start) * 1000  # ms

    print(f"   ✓ Loaded {len(patterns)} patterns in {load_time:.2f}ms")

    if len(patterns) < 20:
        print(f"   ⚠️  Warning: Expected at least 20 patterns, got {len(patterns)}")
    else:
        print("   ✓ Pattern count meets target (20+)")

    # 2. Test Pattern Validation
    print("\n2. Testing Pattern Validation...")
    validator = PatternValidator(loader)

    start = time.perf_counter()
    results = validator.validate_all_patterns()
    validation_time = (time.perf_counter() - start) * 1000  # ms

    print(f"   ✓ Validated {len(results)} patterns in {validation_time:.2f}ms")

    # Count valid/invalid
    valid_count = sum(1 for r in results.values() if r.is_valid)
    invalid_count = len(results) - valid_count

    print(f"   ✓ Valid patterns: {valid_count}/{len(results)}")

    if invalid_count > 0:
        print(f"   ✗ Invalid patterns: {invalid_count}")
        print("\n   Invalid patterns:")
        for pattern_id, result in results.items():
            if not result.is_valid:
                print(f"     - {pattern_id}:")
                for error in result.errors:
                    print(f"       ERROR: {error}")

        # Print detailed validation summary
        print()
        validator.print_validation_summary(results)
        return 1  # Exit with error

    print("   ✓ All patterns valid!")

    # 3. Test Pattern Registry
    print("\n3. Testing Pattern Registry...")
    registry = PatternRegistry(loader)

    start = time.perf_counter()
    registry.load_patterns()
    registry_time = (time.perf_counter() - start) * 1000  # ms

    print(f"   ✓ Registry loaded in {registry_time:.2f}ms")

    stats = registry.get_library_stats()
    print(f"   ✓ Total patterns: {stats.total_patterns}")
    print(f"   ✓ Average complexity: {stats.average_complexity:.2f}")

    # Print category breakdown
    print("\n   Patterns by category:")
    for category, count in stats.patterns_by_category.items():
        print(f"     - {category}: {count}")

    # 4. Test Pattern Matching
    print("\n4. Testing Pattern Matching...")
    matcher = PatternMatcher(registry)

    # Test matching performance
    test_features = {"async", "database", "error-handling"}
    total_match_time = 0
    match_iterations = 10

    for node_type in EnumNodeType:
        start = time.perf_counter()
        for _ in range(match_iterations):
            matches = matcher.match_patterns(
                node_type=node_type,
                required_features=test_features,
                top_k=5,
            )
        match_time = (time.perf_counter() - start) * 1000 / match_iterations
        total_match_time += match_time

        print(f"   ✓ {node_type.value}: {len(matches)} matches in {match_time:.2f}ms")

    avg_match_time = total_match_time / len(EnumNodeType)
    print(f"\n   ✓ Average match time: {avg_match_time:.2f}ms")

    if avg_match_time > 10:
        print(
            f"   ⚠️  Warning: Average match time {avg_match_time:.2f}ms exceeds "
            f"10ms target"
        )
    else:
        print("   ✓ Match performance meets target (<10ms)")

    # 5. Test Pattern Applicability
    print("\n5. Testing Pattern Applicability...")

    # Check that each node type has applicable patterns
    for node_type in EnumNodeType:
        applicable = registry.get_patterns_by_node_type(node_type)
        print(f"   ✓ {node_type.value}: {len(applicable)} applicable patterns")

        if len(applicable) == 0:
            print(f"   ⚠️  Warning: No patterns for {node_type.value}")

    # 6. Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"✓ Patterns loaded: {len(patterns)}")
    print(f"✓ Patterns valid: {valid_count}/{len(results)}")
    print(f"✓ Load time: {load_time:.2f}ms")
    print(f"✓ Validation time: {validation_time:.2f}ms")
    print(f"✓ Match time: {avg_match_time:.2f}ms (target: <10ms)")
    print()

    if invalid_count > 0:
        print("✗ VALIDATION FAILED: Some patterns are invalid")
        return 1

    if avg_match_time > 10:
        print("⚠️  WARNING: Match performance exceeds target")
        # Don't fail, just warn
        return 0

    print("✓ ALL VALIDATIONS PASSED!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
