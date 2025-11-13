#!/usr/bin/env python3
"""
Phase 3 Performance Benchmark Script.

Tests performance of Phase 3 components:
- TemplateSelector: <5ms target
- PatternLibrary: <10ms target
- RequirementsAnalyzer + MixinRecommender: <200ms target
- EnhancedContextBuilder: <50ms target
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def benchmark_template_selector():
    """Benchmark TemplateSelector performance."""
    from omninode_bridge.codegen.template_selector import TemplateSelector

    @dataclass
    class MockContract:
        node_type: str = "effect"
        operations: list = field(default_factory=lambda: ["read", "write"])
        dependencies: dict = field(default_factory=dict)
        io_operations: list = field(default_factory=list)

    selector = TemplateSelector()
    contract = MockContract()

    # Warm up
    for _ in range(5):
        selector.select_template(contract, "effect")

    # Measure
    times = []
    for _ in range(100):
        result = selector.select_template(contract, "effect")
        times.append(result.selection_time_ms)

    avg_time = sum(times) / len(times)
    p99_time = sorted(times)[98]

    print("=" * 60)
    print("TemplateSelector Performance")
    print("=" * 60)
    print(f"  Average:    {avg_time:.2f}ms")
    print(f"  P99:        {p99_time:.2f}ms")
    print("  Target:     <5ms")
    print(f"  Status:     {'PASS' if p99_time < 5 else 'FAIL'}")
    print()

    return p99_time < 5


def benchmark_pattern_library():
    """Benchmark PatternLibrary performance."""
    from omninode_bridge.codegen.pattern_library import ProductionPatternLibrary

    library = ProductionPatternLibrary()

    # Measure
    times = []
    for _ in range(100):
        start = time.perf_counter()
        matches = library.find_matching_patterns(
            operation_type="database",
            features={"health_checks", "metrics", "lifecycle"},
            node_type="effect",
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

    avg_time = sum(times) / len(times)
    p99_time = sorted(times)[98]

    print("=" * 60)
    print("PatternLibrary Performance")
    print("=" * 60)
    print(f"  Average:    {avg_time:.2f}ms")
    print(f"  P99:        {p99_time:.2f}ms")
    print("  Target:     <10ms")
    print(f"  Status:     {'PASS' if p99_time < 10 else 'FAIL'}")
    print(f"  Patterns:   {len(library.get_all_patterns())} available")
    print()

    return p99_time < 10


def benchmark_context_builder():
    """Benchmark EnhancedContextBuilder performance."""
    from omninode_bridge.codegen.context_builder import EnhancedContextBuilder
    from omninode_bridge.codegen.models_contract import EnumTemplateVariant
    from omninode_bridge.codegen.template_selector import ModelTemplateSelection

    @dataclass
    class MockOperation:
        name: str = "test_operation"
        description: str = "Test operation"

    @dataclass
    class MockContract:
        node_type: str = "effect"
        business_description: str = "Test service"
        domain: str = "database"

    builder = EnhancedContextBuilder()
    contract = MockContract()
    operation = MockOperation()
    template_selection = ModelTemplateSelection(
        variant=EnumTemplateVariant.STANDARD,
        confidence=0.9,
        patterns=["lifecycle", "health_checks"],
    )

    # Measure
    times = []
    for _ in range(50):
        start = time.perf_counter()
        context = builder.build_context(
            requirements=contract,
            operation=operation,
            template_selection=template_selection,
            mixin_selection=[],
            pattern_matches=[],
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

    avg_time = sum(times) / len(times)
    p99_time = sorted(times)[int(len(times) * 0.98)]

    print("=" * 60)
    print("EnhancedContextBuilder Performance")
    print("=" * 60)
    print(f"  Average:    {avg_time:.2f}ms")
    print(f"  P99:        {p99_time:.2f}ms")
    print("  Target:     <50ms")
    print(f"  Status:     {'PASS' if p99_time < 50 else 'FAIL'}")
    print()

    return p99_time < 50


def main():
    """Run all benchmarks."""
    print("\n")
    print("*" * 60)
    print("Phase 3 Performance Benchmark")
    print("*" * 60)
    print()

    results = []

    # Run benchmarks
    try:
        results.append(("TemplateSelector", benchmark_template_selector()))
    except Exception as e:
        print(f"TemplateSelector benchmark failed: {e}")
        results.append(("TemplateSelector", False))

    try:
        results.append(("PatternLibrary", benchmark_pattern_library()))
    except Exception as e:
        print(f"PatternLibrary benchmark failed: {e}")
        results.append(("PatternLibrary", False))

    try:
        results.append(("EnhancedContextBuilder", benchmark_context_builder()))
    except Exception as e:
        print(f"EnhancedContextBuilder benchmark failed: {e}")
        results.append(("EnhancedContextBuilder", False))

    # Summary
    print("=" * 60)
    print("Benchmark Summary")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:30s} {status}")
    print()

    # Overall result
    all_passed = all(passed for _, passed in results)
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print()

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
