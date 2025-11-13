"""
MixinSelector Usage Examples

This file demonstrates common usage patterns for the MixinSelector class
in code generation workflows.
"""

from .mixin_selector import MixinSelector, select_base_class_simple


def example_1_standard_nodes():
    """
    Example 1: Standard Nodes (80% Path - Convenience Wrappers)

    For most nodes with standard capabilities, the selector returns
    convenience wrappers (ModelService*) that include pre-composed mixins.
    """
    print("=" * 80)
    print("Example 1: Standard Nodes (80% Path)")
    print("=" * 80)
    print()

    selector = MixinSelector()

    # Effect node (database adapter, API client, file I/O)
    result = selector.select_base_class("effect", {})
    print(f"Effect Node: {result}")
    print("  → ModelServiceEffect includes:")
    print("     - MixinNodeService (persistent service mode)")
    print("     - MixinHealthCheck (health monitoring)")
    print("     - MixinEventBus (event publishing)")
    print("     - MixinMetrics (performance tracking)")
    print()

    # Compute node (data transformation, ML inference)
    result = selector.select_base_class("compute", {})
    print(f"Compute Node: {result}")
    print("  → ModelServiceCompute includes:")
    print("     - MixinNodeService (persistent service mode)")
    print("     - MixinHealthCheck (health monitoring)")
    print("     - MixinCaching (result caching)")
    print("     - MixinMetrics (performance tracking)")
    print()

    # Reducer node (metrics aggregation, log analysis)
    result = selector.select_base_class("reducer", {})
    print(f"Reducer Node: {result}")
    print("  → ModelServiceReducer includes:")
    print("     - MixinNodeService (persistent service mode)")
    print("     - MixinHealthCheck (health monitoring)")
    print("     - MixinCaching (aggregation caching)")
    print("     - MixinMetrics (performance tracking)")
    print()

    # Orchestrator node (workflow coordination)
    result = selector.select_base_class("orchestrator", {})
    print(f"Orchestrator Node: {result}")
    print("  → ModelServiceOrchestrator includes:")
    print("     - MixinNodeService (persistent service mode)")
    print("     - MixinHealthCheck (health monitoring)")
    print("     - MixinEventBus (event coordination)")
    print("     - MixinMetrics (performance tracking)")
    print()


def example_2_fault_tolerant_api_client():
    """
    Example 2: Fault-Tolerant API Client (20% Path - Custom Composition)

    For API clients needing retry and circuit breaker patterns,
    the selector returns custom mixin composition.
    """
    print("=" * 80)
    print("Example 2: Fault-Tolerant API Client (20% Path)")
    print("=" * 80)
    print()

    selector = MixinSelector()

    requirements = {
        "features": ["custom_mixins", "needs_retry", "needs_circuit_breaker"],
        "integrations": ["api"],
    }

    result = selector.select_base_class("effect", requirements)
    print(f"Fault-Tolerant API Client: {result}")
    print()
    print("Mixin Order:")
    for i, mixin in enumerate(result, 1):
        print(f"  {i}. {mixin}")
    print()
    print("Key Features:")
    print("  - MixinRetry: Exponential backoff for transient failures")
    print("  - MixinCircuitBreaker: Prevent cascading failures")
    print("  - MixinHealthCheck: API endpoint health monitoring")
    print("  - MixinMetrics: Request latency, success rate tracking")
    print()


def example_3_high_throughput_stream_processor():
    """
    Example 3: High-Throughput Stream Processor (20% Path)

    For high-throughput data processing, the selector omits caching
    to minimize overhead.
    """
    print("=" * 80)
    print("Example 3: High-Throughput Stream Processor (20% Path)")
    print("=" * 80)
    print()

    selector = MixinSelector()

    requirements = {
        "features": ["custom_mixins"],
        "performance": {"high_throughput": True},
    }

    result = selector.select_base_class("compute", requirements)
    print(f"High-Throughput Processor: {result}")
    print()
    print("Mixin Order:")
    for i, mixin in enumerate(result, 1):
        print(f"  {i}. {mixin}")
    print()
    print("Optimizations:")
    print("  - No MixinCaching (overhead not worth it)")
    print("  - MixinHealthCheck (monitoring only)")
    print("  - MixinMetrics (lightweight tracking)")
    print()


def example_4_secure_data_processor():
    """
    Example 4: Secure Data Processor (20% Path)

    For processing sensitive data, the selector includes validation,
    security, and sensitive field redaction.
    """
    print("=" * 80)
    print("Example 4: Secure Data Processor (20% Path)")
    print("=" * 80)
    print()

    selector = MixinSelector()

    requirements = {
        "features": ["custom_mixins", "needs_validation", "needs_security"],
        "security": {"sensitive_data": True},
    }

    result = selector.select_base_class("compute", requirements)
    print(f"Secure Data Processor: {result}")
    print()
    print("Mixin Order:")
    for i, mixin in enumerate(result, 1):
        print(f"  {i}. {mixin}")
    print()
    print("Security Features:")
    print("  1. MixinValidation: Validate FIRST (fail-fast)")
    print("  2. MixinSecurity: Secure AFTER validation")
    print("  3. MixinSensitiveFieldRedaction: Redact logs/serialization")
    print("  4. MixinHealthCheck: Monitor security components")
    print()
    print("Note: Validation MUST come before Security in MRO!")
    print()


def example_5_decision_logging():
    """
    Example 5: Decision Logging for Debugging

    The selector logs all decisions for debugging and analysis.
    """
    print("=" * 80)
    print("Example 5: Decision Logging")
    print("=" * 80)
    print()

    selector = MixinSelector()

    # Make several decisions
    selector.select_base_class("effect", {})
    selector.select_base_class("compute", {"features": ["custom_mixins"]})
    selector.select_base_class(
        "effect", {"features": ["needs_retry", "needs_circuit_breaker"]}
    )

    # Get decision log
    log = selector.get_decision_log()

    print(f"Total Decisions: {len(log)}")
    print()

    for i, decision in enumerate(log, 1):
        print(f"Decision {i}:")
        print(f"  Path: {decision['path']}")
        print(f"  Result: {decision['result']}")
        print(f"  Reason: {decision['reason']}")
        print()


def example_6_convenience_function():
    """
    Example 6: Simplified Convenience Function

    For simple use cases, use the select_base_class_simple() function.
    """
    print("=" * 80)
    print("Example 6: Convenience Function")
    print("=" * 80)
    print()

    # Simple usage
    result = select_base_class_simple("effect")
    print(f"Simple Effect Node: {result}")
    print()

    # With features
    result = select_base_class_simple("effect", ["custom_mixins", "needs_retry"])
    print(f"Effect Node with Retry: {result}")
    print()


def example_7_integration_specific_nodes():
    """
    Example 7: Integration-Specific Nodes

    The selector recognizes common integration patterns.
    """
    print("=" * 80)
    print("Example 7: Integration-Specific Nodes")
    print("=" * 80)
    print()

    selector = MixinSelector()

    # Database adapter
    result = selector.select_base_class("effect", {"integrations": ["database"]})
    print(f"Database Adapter: {result}")
    print("  → Standard convenience wrapper with transaction support")
    print()

    # Kafka consumer
    result = selector.select_base_class("effect", {"integrations": ["kafka"]})
    print(f"Kafka Consumer: {result}")
    print("  → Standard convenience wrapper with event bus integration")
    print()

    # File I/O processor
    result = selector.select_base_class("effect", {"integrations": ["file_io"]})
    print(f"File I/O Processor: {result}")
    print("  → Standard convenience wrapper with atomic operations")
    print()


def example_8_performance_comparison():
    """
    Example 8: Performance Comparison

    Demonstrate the <1ms selection time guarantee.
    """
    print("=" * 80)
    print("Example 8: Performance Comparison")
    print("=" * 80)
    print()

    import time

    selector = MixinSelector()

    # Test 1: Convenience wrapper selection (fast path)
    start = time.perf_counter()
    for _ in range(1000):
        selector.select_base_class("effect", {})
    end = time.perf_counter()
    avg_ms = (end - start) * 1000 / 1000
    print("Convenience Wrapper Selection (1000 iterations):")
    print(f"  Average: {avg_ms:.3f}ms per selection")
    print("  ✓ PASS (<1ms target)" if avg_ms < 1 else f"  ✗ FAIL (>{avg_ms:.3f}ms)")
    print()

    # Test 2: Custom composition selection (slower path)
    requirements = {
        "features": ["custom_mixins", "needs_retry", "needs_circuit_breaker"],
        "security": {"sensitive_data": True},
    }
    start = time.perf_counter()
    for _ in range(1000):
        selector.select_base_class("effect", requirements)
    end = time.perf_counter()
    avg_ms = (end - start) * 1000 / 1000
    print("Custom Composition Selection (1000 iterations):")
    print(f"  Average: {avg_ms:.3f}ms per selection")
    print("  ✓ PASS (<1ms target)" if avg_ms < 1 else f"  ✗ FAIL (>{avg_ms:.3f}ms)")
    print()


if __name__ == "__main__":
    example_1_standard_nodes()
    example_2_fault_tolerant_api_client()
    example_3_high_throughput_stream_processor()
    example_4_secure_data_processor()
    example_5_decision_logging()
    example_6_convenience_function()
    example_7_integration_specific_nodes()
    example_8_performance_comparison()

    print("=" * 80)
    print("All Examples Complete!")
    print("=" * 80)
