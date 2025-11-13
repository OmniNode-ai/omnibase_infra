#!/usr/bin/env python3
"""
Example usage of StrategySelector for intelligent strategy selection.

Demonstrates:
1. Simple CRUD node → Jinja2Strategy
2. Complex business logic → TemplateLoadStrategy
3. Production-critical → HybridStrategy
4. Fallback strategy handling

Run with:
    python -m omninode_bridge.codegen.strategies.example_usage
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from omninode_bridge.codegen.prd_analyzer import ModelPRDRequirements
from omninode_bridge.codegen.strategies.base import EnumStrategyType
from omninode_bridge.codegen.strategies.selector import StrategySelector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def example_simple_crud() -> None:
    """
    Example 1: Simple CRUD node.

    Expected: Jinja2Strategy (fast, template-only)
    """
    print("\n" + "=" * 80)
    print("Example 1: Simple CRUD Node (PostgreSQL)")
    print("=" * 80)

    requirements = ModelPRDRequirements(
        node_type="effect",
        service_name="postgres_crud",
        domain="database",
        operations=["create", "read", "update", "delete"],
        business_description="Simple PostgreSQL CRUD operations with connection pooling",
        features=["connection_pooling", "logging"],
        complexity_threshold=4,  # Low complexity
        min_test_coverage=0.85,
    )

    selector = StrategySelector(enable_llm=True, enable_validation=True)
    result = selector.select_strategy(requirements)

    print(f"\n📊 Selected Strategy: {result.selected_strategy.value}")
    print(f"📊 Confidence: {result.confidence:.2%}")
    print("\n📋 Reasoning:")
    for reason in result.reasoning:
        print(f"  • {reason}")

    print("\n📊 All Strategy Scores:")
    for score in result.all_scores:
        print(f"  • {score.strategy.value}: {score.total_score:.1f}/100")
        for component, value in score.component_scores.items():
            print(f"    - {component}: {value:.1f}")

    print(f"\n🔄 Fallback Strategies: {[s.value for s in result.fallback_strategies]}")

    # Assertion for expected result
    assert (
        result.selected_strategy == EnumStrategyType.JINJA2
    ), f"Expected Jinja2Strategy for simple CRUD, got {result.selected_strategy.value}"
    print("\n✅ Result matches expected strategy (Jinja2)")


def example_complex_business_logic() -> None:
    """
    Example 2: Complex business logic node.

    Expected: TemplateLoadStrategy (LLM-enhanced)
    """
    print("\n" + "=" * 80)
    print("Example 2: Complex Business Logic (Payment Processing)")
    print("=" * 80)

    requirements = ModelPRDRequirements(
        node_type="orchestrator",
        service_name="payment_processor",
        domain="api",
        operations=["validate", "authorize", "capture", "refund", "reconcile"],
        business_description=(
            "Complex payment processing orchestration with custom fraud detection, "
            "multi-currency support, and intelligent retry logic"
        ),
        features=[
            "circuit_breaker",
            "retry_logic",
            "distributed_tracing",
            "authentication",
        ],
        dependencies={
            "stripe_api": "^2.0",
            "fraud_detection_service": "^1.0",
        },
        performance_requirements={
            "latency_ms": 150,
            "throughput_per_sec": 500,
        },
        complexity_threshold=18,  # High complexity
        min_test_coverage=0.90,
    )

    selector = StrategySelector(enable_llm=True, enable_validation=True)
    result = selector.select_strategy(requirements)

    print(f"\n📊 Selected Strategy: {result.selected_strategy.value}")
    print(f"📊 Confidence: {result.confidence:.2%}")
    print("\n📋 Reasoning:")
    for reason in result.reasoning:
        print(f"  • {reason}")

    print("\n📊 All Strategy Scores:")
    for score in result.all_scores:
        print(f"  • {score.strategy.value}: {score.total_score:.1f}/100")
        for component, value in score.component_scores.items():
            print(f"    - {component}: {value:.1f}")

    print(f"\n🔄 Fallback Strategies: {[s.value for s in result.fallback_strategies]}")

    # Assertion for expected result
    assert (
        result.selected_strategy == EnumStrategyType.TEMPLATE_LOADING
    ), f"Expected TemplateLoadStrategy for complex logic, got {result.selected_strategy.value}"
    print("\n✅ Result matches expected strategy (TemplateLoad)")


def example_production_critical() -> None:
    """
    Example 3: Production-critical node with high quality requirements.

    Expected: HybridStrategy (best quality with validation)
    """
    print("\n" + "=" * 80)
    print("Example 3: Production-Critical (Core Banking)")
    print("=" * 80)

    requirements = ModelPRDRequirements(
        node_type="effect",
        service_name="core_banking_ledger",
        domain="database",
        operations=[
            "create_transaction",
            "update_balance",
            "reconcile_accounts",
            "generate_statement",
            "audit_trail",
        ],
        business_description=(
            "Production-critical core banking ledger with ACID guarantees, "
            "regulatory compliance, and sophisticated audit trails"
        ),
        features=[
            "distributed_tracing",
            "circuit_breaker",
            "authentication",
            "validation",
            "metrics",
        ],
        dependencies={
            "postgresql": "^14.0",
            "compliance_service": "^2.0",
            "audit_service": "^1.5",
        },
        performance_requirements={
            "latency_ms": 50,
            "throughput_per_sec": 2000,
        },
        complexity_threshold=22,  # Very high complexity
        min_test_coverage=0.95,  # Very high test coverage requirement
    )

    selector = StrategySelector(enable_llm=True, enable_validation=True)
    result = selector.select_strategy(requirements)

    print(f"\n📊 Selected Strategy: {result.selected_strategy.value}")
    print(f"📊 Confidence: {result.confidence:.2%}")
    print("\n📋 Reasoning:")
    for reason in result.reasoning:
        print(f"  • {reason}")

    print("\n📊 All Strategy Scores:")
    for score in result.all_scores:
        print(f"  • {score.strategy.value}: {score.total_score:.1f}/100")
        for component, value in score.component_scores.items():
            print(f"    - {component}: {value:.1f}")

    print(f"\n🔄 Fallback Strategies: {[s.value for s in result.fallback_strategies]}")

    # Assertion for expected result
    assert (
        result.selected_strategy == EnumStrategyType.HYBRID
    ), f"Expected HybridStrategy for production-critical, got {result.selected_strategy.value}"
    print("\n✅ Result matches expected strategy (Hybrid)")


def example_fallback_handling() -> None:
    """
    Example 4: Demonstrate fallback strategy handling.
    """
    print("\n" + "=" * 80)
    print("Example 4: Fallback Strategy Handling")
    print("=" * 80)

    selector = StrategySelector(enable_llm=True, enable_validation=True)

    # Test fallback order for each strategy
    strategies = [
        EnumStrategyType.JINJA2,
        EnumStrategyType.TEMPLATE_LOADING,
        EnumStrategyType.HYBRID,
    ]

    for strategy in strategies:
        fallbacks = selector.get_fallback_order(strategy)
        print(f"\n{strategy.value}:")
        print(f"  Fallback order: {[s.value for s in fallbacks]}")


def example_llm_disabled() -> None:
    """
    Example 5: Strategy selection with LLM disabled.

    Expected: Only Jinja2Strategy available
    """
    print("\n" + "=" * 80)
    print("Example 5: Strategy Selection with LLM Disabled")
    print("=" * 80)

    requirements = ModelPRDRequirements(
        node_type="effect",
        service_name="simple_api",
        domain="api",
        operations=["get", "post"],
        business_description="Simple REST API wrapper",
        features=["logging"],
        complexity_threshold=3,
        min_test_coverage=0.80,
    )

    # Selector with LLM disabled
    selector = StrategySelector(enable_llm=False, enable_validation=True)
    result = selector.select_strategy(requirements)

    print(f"\n📊 Selected Strategy: {result.selected_strategy.value}")
    print(f"📊 Confidence: {result.confidence:.2%}")
    print("\n📋 Available Strategies (LLM disabled):")
    for score in result.all_scores:
        print(f"  • {score.strategy.value}: {score.total_score:.1f}/100")

    # Should only have Jinja2 available
    assert len(result.all_scores) == 1, "Expected only Jinja2 with LLM disabled"
    assert result.all_scores[0].strategy == EnumStrategyType.JINJA2
    print("\n✅ Only Jinja2Strategy available with LLM disabled")


def main() -> None:
    """Run all examples."""
    print("\n🎯 StrategySelector Usage Examples")
    print("=" * 80)

    try:
        example_simple_crud()
        example_complex_business_logic()
        example_production_critical()
        example_fallback_handling()
        example_llm_disabled()

        print("\n" + "=" * 80)
        print("✅ All examples completed successfully!")
        print("=" * 80 + "\n")

    except AssertionError as e:
        print(f"\n❌ Assertion failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Example failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
