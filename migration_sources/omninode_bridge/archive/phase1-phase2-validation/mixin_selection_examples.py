#!/usr/bin/env python3
"""
Examples demonstrating Jinja2Strategy mixin selection.

Shows how different requirements trigger convenience wrapper vs custom composition.
"""

from omninode_bridge.codegen.node_classifier import (
    EnumNodeType,
    ModelClassificationResult,
)
from omninode_bridge.codegen.prd_analyzer import ModelPRDRequirements
from omninode_bridge.codegen.template_engine import TemplateEngine


def print_mixin_selection(
    title: str,
    requirements: ModelPRDRequirements,
    classification: ModelClassificationResult,
):
    """Print mixin selection results for given requirements."""
    print(f"\n{'=' * 80}")
    print(f" {title}")
    print(f"{'=' * 80}\n")

    # Create template engine
    engine = TemplateEngine()

    # Get mixin selection
    selection = engine._select_base_class(requirements, classification)

    # Print results
    print("Requirements:")
    print(f"  Service Name: {requirements.service_name}")
    print(f"  Description: {requirements.business_description}")
    print(f"  Operations: {', '.join(requirements.operations)}")
    print(f"  Features: {', '.join(requirements.features)}")
    print(f"  Dependencies: {list(requirements.dependencies.keys())}")
    print()

    print("Selection Result:")
    print(
        f"  Path: {'Convenience Wrapper' if selection['use_convenience_wrapper'] else 'Custom Composition'}"
    )
    print(f"  Base Class: {selection['base_class_name']}")
    print(
        f"  Mixin List: {', '.join(selection['mixin_list']) if selection['mixin_list'] else 'Included in wrapper'}"
    )
    print()

    print("Selection Reasoning:")
    reasoning = selection["selection_reasoning"]
    print(f"  Complexity: {reasoning['complexity']}")
    print(f"  Needs Retry: {reasoning['needs_retry']}")
    print(f"  Needs Circuit Breaker: {reasoning['needs_circuit_breaker']}")
    print(f"  Needs Validation: {reasoning['needs_validation']}")
    print(f"  Needs Security: {reasoning['needs_security']}")
    print(f"  No Service Mode: {reasoning['no_service_mode']}")
    print()

    print("Generated Imports:")
    for category, imports in selection["import_paths"].items():
        print(f"  {category}:")
        for import_line in imports:
            print(f"    {import_line}")


def example_1_simple_crud():
    """Example 1: Simple CRUD → Convenience Wrapper"""
    requirements = ModelPRDRequirements(
        service_name="postgres_crud",
        business_description="Simple PostgreSQL CRUD operations for user management",
        operations=["create", "read", "update", "delete"],
        features=["connection_pooling"],
        domain="database",
        node_type="effect",
        dependencies={},
        performance_requirements={},
        data_models=[],
        best_practices=[],
        code_examples=[],
    )

    classification = ModelClassificationResult(
        node_type=EnumNodeType.EFFECT,
        confidence=0.95,
        reasoning="Standard database CRUD operations",
        template_name="effect",
        template_variant="standard",
    )

    print_mixin_selection(
        "Example 1: Simple CRUD Operations", requirements, classification
    )


def example_2_resilient_api():
    """Example 2: Fault-tolerant API client → Custom Composition"""
    requirements = ModelPRDRequirements(
        service_name="resilient_api_client",
        business_description="Fault-tolerant external API client with retry and circuit breaker for resilient communication",
        operations=["call_api", "handle_failure", "recover"],
        features=["retry_logic", "circuit_breaker", "metrics"],
        domain="api_client",
        node_type="effect",
        dependencies={"requests": "^2.28.0"},
        performance_requirements={"latency_ms": 500},
        data_models=[],
        best_practices=[],
        code_examples=[],
    )

    classification = ModelClassificationResult(
        node_type=EnumNodeType.EFFECT,
        confidence=0.90,
        reasoning="API client with fault tolerance requirements",
        template_name="effect",
        template_variant="resilient",
    )

    print_mixin_selection(
        "Example 2: Resilient API Client", requirements, classification
    )


def example_3_secure_processor():
    """Example 3: Secure data processor → Custom Composition"""
    requirements = ModelPRDRequirements(
        service_name="secure_data_processor",
        business_description="Secure data processor with validation and PII redaction",
        operations=["validate_input", "redact_pii", "process_data"],
        features=["validation", "security_redaction"],
        domain="data_processing",
        node_type="compute",
        dependencies={},
        performance_requirements={},
        data_models=[],
        best_practices=[],
        code_examples=[],
    )

    classification = ModelClassificationResult(
        node_type=EnumNodeType.COMPUTE,
        confidence=0.92,
        reasoning="Secure data processing with validation",
        template_name="compute",
        template_variant="secure",
    )

    print_mixin_selection(
        "Example 3: Secure Data Processor", requirements, classification
    )


def example_4_complex_workflow():
    """Example 4: Complex workflow orchestrator → Custom Composition (high complexity)"""
    requirements = ModelPRDRequirements(
        service_name="complex_workflow_orchestrator",
        business_description="Multi-step workflow orchestrator for complex business processes",
        operations=[
            "validate_inputs",
            "allocate_resources",
            "execute_step_1",
            "execute_step_2",
            "execute_step_3",
            "aggregate_results",
            "cleanup_resources",
            "handle_errors",
        ],
        features=[
            "dependency_management",
            "parallel_execution",
            "error_recovery",
            "state_persistence",
            "monitoring",
            "alerting",
        ],
        domain="workflow_orchestration",
        node_type="orchestrator",
        dependencies={"celery": "^5.0.0", "redis": "^4.0.0"},
        performance_requirements={"latency_ms": 1000, "throughput_per_sec": 100},
        data_models=[],
        best_practices=[],
        code_examples=[],
    )

    classification = ModelClassificationResult(
        node_type=EnumNodeType.ORCHESTRATOR,
        confidence=0.88,
        reasoning="Complex workflow with high operation count",
        template_name="orchestrator",
        template_variant="complex",
    )

    print_mixin_selection(
        "Example 4: Complex Workflow Orchestrator", requirements, classification
    )


def example_5_one_shot_compute():
    """Example 5: One-shot ephemeral computation → Custom Composition (no service mode)"""
    requirements = ModelPRDRequirements(
        service_name="ephemeral_compute",
        business_description="One-shot temporary computation for batch processing",
        operations=["compute"],
        features=["batch_processing"],
        domain="batch_compute",
        node_type="compute",
        dependencies={},
        performance_requirements={},
        data_models=[],
        best_practices=[],
        code_examples=[],
    )

    classification = ModelClassificationResult(
        node_type=EnumNodeType.COMPUTE,
        confidence=0.95,
        reasoning="One-shot batch computation",
        template_name="compute",
        template_variant="batch",
    )

    print_mixin_selection(
        "Example 5: One-Shot Ephemeral Compute", requirements, classification
    )


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" Jinja2Strategy Mixin Selection Examples")
    print("=" * 80)

    # Run examples
    example_1_simple_crud()
    example_2_resilient_api()
    example_3_secure_processor()
    example_4_complex_workflow()
    example_5_one_shot_compute()

    print("\n" + "=" * 80)
    print(" Summary")
    print("=" * 80)
    print()
    print("Convenience Wrapper (ModelService*): Examples 1")
    print("Custom Composition: Examples 2, 3, 4, 5")
    print()
    print("Triggers for Custom Composition:")
    print("  - Complexity > 10 (Example 4)")
    print("  - Retry/Circuit Breaker keywords (Example 2)")
    print("  - Validation/Security keywords (Example 3)")
    print("  - One-shot/Ephemeral keywords (Example 5)")
    print()
