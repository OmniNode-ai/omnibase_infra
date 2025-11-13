#!/usr/bin/env python3
"""
Sample PRD requirements for testing.

Provides a variety of realistic requirements for different node types and complexity levels.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omninode_bridge.codegen.prd_analyzer import ModelPRDRequirements
else:
    # Import at runtime to avoid circular dependencies
    import importlib

    def _get_prd_requirements():
        module = importlib.import_module("omninode_bridge.codegen.prd_analyzer")
        return module.ModelPRDRequirements

    ModelPRDRequirements = None  # Will be set at runtime


def get_simple_crud_requirements():  # -> ModelPRDRequirements
    """
    Simple CRUD effect node requirements.

    Low complexity (< 5), ideal for Jinja2Strategy.
    """
    from omninode_bridge.codegen.prd_analyzer import ModelPRDRequirements

    return ModelPRDRequirements(
        service_name="postgres_crud",
        node_type="effect",
        domain="database",
        business_description="Simple PostgreSQL CRUD operations for user management",
        operations=["create", "read", "update", "delete"],
        features=["logging"],  # Simple feature to keep complexity low
        input_schema={
            "type": "object",
            "properties": {
                "operation": {"type": "string"},
                "user_id": {"type": "string"},
                "data": {"type": "object"},
            },
        },
        output_schema={
            "type": "object",
            "properties": {"success": {"type": "boolean"}, "data": {"type": "object"}},
        },
        performance_requirements={},  # No specific performance requirements for simple CRUD
        error_handling_strategy="return_error_response",
        dependencies={},  # No dependencies for simple CRUD
        complexity_threshold=0,  # Very simple - let calculated complexity determine
        extraction_confidence=0.8,  # High confidence for strict validation
    )


def get_moderate_complexity_requirements():  # -> ModelPRDRequirements
    """
    Moderate complexity compute node requirements.

    Moderate complexity (5-10), ideal for TemplateLoadStrategy.
    """
    from omninode_bridge.codegen.prd_analyzer import ModelPRDRequirements

    return ModelPRDRequirements(
        service_name="data_transformer",
        node_type="compute",
        domain="data_processing",
        business_description="Transform and validate incoming data with standard business logic",
        operations=["transform", "validate", "enrich", "filter"],
        features=["caching", "validation"],
        input_schema={
            "type": "object",
            "properties": {
                "data": {"type": "array"},
                "transformation_rules": {"type": "object"},
            },
        },
        output_schema={
            "type": "object",
            "properties": {
                "transformed_data": {"type": "array"},
                "validation_results": {"type": "object"},
            },
        },
        performance_requirements={},  # No specific performance requirements
        error_handling_strategy="retry_with_exponential_backoff",
        dependencies={},  # No dependencies for moderate complexity
        complexity_threshold=2,  # Moderate - let operations and features drive complexity
        extraction_confidence=0.7,  # Medium confidence
    )


def get_complex_orchestration_requirements():  # -> ModelPRDRequirements
    """
    Complex orchestration node requirements.

    High complexity (> 10), ideal for HybridStrategy.
    """
    from omninode_bridge.codegen.prd_analyzer import ModelPRDRequirements

    return ModelPRDRequirements(
        service_name="payment_orchestrator",
        node_type="orchestrator",
        domain="financial",
        business_description="Production-critical complex multi-step payment orchestration with fraud detection, risk assessment, and dynamic routing",
        operations=[
            "orchestrate_payment",
            "validate_transaction",
            "assess_risk",
            "route_payment",
            "handle_failures",
        ],
        features=[
            "circuit_breaker",
            "rate_limiting",
            "distributed_tracing",
            "authentication",
            "fraud_detection",
            "retry_logic",
        ],
        input_schema={
            "type": "object",
            "properties": {
                "payment_data": {"type": "object"},
                "customer_profile": {"type": "object"},
                "merchant_config": {"type": "object"},
            },
        },
        output_schema={
            "type": "object",
            "properties": {
                "transaction_id": {"type": "string"},
                "status": {"type": "string"},
                "routing_decisions": {"type": "array"},
                "risk_score": {"type": "number"},
            },
        },
        performance_requirements={"max_latency_ms": 2000, "min_throughput_rps": 50},
        error_handling_strategy="circuit_breaker_with_fallback",
        dependencies={
            "fraud_detection_service": "v2.1.0",
            "risk_assessment_service": "v1.5.0",
            "payment_gateway": "v3.0.0",
            "notification_service": "v1.2.0",
        },
        complexity_threshold=18,  # Complex
        min_test_coverage=0.9,  # High coverage requirement
    )


def get_reducer_requirements():  # -> ModelPRDRequirements
    """
    Reducer node requirements for testing.

    Used for testing reducer-specific generation.
    """
    from omninode_bridge.codegen.prd_analyzer import ModelPRDRequirements

    return ModelPRDRequirements(
        service_name="event_aggregator",
        node_type="reducer",
        domain="event_processing",
        business_description="Aggregate streaming events by namespace with time-window grouping",
        operations=["aggregate", "group_by_namespace", "windowed_aggregation"],
        features=["streaming", "time_windows", "state_management"],
        input_schema={
            "type": "object",
            "properties": {
                "events": {"type": "array"},
                "window_size_ms": {"type": "number"},
            },
        },
        output_schema={
            "type": "object",
            "properties": {
                "aggregated_results": {"type": "object"},
                "window_end": {"type": "string"},
            },
        },
        performance_requirements={"max_latency_ms": 100, "min_throughput_rps": 1000},
        error_handling_strategy="accumulate_errors",
        dependencies={"state_store": "v1.0.0"},
        complexity_threshold=6,
    )


def get_invalid_requirements():  # -> ModelPRDRequirements
    """
    Invalid requirements for testing validation.

    Missing required fields and low confidence.
    """
    from omninode_bridge.codegen.prd_analyzer import ModelPRDRequirements

    return ModelPRDRequirements(
        service_name="",  # Missing service name
        node_type="effect",
        domain="",  # Missing domain
        business_description="Incomplete requirements for testing",
        operations=[],  # No operations
        features=[],
        input_schema={},
        output_schema={},
        performance_requirements={},
        error_handling_strategy="",
        dependencies={},
        extraction_confidence=0.3,  # Low confidence
    )
