#!/usr/bin/env python3
"""
Example Usage: Health Check Pattern Generator.

Demonstrates how the template engine uses the health check generator
to create production-ready health check implementations.

This is a reference implementation showing integration patterns.
"""

from omninode_bridge.codegen.patterns.health_checks import (
    HealthCheckGenerator,
    generate_consul_health_check,
    generate_database_health_check,
    generate_health_check_method,
    generate_http_service_health_check,
    generate_kafka_health_check,
    generate_self_health_check,
)


def example_1_generate_complete_health_checks():
    """
    Example 1: Generate complete health check implementation for a node.

    This shows how the template engine would generate all health checks
    for a PostgreSQL CRUD Effect node with Kafka and Consul dependencies.
    """
    print("=" * 80)
    print("Example 1: Complete Health Check Implementation")
    print("=" * 80)

    # Node configuration
    node_type = "NodePostgresCRUDEffect"
    dependencies = ["postgres", "kafka", "consul"]
    operations = ["read", "write", "update", "delete"]

    # Generate complete health check implementation
    health_check_code = generate_health_check_method(
        node_type=node_type,
        dependencies=dependencies,
        operations=operations,
    )

    print("\nGenerated Health Check Code:")
    print("-" * 80)
    print(health_check_code)
    print("-" * 80)
    print(
        f"\nGenerated {len(health_check_code.split('async def'))} health check methods"
    )


def example_2_generate_individual_checks():
    """
    Example 2: Generate individual health check methods.

    Shows how to generate specific health checks one at a time,
    useful when you need fine-grained control.
    """
    print("\n" + "=" * 80)
    print("Example 2: Individual Health Check Methods")
    print("=" * 80)

    # Generate self health check
    print("\n--- Self Health Check ---")
    self_check = generate_self_health_check()
    print(self_check[:500] + "...")  # Print first 500 chars

    # Generate database health check
    print("\n--- Database Health Check ---")
    db_check = generate_database_health_check()
    print(db_check[:500] + "...")

    # Generate Kafka health check
    print("\n--- Kafka Health Check ---")
    kafka_check = generate_kafka_health_check()
    print(kafka_check[:500] + "...")

    # Generate Consul health check
    print("\n--- Consul Health Check ---")
    consul_check = generate_consul_health_check()
    print(consul_check[:500] + "...")


def example_3_custom_http_service_checks():
    """
    Example 3: Generate HTTP service health checks for external dependencies.

    Shows how to create health checks for HTTP-based services like
    OnexTree, MetadataStamping, or other microservices.
    """
    print("\n" + "=" * 80)
    print("Example 3: Custom HTTP Service Health Checks")
    print("=" * 80)

    # Generate health check for OnexTree service
    print("\n--- OnexTree Service Health Check ---")
    onextree_check = generate_http_service_health_check(
        service_name="onextree",
        service_url="http://onextree:8058",
    )
    print(onextree_check[:500] + "...")

    # Generate health check for Metadata Stamping service
    print("\n--- Metadata Stamping Service Health Check ---")
    metadata_check = generate_http_service_health_check(
        service_name="metadata_stamping",
        service_url="http://metadata-stamping:8053",
    )
    print(metadata_check[:500] + "...")


def example_4_template_engine_integration():
    """
    Example 4: How TemplateEngine integrates with HealthCheckGenerator.

    Demonstrates the typical workflow in the template engine:
    1. Parse contract to extract dependencies
    2. Generate health check methods
    3. Inject into node.py template
    """
    print("\n" + "=" * 80)
    print("Example 4: Template Engine Integration Pattern")
    print("=" * 80)

    # Simulate contract parsing (would come from actual contract.yaml)
    contract_data = {
        "node_type": "NodePostgresCRUDEffect",
        "dependencies": {
            "database": {"type": "postgres", "critical": True},
            "event_bus": {"type": "kafka", "critical": False},
            "service_discovery": {"type": "consul", "critical": False},
        },
        "operations": ["create", "read", "update", "delete"],
    }

    # Extract dependencies for health check generation
    dependencies = list(contract_data["dependencies"].keys())
    # Map dependency names to health check types
    dep_mapping = {
        "database": "postgres",
        "event_bus": "kafka",
        "service_discovery": "consul",
    }
    health_check_deps = [dep_mapping.get(d, d) for d in dependencies]

    print(f"\nContract: {contract_data['node_type']}")
    print(f"Dependencies: {dependencies}")
    print(f"Health Check Dependencies: {health_check_deps}")

    # Generate health checks
    generator = HealthCheckGenerator()
    health_check_code = generator.generate_health_check_method(
        node_type=contract_data["node_type"],
        dependencies=health_check_deps,
        operations=contract_data.get("operations"),
    )

    print(
        f"\nGenerated {len(health_check_code.splitlines())} lines of health check code"
    )

    # In template engine, this would be injected into node.py template:
    # {{ health_check_methods }}


def example_5_generated_code_structure():
    """
    Example 5: Show structure of generated health check code.

    Demonstrates what the final generated code looks like in a node.
    """
    print("\n" + "=" * 80)
    print("Example 5: Generated Code Structure")
    print("=" * 80)

    # Generate a simple example
    generator = HealthCheckGenerator()
    code = generator.generate_health_check_method(
        node_type="NodeExampleEffect",
        dependencies=["postgres"],
    )

    # Show code structure
    methods = code.split("async def")
    print(f"\nGenerated {len(methods)} methods:")
    for i, method in enumerate(methods):
        if method.strip():
            # Extract method name
            method_name = method.split("(")[0].strip()
            if not method_name:
                method_name = "_register_component_checks"
            print(f"  {i}. {method_name}")

    print("\nFull generated code preview:")
    print("-" * 80)
    print(code[:1000] + "...")
    print("-" * 80)


def example_6_integration_with_jinja2():
    """
    Example 6: Integration with Jinja2 templates.

    Shows how health check code would be used in Jinja2 templates.
    """
    print("\n" + "=" * 80)
    print("Example 6: Jinja2 Template Integration")
    print("=" * 80)

    # This is how it would be used in a Jinja2 template:
    jinja2_template = """
{# node.py.j2 template #}

class {{ node_class_name }}(NodeEffect, HealthCheckMixin):
    '''{{ node_description }}'''

    def __init__(self, container: ModelContainer):
        super().__init__(container)
        self.initialize_health_checks()

{{ health_check_methods | indent(4) }}

    async def execute(self, input_data: dict) -> dict:
        '''Execute node logic.'''
        # Node implementation here
        pass
"""

    print("Jinja2 Template:")
    print(jinja2_template)

    # In template engine rendering:
    print("\nTemplate Context:")
    context = {
        "node_class_name": "NodeExampleEffect",
        "node_description": "Example Effect Node with Health Checks",
        "health_check_methods": generate_health_check_method(
            node_type="NodeExampleEffect",
            dependencies=["postgres", "kafka"],
        ),
    }
    print(f"  node_class_name: {context['node_class_name']}")
    print(f"  node_description: {context['node_description']}")
    print(f"  health_check_methods: {len(context['health_check_methods'])} characters")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("Health Check Pattern Generator - Example Usage")
    print("=" * 80)

    # Run examples
    example_1_generate_complete_health_checks()
    example_2_generate_individual_checks()
    example_3_custom_http_service_checks()
    example_4_template_engine_integration()
    example_5_generated_code_structure()
    example_6_integration_with_jinja2()

    print("\n" + "=" * 80)
    print("Examples Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
