#!/usr/bin/env python3
"""
Inspect generated code quality.

Renders a template and displays the output for manual inspection.
"""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader


def inspect_generated_code():
    """Render and display generated code."""
    # Get templates directory
    script_dir = Path(__file__).parent
    templates_dir = (
        script_dir.parent / "src" / "omninode_bridge" / "codegen" / "templates"
    )

    # Initialize Jinja2 environment
    env = Environment(loader=FileSystemLoader(str(templates_dir)), autoescape=False)

    # Register custom filters
    import re

    def to_snake_case(text):
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", text)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

    env.filters["to_snake_case"] = to_snake_case
    env.filters["sort_imports"] = sorted
    env.filters["repr"] = repr

    # Build test context with mixins
    context = {
        "node_name": "DatabaseAdapterEffect",
        "class_name": "NodeDatabaseAdapterEffect",
        "description": "Database adapter with connection pooling and transaction support",
        "service_name": "database_adapter",
        "node_type": "effect",
        "domain": "persistence",
        "generation_timestamp": "2025-11-04T12:00:00Z",
        "base_classes": ["NodeEffect", "MixinHealthCheck", "MixinMetrics"],
        "imports": {
            "standard_library": [
                "import logging",
                "from typing import Any, Dict, List, Optional",
                "from datetime import UTC, datetime",
            ],
            "third_party": [],
            "omnibase_core": [
                "from omnibase_core.models.core import ModelContainer",
                "from omnibase_core.nodes.node_effect import NodeEffect",
                "from omnibase_core import ModelOnexError, EnumCoreErrorCode",
                "from omnibase_core.enums.enum_log_level import EnumLogLevel as LogLevel",
                "from omnibase_core.logging.structured import emit_log_event_sync as emit_log_event",
                "from omnibase_core.nodes.model_circuit_breaker import ModelCircuitBreaker",
            ],
            "omnibase_mixins": [
                "from omnibase_core.mixins.mixin_health_check import MixinHealthCheck",
                "from omnibase_core.mixins.mixin_metrics import MixinMetrics",
            ],
            "project_local": [],
        },
        "enabled_mixins": ["MixinHealthCheck", "MixinMetrics"],
        "mixin_configs": {
            "MixinHealthCheck": {
                "check_interval_ms": 30000,
                "timeout_seconds": 5.0,
            },
            "MixinMetrics": {
                "metrics_prefix": "database_adapter",
                "enable_system_metrics": True,
            },
        },
        "mixin_descriptions": {
            "MixinHealthCheck": "Health monitoring and component checks",
            "MixinMetrics": "Performance metrics collection",
        },
        "health_check_components": [
            {"name": "database", "critical": True, "timeout_seconds": 5.0},
            {"name": "connection_pool", "critical": True, "timeout_seconds": 3.0},
        ],
        "io_operations": [
            {
                "name": "execute_query",
                "description": "Execute SQL query with transaction support",
                "input_model": "ModelDatabaseAdapterRequest",
                "output_model": "ModelDatabaseAdapterResponse",
                "operation_type": "database_query",
                "atomic": True,
                "timeout_seconds": 30,
                "validation_enabled": True,
            }
        ],
        "compute_operations": [],
        "reduction_operations": [],
        "aggregation_types": [],
        "workflows": [],
        "advanced_features": {
            "circuit_breaker": {
                "services": {
                    "postgres": {
                        "failure_threshold": 5,
                        "recovery_timeout_ms": 60000,
                    }
                }
            },
            "retry_policy": {
                "max_attempts": 3,
                "initial_delay_ms": 100,
                "max_delay_ms": 5000,
                "exponential_base": 2.0,
            },
        },
        "version_dict": {"major": 1, "minor": 0, "patch": 0},
        "package_path": "omninode_bridge.nodes.database_adapter.v1_0_0.node",
        "input_model": "ModelDatabaseAdapterRequest",
        "output_model": "ModelDatabaseAdapterResponse",
        "operations": ["execute_query", "begin_transaction", "commit_transaction"],
        "features": ["connection_pooling", "transaction_support", "health_monitoring"],
        "performance_requirements": {
            "target_execution_time_ms": 50,
            "max_execution_time_ms": 500,
        },
        "testing": {
            "unit_test_coverage": 90,
            "integration_tests_required": True,
        },
    }

    # Render Effect template
    print("=" * 80)
    print("Generated Effect Node with Mixins:")
    print("=" * 80)
    template = env.get_template("node_templates/node_effect.py.j2")
    output = template.render(**context)
    print(output)

    print("\n" + "=" * 80)
    print("Generated Contract YAML:")
    print("=" * 80)
    contract_template = env.get_template("node_templates/contract.yaml.j2")
    contract_output = contract_template.render(**context)
    print(contract_output)


if __name__ == "__main__":
    inspect_generated_code()
