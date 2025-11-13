#!/usr/bin/env python3
"""
Standalone template validation script.

Validates Jinja2 templates can render without errors and generate valid Python/YAML.
"""

import ast
import sys
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader


def validate_templates():
    """Validate all node templates."""
    # Get templates directory
    script_dir = Path(__file__).parent
    templates_dir = (
        script_dir.parent / "src" / "omninode_bridge" / "codegen" / "templates"
    )

    if not templates_dir.exists():
        print(f"❌ Templates directory not found: {templates_dir}")
        return False

    # Initialize Jinja2 environment
    env = Environment(loader=FileSystemLoader(str(templates_dir)), autoescape=False)

    # Register custom filters
    def to_snake_case(text):
        """Convert CamelCase to snake_case."""
        import re

        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", text)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

    env.filters["to_snake_case"] = to_snake_case
    env.filters["sort_imports"] = sorted
    env.filters["repr"] = repr

    # Build test context
    context = {
        "node_name": "TestServiceEffect",
        "class_name": "NodeTestServiceEffect",
        "description": "A test service for template validation",
        "service_name": "test_service",
        "node_type": "effect",
        "domain": "testing",
        "generation_timestamp": "2025-11-04T00:00:00Z",
        "base_classes": ["NodeEffect"],
        "imports": {
            "standard_library": ["import logging", "from typing import Any"],
            "third_party": [],
            "omnibase_core": ["from omnibase_core.models.core import ModelContainer"],
            "omnibase_mixins": [],
            "project_local": [],
        },
        "enabled_mixins": [],
        "mixin_configs": {},
        "mixin_descriptions": {},
        "health_check_components": [],
        "io_operations": [
            {
                "name": "create_record",
                "description": "Create a new record",
                "input_model": "ModelTestServiceRequest",
                "output_model": "ModelTestServiceResponse",
            }
        ],
        "compute_operations": [],
        "reduction_operations": [],
        "aggregation_types": [],
        "workflows": [],
        "advanced_features": {},
        "version_dict": {"major": 1, "minor": 0, "patch": 0},
        "package_path": "omninode_bridge.nodes.test_service.v1_0_0.node",
        "input_model": "ModelTestServiceRequest",
        "output_model": "ModelTestServiceResponse",
        "operations": ["create", "read", "update"],
        "features": ["caching", "monitoring"],
        "performance_requirements": {},
        "testing": {
            "unit_test_coverage": 85,
            "integration_tests_required": True,
        },
    }

    tests_passed = 0
    tests_failed = 0

    # Test node templates
    node_templates = [
        ("node_templates/node_effect.py.j2", "effect"),
        ("node_templates/node_compute.py.j2", "compute"),
        ("node_templates/node_reducer.py.j2", "reducer"),
        ("node_templates/node_orchestrator.py.j2", "orchestrator"),
    ]

    for template_path, node_type in node_templates:
        print(f"\n🧪 Testing {template_path}...")

        try:
            template = env.get_template(template_path)

            # Update context for node type
            test_context = context.copy()
            test_context["node_type"] = node_type
            test_context["class_name"] = f"NodeTestService{node_type.capitalize()}"

            # Render template
            output = template.render(**test_context)

            # Validate Python syntax
            ast.parse(output)

            print(f"✅ {template_path}: Valid Python generated ({len(output)} chars)")
            tests_passed += 1

        except Exception as e:
            print(f"❌ {template_path}: {e}")
            tests_failed += 1

    # Test contract template
    print("\n🧪 Testing node_templates/contract.yaml.j2...")
    try:
        template = env.get_template("node_templates/contract.yaml.j2")
        output = template.render(**context)

        # Validate YAML syntax
        yaml.safe_load(output)

        print(f"✅ contract.yaml.j2: Valid YAML generated ({len(output)} chars)")
        tests_passed += 1

    except Exception as e:
        print(f"❌ contract.yaml.j2: {e}")
        tests_failed += 1

    # Test __init__ template
    print("\n🧪 Testing node_templates/__init__.py.j2...")
    try:
        template = env.get_template("node_templates/__init__.py.j2")
        output = template.render(**context)

        # Validate Python syntax
        ast.parse(output)

        print(f"✅ __init__.py.j2: Valid Python generated ({len(output)} chars)")
        tests_passed += 1

    except Exception as e:
        print(f"❌ __init__.py.j2: {e}")
        tests_failed += 1

    # Summary
    print(f"\n{'='*60}")
    print("📊 Template Validation Summary:")
    print(f"   ✅ Passed: {tests_passed}")
    print(f"   ❌ Failed: {tests_failed}")
    print(f"   Total:  {tests_passed + tests_failed}")
    print(f"{'='*60}")

    return tests_failed == 0


if __name__ == "__main__":
    success = validate_templates()
    sys.exit(0 if success else 1)
