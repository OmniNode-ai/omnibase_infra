#!/usr/bin/env python3
"""Demo script showing how to use the OmniNode Bridge CI workflow system."""

import sys
from pathlib import Path

# Add the package to Python path if running as script
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from omninode_bridge.ci import (  # Core functionality; Exceptions; Quick access functions; Templates
    WorkflowConfig,
    WorkflowGenerationError,
    WorkflowGenerator,
    WorkflowValidationError,
    WorkflowValidator,
    create_template,
    format_validation_report,
    generate_docker_build,
    generate_python_ci,
    get_available_templates,
)


def demo_template_generation():
    """Demonstrate generating workflows from templates."""
    print("🔧 Template Generation Demo")
    print("=" * 50)

    # List available templates
    print("\n📋 Available Templates:")
    templates = get_available_templates()
    for name, description in templates.items():
        print(f"  • {name:<20} - {description}")

    # Generate a Python CI workflow
    print("\n🐍 Generating Python CI workflow...")
    try:
        workflow = generate_python_ci(
            name="Demo Python CI",
            python_versions=["3.11", "3.12"],
            test_command="pytest --cov=src --cov-report=xml",
            coverage_threshold=85,
        )

        generator = WorkflowGenerator()
        yaml_content = generator.generate_yaml(workflow)

        print("✅ Generated workflow YAML:")
        print("-" * 30)
        print(yaml_content[:500] + "..." if len(yaml_content) > 500 else yaml_content)

    except WorkflowGenerationError as e:
        print(f"❌ Generation failed: {e}")

    return workflow


def demo_validation():
    """Demonstrate workflow validation."""
    print("\n🔍 Validation Demo")
    print("=" * 50)

    # Create a workflow with some issues for demonstration
    try:
        # Generate a basic workflow
        workflow = create_template("python_ci", name="Test Workflow")

        # Validate the workflow
        validator = WorkflowValidator()
        result = validator.validate(workflow)

        print("📊 Validation Results:")
        report = format_validation_report(result)
        print(report)

        if result.is_valid:
            print("✅ Workflow passed validation!")
        else:
            print("❌ Workflow has validation issues")

    except WorkflowValidationError as e:
        print(f"❌ Validation failed: {e}")


def demo_yaml_serialization():
    """Demonstrate YAML serialization and deserialization."""
    print("\n📄 YAML Serialization Demo")
    print("=" * 50)

    try:
        # Create a workflow
        workflow = generate_docker_build(
            name="Demo Docker Build",
            image_name="my-app",
            platforms=["linux/amd64", "linux/arm64"],
        )

        # Convert to YAML
        yaml_content = workflow.to_yaml_string()
        print("📝 Generated YAML:")
        print("-" * 20)
        print(yaml_content[:400] + "..." if len(yaml_content) > 400 else yaml_content)

        # Parse back from YAML
        parsed_workflow = WorkflowConfig.from_yaml_string(yaml_content)
        print(
            f"\n✅ Successfully round-trip serialized workflow: {parsed_workflow.name}"
        )

    except Exception as e:
        print(f"❌ Serialization demo failed: {e}")


def demo_advanced_features():
    """Demonstrate advanced features."""
    print("\n🚀 Advanced Features Demo")
    print("=" * 50)

    # Create a custom workflow using the builder
    print("🏗️  Building custom workflow...")

    from omninode_bridge.ci.generators.workflow_generator import WorkflowBuilder
    from omninode_bridge.ci.models.github_actions import (
        CheckoutAction,
        SetupPythonAction,
    )
    from omninode_bridge.ci.models.workflow import (
        MatrixStrategy,
        WorkflowJob,
        WorkflowStep,
    )

    try:
        # Build a complex workflow
        builder = WorkflowBuilder("Custom Multi-Service CI")

        # Add triggers
        builder.add_trigger("push", branches=["main", "develop"])
        builder.add_trigger("pull_request", branches=["main"])

        # Create a matrix job
        test_steps = [
            WorkflowStep(name="Checkout", **CheckoutAction().model_dump()),
            WorkflowStep(
                name="Setup Python",
                **SetupPythonAction("${{ matrix.python-version }}").model_dump(),
            ),
            WorkflowStep(name="Install deps", run="pip install -r requirements.txt"),
            WorkflowStep(name="Run tests", run="pytest --cov=src"),
        ]

        strategy = MatrixStrategy(
            matrix={
                "python-version": ["3.11", "3.12"],
                "os": ["ubuntu-latest", "windows-latest"],
            },
            fail_fast=False,
        )

        test_job = WorkflowJob(
            name="Test", runs_on="${{ matrix.os }}", strategy=strategy, steps=test_steps
        )

        builder.add_job("test", test_job)

        # Build the workflow
        workflow = builder.build()

        # Validate it
        validator = WorkflowValidator()
        result = validator.validate(workflow)

        print(f"✅ Built custom workflow with {len(workflow.jobs)} jobs")
        print(f"📊 Validation: {'✅ PASSED' if result.is_valid else '❌ FAILED'}")

        if not result.is_valid:
            print(
                f"   Issues found: {result.errors_count} errors, {result.warnings_count} warnings"
            )

    except Exception as e:
        print(f"❌ Advanced demo failed: {e}")


def main():
    """Run all demos."""
    print("🎯 OmniNode Bridge CI Workflow System Demo")
    print("=" * 60)
    print("This demo showcases the type-safe GitHub Actions workflow system.")
    print()

    try:
        # Run demos
        workflow = demo_template_generation()
        demo_validation()
        demo_yaml_serialization()
        demo_advanced_features()

        print("\n🎉 Demo completed successfully!")
        print("\n💡 Try the CLI tool:")
        print("   omninode-workflow template list")
        print("   omninode-workflow generate from-template python_ci -o workflow.yml")
        print("   omninode-workflow validate file workflow.yml")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
