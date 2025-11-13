#!/usr/bin/env python3
"""Validation script for generated Claude workflows."""

from pathlib import Path
from typing import Any

import yaml

from .validators.workflow_validator import WorkflowValidator, format_validation_report
from .workflow_generator import ClaudeWorkflowGenerator


def validate_yaml_syntax(file_path: Path) -> bool:
    """Validate YAML syntax of workflow file.

    Args:
        file_path: Path to YAML workflow file

    Returns:
        True if valid YAML, False otherwise
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            yaml.safe_load(f)
        print(f"✅ YAML syntax valid: {file_path.name}")
        return True
    except yaml.YAMLError as e:
        print(f"❌ YAML syntax invalid: {file_path.name} - {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading file: {file_path.name} - {e}")
        return False


def validate_workflow_structure(file_path: Path) -> bool:
    """Validate GitHub Actions workflow structure.

    Args:
        file_path: Path to workflow file

    Returns:
        True if valid structure, False otherwise
    """
    try:
        validator = WorkflowValidator()
        result = validator.validate_file(file_path)

        if result.is_valid:
            print(f"✅ Workflow structure valid: {file_path.name}")
            if result.warnings_count > 0:
                print(f"⚠️  {result.warnings_count} warnings found")
        else:
            print(f"❌ Workflow structure invalid: {file_path.name}")
            print(
                f"   Errors: {result.errors_count}, Warnings: {result.warnings_count}"
            )

        # Print detailed report if there are issues
        if result.issues:
            print("\nDetailed validation report:")
            print(format_validation_report(result))

        return result.is_valid

    except Exception as e:
        print(f"❌ Workflow validation failed: {file_path.name} - {e}")
        return False


def compare_workflow_functionality(original_path: Path, generated_path: Path) -> bool:
    """Compare functional equivalence of workflows.

    Args:
        original_path: Path to original workflow
        generated_path: Path to generated workflow

    Returns:
        True if functionally equivalent, False otherwise
    """
    try:
        with open(original_path, encoding="utf-8") as f:
            original = yaml.safe_load(f)
        with open(generated_path, encoding="utf-8") as f:
            generated = yaml.safe_load(f)

        # Check core functional elements
        checks = [
            ("name", original.get("name") == generated.get("name")),
            ("triggers", _compare_triggers(original.get("on"), generated.get("on"))),
            (
                "jobs",
                _compare_jobs(original.get("jobs", {}), generated.get("jobs", {})),
            ),
        ]

        all_passed = True
        for check_name, passed in checks:
            if passed:
                print(f"✅ {check_name} functionally equivalent")
            else:
                print(f"❌ {check_name} NOT functionally equivalent")
                all_passed = False

        return all_passed

    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        return False


def _compare_triggers(original_on: Any, generated_on: Any) -> bool:
    """Compare workflow triggers for functional equivalence."""
    # Both should define the same events
    if isinstance(original_on, dict) and isinstance(generated_on, dict):
        return set(original_on.keys()) == set(generated_on.keys())
    elif isinstance(original_on, list) and isinstance(generated_on, list):
        return set(original_on) == set(generated_on)
    elif isinstance(original_on, str) and isinstance(generated_on, str):
        return original_on == generated_on
    else:
        return False


def _compare_jobs(
    original_jobs: dict[str, Any], generated_jobs: dict[str, Any]
) -> bool:
    """Compare jobs for functional equivalence."""
    if set(original_jobs.keys()) != set(generated_jobs.keys()):
        return False

    for job_name in original_jobs:
        original_job = original_jobs[job_name]
        generated_job = generated_jobs[job_name]

        # Check key functional elements
        if original_job.get("runs-on") != generated_job.get("runs-on"):
            return False

        # Compare steps (check uses and run commands)
        original_steps = original_job.get("steps", [])
        generated_steps = generated_job.get("steps", [])

        if len(original_steps) != len(generated_steps):
            return False

        for i, (orig_step, gen_step) in enumerate(
            zip(original_steps, generated_steps, strict=False)
        ):
            if orig_step.get("uses") != gen_step.get("uses"):
                return False
            if orig_step.get("run") != gen_step.get("run"):
                return False

    return True


def test_pydantic_models() -> bool:
    """Test that Pydantic models work correctly.

    Returns:
        True if models work correctly, False otherwise
    """
    try:
        generator = ClaudeWorkflowGenerator()

        # Test creating workflows
        claude_workflow = generator.create_claude_workflow()
        review_workflow = generator.create_claude_code_review_workflow()

        # Test validation
        generator.validate_workflow(claude_workflow)
        generator.validate_workflow(review_workflow)

        # Test YAML generation
        claude_yaml = generator.generate_yaml(claude_workflow)
        review_yaml = generator.generate_yaml(review_workflow)

        # Test that generated YAML is valid
        yaml.safe_load(claude_yaml)
        yaml.safe_load(review_yaml)

        print("✅ Pydantic models working correctly")
        return True

    except Exception as e:
        print(f"❌ Pydantic model test failed: {e}")
        return False


def validate_required_fields() -> bool:
    """Validate that generated workflows have all required fields.

    Returns:
        True if all required fields present, False otherwise
    """
    try:
        project_root = Path(__file__).parents[3]
        workflows_dir = project_root / ".github" / "workflows"

        required_fields = {
            "claude.yml": {
                "name": "Claude Code",
                "on": [
                    "issue_comment",
                    "pull_request_review_comment",
                    "issues",
                    "pull_request_review",
                ],
                "jobs": {
                    "claude": {
                        "runs-on": "ubuntu-latest",
                        "permissions": [
                            "contents",
                            "pull_requests",
                            "issues",
                            "id_token",
                            "actions",
                        ],
                        "steps": ["checkout", "claude-action"],
                    }
                },
            },
            "claude-code-review.yml": {
                "name": "Claude Code Review",
                "on": ["pull_request"],
                "jobs": {
                    "claude-review": {
                        "runs-on": "ubuntu-latest",
                        "permissions": [
                            "contents",
                            "pull_requests",
                            "issues",
                            "id_token",
                        ],
                        "steps": ["checkout", "claude-action"],
                    }
                },
            },
        }

        all_valid = True

        for filename, expected in required_fields.items():
            file_path = workflows_dir / filename

            if not file_path.exists():
                print(f"❌ Missing workflow file: {filename}")
                all_valid = False
                continue

            with open(file_path, encoding="utf-8") as f:
                workflow = yaml.safe_load(f)

            # Check name
            if workflow.get("name") != expected["name"]:
                print(f"❌ {filename}: Incorrect name")
                all_valid = False

            # Check that required events are present
            workflow_events = set(workflow.get("on", {}).keys())
            expected_events = set(expected["on"])
            if not expected_events.issubset(workflow_events):
                print(f"❌ {filename}: Missing required events")
                all_valid = False

            # Check jobs
            for job_name, job_requirements in expected["jobs"].items():
                if job_name not in workflow.get("jobs", {}):
                    print(f"❌ {filename}: Missing job '{job_name}'")
                    all_valid = False
                    continue

                job = workflow["jobs"][job_name]

                # Check runs-on
                if job.get("runs-on") != job_requirements["runs-on"]:
                    print(f"❌ {filename}: Job '{job_name}' incorrect runs-on")
                    all_valid = False

                # Check permissions exist
                if "permissions" not in job:
                    print(f"❌ {filename}: Job '{job_name}' missing permissions")
                    all_valid = False

                # Check steps exist
                if len(job.get("steps", [])) < len(job_requirements["steps"]):
                    print(f"❌ {filename}: Job '{job_name}' insufficient steps")
                    all_valid = False

        if all_valid:
            print("✅ All required fields present in workflows")

        return all_valid

    except Exception as e:
        print(f"❌ Required fields validation failed: {e}")
        return False


def main():
    """Main validation function."""
    print("🔍 Validating generated Claude workflows...")
    print("=" * 50)

    project_root = Path(__file__).parents[3]
    workflows_dir = project_root / ".github" / "workflows"

    # Files to validate
    workflows = [
        ("claude.yml", "claude.yml"),
        ("claude-code-review.yml", "claude-code-review.yml"),
    ]

    all_tests_passed = True

    # Test 1: YAML Syntax Validation
    print("\n📋 Test 1: YAML Syntax Validation")
    print("-" * 30)
    for generated_file, _ in workflows:
        file_path = workflows_dir / generated_file
        if not validate_yaml_syntax(file_path):
            all_tests_passed = False

    # Test 2: Workflow Structure Validation
    print("\n🏗️  Test 2: Workflow Structure Validation")
    print("-" * 30)
    for generated_file, _ in workflows:
        file_path = workflows_dir / generated_file
        if not validate_workflow_structure(file_path):
            all_tests_passed = False

    # Test 3: Required Fields Validation
    print("\n📝 Test 3: Required Fields Validation")
    print("-" * 30)
    if not validate_required_fields():
        all_tests_passed = False

    # Test 4: Pydantic Models Test
    print("\n⚙️  Test 4: Pydantic Models Test")
    print("-" * 30)
    if not test_pydantic_models():
        all_tests_passed = False

    # Final Results
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 All validation tests PASSED!")
        print("✅ Generated workflows are valid and ready for production use")
        return 0
    else:
        print("❌ Some validation tests FAILED!")
        print("🔧 Please review the issues above and regenerate workflows")
        return 1


if __name__ == "__main__":
    exit(main())
