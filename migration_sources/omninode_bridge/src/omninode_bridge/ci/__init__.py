"""OmniNode Bridge CI Workflow Management System.

A comprehensive Pydantic-based GitHub Actions workflow generation and validation system.
"""

from .exceptions import (
    CIWorkflowError,
    FileOperationError,
    PerformanceError,
    SecurityViolationError,
    TemplateError,
    WorkflowGenerationError,
    WorkflowValidationError,
    YAMLProcessingError,
)
from .generators.workflow_generator import (
    WorkflowBuilder,
    WorkflowGenerator,
    YAMLFormatter,
    create_simple_ci_workflow,
)
from .models.github_actions import (
    ActionVersion,
    CacheAction,
    CheckoutAction,
    ClaudeCodeAction,
    DockerContainer,
    DownloadArtifactAction,
    PythonVersion,
    ServiceContainer,
    SetupPythonAction,
    ShellCommand,
    UploadArtifactAction,
)
from .models.workflow import (
    EventType,
    MatrixStrategy,
    PermissionLevel,
    PermissionSet,
    WorkflowConfig,
    WorkflowJob,
    WorkflowStep,
)
from .templates.templates import (
    WorkflowTemplates,
    create_template,
    get_available_templates,
)
from .validators.workflow_validator import (
    ValidationIssue,
    ValidationResult,
    ValidationRule,
    ValidationSeverity,
    WorkflowValidator,
    format_validation_report,
)

__all__ = [
    # Core models
    "WorkflowConfig",
    "WorkflowJob",
    "WorkflowStep",
    "MatrixStrategy",
    "EventType",
    "PermissionSet",
    "PermissionLevel",
    # GitHub Actions models
    "CheckoutAction",
    "SetupPythonAction",
    "UploadArtifactAction",
    "DownloadArtifactAction",
    "CacheAction",
    "ClaudeCodeAction",
    "ShellCommand",
    "DockerContainer",
    "ServiceContainer",
    "PythonVersion",
    "ActionVersion",
    # Generation
    "WorkflowGenerator",
    "YAMLFormatter",
    "WorkflowBuilder",
    "create_simple_ci_workflow",
    # Validation
    "WorkflowValidator",
    "ValidationResult",
    "ValidationIssue",
    "ValidationSeverity",
    "ValidationRule",
    "format_validation_report",
    # Templates
    "WorkflowTemplates",
    "get_available_templates",
    "create_template",
    # Exceptions
    "CIWorkflowError",
    "WorkflowValidationError",
    "WorkflowGenerationError",
    "TemplateError",
    "YAMLProcessingError",
    "FileOperationError",
    "SecurityViolationError",
    "PerformanceError",
]


def quick_start_example():
    """Generate a simple example workflow for demonstration."""
    from .generators.workflow_generator import create_simple_ci_workflow

    # Create a simple Python CI workflow
    workflow = create_simple_ci_workflow(
        name="Quick Start Example",
        python_versions=["3.11", "3.12"],
        test_command="pytest --cov=src",
        lint_command="ruff check src",
    )

    # Generate YAML
    generator = WorkflowGenerator()
    yaml_content = generator.generate_yaml(workflow)

    return yaml_content


def validate_workflow_file(file_path: str) -> ValidationResult:
    """Quick validation of a workflow file.

    Args:
        file_path: Path to workflow YAML file

    Returns:
        Validation result
    """
    validator = WorkflowValidator()
    return validator.validate_file(file_path)


# Quick access functions for common operations
def generate_python_ci(name: str, **kwargs) -> WorkflowConfig:
    """Generate Python CI workflow with defaults."""
    return WorkflowTemplates.python_ci_template(name=name, **kwargs)


def generate_docker_build(name: str, **kwargs) -> WorkflowConfig:
    """Generate Docker build workflow with defaults."""
    return WorkflowTemplates.docker_build_template(name=name, **kwargs)


def generate_security_scan(name: str, **kwargs) -> WorkflowConfig:
    """Generate security scanning workflow with defaults."""
    return WorkflowTemplates.security_scan_template(name=name, **kwargs)
