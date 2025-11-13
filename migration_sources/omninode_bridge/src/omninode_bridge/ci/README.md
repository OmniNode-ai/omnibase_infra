# OmniNode Bridge CI Workflow System

A comprehensive, type-safe GitHub Actions workflow generation and validation system built with Pydantic models. This system provides programmatic creation, validation, and management of GitHub Actions workflows with full type safety and comprehensive error handling.

## 🚀 Features

- **Type-Safe Models**: Complete Pydantic models for GitHub Actions workflows
- **YAML Generation**: Convert Pydantic models to GitHub Actions YAML
- **Comprehensive Validation**: Multi-rule validation system with security, performance, and best practice checks
- **Pre-built Templates**: Ready-to-use templates for common CI/CD scenarios
- **CLI Tools**: Command-line interface for workflow generation and validation
- **Error Handling**: Comprehensive exception system with detailed error reporting
- **Round-trip Serialization**: Full YAML ↔ Pydantic model conversion

## 📦 Installation

The CI workflow system is part of the OmniNode Bridge package:

```bash
# Install from source
pip install -e .

# The CLI tool will be available as:
omninode-workflow --help
```

## 🏃 Quick Start

### Python API

```python
from omninode_bridge.ci import (
    generate_python_ci,
    WorkflowGenerator,
    WorkflowValidator
)

# Generate a Python CI workflow
workflow = generate_python_ci(
    name="My Python CI",
    python_versions=["3.11", "3.12"],
    test_command="pytest --cov=src",
    coverage_threshold=80
)

# Convert to YAML
generator = WorkflowGenerator()
yaml_content = generator.generate_yaml(workflow)

# Validate workflow
validator = WorkflowValidator()
result = validator.validate(workflow)
print(f"Valid: {result.is_valid}")
```

### CLI Usage

```bash
# List available templates
omninode-workflow template list

# Generate workflow from template
omninode-workflow generate from-template python_ci \
    --output .github/workflows/ci.yml \
    --python-versions "3.11,3.12" \
    --coverage-threshold 85

# Validate workflow file
omninode-workflow validate file .github/workflows/ci.yml

# Validate all workflows in directory
omninode-workflow validate directory .github/workflows/ --recursive
```

## 🏗️ Architecture

### Core Components

```
src/omninode_bridge/ci/
├── models/                 # Pydantic models
│   ├── workflow.py        # Core workflow models
│   └── github_actions.py  # GitHub Actions specific models
├── generators/            # YAML generation
│   └── workflow_generator.py
├── validators/            # Workflow validation
│   └── workflow_validator.py
├── templates/             # Pre-built templates
│   ├── templates.py       # Template implementations
│   ├── python-ci.yml     # Example templates
│   └── docker-build.yml
├── cli/                   # Command-line interface
│   ├── main.py           # CLI implementation
│   └── workflow_cli.py   # Entry point script
├── examples/              # Demo and examples
│   └── demo.py           # Comprehensive demo
├── exceptions.py          # Custom exceptions
└── __init__.py           # Public API
```

### Model Hierarchy

```
WorkflowConfig
├── name: str
├── on: Union[EventType, List[EventType], Dict]
├── jobs: Dict[str, WorkflowJob]
└── env, permissions, defaults, concurrency

WorkflowJob
├── runs_on: Union[str, List[str]]
├── steps: List[WorkflowStep]
├── strategy: Optional[MatrixStrategy]
└── needs, permissions, environment, etc.

WorkflowStep
├── name: str
├── uses: Optional[str] (for actions)
├── run: Optional[str] (for shell commands)
└── with_, env, if_, continue_on_error, etc.
```

## 📋 Templates

### Available Templates

| Template | Description | Use Case |
|----------|-------------|----------|
| `python_ci` | Python CI with testing, linting, coverage | Python projects |
| `docker_build` | Docker build and push with multi-platform | Containerized applications |
| `security_scan` | Security scanning (bandit, safety, semgrep) | Security auditing |
| `release` | Release workflow with PyPI publishing | Package releases |
| `performance_test` | Performance testing with benchmarks | Performance monitoring |
| `multi_os` | Multi-OS testing (Ubuntu, Windows, macOS) | Cross-platform projects |

### Template Usage

```python
from omninode_bridge.ci import create_template

# Generate Python CI workflow
workflow = create_template("python_ci",
    name="My CI",
    python_versions=["3.11", "3.12"],
    test_command="pytest --cov=src",
    lint_commands=["ruff check", "black --check", "mypy"],
    coverage_threshold=85
)

# Generate Docker build workflow
workflow = create_template("docker_build",
    name="Docker Build",
    image_name="myorg/myapp",
    platforms=["linux/amd64"],  # Current CI/CD configuration
    push_to_registry=True
)
```

## 🔍 Validation System

### Validation Rules

The system includes comprehensive validation rules:

- **Structure Validation** (`WS001`): Basic workflow structure
- **Job Dependencies** (`JD001`): Valid job dependency references
- **Circular Dependencies** (`CD001`): Detect circular job dependencies
- **Action Versions** (`AV001`): Check for outdated action versions
- **Security** (`SEC001`): Security best practices and vulnerability detection
- **Performance** (`PERF001`): Performance optimization recommendations

### Custom Validation Rules

```python
from omninode_bridge.ci.validators.workflow_validator import ValidationRule, ValidationIssue, ValidationSeverity

class CustomRule(ValidationRule):
    def __init__(self):
        super().__init__("CUSTOM001", "Custom validation rule")

    def validate(self, workflow):
        issues = []
        # Add custom validation logic
        if some_condition:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Custom validation message",
                location="workflow.jobs.test",
                suggestion="Consider doing X instead"
            ))
        return issues

# Add to validator
validator = WorkflowValidator()
validator.add_rule(CustomRule())
```

## 🛠️ Advanced Usage

### Custom Workflow Builder

```python
from omninode_bridge.ci import WorkflowBuilder, WorkflowJob, WorkflowStep

builder = WorkflowBuilder("Custom CI")

# Add triggers
builder.add_trigger("push", branches=["main"])
builder.add_trigger("pull_request")

# Create custom job
steps = [
    WorkflowStep(name="Checkout", uses="actions/checkout@v4"),
    WorkflowStep(name="Setup", uses="actions/setup-python@v5"),
    WorkflowStep(name="Test", run="pytest")
]

job = WorkflowJob(
    name="Test",
    runs_on="ubuntu-latest",
    steps=steps
)

builder.add_job("test", job)
workflow = builder.build()
```

### YAML Round-Trip

```python
from omninode_bridge.ci import WorkflowConfig

# Load from YAML file
workflow = WorkflowConfig.from_yaml_file("workflow.yml")

# Modify programmatically
workflow.jobs["test"].timeout_minutes = 30

# Save back to YAML
workflow.to_yaml_file("updated-workflow.yml")
```

### Error Handling

```python
from omninode_bridge.ci import (
    WorkflowGenerationError,
    WorkflowValidationError,
    ErrorContext
)

try:
    with ErrorContext("workflow generation", "my-workflow.yml"):
        workflow = create_template("python_ci")
        generator = WorkflowGenerator()
        yaml_content = generator.generate_yaml(workflow)

except WorkflowGenerationError as e:
    print(f"Generation failed: {e}")
    print(f"Details: {e.details}")
except WorkflowValidationError as e:
    print(f"Validation failed: {e}")
    for error in e.validation_errors:
        print(f"  - {error}")
```

## 🔧 CLI Reference

### Generate Commands

```bash
# Generate from template
omninode-workflow generate from-template TEMPLATE_NAME \
    --output FILE \
    [--name NAME] \
    [--python-versions VERSIONS] \
    [--test-command COMMAND] \
    [--config CONFIG_FILE] \
    [--dry-run]

# Generate from model file
omninode-workflow generate from-model MODEL_FILE \
    --output FILE \
    [--format json|yaml] \
    [--dry-run]
```

### Validate Commands

```bash
# Validate single file
omninode-workflow validate file WORKFLOW_FILE \
    [--strict] \
    [--rules RULE_IDS] \
    [--exclude-rules RULE_IDS] \
    [--format text|json]

# Validate directory
omninode-workflow validate directory DIRECTORY \
    [--pattern "*.yml,*.yaml"] \
    [--recursive] \
    [--strict] \
    [--summary]
```

### Template Commands

```bash
# List templates
omninode-workflow template list [--detailed]

# Show template output
omninode-workflow template show TEMPLATE_NAME [--format yaml|json]
```

### Convert Commands

```bash
# Convert workflow to model
omninode-workflow convert to-model WORKFLOW_FILE \
    --output FILE \
    [--format json|yaml]

# Convert model to workflow
omninode-workflow convert to-yaml MODEL_FILE \
    --output FILE \
    [--format json|yaml]
```

## 🧪 Testing

Run the demo to see the system in action:

```bash
python src/omninode_bridge/ci/examples/demo.py
```

## 🤝 Contributing

The CI workflow system is designed to be extensible:

1. **Adding Templates**: Create new templates in `templates/templates.py`
2. **Custom Validation Rules**: Extend the validation system with new rules
3. **GitHub Actions Models**: Add new action models in `models/github_actions.py`
4. **CLI Commands**: Extend the CLI with new commands

## 📄 License

This project is part of the OmniNode Bridge system and follows the same MIT license.

## 🔗 Related

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [PyYAML Documentation](https://pyyaml.org/)
