# Workflow System Developer Guide

This guide provides essential information for developers working with the OmniNode Bridge workflow system.

## Overview

The workflow system consists of two main components:
1. **GitHub Actions Workflows** - Standard CI/CD pipelines
2. **Pydantic CI System** - Type-safe workflow generation and validation

## How CI Workflows Work

### Workflow Structure

```
.github/
├── workflows/              # Main workflow files
│   ├── claude-code-review.yml  # Claude AI code review
│   └── claude.yml              # General Claude workflow
└── workflows-generated/    # Auto-generated workflows
    ├── comprehensive-ci.yml    # Full CI/CD pipeline
    ├── python-ci.yml          # Python-specific CI
    ├── docker-build.yml       # Docker build/push
    ├── security-scan.yml      # Security scanning
    └── simple-test.yml        # Basic testing
```

### Workflow Types

| Workflow | Purpose | Triggers |
|----------|---------|----------|
| `claude-code-review.yml` | AI-powered code review | Pull requests |
| `comprehensive-ci.yml` | Full CI/CD with testing, linting, building | Push to main/develop, PRs |
| `python-ci.yml` | Python testing across versions | Push/PR |
| `docker-build.yml` | Container builds with registry push | Main branch pushes |
| `security-scan.yml` | Security vulnerability scanning | Scheduled/manual |

## Pydantic CI System Usage

### Quick Start

```bash
# Install CLI
pip install -e .

# List available templates
omninode-workflow template list

# Generate Python CI workflow
omninode-workflow generate from-template python_ci \
    --output .github/workflows/my-ci.yml \
    --python-versions "3.11,3.12" \
    --coverage-threshold 85

# Validate workflow
omninode-workflow validate file .github/workflows/my-ci.yml
```

### Python API

```python
from omninode_bridge.ci import generate_python_ci, WorkflowGenerator

# Create workflow programmatically
workflow = generate_python_ci(
    name="My Python CI",
    python_versions=["3.11", "3.12"],
    test_command="pytest --cov=src",
    coverage_threshold=80
)

# Generate YAML
generator = WorkflowGenerator()
yaml_content = generator.generate_yaml(workflow)

# Save to file
with open('.github/workflows/ci.yml', 'w') as f:
    f.write(yaml_content)
```

### Available Templates

| Template | Use Case | Command |
|----------|----------|---------|
| `python_ci` | Python testing/linting | `omninode-workflow generate from-template python_ci` |
| `docker_build` | Container builds | `omninode-workflow generate from-template docker_build` |
| `security_scan` | Security auditing | `omninode-workflow generate from-template security_scan` |

## Required Secrets and Configuration

### GitHub Repository Secrets

Add these secrets in GitHub Settings → Secrets and variables → Actions:

| Secret | Purpose | Required For |
|--------|---------|--------------|
| `CLAUDE_CODE_OAUTH_TOKEN` | Claude AI API access | Code review workflows |
| `GITHUB_TOKEN` | GitHub API access | Automated (provided by GitHub) |
| `CODECOV_TOKEN` | Code coverage reporting | Coverage workflows |
| `DOCKER_USERNAME` | Docker registry login | Docker build workflows |
| `DOCKER_PASSWORD` | Docker registry password | Docker build workflows |

### Setting Up Claude Code Review

1. Get Claude Code OAuth token from [Claude AI Console](https://console.anthropic.com/)
2. Add as repository secret: `CLAUDE_CODE_OAUTH_TOKEN`  # pragma: allowlist secret
3. Enable the workflow in `.github/workflows/claude-code-review.yml`

### Environment Variables

Common environment variables used in workflows:

```yaml
env:
  PYTHON_VERSION: '3.12'        # Default Python version
  NODE_VERSION: '20'            # Node.js version for frontend
  COVERAGE_THRESHOLD: '80'      # Minimum coverage percentage
```

## Examples

### Example 1: Basic Python CI

```bash
# Generate basic Python CI
omninode-workflow generate from-template python_ci \
    --name "Basic Python CI" \
    --python-versions "3.12" \
    --test-command "pytest tests/" \
    --output .github/workflows/basic-ci.yml
```

### Example 2: Multi-version Testing

```bash
# Generate multi-version Python CI
omninode-workflow generate from-template python_ci \
    --name "Multi-version CI" \
    --python-versions "3.11,3.12" \
    --test-command "pytest --cov=src --cov-report=xml" \
    --coverage-threshold 85 \
    --output .github/workflows/multi-ci.yml
```

### Example 3: Docker Build with Push

```bash
# Generate Docker workflow
omninode-workflow generate from-template docker_build \
    --name "Docker Build" \
    --image-name "myorg/myapp" \
    --dockerfile "Dockerfile" \
    --push-to-registry true \
    --output .github/workflows/docker.yml
```

### Example 4: Custom Workflow with Python API

```python
from omninode_bridge.ci import WorkflowBuilder, WorkflowJob, WorkflowStep

# Create custom workflow
builder = WorkflowBuilder("Custom CI")
builder.add_trigger("push", branches=["main"])
builder.add_trigger("pull_request")

# Define steps
steps = [
    WorkflowStep(name="Checkout", uses="actions/checkout@v4"),
    WorkflowStep(name="Setup Python", uses="actions/setup-python@v5",
                with_={"python-version": "3.12"}),
    WorkflowStep(name="Install deps", run="pip install -e .[dev]"),
    WorkflowStep(name="Run tests", run="pytest tests/"),
    WorkflowStep(name="Security scan", run="bandit -r src/")
]

# Create job
job = WorkflowJob(name="test", runs_on="ubuntu-latest", steps=steps)
builder.add_job("test", job)

# Generate and save
workflow = builder.build()
generator = WorkflowGenerator()
yaml_content = generator.generate_yaml(workflow)

with open('.github/workflows/custom.yml', 'w') as f:
    f.write(yaml_content)
```

## Troubleshooting Guide

### Common Issues

#### 1. Workflow Not Triggering

**Problem**: Workflow doesn't run on push/PR
**Solutions**:
- Check workflow file is in `.github/workflows/`
- Verify YAML syntax: `omninode-workflow validate file workflow.yml`
- Check branch names in trigger configuration
- Ensure workflow is enabled in repository settings

#### 2. Claude Code Review Not Working

**Problem**: Claude review doesn't post comments
**Solutions**:
- Verify `CLAUDE_CODE_OAUTH_TOKEN` secret is set
- Check token permissions and validity
- Ensure PR has changed files in monitored paths
- Review workflow logs for API errors

#### 3. Permission Denied Errors

**Problem**: Workflow fails with permission errors
**Solutions**:
- Check repository permissions in workflow file:
  ```yaml
  permissions:
    contents: read
    pull-requests: write
    issues: write
  ```
- Verify token has required scopes
- For organization repos, check organization policies

#### 4. Python Version Issues

**Problem**: Tests fail due to Python version mismatch
**Solutions**:
- Update Python version in workflow:
  ```yaml
  strategy:
    matrix:
      python-version: ["3.11", "3.12"]
  ```
- Check `pyproject.toml` Python requirements
- Use compatible package versions

#### 5. Docker Build Failures

**Problem**: Docker build or push fails
**Solutions**:
- Verify Dockerfile exists and is valid
- Check Docker registry credentials
- Ensure sufficient disk space for builds
- Review build context and `.dockerignore`

#### 6. Coverage Reports Not Working

**Problem**: Coverage reports missing or incorrect
**Solutions**:
- Install coverage tools: `pip install pytest-cov`
- Use correct coverage command: `pytest --cov=src --cov-report=xml`
- Check file paths in coverage configuration
- Verify `CODECOV_TOKEN` if using Codecov

### Debugging Workflows

#### View Workflow Logs
1. Go to GitHub repository → Actions tab
2. Click on failed workflow run
3. Click on failed job
4. Expand failed step to see detailed logs

#### Local Testing
```bash
# Validate workflow locally
omninode-workflow validate file .github/workflows/ci.yml

# Test Python CI commands locally
pip install -e .[dev]
ruff check .
black --check .
mypy src/
pytest tests/ --cov=src
```

#### Enable Debug Logging
Add to workflow environment:
```yaml
env:
  ACTIONS_STEP_DEBUG: true
  RUNNER_DEBUG: 1
```

### Performance Issues

#### Slow Workflow Execution
- Use matrix strategy for parallel jobs
- Cache dependencies:
  ```yaml
  - uses: actions/setup-python@v5
    with:
      python-version: '3.12'
      cache: 'pip'
  ```
- Optimize Docker builds with multi-stage builds
- Use workflow concurrency controls

#### Resource Limits
- Monitor workflow usage in repository insights
- Use self-hosted runners for heavy workloads
- Optimize test suites to reduce execution time

### Security Issues

#### Token Exposure
- Never commit secrets to repository
- Use repository secrets for sensitive data
- Rotate tokens regularly
- Use minimal required permissions

#### Dependency Vulnerabilities
- Enable Dependabot alerts
- Use security scanning workflows
- Keep dependencies updated
- Review third-party actions before use

## CLI Reference

### Generate Commands
```bash
# From template
omninode-workflow generate from-template <template> --output <file>

# From model file
omninode-workflow generate from-model <model-file> --output <file>
```

### Validate Commands
```bash
# Single file
omninode-workflow validate file <workflow-file>

# Directory
omninode-workflow validate directory <directory> --recursive
```

### Template Commands
```bash
# List templates
omninode-workflow template list

# Show template
omninode-workflow template show <template>
```

### Convert Commands
```bash
# Workflow to model
omninode-workflow convert to-model <workflow-file> --output <file>

# Model to workflow
omninode-workflow convert to-yaml <model-file> --output <file>
```

## Best Practices

### Workflow Design
- Use descriptive workflow and job names
- Set appropriate timeouts to prevent hanging
- Use matrix strategies for multiple versions/platforms
- Cache dependencies when possible
- Use specific action versions (not `@main`)

### Security
- Minimize permissions for each workflow
- Use repository secrets for sensitive data
- Pin action versions to prevent supply chain attacks
- Review third-party actions before use
- Enable branch protection rules

### Performance
- Cache dependencies between runs
- Use parallel jobs when possible
- Optimize Docker builds with multi-stage builds
- Set appropriate timeouts
- Use concurrency controls to prevent multiple runs

### Maintenance
- Regularly update action versions
- Monitor workflow success rates
- Keep dependencies updated
- Review and clean up unused workflows
- Document custom workflows and their purpose

## Getting Help

- **CI System Documentation**: `src/omninode_bridge/ci/README.md`
- **GitHub Actions Docs**: https://docs.github.com/en/actions
- **Pydantic Documentation**: https://docs.pydantic.dev/
- **Issue Reporting**: Create GitHub issue with workflow logs and configuration
