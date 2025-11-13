# Developer Onboarding Guide

## Welcome to OmniNode Bridge! 🚀

This guide will get you up and running with the OmniNode Bridge development environment quickly and efficiently. Whether you're a new team member or an open-source contributor, this guide provides everything you need to start developing.

## Table of Contents

- [Quick Start](#quick-start)
- [Development Environment Setup](#development-environment-setup)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Contributing Guidelines](#contributing-guidelines)
- [Release Process](#release-process)
- [Troubleshooting](#troubleshooting)
- [Resources & References](#resources--references)

## Quick Start

### 🔧 Prerequisites

Before you begin, ensure you have the following installed:

```bash
# Required Software
Python 3.11+
Docker 20.10+
Docker Compose 2.0+
Git 2.30+
Node.js 18+ (for tooling)

# Required Python Tools
poetry >= 1.4.0
pytest >= 7.0
black >= 22.0
ruff >= 0.0.254
mypy >= 1.0
pre-commit >= 3.0
```

### ⚡ 5-Minute Setup

```bash
# 1. Clone the repository
git clone https://github.com/OmniNode-ai/omninode_bridge.git
cd omninode_bridge

# 2. Install dependencies
poetry install --with dev

# 3. Setup pre-commit hooks
poetry run pre-commit install

# 4. Start development services
docker-compose -f docker-compose.dev.yml up -d

# 5. Run tests to verify setup
poetry run pytest tests/unit/ -v

# 6. Check code quality
poetry run black . && poetry run ruff check .

# ✅ You're ready to develop!
```

## Development Environment Setup

### Local Development Infrastructure

Our development environment uses Docker to provide consistent, isolated services:

```yaml
# docker-compose.dev.yml services:
services:
  postgres:      # Database (localhost:5432)
  redis:         # Cache (localhost:6379)
  kafka:         # Message broker (localhost:9092)
  zookeeper:     # Kafka coordination
  prometheus:    # Metrics (localhost:9090)
  grafana:       # Monitoring (localhost:3000)
```

### Environment Configuration

Create a `.env` file for local development:

```bash
# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=omninode_bridge
POSTGRES_USER=bridge_user
POSTGRES_PASSWORD=bridge_password

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=redis_password

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# Service Configuration
HOOK_RECEIVER_HOST=localhost
HOOK_RECEIVER_PORT=8001
MODEL_METRICS_HOST=localhost
MODEL_METRICS_PORT=8002
WORKFLOW_COORDINATOR_HOST=localhost
WORKFLOW_COORDINATOR_PORT=8003

# Development Settings
DEBUG=true
LOG_LEVEL=DEBUG
ENVIRONMENT=development
```

### IDE Configuration

#### Visual Studio Code

Recommended extensions and settings:

```json
// .vscode/extensions.json
{
    "recommendations": [
        "ms-python.python",
        "ms-python.black-formatter",
        "charliermarsh.ruff",
        "ms-python.mypy-type-checker",
        "ms-python.pytest",
        "redhat.vscode-yaml",
        "ms-vscode.docker"
    ]
}
```

```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "files.exclude": {
        "**/__pycache__": true,
        "**/.pytest_cache": true,
        "**/.mypy_cache": true
    }
}
```

#### PyCharm

```python
# PyCharm configuration checklist:
✅ Set Python interpreter to Poetry virtual environment
✅ Enable Black as code formatter
✅ Configure Ruff as linter
✅ Set pytest as test runner
✅ Enable Docker integration
✅ Configure database connections for PostgreSQL
```

## Project Structure

Understanding the codebase organization:

```
omninode_bridge/
├── src/omninode_bridge/           # Main application code
│   ├── models/                    # Data models and schemas
│   │   ├── events.py             # Event definitions
│   │   ├── metrics.py            # Metrics models
│   │   └── workflows.py          # Workflow definitions
│   ├── services/                  # Core services
│   │   ├── hook_receiver.py      # Webhook handling service
│   │   ├── model_metrics_api.py  # AI model metrics service
│   │   ├── workflow_coordinator.py # Workflow coordination
│   │   ├── kafka_client.py       # Message broker client
│   │   └── postgres_client.py    # Database client
│   ├── security/                  # Security components
│   │   ├── authentication.py     # Auth handlers
│   │   ├── encryption.py         # Encryption utilities
│   │   └── rate_limiting.py      # Rate limiting
│   └── utils/                     # Utility functions
│       ├── logging.py            # Structured logging
│       ├── metrics.py            # Metrics collection
│       └── validation.py        # Input validation
├── tests/                         # Test suite
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   ├── performance/              # Performance tests
│   └── conftest.py               # Test configuration
├── docs/                          # Documentation
│   ├── api/                      # API documentation
│   ├── architecture/             # Architecture docs
│   ├── deployment/               # Deployment guides
│   ├── developers/               # Developer guides
│   └── operations/               # Operational docs
├── monitoring/                    # Monitoring configurations
│   ├── prometheus.yml            # Metrics collection
│   ├── alerts/                   # Alert rules
│   └── grafana/                  # Dashboard configs
└── scripts/                       # Development scripts
    ├── setup-dev.sh              # Development setup
    ├── run-tests.sh              # Test execution
    └── deploy.sh                 # Deployment script
```

### Key Components

| Component | Purpose | Port | Dependencies |
|-----------|---------|------|--------------|
| **HookReceiver** | Webhook handling and event processing | 8001 | PostgreSQL, Kafka |
| **ModelMetrics** | AI model performance tracking | 8002 | PostgreSQL, Redis |
| **WorkflowCoordinator** | Multi-service workflow orchestration | 8003 | PostgreSQL, Kafka |
| **PostgreSQL** | Primary data storage | 5432 | - |
| **Redis** | Caching and session storage | 6379 | - |
| **Kafka** | Event streaming and message queuing | 9092 | Zookeeper |

## Development Workflow

### Branch Strategy

We follow **Git Flow** with feature branches:

```bash
# Main branches
main          # Production-ready code
develop       # Integration branch for features

# Supporting branches
feature/*     # New features (branch from develop)
hotfix/*      # Critical fixes (branch from main)
release/*     # Release preparation (branch from develop)
```

### Feature Development Process

```bash
# 1. Create feature branch
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name

# 2. Implement feature with tests
# ... develop and test your changes ...

# 3. Run quality checks
poetry run black .                 # Format code
poetry run ruff check --fix .      # Lint and auto-fix
poetry run mypy src/               # Type checking
poetry run pytest                 # Run tests

# 4. Commit with conventional commits
git add .
git commit -m "feat: add webhook retry mechanism with exponential backoff

- Implement RetryPolicy class with configurable backoff
- Add circuit breaker pattern for external service calls
- Include comprehensive unit tests and integration tests
- Update API documentation with retry behavior

Closes #123"

# 5. Push and create PR
git push origin feature/your-feature-name
# Create pull request via GitHub UI or CLI
```

### Daily Development Commands

Create these aliases for efficient development:

```bash
# Add to your ~/.bashrc or ~/.zshrc
alias bridge-start="docker-compose -f docker-compose.dev.yml up -d"
alias bridge-stop="docker-compose -f docker-compose.dev.yml down"
alias bridge-logs="docker-compose -f docker-compose.dev.yml logs -f"
alias bridge-test="poetry run pytest"
alias bridge-format="poetry run black . && poetry run ruff check --fix ."
alias bridge-quality="poetry run black . && poetry run ruff check . && poetry run mypy src/"
alias bridge-coverage="poetry run pytest --cov=omninode_bridge --cov-report=html"
```

## Coding Standards

### Python Style Guide

We follow **PEP 8** with these specific guidelines:

#### Naming Conventions
```python
# Classes: PascalCase
class HookReceiver:
    pass

# Functions and variables: snake_case
def process_webhook_event():
    user_id = get_user_id()

# Constants: UPPER_SNAKE_CASE
MAX_RETRY_ATTEMPTS = 3
DEFAULT_TIMEOUT = 30

# Private methods: leading underscore
def _internal_helper():
    pass
```

#### Type Hints
Type hints are **required** for all public functions:

```python
from typing import Dict, List, Optional, Union
from datetime import datetime

def process_event(
    event_data: Dict[str, Any],
    event_type: str,
    timestamp: Optional[datetime] = None
) -> Dict[str, Union[str, int, bool]]:
    """Process webhook event with comprehensive validation.

    Args:
        event_data: Raw event payload from webhook
        event_type: Classification of event (hook, metric, workflow)
        timestamp: Event timestamp (defaults to current time)

    Returns:
        Processing result with status, message, and metadata

    Raises:
        ValidationError: Invalid event data or type
        ProcessingError: Event processing failed
    """
    pass
```

#### Documentation Standards
Use **Google-style docstrings**:

```python
def calculate_response_metrics(
    response_times: List[float],
    error_count: int,
    total_requests: int
) -> Dict[str, float]:
    """Calculate comprehensive response metrics for monitoring.

    This function computes key performance indicators including:
    - Average, median, and 95th percentile response times
    - Error rate and success rate percentages
    - Throughput and capacity metrics

    Args:
        response_times: List of response times in milliseconds
        error_count: Number of failed requests in the period
        total_requests: Total number of requests processed

    Returns:
        Dictionary containing calculated metrics:
        - avg_response_time: Average response time in ms
        - median_response_time: Median response time in ms
        - p95_response_time: 95th percentile response time in ms
        - error_rate: Percentage of failed requests (0.0-1.0)
        - success_rate: Percentage of successful requests (0.0-1.0)
        - throughput: Requests per second

    Raises:
        ValueError: If response_times is empty or negative values provided

    Example:
        >>> response_times = [100, 150, 200, 120, 180]
        >>> metrics = calculate_response_metrics(response_times, 1, 100)
        >>> print(f"Error rate: {metrics['error_rate']:.2%}")
        Error rate: 1.00%
    """
    pass
```

#### Error Handling
Use specific exception types and structured error information:

```python
from omninode_bridge.exceptions import (
    ValidationError,
    ProcessingError,
    ExternalServiceError
)

class WebhookProcessor:
    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Validate input
            if not self._validate_payload(payload):
                raise ValidationError(
                    message="Invalid webhook payload structure",
                    details={"missing_fields": ["event_type", "timestamp"]},
                    error_code="INVALID_PAYLOAD"
                )

            # Process the webhook
            result = self._execute_processing(payload)

        except ValidationError:
            # Re-raise validation errors
            raise
        except ExternalServiceError as e:
            # Handle external service failures
            logger.error(f"External service failure: {e}")
            raise ProcessingError(
                message="Webhook processing failed due to external dependency",
                details={"service": e.service_name, "error": str(e)},
                error_code="EXTERNAL_DEPENDENCY_FAILURE"
            ) from e
        except Exception as e:
            # Handle unexpected errors
            logger.exception("Unexpected error during webhook processing")
            raise ProcessingError(
                message="Unexpected processing error",
                details={"error": str(e)},
                error_code="UNEXPECTED_ERROR"
            ) from e

        return result
```

### Code Quality Tools

#### Pre-commit Configuration
Our `.pre-commit-config.yaml` enforces quality standards:

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict

  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.0.254
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

## Testing Guidelines

### Testing Strategy

We use a comprehensive testing approach:

```
Testing Pyramid:
├── Unit Tests (70%)        # Fast, isolated component tests
├── Integration Tests (20%) # Service interaction tests
├── Performance Tests (5%)  # Load and stress tests
└── E2E Tests (5%)         # Full system workflow tests
```

### Unit Testing Standards

```python
# tests/unit/services/test_hook_receiver.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from omninode_bridge.services.hook_receiver import HookReceiver
from omninode_bridge.models.events import WebhookEvent

class TestHookReceiver:
    """Comprehensive unit tests for HookReceiver service."""

    @pytest.fixture
    def mock_db_client(self):
        """Mock database client for testing."""
        mock = AsyncMock()
        mock.store_event.return_value = {"id": "test-event-id"}
        return mock

    @pytest.fixture
    def mock_kafka_client(self):
        """Mock Kafka client for testing."""
        mock = AsyncMock()
        mock.publish_event.return_value = True
        return mock

    @pytest.fixture
    def hook_receiver(self, mock_db_client, mock_kafka_client):
        """HookReceiver instance with mocked dependencies."""
        config = {
            "database": {"connection_pool_size": 10},
            "kafka": {"bootstrap_servers": "localhost:9092"}
        }
        receiver = HookReceiver(config)
        receiver.db_client = mock_db_client
        receiver.kafka_client = mock_kafka_client
        return receiver

    @pytest.mark.asyncio
    async def test_process_webhook_success(self, hook_receiver, mock_db_client, mock_kafka_client):
        """Test successful webhook processing."""
        # Arrange
        webhook_data = {
            "event_type": "user.created",
            "payload": {"user_id": "123", "email": "test@example.com"},
            "timestamp": "2025-09-22T10:00:00Z"
        }

        # Act
        result = await hook_receiver.process_webhook(webhook_data)

        # Assert
        assert result["status"] == "success"
        assert result["event_id"] is not None

        # Verify database interaction
        mock_db_client.store_event.assert_called_once()
        stored_event_data = mock_db_client.store_event.call_args[0][0]
        assert stored_event_data.event_type == "user.created"

        # Verify Kafka publishing
        mock_kafka_client.publish_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_webhook_validation_error(self, hook_receiver):
        """Test webhook processing with invalid data."""
        # Arrange
        invalid_data = {"invalid": "structure"}

        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            await hook_receiver.process_webhook(invalid_data)

        assert "event_type" in str(exc_info.value)
        assert exc_info.value.error_code == "INVALID_PAYLOAD"

    @pytest.mark.asyncio
    async def test_process_webhook_database_failure(self, hook_receiver, mock_db_client):
        """Test webhook processing with database failure."""
        # Arrange
        mock_db_client.store_event.side_effect = Exception("Database connection failed")
        webhook_data = {
            "event_type": "test.event",
            "payload": {"test": "data"},
            "timestamp": "2025-09-22T10:00:00Z"
        }

        # Act & Assert
        with pytest.raises(ProcessingError) as exc_info:
            await hook_receiver.process_webhook(webhook_data)

        assert "database" in str(exc_info.value).lower()
        assert exc_info.value.error_code == "DATABASE_ERROR"

    @pytest.mark.parametrize("event_type,expected_topic", [
        ("user.created", "user_events"),
        ("order.completed", "order_events"),
        ("system.health", "system_events")
    ])
    @pytest.mark.asyncio
    async def test_event_routing(self, hook_receiver, event_type, expected_topic):
        """Test event routing to correct Kafka topics."""
        # Arrange
        webhook_data = {
            "event_type": event_type,
            "payload": {"test": "data"},
            "timestamp": "2025-09-22T10:00:00Z"
        }

        # Act
        await hook_receiver.process_webhook(webhook_data)

        # Assert
        hook_receiver.kafka_client.publish_event.assert_called_once()
        call_args = hook_receiver.kafka_client.publish_event.call_args
        assert call_args[1]["topic"] == expected_topic
```

### Integration Testing

```python
# tests/integration/test_webhook_flow.py
import pytest
import asyncio
from testcontainers.postgres import PostgresContainer
from testcontainers.kafka import KafkaContainer

class TestWebhookIntegrationFlow:
    """Integration tests for complete webhook processing flow."""

    @pytest.fixture(scope="class")
    def postgres_container(self):
        """PostgreSQL container for integration testing."""
        with PostgresContainer("postgres:15") as postgres:
            yield postgres

    @pytest.fixture(scope="class")
    def kafka_container(self):
        """Kafka container for integration testing."""
        with KafkaContainer() as kafka:
            yield kafka

    @pytest.mark.asyncio
    async def test_end_to_end_webhook_processing(self, postgres_container, kafka_container):
        """Test complete webhook processing from API to data storage."""
        # Setup real services with test containers
        # ... implementation details ...

        # Verify end-to-end processing
        assert webhook_response.status_code == 200
        assert event_stored_in_database
        assert event_published_to_kafka
```

### Performance Testing

```python
# tests/performance/test_load_scenarios.py
import pytest
import asyncio
from locust import HttpUser, task, between

class WebhookLoadTest(HttpUser):
    """Load testing for webhook endpoints."""

    wait_time = between(1, 3)

    @task(3)
    def process_webhook(self):
        """Simulate webhook processing under load."""
        payload = {
            "event_type": "user.action",
            "payload": {"user_id": f"user_{self.environment.runner.user_count}"},
            "timestamp": "2025-09-22T10:00:00Z"
        }

        response = self.client.post("/api/v1/webhooks", json=payload)
        assert response.status_code == 200
```

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run specific test categories
poetry run pytest tests/unit/           # Unit tests only
poetry run pytest tests/integration/    # Integration tests only
poetry run pytest tests/performance/    # Performance tests only

# Run with coverage
poetry run pytest --cov=omninode_bridge --cov-report=html

# Run tests matching pattern
poetry run pytest -k "test_webhook" -v

# Run tests with markers
poetry run pytest -m "slow" -v          # Run slow tests
poetry run pytest -m "not slow" -v      # Skip slow tests

# Parallel testing
poetry run pytest -n auto               # Auto-detect CPU cores
poetry run pytest -n 4                  # Use 4 workers
```

## Contributing Guidelines

### Getting Started with Contributions

1. **Check Existing Issues**: Look for issues labeled `good-first-issue` or `help-wanted`
2. **Discuss First**: For major changes, create an issue to discuss the approach
3. **Fork and Branch**: Fork the repository and create a feature branch
4. **Follow Standards**: Adhere to our coding standards and testing requirements
5. **Document Changes**: Update documentation for user-facing changes

### Pull Request Process

#### PR Checklist
Before submitting a pull request, ensure:

- [ ] **Code Quality**
  - [ ] Code follows project style guidelines
  - [ ] All pre-commit hooks pass
  - [ ] No new linting or type checking errors
  - [ ] Functions and classes have proper docstrings

- [ ] **Testing**
  - [ ] All existing tests pass
  - [ ] New features include comprehensive unit tests
  - [ ] Integration tests updated for service interactions
  - [ ] Test coverage remains above 90%

- [ ] **Documentation**
  - [ ] API documentation updated for endpoint changes
  - [ ] README updated for configuration changes
  - [ ] Architecture docs updated for design changes
  - [ ] Developer docs updated for workflow changes

- [ ] **Security**
  - [ ] No sensitive data in code or commits
  - [ ] Security implications considered and documented
  - [ ] Input validation implemented for new endpoints
  - [ ] Authentication/authorization properly handled

#### PR Description Template

```markdown
## Description
Brief description of the changes and their purpose.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Changes Made
- List specific changes made
- Include any new dependencies
- Mention configuration changes

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed
- [ ] Performance impact assessed

## Documentation
- [ ] Code comments added/updated
- [ ] API documentation updated
- [ ] User documentation updated
- [ ] Developer documentation updated

## Checklist
- [ ] Self-review completed
- [ ] Code follows style guidelines
- [ ] Breaking changes documented
- [ ] Tests pass locally
- [ ] No merge conflicts
```

### Code Review Guidelines

#### For Contributors
- **Self-Review**: Review your own code before submitting
- **Small PRs**: Keep pull requests focused and manageable
- **Clear Commits**: Use conventional commit messages
- **Respond Promptly**: Address review feedback quickly

#### For Reviewers
- **Be Constructive**: Provide helpful, actionable feedback
- **Focus on Logic**: Review for correctness, not just style
- **Security-Minded**: Look for potential security issues
- **Performance-Aware**: Consider performance implications

## Release Process

### Release Types

We follow **Semantic Versioning** (semver):
- **Major** (x.0.0): Breaking changes
- **Minor** (x.y.0): New features, backwards compatible
- **Patch** (x.y.z): Bug fixes, backwards compatible

### Release Workflow

```bash
# 1. Create release branch
git checkout develop
git pull origin develop
git checkout -b release/v1.2.0

# 2. Update version information
# Update pyproject.toml version
# Update CHANGELOG.md
# Update API documentation versions

# 3. Run full test suite
poetry run pytest tests/
poetry run pytest tests/integration/ --slow
poetry run pytest tests/performance/

# 4. Create release PR
git add .
git commit -m "chore: prepare release v1.2.0"
git push origin release/v1.2.0
# Create PR: release/v1.2.0 → main

# 5. After PR approval and merge
git checkout main
git pull origin main
git tag -a v1.2.0 -m "Release version 1.2.0"
git push origin v1.2.0

# 6. Merge back to develop
git checkout develop
git merge main
git push origin develop
```

### Release Checklist

- [ ] **Version Updates**
  - [ ] `pyproject.toml` version updated
  - [ ] `__version__.py` files updated
  - [ ] API documentation versions updated
  - [ ] Docker image tags updated

- [ ] **Documentation**
  - [ ] CHANGELOG.md updated with release notes
  - [ ] Breaking changes documented
  - [ ] Migration guides written (if needed)
  - [ ] API documentation published

- [ ] **Testing**
  - [ ] Full test suite passes
  - [ ] Performance tests show no regressions
  - [ ] Security scans complete
  - [ ] Integration tests with dependencies pass

- [ ] **Deployment**
  - [ ] Production deployment tested in staging
  - [ ] Database migrations tested
  - [ ] Rollback procedure verified
  - [ ] Monitoring alerts updated

### Hotfix Process

For critical production issues:

```bash
# 1. Create hotfix branch from main
git checkout main
git pull origin main
git checkout -b hotfix/critical-security-fix

# 2. Implement minimal fix
# ... make necessary changes ...

# 3. Test thoroughly
poetry run pytest tests/
# Run specific tests for the fix

# 4. Update version (patch bump)
# Update version to next patch (e.g., 1.2.0 → 1.2.1)

# 5. Create hotfix PR
git add .
git commit -m "fix: resolve critical security vulnerability

- Fix SQL injection in webhook validation
- Add input sanitization for all user inputs
- Update security tests

Fixes #456"

# 6. Emergency review and merge
# Fast-track review process for critical fixes

# 7. Tag and deploy
git tag -a v1.2.1 -m "Hotfix version 1.2.1"
git push origin v1.2.1

# 8. Merge back to develop
git checkout develop
git merge main
git push origin develop
```

## Troubleshooting

### Common Development Issues

#### Environment Setup Problems

**Issue**: Poetry installation fails
```bash
# Solution: Update Poetry and clear cache
curl -sSL https://install.python-poetry.org | python3 -
poetry cache clear pypi --all
poetry install --with dev
```

**Issue**: Docker services won't start
```bash
# Check for port conflicts
netstat -tulpn | grep LISTEN

# Free up ports if needed
docker-compose -f docker-compose.dev.yml down
sudo lsof -ti:5432 | xargs kill -9  # Example for PostgreSQL

# Restart services
docker-compose -f docker-compose.dev.yml up -d
```

**Issue**: Pre-commit hooks failing
```bash
# Update pre-commit hooks
poetry run pre-commit autoupdate
poetry run pre-commit install --install-hooks

# Fix common issues
poetry run black .                    # Format code
poetry run ruff check --fix .         # Fix linting issues
poetry run mypy --install-types       # Install missing types
```

#### Testing Issues

**Issue**: Tests fail with database errors
```bash
# Reset test database
docker-compose -f docker-compose.dev.yml restart postgres
poetry run python -m omninode_bridge.database.reset_test_db
poetry run pytest tests/unit/
```

**Issue**: Integration tests are flaky
```bash
# Run tests with more verbose output
poetry run pytest tests/integration/ -v -s --tb=long

# Run specific failing test multiple times
poetry run pytest tests/integration/test_webhook_flow.py::test_end_to_end -v --count=10
```

#### Performance Issues

**Issue**: Slow local development
```bash
# Check system resources
docker stats
df -h

# Optimize Docker performance
docker system prune -f
docker volume prune -f

# Reduce resource usage
export POSTGRES_SHARED_BUFFERS=128MB
export JAVA_OPTS="-Xmx512m"  # For Kafka
```

### Debug Tools and Techniques

#### Logging Configuration
```python
# Enable debug logging
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Service-specific debugging
export LOG_LEVEL=DEBUG
export DEBUG_SQL=true
export DEBUG_KAFKA=true
```

#### Database Debugging
```bash
# Connect to development database
docker exec -it postgres psql -U bridge_user -d omninode_bridge

# Useful SQL commands
\dt                              # List tables
\d+ webhook_events              # Describe table
SELECT * FROM sessions LIMIT 5; # Query data
```

#### API Debugging
```bash
# Test API endpoints
curl -X POST http://localhost:8001/api/v1/webhooks \
  -H "Content-Type: application/json" \
  -d '{"event_type": "test", "payload": {"test": "data"}}'

# Check service health
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health

# View metrics
curl http://localhost:8001/metrics
```

### Getting Help

#### Internal Resources
- **Architecture Documentation**: `/docs/architecture/`
- **API Documentation**: `/docs/api/`
- **Operational Guides**: `/docs/operations/`
- **Security Guidelines**: `/docs/security/`

#### Development Tools
- **Code Quality**: Black, Ruff, MyPy, Pre-commit
- **Testing**: pytest, Coverage.py, Locust
- **Monitoring**: Prometheus, Grafana, Jaeger
- **Debugging**: Python debugger, Docker logs, PostgreSQL logs

#### Support Channels
- **GitHub Issues**: Report bugs and request features
- **GitHub Discussions**: Ask questions and share ideas
- **Wiki**: Browse community documentation
- **Security**: Report security issues privately

## Resources & References

### Essential Documentation
- [FastAPI Documentation](https://fastapi.tiangolo.com/) - Web framework
- [PostgreSQL Documentation](https://www.postgresql.org/docs/) - Database
- [Redis Documentation](https://redis.io/documentation) - Caching
- [Kafka Documentation](https://kafka.apache.org/documentation/) - Messaging
- [Docker Documentation](https://docs.docker.com/) - Containerization
- [Poetry Documentation](https://python-poetry.org/docs/) - Dependency management

### Architecture References
- [Microservices Patterns](https://microservices.io/patterns/) - Design patterns
- [12-Factor App](https://12factor.net/) - Application architecture
- [API Design Guidelines](https://github.com/microsoft/api-guidelines) - REST API best practices
- [Observability Patterns](https://observability.patterns.dev/) - Monitoring and logging

### Security Resources
- [OWASP Top 10](https://owasp.org/www-project-top-ten/) - Security vulnerabilities
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework) - Security standards
- [Python Security](https://python-security.readthedocs.io/) - Python-specific security

### Performance Resources
- [Python Performance Tips](https://wiki.python.org/moin/PythonSpeed/PerformanceTips) - Optimization techniques
- [Database Performance](https://use-the-index-luke.com/) - SQL optimization
- [Redis Performance](https://redis.io/docs/manual/performance/) - Caching optimization

---

**Welcome to the OmniNode Bridge team!** 🎉

This guide should get you productive quickly. If you have questions or suggestions for improvement, please create an issue or submit a pull request.

Happy coding! 🚀
