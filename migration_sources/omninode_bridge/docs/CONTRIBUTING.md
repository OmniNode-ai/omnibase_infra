# Contributing to OmniNode Bridge

**Thank you for your interest in contributing to OmniNode Bridge!**

This guide provides everything you need to know about contributing to the project, from setting up your development environment to submitting pull requests.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Workflow](#development-workflow)
3. [Code Standards](#code-standards)
4. [Testing Requirements](#testing-requirements)
5. [Documentation Requirements](#documentation-requirements)
6. [Pull Request Process](#pull-request-process)
7. [Code Review Guidelines](#code-review-guidelines)
8. [Community Guidelines](#community-guidelines)

---

## Getting Started

### Prerequisites

Before you begin contributing, ensure you have:

1. **Python 3.11+** installed
2. **Docker** and **Docker Compose v2+** (uses `docker compose` command)
3. **Poetry** for dependency management
4. **Git** for version control
5. A **GitHub account** with Personal Access Token (PAT)
6. **GitHub Personal Access Token (GH_PAT)** - Required for building Docker images with private dependencies

**Docker Build Requirements:**
- Docker Compose v2+ (not v1)
- Docker BuildKit enabled (default in Docker 23.0+)
- GitHub PAT with `read:packages` and `repo` permissions

**See**: [Docker Build Requirements in SETUP.md](./SETUP.md#docker-build-requirements) for detailed setup instructions.

### Development Environment Setup

```bash
# 1. Fork and clone the repository
git clone <your-fork-url>
cd omninode_bridge

# 2. Install dependencies
poetry install

# 3. Configure Kafka/Redpanda hostname (ONE-TIME)
echo "127.0.0.1 omninode-bridge-redpanda" | sudo tee -a /etc/hosts

# 4. Setup GitHub PAT for Docker builds (REQUIRED)
# See: https://github.com/settings/tokens
export GH_PAT=ghp_your_token_here...
# Required permissions: read:packages, repo (or read:org)

# 5. Start services (with GH_PAT configured)
docker compose up -d

# 6. Run database migrations
poetry run alembic upgrade head

# 7. Verify setup
poetry run pytest tests/ -v
```

**Full Setup Guide**: [docs/SETUP.md](./SETUP.md)

**Docker Build Guide**: [Docker Build Requirements](./SETUP.md#docker-build-requirements)

---

## Development Workflow

### Branch Strategy

**Main Branch**: `main` (or `master`)
- Always in a deployable state
- Protected, requires PR approval
- CI/CD must pass before merge

**Feature Branches**: `feature/<feature-name>`
- Branch from `main`
- One feature per branch
- Descriptive names (e.g., `feature/add-validation-middleware`)

**Bug Fix Branches**: `fix/<bug-name>`
- Branch from `main`
- One bug per branch
- Reference issue number (e.g., `fix/issue-123-connection-leak`)

**Documentation Branches**: `docs/<doc-name>`
- Branch from `main`
- Documentation-only changes

### Development Cycle

```bash
# 1. Create feature branch
git checkout -b feature/my-new-feature

# 2. Make changes
# ... develop your feature ...

# 3. Run tests locally
poetry run pytest tests/ -v

# 4. Format code
poetry run black src/ tests/
poetry run ruff check --fix src/ tests/

# 5. Type check
poetry run mypy src/

# 6. Commit changes
git add .
git commit -m "feat: add my new feature"

# 7. Push to your fork
git push origin feature/my-new-feature

# 8. Create Pull Request on GitHub
```

### Commit Message Convention

**Format**: `<type>(<scope>): <subject>`

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Test additions or modifications
- `chore`: Build/tooling changes

**Examples**:
```bash
feat(orchestrator): add workflow timeout handling
fix(database): resolve connection pool exhaustion
docs(api): update endpoint documentation
test(integration): add E2E workflow tests
refactor(reducer): simplify aggregation logic
```

---

## Code Standards

### Python Code Style

**Tool Stack**:
- **Black**: Code formatting (line length: 100)
- **Ruff**: Linting and import sorting
- **mypy**: Static type checking

**Run Before Commit**:
```bash
# Format code
poetry run black src/ tests/

# Lint and fix (includes import sorting)
poetry run ruff check --fix src/ tests/

# Type check
poetry run mypy src/

# Optional: Check import sorting specifically
poetry run ruff check --select I src/ tests/
```

### ONEX v2.0 Compliance

**All code must follow ONEX v2.0 standards**:

1. **Suffix-Based Naming**:
   ```python
   # ✅ Correct
   class NodeBridgeOrchestrator(NodeOrchestrator):
       pass

   class ModelBridgeState(BaseModel):
       pass

   class EnumWorkflowState(str, Enum):
       pass

   # ❌ Incorrect
   class BridgeOrchestratorNode(NodeOrchestrator):
       pass

   class BridgeStateModel(BaseModel):
       pass
   ```

2. **Method Signatures**:
   ```python
   # Orchestrator
   async def execute_orchestration(
       self,
       contract: ModelContractOrchestrator
   ) -> ModelStampResponseOutput:
       pass

   # Reducer
   async def execute_reduction(
       self,
       contract: ModelContractReducer
   ) -> ModelReducerOutputState:
       pass
   ```

3. **Error Handling**:
   ```python
   from omnibase_core.errors import OnexError, CoreErrorCode

   # ✅ Correct
   raise OnexError(
       code=CoreErrorCode.VALIDATION_ERROR,
       message="Invalid input data",
       details={"field": "content", "error": "must not be empty"}
   )

   # ❌ Incorrect
   raise ValueError("Invalid input")
   ```

### Type Annotations

**All functions must have type annotations**:
```python
# ✅ Correct
async def process_workflow(
    workflow_id: UUID,
    input_data: dict[str, Any],
    timeout: float = 60.0
) -> ModelStampResponseOutput:
    pass

# ❌ Incorrect
async def process_workflow(workflow_id, input_data, timeout=60.0):
    pass
```

### Documentation Strings

**All public functions/classes must have docstrings**:
```python
class NodeBridgeOrchestrator(NodeOrchestrator):
    """
    Bridge Orchestrator for stamping workflow coordination.

    Coordinates multi-step stamping workflows with FSM state management,
    service routing, and event publishing.

    Performance:
        - Standard workflow: <50ms
        - With OnexTree intelligence: <150ms
        - Throughput: 100+ workflows/second

    Attributes:
        container: ONEX dependency injection container
        fsm: Workflow FSM state tracker
        kafka_producer: Event publisher
    """

    async def execute_orchestration(
        self,
        contract: ModelContractOrchestrator
    ) -> ModelStampResponseOutput:
        """
        Execute stamping workflow with FSM state management.

        Args:
            contract: Orchestrator contract with workflow configuration.
                     Must include:
                     - correlation_id: UUID for workflow tracking
                     - input_data: Input data for stamping

        Returns:
            ModelStampResponseOutput with:
            - stamp_id: Unique stamp identifier
            - file_hash: BLAKE3 hash of content
            - stamped_content: Content with embedded stamp
            - stamp_metadata: Complete stamp metadata

        Raises:
            OnexError: If workflow execution fails
                      Error codes:
                      - VALIDATION_ERROR: Invalid input data
                      - OPERATION_FAILED: Workflow step failed

        Performance:
            - Target: <50ms for standard workflows
            - With OnexTree: <150ms

        Example:
            >>> contract = ModelContractOrchestrator(
            ...     correlation_id=uuid4(),
            ...     input_data={"content": "Hello"}
            ... )
            >>> result = await orchestrator.execute_orchestration(contract)
            >>> print(f"Stamp ID: {result.stamp_id}")
        """
        pass
```

### Pydantic v2 Models

**Use Pydantic v2 BaseModel**:
```python
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from uuid import UUID

class ModelBridgeState(BaseModel):
    """Bridge state model with O.N.E. v0.1 compliance."""

    state_id: UUID
    version: int = Field(..., ge=1)
    namespace: str = Field(..., min_length=1, max_length=200)
    metadata_version: str = Field(default="0.1")
    total_stamps: int = Field(default=0, ge=0)
    total_size_bytes: int = Field(default=0, ge=0)
    unique_file_types: set[str] = Field(default_factory=set)
    current_fsm_state: str
    created_at: datetime
    last_updated: datetime

    @field_validator('namespace')
    @classmethod
    def validate_namespace(cls, v: str) -> str:
        """Validate namespace format."""
        if not v or not v.replace('.', '').replace('_', '').replace('-', '').isalnum():
            raise ValueError("Invalid namespace format")
        return v

    class Config:
        """Pydantic v2 configuration."""
        json_schema_extra = {
            "example": {
                "state_id": "550e8400-e29b-41d4-a716-446655440000",
                "version": 1,
                "namespace": "omniclaude.docs",
                "metadata_version": "0.1",
                "total_stamps": 100,
                "total_size_bytes": 10485760,
                "unique_file_types": ["text/plain", "application/pdf"],
                "current_fsm_state": "AGGREGATING",
                "created_at": "2025-10-15T12:34:56Z",
                "last_updated": "2025-10-15T12:34:56Z"
            }
        }
```

---

## Testing Requirements

### Test Coverage Requirements

**Minimum Coverage**: 90% for new code
**Current Coverage**: 92.8% (target: maintain or improve)

**Coverage by Component**:
- Event schemas: 100% (required)
- Core nodes: >95% (required)
- Database layer: >90% (required)
- Integration tests: >85% (required)

### Test Organization

```
tests/
├── unit/                    # Unit tests (fast, isolated)
│   ├── nodes/
│   │   ├── test_orchestrator.py
│   │   └── test_reducer.py
│   ├── persistence/
│   │   └── test_connection_manager.py
│   └── models/
│       └── test_bridge_state.py
│
├── integration/             # Integration tests (slower, multiple components)
│   ├── test_orchestrator_reducer_flow.py
│   ├── test_kafka_event_publishing.py
│   └── test_database_persistence.py
│
├── performance/             # Performance tests
│   ├── test_hash_generation_performance.py
│   └── test_aggregation_throughput.py
│
└── e2e/                     # End-to-end tests
    └── test_complete_stamping_workflow.py
```

### Writing Tests

**Unit Test Example**:
```python
import pytest
from uuid import uuid4
from omninode_bridge.nodes.orchestrator.v1_0_0.node import NodeBridgeOrchestrator
from omnibase_core.models.contracts import ModelContractOrchestrator

class TestNodeBridgeOrchestrator:
    """Test suite for NodeBridgeOrchestrator."""

    @pytest.fixture
    def orchestrator(self, container):
        """Create orchestrator instance for testing."""
        return NodeBridgeOrchestrator(container)

    @pytest.mark.asyncio
    async def test_execute_orchestration_success(self, orchestrator):
        """Test successful orchestration execution."""
        # Arrange
        contract = ModelContractOrchestrator(
            correlation_id=uuid4(),
            input_data={"content": "test", "namespace": "test"}
        )

        # Act
        result = await orchestrator.execute_orchestration(contract)

        # Assert
        assert result.stamp_id is not None
        assert result.file_hash is not None
        assert result.workflow_state.value == "COMPLETED"
        assert result.processing_time_ms < 50  # Performance assertion

    @pytest.mark.asyncio
    async def test_execute_orchestration_validation_error(self, orchestrator):
        """Test orchestration with invalid input."""
        # Arrange
        contract = ModelContractOrchestrator(
            correlation_id=uuid4(),
            input_data={}  # Missing required fields
        )

        # Act & Assert
        with pytest.raises(OnexError) as exc_info:
            await orchestrator.execute_orchestration(contract)

        assert exc_info.value.code == CoreErrorCode.VALIDATION_ERROR
```

**Integration Test Example**:
```python
@pytest.mark.integration
class TestOrchestratorReducerFlow:
    """Integration tests for orchestrator-reducer workflow."""

    @pytest.mark.asyncio
    async def test_complete_workflow(
        self,
        orchestrator,
        reducer,
        postgresql_client,
        kafka_producer
    ):
        """Test complete stamping workflow with aggregation."""
        # Arrange
        correlation_id = uuid4()
        contract = ModelContractOrchestrator(
            correlation_id=correlation_id,
            input_data={"content": "test", "namespace": "integration.test"}
        )

        # Act - Orchestrate
        orchestration_result = await orchestrator.execute_orchestration(contract)

        # Act - Reduce
        reduction_contract = ModelContractReducer(
            correlation_id=correlation_id,
            input_state={"items": [orchestration_result.stamp_metadata]}
        )
        reduction_result = await reducer.execute_reduction(reduction_contract)

        # Assert - Orchestration
        assert orchestration_result.workflow_state.value == "COMPLETED"

        # Assert - Reduction
        assert reduction_result.total_items == 1
        assert "integration.test" in reduction_result.namespaces

        # Assert - Database persistence
        bridge_state = await postgresql_client.fetchrow(
            "SELECT * FROM bridge_states WHERE correlation_id = $1",
            correlation_id
        )
        assert bridge_state is not None
```

### Running Tests

```bash
# Run all tests
poetry run pytest tests/ -v

# Run specific test category
poetry run pytest tests/unit/ -v
poetry run pytest tests/integration/ -v
poetry run pytest tests/performance/ -m performance

# Run with coverage
poetry run pytest tests/ --cov=src --cov-report=html

# Run specific test file
poetry run pytest tests/unit/nodes/test_orchestrator.py -v

# Run specific test
poetry run pytest tests/unit/nodes/test_orchestrator.py::TestNodeBridgeOrchestrator::test_execute_orchestration_success -v
```

### Performance Test Requirements

**All performance-sensitive code must have benchmarks**:
```python
@pytest.mark.performance
class TestHashGenerationPerformance:
    """Performance benchmarks for hash generation."""

    @pytest.mark.benchmark(group="hash_generation")
    def test_hash_generation_small_file(self, benchmark, hash_generator):
        """Benchmark hash generation for small files (<1KB)."""
        content = b"x" * 1024  # 1KB

        result = benchmark(hash_generator.generate_hash, content)

        # Assert performance target
        assert benchmark.stats['mean'] < 0.001  # <1ms mean
        assert benchmark.stats['max'] < 0.002   # <2ms p99
```

---

## Documentation Requirements

### Documentation for All Changes

**All pull requests must include**:
1. **Code Documentation**: Docstrings for new functions/classes
2. **User Documentation**: Updates to relevant guides if behavior changes
3. **API Documentation**: Updates to API reference for API changes
4. **Migration Guides**: If breaking changes introduced

### Documentation Standards

**Format**: Markdown
**Style**: Clear, concise, with code examples
**Cross-References**: Link to related documentation

**Example**:
```markdown
# New Feature: Workflow Timeout Handling

## Overview

The orchestrator now supports configurable workflow timeouts to prevent
long-running workflows from blocking resources.

## Usage

\```python
from omnibase_core.models.contracts import ModelContractOrchestrator

# Set workflow timeout (in seconds)
contract = ModelContractOrchestrator(
    correlation_id=uuid4(),
    input_data={"content": "test"},
    timeout_seconds=30.0  # Fail after 30 seconds
)

result = await orchestrator.execute_orchestration(contract)
\```

## Configuration

Default timeout: 60 seconds
Minimum timeout: 1 second
Maximum timeout: 300 seconds (5 minutes)

## Related Documentation

- [Orchestrator API Reference](./api/API_REFERENCE.md#nodebridgeorchestrator-api)
- [Error Handling Guide](./guides/ERROR_HANDLING.md)
```

---

## Pull Request Process

### Before Submitting

**Pre-submission Checklist**:
- [ ] Code follows ONEX v2.0 standards
- [ ] All tests pass (`pytest tests/`)
- [ ] Code formatted (`black src/ tests/`)
- [ ] Code linted (`ruff check src/ tests/`)
- [ ] Type checked (`mypy src/`)
- [ ] Test coverage >90% for new code
- [ ] Documentation updated
- [ ] Commit messages follow convention
- [ ] Branch up to date with `main`

### Submitting Pull Request

1. **Push to your fork**:
   ```bash
   git push origin feature/my-feature
   ```

2. **Create PR on GitHub**:
   - Use descriptive title (e.g., "feat(orchestrator): add workflow timeout handling")
   - Fill out PR template completely
   - Reference related issues (e.g., "Closes #123")
   - Add screenshots/logs if relevant

3. **PR Template** (example):
   ```markdown
   ## Description

   Add workflow timeout handling to orchestrator to prevent long-running workflows.

   ## Type of Change

   - [x] New feature
   - [ ] Bug fix
   - [ ] Documentation update
   - [ ] Refactoring
   - [ ] Performance improvement

   ## Testing

   - [x] Unit tests added
   - [x] Integration tests added
   - [x] Performance benchmarks added
   - [x] Manual testing completed

   ## Checklist

   - [x] Code follows ONEX v2.0 standards
   - [x] Tests pass (`pytest tests/`)
   - [x] Code formatted (`black`)
   - [x] Documentation updated
   - [x] Test coverage >90%

   ## Related Issues

   Closes #123
   ```

### CI/CD Requirements

**All PRs must pass**:
1. **Unit Tests**: All unit tests must pass
2. **Integration Tests**: All integration tests must pass
3. **Code Quality**: Black, Ruff, mypy checks
4. **Test Coverage**: Coverage must be >90%
5. **Build**: Docker build must succeed

### Review Process

1. **Automated Checks**: CI/CD runs automatically
2. **Code Review**: At least 1 approval required from maintainers
3. **Discussion**: Address all review comments
4. **Approval**: All checks pass + approval → ready to merge
5. **Merge**: Squash and merge (default) or merge commit

---

## Code Review Guidelines

### For Authors

**When submitting for review**:
1. **Keep PRs focused**: One feature/fix per PR
2. **Write clear descriptions**: Explain what and why
3. **Add context**: Link to issues, design docs
4. **Respond promptly**: Address feedback within 48 hours
5. **Test thoroughly**: Ensure all scenarios covered

### For Reviewers

**When reviewing PRs**:
1. **Be constructive**: Suggest improvements, not just point out issues
2. **Be specific**: Reference line numbers, provide examples
3. **Check for**:
   - ONEX v2.0 compliance
   - Test coverage
   - Documentation completeness
   - Performance implications
   - Security concerns
4. **Approve when**: All checks pass, concerns addressed
5. **Response time**: Review within 48 hours

**Review Checklist**:
- [ ] Code follows ONEX v2.0 standards
- [ ] Naming conventions correct (suffix-based)
- [ ] Type annotations complete
- [ ] Error handling uses OnexError
- [ ] Tests comprehensive (>90% coverage)
- [ ] Documentation updated
- [ ] Performance acceptable
- [ ] No security vulnerabilities

---

## Community Guidelines

### Code of Conduct

**We are committed to providing a welcoming and inclusive environment.**

**Expected Behavior**:
- Be respectful and professional
- Be collaborative and helpful
- Accept constructive criticism gracefully
- Focus on what is best for the project

**Unacceptable Behavior**:
- Harassment or discrimination
- Personal attacks
- Trolling or inflammatory comments
- Publishing others' private information

**Reporting**:
If you experience or witness unacceptable behavior, report to jonah@omninode.ai.

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **Pull Requests**: Code contributions
- **Discussions**: General questions, proposals

### Getting Help

**Before asking for help**:
1. Check [Documentation Index](./INDEX.md)
2. Search existing issues
3. Review [Getting Started](./GETTING_STARTED.md) and [Setup Guide](./SETUP.md)

**When asking for help**:
1. Provide context (what you're trying to do)
2. Show what you've tried
3. Include error messages/logs
4. Specify your environment (OS, Python version, etc.)

---

## Recognition

### Contributors

We value all contributions, large and small. All contributors will be:
- Added to CONTRIBUTORS.md
- Mentioned in release notes (for significant contributions)
- Recognized in project documentation

### Types of Contributions

We welcome:
- **Code**: Features, bug fixes, refactoring
- **Documentation**: Guides, tutorials, examples
- **Testing**: Test improvements, bug reports
- **Design**: Architecture proposals, design docs
- **Community**: Helping others, answering questions

---

## License

By contributing to OmniNode Bridge, you agree that your contributions will be licensed under the same license as the project.

---

## Questions?

- **Documentation**: [docs/INDEX.md](./INDEX.md)
- **Setup Issues**: [docs/SETUP.md](./SETUP.md)
- **Architecture Questions**: [docs/architecture/ARCHITECTURE.md](./architecture/ARCHITECTURE.md)
- **API Questions**: [docs/api/API_REFERENCE.md](./api/API_REFERENCE.md)

---

**Thank you for contributing to OmniNode Bridge!** 🚀

Your contributions help build a better, more robust system for the omninode ecosystem.

---

**Maintained By**: omninode_bridge team
**Last Updated**: October 15, 2025
**Document Version**: 1.0
