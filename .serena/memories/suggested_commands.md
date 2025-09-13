# Essential Commands for ONEX Infrastructure Development

## Agent-Driven Development (MANDATORY)
**NEVER code directly - Always use agents:**

### Orchestration Agents (Complex Tasks)
```bash
# Primary coordination for multi-step tasks
agent-onex-coordinator

# Multi-step execution and progress management  
agent-workflow-coordinator

# Comprehensive ticket operations
agent-ticket-manager
```

### Specialist Agents
```bash
# Code generation and validation
agent-contract-driven-generator
agent-contract-validator
agent-ast-generator

# DevOps and infrastructure
agent-devops-infrastructure
agent-security-audit
agent-performance
agent-production-monitor

# Quality and testing
agent-testing
agent-pr-review
agent-pr-create
agent-address-pr-comments

# Intelligence and research
agent-research
agent-debug-intelligence
agent-rag-query
agent-rag-update
```

## Development Workflow
```bash
# Setup environment
poetry install --with dev
pre-commit install

# Testing
pytest                    # Run all tests
pytest tests/specific/    # Run specific test directory
pytest -v                # Verbose output
pytest --asyncio-mode=auto  # For async tests

# Code Quality
ruff check               # Linting
ruff format             # Code formatting (replaces black)
mypy src/              # Type checking
isort src/             # Import sorting

# Pre-commit validation
pre-commit run --all-files

# Git operations
git status
git add .
git commit -m "message"
git push origin feature-branch
```

## Infrastructure Specific
```bash
# CLI entry point
omni-infra --help

# Container operations
docker-compose -f docker-compose.infrastructure.yml up
docker-compose -f docker-compose.infrastructure.yml down

# Integration testing
python test_postgres_redpanda_integration.py
python validate_integration.py
```

## Key Ports
- PostgreSQL: 5432
- Kafka: 9092 (plaintext), 9093 (SSL)
- Consul: 8500 (HTTP), 8600 (DNS)
- Vault: 8200
- Debug Dashboard: 8096