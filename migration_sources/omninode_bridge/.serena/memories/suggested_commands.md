# OmniNode Bridge Development Commands

## Development Setup
```bash
# Install dependencies
poetry install

# Install development dependencies
poetry install --with dev

# Activate virtual environment
poetry shell
```

## Code Quality and Testing
```bash
# Linting with ruff
poetry run ruff check src/ tests/

# Formatting with black
poetry run black src/ tests/

# Import sorting with isort
poetry run isort src/ tests/

# Type checking with mypy
poetry run mypy src/

# Run tests
poetry run pytest tests/

# Run tests with coverage
poetry run pytest --cov=src/omninode_bridge tests/

# Performance testing
poetry run locust -f tests/performance/
```

## Docker Development
```bash
# Build all services
docker-compose build

# Start infrastructure (PostgreSQL, RedPanda, Consul)
docker-compose up postgres redpanda consul -d

# Start specific service
docker-compose up hook-receiver -d

# View logs
docker-compose logs -f hook-receiver

# Stop all services
docker-compose down

# Clean rebuild
docker-compose down -v && docker-compose build --no-cache
```

## Service Management
```bash
# Start HookReceiver locally
poetry run python -m omninode_bridge.main

# Submit workflow via CLI
poetry run python -m omninode_bridge.cli.workflow_submit

# Check service health
curl http://localhost:8001/health
curl http://localhost:8005/     # Model Metrics
curl http://localhost:8006/health # Workflow Coordinator
```

## Database Operations
```bash
# Connect to PostgreSQL
docker exec -it omninode-bridge-postgres psql -U postgres -d omninode_bridge

# View connection pools
docker-compose logs postgres | grep -i connection
```

## Monitoring and Debugging
```bash
# View Prometheus metrics
curl http://localhost:8001/metrics

# RedPanda UI (development)
open http://localhost:8080

# Service logs
docker-compose logs -f --tail=100 hook-receiver
```
