# Developer Quick Reference

## Essential Commands

### Environment Management
```bash
# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# Stop development environment
docker-compose -f docker-compose.dev.yml down

# Restart specific service
docker-compose -f docker-compose.dev.yml restart postgres

# View logs
docker-compose -f docker-compose.dev.yml logs -f hook-receiver
```

### Code Quality
```bash
# Format code
poetry run black .

# Lint code
poetry run ruff check .
poetry run ruff check --fix .

# Type checking
poetry run mypy src/

# All quality checks
poetry run black . && poetry run ruff check --fix . && poetry run mypy src/
```

### Testing
```bash
# Run all tests
poetry run pytest

# Run specific test types
poetry run pytest tests/unit/
poetry run pytest tests/integration/
poetry run pytest -m "not slow"

# Run with coverage
poetry run pytest --cov=omninode_bridge --cov-report=html

# Run specific test
poetry run pytest tests/unit/services/test_hook_receiver.py::TestHookReceiver::test_process_webhook_success -v
```

### Database Operations
```bash
# Connect to database
docker exec -it postgres psql -U bridge_user -d omninode_bridge

# Reset test database
poetry run python -m omninode_bridge.database.reset_test_db

# Run migrations
poetry run alembic upgrade head

# Create new migration
poetry run alembic revision --autogenerate -m "description"
```

### API Testing
```bash
# Health checks
curl http://localhost:8001/health  # HookReceiver
curl http://localhost:8002/health  # ModelMetrics
curl http://localhost:8003/health  # WorkflowCoordinator

# Test webhook
curl -X POST http://localhost:8001/api/v1/webhooks \
  -H "Content-Type: application/json" \
  -d '{"event_type": "test", "payload": {"test": "data"}}'

# View metrics
curl http://localhost:8001/metrics
```

### Git Workflow
```bash
# Create feature branch
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name

# Commit with conventional commits
git add .
git commit -m "feat: add new webhook validation

- Implement comprehensive input validation
- Add rate limiting for webhook endpoints
- Update API documentation

Closes #123"

# Push and create PR
git push origin feature/your-feature-name
gh pr create --title "feat: add webhook validation" --body "Description of changes"
```

## Configuration Files

### Local Environment (.env)
```bash
# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=omninode_bridge
POSTGRES_USER=bridge_user
POSTGRES_PASSWORD=bridge_password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# Services
HOOK_RECEIVER_PORT=8001
MODEL_METRICS_PORT=8002
WORKFLOW_COORDINATOR_PORT=8003

# Development
DEBUG=true
LOG_LEVEL=DEBUG
ENVIRONMENT=development
```

### VS Code Settings (.vscode/settings.json)
```json
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "python.formatting.provider": "black",
    "python.linting.ruffEnabled": true,
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"]
}
```

## Common Code Patterns

### Service Implementation
```python
from fastapi import FastAPI, HTTPException
from omninode_bridge.core.base_service import BaseService

class YourService(BaseService):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Initialize service-specific components

    async def initialize(self) -> bool:
        """Initialize service with dependencies."""
        try:
            await super().initialize()
            # Service-specific initialization
            return True
        except Exception as e:
            logger.error(f"Service initialization failed: {e}")
            return False

    async def process_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming request."""
        # Implementation
        pass
```

### Error Handling
```python
from omninode_bridge.exceptions import ValidationError, ProcessingError

try:
    result = await some_operation()
except ValidationError:
    raise  # Re-raise validation errors
except ExternalServiceError as e:
    raise ProcessingError(
        message="Operation failed due to external dependency",
        details={"service": e.service_name},
        error_code="EXTERNAL_DEPENDENCY_FAILURE"
    ) from e
```

### Database Operations
```python
async def store_event(self, event_data: Dict[str, Any]) -> str:
    """Store event in database."""
    async with self.db_pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            INSERT INTO events (id, event_type, payload, created_at)
            VALUES ($1, $2, $3, NOW())
            RETURNING id
            """,
            str(uuid.uuid4()),
            event_data["event_type"],
            json.dumps(event_data["payload"])
        )
        return result["id"]
```

### API Endpoint
```python
from fastapi import APIRouter, HTTPException, Depends
from omninode_bridge.models.events import WebhookEvent

router = APIRouter()

@router.post("/webhooks", response_model=Dict[str, Any])
async def process_webhook(
    event: WebhookEvent,
    service: YourService = Depends(get_service)
) -> Dict[str, Any]:
    """Process incoming webhook event."""
    try:
        result = await service.process_webhook(event.dict())
        return {"status": "success", "event_id": result["id"]}
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ProcessingError as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Testing Patterns

### Unit Test Template
```python
import pytest
from unittest.mock import AsyncMock
from omninode_bridge.services.your_service import YourService

class TestYourService:
    @pytest.fixture
    def mock_dependencies(self):
        return {
            "db_client": AsyncMock(),
            "kafka_client": AsyncMock()
        }

    @pytest.fixture
    def service(self, mock_dependencies):
        config = {"test": "config"}
        service = YourService(config)
        service.db_client = mock_dependencies["db_client"]
        return service

    @pytest.mark.asyncio
    async def test_successful_processing(self, service):
        # Arrange
        input_data = {"test": "data"}

        # Act
        result = await service.process(input_data)

        # Assert
        assert result["status"] == "success"
```

### Integration Test Template
```python
import pytest
from httpx import AsyncClient
from testcontainers.postgres import PostgresContainer

@pytest.mark.asyncio
async def test_webhook_integration():
    with PostgresContainer("postgres:15") as postgres:
        # Setup test environment
        db_url = postgres.get_connection_url()

        # Test implementation
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/api/v1/webhooks", json=test_data)
            assert response.status_code == 200
```

## Monitoring & Debugging

### Log Analysis
```bash
# View service logs
docker logs hook-receiver-container

# Follow logs in real-time
docker logs -f hook-receiver-container

# Search logs for errors
docker logs hook-receiver-container 2>&1 | grep ERROR

# Export logs to file
docker logs hook-receiver-container > service.log 2>&1
```

### Metrics Queries (Prometheus)
```promql
# Request rate
rate(http_requests_total[5m])

# Error rate
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])

# Response time 95th percentile
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Active database connections
postgres_active_connections
```

### Database Queries
```sql
-- View recent events
SELECT * FROM webhook_events
WHERE created_at > NOW() - INTERVAL '1 hour'
ORDER BY created_at DESC
LIMIT 10;

-- Check session activity
SELECT status, COUNT(*)
FROM sessions
WHERE updated_at > NOW() - INTERVAL '1 day'
GROUP BY status;

-- Performance analysis
SELECT
    event_type,
    COUNT(*) as total_events,
    AVG(EXTRACT(EPOCH FROM (updated_at - created_at))) as avg_processing_time
FROM webhook_events
WHERE created_at > NOW() - INTERVAL '1 day'
GROUP BY event_type;
```

## Troubleshooting Checklist

### Service Won't Start
- [ ] Check port availability: `netstat -tulpn | grep LISTEN`
- [ ] Verify environment variables: `docker exec service env`
- [ ] Check service dependencies: `docker-compose ps`
- [ ] Review service logs: `docker logs service-name`

### Tests Failing
- [ ] Update dependencies: `poetry install`
- [ ] Reset test database: `poetry run python -m omninode_bridge.database.reset_test_db`
- [ ] Clear cache: `poetry run pytest --cache-clear`
- [ ] Check test isolation: Run individual tests

### Performance Issues
- [ ] Check system resources: `docker stats`
- [ ] Review database connections: `SELECT * FROM pg_stat_activity;`
- [ ] Monitor cache hit rates: `INFO stats` in Redis
- [ ] Check for slow queries: Enable PostgreSQL query logging

### API Issues
- [ ] Verify service health: `curl http://localhost:8001/health`
- [ ] Check request format: Validate JSON schema
- [ ] Review authentication: Check API keys/tokens
- [ ] Monitor rate limits: Check response headers

## Useful Aliases

Add to your shell configuration file:

```bash
# Environment
alias bridge-up="docker-compose -f docker-compose.dev.yml up -d"
alias bridge-down="docker-compose -f docker-compose.dev.yml down"
alias bridge-logs="docker-compose -f docker-compose.dev.yml logs -f"

# Development
alias bridge-test="poetry run pytest"
alias bridge-format="poetry run black . && poetry run ruff check --fix ."
alias bridge-coverage="poetry run pytest --cov=omninode_bridge --cov-report=html"

# Database
alias bridge-db="docker exec -it postgres psql -U bridge_user -d omninode_bridge"
alias bridge-db-reset="poetry run python -m omninode_bridge.database.reset_test_db"

# API Testing
alias bridge-health="curl http://localhost:8001/health && curl http://localhost:8002/health && curl http://localhost:8003/health"
alias bridge-metrics="curl http://localhost:8001/metrics"

# Git
alias bridge-feature="git checkout develop && git pull origin develop && git checkout -b feature/"
alias bridge-ready="poetry run black . && poetry run ruff check --fix . && poetry run pytest"
```

## Port Reference

| Service | Port | Purpose |
|---------|------|---------|
| HookReceiver | 8001 | Webhook processing API |
| ModelMetrics | 8002 | AI model metrics API |
| WorkflowCoordinator | 8003 | Workflow orchestration API |
| PostgreSQL | 5432 | Primary database |
| Redis | 6379 | Caching and sessions |
| Kafka | 9092 | Message broker |
| Zookeeper | 2181 | Kafka coordination |
| Prometheus | 9090 | Metrics collection |
| Grafana | 3000 | Monitoring dashboards |
| Consul | 8500 | Service discovery |

## File Locations

| Type | Location | Purpose |
|------|----------|---------|
| Source Code | `src/omninode_bridge/` | Main application code |
| Tests | `tests/` | All test files |
| Documentation | `docs/` | Project documentation |
| Configuration | `docker-compose.*.yml` | Service configuration |
| Scripts | `scripts/` | Development utilities |
| Monitoring | `monitoring/` | Prometheus/Grafana config |
| Logs | `logs/` | Application logs (dev) |
| Data | `data/` | Persistent data (dev) |
