# OmniNode Bridge - Tech Stack

## Core Technologies

### Language & Framework
- **Python 3.12** - Primary language
- **FastAPI** - REST API framework with automatic OpenAPI documentation
- **Pydantic 2.11.7** - Data validation and serialization
- **Uvicorn** - ASGI server for FastAPI applications

### Dependency Management
- **Poetry** - Modern dependency management and packaging
- **pyproject.toml** - Project configuration and dependencies

### Data & Messaging
- **PostgreSQL 15** - Primary database for persistence
- **asyncpg** - Async PostgreSQL driver
- **psycopg2-binary** - Sync PostgreSQL driver
- **RedPanda/Kafka** - Event streaming and message bus
- **aiokafka** - Async Kafka client

### Service Infrastructure
- **HashiCorp Consul** - Service discovery and configuration
- **Redis** - Caching and session management
- **Docker & Docker Compose** - Containerization and orchestration

### Security & Resilience
- **slowapi** - Rate limiting for FastAPI
- **circuitbreaker** - Circuit breaker pattern implementation
- **cryptography** - Encryption and security utilities
- **httpx** - HTTP client for health checks

### Observability & Monitoring
- **structlog** - Structured logging
- **prometheus-client** - Metrics collection
- **pydantic-settings** - Configuration management

### Development Tools
- **pytest** - Testing framework with asyncio support
- **mypy** - Static type checking
- **black** - Code formatting
- **isort** - Import sorting
- **ruff** - Fast Python linter
- **pre-commit** - Git hooks for code quality

### AI Integration
- **Python-based AI Lab integration** - Custom integration with local AI models
- **Smart Responder Client** - Model performance tracking and routing
