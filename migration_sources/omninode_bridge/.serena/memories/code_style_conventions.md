# OmniNode Bridge Code Style and Conventions

## Python Style Guidelines
- **Python Version**: 3.12
- **Line Length**: 88 characters (Black standard)
- **Formatting**: Black formatter
- **Import Sorting**: isort with Black profile
- **Linting**: Ruff with comprehensive rule set
- **Type Checking**: MyPy with strict mode

## Code Structure
- **Type Hints**: Required for all function signatures and class attributes
- **Docstrings**: Google-style docstrings for all public functions and classes
- **Error Handling**: Use specific exceptions, structured logging with structlog
- **Async/Await**: Use async/await consistently for I/O operations

## Architecture Patterns
- **Dependency Injection**: Services accept configuration via constructor
- **Resource Management**: Use async context managers for connections
- **Circuit Breakers**: Apply to external service calls
- **Structured Logging**: Use structlog with JSON output
- **Metrics**: Prometheus metrics for monitoring

## Security Practices
- **Environment Variables**: Never hardcode secrets
- **SSL/TLS**: Use secure connections for all external services
- **Rate Limiting**: Apply to all public endpoints
- **Authentication**: API key authentication for all services
- **Input Validation**: Pydantic models for all request/response data

## Testing
- **Test Structure**: tests/unit/, tests/integration/, tests/performance/
- **Test Naming**: test_*.py files, test_* functions
- **Async Testing**: Use pytest-asyncio
- **Coverage**: Minimum 80% code coverage
- **Integration Tests**: Use testcontainers for real database testing
