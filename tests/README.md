# Infrastructure Foundation Utilities - Test Suite

Comprehensive pytest test suite for omnibase_infra foundation utilities.

## Prerequisites

⚠️ **Python Version: 3.11 or 3.12 (Required)**

**Important**: asyncpg 0.29.0 has compilation issues with Python 3.13. Use Python 3.11 or 3.12 for running tests.

## Quick Start

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-mock pytest-cov

# Run all tests
pytest tests/infrastructure/ -v

# Run with coverage
pytest tests/infrastructure/ --cov=src/omnibase_infra/infrastructure --cov-report=html --cov-report=term

# Run specific category
pytest tests/infrastructure/postgres/ -v       # PostgreSQL tests
pytest tests/infrastructure/kafka/ -v          # Kafka tests
pytest tests/infrastructure/resilience/ -v     # Resilience tests
pytest tests/infrastructure/observability/ -v  # Observability tests
```

## Test Suite Statistics

- **Total Test Files:** 9
- **Total Test Functions:** 240
- **Total Lines of Test Code:** 3,791
- **Coverage Target:** >90%

## Test Categories

### 1. PostgreSQL (30 tests)
- `test_connection_manager.py` - Connection pooling, query execution, metrics

### 2. Kafka (64 tests)
- `test_producer_pool.py` - Producer pooling and messaging
- `test_topic_registry.py` - Topic naming, validation, metadata

### 3. Resilience (80 tests)
- `test_circuit_breaker_factory.py` - Circuit breaker patterns
- `test_retry_policy.py` - Retry strategies and backoff
- `test_rate_limiter.py` - Rate limiting and token bucket

### 4. Observability (66 tests)
- `test_structured_logger.py` - Structured logging with correlation IDs
- `test_tracer_factory.py` - OpenTelemetry tracing
- `test_metrics_registry.py` - Prometheus metrics

## Features

- ✅ Comprehensive mocking of external dependencies
- ✅ No actual service connections required
- ✅ Fast, isolated tests
- ✅ Async/await support with pytest-asyncio
- ✅ Shared fixtures in conftest.py
- ✅ OnexError integration testing
- ✅ Error path coverage
- ✅ Integration test scenarios

## Documentation

See [TEST_COVERAGE_SUMMARY.md](./TEST_COVERAGE_SUMMARY.md) for detailed coverage analysis.

## pytest Configuration

Configuration is in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
asyncio_mode = "auto"
```

## Shared Fixtures

Available in `conftest.py`:
- `event_loop` - Async event loop
- `mock_async_pool` - Mock asyncpg pool
- `mock_async_connection` - Mock asyncpg connection
- `mock_kafka_producer` - Mock Kafka producer
- `mock_kafka_consumer` - Mock Kafka consumer
- `reset_structlog` - Structlog cleanup
- `reset_logger_factory` - Logger factory reset
- `mock_time` - Time mocking

## Test Organization

Tests are organized by functionality within each file:
1. **Initialization Tests** - Configuration and setup
2. **Operation Tests** - Core functionality
3. **Error Handling Tests** - Exception scenarios
4. **Integration Tests** - End-to-end workflows

## Example Test

```python
@pytest.mark.asyncio
async def test_execute_select_query(manager, mock_pool, mock_connection):
    """Test executing SELECT query."""
    manager.pool = mock_pool
    manager.is_initialized = True
    mock_pool.acquire = AsyncMock(return_value=mock_connection)
    mock_pool.release = AsyncMock()

    mock_records = [Mock(spec=Record), Mock(spec=Record)]
    mock_connection.fetch = AsyncMock(return_value=mock_records)

    result = await manager.execute_query("SELECT * FROM users")

    assert result == mock_records
    assert len(manager.query_metrics) == 1
    assert manager.query_metrics[0].was_successful is True
```

## CI/CD Integration

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install pytest pytest-asyncio pytest-mock pytest-cov
    pytest tests/infrastructure/ --cov=src/omnibase_infra/infrastructure --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

## Contributing

When adding new tests:
1. Follow existing test organization patterns
2. Use descriptive test names: `test_<what>_<scenario>`
3. Mock all external dependencies
4. Test both success and failure paths
5. Add docstrings explaining what is being tested
6. Aim for >90% code coverage

## Troubleshooting

### Import Errors
Ensure the package is installed in development mode:
```bash
pip install -e .
```

### Async Test Issues
Make sure pytest-asyncio is installed and `asyncio_mode = "auto"` is set in pytest configuration.

### Mock Issues
Check that mocks are properly configured in conftest.py and reset between tests.
