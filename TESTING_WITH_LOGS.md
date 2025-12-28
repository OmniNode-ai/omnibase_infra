# Testing with Enhanced Logging

## Quick Start

### Run E2E Tests with Full Logging

```bash
# Set DEBUG level to see all logging
export ONEX_LOG_LEVEL=DEBUG

# Run the runtime E2E test
pytest tests/integration/registration/e2e/test_runtime_e2e.py -v -s

# Or run all E2E tests
pytest tests/integration/registration/e2e/ -v -s
```

### Expected Log Flow (Success Case)

When events are processing correctly, you should see this sequence:

```
1. [INFO] Introspection dispatcher created and wired
2. [INFO] Subscribing to introspection events on Kafka
3. [INFO] Introspection event consumer started successfully
4. [DEBUG] Introspection message callback invoked      ← Message received from Kafka
5. [DEBUG] Parsing message value as bytes
6. [DEBUG] Validating payload as ModelNodeIntrospectionEvent
7. [INFO] Introspection event parsed successfully      ← Event validated
8. [DEBUG] Event envelope created
9. [INFO] Routing to introspection dispatcher          ← Entering handler pipeline
10. [INFO] Introspection event processed successfully  ← Handler completed
11. [DEBUG] Introspection message callback completed
```

### Diagnosing Pipeline Breakage

**If you see steps 1-3 but NOT step 4**:
- **Problem**: Consumer not receiving messages from Kafka
- **Check**:
  - Is Kafka running? `docker ps | grep redpanda`
  - Is topic correct? Check `config.input_topic`
  - Are messages published to Kafka? Check with `kafkacat`

**If you see steps 1-4 but NOT step 7**:
- **Problem**: Event validation failing
- **Check**:
  - Look for ValidationError in logs (debug level)
  - Verify event schema matches `ModelNodeIntrospectionEvent`
  - Check JSON structure of published message

**If you see steps 1-8 but NOT step 10**:
- **Problem**: Handler failing to process event
- **Check**:
  - Look for warning/error logs from dispatcher
  - Check `HandlerNodeIntrospected` implementation
  - Verify projector is initialized and connected to PostgreSQL

## Grep for Specific Correlation IDs

```bash
# Find all logs for a specific correlation_id
grep "correlation_id=abc-123" runtime.log

# Find all callback invocations
grep "Introspection message callback invoked" runtime.log

# Find all processing successes
grep "processed successfully" runtime.log

# Find all failures
grep -E "(failed|error)" runtime.log
```

## Useful Log Filters

```bash
# Only show INFO level and above
pytest ... 2>&1 | grep -E "\[(INFO|WARNING|ERROR)\]"

# Only show callback flow
pytest ... 2>&1 | grep "callback"

# Only show dispatcher flow
pytest ... 2>&1 | grep "dispatcher"

# Show timing information
pytest ... 2>&1 | grep "duration"
```

## Running with Docker Compose

If running via docker-compose, set logging in the environment:

```yaml
# docker-compose.yml
services:
  kernel:
    environment:
      - ONEX_LOG_LEVEL=DEBUG
    volumes:
      - ./logs:/app/logs
```

Then tail logs:
```bash
docker-compose logs -f kernel | grep -E "(correlation_id|Introspection)"
```

## Debugging Checklist

- [ ] Step 1: Verify dispatcher created (logs show "dispatcher created and wired")
- [ ] Step 2: Verify consumer subscribed (logs show "consumer started successfully")
- [ ] Step 3: Publish test event to Kafka topic
- [ ] Step 4: Wait 5 seconds and check logs for "callback invoked"
- [ ] Step 5: If no callback, check Kafka consumer group status
- [ ] Step 6: If callback seen but no success, check validation errors
- [ ] Step 7: If validation passes but no success, check handler logs
- [ ] Step 8: Verify projection written to PostgreSQL

## Log Levels Explained

| Level | When to Use |
|-------|-------------|
| `DEBUG` | Development, debugging pipeline issues, full trace |
| `INFO` | Production, key events only (dispatcher, routing, success) |
| `WARNING` | Production, non-fatal errors (processing failures) |
| `ERROR` | Production, fatal errors requiring investigation |

## Common Issues

**Issue**: Too many logs, can't find relevant information
**Solution**: Use `INFO` level and grep for correlation_id

**Issue**: No callback logs appearing
**Solution**: Verify Kafka consumer is created AND messages are published to correct topic

**Issue**: Callback invoked but events not in database
**Solution**: Check projector logs and PostgreSQL connection

**Issue**: ValidationError appearing for every message
**Solution**: This is normal if topic contains other message types; use DEBUG level to see details
