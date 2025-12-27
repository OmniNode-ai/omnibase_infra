# E2E Test Diagnostic Manual
**Purpose**: Diagnose event processing failures in runtime E2E tests
**Context**: Enhanced logging added to kernel.py for tracing pipeline breakpoints
**Date**: 2025-12-27

---

## 🎯 Objective

Identify **exactly where** the event processing pipeline breaks when the runtime container consumes introspection events from Kafka.

## 📊 Event Processing Pipeline

```
Kafka Message → Callback Invoked → JSON Parsing → Model Validation
    → Envelope Creation → Dispatcher Routing → Handler Execution
    → Projector Persistence → PostgreSQL Projection
```

**Each step has logging** - we'll trace through to find the break point.

---

## 🚀 Step 1: Rebuild Runtime Container

```bash
cd "$(git rev-parse --show-toplevel)/docker"

# Stop existing runtime if running
docker compose -f docker-compose.e2e.yml down runtime

# Rebuild with latest code changes (enhanced logging)
DOCKER_BUILDKIT=1 docker compose -f docker-compose.e2e.yml build runtime

# Start infrastructure + runtime
docker compose -f docker-compose.e2e.yml --profile runtime up -d

# Wait for healthy status
sleep 15
```

**Expected duration**: 2-3 minutes for rebuild, 30-40 seconds for startup

---

## 🏥 Step 2: Verify Container Health

```bash
cd "$(git rev-parse --show-toplevel)/docker"

# Check container status
docker compose -f docker-compose.e2e.yml ps runtime
# Expected: State = "Up" or "Up (healthy)"

# Quick health endpoint test
curl -s http://localhost:8085/health | jq .
# Expected: HTTP 200 with JSON response
```

**If container fails to start**, check startup logs:
```bash
docker compose -f docker-compose.e2e.yml logs runtime | head -100
```

---

## 📝 Step 3: Capture Startup Logs

```bash
cd "$(git rev-parse --show-toplevel)/docker"

# Capture last 100 lines of startup logs
docker compose -f docker-compose.e2e.yml logs runtime | tail -100 > /tmp/runtime_startup.log

# Open in editor for analysis
cat /tmp/runtime_startup.log
```

### ✅ Required Startup Log Messages

Search for these **EXACT log messages** in startup logs:

1. **"Container wiring complete"** - Infrastructure services wired
2. **"HandlerNodeIntrospected resolved successfully"** - Handler found
3. **"Introspection dispatcher created and wired"** - Dispatcher ready
4. **"Subscribing to introspection events on Kafka"** - Starting subscription
5. **"Introspection event consumer started successfully"** - Consumer active
6. **"ONEX Runtime Kernel v"** - Startup banner displayed

**Diagnostic commands**:
```bash
# Check for critical startup messages
grep -i "container wiring complete" /tmp/runtime_startup.log
grep -i "HandlerNodeIntrospected resolved" /tmp/runtime_startup.log
grep -i "dispatcher created and wired" /tmp/runtime_startup.log
grep -i "consumer started successfully" /tmp/runtime_startup.log

# Check for startup errors
grep -i "error\|exception\|failed" /tmp/runtime_startup.log
```

### 🚨 Startup Failure Scenarios

**Scenario S1: "HandlerNodeIntrospected resolved successfully" NOT found**
- **Cause**: Handler wiring failed in `container_wiring.wire_registration_handlers()`
- **Fix**: Check PostgreSQL connection, verify schema migrations ran
- **Evidence**: Look for database connection errors in startup logs

**Scenario S2: "Introspection dispatcher created and wired" NOT found**
- **Cause**: `DispatcherNodeIntrospected` instantiation failed
- **Fix**: Check if `get_handler_node_introspected_from_container()` raised exception
- **Evidence**: Exception traceback before dispatcher creation

**Scenario S3: "Introspection event consumer started successfully" NOT found**
- **Cause**: Kafka subscription failed
- **Fix**: Verify Kafka reachable at `redpanda:9092` from container
- **Evidence**: KafkaEventBus error logs, connection timeouts

**Scenario S4: Container exits immediately after startup**
- **Cause**: Asyncio event loop crashed
- **Fix**: Check for unhandled exceptions in startup coroutines
- **Evidence**: Python traceback in container logs

---

## 🧪 Step 4: Run E2E Test with Verbose Logging

**Open TWO terminals** - one for test, one for logs.

### Terminal 1: Capture Container Logs (Live)

```bash
cd "$(git rev-parse --show-toplevel)/docker"

# Stream container logs to file
docker compose -f docker-compose.e2e.yml logs -f runtime > /tmp/runtime_test_logs.txt
```

**Keep this running** while test executes in Terminal 2.

### Terminal 2: Run E2E Test

```bash
cd "$(git rev-parse --show-toplevel)"

# Set verbose pytest output
export PYTEST_CURRENT_TEST=""

# Run single test with verbose output
poetry run pytest \
  tests/integration/registration/e2e/test_runtime_e2e.py::TestRuntimeE2EFlow::test_introspection_event_processed_by_runtime \
  -v -s --log-cli-level=DEBUG \
  2>&1 | tee /tmp/test_output.log
```

**Expected duration**: 30-45 seconds

### Test Output Analysis

```bash
# Check test result
grep -i "PASSED\|FAILED" /tmp/test_output.log

# If FAILED, check assertion
grep -A 10 "AssertionError" /tmp/test_output.log
```

**Common assertion failure**:
```
AssertionError: Runtime did not create projection for node <UUID> within 30s. Check runtime logs.
```

This means: **event was published but projection not created** → pipeline broke somewhere.

---

## 🔍 Step 5: Analyze Container Logs During Test

**Stop log capture** in Terminal 1 (Ctrl+C), then analyze:

```bash
cd "$(git rev-parse --show-toplevel)"

# Open captured logs
cat /tmp/runtime_test_logs.txt
```

### 🎯 Critical Log Search Commands

Run these searches to identify the failure point:

#### 1. Check if Callback Invoked
```bash
grep -i "introspection message callback invoked" /tmp/runtime_test_logs.txt
```

**Expected**: One or more matches (one per message consumed)
**If NOT found**: Consumer not receiving messages → **Scenario B** below

#### 2. Check Message Parsing
```bash
grep -i "parsing message value" /tmp/runtime_test_logs.txt
```

**Expected**: Logs showing `bytes`, `string`, or `dict` value type
**If NOT found**: Callback exiting before parse → check for "Message value is None"

#### 3. Check Model Validation
```bash
grep -i "validating payload as ModelNodeIntrospectionEvent" /tmp/runtime_test_logs.txt
```

**Expected**: Validation attempt logged
**If NOT found**: JSON parsing failed → check for JSONDecodeError

#### 4. Check Validation Success
```bash
grep -i "introspection event parsed successfully" /tmp/runtime_test_logs.txt
```

**Expected**: Success message with node_id
**If NOT found**: ValidationError occurred → **Scenario C** below

#### 5. Check Dispatcher Routing
```bash
grep -i "routing to introspection dispatcher" /tmp/runtime_test_logs.txt
```

**Expected**: Message showing envelope_correlation_id and node_id
**If NOT found**: Envelope creation failed → **Scenario D** below

#### 6. Check Dispatcher Success
```bash
grep -i "introspection event processed successfully" /tmp/runtime_test_logs.txt
```

**Expected**: Success with timing (e.g., "in 0.123s")
**If NOT found**: Dispatcher call failed → **Scenario E** below

#### 7. Check for Errors
```bash
grep -i "error\|exception\|failed" /tmp/runtime_test_logs.txt
```

**Expected**: No critical errors (warnings OK)
**If found**: Identify error type and context

#### 8. Extract Correlation IDs
```bash
grep -oP "correlation_id=\K[a-f0-9-]+" /tmp/runtime_test_logs.txt | sort -u
```

Use correlation IDs to trace specific message flows.

---

## 🐛 Step 6: Identify Failure Scenario

Based on log analysis above, determine which scenario matches:

### Scenario A: Consumer Never Started
**Symptoms**:
- No "Introspection event consumer started successfully" in startup logs
- No callback invocations during test

**Root Causes**:
1. `introspection_dispatcher is None` (handler wiring failed)
2. Event bus is not `KafkaEventBus` (in-memory bus used instead)
3. Subscription threw exception

**Diagnosis**:
```bash
# Check if dispatcher was created
grep -i "dispatcher created" /tmp/runtime_startup.log

# Check event bus type
grep -i "event bus:" /tmp/runtime_startup.log

# Check for subscription errors
grep -A 5 "subscribing to introspection events" /tmp/runtime_startup.log
```

**Fixes**:
- If handler wiring failed: Fix PostgreSQL connection or schema
- If wrong event bus: Verify `KAFKA_BOOTSTRAP_SERVERS` env var set
- If subscription failed: Check Kafka connectivity from container

---

### Scenario B: Consumer Started But No Messages Received
**Symptoms**:
- "Introspection event consumer started successfully" found
- No "callback invoked" messages during test

**Root Causes**:
1. Topic name mismatch (publishing to different topic than consuming)
2. Consumer group offset issue (already consumed message)
3. Kafka broker not routing messages
4. Test published to wrong Kafka instance

**Diagnosis**:
```bash
# Check subscribed topic
grep -i "subscribing to introspection" /tmp/runtime_startup.log

# Check published topic from test
grep -i "publish" /tmp/test_output.log

# List Kafka topics
docker exec omnibase-infra-redpanda rpk topic list

# Check topic contents
docker exec omnibase-infra-redpanda rpk topic consume \
  dev.onex.evt.node-introspection.v1 \
  --num 10 \
  --brokers redpanda:9092
```

**Fixes**:
- If topic mismatch: Verify `ONEX_INPUT_TOPIC` matches test topic
- If offset issue: Reset consumer group offset
- If no messages in topic: Test not publishing correctly

---

### Scenario C: Message Received But Validation Failed
**Symptoms**:
- "callback invoked" found
- "parsing message value" found
- "validating payload" found
- **"introspection event parsed successfully" NOT found**
- "message is not a valid introspection event, skipping" found

**Root Causes**:
1. JSON structure doesn't match `ModelNodeIntrospectionEvent` schema
2. Missing required fields (node_id, node_type, etc.)
3. Type mismatch (e.g., UUID as string instead of UUID)
4. Pydantic validation failing

**Diagnosis**:
```bash
# Check validation error details
grep -A 10 "validation_error_count" /tmp/runtime_test_logs.txt

# Capture raw message from Kafka
docker exec omnibase-infra-redpanda rpk topic consume \
  dev.onex.evt.node-introspection.v1 \
  --num 1 \
  --brokers redpanda:9092 \
  --format json | jq .
```

**Fixes**:
- Compare raw Kafka message to `ModelNodeIntrospectionEvent` schema
- Fix test fixture to match expected schema
- Add missing required fields

---

### Scenario D: Validation Succeeded But Envelope Creation Failed
**Symptoms**:
- "introspection event parsed successfully" found
- **"routing to introspection dispatcher" NOT found**
- Possible exception between validation and routing

**Root Causes**:
1. `ModelEventEnvelope` creation failed
2. Correlation ID extraction logic failed
3. Exception thrown before dispatcher call

**Diagnosis**:
```bash
# Check for exceptions after validation
grep -A 20 "parsed successfully" /tmp/runtime_test_logs.txt | grep -i "error\|exception"

# Check envelope creation logs
grep -i "event envelope created" /tmp/runtime_test_logs.txt
```

**Fixes**:
- Add exception handling around envelope creation
- Fix correlation ID extraction from headers
- Verify `ModelEventEnvelope` constructor signature

---

### Scenario E: Dispatcher Routed But Handler Failed
**Symptoms**:
- "routing to introspection dispatcher" found
- **"introspection event processed successfully" NOT found**
- "introspection event processing failed" found (with error message)

**Root Causes**:
1. Handler execution raised exception
2. Projector write failed (database error)
3. Handler returned unsuccessful result

**Diagnosis**:
```bash
# Check dispatcher error messages
grep -i "processing failed" /tmp/runtime_test_logs.txt

# Check for database errors
grep -i "postgresql\|database\|asyncpg" /tmp/runtime_test_logs.txt

# Check handler execution details
grep -A 30 "routing to introspection dispatcher" /tmp/runtime_test_logs.txt
```

**Fixes**:
- If database error: Check PostgreSQL connection from container
- If handler exception: Fix handler logic or add error handling
- If projector failed: Verify schema exists and permissions correct

---

### Scenario F: Handler Succeeded But Projection Not Visible
**Symptoms**:
- "introspection event processed successfully" found
- Test assertion fails: "Runtime did not create projection"
- No errors in logs

**Root Causes**:
1. Projection created but test querying wrong database
2. Database connection not shared (transaction isolation)
3. Projection created with wrong entity_id
4. Race condition: projection not committed before test queries

**Diagnosis**:
```bash
# Check if projection exists in database
docker exec omnibase-infra-postgres psql \
  -U postgres \
  -d omninode_bridge \
  -c "SELECT entity_id, node_type, created_at FROM registration_projections ORDER BY created_at DESC LIMIT 5;"

# Compare entity_id in logs vs test
grep -i "node_id" /tmp/runtime_test_logs.txt
grep -i "unique_node_id" /tmp/test_output.log
```

**Fixes**:
- If projection exists: Test querying wrong database or using wrong entity_id
- If projection missing: Projector write failed silently
- Add delay before test query to avoid race condition

---

## 📊 Step 7: Generate Diagnostic Report

Create a comprehensive report with findings:

```bash
# Generate structured report
cat > /tmp/e2e_diagnostic_report.md << 'EOF'
# E2E Test Diagnostic Report
**Date**: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
**Test**: test_introspection_event_processed_by_runtime

## Test Result
<!-- PASSED or FAILED -->

## Container Startup Status
### Required Startup Messages
- [ ] Container wiring complete
- [ ] HandlerNodeIntrospected resolved
- [ ] Dispatcher created and wired
- [ ] Consumer started successfully
- [ ] Runtime kernel banner displayed

### Startup Errors
<!-- Paste any errors from startup logs -->

## Event Processing Status
### Pipeline Checkpoints
- [ ] Callback invoked (message received)
- [ ] Message parsed as bytes/string/dict
- [ ] Validation attempted
- [ ] Validation succeeded
- [ ] Envelope created
- [ ] Dispatcher routing attempted
- [ ] Handler execution succeeded

### Processing Errors
<!-- Paste any errors from test logs -->

## Identified Scenario
<!-- A, B, C, D, E, or F from above -->

## Root Cause Hypothesis
<!-- Your analysis of the failure -->

## Correlation IDs
<!-- Paste unique correlation IDs from logs -->

## Evidence
### Startup Logs
```
<!-- Paste relevant startup log excerpts -->
```

### Test Logs
```
<!-- Paste relevant test log excerpts -->
```

### Kafka Messages
```
<!-- Paste raw Kafka message if inspected -->
```

### Database State
```
<!-- Paste query results if checked -->
```

## Recommended Fix
<!-- Specific code changes or configuration updates -->

## Next Steps
<!-- Additional investigation or verification needed -->
EOF

# Open report for editing
${EDITOR:-nano} /tmp/e2e_diagnostic_report.md
```

---

## 🔧 Step 8: Apply Fix Based on Scenario

Once you've identified the scenario and root cause, apply the appropriate fix:

### Fix Template: Handler Wiring Issue (Scenario A - S1)

```python
# File: src/omnibase_infra/runtime/container_wiring.py
# Problem: HandlerNodeIntrospected not resolving from container

# Check that wire_registration_handlers() includes:
async def wire_registration_handlers(
    container: ModelONEXContainer,
    db_pool: asyncpg.Pool,
) -> None:
    """Wire registration handlers with database pool."""
    # Ensure ProjectionReaderRegistration is registered
    projection_reader = ProjectionReaderRegistration(db_pool)
    container.register_singleton(
        "projection_reader_registration",
        projection_reader,
    )

    # Ensure ProjectorRegistration is registered
    projector = ProjectorRegistration(db_pool)
    container.register_singleton(
        "projector_registration",
        projector,
    )

    # Register HandlerNodeIntrospected
    handler = HandlerNodeIntrospected(
        projection_reader=projection_reader,
        projector=projector,
    )
    container.register_singleton(
        "handler_node_introspected",
        handler,
    )
```

### Fix Template: Topic Mismatch (Scenario B)

```bash
# File: docker/docker-compose.e2e.yml
# Problem: Runtime subscribing to different topic than test publishes to

# Verify environment variable consistency:
services:
  runtime:
    environment:
      # Must match topic in test fixture
      ONEX_INPUT_TOPIC: dev.onex.evt.node-introspection.v1
```

```python
# File: tests/integration/registration/e2e/test_runtime_e2e.py
# Problem: Test publishing to different topic

# Verify topic matches docker-compose:
RUNTIME_INPUT_TOPIC = os.getenv("ONEX_INPUT_TOPIC", "dev.onex.evt.node-introspection.v1")

await real_kafka_event_bus.publish(
    topic=RUNTIME_INPUT_TOPIC,  # Use same variable
    key=str(unique_node_id).encode("utf-8"),
    value=json.dumps(event_json).encode("utf-8"),
)
```

### Fix Template: Schema Mismatch (Scenario C)

```python
# File: tests/integration/registration/e2e/test_runtime_e2e.py
# Problem: Introspection event missing required fields

@pytest.fixture
def introspection_event(unique_node_id: UUID) -> ModelNodeIntrospectionEvent:
    """Create a valid introspection event for testing."""
    return ModelNodeIntrospectionEvent(
        node_id=unique_node_id,
        node_type=EnumNodeKind.EFFECT.value,  # Must be .value not EnumNodeKind
        node_version="1.0.0",
        capabilities=ModelNodeCapabilities(),  # Required field
        endpoints={  # Required field
            "health": f"http://test-node-{unique_node_id.hex[:8]}:8080/health",
            "api": f"http://test-node-{unique_node_id.hex[:8]}:8080/api",
        },
        correlation_id=uuid4(),
        timestamp=datetime.now(UTC),
    )
```

---

## 🎯 Step 9: Verify Fix

After applying fix:

```bash
# Rebuild container with fix
cd "$(git rev-parse --show-toplevel)/docker"
docker compose -f docker-compose.e2e.yml build runtime
docker compose -f docker-compose.e2e.yml up -d runtime

# Wait for healthy
sleep 15

# Rerun test
cd "$(git rev-parse --show-toplevel)"
poetry run pytest \
  tests/integration/registration/e2e/test_runtime_e2e.py::TestRuntimeE2EFlow::test_introspection_event_processed_by_runtime \
  -v -s

# Check result
echo $?  # Should be 0 if PASSED
```

**Success criteria**:
- ✅ Test exits with code 0 (PASSED)
- ✅ Log shows "introspection event processed successfully"
- ✅ Projection exists in database
- ✅ No errors in container logs

---

## 📚 Additional Diagnostic Commands

### Kafka Inspection
```bash
# List all topics
docker exec omnibase-infra-redpanda rpk topic list

# Describe topic details
docker exec omnibase-infra-redpanda rpk topic describe dev.onex.evt.node-introspection.v1

# Check consumer groups
docker exec omnibase-infra-redpanda rpk group list

# Check consumer group lag
docker exec omnibase-infra-redpanda rpk group describe onex-runtime-e2e-introspection
```

### PostgreSQL Inspection
```bash
# Connect to database
docker exec -it omnibase-infra-postgres psql -U postgres -d omninode_bridge

# Query projections
SELECT
  entity_id,
  node_type,
  node_version,
  registration_phase,
  created_at
FROM registration_projections
ORDER BY created_at DESC
LIMIT 10;

# Check table schema
\d registration_projections

# Exit
\q
```

### Container Debugging
```bash
# Shell into runtime container
docker exec -it omnibase-infra-runtime /bin/bash

# Inside container:
# - Check Python environment
python --version
poetry env info

# - Check contract files
ls -la /app/contracts/

# - Check environment variables
env | grep ONEX
env | grep KAFKA
env | grep POSTGRES

# Exit
exit
```

### Network Debugging
```bash
# Check if runtime can reach Kafka
docker exec omnibase-infra-runtime nc -zv redpanda 9092

# Check if runtime can reach PostgreSQL
docker exec omnibase-infra-runtime nc -zv postgres 5432

# Check DNS resolution
docker exec omnibase-infra-runtime nslookup redpanda
```

---

## 🚀 Quick Reference

### One-Command Full Diagnostic
```bash
cd "$(git rev-parse --show-toplevel)"

# Rebuild, restart, test, capture logs
(
  cd docker && \
  docker compose -f docker-compose.e2e.yml build runtime && \
  docker compose -f docker-compose.e2e.yml up -d runtime && \
  sleep 20 && \
  docker compose -f docker-compose.e2e.yml logs runtime > /tmp/runtime_startup.log
) && \
poetry run pytest tests/integration/registration/e2e/test_runtime_e2e.py::TestRuntimeE2EFlow::test_introspection_event_processed_by_runtime -v -s 2>&1 | tee /tmp/test_output.log && \
(
  cd docker && \
  docker compose -f docker-compose.e2e.yml logs runtime > /tmp/runtime_full_logs.log
)

# Analyze results
echo "=== Test Result ==="
grep -i "PASSED\|FAILED" /tmp/test_output.log
echo ""
echo "=== Startup Messages ==="
grep -i "consumer started successfully\|dispatcher created" /tmp/runtime_startup.log
echo ""
echo "=== Processing Messages ==="
grep -i "callback invoked\|parsed successfully\|processed successfully" /tmp/runtime_full_logs.log
echo ""
echo "=== Errors ==="
grep -i "error\|exception\|failed" /tmp/runtime_full_logs.log | head -20
```

---

## 📞 Support

If diagnosis is unclear:
1. Capture logs: `/tmp/runtime_startup.log`, `/tmp/runtime_full_logs.log`, `/tmp/test_output.log`
2. Generate diagnostic report (Step 7)
3. Include correlation IDs from logs
4. Share scenario identification (A-F)
5. Include raw Kafka message dump

**Log locations**:
- Container logs: `docker compose -f docker/docker-compose.e2e.yml logs runtime`
- Test output: `/tmp/test_output.log`
- Diagnostic report: `/tmp/e2e_diagnostic_report.md`

---

## ✅ Success Checklist

Before marking diagnostic complete:

- [ ] Container rebuilt with latest code
- [ ] Container healthy and accessible
- [ ] Startup logs captured and analyzed
- [ ] All required startup messages present
- [ ] Test executed with verbose logging
- [ ] Container logs during test captured
- [ ] Failure scenario identified (A-F)
- [ ] Root cause hypothesis documented
- [ ] Evidence gathered (logs, Kafka, database)
- [ ] Fix applied and tested
- [ ] Test passes consistently (3+ runs)

**Estimated total time**: 15-30 minutes depending on scenario complexity

---

**End of Manual**
