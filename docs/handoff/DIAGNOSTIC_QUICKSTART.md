# E2E Diagnostic Quickstart Guide

**Context**: Runtime E2E test failing - introspection events not creating projections
**Enhanced Logging**: Added comprehensive logging to kernel.py for pipeline tracing
**Goal**: Identify exact failure point and fix

---

## 🚀 Quick Start (5 Minutes)

### Option 1: Automated Diagnostic Script

```bash
cd /workspace/omnibase_infra3

# Run automated diagnostic
./scripts/diagnose_e2e.sh --rebuild --full-report
```

**This script will**:
1. Rebuild Docker container with enhanced logging
2. Verify container health
3. Capture startup logs
4. Run E2E test with verbose output
5. Analyze logs for failure scenario
6. Generate comprehensive diagnostic report

**Output**: Scenario identification + recommended fix

---

### Option 2: Manual Diagnostic (15-30 Minutes)

Follow the comprehensive manual:

```bash
cd /workspace/omnibase_infra3

# Open manual
cat docs/handoff/E2E_DIAGNOSTIC_MANUAL.md
```

**Manual includes**:
- Step-by-step diagnostic procedures
- Log search commands for each pipeline stage
- 6 failure scenarios with fixes
- Database and Kafka inspection commands
- Diagnostic report template

---

## 📊 Understanding the Event Pipeline

```
┌────────────────────────────────────────────────────────────┐
│                    Event Processing Flow                    │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Kafka Message Arrives                                  │
│     ↓ (Log: "callback invoked")                            │
│                                                             │
│  2. JSON Parsing                                            │
│     ↓ (Log: "parsing message value")                       │
│                                                             │
│  3. Model Validation                                        │
│     ↓ (Log: "validating payload")                          │
│                                                             │
│  4. Validation Success                                      │
│     ↓ (Log: "introspection event parsed successfully")     │
│                                                             │
│  5. Envelope Creation                                       │
│     ↓ (Log: "event envelope created")                      │
│                                                             │
│  6. Dispatcher Routing                                      │
│     ↓ (Log: "routing to introspection dispatcher")         │
│                                                             │
│  7. Handler Execution                                       │
│     ↓ (Log: "introspection event processed successfully")  │
│                                                             │
│  8. Projection Persisted                                    │
│     ✓ (Database: registration_projections table)           │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

**Each step has logging** - find the last successful step to identify the break point.

---

## 🎯 Common Failure Scenarios

### Scenario A: Consumer Never Started
**Symptom**: No "consumer started successfully" in startup logs
**Cause**: Handler wiring failed, wrong event bus type, or Kafka unreachable
**Fix**: Check PostgreSQL connection, verify KAFKA_BOOTSTRAP_SERVERS

### Scenario B: No Messages Received
**Symptom**: Consumer started but no "callback invoked"
**Cause**: Topic mismatch, consumer offset issue, or Kafka routing problem
**Fix**: Verify ONEX_INPUT_TOPIC matches test topic, reset consumer group

### Scenario C: Validation Failed
**Symptom**: "callback invoked" but NOT "parsed successfully"
**Cause**: Message schema doesn't match ModelNodeIntrospectionEvent
**Fix**: Compare Kafka message to expected schema, add missing fields

### Scenario D: Envelope Creation Failed
**Symptom**: "parsed successfully" but NOT "routing to dispatcher"
**Cause**: ModelEventEnvelope construction failed
**Fix**: Check envelope creation logic, verify correlation_id extraction

### Scenario E: Handler Failed
**Symptom**: "routing to dispatcher" but NOT "processed successfully"
**Cause**: Handler execution raised exception, projector write failed
**Fix**: Check database connectivity, handler logic, projector errors

### Scenario F: Projection Not Visible
**Symptom**: "processed successfully" but test assertion fails
**Cause**: Database connection mismatch, timing issue, wrong entity_id
**Fix**: Verify test queries correct database, check entity_id matching

---

## 🔍 Key Log Search Commands

After running diagnostic (logs in `/tmp/`):

```bash
# Check startup messages
grep -i "consumer started successfully" /tmp/runtime_startup.log

# Check event processing
grep -i "callback invoked" /tmp/runtime_full_logs.log
grep -i "parsed successfully" /tmp/runtime_full_logs.log
grep -i "processed successfully" /tmp/runtime_full_logs.log

# Check for errors
grep -i "error\|exception\|failed" /tmp/runtime_full_logs.log | grep -v WARNING

# Extract correlation IDs for tracing
grep -oP "correlation_id=\K[a-f0-9-]+" /tmp/runtime_full_logs.log | sort -u
```

---

## 📁 Generated Files

After running `./scripts/diagnose_e2e.sh --full-report`:

```
/tmp/
├── runtime_startup.log           # Container startup logs
├── test_output.log                # Pytest output with assertions
├── runtime_full_logs.log          # Complete container logs during test
└── e2e_diagnostic_report.md       # Full diagnostic report
```

**Report includes**:
- Scenario identification
- Root cause analysis
- Pipeline checkpoint status
- Error excerpts
- Recommended fix
- Database state
- Next steps

---

## 🔧 Quick Fixes by Scenario

### If Scenario A (Consumer Never Started)

```bash
# Check handler wiring
cd /workspace/omnibase_infra3
grep -A 20 "wire_registration_handlers" src/omnibase_infra/runtime/container_wiring.py

# Verify database connection
docker exec omnibase-infra-postgres psql -U postgres -d omninode_bridge -c "SELECT 1;"
```

### If Scenario B (No Messages Received)

```bash
# Check topic in Kafka
docker exec omnibase-infra-redpanda rpk topic consume \
  dev.onex.evt.node-introspection.v1 \
  --num 10 \
  --brokers redpanda:9092

# Reset consumer group offset
docker exec omnibase-infra-redpanda rpk group seek onex-runtime-e2e-introspection \
  --to start \
  --topics dev.onex.evt.node-introspection.v1
```

### If Scenario C (Validation Failed)

```bash
# Capture raw Kafka message
docker exec omnibase-infra-redpanda rpk topic consume \
  dev.onex.evt.node-introspection.v1 \
  --num 1 \
  --format json \
  --brokers redpanda:9092 | jq .

# Compare to ModelNodeIntrospectionEvent schema
cat src/omnibase_infra/models/registration/model_node_introspection_event.py
```

### If Scenario E (Handler Failed)

```bash
# Check database connectivity from container
docker exec omnibase-infra-runtime nc -zv postgres 5432

# Query projection table directly
docker exec omnibase-infra-postgres psql -U postgres -d omninode_bridge \
  -c "SELECT * FROM registration_projections ORDER BY created_at DESC LIMIT 5;"
```

---

## ✅ Verification After Fix

```bash
cd /workspace/omnibase_infra3

# Rebuild with fix
cd docker
docker compose -f docker-compose.e2e.yml build runtime
docker compose -f docker-compose.e2e.yml up -d runtime
sleep 20

# Rerun test
cd ..
poetry run pytest \
  tests/integration/registration/e2e/test_runtime_e2e.py::TestRuntimeE2EFlow::test_introspection_event_processed_by_runtime \
  -v

# Check exit code
echo "Exit code: $?"  # Should be 0 for PASSED
```

**Success criteria**:
- ✅ Exit code 0 (test passed)
- ✅ "processed successfully" in logs
- ✅ Projection exists in database
- ✅ No errors in container logs

**Run 3+ times** to ensure consistency.

---

## 📞 Next Steps

1. **Run automated diagnostic**:
   ```bash
   ./scripts/diagnose_e2e.sh --rebuild --full-report
   ```

2. **Review report**:
   ```bash
   cat /tmp/e2e_diagnostic_report.md
   ```

3. **Identify scenario** (A, B, C, D, E, or F)

4. **Apply recommended fix** from report

5. **Verify fix**:
   ```bash
   ./scripts/diagnose_e2e.sh
   ```

6. **If still failing**, refer to detailed manual:
   ```bash
   cat docs/handoff/E2E_DIAGNOSTIC_MANUAL.md
   ```

---

## 🎓 Understanding Enhanced Logging

The kernel.py now logs at these critical points:

**Startup**:
- Container wiring status
- Handler resolution status
- Dispatcher creation status
- Consumer subscription status

**Event Processing (per message)**:
- Callback invocation (with offset, partition, topic)
- Message parsing (value type: bytes/string/dict)
- Validation attempt and result
- Envelope creation
- Dispatcher routing (with timing)
- Handler execution result (with timing)

**Errors**:
- JSON decode errors
- Pydantic validation errors
- Handler execution errors
- All with correlation IDs for tracing

**Each log includes**:
- Correlation ID (for message tracing)
- Timing information (duration_seconds)
- Contextual data (node_id, topic, etc.)

---

## 📚 Reference Documentation

- **Full Manual**: `docs/handoff/E2E_DIAGNOSTIC_MANUAL.md`
- **Diagnostic Script**: `scripts/diagnose_e2e.sh`
- **Test File**: `tests/integration/registration/e2e/test_runtime_e2e.py`
- **Kernel Code**: `src/omnibase_infra/runtime/kernel.py` (lines 706-930)
- **Docker Compose**: `docker/docker-compose.e2e.yml`

---

**Estimated Time**:
- Automated diagnostic: **5 minutes**
- Manual diagnostic: **15-30 minutes**
- Fix application: **5-15 minutes** (depends on scenario)
- Total: **15-50 minutes** depending on complexity

---

**Questions?** Review the full manual at `docs/handoff/E2E_DIAGNOSTIC_MANUAL.md`
