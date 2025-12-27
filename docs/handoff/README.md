# E2E Test Diagnostic Package

**Status**: Runtime E2E test failing - event processing pipeline investigation
**Date**: 2025-12-27
**Ticket**: OMN-892

---

## 📦 What's Included

This diagnostic package provides tools to identify and fix the event processing failure in the runtime E2E test.

### Files Created

1. **`DIAGNOSTIC_QUICKSTART.md`** - Start here (5-minute quickstart)
2. **`E2E_DIAGNOSTIC_MANUAL.md`** - Comprehensive manual (step-by-step guide)
3. **`scripts/diagnose_e2e.sh`** - Automated diagnostic script (run from project root)

---

## 🚀 Quick Start

### 1. Run Automated Diagnostic

```bash
# Navigate to the repository root (if not already there)
cd "$(git rev-parse --show-toplevel)"

# Run diagnostic with full report
./scripts/diagnose_e2e.sh --rebuild --full-report
```

**This will**:
- ✅ Rebuild Docker container with enhanced logging
- ✅ Verify container health
- ✅ Run E2E test
- ✅ Analyze logs for failure point
- ✅ Identify scenario (A-F)
- ✅ Recommend specific fix
- ✅ Generate comprehensive report

**Output**: `/tmp/e2e_diagnostic_report.md`

### 2. Review Results

```bash
# View diagnostic report
cat /tmp/e2e_diagnostic_report.md

# Check identified scenario
grep "Identified Scenario" /tmp/e2e_diagnostic_report.md

# Check recommended fix
grep -A 10 "Recommended Fix" /tmp/e2e_diagnostic_report.md
```

### 3. Apply Fix

Based on the identified scenario, apply the recommended fix from the report.

**Common fixes**:
- **Scenario A**: Fix handler wiring or Kafka connection
- **Scenario B**: Fix topic name mismatch or reset consumer offset
- **Scenario C**: Fix message schema to match ModelNodeIntrospectionEvent
- **Scenario D**: Fix envelope creation logic
- **Scenario E**: Fix handler execution or database connectivity
- **Scenario F**: Fix database query or timing issue

### 4. Verify Fix

```bash
# Rerun diagnostic after applying fix
./scripts/diagnose_e2e.sh

# Test should pass (exit code 0)
echo $?
```

---

## 📖 Documentation Guide

### For Quick Diagnosis (5 minutes)
→ Read `DIAGNOSTIC_QUICKSTART.md`

### For Detailed Investigation (15-30 minutes)
→ Read `E2E_DIAGNOSTIC_MANUAL.md`

### For Automated Diagnosis
→ Run `./scripts/diagnose_e2e.sh` from project root

---

## 🎯 What Was Changed

### Enhanced Logging Added to `kernel.py`

**Location**: `src/omnibase_infra/runtime/kernel.py` (lines 706-930)

**New logging points**:

1. **Startup logging**:
   - Container wiring status
   - Handler resolution (`HandlerNodeIntrospected`)
   - Dispatcher creation (`DispatcherNodeIntrospected`)
   - Consumer subscription status
   - Startup banner with configuration

2. **Event processing logging** (per message):
   - Callback invocation with Kafka metadata
   - Message parsing (bytes → dict)
   - Model validation attempt
   - Validation success/failure
   - Envelope creation
   - Dispatcher routing with timing
   - Handler execution result with timing

3. **Error logging**:
   - JSON decode errors
   - Pydantic validation errors
   - Handler execution errors
   - All with correlation IDs for tracing

**Log levels**:
- `DEBUG`: Detailed pipeline steps
- `INFO`: Major milestones (consumer started, event processed)
- `WARNING`: Recoverable errors (validation failures)
- `ERROR`: Critical failures (handler exceptions)

**Structured logging**:
All logs include `extra={}` context with:
- Correlation IDs
- Timing information (duration_seconds)
- Node IDs, topics, partitions
- Error details

---

## 🔍 Event Processing Pipeline

The diagnostic tools trace this pipeline:

```
┌──────────────────────────────────────────────┐
│         Event Processing Pipeline             │
├──────────────────────────────────────────────┤
│                                               │
│  Kafka Message                                │
│    ↓                                          │
│  Callback Invoked        (Log checkpoint 1)  │
│    ↓                                          │
│  JSON Parsing            (Log checkpoint 2)  │
│    ↓                                          │
│  Model Validation        (Log checkpoint 3)  │
│    ↓                                          │
│  Validation Success      (Log checkpoint 4)  │
│    ↓                                          │
│  Envelope Creation       (Log checkpoint 5)  │
│    ↓                                          │
│  Dispatcher Routing      (Log checkpoint 6)  │
│    ↓                                          │
│  Handler Execution       (Log checkpoint 7)  │
│    ↓                                          │
│  Projection Persisted    (Database check)    │
│                                               │
└──────────────────────────────────────────────┘
```

**Diagnostic identifies** the last successful checkpoint → failure point.

---

## 🐛 Known Failure Scenarios

Based on log analysis, 6 scenarios are identified:

| Scenario | Description | Common Cause |
|----------|-------------|--------------|
| **A** | Consumer never started | Handler wiring failed |
| **B** | No messages received | Topic mismatch |
| **C** | Validation failed | Schema mismatch |
| **D** | Envelope creation failed | Logic error |
| **E** | Handler execution failed | Database error |
| **F** | Projection not visible | Query mismatch |

Each scenario has:
- Symptom description
- Root cause analysis
- Recommended fix
- Diagnostic commands

**See**: `E2E_DIAGNOSTIC_MANUAL.md` for detailed fixes.

---

## 🛠️ Diagnostic Script Usage

### Basic Usage

```bash
# Quick diagnostic (use existing container)
./scripts/diagnose_e2e.sh

# Full diagnostic with rebuild
./scripts/diagnose_e2e.sh --rebuild

# Generate comprehensive report
./scripts/diagnose_e2e.sh --full-report

# Full rebuild + report (recommended)
./scripts/diagnose_e2e.sh --rebuild --full-report
```

### Script Output

The script outputs:
- ✅/✗ status for each pipeline checkpoint
- Identified scenario (A-F)
- Root cause hypothesis
- Recommended fix
- Log file locations

**Exit codes**:
- `0`: Test PASSED
- `1`: Test FAILED (see scenario for fix)

### Generated Files

```
/tmp/
├── runtime_startup.log         # Container startup
├── test_output.log              # Pytest output
├── runtime_full_logs.log        # Complete container logs
└── e2e_diagnostic_report.md     # Diagnostic report (with --full-report)
```

---

## 📋 Diagnostic Checklist

Follow this checklist when diagnosing:

### Pre-Diagnostic
- [ ] Enhanced logging code deployed to kernel.py
- [ ] Docker daemon running
- [ ] Infrastructure services available (Postgres, Kafka, Consul)
- [ ] Poetry environment activated

### Diagnostic Steps
- [ ] Run automated diagnostic script
- [ ] Review startup logs for required messages
- [ ] Identify which pipeline checkpoint failed
- [ ] Map to failure scenario (A-F)
- [ ] Review recommended fix
- [ ] Check supporting evidence (Kafka, database)

### Fix Application
- [ ] Apply recommended fix to code
- [ ] Rebuild Docker container
- [ ] Restart runtime service
- [ ] Verify container health
- [ ] Rerun diagnostic script

### Verification
- [ ] Test passes (exit code 0)
- [ ] "processed successfully" in logs
- [ ] Projection exists in database
- [ ] No errors in container logs
- [ ] Test passes 3+ consecutive times

---

## 🎓 Understanding the Test

**Test**: `test_introspection_event_processed_by_runtime`
**Location**: `tests/integration/registration/e2e/test_runtime_e2e.py`

**What it does**:
1. Publishes `ModelNodeIntrospectionEvent` to Kafka topic
2. Waits for runtime to consume and process (max 30 seconds)
3. Queries PostgreSQL for projection
4. Asserts projection exists with correct data

**Why it's failing**:
The runtime is NOT creating the projection within 30 seconds, indicating:
- Event not reaching consumer, OR
- Consumer not processing event, OR
- Handler not executing correctly, OR
- Projector not writing to database

**Enhanced logging reveals** exactly which step breaks.

---

## 📊 Metrics to Watch

When running diagnostic, monitor:

### Timing Metrics
- Container startup time (should be < 40 seconds)
- Test execution time (should be < 45 seconds)
- Callback processing time (should be < 1 second)
- Handler execution time (should be < 500ms)

### Success Indicators
- All startup messages present
- Callback invocations match published messages
- Validation success rate 100%
- Handler success rate 100%
- Projection creation within 2 seconds

### Failure Indicators
- Missing startup messages
- Zero callback invocations
- Validation errors
- Handler exceptions
- Database errors

---

## 🔧 Troubleshooting Tips

### If Container Won't Start
```bash
# Check logs for startup errors
docker compose -f docker/docker-compose.e2e.yml logs runtime | head -100

# Verify dependencies healthy
docker compose -f docker/docker-compose.e2e.yml ps

# Check resource constraints
docker stats omnibase-infra-runtime
```

### If Test Times Out
```bash
# Increase timeout in test fixture
# Default: 30 seconds
# Consider: 60 seconds for slow systems

# Check Kafka lag
docker exec omnibase-infra-redpanda rpk group describe onex-runtime-e2e-introspection
```

### If Logs Are Unclear
```bash
# Increase log level to DEBUG
export ONEX_LOG_LEVEL=DEBUG

# Rebuild with debug logging
docker compose -f docker/docker-compose.e2e.yml build runtime
docker compose -f docker/docker-compose.e2e.yml up -d runtime
```

---

## 📞 Support

If diagnostic doesn't identify the issue:

1. **Capture all logs**:
   ```bash
   tar -czf /tmp/e2e_diagnostic_logs.tar.gz \
     /tmp/runtime_startup.log \
     /tmp/test_output.log \
     /tmp/runtime_full_logs.log \
     /tmp/e2e_diagnostic_report.md
   ```

2. **Include correlation IDs**:
   ```bash
   grep -oP "correlation_id=\K[a-f0-9-]+" /tmp/runtime_full_logs.log | sort -u
   ```

3. **Share scenario identification**:
   ```bash
   grep "Identified Scenario" /tmp/e2e_diagnostic_report.md
   ```

4. **Include raw Kafka message** (if available):
   ```bash
   docker exec omnibase-infra-redpanda rpk topic consume \
     dev.onex.evt.node-introspection.v1 \
     --num 1 --format json --brokers redpanda:9092
   ```

---

## ✅ Success Criteria

Diagnostic is complete when:

- [x] Enhanced logging deployed to kernel.py
- [ ] Automated diagnostic script run successfully
- [ ] Failure scenario identified (A-F)
- [ ] Root cause determined with evidence
- [ ] Fix applied and tested
- [ ] Test passes consistently (3+ runs)
- [ ] No errors in container logs
- [ ] Projection verified in database

---

## 🔄 Next Steps

1. **Run diagnostic**:
   ```bash
   ./scripts/diagnose_e2e.sh --rebuild --full-report
   ```

2. **Review quickstart**:
   ```bash
   cat docs/handoff/DIAGNOSTIC_QUICKSTART.md
   ```

3. **Follow scenario-specific fix** from report

4. **Verify fix works**:
   ```bash
   ./scripts/diagnose_e2e.sh
   ```

5. **Run full test suite**:
   ```bash
   poetry run pytest tests/integration/registration/e2e/ -v
   ```

---

**Questions?** See `E2E_DIAGNOSTIC_MANUAL.md` for comprehensive guidance.

**Estimated time**: 15-50 minutes depending on scenario complexity.
