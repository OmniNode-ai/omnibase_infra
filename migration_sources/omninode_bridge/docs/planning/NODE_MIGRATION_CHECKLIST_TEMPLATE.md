# Node Migration Checklist: <NODE_NAME>

**Node**: `<node_name>`
**Priority**: [HIGH | MEDIUM | LOW]
**Assigned To**: `<developer_name>`
**Start Date**: `YYYY-MM-DD`
**Target Date**: `YYYY-MM-DD`

---

## Pre-Migration Assessment

### Current State Analysis
- [ ] **LOC Count**: _____ lines
- [ ] **Current Features Documented**:
  ```
  - [ ] Circuit breaker (manual: ___ LOC | NodeEffect built-in)
  - [ ] Retry logic (manual: ___ LOC | NodeEffect built-in)
  - [ ] Health checks (manual: ___ LOC | MixinHealthCheck candidate)
  - [ ] Metrics tracking (manual: ___ LOC | MixinMetrics candidate)
  - [ ] Event consumption (manual: ___ LOC | MixinEventDrivenNode candidate)
  - [ ] DLQ handling (manual: ___ LOC | advanced_features candidate)
  - [ ] Security validation (manual: ___ LOC | advanced_features candidate)
  - [ ] Other: _______________ (manual: ___ LOC)
  ```

### Baseline Metrics Captured
- [ ] **Test Results**:
  - Unit tests: ___/___  passed
  - Integration tests: ___/___ passed
  - Performance tests: ___/___ passed
- [ ] **Performance Baseline**:
  - Average operation latency: ___ms
  - P95 latency: ___ms
  - Throughput: ___/sec
  - Memory usage: ___MB
- [ ] **Test Coverage**: ___%

### Backup Created
- [ ] **Backup Location**: `src/omninode_bridge/nodes/<node_name>.backup/`
- [ ] **Backup Verified**: Can restore from backup
- [ ] **Git Branch Created**: `feat/migrate-<node_name>-to-mixins`

---

## Contract Enhancement

### Mixin Selection
Selected mixins for this node:

- [ ] **MixinHealthCheck** (replaces manual health checks)
  - Components to monitor:
    - [ ] `database` (critical: true, timeout: 5.0s)
    - [ ] `kafka_consumer` (critical: false, timeout: 3.0s)
    - [ ] `_____________` (critical: ___, timeout: ___s)

- [ ] **MixinMetrics** (replaces manual metrics tracking)
  - Metrics to collect:
    - [ ] Latency (percentiles: [50, 95, 99])
    - [ ] Throughput (requests/second)
    - [ ] Error rates (by error type)
    - [ ] Custom: _______________

- [ ] **MixinEventDrivenNode** (replaces manual Kafka consumer)
  - Topics to consume:
    - [ ] `workflow-started`
    - [ ] `workflow-completed`
    - [ ] `_______________`
  - Consumer group: `_______________`

- [ ] **MixinEventBus** (replaces manual Kafka producer)
  - Topics to publish:
    - [ ] `_______________`

- [ ] **MixinServiceRegistry** (adds service discovery)
  - Service name: `_______________`
  - Consul host: `${CONSUL_HOST}`
  - Health check interval: ___s

- [ ] **MixinLogData** (adds structured logging)
  - Log level: `INFO`
  - Correlation tracking: `true`

- [ ] **MixinSensitiveFieldRedaction** (adds secret redaction)
  - Fields to redact:
    - [ ] `password`
    - [ ] `api_key`
    - [ ] `secret`
    - [ ] `_______________`

- [ ] **Other**: `_______________`

### Advanced Features Configuration
- [ ] **Circuit Breaker** (NodeEffect built-in):
  ```yaml
  circuit_breaker:
    enabled: true
    failure_threshold: 5
    recovery_timeout_ms: 60000
  ```

- [ ] **Retry Policy** (NodeEffect built-in):
  ```yaml
  retry_policy:
    enabled: true
    max_attempts: 3
    backoff_multiplier: 2.0
    retryable_status_codes: [429, 500, 502, 503]
  ```

- [ ] **Dead Letter Queue**:
  ```yaml
  dead_letter_queue:
    enabled: true
    max_retries: 3
    topic_suffix: ".dlq"
    alert_threshold: 100
  ```

- [ ] **Transactions** (NodeEffect built-in):
  ```yaml
  transactions:
    enabled: true
    isolation_level: "READ_COMMITTED"
    timeout_seconds: 30
  ```

- [ ] **Security Validation**:
  ```yaml
  security_validation:
    enabled: true
    sanitize_inputs: true
    validate_sql: true
  ```

### Contract Update
- [ ] **Contract File Updated**: `src/omninode_bridge/nodes/<node_name>/v1_0_0/contract.yaml`
- [ ] **Mixins Section Added** (see above selections)
- [ ] **Advanced Features Section Added** (see above configurations)
- [ ] **Contract Validated**: `omninode-generate --contract <path> --validate-only`
- [ ] **No Validation Errors**

---

## Code Generation

### Regeneration
- [ ] **Generation Command**:
  ```bash
  omninode-generate \
    --contract src/omninode_bridge/nodes/<node_name>/v1_0_0/contract.yaml \
    --enable-mixins \
    --enable-llm \
    --validation-level strict \
    --output src/omninode_bridge/nodes/<node_name>/v1_0_0/ \
    --overwrite \
    --backup
  ```

- [ ] **Generation Successful** (exit code 0)
- [ ] **No Generation Errors**
- [ ] **Generation Time**: ___s (target: < 5s)

### Code Review
- [ ] **Imports Correct**:
  - [ ] All mixin imports present
  - [ ] No import errors
  - [ ] Imports organized correctly

- [ ] **Class Definition Correct**:
  - [ ] Inherits from NodeEffect + Mixins
  - [ ] Docstring comprehensive
  - [ ] Type annotations present

- [ ] **Methods Generated**:
  - [ ] `__init__()` with mixin config
  - [ ] `initialize()` with mixin setup
  - [ ] `shutdown()` with cleanup
  - [ ] Health check methods (if MixinHealthCheck)
  - [ ] Event handlers (if MixinEventDrivenNode)
  - [ ] Business logic methods

- [ ] **Configuration Correct**:
  - [ ] Mixin configs initialized
  - [ ] Circuit breaker configured (if enabled)
  - [ ] Retry policy configured (if enabled)
  - [ ] All configs use proper types

### Comparison with Original
- [ ] **LOC Comparison**:
  - Original: _____ lines
  - Generated: _____ lines
  - Reduction: ___% (target: 30-50%)

- [ ] **Feature Parity**:
  - [ ] All original features preserved
  - [ ] No features removed
  - [ ] New features added via mixins
  - [ ] Behavioral equivalence verified

- [ ] **Code Quality**:
  - [ ] Cleaner than original
  - [ ] More maintainable
  - [ ] Better documented
  - [ ] Follows ONEX patterns

---

## Testing & Validation

### Syntax & Structure
- [ ] **Syntax Validation**: `python -m py_compile src/.../node.py`
- [ ] **AST Validation**: Passed
- [ ] **Import Resolution**: All imports resolve
- [ ] **Type Checking**: `mypy src/.../node.py` (0 errors)

### Unit Tests
- [ ] **Unit Tests Run**: `pytest tests/unit/nodes/<node_name>/ -v`
- [ ] **Results**:
  - Tests passed: ___/___
  - Tests failed: ___/___
  - Pass rate: ___%  (target: 100%)
  - Duration: ___s

- [ ] **New Tests Added** (if mixins require):
  - [ ] Health check tests
  - [ ] Metrics collection tests
  - [ ] Event handling tests
  - [ ] Integration tests

### Integration Tests
- [ ] **Integration Tests Run**: `pytest tests/integration/nodes/<node_name>/ -v`
- [ ] **Results**:
  - Tests passed: ___/___
  - Tests failed: ___/___
  - Pass rate: ___%  (target: 100%)
  - Duration: ___s

### Performance Tests
- [ ] **Performance Tests Run**: `pytest tests/performance/nodes/<node_name>/ -v`
- [ ] **Results**:
  - Average latency: ___ms (baseline: ___ms)
  - P95 latency: ___ms (baseline: ___ms)
  - Throughput: ___/sec (baseline: ___/sec)
  - Memory: ___MB (baseline: ___MB)
  - **Performance Regression**: [NONE | ACCEPTABLE | UNACCEPTABLE]

### Validation Pipeline
- [ ] **Syntax Validation**: ✅ Passed
- [ ] **AST Validation**: ✅ Passed
- [ ] **Import Validation**: ✅ Passed
- [ ] **Type Checking**: ✅ Passed
- [ ] **ONEX Compliance**: ✅ Passed
- [ ] **Security Scan**: ✅ Passed (0 warnings)

---

## Deployment Preparation

### Documentation
- [ ] **Node Documentation Updated**: `src/omninode_bridge/nodes/<node_name>/README.md`
- [ ] **Changelog Updated**: `CHANGELOG.md` (added migration entry)
- [ ] **Contract Documentation**: Examples updated
- [ ] **Migration Notes**: Added to docs/

### Code Review
- [ ] **Self-Review Complete**
- [ ] **Peer Review Requested**: @_______________
- [ ] **Peer Review Approved**: @_______________
- [ ] **All Comments Addressed**

### Version Control
- [ ] **Changes Committed**:
  ```bash
  git add src/omninode_bridge/nodes/<node_name>/
  git commit -m "feat: Migrate <node_name> to mixin-enhanced generation"
  ```
- [ ] **Tests Committed**
- [ ] **Documentation Committed**
- [ ] **PR Created**: #_____
- [ ] **CI/CD Passing**: All checks green

---

## Deployment

### Development Environment
- [ ] **Deployed to Dev**: `dev.omninode.ai`
- [ ] **Dev Health Check**: ✅ Healthy
- [ ] **Dev Smoke Tests**: ✅ Passed
- [ ] **Monitoring Configured**: Metrics visible in dashboard

### Staging Environment
- [ ] **Deployed to Staging**: `staging.omninode.ai`
- [ ] **Staging Health Check**: ✅ Healthy
- [ ] **Staging Integration Tests**: ✅ Passed
- [ ] **Load Testing**: ✅ Passed (no regressions)
- [ ] **Soak Testing**: ✅ Passed (24h stability)

### Production Environment
- [ ] **Deployment Plan Reviewed**
- [ ] **Rollback Plan Documented**
- [ ] **On-Call Engineer Notified**: @_______________
- [ ] **Deployed to Production**: `prod.omninode.ai`
- [ ] **Production Health Check**: ✅ Healthy
- [ ] **Production Smoke Tests**: ✅ Passed
- [ ] **Monitoring Alerts**: Configured and tested
- [ ] **No Immediate Issues**: 1 hour post-deployment

---

## Post-Deployment

### Monitoring (First 24 Hours)
- [ ] **Error Rates**: No increase
- [ ] **Latency**: No regression
- [ ] **Throughput**: No degradation
- [ ] **Memory Usage**: No leaks
- [ ] **CPU Usage**: No spikes
- [ ] **Logs**: No error patterns

### Metrics Collection
- [ ] **Regeneration Metrics Recorded**:
  ```yaml
  node_name: "<node_name>"
  original_loc: _____
  generated_loc: _____
  loc_reduction_percent: ____%
  mixins_applied: [___, ___, ___]
  test_pass_rate: 100%
  generation_time_seconds: ___
  deployment_success: true
  ```

- [ ] **Metrics Added to Summary Report**: `docs/planning/MIGRATION_SUMMARY.md`

### Cleanup
- [ ] **Backup Retention**: Keep for 30 days
- [ ] **Old Branch Cleanup**: Archive after 7 days
- [ ] **Documentation Archive**: Old docs saved to `docs/archive/`

### Retrospective
- [ ] **What Went Well**:
  ```
  -
  -
  ```

- [ ] **What Could Be Improved**:
  ```
  -
  -
  ```

- [ ] **Action Items**:
  ```
  - [ ]
  - [ ]
  ```

- [ ] **Lessons Learned Documented**: Added to `docs/LESSONS_LEARNED.md`

---

## Sign-Off

- [ ] **Developer**: `@_______________` (Date: _______)
- [ ] **Reviewer**: `@_______________` (Date: _______)
- [ ] **QA**: `@_______________` (Date: _______)
- [ ] **Operations**: `@_______________` (Date: _______)

**Migration Status**: [IN_PROGRESS | COMPLETED | BLOCKED | ROLLED_BACK]

---

**Notes**:
```
(Add any additional notes, issues encountered, workarounds, etc.)
```
