# Test Configuration

This directory contains configuration files for test suites, enabling environment-specific adjustments without code changes.

## Performance Thresholds Configuration

### File: `performance_thresholds.yaml`

Defines performance expectations for the two-way registration pattern tests.

### Configuration Priority (Highest to Lowest)

1. **Individual Environment Variables** - Override specific thresholds
2. **Custom Config File** - Override entire config file path
3. **Default Config File** - `tests/config/performance_thresholds.yaml`
4. **Hardcoded Defaults** - Fallback if config file not found

### Usage Examples

#### Using Default Configuration

Tests will automatically load from `tests/config/performance_thresholds.yaml`:

```bash
pytest tests/integration/test_two_way_registration_e2e.py
```

#### Using Custom Config File

Override the entire configuration file path:

```bash
export TEST_PERFORMANCE_CONFIG=/path/to/custom/thresholds.yaml
pytest tests/integration/test_two_way_registration_e2e.py
```

#### Overriding Individual Thresholds

Override specific thresholds via environment variables:

```bash
# Relax introspection latency for slower environments
export TEST_INTROSPECTION_BROADCAST_LATENCY_MS=100

# Increase dual registration time for high-latency networks
export TEST_DUAL_REGISTRATION_TIME_MS=300

# Run performance tests
pytest tests/integration/test_two_way_registration_e2e.py -m performance
```

#### CI/CD Environment Example

```yaml
# .github/workflows/test.yml
env:
  TEST_INTROSPECTION_BROADCAST_LATENCY_MS: 100  # Slower CI environment
  TEST_DUAL_REGISTRATION_TIME_MS: 300           # Account for network latency
  TEST_HEARTBEAT_OVERHEAD_MS: 75                # More lenient heartbeat check
```

### Available Environment Variables

| Variable | Description | Default | Unit |
|----------|-------------|---------|------|
| `TEST_PERFORMANCE_CONFIG` | Path to custom config file | `tests/config/performance_thresholds.yaml` | path |
| `TEST_INTROSPECTION_BROADCAST_LATENCY_MS` | Max latency for introspection broadcast | 50 | milliseconds |
| `TEST_REGISTRY_PROCESSING_LATENCY_MS` | Max latency for registry processing | 100 | milliseconds |
| `TEST_DUAL_REGISTRATION_TIME_MS` | Max time for dual registration | 200 | milliseconds |
| `TEST_HEARTBEAT_INTERVAL_S` | Standard heartbeat interval | 30 | seconds |
| `TEST_HEARTBEAT_OVERHEAD_MS` | Max heartbeat overhead | 50 | milliseconds |

### Environment-Specific Configurations

#### Development (Local)

```yaml
# tests/config/performance_thresholds.yaml (default)
introspection_broadcast_latency_ms: 50
registry_processing_latency_ms: 100
dual_registration_time_ms: 200
heartbeat_interval_s: 30
heartbeat_overhead_ms: 50
```

#### CI/CD (Slower)

```yaml
# tests/config/performance_thresholds.ci.yaml
introspection_broadcast_latency_ms: 100
registry_processing_latency_ms: 150
dual_registration_time_ms: 300
heartbeat_interval_s: 30
heartbeat_overhead_ms: 75
```

Usage:
```bash
export TEST_PERFORMANCE_CONFIG=tests/config/performance_thresholds.ci.yaml
pytest tests/integration/
```

#### Production Validation (Strict)

```yaml
# tests/config/performance_thresholds.prod.yaml
introspection_broadcast_latency_ms: 25
registry_processing_latency_ms: 75
dual_registration_time_ms: 150
heartbeat_interval_s: 30
heartbeat_overhead_ms: 40
```

Usage:
```bash
export TEST_PERFORMANCE_CONFIG=tests/config/performance_thresholds.prod.yaml
pytest tests/integration/ -m performance
```

### Debugging Configuration Issues

If thresholds aren't loading correctly, the test output will show warnings:

```
Warning: Failed to load performance thresholds from /path/to/config.yaml: [error details]
Using default performance thresholds
```

Check:
1. File path is correct and accessible
2. YAML syntax is valid
3. File permissions allow reading
4. Environment variables are set correctly

### Adding New Thresholds

1. Add the threshold to `performance_thresholds.yaml`:
   ```yaml
   new_threshold_ms: 100
   ```

2. Update the loader function in test file:
   ```python
   defaults = {
       "introspection_broadcast_latency_ms": 50,
       # ... existing thresholds ...
       "new_threshold_ms": 100,  # Add new threshold
   }

   env_overrides = {
       # ... existing overrides ...
       "new_threshold_ms": "TEST_NEW_THRESHOLD_MS",
   }
   ```

3. Update this README documentation

4. Use in tests:
   ```python
   assert latency_ms < PERFORMANCE_THRESHOLDS["new_threshold_ms"]
   ```
