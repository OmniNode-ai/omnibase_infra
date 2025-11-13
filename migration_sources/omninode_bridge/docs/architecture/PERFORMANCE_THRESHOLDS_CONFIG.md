# Performance Thresholds Configuration

**Status**: ✅ Implemented (PR Review Response)
**Location**: `tests/config/performance_thresholds.yaml`
**Test File**: `tests/integration/test_two_way_registration_e2e.py`

## Overview

Performance thresholds have been moved from hardcoded constants to a flexible configuration file, enabling environment-specific adjustments without code changes.

## What Changed

### Before (Hardcoded)

```python
PERFORMANCE_THRESHOLDS = {
    "introspection_broadcast_latency_ms": 50,
    "registry_processing_latency_ms": 100,
    "dual_registration_time_ms": 200,
    "heartbeat_interval_s": 30,
    "heartbeat_overhead_ms": 50,
}
```

### After (Configurable)

```python
def _load_performance_thresholds() -> dict[str, int]:
    """Load thresholds with priority: env vars > custom config > default config > hardcoded defaults"""
    # ... implementation ...

PERFORMANCE_THRESHOLDS = _load_performance_thresholds()
```

## Configuration Priority

1. **Individual Environment Variables** (Highest)
   - `TEST_INTROSPECTION_BROADCAST_LATENCY_MS`
   - `TEST_REGISTRY_PROCESSING_LATENCY_MS`
   - `TEST_DUAL_REGISTRATION_TIME_MS`
   - `TEST_HEARTBEAT_INTERVAL_S`
   - `TEST_HEARTBEAT_OVERHEAD_MS`

2. **Custom Config File Path**
   - `TEST_PERFORMANCE_CONFIG` environment variable

3. **Default Config File**
   - `tests/config/performance_thresholds.yaml`

4. **Hardcoded Defaults** (Fallback)
   - Used if config file not found

## Usage Examples

### Development (Default)

```bash
# Uses tests/config/performance_thresholds.yaml
pytest tests/integration/test_two_way_registration_e2e.py -m performance
```

### CI/CD (Relaxed Thresholds)

```bash
# Override via environment variables
export TEST_INTROSPECTION_BROADCAST_LATENCY_MS=100
export TEST_DUAL_REGISTRATION_TIME_MS=300
pytest tests/integration/ -m performance
```

### Custom Environment

```bash
# Use custom config file
export TEST_PERFORMANCE_CONFIG=/path/to/custom/thresholds.yaml
pytest tests/integration/
```

### Production Validation (Strict)

```yaml
# production_thresholds.yaml
introspection_broadcast_latency_ms: 25
registry_processing_latency_ms: 75
dual_registration_time_ms: 150
heartbeat_interval_s: 30
heartbeat_overhead_ms: 40
```

```bash
export TEST_PERFORMANCE_CONFIG=production_thresholds.yaml
pytest tests/integration/ -m performance
```

## Configuration Files

### Main Configuration

**File**: `tests/config/performance_thresholds.yaml`

```yaml
# Performance Thresholds Configuration for Two-Way Registration Tests
introspection_broadcast_latency_ms: 50
registry_processing_latency_ms: 100
dual_registration_time_ms: 200
heartbeat_interval_s: 30
heartbeat_overhead_ms: 50
```

### Documentation

**File**: `tests/config/README.md`

Complete documentation including:
- Configuration priority explanation
- Environment variable reference table
- Environment-specific examples (dev, CI/CD, production)
- Debugging guide
- Instructions for adding new thresholds

## Benefits

1. **Environment Flexibility**
   - Adjust thresholds for CI/CD, staging, production
   - No code changes required

2. **Team Collaboration**
   - Developers can override locally without affecting others
   - CI/CD pipelines can use relaxed thresholds

3. **Debugging Support**
   - Easy to temporarily relax thresholds for debugging
   - Clear error messages when config fails to load

4. **Documentation**
   - Self-documenting YAML with inline comments
   - Centralized threshold management

## CI/CD Integration

### GitHub Actions Example

```yaml
# .github/workflows/test.yml
jobs:
  test-performance:
    runs-on: ubuntu-latest
    env:
      TEST_INTROSPECTION_BROADCAST_LATENCY_MS: 100
      TEST_DUAL_REGISTRATION_TIME_MS: 300
    steps:
      - name: Run performance tests
        run: pytest tests/integration/ -m performance
```

### GitLab CI Example

```yaml
# .gitlab-ci.yml
test:performance:
  variables:
    TEST_INTROSPECTION_BROADCAST_LATENCY_MS: "100"
    TEST_DUAL_REGISTRATION_TIME_MS: "300"
  script:
    - pytest tests/integration/ -m performance
```

## Implementation Details

### Loader Function

```python
def _load_performance_thresholds() -> dict[str, int]:
    """
    Load performance thresholds from configuration file with environment overrides.

    Priority (highest to lowest):
    1. Individual environment variables (TEST_<THRESHOLD_NAME>)
    2. Custom config file path (TEST_PERFORMANCE_CONFIG env var)
    3. Default config file (tests/config/performance_thresholds.yaml)
    4. Hardcoded defaults
    """
    # Implementation includes:
    # - YAML config loading with error handling
    # - Environment variable overrides
    # - Fallback to hardcoded defaults
    # - Warning messages for config failures
```

### Error Handling

- **Missing Config File**: Falls back to hardcoded defaults
- **Invalid YAML**: Prints warning and uses defaults
- **Invalid Env Vars**: Prints warning and uses config/default value
- **All Failures**: Tests still run with sensible defaults

## Adding New Thresholds

1. **Update YAML Config**
   ```yaml
   new_metric_threshold_ms: 150
   ```

2. **Update Loader Defaults**
   ```python
   defaults = {
       # ... existing ...
       "new_metric_threshold_ms": 150,
   }
   ```

3. **Add Environment Override**
   ```python
   env_overrides = {
       # ... existing ...
       "new_metric_threshold_ms": "TEST_NEW_METRIC_THRESHOLD_MS",
   }
   ```

4. **Use in Tests**
   ```python
   assert metric_value < PERFORMANCE_THRESHOLDS["new_metric_threshold_ms"]
   ```

5. **Update Documentation**
   - Add to `tests/config/README.md`
   - Add to this document

## Files Modified

- ✅ `tests/config/performance_thresholds.yaml` - Created
- ✅ `tests/config/README.md` - Created
- ✅ `tests/integration/test_two_way_registration_e2e.py` - Updated
- ✅ `docs/architecture/PERFORMANCE_THRESHOLDS_CONFIG.md` - Created (this file)

## Testing

```bash
# Verify config loads correctly
python3 -c "
from tests.integration.test_two_way_registration_e2e import PERFORMANCE_THRESHOLDS
print(PERFORMANCE_THRESHOLDS)
"

# Test environment override
TEST_INTROSPECTION_BROADCAST_LATENCY_MS=100 python3 -c "
from tests.integration.test_two_way_registration_e2e import PERFORMANCE_THRESHOLDS
print(f'Latency threshold: {PERFORMANCE_THRESHOLDS[\"introspection_broadcast_latency_ms\"]}ms')
"
```

Expected output:
```
{'introspection_broadcast_latency_ms': 50, 'registry_processing_latency_ms': 100, ...}
Latency threshold: 100ms
```

## Related Documentation

- **Test Configuration Guide**: `tests/config/README.md`
- **Two-Way Registration Tests**: `tests/integration/test_two_way_registration_e2e.py`
- **Performance Thresholds Config**: `tests/config/performance_thresholds.yaml`

## PR Review Response

This implementation addresses the medium priority issue from PR review:

> **Issue**: Performance thresholds are hardcoded in test code
> **Location**: `tests/integration/test_two_way_registration_e2e.py:93-99`
> **Solution**: Move to configuration file for environment flexibility
> **Result**: ✅ Complete - Thresholds now in `tests/config/performance_thresholds.yaml` with env var overrides
