# NODE_TYPE Environment Variable Validation

## Overview

This implementation adds robust NODE_TYPE environment variable validation to all generic Dockerfiles, preventing runtime failures with clear, actionable error messages.

## Problem Statement

Previously, containers would fail at runtime with unclear errors if NODE_TYPE was unset:
- Container startup would fail with obscure Python import errors
- Health checks would fail silently without indicating the root cause
- Debugging required inspecting container logs and understanding the runtime path

## Solution

Added comprehensive validation script that:
1. **Validates NODE_TYPE is set and non-empty**
2. **Validates NODE_TYPE contains only valid characters** (alphanumeric and underscore)
3. **Validates NODE_TYPE length** (3-100 characters)
4. **Optionally checks if the node module exists** (when available)
5. **Provides clear, actionable error messages** with examples

## Implementation

### Files Modified

1. **deployment/scripts/validate-node-type.sh** (NEW)
   - Validation script with comprehensive checks
   - Color-coded output (ERROR/WARNING/SUCCESS)
   - Detailed usage examples in error messages

2. **deployment/Dockerfile.generic-effect** (UPDATED)
   - Added validation script COPY in both development and production stages
   - Made script executable during build
   - Updated CMD to call validation before node startup
   - Updated HEALTHCHECK to validate before health check CLI

3. **deployment/Dockerfile.generic-orchestrator** (UPDATED)
   - Same changes as effect Dockerfile
   - Added ENV PYTHONPATH=/app/src for consistency

4. **deployment/Dockerfile.generic-reducer** (UPDATED)
   - Same changes as effect and orchestrator Dockerfiles

### Validation Script Location

```
/usr/local/bin/validate-node-type.sh
```

Script is copied into all containers during build and made executable.

## Usage

### Correct Configuration

**docker-compose.yml:**
```yaml
services:
  my-effect-node:
    build:
      dockerfile: deployment/Dockerfile.generic-effect
    environment:
      NODE_TYPE: database_adapter_effect
```

**docker run:**
```bash
docker run -e NODE_TYPE=database_adapter_effect <image>
```

### Error Messages

#### 1. Missing NODE_TYPE

```
ERROR: NODE_TYPE environment variable is not set

REQUIRED: Set NODE_TYPE to specify which node to run

Valid values by Dockerfile:
  - Dockerfile.generic-effect:
      database_adapter_effect, store_effect
  - Dockerfile.generic-orchestrator:
      codegen_orchestrator, workflow_orchestrator
  - Dockerfile.generic-reducer:
      codegen_metrics_reducer, aggregation_reducer

Example docker-compose.yml configuration:
  environment:
    - NODE_TYPE=database_adapter_effect

Example docker run command:
  docker run -e NODE_TYPE=database_adapter_effect <image>
```

#### 2. Invalid Characters

```
ERROR: NODE_TYPE contains invalid characters: 'invalid-chars!'

NODE_TYPE must contain only alphanumeric characters and underscores
Current value: invalid-chars!
```

#### 3. Too Short

```
ERROR: NODE_TYPE is too short: 'ab'

NODE_TYPE must be at least 3 characters long
```

#### 4. Success

```
SUCCESS: NODE_TYPE validated: 'database_adapter_effect' (module found at /app/src/omninode_bridge/nodes/database_adapter_effect/v1_0_0/node.py)
```

## Validation Points

The validation script runs at **two critical points**:

### 1. Container Startup (CMD)

```dockerfile
CMD ["sh", "-c", "/usr/local/bin/validate-node-type.sh && poetry run python -m omninode_bridge.nodes.${NODE_TYPE}.v1_0_0.node --log-level info"]
```

**Behavior:**
- Validates NODE_TYPE before starting the node
- Container fails immediately with clear error if validation fails
- Exit code 1 triggers container restart policy

### 2. Health Check (HEALTHCHECK)

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD /usr/local/bin/validate-node-type.sh && python -m omninode_bridge.nodes.health_check_cli ${NODE_TYPE}
```

**Behavior:**
- Validates NODE_TYPE before running health check CLI
- Health check fails immediately if NODE_TYPE is invalid
- Prevents misleading "healthy" status when NODE_TYPE is misconfigured

## Testing

### Manual Testing

```bash
# Test missing NODE_TYPE
bash deployment/scripts/validate-node-type.sh
# Expected: Error with clear message and examples

# Test empty NODE_TYPE
NODE_TYPE="" bash deployment/scripts/validate-node-type.sh
# Expected: Error (treats empty as missing)

# Test invalid characters
NODE_TYPE="invalid-chars!" bash deployment/scripts/validate-node-type.sh
# Expected: Error about invalid characters

# Test too short
NODE_TYPE="ab" bash deployment/scripts/validate-node-type.sh
# Expected: Error about minimum length

# Test valid NODE_TYPE
NODE_TYPE="database_adapter_effect" bash deployment/scripts/validate-node-type.sh
# Expected: Success message
```

### Docker Testing

```bash
# Test with missing NODE_TYPE (should fail at startup)
docker build -f deployment/Dockerfile.generic-effect -t test-effect .
docker run --rm test-effect
# Expected: Container exits immediately with clear error message

# Test with valid NODE_TYPE (should start successfully)
docker run --rm -e NODE_TYPE=database_adapter_effect test-effect
# Expected: Container starts and node initializes

# Test health check with invalid NODE_TYPE
docker run -d --name test-unhealthy -e NODE_TYPE="invalid!" test-effect
docker inspect --format='{{.State.Health.Status}}' test-unhealthy
# Expected: unhealthy status after start period
```

## Benefits

### 1. Fail Fast
- Containers fail immediately at startup if NODE_TYPE is invalid
- No need to dig through logs to find root cause

### 2. Clear Error Messages
- Error messages include valid values and configuration examples
- Operators know exactly how to fix the issue

### 3. Validated Health Checks
- Health checks validate NODE_TYPE before reporting status
- Prevents misleading "healthy" status when misconfigured

### 4. Defense in Depth
- Validation at both startup and health check
- Catches configuration issues early in container lifecycle

### 5. Consistency
- Same validation logic across all generic Dockerfiles
- Single validation script for maintainability

## Maintenance

### Adding New Node Types

When adding a new node type, update the validation script error message:

```bash
# Edit deployment/scripts/validate-node-type.sh
# Update the "Valid values by Dockerfile" section
echo "  - Dockerfile.generic-effect:" >&2
echo "      database_adapter_effect, store_effect, new_node_effect" >&2
```

### Modifying Validation Rules

The validation script supports these checks (modify as needed):
1. Empty/unset check (required)
2. Character validation regex (configurable)
3. Length validation (min/max configurable)
4. Module existence check (optional, graceful fallback)

## Docker Compose Examples

All existing docker-compose files already configure NODE_TYPE correctly:

**deployment/docker-compose.metrics.yml:**
- codegen-orchestrator: `NODE_TYPE: codegen_orchestrator`
- codegen-metrics-reducer: `NODE_TYPE: codegen_metrics_reducer`
- store-effect: `NODE_TYPE: store_effect`

**deployment/docker-compose.stamping.yml:**
- database-adapter-effect: `NODE_TYPE: database_adapter_effect`

## Troubleshooting

### Issue: Container exits immediately with "ERROR: NODE_TYPE environment variable is not set"

**Solution:** Add NODE_TYPE to your docker-compose.yml or docker run command:

```yaml
environment:
  NODE_TYPE: database_adapter_effect  # Change to your node type
```

### Issue: Health check fails but container is running

**Solution:** Check if NODE_TYPE is set correctly in the running container:

```bash
docker exec <container_name> printenv NODE_TYPE
```

If unset or invalid, restart with correct NODE_TYPE.

### Issue: "Module not found" warning but validation succeeds

**Behavior:** This is expected if:
1. Source code is mounted as volume (development mode)
2. Module will be created later
3. Node type is valid but not yet implemented

The validation script will warn but not fail, allowing development flexibility.

## Summary

This implementation ensures that:
✅ All containers validate NODE_TYPE before startup
✅ Error messages are clear and actionable
✅ Health checks validate NODE_TYPE before reporting status
✅ No silent failures due to misconfiguration
✅ Development and production stages both have validation
✅ Single validation script for maintainability
