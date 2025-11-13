# Docker Compose Files Validation Report

**Date**: 2025-11-04
**Validated Files**: 5
**Validation Tool**: `docker compose config`

---

## Summary

| File | Status | Issues |
|------|--------|--------|
| `docker-compose.yml` | ✅ VALID | None |
| `docker-compose.bridge.yml` | ❌ INVALID | Cross-file service dependencies |
| `docker-compose.adapters.yml` | ❌ INVALID | Missing required environment variable |
| `docker-compose.codegen.yml` | ✅ VALID | None |
| `docker-compose.metrics.yml` | ✅ VALID | None |

**Overall**: 3/5 files pass validation (60%)

---

## Detailed Results

### ✅ docker-compose.yml (Infrastructure)

**Status**: VALID
**Project Name**: `omninode_bridge` ✓

**Services**:
- consul
- vault
- postgres
- redpanda
- redpanda-topic-manager
- redpanda-ui
- registry
- deployment-receiver

**Issues**: None

---

### ❌ docker-compose.bridge.yml (Bridge Nodes)

**Status**: INVALID
**Project Name**: `omninode_bridge` ✓

**Services**:
- orchestrator
- reducer

**Error**:
```
service "reducer" depends on undefined service "postgres": invalid compose project
```

**Root Cause**:
Both `orchestrator` and `reducer` services declare `depends_on` for services defined in `docker-compose.yml`:
- postgres
- redpanda
- consul

These services are not available when `docker-compose.bridge.yml` is validated standalone.

**Solutions**:

**Option 1: Multi-file deployment (Recommended)**
```bash
docker compose -f docker-compose.yml -f docker-compose.bridge.yml up -d
```

**Option 2: Remove cross-file dependencies**
Remove `depends_on` from bridge services and rely on connection retry logic:
```yaml
# Remove these from orchestrator and reducer:
depends_on:
  - postgres
  - redpanda
  - consul
```

**Option 3: Add external network reference**
Modify bridge services to use external network and assume infrastructure is running:
```yaml
networks:
  omninode-bridge-network:
    external: true
```

---

### ❌ docker-compose.adapters.yml (Effect Adapters)

**Status**: INVALID
**Project Name**: `omninode_bridge` ✓

**Services**:
- metadata-stamping
- onextree
- hook-receiver
- model-metrics
- llm-effect
- vault-effect

**Error**:
```
error while interpolating services.llm-effect.environment.ZAI_API_KEY:
required variable ZAI_API_KEY is missing a value: ZAI_API_KEY must be set in environment
```

**Root Cause**:
The `llm-effect` service requires `ZAI_API_KEY` environment variable with explicit validation:
```yaml
ZAI_API_KEY: ${ZAI_API_KEY:?ZAI_API_KEY must be set in environment}
```

This variable is:
- Not defined in `.env.example`
- Not set in current environment

**Solutions**:

**Option 1: Add to .env.example (Recommended)**
```bash
# Add to .env.example
ZAI_API_KEY=your_z_ai_api_key_here
```

**Option 2: Provide default value**
```yaml
# Change in docker-compose.adapters.yml
ZAI_API_KEY: ${ZAI_API_KEY:-default_key_for_testing}
```

**Option 3: Set in environment before validation**
```bash
export ZAI_API_KEY="test_key"
docker compose -f docker-compose.adapters.yml config
```

**Option 4: Make optional for development**
```yaml
# Remove required validation (for dev only)
ZAI_API_KEY: ${ZAI_API_KEY}
```

---

### ✅ docker-compose.codegen.yml (Code Generation)

**Status**: VALID
**Project Name**: `omninode_bridge` ✓

**Services**:
- topic-creator

**Issues**: None

---

### ✅ docker-compose.metrics.yml (Monitoring)

**Status**: VALID
**Project Name**: `omninode_bridge` ✓

**Services**:
- codegen-orchestrator
- codegen-metrics-reducer
- store-effect

**Issues**: None

---

## Global Validation

### Project Name Consistency
✅ All files use `name: omninode_bridge`

### Network Configuration
✅ All files reference `omninode-bridge-network` (external network pattern)

### Volume Naming
✅ Volume names follow consistent patterns with appropriate prefixes

---

## Recommendations

### Immediate Actions

1. **Fix docker-compose.bridge.yml**:
   - Document that it requires docker-compose.yml to run
   - Update deployment scripts to use multi-file deployment
   - OR remove cross-file dependencies and rely on retry logic

2. **Fix docker-compose.adapters.yml**:
   - Add `ZAI_API_KEY` to `.env.example`
   - Document the requirement in setup documentation
   - OR provide a default/optional value for development environments

3. **Update Documentation**:
   - Document multi-file deployment strategy
   - Add file dependency matrix
   - Update deployment scripts to handle dependencies

### File Dependency Matrix

```
docker-compose.yml (infrastructure)
  ├── docker-compose.bridge.yml (depends on: postgres, redpanda, consul)
  ├── docker-compose.adapters.yml (depends on: network, ZAI_API_KEY env var)
  ├── docker-compose.codegen.yml (independent, but uses network)
  └── docker-compose.metrics.yml (independent, but uses network)
```

### Deployment Strategy

**Recommended deployment order**:
```bash
# 1. Infrastructure first
docker compose -f deployment/docker-compose.yml up -d

# 2. Wait for infrastructure to be healthy
docker compose -f deployment/docker-compose.yml ps

# 3. Deploy domain-specific services
export ZAI_API_KEY="your_key"
docker compose -f deployment/docker-compose.yml -f deployment/docker-compose.bridge.yml up -d
docker compose -f deployment/docker-compose.yml -f deployment/docker-compose.adapters.yml up -d
docker compose -f deployment/docker-compose.codegen.yml up -d
docker compose -f deployment/docker-compose.metrics.yml up -d
```

**OR single command for all**:
```bash
export ZAI_API_KEY="your_key"
docker compose \
  -f deployment/docker-compose.yml \
  -f deployment/docker-compose.bridge.yml \
  -f deployment/docker-compose.adapters.yml \
  -f deployment/docker-compose.codegen.yml \
  -f deployment/docker-compose.metrics.yml \
  up -d
```

---

## Conclusion

The docker-compose file reorganization has successfully separated concerns by domain, but introduced cross-file dependencies that need to be addressed. The validation identified 2 specific issues that can be easily resolved with the recommended solutions above.

**Next Steps**:
1. Choose dependency resolution strategy (multi-file deployment vs. removal)
2. Add missing environment variables to `.env.example`
3. Update deployment scripts and documentation
4. Re-validate after fixes applied
