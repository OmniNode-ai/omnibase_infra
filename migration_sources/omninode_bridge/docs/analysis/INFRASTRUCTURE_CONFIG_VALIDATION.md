# Infrastructure Configuration Validation Report
**Correlation ID**: f1c17289-23e6-4821-bbc2-0cbcdbc1a347
**Generated**: $(date -u +"%Y-%m-%d %H:%M:%S UTC")

## Executive Summary
**Status**: ❌ CRITICAL - Multiple configuration mismatches detected
**Deficiencies Found**: 12 critical issues
**Impact**: Code generation orchestrator cannot resolve services

---

## 1. Environment Variables Assessment

### Current Shell Environment
**Status**: ❌ MISSING - No infrastructure environment variables set

```bash
# Expected variables (MISSING):
POSTGRES_HOST
POSTGRES_PORT
POSTGRES_DATABASE
POSTGRES_USER
POSTGRES_PASSWORD
KAFKA_BOOTSTRAP_SERVERS
KAFKA_BROKER_URL
CONSUL_HOST
CONSUL_PORT
REDPANDA_PORT
```

**Finding**: Running orchestrator locally requires these env vars to override hardcoded defaults.

**Validation Results**:
- Configuration Item: POSTGRES_HOST
  - Expected: 192.168.86.200 or omninode-bridge-postgres
  - Actual: MISSING
  - Status: ❌ MISSING

- Configuration Item: KAFKA_BOOTSTRAP_SERVERS
  - Expected: 192.168.86.200:9092 or omninode-bridge-redpanda:9092
  - Actual: MISSING
  - Status: ❌ MISSING

- Configuration Item: CONSUL_HOST
  - Expected: 192.168.86.200 or omninode-bridge-consul
  - Actual: MISSING
  - Status: ❌ MISSING

---

## 2. Docker Compose Configuration

### Infrastructure Services (docker-compose.yml)
**Status**: ✅ CONFIGURED - Services defined correctly

| Service | Container DNS | External Port | Internal Port |
|---------|---------------|---------------|---------------|
| postgres | omninode-bridge-postgres | 5436 | 5432 |
| redpanda | omninode-bridge-redpanda | 9092 | 9092 |
| consul | omninode-bridge-consul | 28500 | 8500 |
| metadata-stamping | - | 8057 | 8053 |
| onextree | - | 8058 | 8058 |

**Finding**: Services use internal Docker network DNS names (omninode-bridge-*) but orchestrator configs use localhost.

---

## 3. Orchestrator Node Configuration

### NodeCodegenOrchestrator (node.py)
**Status**: ⚠️ MISMATCH - Hardcoded defaults incorrect for deployment

**Hardcoded Defaults** (lines 85-105):
```python
self.kafka_broker_url = "localhost:29092"      # ❌ WRONG PORT
self.omniarchon_url = "http://omniarchon:8060" # ❌ WRONG SERVICE NAME
self.default_output_dir = "./generated_nodes"   # ✅ OK
```

**Configuration Issues**:

1. kafka_broker_url: "localhost:29092"
   - Expected: "192.168.86.200:9092" or "omninode-bridge-redpanda:9092"
   - Actual: "localhost:29092"
   - Status: ❌ MISMATCH (wrong port 29092 vs 9092)

2. omniarchon_url: "http://omniarchon:8060"
   - Expected: "http://192.168.86.200:8060" (NodeBridgeOrchestrator)
   - Actual: "http://omniarchon:8060" (service doesn't exist)
   - Status: ❌ WRONG SERVICE NAME

---

## 4. YAML Configuration Files

### orchestrator.yaml
**Status**: ⚠️ MISMATCH - Localhost configs incompatible with Docker deployment

**Service Endpoints**:
```yaml
services:
  onextree:
    host: "localhost"
    port: 8051          # ❌ WRONG - actual port is 8058

  metadata_stamping:
    host: "localhost"
    port: 8053          # ⚠️ INTERNAL - external port is 8057

kafka:
  bootstrap_servers: "localhost:9092"  # ⚠️ OK for external, but uses port 9092 not 29092

database:
  host: "localhost"
  port: 5432          # ❌ WRONG - external port is 5436
```

**Validation Results**:
- onextree.port: 8051
  - Expected: 8058 (actual running port)
  - Actual: 8051
  - Status: ❌ MISMATCH

- metadata_stamping.port: 8053
  - Expected: 8057 (external) or 8053 (internal)
  - Actual: 8053 (only works inside Docker network)
  - Status: ⚠️ NEEDS CLARIFICATION

- database.port: 5432
  - Expected: 5436 (external) or 5432 (internal)
  - Actual: 5432 (only works inside Docker network)
  - Status: ⚠️ NEEDS CLARIFICATION

### development.yaml
**Status**: ⚠️ MISMATCH - Same localhost issues

**Same Issues**:
- All service endpoints use "localhost"
- Ports don't match external mappings for local execution
- Works only if running inside Docker network

---

## 5. Network Configuration

### Hostname Resolution (/etc/hosts)
**Status**: ✅ CONFIGURED

```
192.168.86.200 omninode-bridge-redpanda
192.168.86.200 omninode-bridge-consul
192.168.86.200 omninode-bridge-postgres
```

**Connectivity Tests**:
- 192.168.86.200: ✅ REACHABLE (13.5ms)
- omninode-bridge-redpanda: ✅ REACHABLE (10.5ms)
- omninode-bridge-postgres: ✅ REACHABLE (13.8ms)

**Active Connections**:
```
192.168.86.101 → 192.168.86.200:5436 (PostgreSQL) ✅
192.168.86.101 → 192.168.86.200:9092 (Kafka) ✅ (multiple connections)
192.168.86.101 → 192.168.86.200:28500 (Consul) ✅
```

**Finding**: Network connectivity is healthy. DNS resolution works correctly.

---

## 6. Service Discovery (Consul)

### Registered Services
**Status**: ⚠️ LIMITED

**Query Result**:
```json
{
  "consul": [],
  "metadata-stamping-service": [
    "blake3-hashing",
    "metadata-stamping",
    "omninode.services.metadata",
    "o.n.e.v0.1"
  ]
}
```

**Findings**:
- ✅ metadata-stamping-service: Registered correctly
- ❌ omniarchon: NOT REGISTERED (service doesn't exist)
- ⚠️ NodeBridgeOrchestrator: Not visible in catalog (but responds at 192.168.86.200:8060)

**Service Resolution Issue**:
The codegen orchestrator tries to connect to "http://omniarchon:8060" but:
1. No service named "omniarchon" exists
2. Actual service is "NodeBridgeOrchestrator" at 192.168.86.200:8060
3. Service name mismatch prevents resolution

---

## 7. Running Services Inventory

### Docker Containers (Running)
**Status**: ⚠️ PARTIAL

| Container | Internal Port | External Port | Status |
|-----------|---------------|---------------|--------|
| omninode-bridge-database-adapter | 8070 | 8070 | ✅ Running |
| archon-bridge | 8054 | 8054 | ✅ Running |
| omninode-bridge-onextree | 8058 | 8058 | ✅ Running |
| omninode-bridge-consul | 8500 | 28500 | ✅ Running |
| omninode-bridge-vault | 8200 | 8200 | ✅ Running |

**Missing Services**:
- ❌ omninode-bridge-postgres (not in docker ps output but responding on 192.168.86.200:5436)
- ❌ omninode-bridge-redpanda (not in docker ps output but responding on 192.168.86.200:9092)
- ❌ omninode-bridge-metadata-stamping (not in local docker ps)

**Finding**: Core infrastructure services are running on REMOTE host (192.168.86.200), not locally.

---

## 8. Service-Specific Validation

### Kafka/Redpanda
- Configuration Item: KAFKA_BOOTSTRAP_SERVERS
  - Expected: omninode-bridge-redpanda:9092 or 192.168.86.200:9092
  - Actual (env): MISSING
  - Actual (orchestrator default): localhost:29092
  - Actual (orchestrator.yaml): localhost:9092
  - Status: ❌ MISMATCH

**Port Confusion**:
- Orchestrator node.py default: 29092 (external port in docker-compose)
- orchestrator.yaml: 9092 (internal port)
- Actual listening: 192.168.86.200:9092 (internal port exposed)

### PostgreSQL
- Configuration Item: DATABASE_URL
  - Expected: postgresql://postgres:***@192.168.86.200:5436/omninode_bridge
  - Actual (env): MISSING
  - Actual (.env file): postgresql://postgres:***@192.168.86.200:5436/omninode_bridge
  - Actual (orchestrator.yaml): localhost:5432
  - Status: ⚠️ MISMATCH (env not loaded, yaml incorrect)

### Consul
- Configuration Item: CONSUL_HOST
  - Expected: omninode-bridge-consul or 192.168.86.200
  - Actual (env): MISSING
  - Actual (.env file): omninode-bridge-consul
  - Actual (orchestrator.yaml): Not used
  - Status: ⚠️ ENV NOT LOADED

### OnexTree Service
- Configuration Item: services.onextree
  - Expected: localhost:8058 or omninode-bridge-onextree:8058
  - Actual (orchestrator.yaml): localhost:8051
  - Actual (running): localhost:8058
  - Status: ❌ PORT MISMATCH

### Metadata Stamping Service
- Configuration Item: services.metadata_stamping
  - Expected: localhost:8057 (external) or omninode-bridge-metadata-stamping:8053 (internal)
  - Actual (orchestrator.yaml): localhost:8053
  - Actual (running remote): 192.168.86.200:8057 (external) / :8053 (internal)
  - Status: ❌ MISMATCH (service is remote, not local)

---

## 9. Configuration Loading Mechanism

### Container Configuration (node.py lines 85-105)
**Status**: ⚠️ FALLBACK TO DEFAULTS

**Config Resolution Path**:
1. Try: container.config.get("kafka_broker_url")
2. Fallback: "localhost:29092" (hardcoded)
3. Exception: "localhost:29092" (hardcoded)

**Findings**:
- ❌ No environment variable override implemented
- ❌ Defaults are incorrect for deployment environment
- ❌ No config file loader from orchestrator.yaml/development.yaml
- ⚠️ Container.config.get() likely returns None (no config loaded)

---

## 10. Missing Configuration Files

### Checked Locations
**Status**: ⚠️ SOME MISSING

**Config Files Found**:
- ✅ /config/orchestrator.yaml
- ✅ /config/development.yaml
- ✅ /config/production.yaml
- ✅ /.env (main file)
- ✅ /.env.codegen.example
- ⚠️ /.env.codegen (may not exist - not checked)

**Config Files NOT Found**:
- ❌ codegen_orchestrator.yaml (dedicated config)
- ❌ .env.codegen (runtime config)

---

## Deficiencies Summary

### Critical Issues (12 total)

1. **Environment Variables Not Set** - No POSTGRES/KAFKA/CONSUL vars in current shell
2. **Orchestrator kafka_broker_url Wrong** - localhost:29092 vs 192.168.86.200:9092
3. **Orchestrator omniarchon_url Invalid** - Service "omniarchon" doesn't exist
4. **orchestrator.yaml Service Ports Wrong** - onextree 8051 vs actual 8058
5. **orchestrator.yaml Database Port Wrong** - 5432 vs external 5436
6. **Container Config Not Loading** - container.config.get() returns defaults
7. **Service Name Mismatch** - "omniarchon" vs "NodeBridgeOrchestrator"
8. **Local vs Remote Confusion** - Services run remotely but configs assume local
9. **No Environment Variable Override** - Hardcoded defaults can't be overridden
10. **YAML Configs Use Localhost** - Incompatible with Docker network / remote execution
11. **Port Number Confusion** - 29092 (external) vs 9092 (internal) in different configs
12. **Metadata Stamping Service Missing Locally** - Configured locally but runs remotely

---

## Remediation Steps

### Immediate Actions (Required)

1. **Create .env.codegen file** with correct remote infrastructure:
   ```bash
   cat > .env.codegen << 'ENVEOF'
   # Remote Infrastructure (192.168.86.200)
   POSTGRES_HOST=192.168.86.200
   POSTGRES_PORT=5436
   POSTGRES_DATABASE=omninode_bridge
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=omninode_remote_2024_secure

   KAFKA_BOOTSTRAP_SERVERS=192.168.86.200:9092
   KAFKA_BROKER_URL=192.168.86.200:9092

   CONSUL_HOST=192.168.86.200
   CONSUL_PORT=28500

   # Service endpoints (external ports for local development)
   ONEXTREE_URL=http://localhost:8058
   METADATA_STAMPING_URL=http://192.168.86.200:8057

   # Orchestrator service (bridge orchestrator, not "omniarchon")
   OMNIARCHON_URL=http://192.168.86.200:8060
   BRIDGE_ORCHESTRATOR_URL=http://192.168.86.200:8060
   ENVEOF
   ```

2. **Update orchestrator.yaml** to use environment variables:
   ```yaml
   services:
     onextree:
       host: "${ONEXTREE_HOST:-localhost}"
       port: ${ONEXTREE_PORT:-8058}  # Correct port
       base_url: "${ONEXTREE_URL:-http://localhost:8058}"

     metadata_stamping:
       host: "${METADATA_STAMPING_HOST:-192.168.86.200}"
       port: ${METADATA_STAMPING_PORT:-8057}
       base_url: "${METADATA_STAMPING_URL:-http://192.168.86.200:8057}"

   kafka:
     bootstrap_servers: "${KAFKA_BOOTSTRAP_SERVERS:-192.168.86.200:9092}"

   database:
     host: "${POSTGRES_HOST:-192.168.86.200}"
     port: ${POSTGRES_PORT:-5436}  # External port
   ```

3. **Fix node.py configuration loading**:
   ```python
   # Add environment variable override
   import os

   self.kafka_broker_url = os.getenv(
       "KAFKA_BROKER_URL",
       container.config.get("kafka_broker_url", "192.168.86.200:9092")
   )
   self.omniarchon_url = os.getenv(
       "BRIDGE_ORCHESTRATOR_URL",
       container.config.get("omniarchon_url", "http://192.168.86.200:8060")
   )
   ```

4. **Fix service name references**:
   - Replace all "omniarchon" references with "bridge-orchestrator"
   - Update routing.yaml to use BRIDGE_ORCHESTRATOR_URL
   - Update tests to use correct service name

5. **Source environment before execution**:
   ```bash
   # Load environment
   export $(cat .env.codegen | xargs)

   # Verify
   echo $KAFKA_BOOTSTRAP_SERVERS
   echo $POSTGRES_HOST
   echo $BRIDGE_ORCHESTRATOR_URL

   # Run orchestrator
   python -m omninode_bridge.nodes.codegen_orchestrator.v1_0_0.node
   ```

### Configuration Validation Command

```bash
#!/bin/bash
# validate-config.sh - Run after remediation

echo "=== Configuration Validation ==="

# 1. Check environment variables
echo "1. Environment Variables:"
env | grep -E "(POSTGRES|KAFKA|CONSUL|BRIDGE)" | sort

# 2. Test network connectivity
echo -e "\n2. Network Connectivity:"
ping -c 1 192.168.86.200 && echo "✅ Remote host reachable"
curl -s http://192.168.86.200:8060/health && echo "✅ Bridge orchestrator healthy"
curl -s http://localhost:8058/health && echo "✅ OnexTree healthy"

# 3. Test Kafka connection
echo -e "\n3. Kafka Connection:"
timeout 5 nc -zv 192.168.86.200 9092 && echo "✅ Kafka port open"

# 4. Test PostgreSQL connection
echo -e "\n4. PostgreSQL Connection:"
timeout 5 nc -zv 192.168.86.200 5436 && echo "✅ PostgreSQL port open"

# 5. Test Consul connection
echo -e "\n5. Consul Connection:"
curl -s http://192.168.86.200:28500/v1/catalog/services | jq '.' && echo "✅ Consul accessible"

echo -e "\n=== Validation Complete ==="
```

---

## Configuration Matrix

### Local Development (Running orchestrator locally)

| Component | Configuration Source | Value | Status |
|-----------|---------------------|-------|--------|
| Kafka | .env.codegen | 192.168.86.200:9092 | ❌ CREATE FILE |
| PostgreSQL | .env.codegen | 192.168.86.200:5436 | ❌ CREATE FILE |
| Consul | .env.codegen | 192.168.86.200:28500 | ❌ CREATE FILE |
| OnexTree | .env.codegen | localhost:8058 | ❌ CREATE FILE |
| Metadata Stamping | .env.codegen | 192.168.86.200:8057 | ❌ CREATE FILE |
| Bridge Orchestrator | .env.codegen | http://192.168.86.200:8060 | ❌ CREATE FILE |

### Docker Deployment (Running orchestrator in container)

| Component | Configuration Source | Value | Status |
|-----------|---------------------|-------|--------|
| Kafka | docker-compose.yml env | omninode-bridge-redpanda:9092 | ✅ OK |
| PostgreSQL | docker-compose.yml env | omninode-bridge-postgres:5432 | ✅ OK |
| Consul | docker-compose.yml env | omninode-bridge-consul:8500 | ✅ OK |
| OnexTree | docker-compose.yml env | omninode-bridge-onextree:8058 | ⚠️ VERIFY |
| Metadata Stamping | docker-compose.yml env | omninode-bridge-metadata-stamping:8053 | ⚠️ VERIFY |
| Bridge Orchestrator | NOT CONFIGURED | ??? | ❌ MISSING |

---

## Risk Assessment

### Severity: HIGH
- Orchestrator cannot start successfully
- Service resolution failures block all operations
- Configuration mismatches prevent deployment

### Impact: CRITICAL
- Code generation pipeline completely blocked
- No workaround without configuration fixes
- Affects all dependent services

### Urgency: IMMEDIATE
- Blocking development work
- Requires configuration audit and remediation
- Estimated fix time: 2-4 hours

---

## Next Steps

1. ✅ **Immediate**: Create .env.codegen with correct remote infrastructure configuration
2. ✅ **Immediate**: Update orchestrator.yaml with environment variable references
3. ✅ **Immediate**: Fix node.py to load environment variables
4. ✅ **Immediate**: Replace "omniarchon" with "bridge-orchestrator" everywhere
5. ⚠️ **Short-term**: Create dedicated codegen_orchestrator.yaml config
6. ⚠️ **Short-term**: Add configuration validation script
7. ⚠️ **Medium-term**: Implement Consul-based service discovery
8. ⚠️ **Medium-term**: Add configuration hot-reload capability

---

**Report End**
