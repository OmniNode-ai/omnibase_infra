# Vault Infrastructure Status

**Date:** 2025-11-14
**Status:** NOT IMPLEMENTED (Future Feature)

---

## Summary

**The Vault adapter node does NOT exist in the omninode_bridge archive.**

The CLAUDE.md file references Vault as part of the desired infrastructure architecture, but it was never implemented in the archive. Only preparatory models exist.

---

## What EXISTS in Archive

### 1. Vault Metrics Model ✅ MIGRATED
**Location:** `src/omnibase_infra/models/health/model_vault_metrics.py`

**Purpose:** Health metrics model for future Vault integration

**Metrics Tracked:**
- Authentication metrics (active tokens, lookups, success rate, renewals)
- Secret engine metrics (mounted engines, read/write operations)
- Lease metrics (active leases, renewals, expirations)
- Audit metrics (requests, failures, latency)
- Seal status and high availability metrics
- Performance metrics (response times, throughput)

**Status:** ✅ Copied to `src/omnibase_infra/models/health/`

---

## What DOES NOT Exist

### 1. Vault Adapter Node ❌ NOT IN ARCHIVE
**Expected Location:** `src/omnibase_infra/nodes/vault_adapter/v1_0_0/`

**What it would do:**
- Message bus bridge to HashiCorp Vault
- Secret retrieval and storage operations
- Token management and renewal
- Lease management
- Encryption/decryption operations
- Audit logging

**Status:** ❌ NOT IMPLEMENTED - Would need to be created from scratch

### 2. Vault Docker Service ❌ NOT IN ARCHIVE
**Expected:** Vault service in docker-compose.infrastructure.yml

**What it would include:**
```yaml
vault:
  image: hashicorp/vault:latest
  container_name: omnibase-infra-vault
  ports:
    - "8200:8200"
  environment:
    VAULT_DEV_ROOT_TOKEN_ID: root
    VAULT_ADDR: http://0.0.0.0:8200
  cap_add:
    - IPC_LOCK
```

**Status:** ❌ NOT IN DOCKER-COMPOSE

### 3. Vault Models ❌ NOT IN ARCHIVE
**Expected:** Shared Vault request/response models

**Would include:**
- `model_vault_secret_request.py`
- `model_vault_secret_response.py`
- `model_vault_token_request.py`
- `model_vault_lease_info.py`

**Status:** ❌ NOT IMPLEMENTED

---

## CLAUDE.md References (Aspirational)

The CLAUDE.md file mentions Vault in several places as part of the **desired** architecture:

1. **Service Integration Architecture:**
   > "Adapter Pattern - External services wrapped in ONEX adapters (Consul, Kafka, **Vault**)"

2. **Infrastructure 4-Node Pattern:**
   > "EFFECT - External service interactions (Consul, Kafka, **Vault adapters**)"

3. **Service Adapters:**
   > "`node_infrastructure_vault_adapter_effect` - **Vault secret management integration**"

4. **Example Directory Structure:**
   ```
   ├── vault_adapter/v1_0_0/
   │   ├── contract.yaml
   │   ├── node.py (NodeEffectService - message bus to vault bridge)
   │   ├── models/
   │   │   ├── model_vault_adapter_input.py
   │   │   └── model_vault_adapter_output.py
   ```

**These are all FUTURE PLANS, not existing implementations.**

---

## Current Credential Management

Without Vault adapter, the infrastructure currently uses:

### 1. Docker Secrets
**Location:** docker-compose.infrastructure.yml
```yaml
secrets:
  github_token:
    environment: "GITHUB_TOKEN"
  postgres_password:
    environment: "POSTGRES_PASSWORD"
```

### 2. Environment Variables
**Location:** .env
```bash
POSTGRES_PASSWORD=your_password
GITHUB_TOKEN=your_token
```

### 3. Credential Manager Utility
**Location:** `src/omnibase_infra/security/credential_manager.py`
- In-memory credential caching with TTL
- Prepared for Vault integration (has Vault URL parameter)
- Can be extended to integrate with Vault when adapter is created

---

## Recommendation: Create Vault Adapter (Future Work)

### When Needed
Create Vault adapter when you need:
- Centralized secret management
- Dynamic secret generation
- Secret rotation and leasing
- Encryption as a service
- Audit logging of secret access

### How to Create

**1. Use Generation Pipeline:**
```bash
python -m omnibase_infra.generation.cli effect \
  --domain infrastructure \
  --microservice vault_adapter \
  --description "HashiCorp Vault secret management adapter" \
  --external-system "HashiCorp Vault"
```

**2. Implement Contract:**
- Input/output models for secret operations
- IO operations: `get_secret`, `set_secret`, `delete_secret`, `renew_lease`
- Dependencies: `protocol_event_bus`, `protocol_http_client`

**3. Add to Docker Compose:**
- Vault service (port 8200)
- Vault adapter service
- Environment configuration

**4. Create Shared Models:**
- `models/vault/model_vault_secret_request.py`
- `models/vault/model_vault_secret_response.py`
- `models/vault/model_vault_token_request.py`

**5. Update Credential Manager:**
- Integrate with Vault adapter via event bus
- Remove in-memory caching, use Vault directly

---

## Current State Summary

| Component | Status | Location |
|-----------|--------|----------|
| **Vault Adapter Node** | ❌ NOT IMPLEMENTED | N/A |
| **Vault Docker Service** | ❌ NOT IN COMPOSE | N/A |
| **Vault Shared Models** | ❌ NOT CREATED | N/A |
| **Vault Metrics Model** | ✅ MIGRATED | `models/health/model_vault_metrics.py` |
| **Credential Manager Utility** | ✅ MIGRATED | `security/credential_manager.py` (Vault-ready) |
| **Docker Secrets** | ✅ IMPLEMENTED | `docker-compose.infrastructure.yml` |

---

## Conclusion

**Vault infrastructure is NOT part of the omninode_bridge archive.**

The references in CLAUDE.md are architectural guidelines for FUTURE implementation, not documentation of existing code.

For now, the infrastructure uses Docker secrets and environment variables for credential management. The Vault adapter would be a valuable future addition but is not required for current functionality.

**All existing infrastructure from the archive HAS been migrated.**
