# MASTER PLAN - Code Generator Fix & Production Readiness

**Status**: Phase 1 In Progress (Task 1/15 complete)
**Goal**: Fix code generator → Use it to build all missing infrastructure
**Timeline**: 10-15 days (Phase 1) | 45-54 days total (all phases)
**Last Updated**: 2025-10-30

---

## 🎯 Quick Status

| Item | Status | Location |
|------|--------|----------|
| **omnibase_core 0.1.0** | ✅ INSTALLED | pyproject.toml:102 |
| **omnibase_core 0.1.0** | ✅ INSTALLED | pyproject.toml:103 |
| **Phase 1 Roadmap** | 📋 READY | docs/planning/IMPLEMENTATION_ROADMAP.md |
| **Todo List** | 📝 14/15 PENDING | See below |

---

## 📋 Current Todo List (Phase 1)

### ✅ Completed
1. **[DONE]** Install omnibase_core package (v0.1.0) + omnibase_core (v0.1.0)

### ⏳ In Progress
2. **[NEXT]** Create ConfigLoader with Vault integration
3. Initialize Vault with secrets
4. Create .env.development and .env.production templates
5. Fix hostname resolution - add extra_hosts to docker-compose
6. Migrate NodeCodegenOrchestrator to ConfigLoader
7. Add event subcontracts to 3 key nodes
8. Implement Consul registration with event metadata
9. Update generator templates - ConfigLoader pattern
10. Update generator templates - event subcontracts
11. Update generator templates - protocol duck typing
12. Apply PostgreSQL migrations (12 migrations) on remote
13. Implement NodeBridgeRegistry backend methods
14. Fix distributed_lock_effect bugs
15. Test code generator end-to-end

---

## 📚 Complete Documentation Index

### Planning Documents
| Document | Purpose | Status | Size |
|----------|---------|--------|------|
| **IMPLEMENTATION_ROADMAP.md** | Complete 3-phase plan | ✅ | 43KB |
| **ROADMAP_VISUAL_SUMMARY.md** | Timeline & dependencies | ✅ | 26KB |
| **MASTER_PLAN.md** | This file - quick reference | ✅ | Current |

**Location**: `docs/planning/`

### Audit Reports
| Report | Finding | Impact | Location |
|--------|---------|--------|----------|
| **omnibase_core Compliance** | 35% prepared, needs install | P0 | /tmp/omnibase_core_compliance_audit.md |
| **Protocol Duck Typing** | 1.5% adoption | P1 | docs/analysis/MODEL_CONTAINER_PROTOCOL_AUDIT.md |
| **Infrastructure Config** | Hardcoded chaos | P0 | docs/analysis/INFRASTRUCTURE_CONFIG_VALIDATION.md |
| **Event Subcontracts** | 0% standardized | P0 | Agent output from parallel-solve |
| **Generator Templates** | 60% compliant | P0 | Agent output from parallel-solve |

### Research Documents
| Document | Purpose | Location |
|----------|---------|----------|
| **Consul/Registry/Introspection** | Service registration design | docs/research/CONSUL_NODE_REGISTRY_INTROSPECTION_RESEARCH.md |

---

## 🎯 Phase 1: MVP Foundation (10-15 days)

**Goal**: Code generator working with proper config/Vault/Consul integration

### Week 1: Configuration & Infrastructure
- **ConfigLoader** with Vault cascade (1d)
- **Vault initialization** with secrets (2h)
- **.env templates** for dev/prod (2h)
- **Hostname resolution** fix (30m)
- **Protocol duck typing** migration (1d)

### Week 2: Node Migrations & Integration
- **Orchestrator** to ConfigLoader (1d)
- **Reducer** to ConfigLoader (1d)
- **Registry** to ConfigLoader (0.5d)
- **Consul registration** with event metadata (1d)

### Week 3: Code Generator & Testing
- **Fix distributed_lock** bugs (4h)
- **Database migrations** (4h)
- **Registry backend** implementation (1d)
- **Update templates** (ConfigLoader + events + protocols) (1d)
- **End-to-end testing** (1d)

**Critical Path**: 6.75 days minimum (with parallelization: ~5 days)

---

## 🔑 Key Architecture Decisions

### 1. Configuration Management
```
Priority Cascade:
1. Vault (secrets only) → postgres_password, api_keys
2. Environment Variables (.env) → endpoints, non-secrets
3. Contract YAML (defaults) → fallback values
4. ❌ NEVER hardcode in Python files
```

**Implementation**: `src/omninode_bridge/config/loader.py`

### 2. Consul Service Registration
```yaml
# Services register with event metadata
Meta:
  events_consumed: "node.introspection.reported"
  events_emitted: "node.registration.completed,node.registration.failed"
  node_type: "registry"
  capabilities: {...}
```

**Implementation**: `NodeBridgeRegistry._register_with_consul()`

### 3. Protocol Duck Typing
```python
# ✅ CORRECT - Type-safe dependency injection
from omninode_bridge.protocols import KafkaClientProtocol
self.kafka: KafkaClientProtocol = container.resolve(KafkaClientProtocol)

# ❌ WRONG - String-based lookup (no type safety)
self.kafka = container.get_service("kafka_client")
```

**Migration**: Convert 24 string-based lookups → protocol-based

### 4. omnibase_core Models
```python
# Use omnibase_core models where available
from omnibase_core.models.core import ModelContainer, ModelOnexError
from omnibase_core.enums import EnumCoreErrorCode

# Migrate 26 entity models to omnibase_core.ModelEntityBase
# Migrate 17 exception files to ModelOnexError
```

**Status**: 35% prepared (stubs exist), need real package

---

## 🚨 Critical Infrastructure Issues (RESOLVED)

### Issue 1: Code Generator Failures ✅ IDENTIFIED
**Root Cause**: Multiple configuration/infrastructure issues
**Symptoms**:
- 7 background orchestrator processes failing
- Service resolution errors (KafkaClientProtocol)
- Hardcoded values (localhost:29092 instead of 192.168.86.200:9092)
- Hostname resolution failures in Docker

**Solution**: ConfigLoader + Vault + hostname fix (Week 1-2 tasks)

### Issue 2: NodeBridgeRegistry Not Functional ✅ IDENTIFIED
**Status**: Running on remote but backend methods stubbed
**Impact**: No service registration happening
**Solution**: Implement backend methods (Week 3, task 13)

### Issue 3: Missing Event Metadata ✅ IDENTIFIED
**Status**: Contracts lack event subcontracts
**Impact**: Can't do event-driven service discovery
**Solution**: Add to contracts + Consul registration (Week 2, task 7-8)

---

## 📊 Compliance Scorecard

| Area | Current | Target | Priority | Effort |
|------|---------|--------|----------|--------|
| omnibase_core | 35% | 100% | P0 | 8-9 weeks |
| Protocol Duck Typing | 1.5% | 100% | P1 | 2-3 weeks |
| Event Subcontracts | 0% | 100% | P0 | 32 hours |
| Generator Templates | 60% | 100% | P0 | 3 weeks |
| Configuration | 0% | 100% | P0 | 1-2 weeks |

---

## 🎯 Phase 2: Production Hardening (15-21 days)

**Goal**: Keycloak + Security + Observability

### Week 4: Keycloak Integration (FOLLOW-UP)
- Deploy Keycloak container
- Configure realms and clients
- Add JWT validation middleware
- Update ConfigLoader for Keycloak

### Week 5: Audit Trail & Monitoring
- Complete audit trail implementation
- Monitoring dashboards (Grafana/Prometheus)
- Alerting rules

### Week 6: Service Mesh & Security
- mTLS between services
- Security hardening
- Penetration testing

---

## 🎯 Phase 3: Scale & Optimize (20-28 days)

**Goal**: Horizontal scaling + Multi-region

### Week 7-8: Horizontal Scaling
- Orchestrator scaling (stateless coordination)
- Reducer scaling (state partitioning)
- Load testing & optimization

### Week 9-10: Advanced Features
- Multi-region deployment
- Advanced Keycloak features (SSO, user federation)
- Performance optimization

---

## 🔧 How to Use This Plan

### Starting Fresh (New Session)
1. Read this file (MASTER_PLAN.md)
2. Check todo list status above
3. Review IMPLEMENTATION_ROADMAP.md for current task details
4. Execute next pending task

### Tracking Progress
- Update todo list status in this file
- Mark completed tasks with ✅
- Add notes/blockers as needed

### Need More Detail?
- **Full roadmap**: `docs/planning/IMPLEMENTATION_ROADMAP.md` (43KB)
- **Visual timeline**: `docs/planning/ROADMAP_VISUAL_SUMMARY.md` (26KB)
- **Audit reports**: `docs/analysis/` directory
- **Research**: `docs/research/` directory

---

## 🚀 Next Immediate Actions

**Now** (Task #2 in todo list):
1. Create `src/omninode_bridge/config/loader.py` with ConfigLoader class
2. Implement Vault cascade: Vault → Env → YAML → Fail
3. Add tests for ConfigLoader
4. Document usage in CLAUDE.md

**Commands to start**:
```bash
# Create config directory if needed
mkdir -p src/omninode_bridge/config

# Create ConfigLoader implementation
# (See IMPLEMENTATION_ROADMAP.md Section 1.1 for code)

# Create .env templates
# (See IMPLEMENTATION_ROADMAP.md Section 1.2 for templates)

# Initialize Vault
# (See IMPLEMENTATION_ROADMAP.md Section 1.3 for Vault setup)
```

---

## 📞 Quick Reference

**Infrastructure**:
- Remote Server: 192.168.86.200
- PostgreSQL: 192.168.86.200:5436/omninode_bridge
- Kafka: 192.168.86.200:9092
- Consul: 192.168.86.200:28500
- Vault: 192.168.86.200:8200

**Key Files**:
- ConfigLoader: `src/omninode_bridge/config/loader.py` (to be created)
- Orchestrator: `src/omninode_bridge/nodes/codegen_orchestrator/v1_0_0/node.py`
- Registry: `src/omninode_bridge/nodes/registry/v1_0_0/node.py`

**Dependencies**:
- omnibase_core: v0.1.0 ✅ INSTALLED
- omnibase_core: v0.1.0 ✅ INSTALLED

---

## ✅ Session Resume Checklist

When starting a new session:
- [ ] Read MASTER_PLAN.md (this file)
- [ ] Check todo list (14 pending items)
- [ ] Review IMPLEMENTATION_ROADMAP.md for current task
- [ ] Verify omnibase_core still installed: `poetry show omnibase-core`
- [ ] Check remote infrastructure: `ssh 192.168.86.200 "docker ps"`
- [ ] Execute next task

**Current Task**: #2 - Create ConfigLoader with Vault integration

---

**End of Master Plan**
**Last Updated**: 2025-10-30
**Next Review**: After completing each week of Phase 1
