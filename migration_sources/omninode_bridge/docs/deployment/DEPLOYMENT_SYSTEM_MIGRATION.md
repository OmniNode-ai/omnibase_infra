# Deployment System Migration Strategy

**Status**: ⏳ In Progress - Parallel Validation Phase
**Last Updated**: 2025-10-25
**Correlation ID**: deployment-docs-001

---

## Executive Summary

This document outlines the strategic migration from **manual shell script-based deployment** to **automated ONEX v2.0 workflow-based deployment** for the OmniNode Bridge system. The migration follows a **phased validation approach** where both systems operate in parallel until the automated system is proven reliable, at which point the manual scripts will be deprecated.

**Key Principle**: **Preserve working systems until new systems are validated**

---

## Table of Contents

1. [Current State](#current-state)
2. [Future State](#future-state)
3. [Migration Strategy](#migration-strategy)
4. [Script Preservation Strategy](#script-preservation-strategy)
5. [Validation Criteria](#validation-criteria)
6. [Rollback Plan](#rollback-plan)
7. [Deployment Workflow Usage Guide](#deployment-workflow-usage-guide)
8. [Transition Checklist](#transition-checklist)
9. [Timeline and Milestones](#timeline-and-milestones)

---

## Current State

### Manual Shell Script-Based Deployment

**Status**: ✅ Production-Ready and Battle-Tested

The current deployment system consists of four shell scripts providing complete container deployment automation:

#### 1. **migrate-to-remote.sh** (Complete Migration)
**Purpose**: Full container migration from local to remote system
**Location**: `/Volumes/PRO-G40/Code/omninode_bridge/migrate-to-remote.sh`
**Remote Target**: 192.168.86.200 (Mac Studio AI Lab)

**Capabilities**:
- ✅ Automated SSH key setup and authentication
- ✅ Docker image export (all omninode_bridge + infrastructure images)
- ✅ Remote environment configuration generation
- ✅ Docker-compose override file creation
- ✅ Complete file transfer via SCP
- ✅ Remote setup script execution
- ✅ Service health verification
- ✅ Automatic cleanup of local exports

**Performance**:
- **Export Time**: ~5-10 minutes for all images
- **Transfer Time**: ~3-5 minutes for ~2GB of images
- **Total Time**: ~15-20 minutes for complete migration
- **Success Rate**: 100% (manual intervention only for initial SSH password)

**Example Usage**:
```bash
# Update username in script (one-time)
REMOTE_USER="jonah"

# Run complete migration
./migrate-to-remote.sh
```

#### 2. **rebuild-service.sh** (Individual Service Rebuild)
**Purpose**: Rebuild and redeploy specific services after code changes
**Location**: `/Volumes/PRO-G40/Code/omninode_bridge/rebuild-service.sh`

**Capabilities**:
- ✅ Build specific Docker image locally
- ✅ Export single image to tar.gz
- ✅ Transfer to remote system
- ✅ Stop and remove old container
- ✅ Deploy and start new container
- ✅ Health check verification
- ✅ Automatic cleanup

**Performance**:
- **Build Time**: ~10-30 seconds per service
- **Transfer Time**: ~5-15 seconds for 100-500MB image
- **Total Time**: ~30-60 seconds per service rebuild

**Example Usage**:
```bash
# Rebuild orchestrator
./rebuild-service.sh orchestrator

# Rebuild reducer on specific host
./rebuild-service.sh reducer 192.168.86.200
```

**Supported Services**:
- `orchestrator` - Workflow coordination
- `reducer` - Metadata aggregation
- `hook-receiver` - Service lifecycle management
- `model-metrics` - AI Lab integration
- `metadata-stamping` - O.N.E. v0.1 Protocol support
- `onextree` - Project structure intelligence
- `registry` - Service registration

#### 3. **setup-remote.sh** (Remote System Configuration)
**Purpose**: Configure and deploy containers on remote system
**Location**: `/Volumes/PRO-G40/Code/omninode_bridge/setup-remote.sh`
**Execution**: Automatically triggered by migrate-to-remote.sh

**Capabilities**:
- ✅ Docker Desktop update via Homebrew
- ✅ Docker image import from tar.gz archives
- ✅ Environment configuration loading
- ✅ Staged service startup (infrastructure → core → bridge nodes)
- ✅ Firewall rule configuration (macOS pf)
- ✅ Management script creation
- ✅ Health check verification

**Staged Startup**:
1. **Infrastructure** (postgres, redpanda, consul, vault) - 45s startup
2. **Core Services** (hook-receiver, model-metrics, metadata-stamping, onextree) - 30s startup
3. **Bridge Nodes** (orchestrator, reducer) - 30s startup
4. **Optional Services** (registry, redpanda-ui) - on-demand

**Performance**:
- **Import Time**: ~2-3 minutes for all images
- **Startup Time**: ~2 minutes for sequential service startup
- **Total Time**: ~5 minutes for complete remote setup

#### 4. **continue-migration.sh** (Recovery/Continuation)
**Purpose**: Continue partial migration after failure
**Location**: `/Volumes/PRO-G40/Code/omninode_bridge/continue-migration.sh`

**Capabilities**:
- ✅ Resume migration from checkpoint
- ✅ Skip already-transferred files
- ✅ Re-attempt remote setup
- ✅ Verify existing remote state

**Use Cases**:
- Network interruption during migration
- Remote system temporarily unavailable
- Partial transfer completion

### Current System Benefits

**Advantages**:
- ✅ **Battle-Tested**: Proven reliable in production use
- ✅ **Simple**: Easy to understand and modify
- ✅ **Fast**: Optimized for common workflows
- ✅ **Portable**: Standard bash scripts, no dependencies
- ✅ **Debuggable**: Clear execution flow, easy to troubleshoot
- ✅ **Self-Contained**: No external service dependencies

**Limitations**:
- ❌ **Manual Invocation**: Requires developer to run scripts
- ❌ **Limited Observability**: No event tracking or metrics
- ❌ **No Workflow Orchestration**: Cannot coordinate multi-service deployments
- ❌ **No Rollback Automation**: Manual rollback required
- ❌ **Single Target**: Designed for one remote system (192.168.86.200)
- ❌ **No Quality Gates**: No automated validation checkpoints

---

## Future State

### Automated ONEX v2.0 Workflow-Based Deployment

**Status**: 🚧 Implementation Complete, Validation Pending

The future deployment system leverages ONEX v2.0 contract-driven workflow orchestration with two specialized Effect nodes and a comprehensive workflow contract.

#### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Deployment Workflow Orchestrator                    │
│                 (deployment_workflow.yaml)                           │
└────────────┬──────────────────────────────────────┬──────────────────┘
             │                                       │
             │ Stage 1: Package                     │ Stage 2: Transfer
             │ Preparation                          │ Initiation
             ▼                                       ▼
┌─────────────────────────────┐         ┌────────────────────────────┐
│ NodeDeploymentSenderEffect  │────────▶│ Remote Hook Receiver       │
│                             │  HTTP   │ (192.168.86.200:8001)      │
│ - Build Docker image        │ Package │                            │
│ - Create deployment package │ Transfer│ - Receive package          │
│ - Generate BLAKE3 checksum  │         │ - Validate checksum        │
│ - Stream to receiver        │         │ - Trigger deployment       │
└────────────┬────────────────┘         └────────────┬───────────────┘
             │                                       │
             │ Kafka Events                          │
             ▼                                       ▼
┌─────────────────────────────┐         ┌────────────────────────────┐
│  Kafka Event Bus            │         │ NodeDeploymentReceiverEffect│
│                             │         │                             │
│ - BUILD_STARTED             │         │ - Load Docker image         │
│ - BUILD_COMPLETED           │         │ - Deploy container          │
│ - TRANSFER_STARTED          │         │ - Run health checks         │
│ - TRANSFER_COMPLETED        │         │ - Publish events            │
│ - DEPLOYMENT_COMPLETED      │         │                             │
└─────────────────────────────┘         └────────────┬───────────────┘
                                                      │
                                                      │ Stage 3: Deploy
                                                      │ Stage 4: Validate
                                                      ▼
                                        ┌────────────────────────────┐
                                        │ Container Deployed         │
                                        │ - Running on remote        │
                                        │ - Health checks passing    │
                                        │ - Service registered       │
                                        └────────────────────────────┘
```

### Components

#### 1. **NodeDeploymentSenderEffect**
**Type**: Effect Node
**Location**: `/Volumes/PRO-G40/Code/omninode_bridge/src/omninode_bridge/nodes/deployment_sender_effect/v1_0_0/`
**Contract**: `contract.yaml`

**Capabilities**:
- ✅ Docker image building from Dockerfiles
- ✅ Image export to compressed packages (gzip/zstd)
- ✅ BLAKE3 checksum generation for integrity
- ✅ HTTP/rsync transfer to remote receivers
- ✅ Kafka event publishing (5 event types)
- ✅ Streaming upload for large packages
- ✅ Transfer verification and retry logic

**Performance Requirements**:
- **Image Build**: <20s target (<15s typical)
- **Package Transfer**: <10s for 1GB package (125MB/s throughput)
- **Throughput**: 100+ deployments/hour
- **Concurrent Transfers**: Up to 5 simultaneous

**IO Operations**:
1. `package_container` - Build, export, compress, checksum (target: <15s)
2. `transfer_package` - Validate, upload, verify (target: <8s)
3. `publish_transfer_event` - Kafka event publishing (target: <50ms)

**Kafka Event Types**:
- `BUILD_STARTED` - Docker build initiated
- `BUILD_COMPLETED` - Image built successfully
- `TRANSFER_STARTED` - Package transfer initiated
- `TRANSFER_COMPLETED` - Package transferred and verified
- `DEPLOYMENT_FAILED` - Deployment operation failed

#### 2. **NodeDeploymentReceiverEffect**
**Type**: Effect Node
**Location**: `/Volumes/PRO-G40/Code/omninode_bridge/src/omninode_bridge/nodes/deployment_receiver_effect/v1_0_0/`
**Contract**: `contract.yaml`

**Capabilities**:
- ✅ HTTP package reception with authentication (HMAC)
- ✅ BLAKE3 hash validation for integrity
- ✅ IP whitelisting for security
- ✅ Docker image loading into daemon
- ✅ Container deployment with configuration
- ✅ Health check monitoring and validation
- ✅ Kafka event publishing

**Performance Requirements**:
- **Image Load**: <3s target
- **Container Start**: <2s target
- **Auth Validation**: <100ms target
- **Health Check**: <5s timeout

**IO Operations**:
1. `receive_package` - HTTP receive with auth (target: <5s)
2. `load_image` - Import into Docker daemon (target: <3s)
3. `deploy_container` - Start container with config (target: <2s)
4. `health_check` - Validate service availability (target: <5s)
5. `publish_deployment_event` - Kafka event publishing (target: <50ms)

**Security Features**:
- HMAC authentication for API requests
- IP whitelisting (192.168.86.0/24)
- Sandbox execution environment
- BLAKE3 checksum verification

#### 3. **Deployment Workflow Contract**
**Type**: Orchestrator Workflow
**Location**: `/Volumes/PRO-G40/Code/omninode_bridge/contracts/workflows/deployment_workflow.yaml`
**Schema Version**: ONEX v2.0

**Workflow Stages**:

| Stage | Name | Duration | Dependencies | Operations |
|-------|------|----------|--------------|------------|
| 1 | **Package Preparation** | <10s | None | Build image, gather metadata, validate |
| 2 | **Transfer Initiation** | <5s | Stage 1 | Connect to receiver, send package, verify |
| 3 | **Deployment Execution** | <10s | Stage 2 | Stop old container, deploy new, start |
| 4 | **Health Validation** | <5s | Stage 3 | Check status, validate endpoint, register |

**FSM States**:
```
PENDING ──▶ PACKAGING ──▶ TRANSFERRING ──▶ DEPLOYING ──▶ VALIDATING ──▶ COMPLETED
   │           │               │                │              │
   └───────────┴───────────────┴────────────────┴──────────────┴───────▶ FAILED
                                                 │
                                                 └───▶ ROLLING_BACK ──▶ ROLLBACK_COMPLETED
```

**Performance Targets**:
- **Small Containers** (<100MB): <30s total
- **Large Containers** (<1GB): <3 minutes total
- **Success Rate**: >95% deployment success
- **Concurrent Deployments**: Up to 5 simultaneous

**Quality Gates**:
1. `image_build_success` - Verify build completed (100% threshold)
2. `image_size_check` - Verify size <2GB (100% threshold)
3. `transfer_success` - Verify package transferred (100% threshold)
4. `deployment_success` - Verify container started (100% threshold)
5. `health_check_pass` - Verify service healthy (100% threshold)
6. `total_duration` - Verify time within target (80% threshold)

**Error Handling**:
- **Retry Policy**: Up to 3 attempts with exponential backoff
- **Rollback Strategy**: Automatic rollback on deployment failure
- **Circuit Breaker**: Abort workflow on critical failures
- **Manual Intervention**: Required only for rollback failures

### Future System Benefits

**Advantages**:
- ✅ **Event-Driven**: Complete observability via Kafka events
- ✅ **Orchestrated**: Multi-service deployment coordination
- ✅ **Validated**: Quality gates at every stage
- ✅ **Automated Rollback**: Automatic recovery from failures
- ✅ **Multi-Target**: Deployable to multiple remote systems
- ✅ **ONEX Compliant**: Contract-driven architecture
- ✅ **Observable**: Full metrics and tracing
- ✅ **Scalable**: Handles concurrent deployments

**Trade-offs**:
- ⚠️ **Complexity**: More components to understand and debug
- ⚠️ **Dependencies**: Requires Kafka, Consul, PostgreSQL
- ⚠️ **Learning Curve**: ONEX v2.0 workflow patterns
- ⚠️ **Validation Needed**: Not yet battle-tested in production

---

## Migration Strategy

### Phased Approach

The migration follows a **three-phase validation strategy** to ensure zero production disruption:

#### Phase 1: Validation (Current)
**Duration**: 2-4 weeks
**Status**: ⏳ In Progress

**Goals**:
- Deploy receiver node to 192.168.86.200 using existing scripts
- Test sender node locally in development environment
- Perform test deployments of non-critical containers
- Validate all workflow stages and error handling
- Measure performance against targets

**Success Criteria**:
- [ ] Receiver node deployed and healthy on remote system
- [ ] Sender node successfully builds and transfers packages
- [ ] End-to-end deployment completes for test container
- [ ] All quality gates pass with >90% success rate
- [ ] Rollback mechanism functions correctly
- [ ] Kafka events published and traced successfully

**Activities**:
1. **Week 1-2**: Deploy receiver node using `rebuild-service.sh`
   ```bash
   # Build receiver node locally
   docker build -f docker/deployment-receiver/Dockerfile \
     -t omninode_bridge-deployment-receiver:latest .

   # Deploy to remote using existing script
   ./rebuild-service.sh deployment-receiver 192.168.86.200
   ```

2. **Week 2-3**: Test sender node locally
   ```python
   # Test deployment workflow
   from omninode_bridge.nodes.deployment_sender_effect.v1_0_0 import NodeDeploymentSenderEffect

   result = await sender.execute_effect({
       "container_name": "test-container",
       "image_tag": "latest",
       "remote_receiver_url": "http://192.168.86.200:8001/deploy"
   })
   ```

3. **Week 3-4**: End-to-end validation
   - Deploy `hook-receiver` (low-risk service)
   - Deploy `model-metrics` (low-risk service)
   - Validate metrics and events
   - Test rollback scenarios

**Rollback Plan**: Continue using `rebuild-service.sh` for all production deployments

#### Phase 2: Parallel Operation
**Duration**: 1-2 weeks
**Status**: 🔜 Pending Phase 1 Completion

**Goals**:
- Use both scripts AND nodes for production deployments
- Compare reliability, performance, and observability
- Document any issues with node-based deployment
- Train team on new workflow patterns

**Success Criteria**:
- [ ] 10+ successful node-based deployments of production services
- [ ] Node-based deployments achieve <10% time variance vs script-based
- [ ] Zero production incidents caused by node-based deployments
- [ ] Team comfortable with ONEX workflow debugging
- [ ] Monitoring dashboards show complete deployment visibility

**Activities**:
1. **Week 1**: Parallel deployments
   - Deploy same service using both methods
   - Compare deployment times and success rates
   - Identify any gaps in node-based approach

2. **Week 2**: Production validation
   - Deploy all services using node-based workflow
   - Keep scripts as immediate rollback option
   - Document lessons learned

**Rollback Plan**: Immediately revert to `rebuild-service.sh` on any production issue

#### Phase 3: Deprecation
**Duration**: 1 month
**Status**: 🔜 Pending Phase 2 Completion

**Goals**:
- Officially deprecate manual scripts
- Archive scripts for historical reference
- Update all documentation to node-based workflow
- Remove scripts from active use

**Success Criteria**:
- [ ] 30+ successful node-based deployments without scripts
- [ ] Team exclusively using node-based workflow
- [ ] Documentation fully updated
- [ ] Scripts archived and no longer in PATH

**Activities**:
1. **Week 1**: Rename scripts to `.deprecated`
   ```bash
   mv migrate-to-remote.sh migrate-to-remote.sh.deprecated
   mv rebuild-service.sh rebuild-service.sh.deprecated
   mv setup-remote.sh setup-remote.sh.deprecated
   mv continue-migration.sh continue-migration.sh.deprecated
   ```

2. **Week 2**: Update documentation
   - Update `CLAUDE.md` to reference node-based deployment
   - Update `REMOTE_MIGRATION_GUIDE.md` for new workflow
   - Add deprecation notices to script headers

3. **Week 3-4**: Archive scripts
   ```bash
   mkdir -p docs/archive/deployment-scripts
   mv *.sh.deprecated docs/archive/deployment-scripts/
   ```

**Rollback Plan**: Restore scripts from archive if critical issues arise

---

## Script Preservation Strategy

### Why We Keep Scripts During Validation

**Rationale**:
1. **Battle-Tested Reliability**: Scripts have proven 100% reliable in production
2. **Fast Rollback**: Immediate recovery path if node-based deployment fails
3. **Validation Baseline**: Compare new system performance against known baseline
4. **Deployment Continuity**: Production deployments continue without risk
5. **Team Confidence**: Developers can fall back to familiar tools

### Script Locations and Lifecycle

| Script | Location | Status | Deprecation Timeline |
|--------|----------|--------|---------------------|
| `migrate-to-remote.sh` | Root directory | ✅ Active | After Phase 2 (1-2 months) |
| `rebuild-service.sh` | Root directory | ✅ Active | After Phase 2 (1-2 months) |
| `setup-remote.sh` | Root directory | ✅ Active | After Phase 2 (1-2 months) |
| `continue-migration.sh` | Root directory | ✅ Active | After Phase 2 (1-2 months) |

### Bootstrap Deployment Pattern

**Critical Use Case**: Use scripts to deploy the new deployment nodes themselves

```bash
# Step 1: Deploy receiver node to remote system using rebuild-service.sh
./rebuild-service.sh deployment-receiver 192.168.86.200

# Step 2: Verify receiver node is healthy
curl http://192.168.86.200:8001/health

# Step 3: Test sender node locally
python -m omninode_bridge.nodes.deployment_sender_effect.v1_0_0.node

# Step 4: Use node-based deployment for subsequent services
# (Receiver is now available to handle node-based deployments)
```

**Bootstrap Paradox Solution**: Manual scripts deploy the automated deployment system, creating a self-sustaining workflow.

### Deprecation Process

**Timeline**: After X successful node-based deployments (target: 20+)

**Step 1: Suffix Rename** (Week 1)
```bash
# Add .deprecated suffix to signal intent
mv migrate-to-remote.sh migrate-to-remote.sh.deprecated
mv rebuild-service.sh rebuild-service.sh.deprecated
mv setup-remote.sh setup-remote.sh.deprecated
mv continue-migration.sh continue-migration.sh.deprecated

# Update shebangs to prevent accidental execution
for file in *.sh.deprecated; do
    echo "#!/bin/bash" > "$file.tmp"
    echo "echo 'WARNING: This script is deprecated. Use ONEX deployment workflow.'" >> "$file.tmp"
    echo "echo 'See docs/deployment/DEPLOYMENT_SYSTEM_MIGRATION.md for migration guide.'" >> "$file.tmp"
    echo "exit 1" >> "$file.tmp"
    cat "$file" >> "$file.tmp"
    mv "$file.tmp" "$file"
done
```

**Step 2: Documentation Update** (Week 2)
- Update `CLAUDE.md` with deployment system overview
- Update `README.md` to reference node-based deployment
- Add deprecation notices to `REMOTE_MIGRATION_GUIDE.md`
- Create `docs/archive/LEGACY_DEPLOYMENT_SCRIPTS.md` guide

**Step 3: Archive** (Week 3-4)
```bash
# Create archive directory
mkdir -p docs/archive/deployment-scripts

# Move deprecated scripts
mv *.sh.deprecated docs/archive/deployment-scripts/

# Create archive README
cat > docs/archive/deployment-scripts/README.md << 'EOF'
# Legacy Deployment Scripts (Archived)

**Status**: Deprecated as of 2025-XX-XX
**Replaced By**: ONEX v2.0 Deployment Workflow

These scripts are preserved for historical reference and emergency rollback only.
For current deployment procedures, see:
- docs/deployment/DEPLOYMENT_SYSTEM_MIGRATION.md
- docs/deployment/REMOTE_MIGRATION_GUIDE.md
EOF
```

**Step 4: Removal** (After 1 month in archive)
```bash
# Final removal after 1 month of successful node-based deployments
# Only if zero production issues and team consensus
rm -rf docs/archive/deployment-scripts
```

---

## Validation Criteria

### When to Retire Manual Scripts

**Quantitative Criteria**:
1. ✅ **20+ successful node-based deployments** across all services
2. ✅ **<5% deployment time variance** compared to script-based baseline
3. ✅ **>95% deployment success rate** for node-based workflow
4. ✅ **Zero production incidents** caused by node-based deployments
5. ✅ **100% team adoption** of node-based workflow
6. ✅ **Full observability** via Kafka events and metrics

**Qualitative Criteria**:
1. ✅ Team **confident** debugging node-based deployment issues
2. ✅ Documentation **complete** and **accurate** for new workflow
3. ✅ Rollback procedures **tested** and **validated**
4. ✅ Performance **meets or exceeds** script-based baseline
5. ✅ Quality gates **prevent** bad deployments from reaching production
6. ✅ Monitoring dashboards **provide visibility** into deployment health

### Metrics to Track

**Deployment Performance**:
```python
# Track these metrics during parallel operation phase
metrics = {
    "deployment_duration_seconds": {
        "script_based": [45, 52, 48, 50],  # Baseline
        "node_based": [47, 49, 51, 48],     # Compare
        "variance_threshold": 0.10          # <10% difference
    },
    "success_rate": {
        "script_based": 1.00,   # 100%
        "node_based": 0.95,      # Target: >95%
        "min_threshold": 0.95
    },
    "rollback_success_rate": {
        "node_based": 1.00,      # Must be 100%
        "min_threshold": 1.00
    },
    "observability_events": {
        "node_based": 7,         # Events per deployment
        "min_threshold": 5       # Minimum event coverage
    }
}
```

**Quality Metrics**:
```yaml
quality_gates:
  - name: "deployment_reliability"
    description: "Node-based deployments succeed consistently"
    threshold: 0.95
    current_value: 0.00  # Update as deployments occur

  - name: "performance_parity"
    description: "Node-based deployments match script performance"
    threshold: 0.90
    current_value: 0.00  # Update as deployments occur

  - name: "rollback_effectiveness"
    description: "Rollback mechanism recovers from all failures"
    threshold: 1.00
    current_value: 0.00  # Update as rollbacks occur

  - name: "team_confidence"
    description: "Team comfortable with new workflow"
    threshold: 0.80
    current_value: 0.00  # Update via team survey
```

### Validation Dashboard

**Recommended Monitoring**:
```python
# Grafana dashboard queries for validation tracking

# Deployment success rate (last 7 days)
sum(rate(deployments_completed_total[7d])) /
sum(rate(deployments_started_total[7d]))

# Average deployment duration by method
avg(deployment_duration_seconds{method="node_based"})
avg(deployment_duration_seconds{method="script_based"})

# Rollback frequency
rate(deployments_rolled_back_total[7d])

# Kafka event coverage
sum(rate(kafka_events_published_total[1h])) by (event_type)
```

---

## Rollback Plan

### Immediate Rollback (During Validation)

**Trigger Conditions**:
- Node-based deployment fails to complete
- Service health check fails after node-based deployment
- Performance degradation >20% vs script baseline
- Team unable to debug node-based deployment issue
- Production incident caused by node-based deployment

**Rollback Procedure**:

**Step 1: Stop Current Deployment**
```bash
# If deployment workflow is running, cancel it
curl -X POST http://localhost:8060/workflows/{workflow_id}/cancel

# Verify workflow stopped
curl http://localhost:8060/workflows/{workflow_id}/status
```

**Step 2: Revert Using Script**
```bash
# Use rebuild-service.sh to redeploy last known good version
./rebuild-service.sh orchestrator 192.168.86.200

# Verify service health
curl http://192.168.86.200:8060/health
```

**Step 3: Document Incident**
```bash
# Capture deployment logs
curl http://192.168.86.200:8060/workflows/{workflow_id}/logs > rollback_incident_$(date +%Y%m%d_%H%M%S).log

# Note: Kafka events are preserved for post-mortem analysis
```

**Step 4: Post-Incident Analysis**
- Review Kafka event timeline
- Analyze deployment metrics
- Identify root cause
- Update validation criteria
- Fix issues before retry

**Recovery Time Objective (RTO)**: <5 minutes

### Emergency Rollback (After Deprecation)

**Scenario**: Node-based deployment system fails after scripts are deprecated

**Recovery Procedure**:

**Step 1: Restore Scripts from Archive**
```bash
# Restore scripts from archive
cp docs/archive/deployment-scripts/*.sh.deprecated .
for file in *.sh.deprecated; do
    mv "$file" "${file%.deprecated}"
done

# Make executable
chmod +x *.sh
```

**Step 2: Execute Emergency Deployment**
```bash
# Use restored script for emergency deployment
./rebuild-service.sh <failed_service> 192.168.86.200
```

**Step 3: Validate Recovery**
```bash
# Verify all services healthy
for port in 8001 8005 8060 8061 8057 8058; do
    curl -s http://192.168.86.200:$port/health || echo "Service on port $port failed"
done
```

**Step 4: Incident Response**
- Escalate to team lead
- Schedule post-mortem review
- Restore node-based deployment system
- Re-validate before deprecating scripts again

**Recovery Time Objective (RTO)**: <15 minutes

---

## Deployment Workflow Usage Guide

### Using NodeDeploymentSenderEffect

**Prerequisites**:
1. Docker daemon running locally
2. Deployment receiver accessible at remote endpoint
3. Kafka broker available for event publishing
4. Proper environment variables configured

**Example: Deploy Orchestrator to Remote System**

```python
#!/usr/bin/env python3
"""
Example: Deploy orchestrator container to remote system using ONEX workflow
"""

import asyncio
from uuid import uuid4
from omninode_bridge.nodes.deployment_sender_effect.v1_0_0 import NodeDeploymentSenderEffect
from omnibase_core.models.core import ModelContainer, ModelContractEffect

async def deploy_orchestrator():
    """Deploy orchestrator to remote Mac Studio"""

    # Initialize container and node
    container = ModelContainer()
    sender = NodeDeploymentSenderEffect(container)

    # Create deployment contract
    contract = ModelContractEffect(
        name="deploy_orchestrator",
        version="1.0.0",
        description="Deploy orchestrator to 192.168.86.200",
        node_type="effect",
        input_state={
            "container_name": "orchestrator",
            "image_tag": "latest",
            "dockerfile_path": "docker/bridge-nodes/Dockerfile.orchestrator",
            "build_context": "/Volumes/PRO-G40/Code/omninode_bridge",
            "build_args": {
                "SERVICE_NAME": "orchestrator",
                "PYTHON_VERSION": "3.11"
            },
            "remote_receiver_url": "http://192.168.86.200:8001/deploy",
            "transfer_method": "http",
            "compression": "gzip",
            "verify_checksum": True,
            "correlation_id": str(uuid4()),
            "deployment_metadata": {
                "environment": "production",
                "target_host": "192.168.86.200",
                "deployment_strategy": "replace"
            }
        }
    )

    # Execute deployment
    print("🚀 Starting deployment of orchestrator...")
    result = await sender.execute_effect(contract)

    # Check result
    if result["success"]:
        print(f"✅ Deployment successful!")
        print(f"   Package ID: {result['package_id']}")
        print(f"   Image ID: {result['image_id']}")
        print(f"   Package Size: {result['package_size_mb']:.2f} MB")
        print(f"   Build Duration: {result['build_duration_ms']} ms")
        print(f"   Transfer Duration: {result['transfer_duration_ms']} ms")
        print(f"   Total Duration: {result['execution_time_ms']} ms")
        print(f"   Kafka Events: {', '.join(result['kafka_events_published'])}")
    else:
        print(f"❌ Deployment failed: {result.get('error_message', 'Unknown error')}")
        print(f"   Error Code: {result.get('error_code', 'N/A')}")

    return result

if __name__ == "__main__":
    asyncio.run(deploy_orchestrator())
```

**Expected Output**:
```
🚀 Starting deployment of orchestrator...
🔨 Building Docker image... (15.2s)
📦 Creating deployment package... (2.1s)
🔐 Generating BLAKE3 checksum... (0.3s)
📤 Transferring package to 192.168.86.200... (7.8s)
✅ Deployment successful!
   Package ID: 3f7a8b2c-4d1e-4f9a-b8c6-1e2d3f4a5b6c
   Image ID: sha256:a1b2c3d4e5f6...
   Package Size: 487.32 MB
   Build Duration: 15234 ms
   Transfer Duration: 7821 ms
   Total Duration: 25387 ms
   Kafka Events: BUILD_STARTED, BUILD_COMPLETED, TRANSFER_STARTED, TRANSFER_COMPLETED
```

### Using NodeDeploymentReceiverEffect

**Deployment**: Receiver runs as service on remote system (192.168.86.200:8001)

**Initial Deployment** (using scripts):
```bash
# Deploy receiver node using rebuild-service.sh
./rebuild-service.sh deployment-receiver 192.168.86.200

# Verify receiver is healthy
curl http://192.168.86.200:8001/health
```

**Receiver automatically handles**:
- Package reception and validation
- Docker image loading
- Container deployment
- Health check monitoring
- Kafka event publishing

### Orchestrating with deployment_workflow.yaml

**Example: Complete Deployment Workflow**

```python
#!/usr/bin/env python3
"""
Example: Execute complete deployment workflow with orchestration
"""

import asyncio
from uuid import uuid4
from omninode_bridge.workflows.deployment_workflow import DeploymentWorkflow

async def orchestrated_deployment():
    """Execute orchestrated deployment with all stages"""

    # Initialize workflow
    workflow = DeploymentWorkflow()

    # Define workflow input
    workflow_input = {
        "container_name": "orchestrator",
        "image_tag": "v1.0.0",
        "remote_host": "192.168.86.200",
        "remote_port": 8001,
        "deployment_config": {
            "environment_variables": {
                "LOG_LEVEL": "INFO",
                "WORKERS": "4"
            },
            "port_mappings": [
                {"host_port": 8060, "container_port": 8060},
                {"host_port": 9094, "container_port": 9091}
            ],
            "network_mode": "bridge",
            "restart_policy": "unless-stopped"
        },
        "build_options": {
            "dockerfile_path": "docker/bridge-nodes/Dockerfile.orchestrator",
            "build_context": ".",
            "no_cache": False
        },
        "health_check_config": {
            "health_endpoint": "/health",
            "expected_status_code": 200,
            "max_retries": 5,
            "retry_interval_ms": 2000
        },
        "rollback_on_failure": True,
        "skip_health_validation": False,
        "workflow_id": str(uuid4())
    }

    # Execute workflow
    print("🚀 Starting orchestrated deployment workflow...")
    print(f"   Container: {workflow_input['container_name']}")
    print(f"   Target: {workflow_input['remote_host']}")
    print(f"   Workflow ID: {workflow_input['workflow_id']}")

    result = await workflow.execute(workflow_input)

    # Print results
    if result["success"]:
        print(f"\n✅ Deployment workflow completed successfully!")
        print(f"   Workflow ID: {result['workflow_id']}")
        print(f"   Deployment ID: {result['deployment_id']}")
        print(f"   Container ID: {result['container_id']}")
        print(f"   Deployed URL: {result['deployed_url']}")
        print(f"   Total Duration: {result['total_duration_ms']} ms")
        print(f"\n📊 Stage Results:")
        for stage in result["stage_results"]:
            status = "✅" if stage["success"] else "❌"
            print(f"   {status} {stage['stage_name']}: {stage['duration_ms']} ms")
        print(f"\n🏥 Health Status:")
        print(f"   Container Running: {result['health_status']['container_running']}")
        print(f"   Health Endpoint: {result['health_status']['health_endpoint_accessible']}")
        print(f"   Service Registered: {result['health_status']['service_registered']}")
    else:
        print(f"\n❌ Deployment workflow failed!")
        print(f"   Error: {result['error_message']}")
        if result.get('rollback_performed'):
            print(f"   Rollback Performed: {result['rollback_result']['success']}")
        print(f"\n📊 Failed Stages: {', '.join(result['stages_failed'])}")

    return result

if __name__ == "__main__":
    asyncio.run(orchestrated_deployment())
```

**Expected Output** (Success):
```
🚀 Starting orchestrated deployment workflow...
   Container: orchestrator
   Target: 192.168.86.200
   Workflow ID: 7b8c9d0e-1f2a-3b4c-5d6e-7f8a9b0c1d2e

✅ Deployment workflow completed successfully!
   Workflow ID: 7b8c9d0e-1f2a-3b4c-5d6e-7f8a9b0c1d2e
   Deployment ID: 1a2b3c4d-5e6f-7a8b-9c0d-1e2f3a4b5c6d
   Container ID: abc123def456
   Deployed URL: http://192.168.86.200:8060
   Total Duration: 28734 ms

📊 Stage Results:
   ✅ package_preparation: 9823 ms
   ✅ transfer_initiation: 4521 ms
   ✅ deployment_execution: 8932 ms
   ✅ health_validation: 5458 ms

🏥 Health Status:
   Container Running: True
   Health Endpoint: True
   Service Registered: True
```

---

## Transition Checklist

### Phase 1: Validation (Weeks 1-4)

#### Week 1-2: Deploy Receiver Node
- [ ] Build deployment receiver image locally
  ```bash
  docker build -f docker/deployment-receiver/Dockerfile \
    -t omninode_bridge-deployment-receiver:latest .
  ```
- [ ] Deploy receiver to 192.168.86.200 using `rebuild-service.sh`
  ```bash
  ./rebuild-service.sh deployment-receiver 192.168.86.200
  ```
- [ ] Verify receiver health endpoint
  ```bash
  curl http://192.168.86.200:8001/health
  ```
- [ ] Configure receiver authentication (HMAC keys, IP whitelist)
- [ ] Test receiver package reception endpoint
  ```bash
  curl -X POST http://192.168.86.200:8001/deploy \
    -H "Content-Type: application/json" \
    -d '{"test": true}'
  ```

#### Week 2-3: Test Sender Node Locally
- [ ] Initialize sender node in development environment
- [ ] Test Docker image building capability
- [ ] Test package creation and compression
- [ ] Test BLAKE3 checksum generation
- [ ] Test HTTP transfer to receiver
- [ ] Validate Kafka event publishing
  ```bash
  # Monitor Kafka events
  docker exec -it omninode-bridge-redpanda rpk topic consume \
    dev.omninode-bridge.deployment.build-started.v1
  ```
- [ ] Test transfer verification logic
- [ ] Test error handling and retry logic

#### Week 3-4: End-to-End Validation
- [ ] Deploy low-risk service (`hook-receiver`) using node-based workflow
- [ ] Deploy second low-risk service (`model-metrics`) using node-based workflow
- [ ] Validate deployment metrics against targets
  - [ ] Build time <20s
  - [ ] Transfer time <10s for 1GB
  - [ ] Total time <60s for complete workflow
- [ ] Test rollback mechanism with intentional failure
- [ ] Verify all Kafka events published correctly
- [ ] Check quality gates all passed
- [ ] Document any issues or deviations from expected behavior

### Phase 2: Parallel Operation (Weeks 5-6)

#### Week 5: Parallel Deployment Comparison
- [ ] Deploy same service using both scripts AND nodes
- [ ] Compare deployment times (variance <10%)
- [ ] Compare success rates (node-based >95%)
- [ ] Compare observability (Kafka events vs logs)
- [ ] Identify any gaps or issues in node-based approach
- [ ] Document performance metrics
  ```python
  {
    "service": "orchestrator",
    "script_duration_s": 48,
    "node_duration_s": 52,
    "variance_pct": 8.3,  # <10% threshold
    "script_success": True,
    "node_success": True,
    "kafka_events": 7
  }
  ```

#### Week 6: Production Validation
- [ ] Deploy all bridge nodes using node-based workflow
  - [ ] orchestrator
  - [ ] reducer
  - [ ] hook-receiver
  - [ ] model-metrics
  - [ ] metadata-stamping
  - [ ] onextree
- [ ] Keep scripts available for immediate rollback
- [ ] Monitor production stability (zero incidents)
- [ ] Train team on ONEX workflow debugging
- [ ] Update team runbooks with new procedures
- [ ] Document lessons learned and best practices

### Phase 3: Deprecation (Weeks 7-10)

#### Week 7: Script Renaming
- [ ] Rename all deployment scripts to `.deprecated`
  ```bash
  mv migrate-to-remote.sh migrate-to-remote.sh.deprecated
  mv rebuild-service.sh rebuild-service.sh.deprecated
  mv setup-remote.sh setup-remote.sh.deprecated
  mv continue-migration.sh continue-migration.sh.deprecated
  ```
- [ ] Add deprecation warnings to script headers
- [ ] Update team communication about deprecation
- [ ] Create archive preparation plan

#### Week 8: Documentation Update
- [ ] Update `CLAUDE.md` with deployment system overview
- [ ] Update `README.md` to reference node-based deployment
- [ ] Update `REMOTE_MIGRATION_GUIDE.md` for new workflow
- [ ] Create `docs/archive/LEGACY_DEPLOYMENT_SCRIPTS.md`
- [ ] Update all runbooks and operational guides
- [ ] Remove script references from CI/CD pipelines

#### Week 9-10: Script Archival
- [ ] Verify 30+ successful node-based deployments without scripts
- [ ] Confirm team exclusively using node-based workflow
- [ ] Create archive directory structure
  ```bash
  mkdir -p docs/archive/deployment-scripts
  ```
- [ ] Move deprecated scripts to archive
  ```bash
  mv *.sh.deprecated docs/archive/deployment-scripts/
  ```
- [ ] Create archive README with historical context
- [ ] Update git history documentation

#### Week 11+: Monitoring and Optimization
- [ ] Monitor deployment metrics for 1 month
- [ ] Optimize workflow performance based on data
- [ ] Address any emerging issues
- [ ] Plan for multi-remote-system support
- [ ] Consider permanent removal of archived scripts (after 1 month)

---

## Timeline and Milestones

### Migration Timeline Overview

```
Week 1-2: Deploy Receiver Node
├─ Build receiver image
├─ Deploy using rebuild-service.sh
├─ Verify health and authentication
└─ Test package reception

Week 2-3: Test Sender Node
├─ Initialize sender in dev environment
├─ Test all IO operations
├─ Validate Kafka event publishing
└─ Test error handling

Week 3-4: End-to-End Validation
├─ Deploy low-risk services
├─ Measure performance metrics
├─ Test rollback mechanism
└─ Document findings

Week 5: Parallel Deployment Comparison
├─ Deploy services with both methods
├─ Compare performance and reliability
├─ Identify gaps
└─ Document metrics

Week 6: Production Validation
├─ Deploy all services with nodes
├─ Monitor for incidents
├─ Train team
└─ Update runbooks

Week 7: Script Deprecation
├─ Rename scripts to .deprecated
├─ Add deprecation warnings
└─ Update team communication

Week 8: Documentation Update
├─ Update all docs
├─ Create archive docs
└─ Update CI/CD

Week 9-10: Script Archival
├─ Verify 30+ deployments
├─ Move scripts to archive
└─ Create archive README

Week 11+: Monitoring and Optimization
├─ Monitor deployment health
├─ Optimize performance
└─ Plan multi-remote support
```

### Key Milestones

| Milestone | Target Date | Status | Success Criteria |
|-----------|-------------|--------|------------------|
| **M1**: Receiver deployed | Week 2 | 🔜 Pending | Receiver healthy on 192.168.86.200 |
| **M2**: Sender validated | Week 3 | 🔜 Pending | All IO operations working |
| **M3**: First node-based deployment | Week 4 | 🔜 Pending | Hook-receiver deployed successfully |
| **M4**: Parallel operation started | Week 5 | 🔜 Pending | Both methods in use |
| **M5**: Production validation complete | Week 6 | 🔜 Pending | 10+ successful deployments |
| **M6**: Scripts deprecated | Week 7 | 🔜 Pending | Scripts renamed to .deprecated |
| **M7**: Documentation updated | Week 8 | 🔜 Pending | All docs reference new workflow |
| **M8**: Scripts archived | Week 10 | 🔜 Pending | Scripts moved to archive |
| **M9**: Full adoption | Week 11+ | 🔜 Pending | Team exclusively using nodes |

### Success Metrics Dashboard

**Track these metrics throughout migration**:

```yaml
# Deployment Performance
deployment_performance:
  script_baseline:
    avg_duration_s: 50
    p95_duration_s: 65
    success_rate: 1.00

  node_based:
    avg_duration_s: TBD  # Update during validation
    p95_duration_s: TBD  # Update during validation
    success_rate: TBD     # Target: >0.95

  variance_threshold: 0.10  # <10% acceptable

# Quality Gates
quality_gates_passed:
  image_build_success: TBD      # Target: 1.00
  image_size_check: TBD         # Target: 1.00
  transfer_success: TBD         # Target: 1.00
  deployment_success: TBD       # Target: 1.00
  health_check_pass: TBD        # Target: 1.00
  total_duration: TBD           # Target: 0.80

# Observability
kafka_events:
  events_per_deployment: TBD    # Target: 5+
  event_publish_success_rate: TBD  # Target: >0.99

# Team Adoption
team_adoption:
  deployments_using_nodes: 0    # Target: 20+
  deployments_using_scripts: TBD
  team_confidence_score: TBD    # Target: >0.80 (survey)
```

---

## Conclusion

This migration strategy ensures a **safe, validated transition** from manual shell script-based deployment to automated ONEX v2.0 workflow-based deployment. By maintaining both systems in parallel during validation, we minimize production risk while gaining the benefits of event-driven orchestration, quality gates, and comprehensive observability.

**Key Success Factors**:
1. ✅ **No production disruption**: Scripts remain available throughout validation
2. ✅ **Measured validation**: Quantitative metrics guide deprecation decision
3. ✅ **Team confidence**: Parallel operation builds familiarity and trust
4. ✅ **Fast rollback**: Immediate recovery path if issues arise
5. ✅ **Bootstrap deployment**: Scripts deploy the automated system itself

**Next Steps**:
1. Begin Phase 1 validation (deploy receiver node)
2. Track metrics against success criteria
3. Update this document with actual performance data
4. Proceed to Phase 2 only after Phase 1 criteria met
5. Deprecate scripts only after Phase 2 criteria met

**Questions or Issues?**
See related documentation:
- [Remote Migration Guide](./REMOTE_MIGRATION_GUIDE.md)
- [Bridge Nodes Guide](../guides/BRIDGE_NODES_GUIDE.md)
- [Pre-Deployment Checklist](./PRE_DEPLOYMENT_CHECKLIST.md)

---

**Document Version**: 1.0.0
**Last Updated**: 2025-10-25
**Maintained By**: OmniNode Team
**Correlation ID**: deployment-docs-001
