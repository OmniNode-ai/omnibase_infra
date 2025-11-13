# Deployment Workflow Integration Guide

**Version**: 1.0.0
**Last Updated**: 2025-10-25
**Status**: Implementation Guide

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Integration Patterns](#integration-patterns)
5. [Configuration](#configuration)
6. [Workflow Stages](#workflow-stages)
7. [Error Handling](#error-handling)
8. [Monitoring](#monitoring)
9. [Testing](#testing)
10. [Practical Examples](#practical-examples)
11. [Troubleshooting](#troubleshooting)

---

## Overview

The **Deployment Workflow** is an ONEX v2.0 compliant orchestrator workflow for automated container deployment to remote systems. It coordinates Docker image packaging, secure transfer to remote hook receivers, deployment execution, and health validation with comprehensive rollback support.

### Purpose

- **Automated Deployment**: End-to-end container deployment automation
- **Remote Execution**: Deploy containers to remote systems (e.g., 192.168.86.200)
- **State Management**: FSM-driven workflow state tracking
- **Event Publishing**: Kafka event streaming for observability
- **Rollback Support**: Automatic rollback on deployment failures

### Key Features

- **4-Stage Workflow**: Package → Transfer → Deploy → Validate
- **Performance**: <30s for small containers, <3min for large containers
- **FSM State Machine**: 9 states with defined transitions
- **Kafka Integration**: 7 event types for lifecycle tracking
- **Quality Gates**: 6 validation gates across workflow stages
- **Health Validation**: Comprehensive health checking with retries

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Deployment Workflow                      │
│              (NodeBridgeOrchestrator-driven)                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 1: Package Preparation (Local System)                 │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ NodeDeploymentSenderEffect                             │  │
│  │  - Build Docker image from Dockerfile                  │  │
│  │  - Export image to tar.gz package                      │  │
│  │  - Generate BLAKE3 checksum                            │  │
│  │  - Gather deployment metadata                          │  │
│  │  - Validate image integrity                            │  │
│  │  Performance: <20s image build, <5s export/checksum    │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │ (Docker image package + metadata)
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 2: Transfer Initiation                                │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ NodeDeploymentSenderEffect                             │  │
│  │  - Establish connection to remote (192.168.86.200)     │  │
│  │  - POST /api/deploy with HMAC authentication           │  │
│  │  - Transfer package via HTTP/rsync                     │  │
│  │  - Verify transfer receipt                             │  │
│  │  Performance: <10s transfer for 1GB packages           │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │ (Transfer receipt + remote deployment ID)
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 3: Deployment Execution (Remote System)               │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ NodeDeploymentReceiverEffect (192.168.86.200:8001)     │  │
│  │  - Validate HMAC authentication                        │  │
│  │  - Verify BLAKE3 checksum                              │  │
│  │  - Stop existing container (if present)                │  │
│  │  - Load Docker image into daemon                       │  │
│  │  - Deploy container with configuration                 │  │
│  │  - Start container with restart policy                 │  │
│  │  Performance: <10s deployment                          │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │ (Container ID + deployment logs)
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 4: Health Validation                                  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ NodeDeploymentReceiverEffect                           │  │
│  │  - Check container status (docker ps)                  │  │
│  │  - Validate health endpoint (HTTP GET /health)         │  │
│  │  - Verify service registration (Consul, optional)      │  │
│  │  - Retry up to 5 times with 2s interval               │  │
│  │  Performance: <5s health checks                        │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │ (Health check results)
                       ▼
                 ┌───────────┐
                 │ COMPLETED │ (Deployment successful)
                 └───────────┘
                       OR
                 ┌────────────┐
                 │ ROLLBACK   │ (Restore previous container)
                 └────────────┘
```

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                   Local Development System                  │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ NodeBridgeOrchestrator                                │  │
│  │  - Workflow coordination                              │  │
│  │  - FSM state management                               │  │
│  │  - Event publishing                                   │  │
│  └───────────────┬───────────────────────────────────────┘  │
│                  │                                           │
│  ┌───────────────▼───────────────────────────────────────┐  │
│  │ NodeDeploymentSenderEffect                            │  │
│  │  - Docker image building                              │  │
│  │  - Package compression                                │  │
│  │  - BLAKE3 hashing                                     │  │
│  │  - HTTP/rsync transfer                                │  │
│  └───────────────┬───────────────────────────────────────┘  │
│                  │                                           │
└──────────────────┼───────────────────────────────────────────┘
                   │ (HTTPS POST with HMAC auth)
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│              Remote System (192.168.86.200)                 │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Hook Receiver API (Port 8001)                         │  │
│  │  - HMAC authentication                                │  │
│  │  - IP whitelisting                                    │  │
│  │  - Package validation                                 │  │
│  └───────────────┬───────────────────────────────────────┘  │
│                  │                                           │
│  ┌───────────────▼───────────────────────────────────────┐  │
│  │ NodeDeploymentReceiverEffect                          │  │
│  │  - Package validation                                 │  │
│  │  - Image loading                                      │  │
│  │  - Container deployment                               │  │
│  │  - Health checking                                    │  │
│  └───────────────┬───────────────────────────────────────┘  │
│                  │                                           │
│  ┌───────────────▼───────────────────────────────────────┐  │
│  │ Docker Daemon                                         │  │
│  │  - Container runtime                                  │  │
│  │  - Image management                                   │  │
│  │  - Network/volume management                          │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                   │
                   ▼
           ┌───────────────┐
           │ Kafka Events  │ (Both systems publish events)
           └───────────────┘
```

### FSM State Machine

```
DEPLOYMENT WORKFLOW STATE MACHINE

┌─────────┐
│ PENDING │ (Initial state - workflow created)
└────┬────┘
     │ start_workflow
     ▼
┌──────────┐
│PACKAGING │ (Building Docker image, gathering metadata)
└────┬─────┘
     │ package_complete
     ▼
┌──────────────┐
│TRANSFERRING  │ (Sending package to remote)
└──────┬───────┘
       │ transfer_complete
       ▼
┌──────────┐
│DEPLOYING │ (Deploying container on remote)
└────┬─────┘
     │ deployment_complete
     ▼
┌────────────┐
│ VALIDATING │ (Health checks)
└──────┬─────┘
       │
       ├──► ┌───────────┐
       │    │ COMPLETED │ (Success - terminal state)
       │    └───────────┘
       │
       └──► ┌──────────────┐
            │ ROLLING_BACK │ (Deployment failed, restoring)
            └──────┬───────┘
                   │
                   ├──► ┌────────────────────┐
                   │    │ROLLBACK_COMPLETED  │ (Restored previous state)
                   │    └────────────────────┘
                   │
                   └──► ┌────────┐
                        │ FAILED │ (Rollback also failed)
                        └────────┘
```

**State Transitions**:
- `PENDING → PACKAGING`: Workflow initiated
- `PACKAGING → TRANSFERRING`: Image built and validated
- `TRANSFERRING → DEPLOYING`: Package transferred successfully
- `DEPLOYING → VALIDATING`: Container deployed and started
- `VALIDATING → COMPLETED`: Health checks passed
- `VALIDATING → ROLLING_BACK`: Health checks failed
- `DEPLOYING → ROLLING_BACK`: Deployment failed
- `ROLLING_BACK → ROLLBACK_COMPLETED`: Previous container restored
- `ROLLING_BACK → FAILED`: Rollback failed (manual intervention required)

### Kafka Event Flow

```
Deployment Lifecycle Events:

1. DEPLOYMENT_STARTED
   Topic: dev.omninode-bridge.deployment.started.v1
   Emitted: Workflow initiation (PENDING → PACKAGING)

2. DEPLOYMENT_STAGE_COMPLETED
   Topic: dev.omninode-bridge.deployment.stage-completed.v1
   Emitted: After each stage completion (4 events per successful deployment)

3. DEPLOYMENT_HEALTH_CHECK
   Topic: dev.omninode-bridge.deployment.health-check.v1
   Emitted: During health validation (multiple events per validation)

4. DEPLOYMENT_COMPLETED
   Topic: dev.omninode-bridge.deployment.completed.v1
   Emitted: Successful deployment (VALIDATING → COMPLETED)

5. DEPLOYMENT_FAILED
   Topic: dev.omninode-bridge.deployment.failed.v1
   Emitted: Deployment failure (any state → FAILED)

6. DEPLOYMENT_ROLLBACK_INITIATED
   Topic: dev.omninode-bridge.deployment.rollback-initiated.v1
   Emitted: Rollback started (DEPLOYING/VALIDATING → ROLLING_BACK)

7. DEPLOYMENT_ROLLBACK_COMPLETED
   Topic: dev.omninode-bridge.deployment.rollback-completed.v1
   Emitted: Rollback finished (ROLLING_BACK → ROLLBACK_COMPLETED)
```

---

## Quick Start

### Basic Deployment Example

```python
from uuid import uuid4
from omninode_bridge.nodes.orchestrator.v1_0_0.node import NodeBridgeOrchestrator
from omninode_bridge.nodes.deployment_sender_effect.v1_0_0.node import (
    NodeDeploymentSenderEffect,
)
from omnibase_core.models.contracts.model_contract_orchestrator import (
    ModelContractOrchestrator,
)
from omnibase_core.models.container import ModelONEXContainer

async def deploy_container_to_remote():
    """Deploy orchestrator container to remote system."""

    # Initialize ONEX container with dependencies
    container = ModelONEXContainer(
        postgresql_client=postgresql_service,
        kafka_producer=kafka_service,
        config=config_service,
    )

    # Initialize orchestrator
    orchestrator = NodeBridgeOrchestrator(container)

    # Create deployment contract
    contract = ModelContractOrchestrator(
        correlation_id=uuid4(),
        workflow_name="deployment_workflow",
        input_state={
            "container_name": "omninode-orchestrator",
            "image_tag": "latest",
            "remote_host": "192.168.86.200",
            "remote_port": 8001,
            "deployment_config": {
                "environment_variables": {
                    "POSTGRES_HOST": "192.168.86.200",
                    "KAFKA_BOOTSTRAP_SERVERS": "192.168.86.200:29092",
                },
                "port_mappings": [
                    {"host_port": 8060, "container_port": 8060}
                ],
                "restart_policy": "unless-stopped",
                "network_mode": "bridge",
            },
            "build_options": {
                "dockerfile_path": "Dockerfile",
                "build_context": ".",
                "no_cache": False,
            },
            "health_check_config": {
                "health_endpoint": "/health",
                "expected_status_code": 200,
                "max_retries": 5,
                "retry_interval_ms": 2000,
            },
            "rollback_on_failure": True,
        },
    )

    # Execute deployment workflow
    result = await orchestrator.execute_orchestration(contract)

    # Check result
    if result.success:
        print(f"✅ Deployment successful!")
        print(f"   Container ID: {result.container_id}")
        print(f"   Deployed URL: {result.deployed_url}")
        print(f"   Duration: {result.total_duration_ms}ms")
        print(f"   Stages completed: {result.stages_completed}")
    else:
        print(f"❌ Deployment failed: {result.error_message}")
        if result.rollback_performed:
            print(f"   Rollback: {'Success' if result.rollback_result.success else 'Failed'}")

    return result
```

### Minimal Example (Default Configuration)

```python
async def quick_deploy():
    """Quick deployment with default settings."""

    orchestrator = NodeBridgeOrchestrator(container)

    result = await orchestrator.execute_orchestration(
        ModelContractOrchestrator(
            correlation_id=uuid4(),
            workflow_name="deployment_workflow",
            input_state={
                "container_name": "orchestrator",
                "remote_host": "192.168.86.200",
                # All other settings use defaults
            },
        )
    )

    return result
```

---

## Integration Patterns

### Pattern 1: Orchestrator Workflow Registration

**Registering deployment workflow with orchestrator:**

```python
from omninode_bridge.nodes.orchestrator.v1_0_0.node import NodeBridgeOrchestrator
from omninode_bridge.workflows.deployment_workflow import DeploymentWorkflow

class EnhancedOrchestrator(NodeBridgeOrchestrator):
    """Orchestrator with deployment workflow support."""

    def __init__(self, container: ModelONEXContainer):
        super().__init__(container)

        # Register deployment workflow
        self.register_workflow(
            workflow_name="deployment_workflow",
            workflow_class=DeploymentWorkflow,
            workflow_contract_path="contracts/workflows/deployment_workflow.yaml",
        )

    async def trigger_deployment(
        self,
        container_name: str,
        remote_host: str,
        **kwargs,
    ) -> dict:
        """Trigger deployment workflow."""

        workflow_input = {
            "container_name": container_name,
            "remote_host": remote_host,
            "image_tag": kwargs.get("image_tag", "latest"),
            "remote_port": kwargs.get("remote_port", 8001),
            **kwargs,
        }

        contract = ModelContractOrchestrator(
            correlation_id=uuid4(),
            workflow_name="deployment_workflow",
            input_state=workflow_input,
        )

        result = await self.execute_orchestration(contract)
        return result.to_dict()
```

### Pattern 2: Monitoring Deployment via Kafka Events

**Subscribe to deployment events for real-time monitoring:**

```python
from aiokafka import AIOKafkaConsumer
import asyncio

async def monitor_deployment(workflow_id: str):
    """Monitor deployment workflow via Kafka events."""

    consumer = AIOKafkaConsumer(
        "dev.omninode-bridge.deployment.started.v1",
        "dev.omninode-bridge.deployment.stage-completed.v1",
        "dev.omninode-bridge.deployment.health-check.v1",
        "dev.omninode-bridge.deployment.completed.v1",
        "dev.omninode-bridge.deployment.failed.v1",
        bootstrap_servers="localhost:29092",
        group_id=f"deployment-monitor-{workflow_id}",
    )

    await consumer.start()

    try:
        async for message in consumer:
            event = json.loads(message.value)

            # Filter by workflow ID
            if event["envelope"]["correlation_id"] != workflow_id:
                continue

            event_type = event["envelope"]["event_type"]
            payload = event["payload"]

            if event_type == "DEPLOYMENT_STARTED":
                print(f"🚀 Deployment started: {payload['container_name']}")

            elif event_type == "DEPLOYMENT_STAGE_COMPLETED":
                print(f"✅ Stage completed: {payload['stage_name']}")
                print(f"   Duration: {payload['duration_ms']}ms")

            elif event_type == "DEPLOYMENT_HEALTH_CHECK":
                print(f"❤️  Health check: {payload['status']}")

            elif event_type == "DEPLOYMENT_COMPLETED":
                print(f"🎉 Deployment completed successfully!")
                print(f"   Total duration: {payload['total_duration_ms']}ms")
                break

            elif event_type == "DEPLOYMENT_FAILED":
                print(f"❌ Deployment failed: {payload['error_message']}")
                break

    finally:
        await consumer.stop()
```

### Pattern 3: Querying Deployment State from PostgreSQL

**Query deployment state and history:**

```python
async def query_deployment_state(workflow_id: str) -> dict:
    """Query deployment state from PostgreSQL."""

    async with postgresql_client.connection() as conn:
        # Query workflow state
        workflow_state = await conn.fetchrow(
            """
            SELECT workflow_id, current_state, stages_completed, stages_failed,
                   total_duration_ms, success, error_message, created_at
            FROM workflow_executions
            WHERE workflow_id = $1
            """,
            workflow_id,
        )

        # Query stage results
        stage_results = await conn.fetch(
            """
            SELECT stage_name, success, duration_ms, operations, error_message
            FROM workflow_stage_results
            WHERE workflow_id = $1
            ORDER BY stage_number
            """,
            workflow_id,
        )

        return {
            "workflow_state": dict(workflow_state),
            "stage_results": [dict(r) for r in stage_results],
        }
```

### Pattern 4: Programmatic Deployment Trigger via API

**Expose deployment workflow via HTTP API:**

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/deployments", tags=["deployments"])

class DeploymentRequest(BaseModel):
    container_name: str
    remote_host: str
    image_tag: str = "latest"
    remote_port: int = 8001
    deployment_config: dict = {}
    build_options: dict = {}
    health_check_config: dict = {}

@router.post("/trigger")
async def trigger_deployment(request: DeploymentRequest):
    """Trigger deployment workflow via API."""

    try:
        orchestrator = get_orchestrator()  # Retrieve from DI container

        result = await orchestrator.trigger_deployment(
            container_name=request.container_name,
            remote_host=request.remote_host,
            image_tag=request.image_tag,
            remote_port=request.remote_port,
            deployment_config=request.deployment_config,
            build_options=request.build_options,
            health_check_config=request.health_check_config,
        )

        return {
            "status": "success" if result["success"] else "failed",
            "workflow_id": str(result["workflow_id"]),
            "deployment_id": str(result["deployment_id"]),
            "deployed_url": result.get("deployed_url"),
            "duration_ms": result["total_duration_ms"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{workflow_id}")
async def get_deployment_status(workflow_id: str):
    """Get deployment workflow status."""
    state = await query_deployment_state(workflow_id)
    return state
```

---

## Configuration

### Environment Variables

**Required for Sender (Local System):**
```bash
# Docker daemon connection
DOCKER_HOST=unix:///var/run/docker.sock

# Remote hook receiver
REMOTE_HOST=192.168.86.200
REMOTE_PORT=8001
HMAC_SECRET_KEY=your-secret-key-here  # Generate with: openssl rand -hex 32

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS=localhost:29092
KAFKA_ENABLE_LOGGING=true

# Deployment settings
MAX_CONCURRENT_DEPLOYMENTS=5
DEPLOYMENT_TIMEOUT_MS=180000  # 3 minutes
```

**Required for Receiver (Remote System):**
```bash
# Hook receiver API
HOOK_RECEIVER_PORT=8001
HOOK_RECEIVER_HOST=0.0.0.0

# Security
HMAC_SECRET_KEY=your-secret-key-here  # Must match sender
IP_WHITELIST=192.168.86.0/24,10.0.0.0/8  # Comma-separated CIDR blocks

# Docker daemon connection
DOCKER_HOST=unix:///var/run/docker.sock
DOCKER_API_VERSION=1.43

# Health check settings
HEALTH_CHECK_MAX_RETRIES=5
HEALTH_CHECK_RETRY_INTERVAL_MS=2000
HEALTH_CHECK_TIMEOUT_MS=5000

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS=192.168.86.200:29092
KAFKA_ENABLE_LOGGING=true
```

### Security Configuration

**HMAC Authentication Setup:**

```bash
# Generate shared secret
openssl rand -hex 32 > /etc/omninode/hmac_secret.key

# Set environment variable on both systems
export HMAC_SECRET_KEY=$(cat /etc/omninode/hmac_secret.key)
```

**IP Whitelisting:**

```yaml
# config/deployment_security.yaml
security:
  ip_whitelist:
    - 192.168.86.0/24   # Local network
    - 10.0.0.0/8        # Private network
    - 172.16.0.0/12     # Docker networks

  hmac_authentication:
    enabled: true
    secret_key_env: HMAC_SECRET_KEY
    signature_header: X-Deployment-Signature
    timestamp_tolerance_seconds: 300
```

### Docker Daemon Requirements

**Sender Side:**
- Docker Engine >= 20.10.0
- BuildKit support (optional, improves build speed)
- Sufficient disk space for image building (>10GB recommended)

**Receiver Side:**
- Docker Engine >= 20.10.0
- API access enabled (unix socket or TCP)
- Sufficient disk space for images (>20GB recommended)

**Verify Docker Access:**
```bash
# Test Docker daemon access
docker info

# Expected output: Server Version: 20.10.x or higher
```

### Network Requirements

**Firewall Rules:**
```bash
# On remote system, allow Hook Receiver port
sudo ufw allow 8001/tcp comment "OmniNode Hook Receiver"

# Optional: Allow from specific IPs only
sudo ufw allow from 192.168.86.0/24 to any port 8001 proto tcp
```

**DNS/Hostname Resolution:**
```bash
# Add remote host to /etc/hosts if needed
echo "192.168.86.200 omninode-remote" | sudo tee -a /etc/hosts
```

---

## Workflow Stages

### Stage 1: Package Preparation (Target: 10s)

**Purpose**: Build Docker image and gather deployment metadata.

**Operations**:

1. **build_docker_image** (Timeout: 45s)
   ```python
   # Build Docker image with BuildKit
   docker build -t omninode-orchestrator:latest \
       --build-arg BUILDKIT_INLINE_CACHE=1 \
       --cache-from omninode-orchestrator:latest \
       -f Dockerfile .
   ```

2. **gather_metadata** (Timeout: 2s)
   ```python
   # Inspect image and gather metadata
   metadata = {
       "image_id": image.id,
       "image_size_bytes": image.attrs['Size'],
       "creation_time": image.attrs['Created'],
       "layers_count": len(image.attrs['RootFS']['Layers']),
       "config": image.attrs['Config'],
   }
   ```

3. **validate_image** (Timeout: 3s)
   ```python
   # Validate image integrity and requirements
   - Check image exists
   - Verify image size < 2GB (configurable)
   - Validate required labels present
   - Check for security vulnerabilities (optional)
   ```

**Output Artifacts**:
- `docker_image`: Built image reference
- `deployment_metadata.json`: Container metadata
- `image_manifest.json`: Image layers and configuration

**Quality Gates**:
- ✅ Image built successfully
- ✅ Image size within limits (<2GB)
- ✅ All required metadata collected

---

### Stage 2: Transfer Initiation (Target: 5s)

**Purpose**: Transfer container package to remote hook receiver.

**Operations**:

1. **establish_connection** (Timeout: 5s)
   ```python
   # Connect to remote hook receiver
   async with aiohttp.ClientSession() as session:
       async with session.get(
           f"http://{remote_host}:{remote_port}/health",
           timeout=aiohttp.ClientTimeout(total=5),
       ) as response:
           assert response.status == 200
   ```

2. **send_deployment_request** (Timeout: 15s)
   ```python
   # Send deployment request with HMAC authentication
   import hmac
   import hashlib
   import time

   # Prepare payload
   payload = {
       "container_name": "omninode-orchestrator",
       "image_tag": "latest",
       "deployment_config": {...},
   }

   # Generate HMAC signature
   timestamp = str(int(time.time()))
   message = f"{timestamp}.{json.dumps(payload)}"
   signature = hmac.new(
       hmac_secret.encode(),
       message.encode(),
       hashlib.sha256,
   ).hexdigest()

   # Send request
   headers = {
       "X-Deployment-Signature": signature,
       "X-Deployment-Timestamp": timestamp,
   }

   async with session.post(
       f"http://{remote_host}:{remote_port}/api/deploy",
       json=payload,
       headers=headers,
   ) as response:
       result = await response.json()
       deployment_id = result["deployment_id"]
   ```

3. **verify_transfer** (Timeout: 5s)
   ```python
   # Confirm transfer success and remote readiness
   async with session.get(
       f"http://{remote_host}:{remote_port}/api/deploy/{deployment_id}/status"
   ) as response:
       status = await response.json()
       assert status["state"] == "ready_for_deployment"
   ```

**Output Artifacts**:
- `transfer_receipt.json`: Transfer confirmation
- `remote_deployment_id`: Unique deployment identifier on remote

**Quality Gates**:
- ✅ Remote endpoint reachable
- ✅ Transfer acknowledged by receiver
- ✅ HMAC authentication successful

---

### Stage 3: Deployment Execution (Target: 10s)

**Purpose**: Deploy and start container on remote system.

**Operations** (Executed on remote via NodeDeploymentReceiverEffect):

1. **stop_existing_container** (Timeout: 10s, allow_failure: true)
   ```python
   # Stop existing container if present (graceful + force)
   try:
       container = docker_client.containers.get(container_name)
       container.stop(timeout=10)
       container.remove(force=True)
   except docker.errors.NotFound:
       pass  # No existing container, continue
   ```

2. **deploy_container** (Timeout: 30s)
   ```python
   # Load image from package and deploy
   with open(image_package_path, 'rb') as f:
       docker_client.images.load(f)

   # Verify image loaded
   image = docker_client.images.get(f"{container_name}:{image_tag}")
   ```

3. **start_container** (Timeout: 15s)
   ```python
   # Start container with configuration
   container = docker_client.containers.run(
       image=f"{container_name}:{image_tag}",
       name=container_name,
       detach=True,
       environment=deployment_config["environment_variables"],
       ports=deployment_config["port_mappings"],
       volumes=deployment_config.get("volumes", []),
       network_mode=deployment_config.get("network_mode", "bridge"),
       restart_policy={"Name": deployment_config.get("restart_policy", "unless-stopped")},
   )

   container_id = container.id
   ```

**Output Artifacts**:
- `container_id`: Docker container ID
- `deployment_logs.txt`: Container startup logs

**Quality Gates**:
- ✅ Container deployed successfully
- ✅ Container started without errors
- ✅ No port conflicts detected

---

### Stage 4: Health Validation (Target: 5s)

**Purpose**: Verify deployment success and service health.

**Operations**:

1. **check_container_status** (Timeout: 3s)
   ```python
   # Verify container is running
   container = docker_client.containers.get(container_id)
   assert container.status == "running"

   # Check for restart loops
   restart_count = container.attrs['RestartCount']
   assert restart_count == 0, f"Container restarting (count: {restart_count})"
   ```

2. **validate_health_endpoint** (Timeout: 5s, retries: 5)
   ```python
   # Check service health endpoint with retries
   max_retries = 5
   retry_interval = 2  # seconds

   for attempt in range(max_retries):
       try:
           async with session.get(
               f"http://{remote_host}:{container_port}/health",
               timeout=aiohttp.ClientTimeout(total=3),
           ) as response:
               if response.status == 200:
                   health_data = await response.json()
                   return health_data

       except (aiohttp.ClientError, asyncio.TimeoutError) as e:
           if attempt < max_retries - 1:
               await asyncio.sleep(retry_interval)
               continue
           raise

   raise Exception(f"Health check failed after {max_retries} attempts")
   ```

3. **verify_service_registration** (Timeout: 3s, allow_failure: true)
   ```python
   # Confirm service registered with Consul (optional)
   if consul_enabled:
       services = consul_client.catalog.service(container_name)
       assert len(services) > 0, "Service not registered with Consul"
   ```

**Output Artifacts**:
- `health_check_results.json`: Health check responses
- `deployment_report.json`: Comprehensive deployment summary

**Quality Gates**:
- ✅ Container running without restarts
- ✅ Health endpoint responding with 200 OK
- ✅ Service registered (if Consul enabled)

---

## Error Handling

### Retry Policies

**Default Retry Configuration**:
```yaml
error_handling:
  retry_policy:
    max_attempts: 3
    backoff_multiplier: 2.0
    initial_backoff_ms: 1000
    max_backoff_ms: 30000
    timeout_ms: 180000  # 3 minutes total

  # Operation-specific overrides
  operation_retries:
    build_docker_image:
      max_attempts: 2
      timeout_ms: 60000

    establish_connection:
      max_attempts: 5
      backoff_multiplier: 1.5

    validate_health_endpoint:
      max_attempts: 5
      backoff_multiplier: 1.2
```

**Implementing Custom Retry Logic**:
```python
import asyncio
from typing import Callable, Any

async def retry_with_backoff(
    operation: Callable,
    max_attempts: int = 3,
    backoff_multiplier: float = 2.0,
    initial_backoff_ms: int = 1000,
) -> Any:
    """Execute operation with exponential backoff retry."""

    for attempt in range(max_attempts):
        try:
            return await operation()

        except Exception as e:
            if attempt >= max_attempts - 1:
                raise

            backoff_ms = initial_backoff_ms * (backoff_multiplier ** attempt)
            print(f"⚠️  Attempt {attempt + 1} failed: {e}")
            print(f"   Retrying in {backoff_ms}ms...")
            await asyncio.sleep(backoff_ms / 1000)
```

### Rollback Procedures

**Automatic Rollback Trigger Conditions**:
- Deployment stage fails
- Health validation fails
- Container crashes immediately after start
- Timeout exceeded in any stage

**Rollback Implementation**:
```python
async def perform_rollback(
    deployment_id: str,
    previous_container_id: str | None,
) -> dict:
    """Rollback failed deployment and restore previous container."""

    rollback_result = {
        "success": False,
        "previous_container_restored": False,
        "rollback_duration_ms": 0,
    }

    start_time = time.time()

    try:
        # 1. Stop failed container
        try:
            failed_container = docker_client.containers.get(container_name)
            failed_container.stop(timeout=5)
            failed_container.remove(force=True)
        except docker.errors.NotFound:
            pass

        # 2. Restore previous container (if exists)
        if previous_container_id:
            try:
                # Restart previous container
                previous_container = docker_client.containers.get(previous_container_id)
                previous_container.start()

                # Verify it started
                await asyncio.sleep(2)
                previous_container.reload()

                if previous_container.status == "running":
                    rollback_result["previous_container_restored"] = True

            except docker.errors.NotFound:
                print("⚠️  Previous container not found, cannot restore")

        rollback_result["success"] = True
        rollback_result["rollback_duration_ms"] = int((time.time() - start_time) * 1000)

    except Exception as e:
        print(f"❌ Rollback failed: {e}")
        rollback_result["error_message"] = str(e)

    return rollback_result
```

### Error Codes and Actions

| Error Code | Description | Severity | Action | Retry Allowed |
|------------|-------------|----------|--------|---------------|
| `BUILD_FAILED` | Docker image build failed | error | retry_build | ✅ Yes |
| `TRANSFER_FAILED` | Container transfer to remote failed | error | retry_transfer | ✅ Yes |
| `DEPLOYMENT_FAILED` | Container deployment failed | error | initiate_rollback | ✅ Yes |
| `HEALTH_CHECK_FAILED` | Health check validation failed | error | initiate_rollback | ✅ Yes |
| `REMOTE_ENDPOINT_UNREACHABLE` | Cannot connect to remote hook receiver | error | abort_workflow | ❌ No |
| `ROLLBACK_FAILED` | Rollback operation failed | critical | manual_intervention_required | ❌ No |
| `TIMEOUT` | Stage execution timeout | error | retry_stage | ✅ Yes |

### DLQ Monitoring

**Dead Letter Queue (DLQ) Monitoring for Failed Deployments**:

```python
from aiokafka import AIOKafkaConsumer

async def monitor_deployment_dlq():
    """Monitor deployment failures in DLQ."""

    consumer = AIOKafkaConsumer(
        "dev.omninode-bridge.deployment.dlq.v1",
        bootstrap_servers="localhost:29092",
        group_id="deployment-dlq-monitor",
    )

    await consumer.start()

    try:
        async for message in consumer:
            event = json.loads(message.value)

            deployment_id = event["envelope"]["correlation_id"]
            error_code = event["payload"]["error_code"]
            error_message = event["payload"]["error_message"]
            retry_count = event["payload"]["retry_count"]

            print(f"🚨 DLQ Event: Deployment {deployment_id}")
            print(f"   Error: {error_code} - {error_message}")
            print(f"   Retry count: {retry_count}")

            # Alert on critical failures
            if error_code in ["ROLLBACK_FAILED", "REMOTE_ENDPOINT_UNREACHABLE"]:
                await send_alert(
                    severity="critical",
                    message=f"Deployment {deployment_id} requires manual intervention",
                    details=event,
                )

    finally:
        await consumer.stop()
```

### Manual Intervention Scenarios

**Scenario 1: Rollback Failed**
```bash
# Manual steps when rollback fails:

# 1. SSH to remote system
ssh user@192.168.86.200

# 2. Check container status
docker ps -a | grep omninode-orchestrator

# 3. Manually stop failed container
docker stop omninode-orchestrator
docker rm omninode-orchestrator

# 4. Restore previous image (if available)
docker images | grep omninode-orchestrator
docker run -d --name omninode-orchestrator \
    --restart unless-stopped \
    -p 8060:8060 \
    omninode-orchestrator:previous

# 5. Verify health
curl http://localhost:8060/health
```

**Scenario 2: Remote Endpoint Unreachable**
```bash
# Troubleshooting steps:

# 1. Check network connectivity
ping 192.168.86.200

# 2. Check hook receiver status
ssh user@192.168.86.200 "docker ps | grep hook-receiver"

# 3. Check firewall rules
ssh user@192.168.86.200 "sudo ufw status | grep 8001"

# 4. Check hook receiver logs
ssh user@192.168.86.200 "docker logs hook-receiver --tail 100"

# 5. Restart hook receiver if needed
ssh user@192.168.86.200 "docker restart hook-receiver"
```

---

## Monitoring

### Kafka Topics to Monitor

**Primary Topics**:
1. `dev.omninode-bridge.deployment.started.v1` - Workflow initiations
2. `dev.omninode-bridge.deployment.stage-completed.v1` - Stage completions
3. `dev.omninode-bridge.deployment.completed.v1` - Successful deployments
4. `dev.omninode-bridge.deployment.failed.v1` - Failed deployments
5. `dev.omninode-bridge.deployment.health-check.v1` - Health check results
6. `dev.omninode-bridge.deployment.rollback-initiated.v1` - Rollback events
7. `dev.omninode-bridge.deployment.rollback-completed.v1` - Rollback completions

**Monitoring Commands**:
```bash
# Monitor all deployment events in real-time
kafka-console-consumer \
    --bootstrap-server localhost:29092 \
    --topic "dev.omninode-bridge.deployment.*" \
    --from-beginning

# Monitor failures only
kafka-console-consumer \
    --bootstrap-server localhost:29092 \
    --topic dev.omninode-bridge.deployment.failed.v1 \
    --from-beginning
```

### Prometheus Metrics

**Key Metrics Exported**:

```python
# Counter metrics
deployments_started_total{container_name, remote_host}
deployments_completed_total{container_name, remote_host}
deployments_failed_total{container_name, remote_host, failure_stage}
deployments_rolled_back_total{container_name, remote_host}

# Histogram metrics
deployment_duration_seconds{container_name}
  # Buckets: [15.0, 30.0, 60.0, 120.0, 180.0]

stage_duration_seconds{stage_name, container_name}
  # Buckets: [5.0, 10.0, 15.0, 30.0, 60.0]

image_size_bytes{container_name}
  # Buckets: [10MB, 100MB, 500MB, 1GB, 2GB]

# Gauge metrics
health_check_success_rate{container_name, remote_host}
current_deployment_state{workflow_id, container_name, state}
concurrent_deployments
```

**Example Prometheus Queries**:
```promql
# Deployment success rate (last 24h)
sum(rate(deployments_completed_total[24h])) /
sum(rate(deployments_started_total[24h]))

# Average deployment duration by container
avg(deployment_duration_seconds{container_name="orchestrator"})

# P95 deployment duration
histogram_quantile(0.95, deployment_duration_seconds_bucket)

# Failed deployments by stage
sum by (failure_stage) (deployments_failed_total)
```

### Log Correlation IDs

**Tracking Deployments Across Services**:

```python
import logging
from uuid import uuid4

# Generate correlation ID at workflow start
correlation_id = str(uuid4())

# Include in all log entries
logger = logging.getLogger(__name__)
logger.info(
    "Deployment started",
    extra={
        "correlation_id": correlation_id,
        "container_name": "orchestrator",
        "remote_host": "192.168.86.200",
    },
)

# Query logs by correlation ID
# Example: Splunk query
# index=omninode correlation_id="550e8400-e29b-41d4-a716-446655440000"

# Example: Elasticsearch query
# GET /omninode-logs/_search
# {
#   "query": {
#     "term": { "correlation_id": "550e8400-e29b-41d4-a716-446655440000" }
#   }
# }
```

### Health Check Endpoints

**Deployment System Health Endpoints**:

```bash
# Orchestrator health
curl http://localhost:8060/health
# Expected: {"status": "healthy", "version": "1.0.0", "services": {...}}

# Hook Receiver health (remote)
curl http://192.168.86.200:8001/health
# Expected: {"status": "healthy", "docker_connected": true}

# Deployed container health
curl http://192.168.86.200:8060/health  # Orchestrator
curl http://192.168.86.200:8061/health  # Reducer
```

---

## Testing

### Unit Test Examples

**Test 1: Stage Execution**

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from omninode_bridge.workflows.deployment_workflow import DeploymentWorkflow

@pytest.mark.asyncio
async def test_package_preparation_stage():
    """Test package preparation stage execution."""

    # Mock dependencies
    mock_docker_client = MagicMock()
    mock_image = MagicMock()
    mock_image.id = "sha256:abc123"
    mock_image.attrs = {
        "Size": 500_000_000,  # 500MB
        "Created": "2025-10-25T12:00:00Z",
        "RootFS": {"Layers": ["layer1", "layer2", "layer3"]},
    }
    mock_docker_client.images.build.return_value = mock_image

    workflow = DeploymentWorkflow(docker_client=mock_docker_client)

    # Execute stage
    result = await workflow.execute_stage_package_preparation(
        container_name="test-container",
        image_tag="latest",
        build_options={},
    )

    # Assertions
    assert result["success"] is True
    assert result["image_id"] == "sha256:abc123"
    assert result["image_size_bytes"] == 500_000_000
    assert result["duration_ms"] < 10000  # Should complete in <10s

    # Verify Docker API calls
    mock_docker_client.images.build.assert_called_once()
```

**Test 2: Error Handling and Retry**

```python
@pytest.mark.asyncio
async def test_transfer_with_retry():
    """Test transfer stage with retry on failure."""

    # Mock HTTP client that fails twice then succeeds
    mock_session = AsyncMock()
    call_count = 0

    async def mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise aiohttp.ClientError("Connection failed")
        response = AsyncMock()
        response.status = 200
        response.json = AsyncMock(return_value={"deployment_id": "deploy-123"})
        return response

    mock_session.post = mock_post

    workflow = DeploymentWorkflow(http_session=mock_session)

    # Execute stage
    result = await workflow.execute_stage_transfer_initiation(
        remote_host="192.168.86.200",
        remote_port=8001,
        deployment_package={},
    )

    # Assertions
    assert result["success"] is True
    assert result["deployment_id"] == "deploy-123"
    assert call_count == 3  # Failed twice, succeeded on third attempt
```

### Integration Test Examples

**Test 3: End-to-End Deployment Workflow**

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_end_to_end_deployment():
    """Test complete deployment workflow end-to-end."""

    # Initialize orchestrator with real services
    orchestrator = NodeBridgeOrchestrator(container)

    # Prepare test container
    test_container = {
        "container_name": "test-orchestrator",
        "image_tag": "test",
        "remote_host": "192.168.86.200",
        "remote_port": 8001,
        "deployment_config": {
            "environment_variables": {"TEST_MODE": "true"},
            "port_mappings": [{"host_port": 9999, "container_port": 8060}],
            "restart_policy": "no",  # Don't restart test containers
        },
        "health_check_config": {
            "health_endpoint": "/health",
            "max_retries": 3,
        },
    }

    # Execute workflow
    contract = ModelContractOrchestrator(
        correlation_id=uuid4(),
        workflow_name="deployment_workflow",
        input_state=test_container,
    )

    result = await orchestrator.execute_orchestration(contract)

    # Assertions
    assert result.success is True
    assert result.container_id is not None
    assert len(result.stages_completed) == 4
    assert result.health_status["container_running"] is True
    assert result.health_status["health_endpoint_accessible"] is True

    # Cleanup: Stop test container
    cleanup_result = await cleanup_deployment(result.container_id)
    assert cleanup_result["success"] is True
```

**Test 4: Rollback on Health Check Failure**

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_rollback_on_health_check_failure():
    """Test automatic rollback when health check fails."""

    # Deploy container with intentionally failing health check
    test_container = {
        "container_name": "test-failing-container",
        "image_tag": "test",
        "remote_host": "192.168.86.200",
        "health_check_config": {
            "health_endpoint": "/nonexistent",  # Will fail
            "max_retries": 2,
        },
        "rollback_on_failure": True,
    }

    contract = ModelContractOrchestrator(
        correlation_id=uuid4(),
        workflow_name="deployment_workflow",
        input_state=test_container,
    )

    result = await orchestrator.execute_orchestration(contract)

    # Assertions
    assert result.success is False
    assert "health_validation" in result.stages_failed
    assert result.rollback_performed is True
    assert result.rollback_result["success"] is True
```

### Performance Tests

**Test 5: Deployment Duration Benchmark**

```python
@pytest.mark.performance
@pytest.mark.asyncio
async def test_deployment_performance_benchmark():
    """Benchmark deployment workflow performance."""

    import time

    # Small container (100MB)
    small_container = {
        "container_name": "small-test",
        "image_tag": "alpine",  # ~5MB
        "remote_host": "192.168.86.200",
    }

    start_time = time.time()
    result = await orchestrator.execute_orchestration(
        ModelContractOrchestrator(
            correlation_id=uuid4(),
            workflow_name="deployment_workflow",
            input_state=small_container,
        )
    )
    duration_ms = (time.time() - start_time) * 1000

    # Assertions (performance targets from contract)
    assert result.success is True
    assert duration_ms < 30000  # <30s for small containers
    assert result.deployment_metrics["build_duration_ms"] < 10000
    assert result.deployment_metrics["transfer_duration_ms"] < 5000
    assert result.deployment_metrics["deployment_duration_ms"] < 10000
    assert result.deployment_metrics["health_validation_duration_ms"] < 5000
```

### Security Validation Tests

**Test 6: HMAC Authentication**

```python
@pytest.mark.security
@pytest.mark.asyncio
async def test_hmac_authentication():
    """Test HMAC authentication for deployment requests."""

    # Test with invalid HMAC signature
    invalid_request = {
        "container_name": "test",
        "headers": {
            "X-Deployment-Signature": "invalid_signature",
            "X-Deployment-Timestamp": str(int(time.time())),
        },
    }

    with pytest.raises(Exception, match="Authentication failed"):
        await send_deployment_request(invalid_request)

    # Test with valid HMAC signature
    valid_request = generate_authenticated_request(
        container_name="test",
        hmac_secret=os.getenv("HMAC_SECRET_KEY"),
    )

    result = await send_deployment_request(valid_request)
    assert result["authenticated"] is True
```

---

## Practical Examples

### Example 1: Deploy Orchestrator to Remote

```python
from omninode_bridge.workflows.deployment_workflow import deploy_container

async def deploy_orchestrator_to_production():
    """Deploy orchestrator to production remote system."""

    result = await deploy_container(
        container_name="omninode-orchestrator",
        image_tag="v1.2.0",
        remote_host="192.168.86.200",
        remote_port=8001,
        deployment_config={
            "environment_variables": {
                "POSTGRES_HOST": "192.168.86.200",
                "POSTGRES_PORT": "5436",
                "POSTGRES_DATABASE": "omninode_bridge",
                "KAFKA_BOOTSTRAP_SERVERS": "192.168.86.200:29092",
                "CONSUL_HOST": "192.168.86.200",
                "LOG_LEVEL": "INFO",
            },
            "port_mappings": [
                {"host_port": 8060, "container_port": 8060}
            ],
            "volumes": [
                {
                    "host_path": "/data/orchestrator",
                    "container_path": "/app/data",
                }
            ],
            "restart_policy": "unless-stopped",
            "network_mode": "bridge",
        },
        build_options={
            "dockerfile_path": "Dockerfile",
            "build_context": ".",
            "build_args": {
                "VERSION": "1.2.0",
                "BUILD_DATE": "2025-10-25",
            },
            "no_cache": False,
        },
        health_check_config={
            "health_endpoint": "/health",
            "expected_status_code": 200,
            "max_retries": 5,
            "retry_interval_ms": 2000,
        },
        rollback_on_failure=True,
    )

    if result["success"]:
        print(f"✅ Orchestrator deployed successfully!")
        print(f"   URL: http://192.168.86.200:8060")
        print(f"   Container ID: {result['container_id']}")
        print(f"   Duration: {result['total_duration_ms']}ms")
    else:
        print(f"❌ Deployment failed: {result['error_message']}")

    return result
```

### Example 2: Deploy Reducer to Remote

```python
async def deploy_reducer_to_production():
    """Deploy reducer to production remote system."""

    result = await deploy_container(
        container_name="omninode-reducer",
        image_tag="v1.2.0",
        remote_host="192.168.86.200",
        deployment_config={
            "environment_variables": {
                "POSTGRES_HOST": "192.168.86.200",
                "KAFKA_BOOTSTRAP_SERVERS": "192.168.86.200:29092",
                "AGGREGATION_WINDOW_MS": "5000",
                "BATCH_SIZE": "100",
            },
            "port_mappings": [
                {"host_port": 8061, "container_port": 8061}
            ],
            "restart_policy": "unless-stopped",
        },
    )

    return result
```

### Example 3: Rolling Update with Zero Downtime

```python
async def rolling_update_orchestrator():
    """Perform rolling update with zero downtime."""

    print("🔄 Starting rolling update...")

    # Step 1: Deploy new version alongside old version
    new_container_name = "omninode-orchestrator-new"

    deploy_result = await deploy_container(
        container_name=new_container_name,
        image_tag="v1.3.0",
        remote_host="192.168.86.200",
        deployment_config={
            "port_mappings": [
                {"host_port": 8070, "container_port": 8060}  # Temporary port
            ],
            "restart_policy": "unless-stopped",
        },
    )

    if not deploy_result["success"]:
        print("❌ New version deployment failed, aborting rolling update")
        return deploy_result

    print("✅ New version deployed on port 8070")

    # Step 2: Wait for new version health checks
    await asyncio.sleep(10)

    # Step 3: Switch traffic (update load balancer/proxy)
    print("🔀 Switching traffic to new version...")
    await update_load_balancer(
        old_backend="192.168.86.200:8060",
        new_backend="192.168.86.200:8070",
    )

    # Step 4: Wait for traffic to drain from old version
    await asyncio.sleep(30)

    # Step 5: Stop old version
    print("🛑 Stopping old version...")
    await stop_container(
        container_name="omninode-orchestrator",
        remote_host="192.168.86.200",
    )

    # Step 6: Rename new container and update to standard port
    print("📝 Finalizing deployment...")
    await rename_and_update_port(
        container_name=new_container_name,
        new_name="omninode-orchestrator",
        new_port=8060,
    )

    print("✅ Rolling update completed successfully!")
    return {"success": True, "old_version": "v1.2.0", "new_version": "v1.3.0"}
```

### Example 4: Parallel Deployments (Multiple Containers)

```python
async def deploy_all_services():
    """Deploy multiple services in parallel."""

    services = [
        {
            "container_name": "omninode-orchestrator",
            "image_tag": "v1.2.0",
            "port": 8060,
        },
        {
            "container_name": "omninode-reducer",
            "image_tag": "v1.2.0",
            "port": 8061,
        },
        {
            "container_name": "omninode-metadata-stamping",
            "image_tag": "v1.2.0",
            "port": 8057,
        },
        {
            "container_name": "omninode-onextree",
            "image_tag": "v1.2.0",
            "port": 8058,
        },
    ]

    # Create deployment tasks
    deployment_tasks = []

    for service in services:
        task = deploy_container(
            container_name=service["container_name"],
            image_tag=service["image_tag"],
            remote_host="192.168.86.200",
            deployment_config={
                "port_mappings": [
                    {"host_port": service["port"], "container_port": service["port"]}
                ],
                "restart_policy": "unless-stopped",
            },
        )
        deployment_tasks.append(task)

    # Execute all deployments in parallel
    results = await asyncio.gather(*deployment_tasks, return_exceptions=True)

    # Analyze results
    successful = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
    failed = len(results) - successful

    print(f"✅ Successful deployments: {successful}/{len(services)}")
    print(f"❌ Failed deployments: {failed}/{len(services)}")

    # Report detailed results
    for service, result in zip(services, results):
        if isinstance(result, Exception):
            print(f"   ❌ {service['container_name']}: {str(result)}")
        elif result.get("success"):
            print(f"   ✅ {service['container_name']}: {result['deployed_url']}")
        else:
            print(f"   ❌ {service['container_name']}: {result.get('error_message')}")

    return results
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Docker Build Fails

**Symptoms**:
- Stage 1 (package_preparation) fails
- Error: "docker build failed"
- Event: `DEPLOYMENT_FAILED` with error_code `BUILD_FAILED`

**Diagnosis**:
```bash
# Check Docker daemon status
docker info

# Check disk space (builds require space)
df -h /var/lib/docker

# Check build logs
docker build --progress=plain -t test:latest .
```

**Solutions**:
1. **Insufficient disk space**:
   ```bash
   # Clean up unused images
   docker image prune -a --force

   # Remove build cache
   docker builder prune --force
   ```

2. **Dockerfile syntax errors**:
   ```bash
   # Validate Dockerfile
   docker build --dry-run -t test:latest .

   # Check for missing dependencies
   ```

3. **Network issues (pulling base images)**:
   ```bash
   # Use local registry or cache
   # Add to deployment config:
   build_options = {
       "pull": False,  # Don't pull base images
       "cache_from": ["myregistry.com/base:latest"],
   }
   ```

---

#### Issue 2: Remote Endpoint Unreachable

**Symptoms**:
- Stage 2 (transfer_initiation) fails
- Error: "Cannot connect to remote hook receiver"
- Event: `DEPLOYMENT_FAILED` with error_code `REMOTE_ENDPOINT_UNREACHABLE`

**Diagnosis**:
```bash
# Test network connectivity
ping 192.168.86.200

# Test hook receiver endpoint
curl http://192.168.86.200:8001/health

# Check firewall rules on remote
ssh user@192.168.86.200 "sudo ufw status | grep 8001"

# Check hook receiver logs
ssh user@192.168.86.200 "docker logs hook-receiver --tail 50"
```

**Solutions**:
1. **Firewall blocking connection**:
   ```bash
   # On remote system
   sudo ufw allow 8001/tcp comment "OmniNode Hook Receiver"
   sudo ufw reload
   ```

2. **Hook receiver not running**:
   ```bash
   # Start hook receiver
   ssh user@192.168.86.200 "docker start hook-receiver"

   # Or deploy hook receiver first
   ```

3. **Wrong port or host**:
   ```python
   # Verify configuration
   deployment_config = {
       "remote_host": "192.168.86.200",  # Correct IP
       "remote_port": 8001,              # Correct port
   }
   ```

---

#### Issue 3: Health Check Failing

**Symptoms**:
- Stage 4 (health_validation) fails
- Container deployed but health check fails
- Event: `DEPLOYMENT_HEALTH_CHECK` with status "failed"

**Diagnosis**:
```bash
# Check container status
ssh user@192.168.86.200 "docker ps | grep orchestrator"

# Check container logs
ssh user@192.168.86.200 "docker logs orchestrator --tail 100"

# Test health endpoint manually
curl http://192.168.86.200:8060/health

# Check port mapping
ssh user@192.168.86.200 "docker port orchestrator"
```

**Solutions**:
1. **Container not fully started**:
   ```python
   # Increase retry interval
   health_check_config = {
       "max_retries": 10,  # More retries
       "retry_interval_ms": 5000,  # Wait longer between retries
   }
   ```

2. **Wrong health endpoint**:
   ```python
   # Verify health endpoint path
   health_check_config = {
       "health_endpoint": "/health",  # Correct path
       "expected_status_code": 200,
   }

   # Check container logs for actual endpoint
   ```

3. **Port mapping incorrect**:
   ```python
   # Verify port mappings
   deployment_config = {
       "port_mappings": [
           {"host_port": 8060, "container_port": 8060}  # Must match
       ],
   }
   ```

---

#### Issue 4: Container Keeps Restarting

**Symptoms**:
- Container deployed but immediately restarts
- Health check detects restart loop
- Container status shows "restarting"

**Diagnosis**:
```bash
# Check restart count
ssh user@192.168.86.200 "docker inspect orchestrator | grep RestartCount"

# Check container logs for errors
ssh user@192.168.86.200 "docker logs orchestrator --tail 200"

# Check container exit code
ssh user@192.168.86.200 "docker inspect orchestrator | grep ExitCode"
```

**Solutions**:
1. **Missing environment variables**:
   ```python
   # Ensure all required env vars are set
   deployment_config = {
       "environment_variables": {
           "POSTGRES_HOST": "192.168.86.200",  # Required
           "KAFKA_BOOTSTRAP_SERVERS": "...",    # Required
           # Add all required variables
       },
   }
   ```

2. **Database connection failed**:
   ```bash
   # Verify database is accessible from container
   ssh user@192.168.86.200 "docker exec orchestrator nc -zv postgres_host 5432"

   # Check database credentials
   ```

3. **Application crash on startup**:
   ```bash
   # Run container interactively to debug
   ssh user@192.168.86.200 "docker run -it --rm orchestrator:latest sh"

   # Check application logs for stack trace
   ```

---

#### Issue 5: HMAC Authentication Failed

**Symptoms**:
- Transfer request rejected by remote
- Error: "Authentication failed: Invalid signature"
- HTTP 401 Unauthorized response

**Diagnosis**:
```bash
# Check HMAC secret on both systems
echo $HMAC_SECRET_KEY  # Local
ssh user@192.168.86.200 "echo \$HMAC_SECRET_KEY"  # Remote

# Verify timestamp tolerance
# HMAC uses timestamp to prevent replay attacks
```

**Solutions**:
1. **Mismatched HMAC secrets**:
   ```bash
   # Regenerate and sync secrets
   SECRET=$(openssl rand -hex 32)

   # Set on local system
   export HMAC_SECRET_KEY="$SECRET"

   # Set on remote system
   ssh user@192.168.86.200 "echo 'export HMAC_SECRET_KEY=$SECRET' >> ~/.bashrc"

   # Restart hook receiver
   ssh user@192.168.86.200 "docker restart hook-receiver"
   ```

2. **Clock skew between systems**:
   ```bash
   # Check time on both systems
   date
   ssh user@192.168.86.200 "date"

   # Sync clocks (NTP)
   sudo ntpdate -u time.google.com
   ssh user@192.168.86.200 "sudo ntpdate -u time.google.com"
   ```

3. **Signature generation bug**:
   ```python
   # Verify signature generation code
   import hmac
   import hashlib
   import time

   secret = os.getenv("HMAC_SECRET_KEY")
   timestamp = str(int(time.time()))
   message = f"{timestamp}.{json.dumps(payload)}"
   signature = hmac.new(
       secret.encode(),
       message.encode(),
       hashlib.sha256,
   ).hexdigest()

   # Include in headers
   headers = {
       "X-Deployment-Signature": signature,
       "X-Deployment-Timestamp": timestamp,
   }
   ```

---

### Debug Checklist

**Pre-Deployment Checklist**:
- [ ] Docker daemon running on local system
- [ ] Docker daemon running on remote system
- [ ] Remote hook receiver is running and healthy
- [ ] HMAC secrets match on both systems
- [ ] Network connectivity between systems
- [ ] Firewall allows traffic on hook receiver port
- [ ] Sufficient disk space on both systems (>10GB)
- [ ] All required environment variables configured

**During Deployment**:
- [ ] Monitor Kafka events for stage progress
- [ ] Check logs for errors or warnings
- [ ] Verify each stage completes within timeout
- [ ] Monitor resource usage (CPU, memory, disk)

**Post-Deployment**:
- [ ] Container status is "running"
- [ ] Health endpoint returns 200 OK
- [ ] Service registered with Consul (if applicable)
- [ ] No restart loops detected
- [ ] Application logs show normal operation

---

### Log Analysis Tips

**Finding Deployment Logs**:

```bash
# Local orchestrator logs
docker logs omninode-orchestrator | grep deployment

# Remote receiver logs
ssh user@192.168.86.200 "docker logs hook-receiver" | grep deployment

# Filter by correlation ID
docker logs omninode-orchestrator | grep "550e8400-e29b-41d4-a716-446655440000"

# Filter by stage
docker logs omninode-orchestrator | grep "PACKAGING"
docker logs omninode-orchestrator | grep "TRANSFERRING"
```

**Analyzing Stage Performance**:

```bash
# Extract stage durations from logs
docker logs omninode-orchestrator | \
    grep "stage_completed" | \
    jq '.duration_ms'

# Find slowest stage
docker logs omninode-orchestrator | \
    grep "stage_completed" | \
    jq '{stage: .stage_name, duration: .duration_ms}' | \
    sort -k2 -rn | \
    head -1
```

**Identifying Error Patterns**:

```bash
# Count errors by type
docker logs omninode-orchestrator | \
    grep "ERROR" | \
    jq '.error_code' | \
    sort | \
    uniq -c

# Find most common failure stage
docker logs omninode-orchestrator | \
    grep "DEPLOYMENT_FAILED" | \
    jq '.payload.failure_stage' | \
    sort | \
    uniq -c
```

---

## References

### Internal Documentation
- **[Workflow Contract](../../contracts/workflows/deployment_workflow.yaml)** - Complete workflow specification
- **[Bridge Nodes Guide](./BRIDGE_NODES_GUIDE.md)** - ONEX bridge node patterns
- **[Remote Migration Guide](../deployment/REMOTE_MIGRATION_GUIDE.md)** - Manual deployment guide
- **[API Reference](../api/API_REFERENCE.md)** - API documentation

### External References
- **[ONEX Architecture Patterns](https://github.com/OmniNode-ai/Archon/blob/main/docs/ONEX_ARCHITECTURE_PATTERNS_COMPLETE.md)** - ONEX v2.0 patterns
- **[Docker SDK for Python](https://docker-py.readthedocs.io/)** - Docker client library
- **[aiokafka Documentation](https://aiokafka.readthedocs.io/)** - Async Kafka client

### Related Scripts
- **[migrate-to-remote.sh](../../migrate-to-remote.sh)** - Automated migration script
- **[rebuild-service.sh](../../rebuild-service.sh)** - Service rebuild script
- **[setup-remote.sh](../../setup-remote.sh)** - Remote system setup

---

**Document Version**: 1.0.0
**Maintained By**: OmniNode Bridge Team
**Last Review**: 2025-10-25
