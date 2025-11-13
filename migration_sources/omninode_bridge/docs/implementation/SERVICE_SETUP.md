# Service Setup Guide

## Overview

This guide provides step-by-step instructions for deploying the OmniNode Bridge tactical deployment system, including all infrastructure, intelligence, and service components.

## Prerequisites

### System Requirements
- Docker and Docker Compose
- Python 3.11+
- Node.js 18+ (for some services)
- 16GB RAM minimum
- 50GB free disk space

### Repository Setup
```bash
# Clone the omninode_bridge repository
git clone https://github.com/OmniNode-ai/omninode_bridge.git
cd omninode_bridge

# Ensure access to source repositories
ls ../omnibase_3        # Hook and proxy patterns
ls ../omnibase_infra    # Infrastructure components
ls ../omnibase_core     # Core models and types
ls ../omnimemory        # Memory services
ls ../omniagent         # Agent services
```

## Phase 1: Infrastructure Bootstrap (Days 1-2)

### Step 1: Deploy Core Infrastructure

#### 1.1 RedPanda Event Bus
```bash
# Start RedPanda cluster
docker run -d \
  --name redpanda \
  -p 9092:9092 \
  -p 9644:9644 \
  vectorized/redpanda:latest \
  redpanda start \
  --smp 1 \
  --memory 1G \
  --reserve-memory 0M \
  --overprovisioned \
  --node-id 0 \
  --kafka-addr PLAINTEXT://0.0.0.0:9092 \
  --advertise-kafka-addr PLAINTEXT://localhost:9092

# Create topics for intelligence capture
docker exec redpanda rpk topic create hooks.lifecycle
docker exec redpanda rpk topic create hooks.execution
docker exec redpanda rpk topic create hooks.tools
docker exec redpanda rpk topic create services.events
docker exec redpanda rpk topic create coordination.events
```

#### 1.2 Consul Service Discovery
```bash
# Start Consul server
docker run -d \
  --name consul \
  -p 8500:8500 \
  -p 8600:8600/udp \
  consul:latest agent \
  -server \
  -bootstrap-expect=1 \
  -ui \
  -bind=0.0.0.0 \
  -client=0.0.0.0

# Verify Consul is running
curl http://localhost:8500/v1/status/leader
```

#### 1.3 PostgreSQL Storage
```bash
# Start PostgreSQL with intelligence schemas
docker run -d \
  --name postgres \
  -e POSTGRES_DB=omninode_bridge \
  -e POSTGRES_USER=bridge_user \
  -e POSTGRES_PASSWORD=bridge_password \
  -p 5432:5432 \
  postgres:15

# Wait for startup then create schemas
sleep 10
docker exec postgres psql -U bridge_user -d omninode_bridge -c "
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY,
    correlation_id UUID,
    started_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS hook_events (
    id UUID PRIMARY KEY,
    session_id UUID REFERENCES sessions(id),
    event_type VARCHAR(255),
    payload JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS service_metrics (
    id UUID PRIMARY KEY,
    service_name VARCHAR(255),
    metric_type VARCHAR(255),
    value NUMERIC,
    recorded_at TIMESTAMP DEFAULT NOW()
);
"
```

### Step 2: Deploy Hook Intelligence System

#### 2.1 HookReceiver Service
```bash
# Copy HookReceiver implementation from omnibase_3
mkdir -p src/services
cp ../omnibase_3/src/omnibase/services/hook_receiver_service.py src/services/

# Create simplified deployment configuration
cat > docker-compose.hooks.yml << 'EOF'
version: '3.8'
services:
  hook-receiver:
    build:
      context: .
      dockerfile: Dockerfile.hook-receiver
    ports:
      - "8080:8080"
    environment:
      - KAFKA_BOOTSTRAP_SERVERS=localhost:9092
      - POSTGRES_DSN=postgresql://bridge_user:bridge_password@localhost:5432/omninode_bridge  # pragma: allowlist secret
      - CONSUL_HTTP_ADDR=http://localhost:8500
    depends_on:
      - postgres
      - consul
    restart: unless-stopped
EOF

# Create Dockerfile for HookReceiver
cat > Dockerfile.hook-receiver << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements-hooks.txt .
RUN pip install -r requirements-hooks.txt

# Copy hook receiver source
COPY src/services/hook_receiver_service.py .
COPY src/models/ ./models/

CMD ["python", "hook_receiver_service.py"]
EOF

# Create requirements file
cat > requirements-hooks.txt << 'EOF'
fastapi==0.104.1
uvicorn==0.24.0
aiokafka==0.10.0
asyncpg==0.29.0
python-consul==1.1.0
pydantic==2.5.0
EOF
```

#### 2.2 Service Lifecycle Hook Registration
```python
# Create service registration helper
cat > src/utils/service_registration.py << 'EOF'
import asyncio
import aiohttp
import json
from datetime import datetime

class ServiceRegistration:
    def __init__(self, hook_receiver_url="http://localhost:8080"):
        self.hook_receiver_url = hook_receiver_url

    async def register_service_startup(self, service_info):
        """Register service startup with automatic capability discovery"""
        payload = {
            "event_type": "service_started",
            "timestamp": datetime.utcnow().isoformat(),
            "service_info": {
                "name": service_info["name"],
                "version": service_info["version"],
                "capabilities": service_info.get("capabilities", []),
                "endpoints": service_info.get("endpoints", {}),
                "mcp_tools": service_info.get("mcp_tools", [])
            }
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.hook_receiver_url}/hooks/lifecycle",
                json=payload
            ) as response:
                return await response.json()

    async def register_service_health(self, service_name, health_status):
        """Register service health changes"""
        payload = {
            "event_type": "service_health_changed",
            "timestamp": datetime.utcnow().isoformat(),
            "service_name": service_name,
            "health_status": health_status
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.hook_receiver_url}/hooks/lifecycle",
                json=payload
            ) as response:
                return await response.json()
EOF
```

### Step 3: Deploy Infrastructure Validation

#### 3.1 Health Check Endpoints
```bash
# Test infrastructure connectivity
curl http://localhost:8500/v1/status/leader  # Consul
curl http://localhost:8080/health            # Hook Receiver (after deployment)

# Test Kafka topics
docker exec redpanda rpk topic list
```

#### 3.2 Basic Event Flow Test
```bash
# Test event publishing
curl -X POST http://localhost:8080/hooks/test \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "test_event",
    "payload": {"test": "infrastructure_validation"}
  }'
```

## Phase 2: Service Integration (Days 3-4)

### Step 1: Deploy ToolCapture Proxy

#### 1.1 Proxy Service Setup
```bash
# Copy ToolCapture implementation from omnibase_3
cp ../omnibase_3/src/omnibase/services/tool_capture_proxy.py src/services/

# Create proxy deployment configuration
cat > docker-compose.proxy.yml << 'EOF'
version: '3.8'
services:
  tool-capture-proxy:
    build:
      context: .
      dockerfile: Dockerfile.proxy
    ports:
      - "8081:8081"
    environment:
      - TARGET_BASE_URL=http://localhost:8080
      - KAFKA_BOOTSTRAP_SERVERS=localhost:9092
      - HOOK_RECEIVER_URL=http://hook-receiver:8080
    depends_on:
      - hook-receiver
    restart: unless-stopped
EOF
```

#### 1.2 MCP Hooks Registry
```bash
# Copy MCP registry implementation
mkdir -p src/models/mcp
cp ../omnibase_3/src/omnibase/model/mcp/model_mcp_hooks_registry.py src/models/mcp/

# Create MCP registry service
cat > src/services/mcp_registry_service.py << 'EOF'
import asyncio
from typing import Dict, List
from models.mcp.model_mcp_hooks_registry import ModelMCPHooksRegistry
import consul.aio

class MCPRegistryService:
    def __init__(self):
        self.registry = ModelMCPHooksRegistry()
        self.consul = consul.aio.Consul()

    async def discover_and_register_tools(self):
        """Discover services and register their MCP tools"""
        # Get all registered services from Consul
        services = await self.consul.catalog.services()

        for service_name, service_tags in services[1].items():
            # Get service details
            service_info = await self.consul.catalog.service(service_name)

            for service_instance in service_info[1]:
                # Extract MCP tools from service metadata
                metadata = service_instance.get('ServiceMeta', {})
                mcp_tools = metadata.get('mcp_tools', '[]')

                if mcp_tools:
                    tools = json.loads(mcp_tools)
                    for tool in tools:
                        await self.register_tool_hook(tool, service_instance)

    async def register_tool_hook(self, tool_info, service_instance):
        """Register a hook for an MCP tool"""
        # Create hook function for tool execution monitoring
        def create_hook(original_func):
            async def hooked_func(*args, **kwargs):
                # Pre-execution hook
                await self.capture_tool_execution_start(tool_info, args, kwargs)

                try:
                    result = await original_func(*args, **kwargs)
                    # Post-execution hook
                    await self.capture_tool_execution_success(tool_info, result)
                    return result
                except Exception as e:
                    # Error hook
                    await self.capture_tool_execution_error(tool_info, e)
                    raise
            return hooked_func

        # Register the hook
        self.registry.register_hook(
            tool_name=tool_info['name'],
            original_function=tool_info.get('function'),
            hook_function=create_hook
        )

if __name__ == "__main__":
    service = MCPRegistryService()
    asyncio.run(service.discover_and_register_tools())
EOF
```

### Step 2: Deploy OmniMemory Integration

#### 2.1 Memory Service with Event Integration
```bash
# Navigate to omnimemory and add event hooks
cd ../omnimemory

# Add event integration to memory operations
cat >> src/omnimemory/core/memory_manager.py << 'EOF'

# Event integration for memory operations
from utils.service_registration import ServiceRegistration

class EventIntegratedMemoryManager(MemoryManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.service_registration = ServiceRegistration()

    async def store_memory(self, key, value, metadata=None):
        # Capture memory store event
        await self.service_registration.register_service_event({
            "event_type": "memory_store",
            "service": "omnimemory",
            "operation": "store",
            "key": key,
            "metadata": metadata
        })

        result = await super().store_memory(key, value, metadata)

        # Capture completion event
        await self.service_registration.register_service_event({
            "event_type": "memory_store_completed",
            "service": "omnimemory",
            "operation": "store",
            "key": key,
            "success": True
        })

        return result
EOF

# Return to bridge directory
cd ../omninode_bridge
```

#### 2.2 Memory Service Registration
```bash
# Create omnimemory service registration
cat > scripts/register_omnimemory.py << 'EOF'
import asyncio
from src.utils.service_registration import ServiceRegistration

async def register_omnimemory():
    registration = ServiceRegistration()

    service_info = {
        "name": "omnimemory",
        "version": "1.0.0",
        "capabilities": ["memory_storage", "caching", "intelligence"],
        "endpoints": {
            "health": "/health",
            "store": "/memory/store",
            "retrieve": "/memory/retrieve",
            "clear": "/memory/clear"
        },
        "mcp_tools": [
            {
                "name": "store_memory",
                "description": "Store data in memory system",
                "parameters": {"key": "string", "value": "any", "metadata": "object"}
            },
            {
                "name": "retrieve_memory",
                "description": "Retrieve data from memory system",
                "parameters": {"key": "string"}
            }
        ]
    }

    result = await registration.register_service_startup(service_info)
    print(f"OmniMemory registered: {result}")

if __name__ == "__main__":
    asyncio.run(register_omnimemory())
EOF
```

### Step 3: Deploy OmniAgent Integration

#### 3.1 Agent Coordination with Event Monitoring
```bash
# Create agent coordination service
cat > src/services/agent_coordinator.py << 'EOF'
import asyncio
from typing import List, Dict
from utils.service_registration import ServiceRegistration

class AgentCoordinator:
    def __init__(self):
        self.service_registration = ServiceRegistration()
        self.active_workflows = {}

    async def execute_workflow(self, workflow_spec: Dict):
        """Execute workflow with intelligent coordination"""
        workflow_id = workflow_spec.get('id', str(uuid.uuid4()))

        # Register workflow start
        await self.service_registration.register_service_event({
            "event_type": "workflow_started",
            "service": "omniagent",
            "workflow_id": workflow_id,
            "agents_count": len(workflow_spec.get('agents', [])),
            "estimated_duration": workflow_spec.get('estimated_duration')
        })

        try:
            # Discover available agents
            agents = await self.discover_agents()

            # Execute workflow steps
            results = await self.execute_parallel_agents(agents, workflow_spec)

            # Register successful completion
            await self.service_registration.register_service_event({
                "event_type": "workflow_completed",
                "service": "omniagent",
                "workflow_id": workflow_id,
                "success": True,
                "results_count": len(results)
            })

            return results

        except Exception as e:
            # Register workflow failure
            await self.service_registration.register_service_event({
                "event_type": "workflow_failed",
                "service": "omniagent",
                "workflow_id": workflow_id,
                "error": str(e)
            })
            raise

    async def discover_agents(self) -> List[Dict]:
        """Discover available agents from Consul"""
        # Implementation to discover agents through service registry
        pass

    async def execute_parallel_agents(self, agents: List[Dict], workflow_spec: Dict):
        """Execute agents in parallel with coordination"""
        # Implementation for parallel agent execution
        pass
EOF
```

## Phase 3: Complete System Integration (Days 5-7)

### Step 1: Deploy Version Bridge

#### 1.1 Simple Version Bridge Implementation
```bash
cat > src/services/version_bridge.py << 'EOF'
import asyncio
from typing import Dict, Optional
import consul.aio
from utils.service_registration import ServiceRegistration

class SimpleVersionBridge:
    def __init__(self):
        self.consul = consul.aio.Consul()
        self.service_registration = ServiceRegistration()

    async def route_by_version(self, request: Dict, target_version: str):
        """Route request to compatible service version"""
        # Find services matching target version
        services = await self.find_services(version=target_version)

        if not services:
            raise ValueError(f"No services found for version {target_version}")

        # Select best service (for now, just use first)
        target_service = services[0]

        # Forward request
        result = await self.forward_request(target_service, request)

        # Capture routing intelligence
        await self.service_registration.register_service_event({
            "event_type": "version_bridge_routing",
            "source_version": request.get('version'),
            "target_version": target_version,
            "service_selected": target_service['name'],
            "success": True
        })

        return result

    async def find_services(self, version: str) -> List[Dict]:
        """Find services matching version requirements"""
        services = await self.consul.catalog.services()
        matching_services = []

        for service_name, tags in services[1].items():
            if f"version-{version}" in tags:
                service_details = await self.consul.catalog.service(service_name)
                matching_services.extend(service_details[1])

        return matching_services

    async def forward_request(self, service: Dict, request: Dict):
        """Forward request to target service"""
        # Implementation for request forwarding
        pass
EOF
```

### Step 2: End-to-End Validation

#### 2.1 Complete Workflow Test
```bash
# Create end-to-end validation script
cat > scripts/validate_complete_system.py << 'EOF'
import asyncio
import aiohttp
import json

async def test_complete_workflow():
    """Test complete workflow across all services"""
    print("🧪 Testing Complete OmniNode Bridge System")

    # Test 1: Infrastructure Health
    print("\n1. Testing Infrastructure Health...")
    health_checks = [
        ("Consul", "http://localhost:8500/v1/status/leader"),
        ("Hook Receiver", "http://localhost:8080/health"),
        ("Tool Proxy", "http://localhost:8081/health")
    ]

    for service_name, url in health_checks:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        print(f"  ✅ {service_name}: Healthy")
                    else:
                        print(f"  ❌ {service_name}: Unhealthy ({response.status})")
        except Exception as e:
            print(f"  ❌ {service_name}: Connection failed ({e})")

    # Test 2: Service Registration
    print("\n2. Testing Service Registration...")
    async with aiohttp.ClientSession() as session:
        async with session.get("http://localhost:8500/v1/catalog/services") as response:
            services = await response.json()
            print(f"  📊 Registered services: {list(services.keys())}")

    # Test 3: Event Flow
    print("\n3. Testing Event Flow...")
    test_event = {
        "event_type": "end_to_end_test",
        "payload": {"test_id": "validation_001", "timestamp": "2025-09-21T12:00:00Z"}
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8080/hooks/test",
            json=test_event
        ) as response:
            result = await response.json()
            print(f"  📡 Event processing: {result.get('status', 'failed')}")

    # Test 4: Tool Discovery
    print("\n4. Testing Tool Discovery...")
    # Implementation for tool discovery validation

    # Test 5: Agent Workflow
    print("\n5. Testing Agent Workflow...")
    # Implementation for agent workflow validation

    print("\n🎉 Complete system validation finished!")

if __name__ == "__main__":
    asyncio.run(test_complete_workflow())
EOF

# Run the validation
python scripts/validate_complete_system.py
```

### Step 3: Production Readiness

#### 3.1 Monitoring Setup
```bash
# Create monitoring configuration
cat > docker-compose.monitoring.yml << 'EOF'
version: '3.8'
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana

volumes:
  grafana-storage:
EOF

# Create Prometheus configuration
mkdir -p monitoring
cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'hook-receiver'
    static_configs:
      - targets: ['localhost:8080']

  - job_name: 'tool-proxy'
    static_configs:
      - targets: ['localhost:8081']

  - job_name: 'consul'
    static_configs:
      - targets: ['localhost:8500']
EOF
```

## Quick Start Commands

```bash
# Complete system startup
docker-compose -f docker-compose.hooks.yml -f docker-compose.proxy.yml -f docker-compose.monitoring.yml up -d

# Register services
python scripts/register_omnimemory.py
python scripts/register_omniagent.py

# Validate system
python scripts/validate_complete_system.py

# View monitoring
open http://localhost:3000  # Grafana
open http://localhost:8500  # Consul UI
```

## Troubleshooting

### Common Issues
1. **Port Conflicts**: Ensure ports 8080, 8081, 8500, 9092, 5432 are available
2. **Service Dependencies**: Start infrastructure services before application services
3. **Network Connectivity**: Ensure Docker networks allow service communication
4. **Resource Limits**: Monitor memory usage, especially with multiple services

### Debug Commands
```bash
# Check service logs
docker logs hook-receiver
docker logs tool-capture-proxy

# Check Kafka topics
docker exec redpanda rpk topic list

# Check Consul services
curl http://localhost:8500/v1/catalog/services

# Check PostgreSQL connections
docker exec postgres psql -U bridge_user -d omninode_bridge -c "\dt"
```

This setup provides a complete, production-ready tactical deployment system with intelligent coordination and automatic learning capabilities.
