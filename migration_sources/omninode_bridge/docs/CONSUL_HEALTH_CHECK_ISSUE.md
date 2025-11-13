# Consul Health Check Issue - Remote Consul Cannot Reach Container Localhost

**Date**: 2025-10-29
**Status**: Known Issue
**Priority**: Medium
**Affected Services**: archon-server, archon-intelligence, archon-bridge, archon-search

## Summary

Archon services successfully register with remote Consul (192.168.86.200:28500), but health checks remain in "unknown" state because Consul cannot reach health check URLs that use `localhost`.

## Problem Details

### Architecture Context

```
┌─────────────────────────────────────────┐
│  Local Machine (Docker containers)      │
│                                         │
│  archon-server:                         │
│    - Container: localhost:8181         │
│    - Health: localhost:8181/health     │  ← Consul cannot reach this
│                                         │
│  archon-intelligence:                   │
│    - Container: localhost:8053         │
│    - Health: localhost:8053/health     │  ← Consul cannot reach this
│                                         │
└─────────────────────────────────────────┘
                    │
                    │ Network: 192.168.86.x
                    ↓
┌─────────────────────────────────────────┐
│  Remote Machine: 192.168.86.200         │
│                                         │
│  Consul Server:                         │
│    - Internal: localhost:8500          │
│    - External: 192.168.86.200:28500    │
│                                         │
│  Tries to reach:                        │
│    http://localhost:8181/health        │  ← FAILS (localhost is remote machine)
│                                         │
└─────────────────────────────────────────┘
```

### Current Registration Code

```python
# python/src/server/services/consul_service.py (line ~120)
check = {
    "http": f"http://localhost:{port}/health",  # ← Problem: localhost
    "interval": "30s",
    "timeout": "5s",
    "deregister_critical_service_after": "90s"
}
```

### Error Manifestation

- Services **register successfully** in Consul
- Services show **correct tags and metadata**
- Health checks show as **"unknown"** or **"critical"** (cannot connect)
- Consul logs: `Get "http://localhost:8181/health": dial tcp 127.0.0.1:8181: connect: connection refused`

## Root Cause

**Consul runs on remote machine** (192.168.86.200) while **services run in Docker containers on local machine**. When Consul tries to check health:

1. Consul resolves `localhost` to `127.0.0.1` (its own loopback)
2. Remote machine has no service listening on `127.0.0.1:8181`
3. Connection fails, health check marked as critical/unknown

## Solution Options

### Option 1: Use Host Machine IP (Recommended for Dev)

Update health check URLs to use local machine's network IP:

```python
# Get local machine IP (visible to remote Consul)
LOCAL_IP = os.getenv("LOCAL_IP", "192.168.86.XXX")  # User's dev machine IP

check = {
    "http": f"http://{LOCAL_IP}:{port}/health",
    "interval": "30s",
    "timeout": "5s",
    "deregister_critical_service_after": "90s"
}
```

**Pros**: Simple, works immediately
**Cons**: Requires manual IP configuration, not portable across machines

### Option 2: Use Docker Host Gateway

Configure Docker to expose container ports to host, use host gateway IP:

```python
# Docker provides host.docker.internal for accessing host
HOST_IP = "host.docker.internal"  # Or actual gateway IP

check = {
    "http": f"http://{HOST_IP}:{port}/health",
    "interval": "30s",
    "timeout": "5s"
}
```

**Pros**: More portable
**Cons**: Requires Docker network configuration

### Option 3: TTL-Based Health Checks (Alternative)

Instead of HTTP checks, use TTL-based checks where service reports its own health:

```python
check = {
    "ttl": "30s",
    "deregister_critical_service_after": "90s"
}

# Service must periodically call Consul to update health:
# consul_client.agent.check.ttl_pass(check_id)
```

**Pros**: No network accessibility issues
**Cons**: Requires service-side health reporting logic

### Option 4: Run Consul Locally (Production Alternative)

Run Consul agent on same machine as services, federate with remote Consul:

```yaml
# docker-compose.yml
consul-agent:
  image: consul:latest
  ports:
    - "8500:8500"
  environment:
    - CONSUL_BIND_INTERFACE=eth0
    - CONSUL_CLIENT_INTERFACE=eth0
  command: agent -retry-join=192.168.86.200
```

**Pros**: Production-ready, scalable
**Cons**: More complex setup

## Temporary Workaround (Current State)

Services are registered and discoverable in Consul catalog. Health checks can be ignored for development, as service discovery still functions:

```python
from server.utils.consul_utils import get_service_url

# This works regardless of health check state:
intelligence_url = get_service_url(
    "archon-intelligence",
    fallback_url="http://localhost:8053"
)
```

## Recommended Fix (Short-Term)

Add environment variable for local IP and update health check registration:

**1. Update `.env`:**
```bash
LOCAL_IP=192.168.86.XXX  # User's actual dev machine IP
```

**2. Update `consul_service.py`:**
```python
def register_service(self, name: str, port: int, tags: List[str], metadata: Dict[str, str]):
    local_ip = os.getenv("LOCAL_IP", "localhost")

    check = {
        "http": f"http://{local_ip}:{port}/health",
        "interval": "30s",
        "timeout": "5s",
        "deregister_critical_service_after": "90s"
    }
```

## Testing After Fix

```bash
# 1. Update .env with your local IP
echo "LOCAL_IP=192.168.86.XXX" >> .env

# 2. Restart archon-server
docker compose restart archon-server

# 3. Verify health checks from remote Consul
curl http://192.168.86.200:28500/v1/health/service/archon-server

# Expected: "Status": "passing"
```

## References

- Consul Health Checks: https://developer.hashicorp.com/consul/docs/services/usage/checks
- Docker Networking: https://docs.docker.com/network/
- Related Files:
  - `python/src/server/services/consul_service.py:120` (health check registration)
  - `python/src/server/utils/consul_utils.py` (service discovery utilities)
  - `.env` (Consul configuration)

## Implementation Status (OmniNode Bridge)

**Status**: ✅ **Fixed** (2025-10-29)

### What Was Fixed

1. **Settings Configuration** (`src/omninode_bridge/services/metadata_stamping/config/settings.py`)
   - Added `local_ip` field with environment variable support (`METADATA_STAMPING_LOCAL_IP`)
   - Defaults to "localhost" for backward compatibility
   - Falls back to `service_host` if not set

2. **Consul Client Files** (both locations)
   - `src/metadata_stamping/registry/consul_client.py`
   - `src/omninode_bridge/services/metadata_stamping/registry/consul_client.py`
   - Updated `register_service()` to use `local_ip` for health check URLs
   - Added comments explaining remote Consul support

3. **Environment Files**
   - `config/development.env` - Added `METADATA_STAMPING_LOCAL_IP=localhost` with instructions
   - `remote.env` - Added `METADATA_STAMPING_LOCAL_IP=192.168.86.200` for remote deployment

### Code Changes

```python
# Settings configuration now includes:
local_ip: str = Field(
    default="localhost",
    description="Local machine IP for health check URLs (use when Consul is remote)",
)

# Consul client now uses:
local_ip = getattr(settings, "local_ip", service_host)
health_check_url = f"http://{local_ip}:{service_port}/api/v1/metadata-stamping/health/ready"
```

### Configuration

**For Local Development with Remote Consul:**
```bash
# In .env or config/development.env
METADATA_STAMPING_LOCAL_IP=192.168.86.XXX  # Your dev machine's IP
```

**For Remote Deployment (services and Consul on same machine):**
```bash
# In remote.env
METADATA_STAMPING_LOCAL_IP=192.168.86.200  # Remote server IP
```

### Testing

```bash
# 1. Set LOCAL_IP in environment
export METADATA_STAMPING_LOCAL_IP="192.168.86.XXX"

# 2. Restart metadata stamping service
docker compose restart metadata-stamping

# 3. Verify health check passes in Consul
curl http://192.168.86.200:28500/v1/health/service/metadata-stamping-service

# Expected: "Status": "passing"
```

## Related Issues

- Supabase removed from archon-server (2025-10-29) ✅
- Consul integration added (2025-10-29) ✅
- Health check accessibility (OmniNode Bridge) (2025-10-29) ✅ **Fixed**
- Health check accessibility (Archon services) (2025-10-29) ⏳ Pending fix

---

**Impact**: Medium (health monitoring critical for production)
**Effort**: Low (~30 minutes to implement, already fixed in omninode_bridge)
**Owner**: DevOps/Infrastructure Team
**Status (OmniNode Bridge)**: ✅ Complete
**Status (Archon)**: ⏳ Pending (same fix pattern can be applied)
