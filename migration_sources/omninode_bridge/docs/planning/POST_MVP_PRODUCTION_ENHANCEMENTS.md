# Post-MVP Production Enhancements

**Status**: 📋 **ROADMAP ONLY** - Post-MVP Planning Document
**Timeline**: 12-18 weeks (Estimated - Not Currently Scheduled)
**Priority**: Future Production Enhancements
**Last Updated**: October 29, 2025 (Status Review)

## Overview

This document outlines the comprehensive production roadmap for transforming the **MVP two-way node registration system** into a **production-grade composable network** with enterprise security, robustness, and scalability features.

The MVP (Phase 1a) establishes the foundation with basic self-registration, but lacks critical production features identified through architectural review. This roadmap addresses those gaps through a phased implementation plan.

## Architecture Review Summary

### What We Got Right ✅

**Solid Foundation**:
- ✅ **Event-Driven Architecture**: Kafka-based introspection events enable loose coupling
- ✅ **Dual Registration**: Consul + PostgreSQL provides both service discovery and persistent registry
- ✅ **Production-Ready Infrastructure**: Circuit breakers, TTL cache, atomic registration
- ✅ **Comprehensive Observability**: Structured logging, metrics, health checks
- ✅ **Performance Optimization**: Caching, async operations, graceful degradation

**Good Design Patterns**:
- ✅ **IntrospectionMixin**: Clean separation of concerns, reusable across all nodes
- ✅ **OnexEnvelopeV1**: Standardized event format with metadata
- ✅ **Heartbeat Monitoring**: Liveness detection with configurable intervals
- ✅ **TTL-Based Cleanup**: Automatic removal of stale nodes

### Critical Gaps Identified ⚠️

**Security Deficiencies**:
- ❌ **No Trust Model**: Introspection events are unsigned and unverified
- ❌ **No Identity System**: Nodes can impersonate each other (spoofing attacks)
- ❌ **No Authentication**: No verification of node identity during registration
- ❌ **No Authorization**: No policy enforcement for registration permissions
- ❌ **Plaintext Communication**: Kafka messages and Consul API calls unencrypted

**Robustness Gaps**:
- ❌ **No Lease Semantics**: Registrations don't expire automatically in Consul
- ❌ **No Tombstones**: Deleted nodes don't publish explicit removal events
- ❌ **No Topology Diff Stream**: Consumers must poll for topology changes
- ❌ **Single Point of Failure**: Single registry instance with no failover

**Control Plane vs Data Plane Confusion**:
- ❌ **Mixed Responsibilities**: Registry handles both discovery (control) and registration (data)
- ❌ **No Separation**: All operations go through the same Kafka topics
- ❌ **No Priority Handling**: Critical control messages compete with data messages

**Multi-Tenancy Limitations**:
- ❌ **No Network Isolation**: All nodes share the same Kafka topics and Consul datacenter
- ❌ **No Namespace Support**: Cannot separate tenants or environments logically
- ❌ **No Policy Engine**: Cannot enforce tenant-specific registration policies

**Federation Gaps**:
- ❌ **Single Network Only**: No support for cross-datacenter or cross-cloud deployments
- ❌ **No Bridge Nodes**: Cannot connect separate network segments
- ❌ **No Cross-Network Policy**: Cannot enforce policies across network boundaries

## Phased Enhancement Plan

### ✅ Phase 0: MVP Foundation (COMPLETE - October 2025)

**Timeline**: Completed October 2025
**Status**: ✅ **COMPLETE**

**Delivered**:
- ✅ Bridge nodes (orchestrator, reducer, registry) with ONEX v2.0 compliance
- ✅ Event infrastructure (13 Kafka topics, OnexEnvelopeV1 format)
- ✅ Database persistence (PostgreSQL with 5 migrations)
- ✅ Comprehensive test suite (501 tests, 92.8% passing)
- ✅ LlamaIndex workflow integration
- ✅ Performance benchmarks exceeded (92.8% test coverage)

**Reference**: See [ROADMAP.md](../ROADMAP.md) for Phase 1 & 2 details.

---

### ⏳ Phase 1a: MVP Self-Registration (PLANNED)

**Timeline**: 1-2 days (Not Currently Scheduled)
**Status**: ⏳ **PLANNED** - Implementation Pending

**Scope**:
- Registry publishes introspection on startup
- Registry registers itself in Consul and PostgreSQL
- Basic heartbeat and health monitoring
- **See**: [MVP_REGISTRY_SELF_REGISTRATION.md](./MVP_REGISTRY_SELF_REGISTRATION.md)

**Known Limitations**:
- No trust model, no signed introspection
- No lease semantics, no tombstones
- Single-tenant only, no federation

---

### ⏳ Phase 1b: Trust & Security (PLANNED)

**Timeline**: 2 weeks (Not Currently Scheduled)
**Status**: ⏳ **PLANNED** - Future Enhancement

**Goals**:
- Implement SPIFFE-based identity system
- Add signed introspection with cryptographic verification
- Establish bootstrap controller with certificate authority
- Secure Kafka and Consul communication

**Scope**:

#### 1. SPIFFE Identity System

**Background**: [SPIFFE](https://spiffe.io/) (Secure Production Identity Framework For Everyone) provides:
- Cryptographic identity for services (SVID - SPIFFE Verifiable Identity Document)
- Automatic credential rotation and distribution
- Platform-agnostic identity (works with Docker, Kubernetes, VMs)
- Zero-trust security model (trust nothing, verify everything)

**Implementation**:

```mermaid
graph TD
    A[Bootstrap Controller] -->|Issues SVID| B[Registry Node]
    A -->|Issues SVID| C[Orchestrator Node]
    A -->|Issues SVID| D[Reducer Node]
    B -->|Signed Introspection| E[Kafka]
    C -->|Signed Introspection| E
    D -->|Signed Introspection| E
    E -->|Verify Signature| B
```

**Components**:

1. **Bootstrap Controller**:
   - Central certificate authority (CA) for the network
   - Issues SPIFFE IDs to nodes on startup
   - Rotates credentials automatically (configurable TTL)
   - Maintains trust bundle (root CA certificates)

2. **SPIFFE Agent** (on each node):
   - Requests SVID from bootstrap controller
   - Stores SVID securely (in-memory or secret manager)
   - Rotates credentials before expiry
   - Provides signing/verification APIs to node

3. **Modified IntrospectionMixin**:
   - Sign introspection events with node's private key
   - Include SVID certificate in envelope metadata
   - Verify signatures of received events before processing

**SPIFFE ID Format**:
```
spiffe://omninode.network/node/{node_type}/{node_id}

Examples:
- spiffe://omninode.network/node/registry/registry-abc123
- spiffe://omninode.network/node/orchestrator/orchestrator-001
- spiffe://omninode.network/node/reducer/reducer-001
```

**Signed Introspection Event**:

```json
{
  "envelope_version": "1.0",
  "event_id": "550e8400-e29b-41d4-a716-446655440000",
  "event_type": "NODE_INTROSPECTION",
  "timestamp": "2025-10-28T12:00:00Z",

  "security": {
    "spiffe_id": "spiffe://omninode.network/node/registry/registry-abc123",
    "svid_certificate": "-----BEGIN CERTIFICATE-----\n...\n-----END CERTIFICATE-----",
    "signature": "base64_encoded_signature",
    "signature_algorithm": "SHA256withRSA",
    "signed_at": "2025-10-28T12:00:00Z"
  },

  "payload": {
    "node_id": "registry-abc123",
    "node_type": "registry",
    # ... rest of introspection payload ...
  }
}
```

**Verification Process**:

1. Registry receives introspection event
2. Extract SPIFFE ID and signature from envelope
3. Verify SVID certificate against trust bundle (check CA signature, expiry)
4. Verify event signature using public key from SVID certificate
5. Check SPIFFE ID matches expected format and node_id
6. If verification fails → log security event, discard message, alert
7. If verification succeeds → proceed with registration

**Security Properties**:
- **Authenticity**: Only nodes with valid SVIDs can publish introspection
- **Integrity**: Tampered messages are detected via signature verification
- **Non-Repudiation**: Signed messages prove sender identity
- **Replay Protection**: Timestamps prevent replay attacks

#### 2. Bootstrap Controller Implementation

**Responsibilities**:
- Act as SPIFFE server (issue, rotate, revoke SVIDs)
- Maintain certificate authority (CA) for the network
- Provide SVID workload API for nodes
- Publish trust bundle updates to all nodes

**Architecture**:

```
┌─────────────────────────────────────────────────────────┐
│           Bootstrap Controller (SPIRE Server)           │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────┐  ┌──────────────────┐           │
│  │ Certificate      │  │ Workload API     │           │
│  │ Authority (CA)   │  │ (SPIFFE)         │           │
│  └────────┬─────────┘  └────────┬─────────┘           │
│           │                     │                      │
│  ┌────────▼─────────────────────▼─────────┐           │
│  │   Trust Bundle Management              │           │
│  │   - Root CA certificates               │           │
│  │   - CRL (Certificate Revocation List)  │           │
│  │   - Policy Engine (who can get SVIDs)  │           │
│  └────────────────────────────────────────┘           │
│                                                          │
└─────────────────────────────────────────────────────────┘
                       ↓
              ┌────────────────┐
              │ SPIRE Agent    │
              │ (on each node) │
              └────────────────┘
```

**Implementation Options**:

1. **Option A: SPIRE (Recommended)**:
   - Use [SPIRE](https://spiffe.io/docs/latest/spire-about/) (SPIFFE Runtime Environment)
   - Production-ready, battle-tested implementation
   - Automatic credential rotation, policy engine, multi-platform support
   - Deployment: Run SPIRE server as bootstrap controller, SPIRE agent on each node

2. **Option B: Custom CA**:
   - Implement simple CA using Python cryptography library
   - Less features but easier to customize
   - Suitable for single-datacenter deployments only

**Recommended**: Use SPIRE for production deployments.

**SPIRE Deployment**:

```bash
# Bootstrap controller (SPIRE server)
docker run -d \
  --name spire-server \
  -p 8081:8081 \
  -v /opt/spire/server:/opt/spire/data \
  ghcr.io/spiffe/spire-server:latest \
  -config /opt/spire/server/server.conf

# Node agent (SPIRE agent)
docker run -d \
  --name spire-agent \
  -v /opt/spire/agent:/opt/spire/data \
  -v /run/spire/sockets:/run/spire/sockets \
  ghcr.io/spiffe/spire-agent:latest \
  -config /opt/spire/agent/agent.conf
```

**SPIRE Server Configuration**:

```hcl
server {
  bind_address = "0.0.0.0"
  bind_port = "8081"
  trust_domain = "omninode.network"
  data_dir = "/opt/spire/data/server"
  log_level = "INFO"
}

plugins {
  DataStore "sql" {
    plugin_data {
      database_type = "postgres"
      connection_string = "postgresql://spire:password@postgres:5432/spire" # pragma: allowlist secret
    }
  }

  KeyManager "disk" {
    plugin_data {
      keys_path = "/opt/spire/data/server/keys"
    }
  }

  NodeAttestor "join_token" {
    plugin_data {}
  }

  UpstreamAuthority "disk" {
    plugin_data {
      cert_file_path = "/opt/spire/data/server/ca.crt"
      key_file_path = "/opt/spire/data/server/ca.key"
    }
  }
}
```

**Node Registration with SPIRE**:

```python
# File: src/omninode_bridge/nodes/mixins/spiffe_mixin.py

import asyncio
from spiffe import WorkloadApiClient
from spiffe.svid.x509_svid import X509Svid

class SpiffeMixin:
    """Mixin providing SPIFFE identity and signing capabilities."""

    async def initialize_spiffe(self) -> None:
        """Initialize SPIFFE agent and obtain SVID."""
        try:
            # Connect to SPIRE agent via Unix socket
            self.spiffe_client = WorkloadApiClient(
                socket_path="/run/spire/sockets/agent.sock"
            )

            # Fetch X.509 SVID
            self.svid: X509Svid = await asyncio.to_thread(
                self.spiffe_client.fetch_x509_svid
            )

            # Extract SPIFFE ID
            self.spiffe_id = self.svid.spiffe_id.to_string()

            # Get trust bundle (root CA certificates)
            self.trust_bundle = await asyncio.to_thread(
                self.spiffe_client.fetch_x509_bundles
            )

            logger.info(f"SPIFFE identity obtained: {self.spiffe_id}")

        except Exception as e:
            logger.error(f"Failed to initialize SPIFFE: {e}")
            raise

    async def sign_data(self, data: bytes) -> bytes:
        """Sign data using SVID private key."""
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        signature = self.svid.private_key.sign(
            data,
            padding.PKCS1v15(),
            hashes.SHA256()
        )
        return signature

    async def verify_signature(
        self, data: bytes, signature: bytes, spiffe_id: str
    ) -> bool:
        """Verify signature using trust bundle."""
        try:
            # Get certificate for SPIFFE ID from trust bundle
            cert = self.trust_bundle.get_x509_bundle_for_trust_domain(
                spiffe_id.split("//")[1].split("/")[0]
            )

            # Verify signature
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding

            cert.public_key().verify(
                signature,
                data,
                padding.PKCS1v15(),
                hashes.SHA256()
            )
            return True

        except Exception as e:
            logger.warning(f"Signature verification failed: {e}")
            return False
```

**Modified IntrospectionMixin**:

```python
# File: src/omninode_bridge/nodes/mixins/introspection_mixin.py

class IntrospectionMixin(SpiffeMixin):  # Inherit from SpiffeMixin
    """Enhanced introspection with SPIFFE-based signing."""

    async def publish_introspection(
        self,
        reason: str = "periodic",
        correlation_id: Optional[UUID] = None,
        force_refresh: bool = False,
    ) -> bool:
        """Publish signed introspection data to Kafka."""
        try:
            # Get introspection data
            introspection_data = await self.get_introspection_data()

            # Serialize payload
            import json
            payload_bytes = json.dumps(introspection_data).encode("utf-8")

            # Sign payload with SVID private key
            signature = await self.sign_data(payload_bytes)

            # Create signed envelope
            envelope = {
                "envelope_version": "1.0",
                "event_id": str(uuid4()),
                "event_type": "NODE_INTROSPECTION",
                "timestamp": datetime.now(UTC).isoformat(),
                "security": {
                    "spiffe_id": self.spiffe_id,
                    "svid_certificate": self.svid.cert.public_bytes(
                        serialization.Encoding.PEM
                    ).decode("utf-8"),
                    "signature": base64.b64encode(signature).decode("utf-8"),
                    "signature_algorithm": "SHA256withRSA",
                    "signed_at": datetime.now(UTC).isoformat(),
                },
                "payload": introspection_data,
            }

            # Publish to Kafka
            success = await self._publish_to_kafka(envelope)

            return success

        except Exception as e:
            logger.error(f"Failed to publish signed introspection: {e}")
            return False
```

#### 3. Secure Communication Channels

**Kafka TLS Encryption**:

```yaml
# Kafka broker configuration (server.properties)
listeners=SSL://0.0.0.0:9093
ssl.keystore.location=/var/private/ssl/kafka.server.keystore.jks
ssl.keystore.password=<keystore_password>
ssl.key.password=<key_password>
ssl.truststore.location=/var/private/ssl/kafka.server.truststore.jks
ssl.truststore.password=<truststore_password>
ssl.client.auth=required
```

```python
# KafkaClient configuration
kafka_client = KafkaClient(
    bootstrap_servers="kafka:9093",
    security_protocol="SSL",
    ssl_cafile="/etc/ssl/ca-cert.pem",
    ssl_certfile="/etc/ssl/client-cert.pem",
    ssl_keyfile="/etc/ssl/client-key.pem",
)
```

**Consul TLS Encryption**:

```hcl
# Consul server configuration
ports {
  https = 8501
  http = -1  # Disable HTTP
}

tls {
  defaults {
    ca_file = "/etc/consul.d/ca.pem"
    cert_file = "/etc/consul.d/server.pem"
    key_file = "/etc/consul.d/server-key.pem"
    verify_incoming = true
    verify_outgoing = true
  }
}
```

**Timeline**: 2 weeks
**Dependencies**: None (can start immediately after Phase 1a)
**Estimated Effort**: 60-80 hours

**Success Criteria**:
- ✅ All nodes obtain SPIFFE IDs on startup
- ✅ Introspection events are cryptographically signed
- ✅ Registry verifies signatures before processing
- ✅ Kafka and Consul use TLS encryption
- ✅ Unauthorized nodes cannot register
- ✅ Signature verification failures are logged and alerted

---

### ⏳ Phase 1c: Robustness & Resilience (PLANNED)

**Timeline**: 1 week (Not Currently Scheduled)
**Status**: ⏳ **PLANNED** - Future Enhancement

**Goals**:
- Implement lease semantics with automatic expiry
- Add tombstone events for explicit node removal
- Create topology diff stream for incremental updates
- Add Consul adapter for service discovery integration

**Scope**:

#### 1. Lease Semantics with TTL

**Problem**: Currently, node registrations in Consul don't expire automatically. If a node crashes without deregistering, its registration remains indefinitely, causing stale service discovery.

**Solution**: Implement TTL-based leases in Consul with automatic renewal.

**Architecture**:

```mermaid
sequenceDiagram
    participant Node
    participant Registry
    participant Consul

    Node->>Registry: Introspection Event (TTL: 60s)
    Registry->>Consul: Register Service with TTL Check
    Note over Consul: Consul creates TTL check (60s)

    loop Every 30s (renewal interval)
        Node->>Registry: Heartbeat Event
        Registry->>Consul: Pass TTL Check
        Note over Consul: TTL check renewed
    end

    Note over Node: Node crashes
    Note over Consul: Wait 60s (TTL expiry)
    Consul->>Consul: Mark service as critical
    Note over Consul: Service removed from discovery
```

**Implementation**:

```python
# Modified Consul registration with TTL check
async def _register_with_consul(
    self, introspection: ModelNodeIntrospectionEvent
) -> dict[str, Any]:
    """Register node with Consul using TTL-based health check."""

    # Extract TTL from introspection metadata (default: 60 seconds)
    lease_ttl_seconds = introspection.metadata.get("lease_ttl_seconds", 60)

    # Create Consul service registration with TTL check
    service_registration = {
        "ID": introspection.node_id,
        "Name": introspection.node_type,
        "Port": service_port,
        "Tags": [introspection.node_type, "omninode_bridge"],
        "Meta": {
            "node_type": introspection.node_type,
            "capabilities": ",".join(introspection.capabilities or []),
        },
        "Check": {
            "CheckID": f"ttl-{introspection.node_id}",
            "Name": f"Node Lease TTL for {introspection.node_id}",
            "TTL": f"{lease_ttl_seconds}s",
            "Status": "passing",
            "DeregisterCriticalServiceAfter": f"{lease_ttl_seconds * 2}s",
        },
    }

    # Register service with Consul
    await self.consul_client.agent.service.register(service_registration)

    logger.info(
        f"Registered {introspection.node_id} with TTL check ({lease_ttl_seconds}s)"
    )

    return {
        "success": True,
        "node_id": introspection.node_id,
        "lease_ttl_seconds": lease_ttl_seconds,
    }
```

**Heartbeat-Based Lease Renewal**:

```python
# Modified heartbeat handler in registry
async def _process_heartbeat_event(self, heartbeat_event: ModelNodeHeartbeatEvent) -> None:
    """Process heartbeat event and renew Consul TTL check."""

    node_id = heartbeat_event.node_id

    # Update node last seen timestamp
    self.node_last_seen[node_id] = datetime.now(UTC)

    # Renew Consul TTL check
    if self.consul_client:
        try:
            await self.consul_client.agent.check.ttl_pass(
                check_id=f"ttl-{node_id}",
                note="Heartbeat received"
            )
            logger.debug(f"Renewed TTL check for {node_id}")
        except Exception as e:
            logger.error(f"Failed to renew TTL check for {node_id}: {e}")
```

**Benefits**:
- Automatic cleanup of crashed nodes (no manual intervention)
- Stale registrations expire after `lease_ttl_seconds`
- Consul service discovery always accurate
- Reduced operational burden

#### 2. Tombstone Events

**Problem**: When a node is deleted or deregistered, there's no explicit notification. Consumers must wait for TTL expiry to detect removal, causing delays in topology updates.

**Solution**: Publish TOMBSTONE events for explicit node removal notifications.

**Event Format**:

```json
{
  "envelope_version": "1.0",
  "event_id": "550e8400-e29b-41d4-a716-446655440000",
  "event_type": "NODE_TOMBSTONE",
  "timestamp": "2025-10-28T12:30:00Z",
  "source_node": "orchestrator-001",

  "payload": {
    "node_id": "orchestrator-001",
    "node_type": "orchestrator",
    "removal_reason": "graceful_shutdown",  # or "lease_expired", "manual_deregistration"
    "removal_timestamp": "2025-10-28T12:30:00Z",
    "final_status": "healthy",
    "metadata": {
      "uptime_seconds": 3600,
      "total_operations": 1500
    }
  }
}
```

**Implementation**:

```python
# File: src/omninode_bridge/nodes/mixins/introspection_mixin.py

async def publish_tombstone(
    self, reason: str = "graceful_shutdown"
) -> bool:
    """Publish tombstone event for node removal."""
    try:
        tombstone_event = {
            "node_id": self.node_id,
            "node_type": self._get_node_type_cached(),
            "removal_reason": reason,
            "removal_timestamp": datetime.now(UTC).isoformat(),
            "final_status": self.health_status.value if hasattr(self, "health_status") else "unknown",
            "metadata": {
                "uptime_seconds": int(time.time() - self._startup_time) if hasattr(self, "_startup_time") else 0,
            },
        }

        # Publish to Kafka
        success = await self.kafka_client.publish_with_envelope(
            event_type="node-tombstone",
            source_node_id=self.node_id,
            payload=tombstone_event,
            topic=f"{self._get_environment()}.omninode_bridge.onex.evt.node-tombstone.v1",
        )

        logger.info(f"Published tombstone event for {self.node_id}, reason={reason}")
        return success

    except Exception as e:
        logger.error(f"Failed to publish tombstone event: {e}")
        return False
```

**Registry Tombstone Handling**:

```python
# File: src/omninode_bridge/nodes/registry/v1_0_0/node.py

async def _process_tombstone_event(self, tombstone_event: dict) -> None:
    """Process tombstone event and deregister node immediately."""

    node_id = tombstone_event["node_id"]

    logger.info(f"Received tombstone event for {node_id}, deregistering immediately")

    # Remove from local tracking
    if node_id in self.registered_nodes:
        del self.registered_nodes[node_id]
    if node_id in self.node_last_seen:
        del self.node_last_seen[node_id]

    # Deregister from Consul
    if self.consul_client:
        try:
            await self.consul_client.agent.service.deregister(node_id)
            logger.info(f"Deregistered {node_id} from Consul via tombstone")
        except Exception as e:
            logger.error(f"Failed to deregister {node_id} from Consul: {e}")

    # Mark as deleted in PostgreSQL (soft delete)
    if self.node_repository:
        try:
            await self.node_repository.mark_deleted(
                node_id=node_id,
                deletion_reason=tombstone_event.get("removal_reason", "tombstone"),
                deleted_at=datetime.now(UTC),
            )
            logger.info(f"Marked {node_id} as deleted in PostgreSQL")
        except Exception as e:
            logger.error(f"Failed to mark {node_id} as deleted in PostgreSQL: {e}")
```

**Benefits**:
- Immediate topology updates (no TTL wait)
- Explicit removal reasons for debugging
- Audit trail of node removals
- Better UX for consumers (instant notification)

#### 3. Topology Diff Stream

**Problem**: Consumers must poll Consul or query PostgreSQL to detect topology changes. This is inefficient and has high latency.

**Solution**: Create a **topology diff stream** that publishes incremental updates.

**Architecture**:

```
Registry maintains topology snapshot in memory
    ↓
On every introspection/heartbeat/tombstone event:
    - Compare current topology with previous snapshot
    - If changes detected → publish TOPOLOGY_DIFF event
    ↓
Consumers subscribe to TOPOLOGY_DIFF events
    - Receive only changes (added/updated/removed nodes)
    - Apply diffs incrementally to local topology cache
```

**Event Format**:

```json
{
  "envelope_version": "1.0",
  "event_id": "550e8400-e29b-41d4-a716-446655440000",
  "event_type": "TOPOLOGY_DIFF",
  "timestamp": "2025-10-28T12:35:00Z",
  "source_node": "registry-abc123",

  "payload": {
    "topology_version": 42,  # Monotonically increasing version
    "previous_version": 41,
    "changes": [
      {
        "change_type": "added",
        "node_id": "orchestrator-002",
        "node_type": "orchestrator",
        "endpoints": { "health": "http://orchestrator-002:8053/health" },
        "capabilities": { "max_concurrent_workflows": 100 }
      },
      {
        "change_type": "updated",
        "node_id": "reducer-001",
        "node_type": "reducer",
        "changed_fields": ["endpoints", "capabilities"],
        "endpoints": { "health": "http://reducer-001:8060/health" },
        "capabilities": { "aggregation_throughput": 2000 }
      },
      {
        "change_type": "removed",
        "node_id": "orchestrator-001",
        "removal_reason": "lease_expired"
      }
    ],
    "summary": {
      "total_nodes": 15,
      "nodes_added": 1,
      "nodes_updated": 1,
      "nodes_removed": 1
    }
  }
}
```

**Implementation**:

```python
# File: src/omninode_bridge/nodes/registry/v1_0_0/node.py

class NodeBridgeRegistry(NodeEffect, HealthCheckMixin, IntrospectionMixin):
    """Registry with topology diff stream support."""

    def __init__(self, container: ModelONEXContainer, environment: str = "development") -> None:
        super().__init__(container)

        # ... existing initialization ...

        # Topology diff tracking
        self.topology_version = 0  # Monotonically increasing
        self.topology_snapshot: dict[str, ModelNodeIntrospectionEvent] = {}  # Previous snapshot

    async def dual_register(
        self, introspection: ModelNodeIntrospectionEvent
    ) -> dict[str, Any]:
        """Perform dual registration and publish topology diff."""

        # Perform registration
        result = await super().dual_register(introspection)

        # Detect topology changes
        await self._publish_topology_diff()

        return result

    async def _publish_topology_diff(self) -> None:
        """Publish topology diff event if changes detected."""

        # Compare current topology with previous snapshot
        changes = []

        # Check for added/updated nodes
        for node_id, introspection in self.registered_nodes.items():
            if node_id not in self.topology_snapshot:
                # Node added
                changes.append({
                    "change_type": "added",
                    "node_id": node_id,
                    "node_type": introspection.node_type,
                    "endpoints": introspection.endpoints,
                    "capabilities": introspection.capabilities,
                })
            elif introspection != self.topology_snapshot[node_id]:
                # Node updated
                changes.append({
                    "change_type": "updated",
                    "node_id": node_id,
                    "node_type": introspection.node_type,
                    "changed_fields": self._detect_changed_fields(
                        self.topology_snapshot[node_id], introspection
                    ),
                    "endpoints": introspection.endpoints,
                    "capabilities": introspection.capabilities,
                })

        # Check for removed nodes
        for node_id in self.topology_snapshot:
            if node_id not in self.registered_nodes:
                changes.append({
                    "change_type": "removed",
                    "node_id": node_id,
                    "removal_reason": "unknown",
                })

        # If no changes, skip publishing
        if not changes:
            return

        # Increment topology version
        self.topology_version += 1

        # Create topology diff event
        diff_event = {
            "topology_version": self.topology_version,
            "previous_version": self.topology_version - 1,
            "changes": changes,
            "summary": {
                "total_nodes": len(self.registered_nodes),
                "nodes_added": sum(1 for c in changes if c["change_type"] == "added"),
                "nodes_updated": sum(1 for c in changes if c["change_type"] == "updated"),
                "nodes_removed": sum(1 for c in changes if c["change_type"] == "removed"),
            },
        }

        # Publish to Kafka
        await self.kafka_client.publish_with_envelope(
            event_type="topology-diff",
            source_node_id=self.registry_id,
            payload=diff_event,
            topic=f"{self._get_environment()}.omninode_bridge.onex.evt.topology-diff.v1",
        )

        logger.info(
            f"Published topology diff v{self.topology_version} with {len(changes)} changes"
        )

        # Update snapshot
        self.topology_snapshot = self.registered_nodes.copy()
```

**Consumer Usage**:

```python
# Example consumer subscribing to topology diff stream
class TopologyConsumer:
    def __init__(self):
        self.local_topology: dict[str, dict] = {}
        self.topology_version = 0

    async def consume_topology_diffs(self):
        """Consume topology diff events and apply changes."""

        async for message in kafka_client.consume_messages(
            topic="dev.omninode_bridge.onex.evt.topology-diff.v1",
            group_id="topology-consumer",
        ):
            diff_event = message.payload

            # Validate version sequence
            if diff_event["topology_version"] != self.topology_version + 1:
                logger.warning(
                    f"Topology version mismatch: expected {self.topology_version + 1}, "
                    f"got {diff_event['topology_version']}, resyncing full topology"
                )
                await self.resync_full_topology()
                continue

            # Apply changes incrementally
            for change in diff_event["changes"]:
                if change["change_type"] == "added":
                    self.local_topology[change["node_id"]] = change
                elif change["change_type"] == "updated":
                    self.local_topology[change["node_id"]].update(change)
                elif change["change_type"] == "removed":
                    del self.local_topology[change["node_id"]]

            # Update version
            self.topology_version = diff_event["topology_version"]

            logger.info(f"Applied topology diff v{self.topology_version}")
```

**Benefits**:
- Consumers get instant topology updates (no polling)
- Incremental updates reduce bandwidth and processing
- Version tracking enables consistency checks
- Resync capability for out-of-sync consumers

#### 4. Consul Adapter for Service Discovery

**Problem**: External tools (e.g., Prometheus, Grafana, monitoring systems) need to discover services but don't have direct access to our Kafka-based introspection system.

**Solution**: Create a **Consul Adapter** that bridges Kafka introspection events to Consul service catalog.

**Architecture**:

```
Kafka Introspection Events → Consul Adapter → Consul Service Catalog
                                              ↓
                                   External tools query Consul
                                   (Prometheus, Grafana, etc.)
```

**Implementation**: Already implemented in `_register_with_consul()` method. No additional work needed.

**Timeline**: 1 week
**Dependencies**: Phase 1b (SPIFFE for secure lease renewal)
**Estimated Effort**: 40-50 hours

**Success Criteria**:
- ✅ Consul registrations use TTL-based health checks
- ✅ Heartbeats renew Consul TTL checks automatically
- ✅ Tombstone events trigger immediate deregistration
- ✅ Topology diff stream publishes incremental updates
- ✅ Consumers can subscribe to topology changes
- ✅ Version tracking prevents out-of-sync consumers

---

### ⏳ Phase 2: Scale & Multi-Tenancy (PLANNED)

**Timeline**: 4-6 weeks (Not Currently Scheduled)
**Status**: ⏳ **PLANNED** - Future Enhancement

**Goals**:
- Implement network capabilities and namespace isolation
- Add policy engine for registration authorization
- Implement idempotency and deduplication at registry level
- Add multi-registry support for horizontal scaling

**Scope**:

#### 1. Network Capabilities & Namespace Isolation

**Problem**: All nodes currently share the same Kafka topics and Consul datacenter, preventing multi-tenant deployments.

**Solution**: Implement **network namespaces** for logical isolation.

**Architecture**:

```
┌─────────────────────────────────────────────────────────────┐
│                      Network: omninode-prod                  │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────┐  ┌──────────────────────┐        │
│  │ Namespace: tenant-A  │  │ Namespace: tenant-B  │        │
│  ├──────────────────────┤  ├──────────────────────┤        │
│  │ - Registry           │  │ - Registry           │        │
│  │ - Orchestrator-A1    │  │ - Orchestrator-B1    │        │
│  │ - Reducer-A1         │  │ - Reducer-B1         │        │
│  └──────────────────────┘  └──────────────────────┘        │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

**Kafka Topic Naming**:

```
{network}.{namespace}.{service}.{event_type}.{version}

Examples:
- omninode-prod.tenant-a.omninode_bridge.onex.evt.node-introspection.v1
- omninode-prod.tenant-b.omninode_bridge.onex.evt.node-introspection.v1
```

**Consul Namespace Support** (Consul Enterprise feature):

```hcl
# Consul configuration
namespaces {
  enabled = true
}

# Register service in namespace
service {
  name = "orchestrator-001"
  namespace = "tenant-a"
  # ... rest of configuration ...
}
```

**PostgreSQL Schema with Namespaces**:

```sql
-- Add namespace column to node_registrations table
ALTER TABLE node_registrations
ADD COLUMN namespace VARCHAR(128) NOT NULL DEFAULT 'default';

-- Add composite index for namespace queries
CREATE INDEX idx_node_registrations_namespace_node_id
ON node_registrations (namespace, node_id);

-- Add namespace to unique constraint
ALTER TABLE node_registrations
DROP CONSTRAINT node_registrations_node_id_key;

ALTER TABLE node_registrations
ADD CONSTRAINT node_registrations_namespace_node_id_key
UNIQUE (namespace, node_id);
```

**Implementation**:

```python
# File: src/omninode_bridge/nodes/registry/v1_0_0/node.py

class NodeBridgeRegistry(NodeEffect, HealthCheckMixin, IntrospectionMixin):
    """Registry with namespace support."""

    def __init__(self, container: ModelONEXContainer, environment: str = "development") -> None:
        super().__init__(container)

        # ... existing initialization ...

        # Namespace configuration
        self.namespace = container.value.get("namespace", "default")
        self.network_id = container.value.get("network_id", "omninode-default")

        logger.info(
            f"Registry initialized with namespace={self.namespace}, network={self.network_id}"
        )

    async def dual_register(
        self, introspection: ModelNodeIntrospectionEvent
    ) -> dict[str, Any]:
        """Perform dual registration with namespace isolation."""

        # Extract namespace from introspection metadata
        node_namespace = introspection.metadata.get("namespace", self.namespace)

        # Validate namespace matches registry namespace
        if node_namespace != self.namespace:
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.AUTHORIZATION_ERROR,
                message=f"Node namespace mismatch: node={node_namespace}, registry={self.namespace}",
                context={
                    "node_id": introspection.node_id,
                    "node_namespace": node_namespace,
                    "registry_namespace": self.namespace,
                },
            )

        # Proceed with registration (namespace included in Consul/PostgreSQL)
        result = await super().dual_register(introspection)

        return result
```

#### 2. Policy Engine for Authorization

**Problem**: Any node can register with any node_id or node_type, enabling spoofing and unauthorized access.

**Solution**: Implement **policy engine** for registration authorization based on node attributes and SPIFFE identity.

**Policy Language** (using Open Policy Agent - OPA):

```rego
# File: policies/node_registration.rego

package omninode.registration

import future.keywords.in

# Default deny
default allow = false

# Allow registration if:
# 1. Node has valid SPIFFE ID matching expected format
# 2. Node type is allowed for this namespace
# 3. Node capabilities are within limits
allow {
    valid_spiffe_id
    allowed_node_type
    within_capability_limits
}

# Validate SPIFFE ID format
valid_spiffe_id {
    # Extract SPIFFE ID from introspection event
    spiffe_id := input.introspection.security.spiffe_id

    # Check format: spiffe://omninode.network/node/{node_type}/{node_id}
    regex.match(`^spiffe://omninode.network/node/[a-z]+/[a-z0-9-]+$`, spiffe_id)

    # Extract node_type and node_id from SPIFFE ID
    parts := split(spiffe_id, "/")
    node_type := parts[4]
    node_id := parts[5]

    # Verify node_type and node_id match introspection payload
    node_type == input.introspection.node_type
    node_id == input.introspection.node_id
}

# Check if node type is allowed in this namespace
allowed_node_type {
    node_type := input.introspection.node_type
    namespace := input.namespace

    # Lookup allowed node types for namespace
    allowed_types := data.namespaces[namespace].allowed_node_types

    # Check if node type is in allowed list
    node_type in allowed_types
}

# Check if capabilities are within limits
within_capability_limits {
    capabilities := input.introspection.capabilities

    # Example: Orchestrator max concurrent workflows
    input.introspection.node_type == "orchestrator"
    capabilities.max_concurrent_workflows <= 100
}

within_capability_limits {
    capabilities := input.introspection.capabilities

    # Example: Reducer aggregation throughput
    input.introspection.node_type == "reducer"
    capabilities.aggregation_throughput <= 2000
}
```

**Policy Data** (namespace configuration):

```json
{
  "namespaces": {
    "tenant-a": {
      "allowed_node_types": ["orchestrator", "reducer"],
      "max_nodes_per_type": 10
    },
    "tenant-b": {
      "allowed_node_types": ["orchestrator", "reducer", "registry"],
      "max_nodes_per_type": 20
    }
  }
}
```

**Integration with Registry**:

```python
# File: src/omninode_bridge/nodes/registry/v1_0_0/node.py

from opa_client import OPAClient

class NodeBridgeRegistry(NodeEffect, HealthCheckMixin, IntrospectionMixin):
    """Registry with policy-based authorization."""

    def __init__(self, container: ModelONEXContainer, environment: str = "development") -> None:
        super().__init__(container)

        # ... existing initialization ...

        # Initialize OPA client
        self.opa_client = OPAClient(
            url=os.getenv("OPA_URL", "http://localhost:8181"),
            policy_path="omninode/registration",
        )

    async def dual_register(
        self, introspection: ModelNodeIntrospectionEvent
    ) -> dict[str, Any]:
        """Perform dual registration with policy-based authorization."""

        # Build policy evaluation input
        policy_input = {
            "introspection": introspection.to_dict(),
            "namespace": self.namespace,
        }

        # Evaluate policy
        policy_result = await self.opa_client.evaluate(policy_input)

        # Check if registration is allowed
        if not policy_result.get("result", {}).get("allow", False):
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.AUTHORIZATION_ERROR,
                message="Node registration denied by policy",
                context={
                    "node_id": introspection.node_id,
                    "policy_result": policy_result,
                },
            )

        # Proceed with registration
        result = await super().dual_register(introspection)

        return result
```

#### 3. Idempotency & Deduplication

**Problem**: Duplicate introspection events can cause unnecessary registration operations and database writes.

**Solution**: Implement **idempotency keys** and **deduplication** at registry level.

**Implementation**:

```python
# File: src/omninode_bridge/nodes/registry/v1_0_0/node.py

class NodeBridgeRegistry(NodeEffect, HealthCheckMixin, IntrospectionMixin):
    """Registry with idempotency support."""

    async def dual_register(
        self, introspection: ModelNodeIntrospectionEvent
    ) -> dict[str, Any]:
        """Perform idempotent dual registration."""

        # Generate idempotency key from introspection data
        idempotency_key = self._generate_idempotency_key(introspection)

        # Check if this registration was already processed
        cached_result = self._idempotency_cache.get(idempotency_key)
        if cached_result is not None:
            logger.info(
                f"Registration for {introspection.node_id} is duplicate (idempotency key: {idempotency_key}), "
                "returning cached result"
            )
            return cached_result

        # Perform registration
        result = await super().dual_register(introspection)

        # Cache result with TTL (e.g., 5 minutes)
        self._idempotency_cache.put(
            idempotency_key,
            result,
            ttl_seconds=300,
        )

        return result

    def _generate_idempotency_key(self, introspection: ModelNodeIntrospectionEvent) -> str:
        """Generate idempotency key from introspection data."""
        import hashlib
        import json

        # Use node_id, timestamp, and capabilities hash
        data = {
            "node_id": introspection.node_id,
            "timestamp": introspection.timestamp.isoformat(),
            "capabilities_hash": hashlib.sha256(
                json.dumps(introspection.capabilities, sort_keys=True).encode()
            ).hexdigest()[:16],
        }

        # Create idempotency key
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()[:32]
```

#### 4. Multi-Registry Support (Horizontal Scaling)

**Problem**: Single registry instance is a bottleneck and single point of failure.

**Solution**: Support **multiple registry instances** with leader election and work partitioning.

**Architecture**:

```
┌────────────────────────────────────────────────────────────┐
│                  Consul (Leader Election)                   │
└────────────────────────────────────────────────────────────┘
                        ↓
        ┌───────────────┴───────────────┐
        │                               │
  ┌─────▼─────┐                  ┌─────▼─────┐
  │ Registry-1│ (Leader)          │ Registry-2│ (Standby)
  │ Active    │                   │ Passive   │
  └─────┬─────┘                  └─────┬─────┘
        │                               │
        └───────────────┬───────────────┘
                        ↓
          Kafka (Introspection Events)
```

**Leader Election using Consul**:

```python
# File: src/omninode_bridge/nodes/registry/v1_0_0/node.py

import consul

class NodeBridgeRegistry(NodeEffect, HealthCheckMixin, IntrospectionMixin):
    """Registry with leader election support."""

    async def on_startup(self) -> dict[str, Any]:
        """Start registry with leader election."""

        # ... existing startup logic ...

        # Attempt to acquire leadership
        await self._acquire_leadership()

        # If leader, start consuming introspection events
        if self.is_leader:
            await self.start_consuming()
            logger.info("Registry started as LEADER")
        else:
            # If standby, monitor leadership and wait
            logger.info("Registry started as STANDBY")
            asyncio.create_task(self._monitor_leadership())

        return startup_info

    async def _acquire_leadership(self) -> bool:
        """Attempt to acquire leadership via Consul session."""

        # Create Consul session
        self.session_id = await self.consul_client.session.create(
            name=f"registry-leader-{self.registry_id}",
            ttl="30s",  # Session TTL
            behavior="delete",  # Delete lock on session expiry
        )

        # Attempt to acquire lock
        self.is_leader = await self.consul_client.kv.put(
            key="omninode/registry/leader",
            value=self.registry_id,
            acquire=self.session_id,
        )

        logger.info(f"Leadership acquisition: is_leader={self.is_leader}")

        # Start session renewal task
        if self.is_leader:
            asyncio.create_task(self._renew_leadership_session())

        return self.is_leader

    async def _renew_leadership_session(self) -> None:
        """Renew Consul session to maintain leadership."""

        while self.is_leader and self._running:
            try:
                # Renew session every 15 seconds (TTL is 30s)
                await asyncio.sleep(15)
                await self.consul_client.session.renew(self.session_id)
                logger.debug("Leadership session renewed")
            except Exception as e:
                logger.error(f"Failed to renew leadership session: {e}")
                self.is_leader = False
                break

    async def _monitor_leadership(self) -> None:
        """Monitor leadership and take over if leader fails."""

        while self._running:
            try:
                # Check if leadership is available
                result = await self.consul_client.kv.get("omninode/registry/leader")

                if result[1] is None or result[1]["Session"] is None:
                    # Leadership is available, attempt to acquire
                    logger.info("Leadership available, attempting to acquire")
                    success = await self._acquire_leadership()

                    if success:
                        logger.info("Became LEADER, starting consumer")
                        await self.start_consuming()
                        break

                # Wait before checking again
                await asyncio.sleep(10)

            except Exception as e:
                logger.error(f"Error monitoring leadership: {e}")
                await asyncio.sleep(10)
```

**Work Partitioning** (Alternative to Leader Election):

Instead of leader election, partition work by namespace or node_id hash:

```python
# Registry-1 handles namespaces starting with A-M
# Registry-2 handles namespaces starting with N-Z

async def should_process_introspection(
    self, introspection: ModelNodeIntrospectionEvent
) -> bool:
    """Determine if this registry instance should process the introspection."""

    # Hash node_id to determine partition
    node_hash = hash(introspection.node_id) % self.total_partitions

    # Check if this registry owns the partition
    return node_hash in self.owned_partitions
```

**Timeline**: 4-6 weeks
**Dependencies**: Phase 1b (SPIFFE for policy enforcement), Phase 1c (Lease semantics)
**Estimated Effort**: 160-240 hours

**Success Criteria**:
- ✅ Namespaces isolate tenants logically
- ✅ Policy engine enforces registration rules
- ✅ Idempotency prevents duplicate registrations
- ✅ Multiple registry instances can run concurrently
- ✅ Leader election prevents split-brain scenarios
- ✅ Work partitioning distributes load evenly

---

### ⏳ Phase 3: Federation (PLANNED)

**Timeline**: 6-8 weeks (Not Currently Scheduled)
**Status**: ⏳ **PLANNED** - Future Enhancement

**Goals**:
- Implement federated bridge nodes for cross-network communication
- Add cross-network policy enforcement
- Support blue-green deployment with epoch management
- Enable multi-region and multi-cloud deployments

**Scope**:

#### 1. Federated Bridge Nodes

**Problem**: Single network deployments cannot span multiple datacenters or cloud providers.

**Solution**: Implement **federated bridge nodes** that connect separate network segments.

**Architecture**:

```
┌─────────────────────────────────────┐       ┌─────────────────────────────────────┐
│     Network: omninode-us-west-2     │       │     Network: omninode-eu-west-1     │
├─────────────────────────────────────┤       ├─────────────────────────────────────┤
│                                     │       │                                     │
│  ┌───────────┐  ┌───────────┐     │       │  ┌───────────┐  ┌───────────┐     │
│  │ Registry  │  │Orchestrator│     │       │  │ Registry  │  │Orchestrator│     │
│  └─────┬─────┘  └───────────┘     │       │  └─────┬─────┘  └───────────┘     │
│        │                            │       │        │                            │
│  ┌─────▼─────────────┐             │       │  ┌─────▼─────────────┐             │
│  │ Federation Bridge │◄────────────┼───────┼──►│ Federation Bridge │             │
│  │ (Kafka Connector) │             │       │  │ (Kafka Connector) │             │
│  └───────────────────┘             │       │  └───────────────────┘             │
│                                     │       │                                     │
└─────────────────────────────────────┘       └─────────────────────────────────────┘
           US West 2 Datacenter                        EU West 1 Datacenter
```

**Federation Bridge Implementation**:

```python
# File: src/omninode_bridge/nodes/federation/v1_0_0/node.py

class NodeFederationBridge(NodeEffect):
    """Bridge node for cross-network federation."""

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)

        # Local network configuration
        self.local_network_id = container.value["local_network_id"]
        self.local_kafka_broker = container.value["local_kafka_broker"]

        # Remote network configuration
        self.remote_network_id = container.value["remote_network_id"]
        self.remote_kafka_broker = container.value["remote_kafka_broker"]

        # Create Kafka clients for both networks
        self.local_kafka_client = KafkaClient(bootstrap_servers=self.local_kafka_broker)
        self.remote_kafka_client = KafkaClient(bootstrap_servers=self.remote_kafka_broker)

    async def on_startup(self) -> None:
        """Start federation bridge."""

        # Start bidirectional event forwarding
        asyncio.create_task(self._forward_local_to_remote())
        asyncio.create_task(self._forward_remote_to_local())

        logger.info(
            f"Federation bridge started: {self.local_network_id} ↔ {self.remote_network_id}"
        )

    async def _forward_local_to_remote(self) -> None:
        """Forward events from local network to remote network."""

        async for message in self.local_kafka_client.consume_messages(
            topic=f"{self.local_network_id}.*.onex.evt.node-introspection.v1",
            group_id=f"federation-bridge-{self.local_network_id}-to-{self.remote_network_id}",
        ):
            # Transform event for remote network
            transformed_event = self._transform_event_for_remote(message)

            # Publish to remote network
            await self.remote_kafka_client.publish_with_envelope(
                event_type="node-introspection",
                source_node_id=self.node_id,
                payload=transformed_event,
                topic=f"{self.remote_network_id}.federated.onex.evt.node-introspection.v1",
                metadata={
                    "federated_from": self.local_network_id,
                    "original_event_id": message.event_id,
                },
            )

            logger.debug(
                f"Forwarded event from {self.local_network_id} to {self.remote_network_id}"
            )

    def _transform_event_for_remote(self, event: dict) -> dict:
        """Transform event for remote network (rewrite endpoints, add federation metadata)."""

        # Add federation metadata
        event["metadata"]["federated"] = True
        event["metadata"]["source_network"] = self.local_network_id

        # Rewrite endpoints to use federation bridge as proxy
        if "endpoints" in event["payload"]:
            for endpoint_type, endpoint_url in event["payload"]["endpoints"].items():
                event["payload"]["endpoints"][endpoint_type] = (
                    self._rewrite_endpoint_for_federation(endpoint_url)
                )

        return event

    def _rewrite_endpoint_for_federation(self, endpoint_url: str) -> str:
        """Rewrite endpoint URL to route through federation bridge."""

        # Replace local hostname with federation bridge hostname
        # Example: http://orchestrator-001:8053/health → http://federation-bridge:8080/proxy/orchestrator-001/health

        from urllib.parse import urlparse

        parsed = urlparse(endpoint_url)

        # Extract node_id from hostname
        node_id = parsed.hostname

        # Construct federated endpoint
        return f"http://{self.node_id}:8080/proxy/{node_id}{parsed.path}"
```

#### 2. Cross-Network Policy Enforcement

**Problem**: Federated networks need to enforce policies across network boundaries.

**Solution**: Implement **cross-network policy** with distributed policy evaluation.

**Policy Example**:

```rego
# File: policies/federation.rego

package omninode.federation

# Allow federation if:
# 1. Networks are mutually trusted
# 2. Node type is allowed to be federated
# 3. Cross-network communication is encrypted
allow {
    trusted_networks
    allowed_node_type
    encrypted_communication
}

trusted_networks {
    # Check if source and destination networks trust each other
    input.source_network in data.federation.trusted_networks[input.destination_network]
}

allowed_node_type {
    # Only specific node types can be federated
    input.node_type in data.federation.allowed_federated_node_types
}

encrypted_communication {
    # Ensure communication uses TLS
    input.security.tls_enabled == true
}
```

#### 3. Blue-Green Deployment with Epoch Management

**Problem**: Deploying new versions of services without downtime requires coordination.

**Solution**: Implement **epoch-based blue-green deployment** with gradual traffic shifting.

**Architecture**:

```
Epoch 1 (Blue - Current):
┌─────────────────────────────────────┐
│ Registry-v1, Orchestrator-v1        │
│ deployment_epoch: 2025-10-28T00:00  │
└─────────────────────────────────────┘

Epoch 2 (Green - New):
┌─────────────────────────────────────┐
│ Registry-v2, Orchestrator-v2        │
│ deployment_epoch: 2025-10-28T12:00  │
└─────────────────────────────────────┘

Traffic Shifting:
- 0%   → Epoch 2
- 25%  → Epoch 2 (canary)
- 50%  → Epoch 2
- 100% → Epoch 2 (complete)
```

**Implementation**:

```python
# File: src/omninode_bridge/nodes/registry/v1_0_0/node.py

class NodeBridgeRegistry(NodeEffect, HealthCheckMixin, IntrospectionMixin):
    """Registry with epoch-aware routing."""

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)

        # Deployment epoch
        self.deployment_epoch = container.value.get(
            "deployment_epoch",
            datetime.now(UTC).isoformat()
        )

    async def route_request_to_node(
        self, node_type: str, request_metadata: dict
    ) -> str:
        """Route request to appropriate epoch based on traffic policy."""

        # Get traffic policy (e.g., 50% to epoch 1, 50% to epoch 2)
        traffic_policy = await self._get_traffic_policy()

        # Select epoch based on policy
        selected_epoch = self._select_epoch_by_policy(traffic_policy)

        # Find nodes in selected epoch
        candidate_nodes = [
            node_id
            for node_id, introspection in self.registered_nodes.items()
            if introspection.node_type == node_type
            and introspection.metadata.get("deployment_epoch") == selected_epoch
        ]

        # Round-robin selection
        if not candidate_nodes:
            raise ValueError(f"No nodes found for type={node_type}, epoch={selected_epoch}")

        return candidate_nodes[hash(request_metadata.get("request_id", "")) % len(candidate_nodes)]
```

**Timeline**: 6-8 weeks
**Dependencies**: Phase 2 (Multi-tenancy, policy engine)
**Estimated Effort**: 240-320 hours

**Success Criteria**:
- ✅ Federation bridges connect multiple networks
- ✅ Cross-network events are forwarded bidirectionally
- ✅ Cross-network policies are enforced
- ✅ Blue-green deployments can be performed safely
- ✅ Traffic shifting works as configured
- ✅ Multi-region deployments are supported

---

## Security Model Evolution

### Current State (MVP)

```
┌──────────┐
│ No Trust │  Introspection events are unsigned
└──────────┘  Any node can impersonate any other node
```

### Phase 1b: Trust Foundation

```
┌──────────────────────────────────────┐
│ SPIFFE-Based Identity                │
├──────────────────────────────────────┤
│ - Bootstrap Controller (CA)          │
│ - SVID issuance and rotation         │
│ - Signed introspection events        │
│ - TLS encryption (Kafka + Consul)    │
└──────────────────────────────────────┘
```

### Phase 2: Policy-Based Authorization

```
┌──────────────────────────────────────┐
│ Policy Engine (OPA)                  │
├──────────────────────────────────────┤
│ - Registration authorization         │
│ - Namespace-based policies           │
│ - Capability limits enforcement      │
│ - Audit logging                      │
└──────────────────────────────────────┘
```

### Phase 3: Zero-Trust Network

```
┌──────────────────────────────────────┐
│ Zero-Trust Federation                │
├──────────────────────────────────────┤
│ - Cross-network authentication       │
│ - mTLS for all communication         │
│ - Network-level policies             │
│ - Dynamic trust bundle distribution  │
└──────────────────────────────────────┘
```

## Control Plane vs Data Plane Separation

### Current State (MVP)

**Mixed Responsibilities**:
- Registry handles both node discovery (control) and event consumption (data)
- All events go through same Kafka topics
- No priority separation

### Post-MVP Separation

**Control Plane**:
- Dedicated topics: `{env}.control.{service}.{event_type}.v1`
- High-priority Kafka consumer group
- Operations: Node registration, topology updates, policy changes
- Low latency, high availability requirements

**Data Plane**:
- Dedicated topics: `{env}.data.{service}.{event_type}.v1`
- Standard priority Kafka consumer group
- Operations: Heartbeats, metrics, logs
- High throughput, eventual consistency acceptable

**Implementation**:

```python
# Control plane consumer (high priority)
control_consumer = kafka_client.create_consumer(
    topics=[f"{env}.control.omninode_bridge.onex.evt.node-introspection.v1"],
    group_id="registry-control-plane",
    max_poll_records=10,  # Small batches for low latency
    session_timeout_ms=6000,  # Fast failure detection
)

# Data plane consumer (standard priority)
data_consumer = kafka_client.create_consumer(
    topics=[f"{env}.data.omninode_bridge.onex.evt.node-heartbeat.v1"],
    group_id="registry-data-plane",
    max_poll_records=1000,  # Large batches for throughput
    session_timeout_ms=30000,  # Tolerate temporary failures
)
```

## Failure Modes & Resilience Patterns

### Single Registry Failure

**MVP Behavior**: Registry crash causes loss of new registrations until restart.

**Post-MVP (Phase 2)**: Multi-registry with leader election ensures automatic failover.

**Recovery Time**:
- MVP: Manual restart required (~5 minutes)
- Post-MVP: Automatic failover (~30 seconds)

### Kafka Outage

**MVP Behavior**: Graceful degradation, local caching continues to work.

**Post-MVP (Phase 1c)**: Tombstones and lease semantics ensure topology accuracy.

**Recovery Time**:
- MVP: Full topology resync required (~10 minutes)
- Post-MVP: Incremental catch-up via topology diff stream (~1 minute)

### Consul Outage

**MVP Behavior**: Service discovery fails, nodes cannot register.

**Post-MVP (Phase 1c)**: PostgreSQL fallback for node discovery.

**Recovery Time**:
- MVP: Consul restart required (~5 minutes)
- Post-MVP: Automatic fallback to PostgreSQL (~0 seconds)

### Network Partition

**MVP Behavior**: Nodes in separate partitions cannot discover each other.

**Post-MVP (Phase 3)**: Federation bridges provide cross-partition communication.

**Recovery Time**:
- MVP: Manual intervention required (hours)
- Post-MVP: Automatic bridge reconnection (~5 minutes)

## Summary Roadmap Timeline

| Phase | Duration | Status | Deliverables | Dependencies |
|-------|----------|--------|--------------|--------------|
| **0: MVP Foundation** | Completed | ✅ **COMPLETE** | Bridge nodes, event infrastructure, database, tests | None |
| **1a: Self-Registration** | 1-2 days | ⏳ **PLANNED** | Self-registration, basic introspection | Phase 0 |
| **1b: Trust & Security** | 2 weeks | ⏳ **PLANNED** | SPIFFE, signed events, TLS encryption | 1a |
| **1c: Robustness** | 1 week | ⏳ **PLANNED** | Lease semantics, tombstones, topology diff | 1b |
| **2: Multi-Tenancy** | 4-6 weeks | ⏳ **PLANNED** | Namespaces, policy engine, multi-registry | 1b, 1c |
| **3: Federation** | 6-8 weeks | ⏳ **PLANNED** | Cross-network bridges, blue-green | 2 |

**Phase 0 Status**: ✅ **Completed October 2025** (MVP Foundation Delivered)
**Remaining Timeline**: 12-18 weeks (3-4.5 months) - **Not Currently Scheduled**

## Success Metrics

### Security Metrics

- ✅ 100% of introspection events signed with valid SVIDs
- ✅ 0 unauthorized registrations (blocked by policy engine)
- ✅ 100% of communication encrypted (TLS)
- ✅ <1 second SVID rotation time

### Robustness Metrics

- ✅ 0 stale registrations (automatic TTL expiry)
- ✅ <5 seconds topology update latency (tombstone + diff stream)
- ✅ <30 seconds registry failover time (multi-registry)
- ✅ 99.9% uptime for registry service

### Scalability Metrics

- ✅ Support 1000+ concurrent nodes per namespace
- ✅ Support 100+ namespaces per network
- ✅ Support 10+ federated networks
- ✅ <100ms registration latency (P95)
- ✅ >10,000 registrations/second throughput

## Related Documentation

- **[MVP Registry Self-Registration](./MVP_REGISTRY_SELF_REGISTRATION.md)** - Current implementation (Phase 1a)
- **[Node Introspection Integration Guide](../architecture/two-way-registration/NODE_INTROSPECTION_INTEGRATION.md)** - Introspection system
- **[Bridge Nodes Guide](../guides/BRIDGE_NODES_GUIDE.md)** - Bridge node architecture
- **[SPIFFE Documentation](https://spiffe.io/docs/)** - SPIFFE identity system (external)
- **[Open Policy Agent](https://www.openpolicyagent.org/)** - Policy engine (external)
- **[Consul Documentation](https://www.consul.io/docs)** - Service discovery (external)

---

**Document Version**: 1.0.1
**Last Updated**: October 29, 2025 (Status Review)
**Author**: Planning Documentation Task
**Status**: 📋 **ROADMAP ONLY** - Post-MVP Planning Document

**Note**: This document describes future production enhancements for the registry self-registration system. The MVP foundation (Phase 1 & 2) was completed in October 2025. All items in this document represent **future work** that is not currently scheduled or resourced. Implementation will be considered based on production requirements and business priorities.
