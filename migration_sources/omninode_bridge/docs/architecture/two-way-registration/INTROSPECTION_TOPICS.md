# Node Introspection Topics Documentation

## Overview

This document describes the Kafka topics used for ONEX node introspection broadcasting, enabling real-time visibility into node health, capabilities, and runtime state across the OmniNode Bridge ecosystem.

## Topic Summary

| Topic Name | Purpose | Message Rate | Retention | Consumers |
|------------|---------|--------------|-----------|-----------|
| `dev.omninode_bridge.onex.evt.node-introspection.v1` | Node capability and state broadcasts | ~10/min per node | 1 hour | Registry, Monitoring, Analytics |
| `dev.omninode_bridge.onex.evt.registry-request-introspection.v1` | On-demand introspection requests | ~5/min | 1 hour | All active nodes |
| `dev.omninode_bridge.onex.evt.node-heartbeat.v1` | Node health and liveness signals | ~60/min per node | 30 min | Registry, Health Monitor |

## Naming Convention Compliance

All introspection topics follow the ONEX v2.0 naming convention:

```text
{env}.{namespace}.onex.evt.{event-type}.v{version}
```

### Components Breakdown:

- **Environment**: `dev` (development), `staging`, `prod`
- **Namespace**: `omninode_bridge` (service ownership)
- **Protocol**: `onex` (ONEX v2.0 compliance)
- **Message Type**: `evt` (event/notification)
- **Event Type**: Descriptive kebab-case (e.g., `node-introspection`)
- **Version**: `v1` (schema version for evolution)

## Topic Details

### 1. Node Introspection Topic

**Topic**: `dev.omninode_bridge.onex.evt.node-introspection.v1`

**Purpose**: Broadcasts comprehensive node state including capabilities, performance metrics, and operational status.

**Message Schema**:
```json
{
  "node_id": "orchestrator-001",
  "node_type": "orchestrator|reducer|effect|compute",
  "timestamp": "2025-10-03T12:00:00Z",
  "capabilities": {
    "contract_version": "1.0.0",
    "supported_operations": ["workflow_coordination", "fsm_management"],
    "performance_profile": {
      "avg_latency_ms": 45,
      "throughput_per_sec": 150,
      "memory_usage_mb": 256
    }
  },
  "state": {
    "status": "healthy|degraded|unhealthy",
    "active_workflows": 12,
    "queue_depth": 45,
    "last_error": null
  },
  "subcontracts": [
    {
      "type": "FSMSubcontract",
      "state_count": 4,
      "transition_count": 8
    },
    {
      "type": "RoutingSubcontract",
      "service_count": 3,
      "route_count": 5
    }
  ],
  "metadata": {
    "environment": "dev",
    "namespace": "omninode.bridge",
    "version": "0.1.0"
  }
}
```

**Partitioning Strategy**:
- **Key**: `node_id` (ensures ordered delivery per node)
- **Partitions**: 3 (balanced load distribution)
- **Replication**: 1 (development), 3 (production)

**Expected Message Rate**:
- Development: ~10 messages/min per node
- Production: ~5 messages/min per node (optimized polling)

**Retention Policy**:
- **Duration**: 1 hour (development), 24 hours (staging), 7 days (production)
- **Cleanup**: Delete policy (automatic cleanup after retention)
- **Compression**: Snappy (balance between CPU and storage)

**Consumer Groups**:
1. **`registry-introspection-collectors`**: Node Registry service
2. **`monitoring-introspection-agents`**: Prometheus/Grafana exporters
3. **`analytics-introspection-processors`**: Trend analysis and capacity planning

### 2. Registry Request Introspection Topic

**Topic**: `dev.omninode_bridge.onex.evt.registry-request-introspection.v1`

**Purpose**: Enables on-demand introspection requests from Node Registry to specific nodes or node types.

**Message Schema**:
```json
{
  "request_id": "req-123e4567-e89b",
  "timestamp": "2025-10-03T12:00:00Z",
  "target": {
    "node_id": "orchestrator-001",  // Optional: specific node
    "node_type": "orchestrator",     // Optional: all nodes of type
    "namespace": "omninode.bridge"   // Optional: namespace filter
  },
  "requested_data": [
    "capabilities",
    "performance_metrics",
    "active_workflows",
    "subcontract_state"
  ],
  "response_topic": "dev.omninode_bridge.onex.evt.node-introspection.v1",
  "timeout_ms": 5000,
  "metadata": {
    "requester": "node-registry",
    "correlation_id": "corr-456e7890-abc"
  }
}
```

**Partitioning Strategy**:
- **Key**: `target.node_type` (distributes requests by node type)
- **Partitions**: 3 (balanced request distribution)
- **Replication**: 1 (development), 3 (production)

**Expected Message Rate**:
- Development: ~5 messages/min (on-demand queries)
- Production: ~10 messages/min (registry refresh cycles)

**Retention Policy**:
- **Duration**: 1 hour (short retention for request/response)
- **Cleanup**: Delete policy
- **Compression**: Snappy

**Consumer Groups**:
1. **`node-introspection-handlers-orchestrator`**: Orchestrator nodes
2. **`node-introspection-handlers-reducer`**: Reducer nodes
3. **`node-introspection-handlers-effect`**: Effect nodes
4. **`node-introspection-handlers-compute`**: Compute nodes

**Response Pattern**:
Nodes respond by publishing to `node-introspection` topic with `correlation_id` matching the request.

### 3. Node Heartbeat Topic

**Topic**: `dev.omninode_bridge.onex.evt.node-heartbeat.v1`

**Purpose**: Lightweight health signals for node liveness detection and failure recovery.

**Message Schema**:
```json
{
  "node_id": "orchestrator-001",
  "node_type": "orchestrator",
  "timestamp": "2025-10-03T12:00:00Z",
  "status": "alive|degraded|shutting_down",
  "uptime_seconds": 3600,
  "metrics": {
    "cpu_percent": 45.2,
    "memory_mb": 256,
    "active_tasks": 12,
    "error_rate": 0.001
  },
  "metadata": {
    "environment": "dev",
    "namespace": "omninode.bridge",
    "version": "0.1.0"
  }
}
```

**Partitioning Strategy**:
- **Key**: `node_id` (ordered heartbeats per node)
- **Partitions**: 3 (high throughput support)
- **Replication**: 1 (development), 3 (production)

**Expected Message Rate**:
- Development: ~60 messages/min per node (1 heartbeat/sec)
- Production: ~20 messages/min per node (1 heartbeat/3sec, optimized)

**Retention Policy**:
- **Duration**: 30 minutes (short-lived health signals)
- **Cleanup**: Delete policy (aggressive cleanup)
- **Compression**: Snappy

**Consumer Groups**:
1. **`health-monitors`**: Node Registry health tracking
2. **`failure-detectors`**: Automatic failover systems
3. **`alerting-systems`**: PagerDuty/Slack integrations

**Failure Detection**:
- **Missing Heartbeat Threshold**: 3 missed heartbeats (9 seconds in prod)
- **Degraded Status**: Triggered by high error rate or resource utilization
- **Automatic Recovery**: Node restart or traffic redirection

## Monitoring and Metrics

### Prometheus Integration

All introspection topics are automatically monitored through the existing `kafka-exporter` job in Prometheus:

```yaml
# monitoring/prometheus.yml (existing configuration)
- job_name: 'kafka-exporter'
  static_configs:
    - targets: ['kafka-exporter:9308']
  scrape_interval: 30s
```

**Key Metrics Tracked**:
- `kafka_topic_partition_current_offset{topic="dev.omninode_bridge.onex.evt.node-introspection.v1"}`
- `kafka_topic_partition_lag{topic="dev.omninode_bridge.onex.evt.node-introspection.v1"}`
- `kafka_consumer_group_lag{topic="dev.omninode_bridge.onex.evt.node-heartbeat.v1"}`

### Grafana Dashboards

**Recommended Dashboards**:
1. **Node Introspection Overview**
   - Active nodes by type
   - Capability distribution
   - Performance trends

2. **Heartbeat Health Monitoring**
   - Heartbeat frequency per node
   - Missing heartbeat alerts
   - Node status distribution

3. **Registry Request Analytics**
   - Request volume over time
   - Response latency (P50, P95, P99)
   - Failed request rate

### Alert Rules

**Critical Alerts**:
```yaml
# Example Prometheus alert rules
groups:
  - name: introspection_alerts
    interval: 30s
    rules:
      - alert: NodeHeartbeatMissing
        expr: |
          (time() - kafka_topic_partition_current_offset{
            topic="dev.omninode_bridge.onex.evt.node-heartbeat.v1"
          }) > 30
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Node heartbeat missing for {{ $labels.node_id }}"

      - alert: IntrospectionConsumerLag
        expr: |
          kafka_consumer_group_lag{
            topic=~".*introspection.*"
          } > 1000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High consumer lag on introspection topic"
```

## Performance Characteristics

### Throughput Targets

| Topic | Dev Throughput | Prod Throughput | Latency Target |
|-------|---------------|-----------------|----------------|
| node-introspection | ~50 msg/sec | ~100 msg/sec | < 50ms P95 |
| registry-request-introspection | ~10 msg/sec | ~20 msg/sec | < 100ms P95 |
| node-heartbeat | ~200 msg/sec | ~500 msg/sec | < 10ms P95 |

### Resource Usage

**Estimated Storage** (per environment):
- **Development**: ~50MB/day (1-hour retention)
- **Staging**: ~1.2GB/day (24-hour retention)
- **Production**: ~8GB/day (7-day retention with compression)

**Network Bandwidth**:
- **Peak**: ~5Mbps (all introspection topics combined)
- **Average**: ~2Mbps (steady-state operation)

## Implementation Best Practices

### Producer Guidelines

1. **Use Consistent Partitioning**:
```python
# Always use node_id as partition key for ordered delivery
producer.produce(
    topic="dev.omninode_bridge.onex.evt.node-introspection.v1",
    key=node_id.encode('utf-8'),
    value=json.dumps(introspection_data).encode('utf-8')
)
```

2. **Include Correlation IDs**:
```python
# For request/response patterns
introspection_data = {
    "node_id": node_id,
    "correlation_id": request.correlation_id,  # Match request
    # ... rest of data
}
```

3. **Implement Exponential Backoff**:
```python
# For heartbeat failures
async def send_heartbeat_with_retry():
    for attempt in range(max_retries):
        try:
            await producer.produce(heartbeat_topic, heartbeat_data)
            break
        except KafkaException:
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

### Consumer Guidelines

1. **Use Pattern Subscriptions**:
```python
# Subscribe to all introspection events
consumer.subscribe([
    r"dev\.omninode_bridge\.onex\.evt\.(node-introspection|registry-request|node-heartbeat)\.v1"
])
```

2. **Implement Idempotent Processing**:
```python
# Handle duplicate messages gracefully
async def process_introspection(message):
    if await is_already_processed(message.key):
        return  # Skip duplicate

    await store_introspection_data(message)
    await mark_as_processed(message.key)
```

3. **Monitor Consumer Lag**:
```python
# Alert on high lag
if consumer_lag > 1000:
    logger.warning(f"High consumer lag: {consumer_lag}")
    metrics.increment("introspection.consumer.lag.high")
```

## Migration and Versioning

### Schema Evolution

When updating introspection schemas:

1. **Create v2 Topic**: `dev.omninode_bridge.onex.evt.node-introspection.v2`
2. **Dual Publishing**: Publish to both v1 and v2 during migration
3. **Migrate Consumers**: Update consumers to read from v2
4. **Deprecate v1**: Remove v1 after 30-day migration window

### Backward Compatibility

All schema changes must maintain backward compatibility:
- **Additive Changes**: New fields are optional
- **Deprecated Fields**: Maintained for 2 major versions
- **Breaking Changes**: Require new major version (v2, v3, etc.)

## Security Considerations

### Access Control

**Production ACLs**:
```bash
# Producers (nodes only)
kafka-acls --add --allow-principal User:node-service \
  --operation Write \
  --topic dev.omninode_bridge.onex.evt.node-introspection.v1

# Consumers (registry and monitoring only)
kafka-acls --add --allow-principal User:node-registry \
  --operation Read \
  --topic dev.omninode_bridge.onex.evt.node-introspection.v1
```

### Data Privacy

- **PII**: No personally identifiable information in introspection data
- **Secrets**: Never include credentials, tokens, or connection strings
- **Sensitive Metrics**: Obfuscate internal IPs and resource paths

## Testing Strategy

### Integration Tests

```python
@pytest.mark.asyncio
async def test_node_introspection_flow():
    """Test complete introspection request/response cycle."""

    # 1. Send registry request
    request = create_introspection_request(node_id="test-node-001")
    await producer.produce(registry_request_topic, request)

    # 2. Node responds with introspection data
    response = await consumer.poll(timeout=5.0)
    assert response.correlation_id == request.request_id

    # 3. Verify introspection data
    assert response.capabilities["contract_version"] == "1.0.0"
    assert response.state["status"] == "healthy"
```

### Performance Tests

```python
@pytest.mark.benchmark
async def test_heartbeat_throughput():
    """Verify heartbeat topic can handle 500 msg/sec."""

    start_time = time.time()
    for i in range(5000):
        await producer.produce(heartbeat_topic, create_heartbeat())

    duration = time.time() - start_time
    throughput = 5000 / duration

    assert throughput >= 500, f"Throughput {throughput} msg/sec below target"
```

## Troubleshooting

### Common Issues

**1. Missing Heartbeats**
- **Symptom**: Node appears offline despite being active
- **Cause**: Network partition or producer failure
- **Solution**: Check network connectivity, restart heartbeat producer

**2. High Consumer Lag**
- **Symptom**: Delayed introspection data in registry
- **Cause**: Slow consumer processing or insufficient partitions
- **Solution**: Scale consumers, increase partitions (requires topic recreation)

**3. Duplicate Introspection Events**
- **Symptom**: Registry shows duplicate node entries
- **Cause**: Producer retries without idempotency
- **Solution**: Implement idempotent consumers with message deduplication

### Debug Commands

```bash
# Check topic status
rpk topic describe dev.omninode_bridge.onex.evt.node-introspection.v1

# Monitor consumer lag
rpk group describe registry-introspection-collectors

# Tail live messages
rpk topic consume dev.omninode_bridge.onex.evt.node-heartbeat.v1 \
  --format json | jq .
```

## References

- [ONEX Architecture Patterns](./ONEX_ARCHITECTURE_PATTERNS_COMPLETE.md)
- [Kafka Topic Strategy](./planning/KAFKA_TOPIC_STRATEGY.md)
- [Node Registry Design](./NODE_REGISTRY_DESIGN.md)
- [Bridge Nodes Guide](./BRIDGE_NODES_GUIDE.md)

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2025-10-03 | Initial introspection topics implementation |

---

**Maintained by**: OmniNode Bridge Team
**Last Updated**: 2025-10-03
**Status**: Active
