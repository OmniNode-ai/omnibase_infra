> **Navigation**: [Home](../index.md) > [Operations](../operations/README.md) > Snapshot Topic Configuration

# Snapshot Topic Configuration

Operational guide for creating and managing the Redpanda compacted topic used by the ONEX registration snapshot publisher.

## Overview

The snapshot publisher writes compacted snapshots of node registration state to a Kafka topic so that dashboards and orchestrators can perform fast O(1) reads without replaying the event log. The topic uses **log compaction** (`cleanup.policy=compact`) so that only the latest snapshot per node is retained.

**Topic Name**: `onex.snapshot.platform.registration-snapshots.v1`

**Programmatic Config Source**: `src/omnibase_infra/models/projection/model_snapshot_topic_config.py` (`ModelSnapshotTopicConfig`)

**Infrastructure**: Redpanda (Kafka-compatible) running on `192.168.86.200:29092`

## Topic Configuration Reference

These values are sourced from `ModelSnapshotTopicConfig.default()`:

| Property | Value | Description |
|----------|-------|-------------|
| `topic` | `onex.snapshot.platform.registration-snapshots.v1` | Full topic name per ONEX topic taxonomy |
| `partitions` | 12 | Matches projection partitioning |
| `replication_factor` | 3 | High durability (use 1 for single-node dev) |
| `cleanup.policy` | `compact` | Log compaction retains only latest per key |
| `min.compaction.lag.ms` | `60000` (1 minute) | Minimum time before a message can be compacted |
| `max.compaction.lag.ms` | `300000` (5 minutes) | Maximum time before compaction is forced |
| `segment.bytes` | `104857600` (100 MB) | Segment size; smaller = more frequent compaction |
| `retention.ms` | `-1` (infinite) | Required for compacted topics |
| `min.insync.replicas` | 2 | Minimum in-sync replicas for writes (use 1 for single-node dev) |

## Key Format and Compaction Behavior

Kafka compaction uses the message **key** to determine which records to retain. For this topic, the key is the **node_id as a UUID string only** -- not `domain:entity_id`.

Example key:

```
550e8400-e29b-41d4-a716-446655440000
```

This means compaction retains exactly one record per node. When a new snapshot is published for a given node_id, the previous record for that key will be removed during the next compaction cycle. Publishing a **tombstone** (null value) for a key causes the key to be deleted entirely during compaction.

The key format is defined by `ModelRegistrationSnapshot.to_kafka_key()` which returns `str(self.entity_id)`.

## Creating the Topic

### Production (multi-broker cluster)

```bash
rpk topic create onex.snapshot.platform.registration-snapshots.v1 \
  --brokers 192.168.86.200:29092 \
  --partitions 12 \
  --replicas 3 \
  --topic-config cleanup.policy=compact \
  --topic-config min.compaction.lag.ms=60000 \
  --topic-config max.compaction.lag.ms=300000 \
  --topic-config segment.bytes=104857600 \
  --topic-config retention.ms=-1 \
  --topic-config min.insync.replicas=2
```

### Development (single-node Redpanda)

```bash
rpk topic create onex.snapshot.platform.registration-snapshots.v1 \
  --brokers 192.168.86.200:29092 \
  --partitions 12 \
  --replicas 1 \
  --topic-config cleanup.policy=compact \
  --topic-config min.compaction.lag.ms=60000 \
  --topic-config max.compaction.lag.ms=300000 \
  --topic-config segment.bytes=104857600 \
  --topic-config retention.ms=-1 \
  --topic-config min.insync.replicas=1
```

## Development vs Production

| Setting | Development (single-node) | Production (multi-broker) |
|---------|---------------------------|---------------------------|
| `--replicas` | `1` | `3` |
| `min.insync.replicas` | `1` | `2` |
| `--partitions` | `12` | `12` (increase if needed for throughput) |
| All other config | Same | Same |

Single-node Redpanda cannot honor `replication_factor > 1` or `min.insync.replicas > 1` because there is only one broker. Using values greater than 1 for either setting will cause topic creation to fail on a single-node cluster. Always use `1` for both when targeting a single-node development instance.

## Verifying Topic Configuration

### Check that the topic exists

```bash
rpk topic list --brokers 192.168.86.200:29092
```

### Describe topic configuration

```bash
rpk topic describe onex.snapshot.platform.registration-snapshots.v1 \
  --brokers 192.168.86.200:29092
```

This shows partition count, replication factor, and all topic-level config overrides. Verify that:

- `cleanup.policy` is `compact`
- `min.compaction.lag.ms` is `60000`
- `max.compaction.lag.ms` is `300000`
- `segment.bytes` is `104857600`
- `retention.ms` is `-1`
- `min.insync.replicas` matches your environment (1 for dev, 2 for prod)

### Describe topic configuration (config only)

```bash
rpk topic describe onex.snapshot.platform.registration-snapshots.v1 \
  --brokers 192.168.86.200:29092 \
  -c
```

The `-c` flag shows only the configuration section, which is useful for quick verification.

## Debugging: Listing Messages

### Read all current snapshots from the topic

```bash
rpk topic consume onex.snapshot.platform.registration-snapshots.v1 \
  --brokers 192.168.86.200:29092 \
  --offset start \
  --format '%k\t%v\n'
```

This prints each message as `<key>\t<value>`. Tombstones appear as a key with an empty value.

### Read a limited number of messages

```bash
rpk topic consume onex.snapshot.platform.registration-snapshots.v1 \
  --brokers 192.168.86.200:29092 \
  --offset start \
  --num 10 \
  --format '%k\t%v\n'
```

### Read messages with full metadata (partition, offset, timestamp)

```bash
rpk topic consume onex.snapshot.platform.registration-snapshots.v1 \
  --brokers 192.168.86.200:29092 \
  --offset start \
  --num 10 \
  --format 'partition:%p offset:%o timestamp:%d{ms} key:%k value:%v\n'
```

### Read from a specific partition

```bash
rpk topic consume onex.snapshot.platform.registration-snapshots.v1 \
  --brokers 192.168.86.200:29092 \
  --partitions 0 \
  --offset start \
  --num 10
```

## Checking Consumer Group Offsets

### List all consumer groups

```bash
rpk group list --brokers 192.168.86.200:29092
```

### Describe a specific consumer group

```bash
rpk group describe <group-id> --brokers 192.168.86.200:29092
```

This shows per-partition current offset, log end offset, and **lag** for each topic the group is consuming. High lag on the snapshot topic may indicate a slow consumer or a consumer that has fallen behind.

### Example: check lag for the snapshot reader group

```bash
rpk group describe snapshot-reader --brokers 192.168.86.200:29092
```

Output columns of interest:

| Column | Description |
|--------|-------------|
| `CURRENT-OFFSET` | Last committed offset for this partition |
| `LOG-END-OFFSET` | Latest offset in the partition |
| `LAG` | Difference between log-end and current offset |

A lag of 0 means the consumer is fully caught up.

## Environment Variable Overrides

`ModelSnapshotTopicConfig.default()` applies environment variable overrides on top of the compiled defaults. Set these in your `.env` or shell environment to customize without changing code:

| Environment Variable | Config Field | Type | Default |
|---------------------|--------------|------|---------|
| `SNAPSHOT_TOPIC` | `topic` | string | `onex.snapshot.platform.registration-snapshots.v1` |
| `SNAPSHOT_PARTITION_COUNT` | `partition_count` | int | `12` |
| `SNAPSHOT_REPLICATION_FACTOR` | `replication_factor` | int | `3` |
| `SNAPSHOT_MIN_COMPACTION_LAG_MS` | `min_compaction_lag_ms` | int | `60000` |
| `SNAPSHOT_MAX_COMPACTION_LAG_MS` | `max_compaction_lag_ms` | int | `300000` |
| `SNAPSHOT_SEGMENT_BYTES` | `segment_bytes` | int | `104857600` |
| `SNAPSHOT_RETENTION_MS` | `retention_ms` | int | `-1` |
| `SNAPSHOT_MIN_INSYNC_REPLICAS` | `min_insync_replicas` | int | `2` |

`cleanup_policy` is intentionally **not overridable** via environment variable because snapshot topics must always use compaction.

### Example .env overrides for development

```bash
SNAPSHOT_REPLICATION_FACTOR=1
SNAPSHOT_MIN_INSYNC_REPLICAS=1
```

## Programmatic Topic Creation

The `ModelSnapshotTopicConfig` model provides a `to_kafka_config()` method that outputs Kafka-compatible configuration suitable for the AdminClient API:

```python
from omnibase_infra.models.projection import ModelSnapshotTopicConfig

config = ModelSnapshotTopicConfig.default()

# Get Kafka-compatible config dict
kafka_config = config.to_kafka_config()
# {
#     "cleanup.policy": "compact",
#     "min.compaction.lag.ms": "60000",
#     "max.compaction.lag.ms": "300000",
#     "segment.bytes": "104857600",
#     "retention.ms": "-1",
#     "min.insync.replicas": "2",
# }

# Use with aiokafka AdminClient or similar
print(f"Topic: {config.topic}")
print(f"Partitions: {config.partition_count}")
print(f"Replication Factor: {config.replication_factor}")
```

You can also load configuration from a YAML file:

```python
from pathlib import Path

config = ModelSnapshotTopicConfig.from_yaml(Path("config/snapshot_topic.yaml"))
```

## Compaction Timing

```
|--- min_compaction_lag_ms (60s) ---|--- max_compaction_lag_ms (300s) ---|
|<-- Cannot compact here -->|<-- May compact -->|<-- Must compact -->|
```

- **min_compaction_lag_ms** (60 seconds): A message must exist for at least this long before it can be compacted away. This protects consumers that are slightly behind from missing a snapshot version that was quickly superseded.

- **max_compaction_lag_ms** (5 minutes): After this threshold, Redpanda will force compaction even if the log cleaner has not scheduled it. This keeps the topic size bounded and ensures stale snapshots are cleaned up promptly.

- **segment.bytes** (100 MB): Compaction operates on closed segments. Smaller segments close more frequently, leading to more frequent compaction but higher I/O overhead. The 100 MB default balances compaction frequency against disk I/O for moderate write volumes.

## Modifying Topic Configuration After Creation

If you need to change configuration on an existing topic (for example, adjusting compaction lag):

```bash
rpk topic alter-config onex.snapshot.platform.registration-snapshots.v1 \
  --brokers 192.168.86.200:29092 \
  --set min.compaction.lag.ms=120000 \
  --set max.compaction.lag.ms=600000
```

Partition count can be increased (but never decreased):

```bash
rpk topic add-partitions onex.snapshot.platform.registration-snapshots.v1 \
  --brokers 192.168.86.200:29092 \
  --num 24
```

## Deleting the Topic

To delete and recreate the topic (for example, during development reset):

```bash
rpk topic delete onex.snapshot.platform.registration-snapshots.v1 \
  --brokers 192.168.86.200:29092
```

Snapshots can always be rebuilt from projections, so deleting the snapshot topic does not cause data loss.

## Related Documentation

- [Snapshot Publishing Architecture](../architecture/SNAPSHOT_PUBLISHING.md)
- [Event Bus Operations Runbook](../operations/EVENT_BUS_OPERATIONS_RUNBOOK.md)
- [ModelSnapshotTopicConfig](../../src/omnibase_infra/models/projection/model_snapshot_topic_config.py) (programmatic config source)
- [SnapshotPublisherRegistration](../../src/omnibase_infra/projectors/snapshot_publisher_registration.py) (publisher implementation)
- [Platform Topic Suffixes](../../src/omnibase_infra/topics/platform_topic_suffixes.py) (topic name definition)

### Tickets

- **OMN-1932**: Wire snapshot publisher (Phase 3)
- **OMN-947** (F2): Snapshot Publishing
- **OMN-944** (F1): Implement Registration Projection Schema
