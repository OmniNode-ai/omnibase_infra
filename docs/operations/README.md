> **Navigation**: [Home](../index.md) > Operations

# Operations Documentation

Operational runbooks and guides for deploying and managing omnibase_infra in production environments.

## Runbooks

| Document | Description |
|----------|-------------|
| [Database Index Monitoring](DATABASE_INDEX_MONITORING_RUNBOOK.md) | GIN index performance monitoring, write overhead assessment, and alerting thresholds |
| [DLQ Replay Guide](DLQ_REPLAY_RUNBOOK.md) | Dead Letter Queue replay mechanism: manual procedures, automated design, safety considerations |
| [Event Bus Operations](EVENT_BUS_OPERATIONS_RUNBOOK.md) | Deployment, configuration, monitoring, and troubleshooting for KafkaEventBus |
| [Thread Pool Tuning](THREAD_POOL_TUNING_RUNBOOK.md) | Guide for tuning thread pool configurations in synchronous adapters (Infisical uses async-native SDK) |
| [Infisical Secrets Guide](../guides/INFISICAL_SECRETS_GUIDE.md) | Bootstrap sequence, seed script usage, config prefetch, and local fallback behavior |
| [Database Migrations](../../docker/migrations/README.md) | Migration versioning conventions and execution guidelines |

## Purpose

This directory contains operational documentation for:

- **Tuning Guides**: Performance tuning for different deployment scenarios
- **Troubleshooting**: Common issues and resolution steps
- **Monitoring**: Metrics collection and alerting guidance
- **Deployment**: Production deployment checklists

## Current Operational Commands

```bash
# Seed Infisical from contracts after identity setup
uv run python scripts/seed-infisical.py \
  --contracts-dir src/omnibase_infra/nodes \
  --import-env .env \
  --set-values \
  --execute

# Apply database migrations
uv run python scripts/run-migrations.py

# Create or validate Kafka topics
uv run python scripts/create_kafka_topics.py
uv run python scripts/check_topic_drift.py

# Validate runtime environment variables
uv run python scripts/validate_env.py
uv run python scripts/check_required_env_vars.py
```

## Adding New Runbooks

When creating operational documentation:

1. Use the naming pattern: `<TOPIC>_RUNBOOK.md`
2. Include sections for: Overview, Configuration, Tuning, Monitoring, Troubleshooting
3. Provide concrete examples and recommended settings
4. Link to related source code and documentation

## Recently Fixed Runtime Defects

Quick lookup for defects fixed on `dev` since 2026-06-01. If a deployed runtime version predates the PR listed, apply the update to resolve the symptom.

### Dispatch savings-consumer typed-decode failure

**Symptom**: The `savings-estimation-consumer` service logs decode errors for `ModelSavingsEstimationEvent` payloads and silently drops events; the savings projection table stops updating.

**Root cause**: The consumer used an untyped `dict` decode path for the event payload instead of the declared Pydantic model. A schema change to `ModelSavingsEstimationEvent` added a required field that the untyped path did not populate, causing a `ValidationError` at projection time.

**Fix**: PR merged to `dev`. The consumer now decodes through the typed Pydantic model. Existing rows in `savings_estimation_projection` are not back-filled — only events received after the fix are processed correctly.

**Verification**: After deploying the fix, publish a synthetic `ModelSavingsEstimationEvent` to the savings topic and confirm the `savings_estimation_projection` table receives a new row within the consumer lag window.

---

### Runtime executor `operation_match` import-skip failure

**Symptom**: The runtime executor silently skips handlers whose contracts declare `routing_strategy: operation_match`. The handler is loaded but never invoked; the operation returns an empty result without error.

**Root cause**: The `operation_match` routing path imported `ModelHandlerRoutingEntry` from `omnibase_core.models.routing` but the live contracts declared `ModelHandlerRoutingEntry` from `omnibase_infra.models.routing` — two different classes. The isinstance check failed silently, the operation was not matched, and the handler was skipped.

**Fix**: PR merged to `dev` (commit `b56a3ffa3`). The executor now imports from the canonical infra routing model. OCC #2620 was armed to propagate the fix.

**Verification**: After deploying, trigger an `operation_match`-routed handler (e.g., `register_node` on `node_registry_effect`) and confirm the handler is invoked and the response contains the expected fields.

---

## Related Documentation

- [Architecture Documentation](../architecture/)
- [Validation Documentation](../validation/)
- [Message Dispatch Engine — terminal correlator](../architecture/MESSAGE_DISPATCH_ENGINE.md#terminal-correlator-pattern-omn-13118)
