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
| [Database Migrations](../../docker/migrations/README.md) | Migration versioning conventions and execution guidelines |

## Purpose

This directory contains operational documentation for:

- **Tuning Guides**: Performance tuning for different deployment scenarios
- **Troubleshooting**: Common issues and resolution steps
- **Monitoring**: Metrics collection and alerting guidance
- **Deployment**: Production deployment checklists

## Adding New Runbooks

When creating operational documentation:

1. Use the naming pattern: `<TOPIC>_RUNBOOK.md`
2. Include sections for: Overview, Configuration, Tuning, Monitoring, Troubleshooting
3. Provide concrete examples and recommended settings
4. Link to related source code and documentation

## Related Documentation

- [Architecture Documentation](../architecture/)
- [Validation Documentation](../validation/)
