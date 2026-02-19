> **Navigation**: [Home](../index.md) > [Decisions](README.md) > ADR-008 Realm-Agnostic Topics

# ADR-008: Realm-Agnostic Topic Format

## Status

Accepted

## Date

2026-02-19

## Context

Prior to the 0.4.0 release, ONEX Kafka topic names were constructed by
prepending an environment identifier to a topic suffix inside
`resolve_topic()`:

```python
# Old behavior (pre-0.4.0)
def resolve_topic(suffix: str) -> str:
    env = os.getenv("ONEX_ENV", "dev")
    return f"{env}.{suffix}"
```

A call such as `resolve_topic("archon-intelligence.intelligence.code-analysis-requested.v1")`
would return `"dev.archon-intelligence.intelligence.code-analysis-requested.v1"`
in a development environment and `"prod.archon-intelligence.intelligence.code-analysis-requested.v1"`
in production.

This design encoded deployment context directly into the Kafka topic name.
The problems that emerged as the platform scaled were:

**Cross-environment routing is impossible.** When Claude Code sessions run
in a development environment but the intelligence pipeline runs in a
production cluster, events cannot flow between them through Kafka because
the topics are literally different strings. Every cross-environment flow
required either topic mirroring or a dedicated bridge topic.

**Topic format was inconsistent.** The `dev.*` prefix was appended ad hoc;
there was no structural schema governing the format of the suffix. Topic
names like `dev.archon-intelligence.intelligence.code-analysis-requested.v1`
had no machine-parseable kind marker (`evt`, `cmd`, `intent`, etc.) and no
consistent producer segment.

**Consumer configuration was environment-specific.** Every consumer had to
be reconfigured with the correct `ONEX_ENV` value, and any mismatch silently
produced zero traffic rather than a startup error.

**OMN-2278** addressed all three issues by removing the environment prefix
from `resolve_topic()` and introducing the structured topic suffix format
documented in `docs/standards/TOPIC_TAXONOMY.md`.

## Decision

**Topic suffixes are realm-agnostic. The `resolve_topic()` function returns
the suffix unchanged. Environment isolation is the caller's responsibility,
not the topic name's.**

### New topic suffix format

```text
onex.<kind>.<producer>.<event-name>.v<version>
```

| Component | Rules |
|-----------|-------|
| `onex` | Literal prefix; always lowercase; always first |
| `<kind>` | One of: `evt`, `cmd`, `intent`, `snapshot`, `dlq` |
| `<producer>` | `platform` for infrastructure; domain name for plugins (e.g., `omniintelligence`) |
| `<event-name>` | Lowercase kebab-case |
| `v<version>` | `v1`, `v2`, etc. (no dot-notation) |

Examples:

```text
onex.evt.platform.node-registration.v1
onex.cmd.platform.request-introspection.v1
onex.evt.omniintelligence.llm-call-completed.v1
onex.dlq.platform.node-registration.v1
```

### What changed in `resolve_topic()`

```python
# Before (0.3.x and earlier)
def resolve_topic(suffix: str) -> str:
    env = os.getenv("ONEX_ENV", "dev")
    return f"{env}.{suffix}"

# After (0.4.0+)
def resolve_topic(suffix: str) -> str:
    return suffix  # suffix IS the full topic name
```

The `ONEX_ENV` variable is no longer consulted for topic name construction.
It may still be used by other parts of the platform (e.g., Infisical
environment slug, log context), but it has no effect on Kafka topic routing.

### Suffix validation

All suffix constants are validated at import time via `validate_topic_suffix()`
from `omnibase_core.validation`. An invalid suffix raises `OnexError`
immediately at module load, preventing misconfigured deployments from starting.

Valid suffix pattern:
```text
^onex\.(evt|cmd|intent|snapshot|dlq)\.[a-z][a-z0-9-]*\.[a-z][a-z0-9-]*\.v[0-9]+$
```

### Topic provisioning

`TopicProvisioner` (invoked during runtime startup) creates topics using
suffix strings directly as Kafka topic names — no additional prefix is
applied. The three registry constants that feed `TopicProvisioner` are:

| Constant | Contents |
|----------|----------|
| `ALL_PLATFORM_TOPIC_SPECS` | Platform-reserved `ModelTopicSpec` objects |
| `ALL_INTELLIGENCE_TOPIC_SPECS` | Intelligence domain `ModelTopicSpec` objects |
| `ALL_PROVISIONED_TOPIC_SPECS` | Combined registry used at startup |

### Impact on `EventBusSubcontractWiring`

The `EventBusSubcontractWiring.__init__()` signature gained `service` and
`version` parameters in 0.4.0. These are used for consumer group derivation
(see `adr-consumer-group-naming`) but have no bearing on topic name
construction.

### Environment isolation going forward

Environment isolation is achieved through envelope identity fields
(`ONEX_ENV`, `session_id`, `correlation_id`) rather than topic-name
segregation. Cross-environment consumers that need to filter can inspect
these fields. This aligns with Kafka best practices: topics represent event
streams, not deployment namespaces.

## Consequences

### Positive

- **Cross-environment routing enabled.** A development Claude Code session
  and a production intelligence pipeline can share the same topic string.
  The event envelope carries environment context; consumers decide whether
  to process or skip.
- **Structured suffix format.** The `onex.<kind>.<producer>.<name>.v<N>`
  format is machine-parseable. Dead-letter queue routing derives its topic
  name from the `kind` segment (e.g., `onex.dlq.platform.node-registration.v1`).
- **Import-time validation.** Suffix errors surface at module load rather than
  at runtime message dispatch.
- **Simpler consumer configuration.** Consumers no longer need `ONEX_ENV`
  to construct topic names. A single configuration value covers all
  deployments.
- **E2E tests simplified.** Tests that previously needed different `.env.docker`
  files per environment now use suffix constants directly.

### Negative

- **Breaking change for existing consumers.** Any consumer subscribed to
  `dev.*` or `prod.*` prefixed topics received zero messages after the
  upgrade until reconfigured. The CHANGELOG documents the migration.
- **Historical topic data inaccessible.** Events already in environment-prefixed
  topics are not replayed into the new suffix-named topics. No migration tool
  was provided.
- **Environment isolation is opt-in at the application layer.** Topics no
  longer enforce isolation by name. Teams that relied on environment-prefixed
  topics for access control must implement alternative controls at the Kafka
  ACL level.

### Neutral

- The old format (`dev.archon-intelligence.intelligence.code-analysis-requested.v1`)
  is retired. All active consumers and producers were updated in the 0.4.0
  release cycle.
- `ONEX_ENV` is still available for log context, Infisical environment slug,
  and other non-Kafka purposes.
- The `adr-consumer-group-naming` ADR governs the consumer group ID format,
  which is a separate concern from topic naming.

## References

- **OMN-2278**: Realm-agnostic topic format (remove environment prefix)
- **Topic taxonomy**: `docs/standards/TOPIC_TAXONOMY.md`
- **Suffix constants**: `src/omnibase_infra/topics/platform_topic_suffixes.py`
- **Topic spec model**: `src/omnibase_infra/topics/model_topic_spec.py`
- **Topic resolver**: `src/omnibase_infra/topics/topic_resolver.py`
- **CHANGELOG**: 0.4.0 Breaking Changes — Realm-Agnostic Topics
