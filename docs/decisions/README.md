> **Navigation**: [Home](../index.md) > Decisions (ADRs)

# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records documenting significant design decisions in ONEX Infrastructure.

> **Note**: ADRs explain *why* decisions were made. For the resulting coding rules and standards, see [CLAUDE.md](../../CLAUDE.md) which is the authoritative source for all development guidelines.

## What is an ADR?

An ADR captures:
- **Context**: The situation that led to needing a decision
- **Decision**: What was decided
- **Consequences**: The trade-offs and implications

ADRs are immutable once accepted. Superseded decisions are marked but not deleted.

## ADR Index

### Numbered ADRs

| ADR | Title | Status |
|-----|-------|--------|
| [ADR-001](adr-001-graceful-shutdown-drain-period.md) | Graceful Shutdown Drain Period | Accepted |
| [ADR-002](adr-002-enum-message-category-node-output-separation.md) | Enum Message Category / Node Output Separation | Accepted |
| [ADR-003](adr-003-remove-handler-health-checks.md) | Remove Handler Health Checks | Accepted |
| [ADR-004](adr-004-performance-baseline-thresholds.md) | Performance Baseline Thresholds | Accepted |
| [ADR-005](adr-005-denormalize-capability-fields-in-registration-projections.md) | Denormalize Capability Fields in Registration Projections | Accepted |
| [ADR-006](adr-006-message-dispatch-engine-canonical-routing.md) | MessageDispatchEngine as Canonical Consumer Routing Pattern | Accepted |
| [ADR-007](adr-007-vault-to-infisical-migration.md) | Vault to Infisical Migration | Accepted |
| [ADR-008](adr-008-realm-agnostic-topics.md) | Realm-Agnostic Topic Format | Accepted |
| [ADR-009](adr-009-llm-cost-tracking-spi.md) | LLM Cost Tracking at the Infrastructure Layer | Accepted |

### Topic-Based ADRs

| ADR | Title | Category |
|-----|-------|----------|
| [adr-any-type-pydantic-workaround](adr-any-type-pydantic-workaround.md) | Any Type Pydantic Workaround | Type System |
| [adr-canonical-publish-interface-policy](adr-canonical-publish-interface-policy.md) | Canonical Publish Interface Policy | Architecture |
| [adr-consumer-group-naming](adr-consumer-group-naming.md) | Kafka Consumer Group Naming Convention | Event Bus |
| [adr-cryptography-upgrade-46](adr-cryptography-upgrade-46.md) | Cryptography Upgrade to 46.x | Security |
| [adr-custom-bool-result-models](adr-custom-bool-result-models.md) | Custom `__bool__` for Result Models | Type System |
| [adr-dsn-validation-strengthening](adr-dsn-validation-strengthening.md) | DSN Validation Strengthening | Security |
| [adr-enum-message-category-vs-node-output-type](adr-enum-message-category-vs-node-output-type.md) | Enum Message Category vs Node Output Type | Type System |
| [adr-error-context-factory-pattern](adr-error-context-factory-pattern.md) | Error Context Factory Pattern | Error Handling |
| [adr-handler-contract-schema-evolution](adr-handler-contract-schema-evolution.md) | Handler Contract Schema Evolution | Contracts |
| [adr-handler-plugin-loader-security](adr-handler-plugin-loader-security.md) | Handler Plugin Loader Security | Security |
| [adr-handler-type-vs-handler-category](adr-handler-type-vs-handler-category.md) | Handler Type vs Handler Category | Architecture |
| [adr-projector-composite-key-upsert-workaround](adr-projector-composite-key-upsert-workaround.md) | Projector Composite Key Upsert Workaround | Data Layer |
| [adr-protocol-design-guidelines](adr-protocol-design-guidelines.md) | Protocol Design Guidelines | Architecture |
| [adr-soft-validation-env-parsing](adr-soft-validation-env-parsing.md) | Soft Validation for Environment Parsing | Configuration |

## Categories

| Category | Description | ADRs |
|----------|-------------|------|
| **Type System** | Type safety, generics, Pydantic patterns | 4 |
| **Security** | Authentication, encryption, plugin security | 4 |
| **Architecture** | Node types, handlers, protocols, runtime lifecycle | 6 |
| **Error Handling** | Error patterns, context, recovery | 1 |
| **Contracts** | Contract format, evolution | 1 |
| **Data Layer** | Persistence, projections, cost tracking | 3 |
| **Configuration** | Environment, settings, secret management | 2 |
| **Performance** | Baselines, thresholds | 1 |
| **Event Bus** | Kafka, consumer groups, event routing, topic naming | 2 |

**Note**: All numbered ADRs are included in these category counts.
- ADR-001: Architecture (graceful shutdown)
- ADR-002: Type System (enum separation)
- ADR-003: Architecture (remove handler health checks)
- ADR-004: Performance (baseline thresholds)
- ADR-005: Data Layer (registration projections)
- ADR-006: Architecture (MessageDispatchEngine canonical routing)
- ADR-007: Security + Configuration (Vault to Infisical migration)
- ADR-008: Event Bus (realm-agnostic topic format)
- ADR-009: Data Layer (LLM cost tracking SPI)
- The canonical publish interface policy ADR is also Architecture.

## Writing ADRs

New ADRs should:
1. Use kebab-case filename: `adr-<topic>.md`
2. Include standard sections: Status, Context, Decision, Consequences
3. Be reviewed before merging
4. Never be deleted (mark superseded instead)

### Template

```markdown
# ADR: <Title>

## Status

Accepted | Superseded by ADR-XXX | Deprecated

## Context

What is the situation that requires a decision?

## Decision

What was decided?

## Consequences

### Positive
- Benefit 1
- Benefit 2

### Negative
- Trade-off 1
- Trade-off 2

### Neutral
- Side effect 1
```
