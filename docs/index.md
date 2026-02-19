> **Navigation**: Home (You are here)

# ONEX Infrastructure Documentation

Welcome to the ONEX Infrastructure (`omnibase_infra`) documentation.

## Documentation Authority Model

| Source | Purpose | Authority |
|--------|---------|-----------|
| **[CLAUDE.md](../CLAUDE.md)** | Coding standards, rules, architecture invariants, handler system, intent model, error hierarchy, Pydantic patterns | **Authoritative** - definitive rules for agents and developers |
| **docs/** | Explanations, examples, tutorials, runbooks, ADRs, design context | Supplementary - context and how-to guidance |

**When in conflict, CLAUDE.md takes precedence.** This separation ensures:
- Rules are concise and enforceable in CLAUDE.md
- Documentation provides depth without bloating the rules file
- Agents and developers have a single source of truth for coding standards

**Quick Reference:**
- Need a rule? Check [CLAUDE.md](../CLAUDE.md)
- Need an explanation? Check [docs/](.)
- Need an example? Check [docs/patterns/](patterns/README.md) or [docs/guides/](guides/README.md)

---

## Quick Navigation

| If you want to... | Document | Location |
|-------------------|----------|----------|
| Get running in 5 minutes | Quick Start Guide | [getting-started/quickstart.md](getting-started/quickstart.md) |
| Understand the four-node architecture | Architecture Overview | [architecture/overview.md](architecture/overview.md) |
| See the complete registration flow end-to-end | Registration Workflow | [architecture/REGISTRATION_WORKFLOW.md](architecture/REGISTRATION_WORKFLOW.md) |
| Write a contract.yaml | Contract Reference | [reference/contracts.md](reference/contracts.md) |
| Look up node archetypes (EFFECT/COMPUTE/REDUCER/ORCHESTRATOR) | Node Archetypes | [reference/node-archetypes.md](reference/node-archetypes.md) |
| Author a new handler | Handler Authoring Guide | [guides/HANDLER_AUTHORING_GUIDE.md](guides/HANDLER_AUTHORING_GUIDE.md) |
| Integrate Infisical secrets | Infisical Secrets Guide | [guides/INFISICAL_SECRETS_GUIDE.md](guides/INFISICAL_SECRETS_GUIDE.md) |
| Expose node operations as MCP tools | MCP Integration Guide | [guides/MCP_INTEGRATION_GUIDE.md](guides/MCP_INTEGRATION_GUIDE.md) |
| Handle errors with proper context | Error Handling Patterns | [patterns/error_handling_patterns.md](patterns/error_handling_patterns.md) |
| Add circuit breaker protection | Circuit Breaker Implementation | [patterns/circuit_breaker_implementation.md](patterns/circuit_breaker_implementation.md) |
| Inject dependencies via container | Container DI Pattern | [patterns/container_dependency_injection.md](patterns/container_dependency_injection.md) |
| Understand why a decision was made | ADR Index | [decisions/README.md](decisions/README.md) |
| Operate Kafka in production | Event Bus Operations Runbook | [operations/EVENT_BUS_OPERATIONS_RUNBOOK.md](operations/EVENT_BUS_OPERATIONS_RUNBOOK.md) |
| Run or fix validation | Validation Framework | [validation/README.md](validation/README.md) |
| Understand Kafka topic naming | Topic Taxonomy | [standards/TOPIC_TAXONOMY.md](standards/TOPIC_TAXONOMY.md) |
| Migrate from wire_default_handlers() | Migration Guide | [migration/MIGRATION_WIRE_DEFAULT_HANDLERS.md](migration/MIGRATION_WIRE_DEFAULT_HANDLERS.md) |

---

## Documentation Structure

### Architecture

Understand how ONEX infrastructure is designed.

| Document | Description |
|----------|-------------|
| [overview.md](architecture/overview.md) | High-level system architecture with diagrams |
| [CURRENT_NODE_ARCHITECTURE.md](architecture/CURRENT_NODE_ARCHITECTURE.md) | Detailed node architecture: declarative nodes, handler routing, container DI |
| [REGISTRATION_WORKFLOW.md](architecture/REGISTRATION_WORKFLOW.md) | Complete 2-way registration flow: all 4 nodes, FSM states, Kafka topics, intent construction, error paths, E2E test coverage |
| [EVENT_BUS_INTEGRATION_GUIDE.md](architecture/EVENT_BUS_INTEGRATION_GUIDE.md) | Kafka event streaming integration: adapters, publish/subscribe patterns, envelope format |
| [EVENT_STREAMING_TOPICS.md](architecture/EVENT_STREAMING_TOPICS.md) | Topic catalog, schemas, and usage patterns |
| [HANDLER_PROTOCOL_DRIVEN_ARCHITECTURE.md](architecture/HANDLER_PROTOCOL_DRIVEN_ARCHITECTURE.md) | Handler system design: routing strategies, classification, no-publish constraint |
| [MESSAGE_DISPATCH_ENGINE.md](architecture/MESSAGE_DISPATCH_ENGINE.md) | MessageDispatchEngine internals: routing, dispatcher resilience, canonical pattern |
| [MCP_SERVICE_ARCHITECTURE.md](architecture/MCP_SERVICE_ARCHITECTURE.md) | MCP (Model Context Protocol) service layer: tool registration, schema generation, skip_server testing |
| [CONFIG_DISCOVERY.md](architecture/CONFIG_DISCOVERY.md) | Contract-driven config discovery: Infisical-backed prefetch, transport config map, bootstrap sequence |
| [TOPIC_CATALOG_ARCHITECTURE.md](architecture/TOPIC_CATALOG_ARCHITECTURE.md) | Topic catalog service architecture: discovery, validation, response channels |
| [LLM_INFRASTRUCTURE.md](architecture/LLM_INFRASTRUCTURE.md) | LLM infrastructure: multi-server topology, endpoint selection, cost tracking SPI |
| [SNAPSHOT_PUBLISHING.md](architecture/SNAPSHOT_PUBLISHING.md) | Snapshot publication patterns for state broadcast |
| [DLQ_MESSAGE_FORMAT.md](architecture/DLQ_MESSAGE_FORMAT.md) | Dead Letter Queue message schema and replay fields |
| [CIRCUIT_BREAKER_THREAD_SAFETY.md](architecture/CIRCUIT_BREAKER_THREAD_SAFETY.md) | Concurrency safety in MixinAsyncCircuitBreaker |
| [README.md](architecture/README.md) | Architecture section index |

### Conventions

Coding conventions specific to `omnibase_infra`.

| Document | Description |
|----------|-------------|
| [NAMING_CONVENTIONS.md](conventions/NAMING_CONVENTIONS.md) | File and class naming patterns for all infra artifact types, with real examples |
| [PYDANTIC_BEST_PRACTICES.md](conventions/PYDANTIC_BEST_PRACTICES.md) | ConfigDict requirements, field patterns, immutability rules, custom `__bool__` documentation |
| [ERROR_HANDLING_BEST_PRACTICES.md](conventions/ERROR_HANDLING_BEST_PRACTICES.md) | Infra error hierarchy, `ModelInfraErrorContext` usage, error class selection, sanitization rules |
| [TERMINOLOGY_GUIDE.md](conventions/TERMINOLOGY_GUIDE.md) | Canonical definitions for ONEX architectural terms used in code, comments, and documentation |
| [README.md](conventions/README.md) | Conventions section index |

### Decisions (ADRs)

Architecture Decision Records explaining why things work the way they do. Full index: [decisions/README.md](decisions/README.md).

| Document | Description |
|----------|-------------|
| [README.md](decisions/README.md) | Full ADR index: 9 numbered ADRs + 14 topic-based ADRs, with categories |
| [adr-001-graceful-shutdown-drain-period.md](decisions/adr-001-graceful-shutdown-drain-period.md) | Graceful shutdown drain period for in-flight message processing |
| [adr-002-enum-message-category-node-output-separation.md](decisions/adr-002-enum-message-category-node-output-separation.md) | Enum separation: message category vs node output type |
| [adr-003-remove-handler-health-checks.md](decisions/adr-003-remove-handler-health-checks.md) | Why handler-level health checks were removed |
| [adr-004-performance-baseline-thresholds.md](decisions/adr-004-performance-baseline-thresholds.md) | Calibration methodology for E2E performance thresholds |
| [adr-005-denormalize-capability-fields-in-registration-projections.md](decisions/adr-005-denormalize-capability-fields-in-registration-projections.md) | Denormalize capability fields in registration projections |
| [adr-006-message-dispatch-engine-canonical-routing.md](decisions/adr-006-message-dispatch-engine-canonical-routing.md) | MessageDispatchEngine as the canonical consumer routing pattern |
| [adr-007-vault-to-infisical-migration.md](decisions/adr-007-vault-to-infisical-migration.md) | Vault to Infisical migration rationale |
| [adr-008-realm-agnostic-topics.md](decisions/adr-008-realm-agnostic-topics.md) | Realm-agnostic Kafka topic format |
| [adr-009-llm-cost-tracking-spi.md](decisions/adr-009-llm-cost-tracking-spi.md) | LLM cost tracking at the infrastructure layer via SPI |
| [adr-any-type-pydantic-workaround.md](decisions/adr-any-type-pydantic-workaround.md) | When `Any` is permitted in Pydantic models |
| [adr-canonical-publish-interface-policy.md](decisions/adr-canonical-publish-interface-policy.md) | Canonical interface for event publishing |
| [adr-consumer-group-naming.md](decisions/adr-consumer-group-naming.md) | Kafka consumer group naming convention |
| [adr-cryptography-upgrade-46.md](decisions/adr-cryptography-upgrade-46.md) | Cryptography library upgrade to 46.x |
| [adr-custom-bool-result-models.md](decisions/adr-custom-bool-result-models.md) | Custom `__bool__` for result models |
| [adr-dsn-validation-strengthening.md](decisions/adr-dsn-validation-strengthening.md) | DSN validation strengthening for database URLs |
| [adr-enum-message-category-vs-node-output-type.md](decisions/adr-enum-message-category-vs-node-output-type.md) | Enum message category vs node output type distinction |
| [adr-error-context-factory-pattern.md](decisions/adr-error-context-factory-pattern.md) | Error context factory pattern: `ModelInfraErrorContext.with_correlation()` |
| [adr-handler-contract-schema-evolution.md](decisions/adr-handler-contract-schema-evolution.md) | Handler contract schema evolution strategy |
| [adr-handler-plugin-loader-security.md](decisions/adr-handler-plugin-loader-security.md) | Handler plugin loader security model and threat analysis |
| [adr-handler-type-vs-handler-category.md](decisions/adr-handler-type-vs-handler-category.md) | Handler type vs handler category classification |
| [adr-kafka-required-infrastructure.md](decisions/adr-kafka-required-infrastructure.md) | Kafka as required infrastructure dependency |
| [adr-projector-composite-key-upsert-workaround.md](decisions/adr-projector-composite-key-upsert-workaround.md) | Composite key upsert workaround for projectors |
| [adr-protocol-design-guidelines.md](decisions/adr-protocol-design-guidelines.md) | Protocol design guidelines: Protocol vs ABC, structural typing |
| [adr-soft-validation-env-parsing.md](decisions/adr-soft-validation-env-parsing.md) | Soft validation for environment variable parsing |

### Getting Started

New to ONEX? Start here.

| Document | Description |
|----------|-------------|
| [quickstart.md](getting-started/quickstart.md) | Get ONEX running in 5 minutes |
| [README.md](getting-started/README.md) | Getting started section index |

### Guides

Step-by-step tutorials and examples.

| Document | Description |
|----------|-------------|
| [registration-example.md](guides/registration-example.md) | Complete 2-way registration walkthrough: all four node archetypes with code examples |
| [HANDLER_AUTHORING_GUIDE.md](guides/HANDLER_AUTHORING_GUIDE.md) | How to create a handler: classification properties, no-publish constraint, contract YAML registration |
| [MCP_INTEGRATION_GUIDE.md](guides/MCP_INTEGRATION_GUIDE.md) | Expose ONEX node operations as MCP tools: tool registration, schema generation, skip_server testing |
| [INFISICAL_SECRETS_GUIDE.md](guides/INFISICAL_SECRETS_GUIDE.md) | Retrieve secrets via HandlerInfisical: path convention, bootstrap sequence, local development fallback |
| [README.md](guides/README.md) | Guides section index |

### Migration

Guides for migrating from legacy patterns.

| Document | Description |
|----------|-------------|
| [MIGRATION_WIRE_DEFAULT_HANDLERS.md](migration/MIGRATION_WIRE_DEFAULT_HANDLERS.md) | Migrate from `wire_default_handlers()` to contract-driven handler discovery |
| [README.md](migration/README.md) | Migration section index |

### Operations

Runbooks for operating ONEX infrastructure in production.

| Document | Description |
|----------|-------------|
| [DATABASE_INDEX_MONITORING_RUNBOOK.md](operations/DATABASE_INDEX_MONITORING_RUNBOOK.md) | GIN index performance monitoring, write overhead assessment, alerting thresholds |
| [DLQ_REPLAY_RUNBOOK.md](operations/DLQ_REPLAY_RUNBOOK.md) | Dead Letter Queue replay: manual procedures, automated design, safety considerations |
| [EVENT_BUS_OPERATIONS_RUNBOOK.md](operations/EVENT_BUS_OPERATIONS_RUNBOOK.md) | Deployment, configuration, monitoring, and troubleshooting for KafkaEventBus |
| [THREAD_POOL_TUNING_RUNBOOK.md](operations/THREAD_POOL_TUNING_RUNBOOK.md) | Thread pool configuration tuning for synchronous adapters |
| [README.md](operations/README.md) | Operations section index |

### Patterns

Implementation patterns and best practices. See [patterns/README.md](patterns/README.md) for the full index with quick-reference tables and pattern dependency graph.

| Document | Description |
|----------|-------------|
| [error_handling_patterns.md](patterns/error_handling_patterns.md) | Error classification, `ModelInfraErrorContext`, transport-aware codes, hierarchy |
| [error_sanitization_patterns.md](patterns/error_sanitization_patterns.md) | Data classification, sanitization rules, secure error reporting |
| [error_recovery_patterns.md](patterns/error_recovery_patterns.md) | Exponential backoff, circuit breakers, graceful degradation, credential refresh |
| [retry_backoff_compensation_strategy.md](patterns/retry_backoff_compensation_strategy.md) | Formal retry policies, backoff formulas, compensation for partial failures |
| [circuit_breaker_implementation.md](patterns/circuit_breaker_implementation.md) | Production-ready circuit breaker with state machine |
| [dispatcher_resilience.md](patterns/dispatcher_resilience.md) | Dispatcher-owned resilience for MessageDispatchEngine |
| [operation_routing.md](patterns/operation_routing.md) | Contract-driven operation dispatch with parallel execution and partial failure handling |
| [correlation_id_tracking.md](patterns/correlation_id_tracking.md) | Request tracing, envelope pattern, distributed logging |
| [container_dependency_injection.md](patterns/container_dependency_injection.md) | Service registration, resolution, and testing patterns |
| [protocol_patterns.md](patterns/protocol_patterns.md) | Protocol vs ABC vs concrete base, mixin composition, plugin interfaces |
| [mixin_dependencies.md](patterns/mixin_dependencies.md) | Mixin composition patterns, dependency requirements, inheritance order |
| [async_thread_safety_pattern.md](patterns/async_thread_safety_pattern.md) | Async thread safety: lock patterns, shared state in coroutines |
| [utility_directory_structure.md](patterns/utility_directory_structure.md) | Distinction between `utils/` and `shared/utils/` directories |
| [registry_clear_policy.md](patterns/registry_clear_policy.md) | When and how to implement `clear()` methods for test isolation |
| [consul_integration.md](patterns/consul_integration.md) | Consul connection, health checks, service registration, security patterns |
| [security_patterns.md](patterns/security_patterns.md) | Comprehensive security: error sanitization, input validation, auth, secrets, network security |
| [security_validation_architecture.md](patterns/security_validation_architecture.md) | Security validation architecture: input validation layers and enforcement points |
| [secret_resolver.md](patterns/secret_resolver.md) | Centralized secret resolution with caching; env, file, and Infisical sources |
| [binding_config_resolver.md](patterns/binding_config_resolver.md) | Multi-source handler configuration resolution with environment overrides and caching |
| [policy_registry_trust_model.md](patterns/policy_registry_trust_model.md) | Trust assumptions, validation boundaries, security mitigations for policy registration |
| [handler_plugin_loader.md](patterns/handler_plugin_loader.md) | Contract-driven handler discovery: threat model, security, deployment checklist |
| [db_url_contract.md](patterns/db_url_contract.md) | Database URL construction contract and DSN validation rules |
| [testing_patterns.md](patterns/testing_patterns.md) | Assertion patterns, "HOW TO FIX" error messages, test best practices |
| [README.md](patterns/README.md) | Patterns section index with quick-reference tables and dependency graph |

### Performance

Performance baselines and SLO definitions.

| Document | Description |
|----------|-------------|
| [LLM_ENDPOINT_SLO.md](performance/LLM_ENDPOINT_SLO.md) | SLOs for distributed LLM endpoints: TTFT, throughput, availability |
| [README.md](performance/README.md) | Performance section index |

### Plugins

Documentation for the ONEX plugin system and plugin implementations.

| Document | Description |
|----------|-------------|
| [json_normalizer_optimization.md](plugins/json_normalizer_optimization.md) | JSON normalizer plugin performance optimization |
| [plugin_determinism_verification.md](plugin_determinism_verification.md) | Compute plugin determinism verification guide |
| [README.md](plugins/README.md) | Plugins section index |

### Reference

Comprehensive reference documentation.

| Document | Description |
|----------|-------------|
| [node-archetypes.md](reference/node-archetypes.md) | EFFECT, COMPUTE, REDUCER, ORCHESTRATOR — complete documentation |
| [contracts.md](reference/contracts.md) | Complete contract.yaml format specification |
| [README.md](reference/README.md) | Reference section index |

### Standards

Normative specifications for documentation, terminology, and topic naming.

| Document | Description |
|----------|-------------|
| [STANDARD_DOC_LAYOUT.md](standards/STANDARD_DOC_LAYOUT.md) | Canonical directory structure, file naming rules, required doc sections, deletion policy |
| [ONEX_TERMINOLOGY.md](standards/ONEX_TERMINOLOGY.md) | Canonical vs. deprecated terms: transport types, model names, runtime terms |
| [TOPIC_TAXONOMY.md](standards/TOPIC_TAXONOMY.md) | Kafka topic suffix format, defined suffix constants, topic spec registry, validation rules |
| [README.md](standards/README.md) | Standards section index |

### Testing

Test strategy and integration testing documentation.

| Document | Description |
|----------|-------------|
| [CI_TEST_STRATEGY.md](testing/CI_TEST_STRATEGY.md) | Pytest markers, 60% coverage requirement, test directory structure, common commands |
| [INTEGRATION_TESTING.md](testing/INTEGRATION_TESTING.md) | E2E test infrastructure: two-layer env loading, 53-test E2E suite, runtime container alignment |
| [README.md](testing/README.md) | Testing section index |

### Validation

Validation framework documentation.

| Document | Description |
|----------|-------------|
| [validator_reference.md](validation/validator_reference.md) | All validators: architecture, contract, pattern, union, circular import |
| [framework_integration.md](validation/framework_integration.md) | Integration with omnibase_core validation framework |
| [troubleshooting.md](validation/troubleshooting.md) | Common validation errors and how to fix them |
| [performance_notes.md](validation/performance_notes.md) | Performance considerations for running validators |
| [circular_import_validator_improvements.md](validation/circular_import_validator_improvements.md) | Circular import validator improvements and configuration |
| [EVENT_BUS_COVERAGE_REPORT.md](validation/EVENT_BUS_COVERAGE_REPORT.md) | Event bus validation coverage report |
| [README.md](validation/README.md) | Validation framework overview with quick start and CI/CD integration |

---

## Document Status

| Section | Count | Notes |
|---------|-------|-------|
| Architecture | 15 docs | Includes MCP, config discovery, topic catalog, LLM infrastructure, registration workflow |
| Conventions | 4 docs | Naming, Pydantic, error handling, terminology |
| Decisions (ADRs) | 23 ADRs | 9 numbered + 14 topic-based; see [decisions/README.md](decisions/README.md) for categories |
| Getting Started | 1 guide | Quick start |
| Guides | 4 guides | Registration, handler authoring, MCP integration, Infisical secrets |
| Migration | 1 guide | wire_default_handlers() migration |
| Operations | 4 runbooks | Database, DLQ, event bus, thread pool |
| Patterns | 23 patterns | Full index with dependency graph at [patterns/README.md](patterns/README.md) |
| Performance | 1 doc | LLM endpoint SLO |
| Plugins | 2 docs | JSON normalizer, plugin determinism |
| Reference | 2 docs | Node archetypes, contract reference |
| Standards | 3 docs | Doc layout, terminology, topic taxonomy |
| Testing | 2 docs | CI strategy, integration testing |
| Validation | 6 docs | Validator reference, framework integration, troubleshooting, performance, circular imports, coverage |
| **Total** | **~91 docs** | Across 14 sections |
