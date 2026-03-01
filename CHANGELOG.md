# Changelog

All notable changes to the ONEX Infrastructure (omnibase_infra) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.13.0] - 2026-02-28

### Added
- Contract-driven Kafka topic creator script `create_kafka_topics.py` (OMN-2965, #488)
- `TopicEnumGenerator` — per-producer enum rendering (OMN-2964, #487)
- `ContractTopicExtractor` for contract-driven topic parsing (OMN-2963, #486)
- `generate_topic_enums.py` script and initial generated enum files (OMN-2966, #490)
- AI-slop checker Phase 2 rollout (#491)
- Catalog responder for `topic-catalog-request.v1` (OMN-2923, #469)
- `NodeBaselinesBatchCompute` EFFECT node (OMN-3039, #497)
- Skill lifecycle consumer and topic provisioning (OMN-2934, #475)
- Consumer for manifest injection lifecycle events (OMN-2942, #481)
- Decision-recorded topics to intelligence provisioning registry (OMN-2943, #477)
- Reconnect backoff kwargs wired to AIOKafkaProducer/Consumer sites (OMN-2919, #467)
- `reconnect_backoff_ms`/`max_ms` to `ModelKafkaEventBusConfig` (OMN-2916, #466)
- Configurable Redpanda memory and connection limits (OMN-2917, #465)
- Canonical `ModelRewardAssignedEvent` with policy signal fields (OMN-2928, #470)
- `omninode-claude` plugin install in `Dockerfile.runtime` (OMN-3182, #498)
- E2E automated regression for contract-driven topic enum pipeline (OMN-3186, #499)

### Fixed
- Add 21 missing omnimemory topics to provisioning registry (OMN-2941, #480)
- Add `agent-observability` DLQ topic to provisioning registry (OMN-2959, #484)
- Gate omnimemory topic provisioning behind `OMNIMEMORY_ENABLED` flag (OMN-2944, #479)
- Resolve 503 health check and DLQ validation failures in agent-actions (OMN-2986, #494)
- Extend contract discovery to find `contract_*.yaml` files (OMN-2995, #496)
- CAS-atomic topic subscriber index writes in Consul (OMN-2345, #483)
- Correct `TOPIC_SESSION_OUTCOME_CANONICAL` producer segment to `omniclaude` (OMN-2946, #476)
- Correct gmail-archive-purged topic name hyphen in producer segment (OMN-2937, #473)
- Retire orphan `policy-state-updated` topic constant (OMN-2931, #474)
- Retire orphan `run-evaluated` topic and stale model (OMN-2929, #471)
- Remove stale `run_evaluated` capability from registry (OMN-2930, #472)
- Replace stub `ModelScoreVector` with canonical omnibase_core model (OMN-2927, #468)
- Update handler count assertions for `HandlerCatalogRequest` (#485)
- Show full correlation UUID in Slack context block (#489)
- Remove stale omninode_bridge comment from docker-build workflow (#482)
- Tune AI-slop checker v1.0 — scope `step_narration` to markdown only (OMN-3191, #500)

### Changed
- Renamed `PYPI_PRIVATE_*` secrets to `PYPI_*` for public PyPI (#495)

### Dependencies
- `omnibase-core` pinned to ==0.22.0 (was >=0.21.0,<0.22.0)
- `omnibase-spi` pinned to ==0.15.0 (was >=0.14.0,<0.15.0)

## [0.12.0] - 2026-02-27

### Changed
- Version bump as part of coordinated OmniNode platform release run release-20260227-eceed7

### Dependencies
- omnibase-core bumped to >=0.21.0,<0.22.0 (was >=0.20.0,<0.21.0); git source override removed
- omnibase-spi bumped to >=0.14.0,<0.15.0 (was >=0.13.0,<0.14.0)

## [0.11.0] - 2026-02-25

### Added

#### Event Bus Registry (OMN-2700, MCP-04)

- **Replace Consul discovery with event bus registry queries**: `HandlerMcpRegistryEffect` now queries the event bus registry instead of Consul for service discovery, removing Consul as a hard dependency for MCP-04 discovery flows (#421)

#### Topic Catalog PostgreSQL Backend (OMN-2746)

- **Replace `ServiceTopicCatalog` Consul KV backend with PostgreSQL**: Topic catalog persistence migrated from Consul KV to PostgreSQL, eliminating Consul as a runtime dependency for topic catalog operations (#422)

#### Runtime Observability (OMN-2292)

- **Runtime source-hash and compose-project startup banner**: Services now log a structured startup banner including source hash, compose project name, and environment at boot time for improved traceability (#412)

#### Deployment Safety (OMN-2296)

- **Detect compose project name collisions in `deploy-runtime.sh`**: The deploy script now detects and rejects duplicate compose project names before starting services, preventing silent container conflicts (#413)

### Changed

- Bumped version to 0.11.0

### Tests

- **Adversarial fingerprint CI twins** (OMN-2293): Tests that prove fingerprint CI twins catch drift between runtime and test environments (#414)

### Documentation

- **ADR: two-handler-system architecture** (OMN-1973): Decision record documenting the dual-handler pattern for protocol binding separation (#411)

## [0.10.0] - 2026-02-23

### Added

#### Zero-Repo-Env Policy (OMN-2287)

- **`scripts/register-repo.py`** — central Infisical onboarding CLI with `seed-shared` and `onboard-repo` subcommands; dry-run by default, `--execute` required to write; replaces ~80 lines of hardcoded secret declarations with YAML-driven loading (#387, #400)
- **`config/shared_key_registry.yaml`** — versioned authoritative registry of 39 shared platform keys across 8 transport folders (`db`, `kafka`, `consul`, `vault`, `llm`, `auth`, `valkey`, `env`); single source of truth replacing the hardcoded `SHARED_PLATFORM_SECRETS` dict (#393, #400)
- **`contract_config_extractor.py`**: extended `_TRANSPORT_ALIASES` to cover 13 previously unmapped keys (#387)
- **Pre-commit hook** (OMN-2476): rejects `.env` files anywhere in the repo tree; `.env` removed from the allowed root file list, enforcing the zero-repo-env policy (#388, #389)

#### LLM Metrics Observability (OMN-2443)

- **`ServiceLlmMetricsPublisher`** — service-layer wrapper around `HandlerLlmOpenaiCompatible` that reads `last_call_metrics` after each inference call and publishes to `onex.evt.omniintelligence.llm-call-completed.v1`; fixes zero-data `/cost-trends` dashboard (#390)
- **`register_openai_compatible_with_metrics()`** and **`register_ollama_with_metrics()`** factory methods on `RegistryInfraLlmInferenceEffect` for wiring the publisher at container bootstrap time (#390)

### Fixed

- **`ConfigSessionStorage`** (session): removed `env_prefix="OMNIBASE_INFRA_SESSION_STORAGE_"` so the config reads standard `POSTGRES_*` vars rather than the non-existent prefixed variants (#391)
- **`config_store.py`**: set `env_file=None` to prevent stale `.env` file reads after zero-repo-env migration (#400)

### Changed

#### Dependencies

- **Bump `omnibase-core`** from `>=0.18.1,<0.19.0` → `>=0.19.0,<0.20.0` (DecisionRecord, NodeReducer projection effect)
- **Bump `omnibase-spi`** from `>=0.10.0,<0.11.0` → `>=0.12.0,<0.13.0` (ProtocolEffect, ProtocolNodeProjectionEffect, ContractProjectionResult — OMN-2508)
- **Bump `omniintelligence`** from `0.4.0` → `0.5.0` in `docker/Dockerfile.runtime` (#386)
- Bumped version to 0.10.0

## [0.9.0] - 2026-02-20

### Added

#### OmniMemory Topics

- **OMN-2383**: `platform_topic_suffixes` — OmniMemory Kafka topic suffix constants (`store`, `retrieve`, `retrieved`, `delete`, `deleted`, `search`, `search_results`, `error`) with package exports and full unit test coverage (#383)

#### Topic Catalog

- **OMN-2314**: Topic catalog change notification emission with CAS (compare-and-swap) versioning — catalog mutations now emit `TopicCatalogChangedEvent` with a version vector for optimistic concurrency control (#379)
- **OMN-2312**: Topic catalog response warnings channel — catalog query responses now carry a `warnings` field for non-fatal advisory messages (e.g. deprecated topic references, schema drift) (#377)

#### LLM-Driven Code Generation Handlers

- **OMN-2278**: `HandlerCodeReviewAnalysis` — code review analysis handler via Coder-14B LLM, producing structured review results from git diff input (#376)
- **OMN-2277**: `HandlerTestBoilerplateGeneration` — test boilerplate generation handler via Coder-14B LLM, scaffolding pytest unit tests from source signatures (#375)

#### Tests

- **OMN-1686**: Unit tests for `NodeLedgerWriteEffect` handlers — full coverage of ledger write effect handler behaviour including error paths (#382)
- **OMN-2317**: Topic catalog multi-client no-cross-talk E2E test — validates that concurrent catalog clients do not observe each other's in-flight mutations (#378)

### Changed

#### Dependencies

- **Bump `omnibase-core`** from `>=0.18.0,<0.19.0` → `>=0.18.1,<0.19.0`
- **Bump `aquasecurity/trivy-action`** from `0.33.1` → `0.34.0` in CI vulnerability scanning workflow (#365)

## [0.8.1] - 2026-02-19

### Changed

#### Runtime Plugin

- **Bump `omniintelligence`** from `0.2.3` → `0.4.0` in `docker/Dockerfile.runtime`

## [0.8.0] - 2026-02-19

### Added

#### LLM Inference Infrastructure

- **OMN-2104**: `MixinLlmHttpTransport` for structured LLM HTTP calls with sanitized response bodies, case-insensitive content-type handling, and locked client teardown (#320, #322)
- **OMN-2107**: `HandlerLlmOpenaiCompatible` for OpenAI wire-format inference (chat completions, embeddings) against local vLLM/Ollama-compatible servers (#325)
- **OMN-2108**: `HandlerLlmOllama` with node scaffold for Ollama-native inference (#328)
- **OMN-2112**: `node_llm_embedding_effect` with models, handlers, node, contract, and registry for embedding extraction (#327)
- **OMN-2105**: `ModelLlmInferenceRequest` and `ModelLlmMessage` for typed LLM request construction (#321)
- **OMN-2106**: `ModelLlmInferenceResponse` with `text XOR tool_calls` invariant enforcement (#324)
- **OMN-2103**: `ModelLlmShared` — shared LLM models for inference and embedding nodes (#318)
- **OMN-2111**: Inference node assembly with contract, registry, and operation validation (#335)
- **OMN-2255**: LLM endpoint health checker service with per-endpoint liveness probes (#352)
- **OMN-2249**: LLM endpoint SLO profiling and load test scaffolding (#347)
- **OMN-2250**: CIDR allowlist and HMAC request signing on LLM HTTP transport (#350)
- **OMN-2109**: Inference handler unit tests (#331)
- **OMN-2110**: Inference model validation tests (#334)
- **OMN-2113**: Embedding node unit tests (#329)
- **OMN-2114**: `MixinLlmHttpTransport` unit tests (#337)

#### LLM Cost Tracking

- **OMN-2238**: Token usage extraction and normalization from LLM API responses (#346)
- **OMN-2240**: LLM cost aggregation service with per-session and per-call rollups (#348)
- **OMN-2241**: Static context token cost attribution for system prompt overhead (#361)
- **OMN-2239**: `ModelPricingTable` with YAML manifest and cost estimation utilities (#360)
- **OMN-2236**: `llm_call_metrics` and `llm_cost_aggregates` database migration 031 (#343)
- **OMN-2295**: LLM cost tracking input validation and edge case tests (#358)
- **OMN-2318**: Integrate SPI 0.9.0 LLM cost tracking contracts (#345)
- **OMN-2319**: SPI LLM protocol adapters for `ProtocolLlmCostTracker` and `ProtocolLlmPricingTable` (#353)

#### Enrichment Handlers

- **OMN-2260**: `HandlerCodeAnalysisEnrichment` for git diff analysis via Coder-14B LLM (#363)
- **OMN-2261**: Embedding similarity enrichment handler for vector-based context relevance scoring (#366)
- **OMN-2262**: Context summarization enrichment handler for token-efficient context compression (#367)
- **OMN-2276**: Documentation generation handler via Qwen-72B for automated doc synthesis (#371)

#### Topic Catalog

- **OMN-2310**: Topic Catalog model and suffix foundation (#357)
- **OMN-2311**: `ServiceTopicCatalog` with KV (Valkey) precedence and in-memory caching (#370)
- **OMN-2313**: Topic catalog query handler, dispatcher, and contract wiring (#372)

#### Baselines and Effectiveness Metrics

- **OMN-2155**: A/B baseline comparison compute node with delta scoring (#332)
- **OMN-2303**: Batch compute effectiveness metrics and cache invalidation notifier (#362)
- **OMN-2305**: Baselines tables and batch compute service with Postgres persistence (#369)

#### Secret Management — Infisical Backend

- **OMN-2286**: Infisical secret backend: adapter, handler, and config resolution layer (#355)
- **OMN-2287**: Contract-driven config discovery, Infisical seed script, and bootstrap orchestration (#359)
- **OMN-2288**: Remove Vault handler; migrate all secret resolution references to Infisical (#368)

#### Schema and Event Registry Integrity

- **OMN-2087**: Schema fingerprint manifest with startup assertion gate (#317)
- **OMN-2088**: Event registry fingerprint with startup assertion gate (#326)
- **OMN-2149**: CI twins for schema and event registry fingerprint drift detection (#338)
- **OMN-2151**: Full check catalog, artifact storage, and flake detection (#330)

#### Runtime and Bootstrap

- **OMN-2089**: Bootstrap attestation gate in kernel handshake phase (#336)
- **OMN-2081**: Runtime contract routing verification tests and demo (#312)
- **OMN-2192**: Install `omniintelligence` in runtime Docker image (#323)
- **OMN-2233**: Stable runtime deployment script for repeatable container launches (#340)
- **OMN-2243**: Intelligence topic provisioning; bump omniintelligence to 0.2.0 (#342)
- **OMN-2342**: Set `OMNIINTELLIGENCE_PUBLISH_INTROSPECTION` on `omninode-runtime` only (#364)

#### Demo and Test Tooling

- **OMN-2297**: Demo loop assertion gate for canonical event loop validation (#349)
- **OMN-2299**: Demo reset scoped command for safe environment reset between runs (#354)

#### Error Taxonomy

- **OMN-2103**: `InfraRateLimitedError` exception class added to infrastructure error hierarchy (#315)

#### Registration

- **OMN-996**: Implement `reduce_confirmation()` for registration reducer (#319)
- Reducer-authoritative registration with E2E integration follow-ups (#316)

### Fixed

- **OMN-2251**: Consumer group instance discriminator for multi-container dev environments (#351)
- Sanitize response bodies, case-insensitive content-type, lock client teardown in LLM transport (#322)

### Changed

#### Dependencies

- Update `omnibase-core` from `^0.17.0` to `^0.18.0` (SPI 0.10.0 compatibility)
- Update `omnibase-spi` from `^0.8.0` to `^0.10.0` (enrichment contracts: `ProtocolContextEnrichment`, `ContractEnrichmentResult`, LLM cost tracking protocols)

#### Build Tooling

- **Migrate from Poetry to uv** for all dependency management and virtual environment workflows (#341)
  - All commands now use `uv run` (e.g., `uv run pytest`, `uv run mypy`, `uv run ruff`)
  - `uv.lock` replaces `poetry.lock` as the canonical lockfile
  - Deploy scripts updated for uv migration (#356)

#### CI/CD

- **OMN-2184**: Required status checks added to branch protection rules (#333)
- **OMN-2160**: Extract duplicated rules from CLAUDE.md to shared config (#339)

## [0.7.0] - 2026-02-12

### Changed

#### Dependencies
- Update `omnibase-core` from `^0.16.0` to `^0.17.0`
- Update `omnibase-spi` from `^0.7.0` to `^0.8.0`

## [0.6.0] - 2026-02-09

### Changed

#### Dependencies
- Update `omnibase-core` from `^0.15.0` to `^0.16.0`
- Update `omnibase-spi` from `^0.6.4` to `^0.7.0`

## [0.4.1] - 2026-02-06

### Changed

#### Dependencies
- Update `omnibase-core` from `^0.14.0` to `^0.15.0`

## [0.4.0] - 2026-02-05

### Breaking Changes

#### EventBusSubcontractWiring API Change
- **`EventBusSubcontractWiring.__init__()`** now requires two new parameters: `service` and `version`
  - **Old**: `EventBusSubcontractWiring(event_bus, contract)`
  - **New**: `EventBusSubcontractWiring(event_bus, contract, service="my-service", version="1.0.0")`
  - **Migration**: Add `service` and `version` parameters to all `EventBusSubcontractWiring` instantiations

#### Realm-Agnostic Topics
- **Topics no longer include environment prefix**: The `resolve_topic()` function now returns topic suffixes unchanged
  - **Old**: `resolve_topic("events.v1")` returned `"dev.events.v1"` (with env prefix)
  - **New**: `resolve_topic("events.v1")` returns `"events.v1"` (no prefix)
  - **Impact**: Cross-environment event routing now possible; isolation maintained through envelope identity

#### Subscribe Signature Change (omnibase-core 0.14.0)
- **`ProtocolEventBus.subscribe()`** parameter changed from `group_id: str` to `node_identity: ProtocolNodeIdentity`
  - **Old**: `event_bus.subscribe(topic, group_id="my-group", on_message=handler)`
  - **New**: `event_bus.subscribe(topic, node_identity=ModelEmitterIdentity(...), on_message=handler)`
  - **Migration**: Replace `group_id` with `ModelEmitterIdentity(env, service, node_name, version)`

#### ModelIntrospectionConfig Requires node_name
- **`ModelIntrospectionConfig`** now requires `node_name` as a mandatory field
  - **Old**: Could instantiate with only `node_id` and `node_type`
  - **New**: Must also provide `node_name` parameter
  - **Migration**: Add `node_name=<your_node_name>` to all `ModelIntrospectionConfig` instantiations
  - **Failure**: Omitting `node_name` raises `ValidationError`

#### ModelPostgresIntentPayload.endpoints Validation
- **`ModelPostgresIntentPayload.endpoints`** validator now raises `ValueError` for empty Mapping
  - **Old**: Empty `{}` logged a warning and returned empty tuple
  - **New**: Empty `{}` raises `ValueError("endpoints cannot be an empty Mapping")`
  - **Migration**: Ensure `endpoints` is either `None` or a non-empty Mapping

### Deprecated

#### RegistryPolicy.register_policy()
- **`RegistryPolicy.register_policy()`** method is deprecated
  - **Old**: `policy.register_policy(policy_type, priority, handler)`
  - **New**: `policy.register(ModelPolicyRegistration(policy_type, priority, handler))`
  - **Migration**: Replace `register_policy()` calls with `register(ModelPolicyRegistration(...))`
  - **Warning**: Emits `DeprecationWarning` at call site

### Added

#### Slack Webhook Handler (OMN-1905)
- **HandlerSlackWebhook**: Async handler with Block Kit formatting, retry with exponential backoff, and 429 rate limit handling
- **NodeSlackAlerterEffect**: Pure declarative effect node for Slack alerts
- **EnumAlertSeverity**: Severity levels (critical/error/warning/info)
- **ModelSlackAlert/ModelSlackAlertResult**: Type-safe frozen Pydantic models
- Features: Correlation ID tracking, exponential backoff retry (1s → 2s → 4s), 429 rate limit handling

#### Contract Dependency Resolution (OMN-1903, OMN-1732)
- **ContractDependencyResolver**: Reads protocol dependencies from `contract.yaml` and resolves from container
- **ModelResolvedDependencies**: Pydantic model for resolved protocol instances
- **ProtocolDependencyResolutionError**: Fail-fast error for missing protocols
- **RuntimeHostProcess integration**: Automatic dependency resolution during node discovery
- Zero-code nodes can now receive injected dependencies via constructor

#### Event Ledger Integration Tests (OMN-1649)
- Added comprehensive integration tests for Event Ledger runtime wiring

### Changed

#### Dependencies
- Update `omnibase-core` from `^0.13.1` to `^0.14.0`

## [0.3.2] - 2026-02-02


### Changed

#### Dependencies
- Update `omnibase-core` from `^0.12.0` to `^0.13.1`

#### Database Repository Models Migration
- Moved `ModelDbOperation`, `ModelDbParam`, `ModelDbRepositoryContract`, `ModelDbReturn`, `ModelDbSafetyPolicy` from `omnibase_core.models.contracts` to `omnibase_infra.runtime.db.models`
- These infrastructure-specific models are now owned by omnibase_infra
- Import path changed: `from omnibase_infra.runtime.db import ModelDbRepositoryContract, ...`

## [0.3.1] - 2026-02-02

### Fixed

- **OMN-1842**: Fix ORDER BY injection position when LIMIT clause exists in `PostgresRepositoryRuntime` (#229)
  - ORDER BY is now correctly inserted BEFORE existing LIMIT clause to produce valid SQL
  - Added detection for parameterized LIMIT (`$n`) to prevent duplicate LIMIT injection
  - Before (invalid): `SELECT ... LIMIT $1 ORDER BY id`
  - After (valid): `SELECT ... ORDER BY id LIMIT $1`

## [0.3.0] - 2026-02-01

### Added

- Minor version release

### Changed

- Version bump from 0.2.x to 0.3.0

## [0.2.8] - 2026-01-30

### Changed

#### Dependencies
- Update `omnibase-core` from `^0.9.10` to `^0.9.11`
- Update `omnibase-spi` from `^0.6.3` to `^0.6.4`

## [0.2.7] - 2026-01-30

### Changed

#### Dependencies
- Update `omnibase-core` from `^0.9.9` to `^0.9.10` for OMN-1551 contract-driven topics

## [0.2.6] - 2026-01-30

### Added

#### Contract Registry System
- **OMN-1654**: `KafkaContractSource` for cache-based contract discovery from Kafka topics (#213)
- **OMN-1653**: Contract registry reducer with Postgres projection for persistent contract storage (#212)

#### Event Ledger Persistence
- **OMN-1648**: `NodeLedgerProjectionCompute` for event ledger persistence with compute node pattern (#211)
- **OMN-1647**: PostgreSQL handlers for event ledger persistence operations (#209)
- **OMN-1646**: Event ledger schema and models for tracking event processing state (#208)

#### Declarative Configuration & Routing
- **OMN-1519**: `RuntimeContractConfigLoader` for declarative operation bindings from contract.yaml (#210)
- **OMN-1518**: Declarative topic→operation→handler routing with contract-driven dispatch (#198)
- **OMN-1621**: Contract-driven event bus subscription wiring for automatic topic binding (#200)

#### Emit Daemon
- **OMN-1610**: Emit daemon for persistent Kafka connections with connection pooling (#207)

#### Kafka & Event Bus Improvements
- **OMN-1613**: Event bus topic storage in registry for dynamic topic routing (#199)
- **OMN-1602**: Derived Kafka consumer group IDs with deterministic naming (#197)
- **OMN-1547**: Replace hardcoded topics with validated suffix constants (#206)

#### Handler & Intent Improvements
- **OMN-1509**: Intent storage effect node with integration tests (#195)
- **OMN-1515**: `execute()` dispatcher to `HandlerGraph` for contract discovery (#193)
- **OMN-1614, OMN-1616**: Canonical publish interface ADR and test adapter (#201)

### Changed

#### Dependencies
- **omnibase-core**: Updated from ^0.9.6 to ^0.9.9 (baseline topic constants export)
- **omnibase-spi**: Updated from ^0.6.2 to ^0.6.3

## [0.2.3] - 2026-01-25

### Added

#### OMN-1524: Infrastructure Primitives for Atomic Operations
- `write_atomic_bytes()` / `write_atomic_bytes_async()` for crash-safe file writes with temp file + rename pattern
- `transaction_context()` async context manager with configurable isolation levels, read-only/deferrable options, and per-transaction timeouts
- `retry_on_optimistic_conflict()` decorator/helper with exponential backoff, jitter, and attempt tracking
- Comprehensive test coverage (103 unit tests) for all new utilities

#### OMN-1515: Intent Handler Routing (Demo)
- `HANDLER_TYPE_GRAPH` and `HANDLER_TYPE_INTENT` constants for handler registration
- `HandlerIntent` class wrapping graph operations for intent storage
- Operations: `intent.store`, `intent.query_session`, `intent.query_distribution`
- Auto-routing registration for `HandlerGraph` and `HandlerIntent` in `util_wiring.py`

### Changed

#### Dependencies
- **omnibase-core**: Updated from ^0.9.1 to ^0.9.4 (core release with latest updates)

## [0.2.0] - 2026-01-17

### Breaking Changes

> **IMPORTANT**: This section documents API changes that may require code modifications when upgrading. Review each item carefully before upgrading.

#### File and Class Naming Standardization (OMN-1305, PR #151)

This refactoring enforces consistent naming conventions across the entire codebase per CLAUDE.md standards. **All import paths and class names have changed.**

##### Summary of Changes

| Category | Count | Pattern Change |
|----------|-------|----------------|
| Event Bus | 2 files, 2 classes | `{name}_event_bus` → `event_bus_{name}` |
| Handlers | 6 files, 6 classes | Suffix → Prefix standardization |
| Protocols | 4 files, 4 classes | Removed `Handler` suffix, domain-specific naming |
| Runtime | 6 files, 6 classes | Added `service_`, `util_`, `registry_` prefixes |
| Validation | 8 files, 8 classes | `{name}_validator` → `validator_{name}` |
| Stores | 2 classes | Suffix → Prefix standardization |

##### Event Bus Renames

| Old File | New File |
|----------|----------|
| `inmemory_event_bus.py` | `event_bus_inmemory.py` |
| `kafka_event_bus.py` | `event_bus_kafka.py` |

| Old Class | New Class |
|-----------|-----------|
| `InMemoryEventBus` | `EventBusInmemory` |
| `KafkaEventBus` | `EventBusKafka` |

**Migration**:
```python
# BEFORE
from omnibase_infra.event_bus.inmemory_event_bus import InMemoryEventBus
from omnibase_infra.event_bus.kafka_event_bus import KafkaEventBus

# AFTER
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
```

##### Handler Renames

| Old File | New File |
|----------|----------|
| `handler_mock_registration_storage.py` | `handler_registration_storage_mock.py` |
| `handler_postgres_registration_storage.py` | `handler_registration_storage_postgres.py` |
| `handler_consul_service_discovery.py` | `handler_service_discovery_consul.py` |
| `handler_mock_service_discovery.py` | `handler_service_discovery_mock.py` |

| Old Class | New Class |
|-----------|-----------|
| `MockRegistrationStorageHandler` | `HandlerRegistrationStorageMock` |
| `PostgresRegistrationStorageHandler` | `HandlerRegistrationStoragePostgres` |
| `ConsulServiceDiscoveryHandler` | `HandlerServiceDiscoveryConsul` |
| `MockServiceDiscoveryHandler` | `HandlerServiceDiscoveryMock` |
| `HttpRestHandler` | `HandlerHttpRest` |

**Migration**:
```python
# BEFORE
from omnibase_infra.handlers.registration_storage.handler_postgres_registration_storage import (
    PostgresRegistrationStorageHandler,
)

# AFTER
from omnibase_infra.handlers.registration_storage.handler_registration_storage_postgres import (
    HandlerRegistrationStoragePostgres,
)
```

##### Protocol Renames

| Old File | New File |
|----------|----------|
| `protocol_registration_storage_handler.py` | `protocol_registration_persistence.py` |
| `protocol_service_discovery_handler.py` | `protocol_discovery_operations.py` |

| Old Class | New Class |
|-----------|-----------|
| `ProtocolRegistrationStorageHandler` | `ProtocolRegistrationPersistence` |
| `ProtocolServiceDiscoveryHandler` | `ProtocolDiscoveryOperations` |

**Migration**:
```python
# BEFORE
from omnibase_infra.handlers.registration_storage.protocol_registration_storage_handler import (
    ProtocolRegistrationStorageHandler,
)

# AFTER
from omnibase_infra.handlers.registration_storage.protocol_registration_persistence import (
    ProtocolRegistrationPersistence,
)
```

##### Runtime File Renames

| Old File | New File | Rationale |
|----------|----------|-----------|
| `policy_registry.py` | `registry_policy.py` | Registry prefix pattern |
| `message_dispatch_engine.py` | `service_message_dispatch_engine.py` | Service prefix pattern |
| `runtime_host_process.py` | `service_runtime_host_process.py` | Service prefix pattern |
| `wiring.py` | `util_wiring.py` | Util prefix pattern |
| `container_wiring.py` | `util_container_wiring.py` | Util prefix pattern |
| `validation.py` | `util_validation.py` | Util prefix pattern |

| Old Class | New Class |
|-----------|-----------|
| `PolicyRegistry` | `RegistryPolicy` |
| `ProtocolBindingRegistry` | `RegistryProtocolBinding` |
| `MessageTypeRegistry` | `RegistryMessageType` |
| `EventBusBindingRegistry` | `RegistryEventBusBinding` |

**Migration**:
```python
# BEFORE
from omnibase_infra.runtime.policy_registry import PolicyRegistry
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

# AFTER
from omnibase_infra.runtime.registry_policy import RegistryPolicy
from omnibase_infra.runtime.service_message_dispatch_engine import MessageDispatchEngine
```

##### Validation File Renames

| Old File | New File |
|----------|----------|
| `any_type_validator.py` | `validator_any_type.py` |
| `chain_propagation_validator.py` | `validator_chain_propagation.py` |
| `contract_linter.py` | `linter_contract.py` |
| `registration_security_validator.py` | `validator_registration_security.py` |
| `routing_coverage_validator.py` | `validator_routing_coverage.py` |
| `runtime_shape_validator.py` | `validator_runtime_shape.py` |
| `security_validator.py` | `validator_security.py` |
| `topic_category_validator.py` | `validator_topic_category.py` |
| `validation_aggregator.py` | `service_validation_aggregator.py` |

> **Note**: Class names within validation files remain unchanged (e.g., `AnyTypeDetector`, `ChainPropagationValidator`). Only import paths changed.

**Migration**:
```python
# BEFORE
from omnibase_infra.validation.any_type_validator import AnyTypeDetector
from omnibase_infra.validation.chain_propagation_validator import ChainPropagationValidator

# AFTER
from omnibase_infra.validation.validator_any_type import AnyTypeDetector
from omnibase_infra.validation.validator_chain_propagation import ChainPropagationValidator
```

##### Store Class Renames

| Old Class | New Class |
|-----------|-----------|
| `InMemoryIdempotencyStore` | `StoreIdempotencyInmemory` |
| `PostgresIdempotencyStore` | `StoreIdempotencyPostgres` |

##### Automated Migration

Run these commands to find affected imports in your codebase:

```bash
# Find all affected imports
grep -rE "(InMemoryEventBus|KafkaEventBus|PolicyRegistry|inmemory_event_bus|kafka_event_bus)" \
    --include="*.py" /path/to/your/code

# Specific patterns for each category
grep -r "from omnibase_infra.event_bus.inmemory_event_bus" --include="*.py" .
grep -r "from omnibase_infra.event_bus.kafka_event_bus" --include="*.py" .
grep -r "from omnibase_infra.runtime.policy_registry" --include="*.py" .
grep -r "from omnibase_infra.validation.any_type_validator" --include="*.py" .
```

##### CI Enforcement

A new naming validator (`scripts/validation/validate_naming.py`) enforces these conventions. The CI pipeline will reject PRs that violate naming standards.

#### MixinNodeIntrospection API (OMN-881, PR #54)

##### 1. Cache Invalidation Method Signature Change

**`invalidate_introspection_cache()` is now synchronous (was async)**

This is a **breaking change** for any code that awaits this method.

| Aspect | Details |
|--------|---------|
| **What changed** | Method signature changed from `async def` to `def` (synchronous) |
| **Why it changed** | Cache invalidation is a simple in-memory operation (setting `_introspection_cache = None`) that does not require async I/O. Synchronous semantics simplify usage and avoid unnecessary coroutine overhead. |
| **Error if not migrated** | `TypeError: object NoneType can't be used in 'await' expression` |

**Migration Steps**:

```python
# BEFORE (will cause TypeError after upgrade)
await node.invalidate_introspection_cache()

# AFTER (correct usage)
node.invalidate_introspection_cache()
```

**Search pattern** to find affected code:
```bash
grep -r "await.*invalidate_introspection_cache" --include="*.py"
```

##### 2. Configuration Model API

**`initialize_introspection()` requires `ModelIntrospectionConfig`**

The initialization method uses a typed configuration model for all parameters.

| Aspect | Details |
|--------|---------|
| **What changed** | `initialize_introspection(config: ModelIntrospectionConfig)` is the initialization API |
| **Why** | Typed configuration model provides validation, IDE support, and extensibility |
| **Model location** | `omnibase_infra.models.discovery.ModelIntrospectionConfig` |

**Usage Example**:

```python
from uuid import uuid4
from omnibase_infra.models.discovery import ModelIntrospectionConfig
from omnibase_infra.mixins import MixinNodeIntrospection

class MyNode(MixinNodeIntrospection):
    def __init__(self, event_bus=None):
        config = ModelIntrospectionConfig(
            node_id=uuid4(),
            node_type="EFFECT",
            node_name="my_effect_node",
            event_bus=event_bus,
            version="1.0.0",
            cache_ttl=300.0,
        )
        self.initialize_introspection(config)

    async def shutdown(self):
        # Note: invalidate_introspection_cache() is now SYNC (see above)
        self.invalidate_introspection_cache()
```

#### Error Code for Unhandled node_kind (OMN-990, PR #73)
- **Error code changed from `VALIDATION_ERROR` to `INTERNAL_ERROR`**: When `DispatchContextEnforcer.create_context_for_dispatcher()` encounters an unhandled `node_kind` value, it now raises `ModelOnexError` with `INTERNAL_ERROR` instead of `VALIDATION_ERROR`.
  - **Old**: `error_code=EnumCoreErrorCode.VALIDATION_ERROR`
  - **New**: `error_code=EnumCoreErrorCode.INTERNAL_ERROR`
  - **Migration**: If you catch `ModelOnexError` and check for `VALIDATION_ERROR` when calling context creation methods, update to check for `INTERNAL_ERROR`.
  - **Rationale**: Unhandled `node_kind` values represent internal implementation errors (missing switch cases in exhaustive pattern matching) rather than user input validation failures. `INTERNAL_ERROR` more accurately reflects that this indicates a bug in the code rather than invalid configuration.

#### Handler Types (PR #33)
- **HANDLER_TYPE_REDIS renamed to HANDLER_TYPE_VALKEY**: The handler type constant for Redis-compatible cache has been renamed to accurately reflect the service name.
  - **Old**: `HANDLER_TYPE_REDIS = "redis"`
  - **New**: `HANDLER_TYPE_VALKEY = "valkey"`
  - **Migration**: Update any references from `HANDLER_TYPE_REDIS` to `HANDLER_TYPE_VALKEY`
  - **Rationale**: Valkey is the correct service name for the Redis-compatible cache used in the infrastructure. This aligns the codebase with the actual service naming.

#### Dependency Updates (OMN-1361, PR #156)
- **omnibase_core upgraded to 0.7.0**: Breaking changes in core dependency
- **omnibase_spi upgraded to 0.5.0**: Breaking changes in SPI dependency
- **pytest-asyncio 0.25+ compatibility**: Test framework compatibility updates, requires `asyncio_mode = "auto"` in pyproject.toml
- **Infrastructure IP defaults changed to localhost**: Default infrastructure IPs changed from remote server to localhost for local development

#### Error Handling (OMN-1181, PR #158)
- **RuntimeError replaced with structured domain errors**: All generic `RuntimeError` raises have been replaced with specific domain errors from the error taxonomy. If you were catching `RuntimeError`, update to catch the specific error types:
  - `ProtocolConfigurationError` for configuration issues
  - `InfraConnectionError` for connection failures
  - `InfraTimeoutError` for timeout issues
  - `InfraUnavailableError` for unavailable resources

### Added

#### Node Introspection (OMN-881, PR #54)
- **ModelIntrospectionConfig**: Configuration model for `MixinNodeIntrospection` that provides typed configuration
  - `node_id` (required): Unique identifier for this node instance (UUID)
  - `node_type` (required): Type of node (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR). Cannot be empty (min_length=1).
  - `event_bus`: Optional event bus for publishing introspection and heartbeat events. Uses duck typing (`object | None`) to accept any object implementing `ProtocolEventBus` protocol.
  - `version`: Node version string (default: `"1.0.0"`)
  - `cache_ttl`: Cache time-to-live in seconds (default: `300.0`, minimum: `0.0`)
  - `operation_keywords`: Optional set of keywords to identify operation methods. If None, uses `MixinNodeIntrospection.DEFAULT_OPERATION_KEYWORDS`.
  - `exclude_prefixes`: Optional set of prefixes to exclude from capability discovery. If None, uses `MixinNodeIntrospection.DEFAULT_EXCLUDE_PREFIXES`.
  - `introspection_topic`: Topic for publishing introspection events (default: `"node.introspection"`). ONEX topics (starting with `onex.`) require version suffix (e.g., `.v1`).
  - `heartbeat_topic`: Topic for publishing heartbeat events (default: `"node.heartbeat"`). ONEX topics require version suffix.
  - `request_introspection_topic`: Topic for receiving introspection requests (default: `"node.request_introspection"`). ONEX topics require version suffix.
  - Model is frozen and forbids extra fields for immutability and strict validation.
- **Performance Metrics Tracking**:
  - Added `IntrospectionPerformanceMetrics` dataclass (internal) and `ModelIntrospectionPerformanceMetrics` Pydantic model (for event payloads)
  - Added `get_performance_metrics()` method for monitoring introspection operation timing and threshold violations
  - Performance thresholds: `get_capabilities` <50ms, `discover_capabilities` <30ms, `total_introspection` <50ms, `cache_hit` <1ms
- **Topic Default Constants**: Exported constants for default topic names:
  - `DEFAULT_INTROSPECTION_TOPIC = "node.introspection"`
  - `DEFAULT_HEARTBEAT_TOPIC = "node.heartbeat"`
  - `DEFAULT_REQUEST_INTROSPECTION_TOPIC = "node.request_introspection"`

#### Documentation
- **Protocol Patterns Documentation** (OMN-1079, PR #166): Added comprehensive documentation for protocol design patterns, cross-mixin composition, and TYPE_CHECKING patterns in `docs/patterns/protocol_patterns.md`

#### Testing
- **Correlation ID Integration Tests** (OMN-1349, PR #160): Added integration tests for correlation ID propagation across service boundaries

#### Handlers
- **HttpHandler** (OMN-237, PR #26): HTTP REST protocol handler for MVP
  - GET and POST operations using httpx async client
  - Fixed 30s timeout (configurable timeout deferred to Beta)
  - Returns `EnumHandlerType.HTTP`
  - Error handling mapping to infrastructure errors (`InfraTimeoutError`, `InfraConnectionError`)
  - Full lifecycle support (initialize, shutdown, health_check, describe)
  - 46 unit tests with 97.93% coverage

#### Event Bus
- **InMemoryEventBus** (OMN-239, PR #25): In-memory event bus for local development and testing
  - Implements `ProtocolEventBus` from omnibase_core
  - Topic-based pub/sub with `asyncio.Queue` per topic
  - Thread-safe subscription management
  - Automatic cleanup on unsubscribe
  - Consumer groups with load balancing
  - Graceful shutdown with message draining
  - Comprehensive error handling
  - 1336+ lines of test coverage

#### Runtime
- **ProtocolBindingRegistry** (OMN-240, PR #24): Handler and event bus registration system
  - Single source of truth for handler registration
  - Thread-safe registration operations
  - Support for handler type constants (HTTP, DATABASE, KAFKA, etc.)
  - Event bus registry (InMemory, Kafka)
  - Protocol resolution utilities

#### Errors
- **Infrastructure Error Taxonomy** (OMN-290, PR #23): Structured error hierarchy
  - `RuntimeHostError`: Base infrastructure error class
  - `ProtocolConfigurationError`: Protocol configuration validation errors
  - `SecretResolutionError`: Secret/credential resolution errors
  - `InfraConnectionError`: Infrastructure connection errors (transport-aware)
  - `InfraTimeoutError`: Infrastructure timeout errors
  - `InfraAuthenticationError`: Infrastructure authentication errors
  - `InfraUnavailableError`: Infrastructure resource unavailable errors
  - `ModelInfraErrorContext`: Structured error context model
  - `EnumInfraTransportType`: Transport type classification

#### Infrastructure
- **Directory Structure** (OMN-236, PR #21): Initial MVP directory structure
  - `handlers/`: Protocol handler implementations
  - `event_bus/`: Event bus implementations
  - `runtime/`: Runtime host components
  - `errors/`: Infrastructure error classes
  - `enums/`: Infrastructure enumerations
  - `validation/`: Contract validation utilities

### Changed

#### Handler to Dispatcher Terminology Migration (OMN-977, PR #63)

The codebase has migrated from "handler" to "dispatcher" terminology for message routing components to better reflect their purpose as message dispatchers rather than generic handlers.

- **Protocol Rename**: `ProtocolHandler` → `ProtocolMessageDispatcher`
- **Class Naming**: Handler implementations renamed to Dispatcher (e.g., `UserEventHandler` → `UserEventDispatcher`)
- **ID Convention**: `dispatcher_id` values now use `-dispatcher` suffix instead of `-handler`
- **Enum Rename**: `EnumDispatchStatus.NO_HANDLER` renamed to `NO_DISPATCHER` with new value `no_dispatcher` for consistency with dispatcher terminology
- **Enum Value**: `EnumDispatchStatus.HANDLER_ERROR` retains its current value `handler_error` (rename deferred; see ADR for rationale)
- **Full Migration Guide**: See `docs/migrations/HANDLER_TO_DISPATCHER_MIGRATION.md` for complete migration details and code examples

#### CI/CD
- **Pre-commit Configuration**: Migrated to fix deprecated stage warnings (PR #25)

#### Dependencies
- **omnibase_core**: Updated from 0.6.x to 0.7.0
- **omnibase_spi**: Updated from 0.4.x to 0.5.0
- **pytest-asyncio**: Updated compatibility for 0.25+

---

## Architecture

```
omnibase_infra (YOU ARE HERE)
    ├── handlers/          # Protocol handler implementations
    │   ├── http_handler   # HTTP REST handler (MVP)
    │   └── db_handler     # PostgreSQL handler (MVP)
    ├── event_bus/         # Event bus implementations
    │   ├── inmemory       # InMemory bus (MVP)
    │   └── kafka          # Kafka bus (Beta)
    ├── runtime/           # Runtime host components
    │   ├── handler_registry
    │   └── runtime_host_process
    └── errors/            # Infrastructure errors
        └── infra_errors

DEPENDENCY RULE: infra -> spi -> core (never reverse)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### ONEX Standards
- Zero tolerance for `Any` types
- Contract-driven development
- Protocol-based dependency injection
- Comprehensive test coverage (>80% target)

## License

MIT License - See LICENSE file for details
