## v0.38.0 (2026-05-31)

### Features
- feat: drop bridge-backed delegation dispatch, use pure Kafka port (stage 2) (#1779)
- feat: remove dead instance-discovery fields from registry API (#1777)
- feat: skip Bifrost render for projection API (#1780)

### Bug Fixes
- fix: authenticate uv git+https fetches in CI (#1789)
- fix: clean up non-standard noqa suppressions (#1778)

### Changed
- Bumps omnibase-core pin to >=0.43.0,<0.44.0
- And 33+ additional commits since v0.37.0

## v0.32.0 (2026-04-03)

### Features
- feat(scripts): subscribe-topic wiring health check (#1136)
- feat: wire friction emission into build loop orchestrator (#1134)
- feat: add wire schema contracts for delegation + intelligence (#1133)
- feat: add verify-plugin-cache.sh for content-hash staleness detection (#1132)
- feat: autonomous build loop ONEX nodes (#1128)
- feat(validation): add sweep validators for stale TODOs, CLAUDE.md refs, migration conflicts (#1129)
- feat(router): intelligent model router MVP (#1125)
- feat(migration): add user_persona_snapshots table (#1124)
- feat(db): agent identity and session snapshot tables (#1120)
- feat(models): add AgentEntity and AgentBinding models for persistent agent identity (#1118)
- feat(registry): add YAML-based agent identity registry (#1119)
- feat(consumer): add agent learning extraction functions (#1110)
- feat(watchdog): add CAIA session watchdog for continuous pipeline (#1117)
- feat(consumer): add learning extraction consumer entrypoint stub (#1116)
- feat(migration): migrate 3 pilot handlers to declarative pattern (#1113)
- feat(llm): add resolution summary generator (#1109)
- feat(db): add agent_learnings table (#1107)
- feat(models): add agent learning record models for memory fabric (#1111)
- feat(store): add learning store freshness scoring utility (#1108)
- feat(delegation): wire delegation pipeline into runtime kernel (#1103)
- feat: port archive feature nodes (event-forward, llm, vector-store) (#1105)
- feat: add RSD Priority Engine four-node pattern (#1104)
- feat: add integration catalog contract YAML (#1102)
- feat(chain-learning): prompt-chain learning system with 5 ONEX nodes (#1101)
- feat: scope-check skill-to-node canary — 5 ONEX nodes with event-driven workflow (#1100)
- feat(infra): parameterize skills volume mount and remove hardcoded paths (#1099)
- feat(waitlist): Slack notification on new waitlist signup (#1091)
- feat(eval): add eval runner, baseline passthrough, metric collector, event emitter, regression check (#1090)
- feat: add GITHUB_WEBHOOK_SECRET contract for Infisical seeding (#1088)

### Bug Fixes
- fix: purge localhost fallbacks from omnibase_infra (#1127)
- fix: register ModelDispatchRoute for delegation dispatchers (#1135)
- fix(redpanda): raise topic_partitions_per_shard to 7000 (#1123)
- fix(runtime): increase healthcheck start_period and fix delegation contract (#1112)
- fix(migrations): move migrations 044-050 to docker runner path (#1106)
- fix(runtime): gate MCP handler on MCP_SERVER_ENABLED env var (#1097)
- fix(infra): use REDPANDA_ADVERTISE_HOST env var instead of hardcoded localhost (#1096)
- fix(infra): point check-stale-images.sh at generated compose (#1095)
- fix: populate introspection node_name from contract (#1092)
- fix: resolve macOS /var symlink false positive in path validation (#1093)
- fix: infra-up-runtime auto-rebuilds when code is newer than image (#1094)
- fix: populate introspection node_name from contract (#1089)
- fix: resolve macOS /var symlink false positive in path validation (#1087)
- fix(ci): auto-tag workflow matches chore: release PR titles (#1086)

### Other Changes
- test(integration): add agent identity lifecycle round-trip test (#1126)
- test(integration): agent identity lifecycle round-trip test (#1122)
- test(integration): add learning record construction test (#1115)
- test: add E2E verification script for error triage pipeline (#1114)
- chore(deps): bump aiohttp in the uv group across 1 directory (#1098)

## v0.31.0 (2026-03-31)

### Added
- feat: runtime contract compliance verification (#1064)
- feat: add TOPIC_EVAL_COMPLETED constant (#1080)
- feat: node-based LLM delegation — Kafka consumer, GLM, Gemini/Codex CLI (#1083)
- feat: infra unification — backend probes, registry auto-config, inmemory migration (#1071)

### Fixed
- fix: add node_name field to infra ModelNodeIntrospectionEvent (#1084)
- fix: add metadata.description to all node contract YAMLs (#1081)
- fix(topics): register 4 missing topics in provisioner (#1082)

### Changed
- refactor(event_bus): re-export EventBusInmemory from core (#1067)

### Dependencies
- chore(deps): bump actions/checkout from 4 to 6 (#1075)
- chore(deps): bump actions/upload-artifact from 4 to 7 (#1077)
- chore(deps): update opentelemetry-instrumentation-asyncpg requirement (#1072)
- chore(deps): update opentelemetry-instrumentation-redis requirement (#1073)
- chore(deps): update fastapi requirement (#1074)
- chore(deps-dev): update pytest-split requirement (#1078)

## v0.30.1 (2026-03-31)

### Fixed
- fix: increase runtime healthcheck start_period to 120s [F-CO-009] (#1068)
- fix: start health server before runtime.start() (#1069)

### Changed
- chore(deps): bump omnibase_core to 0.36.0
- ci: add onex compliance check to CI (#1070)

## v0.30.0 (2026-03-30)

### Changed
- chore(deps): bump omnibase_core to 0.35.0 (#1066)

## v0.29.0 (2026-03-28)

### Added
- feat: decision projector consumes Kafka events into Qdrant (#1040)
- feat: add Kafka-to-Memgraph session graph projector (#1038)
- feat: define session graph schema for Memgraph (#1036)
- feat: create session registry projector (#1034)
- feat: create session registry Postgres table and models (#1031)
- feat(ci): add auto-merge-on-open workflow (#1028)

### Fixed
- fix(runtime): add omnimemory to trusted plugin namespace prefixes (#1027)
- fix(ci): add SUFFIX_* export completeness test (#1022)
- fix(tooling): add cache-bust headers to PyPI requests (#1021)
- fix(infra): Phoenix healthcheck CMD format and Infisical provision bugs (#1018)
- fix(ci): topic suffix export check + consumer-health projection (#1017)

### Changed
- refactor: consolidate runner health exports and clean up models (#1033)
- chore(deps): bump omnibase-core to 0.34.0

### Dependencies
- omnibase-core 0.33.1 -> 0.34.0

## v0.28.0 (2026-03-27)

### Added
- feat: post-merge consumer chain (#1015)
- feat: GitHub PR merged event producer (webhook -> Kafka) (#1013)

### Fixed
- fix: wire correct consumption_source for WiringHealthChecker (#994)
- fix: deploy-runtime.sh copies omnibase_core runtime contracts (#1010)
- fix: entrypoint stamps schema fingerprints for all databases (#1011)
- fix: adjust Phoenix health check Python path and timings (#1008)
- fix: Phoenix health check uses python3 instead of missing curl (#1012)
- fix: strip v-prefix in plugin pin cascade verification (#1009)

### Changed
- chore: register custom noqa codes as ruff external linters (#1014)
- chore(deps): bump omnibase-core to 0.33.1, omnibase-spi to 0.20.2

## v0.27.1 (2026-03-26)

### Fixed
- fix(tests): remove dead Consul skips, convert flaky CI benchmarks (#1002)

### Changed
- chore: remove dead imports across src/ and tests/ (#1004)
- chore: standardize TODO markers with ticket references (#1005)
- chore: bump omnibase-spi to 0.20.1, omnibase-core to 0.33.0

### Dependencies
- omnibase-core 0.32.0 -> 0.33.0
- omnibase-spi 0.20.0 -> 0.20.1

## v0.27.0 (2026-03-25)

### Fixed
- fix: register AST extraction Kafka topics in platform topic suffixes (#999)
- fix(tests): add max_files guard to prevent OOM in infra scan (#998)
- fix: health check graceful degradation and start_period increase (#997)
- fix(ci): enable cancel-in-progress for merge queue events (#995)
- fix(ci): skip health check progression test when Postgres unavailable (#992)
- fix(ci): add contract path pre-flight validation before test splits (#989)
- fix: redeploy bugfix trio -- 3 Docker runtime bugs (#987)
- fix(ci): add test duration tracking for split rebalancing (#990)
- fix(ci): export PKG_HYPHEN before Python subprocess in dependency cascade (#986)

### Tests
- test: regression test for introspection_service wiring in ServiceKernel (#993)

### Dependencies
- chore(deps): bump requests in the uv group (#996)
- chore(deps): pin omnibase-core==0.32.0

## v0.26.0 (2026-03-24)

### Added
- feat: contract health Phase A+B -- handler unification + runtime config model (#983)
- feat(kafka): add max_request_size config, pass to producer, set broker default 4MB (#980)
- feat: env var alignment probe + seed profiles for cloud parity (#960)

### Fixed
- fix(ci): prevent xdist race in topic pipeline E2E tests (#985)
- fix(ci): use correct GHCR tag for Trivy and image size jobs (#979)
- fix(kafka): session timeout tuning + consumer self-healing restart (#962)

### Dependencies
- chore(deps): bump omnibase-core from 0.31.0 to 0.31.1 (#981, #984)

### Docs
- docs: add INTELLIGENCE_SERVICE_URL to env-example-full.txt (#977)

## v0.25.0 (2026-03-23)

### Added
- feat(runtime): wire emission pipelines for omnidash infra monitoring (#965)
- feat(runner-health): Runner Health Monitoring MVP (#961)

### Fixed
- fix(deps): update stale omnibase-core and spi version pins (#963)

### Dependencies
- chore(deps): bump tj-actions/changed-files from 45 to 47 (#967)
- chore(deps): bump aws-actions/configure-aws-credentials from 4 to 6 (#968)
- chore(deps): update opentelemetry-instrumentation-kafka-python (#969)
- chore(deps): update opentelemetry-instrumentation-fastapi (#970)
- chore(deps): update structlog (#971)
- chore(deps): bump actions/setup-python from 5 to 6 (#972)
- chore(deps): update uvicorn (#973)
- chore(deps): update aiofiles (#974)
- chore(deps): bump actions/checkout from 4 to 6 (#975)
- chore(ci): rename test.yml -> ci.yml for cross-repo standardization (#966)

## v0.24.0 (2026-03-22)

### Added
- feat: verify_container_manifest.py + CLI entry point (#947)
- feat(ci): add cross-repo migration conflict check (#943)
- feat(monitor): emit runtime errors to Kafka (#934)
- feat(contracts): create service-level feature flag contracts (#926)
- feat(introspection): wire contract data into introspection emission path (#923)
- feat(scripts): add post-release version verification script (#922)

### Fixed
- fix(ci): skip infra-dependent scheduled jobs on ubuntu-latest (#945)
- fix(deep-dive): include onex repos and add ticket summary section (#933)
- fix(docker): extend runtime-effects healthcheck timing to prevent false unhealthy (#931)
- fix(deploy): guard contracts/ rsync for missing directory (#928)
- fix(topics): register 5 missing Kafka topics in services manifest (#929)
- fix(deploy): add migration scripts rsync to deploy-runtime.sh (#925)
- fix(event_bus): wire retry_backoff_ms into AIOKafka constructors (#924)

### Changed
- ci: deploy TODO enforcement hooks and workflows (#941)
- chore: remove stale/canceled TODOs (#937, #939)
- chore: re-tag deferred and unfinished TODOs (#940)
- chore: delete orphaned handler contract scaffolds (#936)
- chore: update exemption markers to ONEX_FLAG_EXEMPT (#927)

## v0.23.0 (2026-03-20)

### Added
- feat: feature flag model wiring (#918)
- feat: Runtime Health Event Pipeline Waves 0-2 (#911)
- feat: add ServiceSavingsEstimator Kafka consumer (#917)
- feat(bifrost): shadow mode for learned routing policy comparison (#920)
- feat(capabilities): extract feature flags from contract YAML (#916)
- feat: populate contract YAMLs with feature_flags declarations (#914)
- feat(infra): add savings estimation compute handler with tiered attribution (#913)
- feat(ci): guard required docker-compose env vars (#900)
- feat(kafka): wire session timeout to all consumers (#905)
- feat(topics): package discovery + kill fallback paths (#910)
- ci: add compose required-env coverage guard (#899)

### Fixed
- fix(scripts): prune stale worktrees before fetch in pull-all.sh (#906)
- fix(deploy): add timeout and --progress=plain to docker build (#907)
- fix(catalog): pre-cleanup dead containers before compose up (#908)
- fix(mypy): resolve no-any-return in _load_stack via explicit type narrowing (#903)
- fix(docker): switch plugin pins to ranges for forward compatibility (#909)

### Changed
- chore: wire no-bare-feature-flags pre-commit hook (#915)
- chore(infra): register savings and validator-catch topics (#912)
- chore(deps): switch uv.sources from release branches to tags (#904)
- Dependency pins will be updated in a follow-up after upstream releases are tagged

### Tests
- test: add e2e integration test for savings estimation pipeline (#919)

## v0.22.0 (2026-03-19)

### Added
- feat(ci): deploy CodeQL security scanning to omnibase_infra (#896)
- feat: service catalog architecture with typed manifests and bundle definitions (#897)
- feat: activation-aware handler wiring (#886)
- feat(ci): add INV-4 contract-declared handler wiring completeness check (#889)
- feat: contract-driven topic discovery and drift CI (#865)
- feat(event_bus): wire ONEX topic format gate into publish() (#863)
- feat(event_bus): add debounced Slack alerting for topic violations (#862)
- feat(consumers): add ContextAuditConsumer for context integrity audit events (#877)
- feat(infra): centralized onboarding system (#869)
- feat: emit wiring-health-snapshot.v1 and llm-call-completed.v1 events
- feat(validation): cross-repo validation event models (#864)
- feat: multi-package entry point discovery for create_kafka_topics.py (#891)
- feat(ci): upgrade plugin-pin-cascade to full reconciliation workflow (#893)

### Fixed
- fix: eliminate empty-default env var fallbacks from compose
- fix: hardcode container-internal addresses for valkey, memgraph, keycloak
- fix: graph handler reads OMNIMEMORY_MEMGRAPH_HOST/PORT for URI resolution (#884)
- fix: replace wrong localhost:9092 defaults with localhost:19092 (#867)
- fix: remove vestigial ONEX_ENV and environment-prefixed topic names (#860)
- fix(docker): remove --admin-addr crash-loop and fix Memgraph healthcheck (#876)
- fix(mypy): resolve 25 pre-existing mypy errors, make CI job fully blocking (#895)

### Changed
- chore(deps): bump omnibase-core to 0.29.0, omnibase-spi to 0.18.0
- chore(deps): bump plugin pins (omninode-claude 0.9.0, omninode-memory 0.9.0, omninode-intelligence 0.15.0)
- refactor: replace legacy Consul ServiceTopicCatalog with contract-driven impl (#871)
- ci(omnibase_infra): add standards compliance workflow with blocking UP007 [std-sweep-v2] (#894)
- chore(deps): multiple Dependabot updates (trivy-action, setup-buildx, codeql-action, etc.)

## v0.20.0 (2026-03-13)

### Features
- feat(scripts): rehome cross-repo governance scripts from omni_home (#820)
- feat(runtime): add build_topic_router_from_contract() utility for per-event topic routing (#811)
- feat(runtime): add topic_router to DispatchResultApplier for per-event-type topic routing (#810)

### Bug Fixes
- fix(infra): expose Redpanda Admin API port 9644 in docker-compose (#819)
- fix(docker): bump Dockerfile.runtime plugin pins to latest releases (#818)
- fix(registration): wire topic_router into DispatchResultApplier from contract published_events (#812)
- fix(registration): short-circuit handler_node_heartbeat for terminal states (#799)
- fix(health): use mode="json" in readiness model_dump to convert tuples to lists (#815)
- fix(cleanup): purge dead endpoints and add skip markers (#805)
- fix(registration): add terminal-state guard to decide_heartbeat (#797)
- fix(runtime): eradicate inmemory default from ModelEventBusConfig and kernel (#809)
- fix(consul): remove Consul handler and add recurrence-prevention (#804)
- feat(cleanup): remove Ollama handlers, registries, and adapter (#808)

### Other Changes
- docs(linear-relay): add activation guide for Linear snapshot automation (#823)
- docs(pr-poller): add activation guide for GitHub PR poller effect node (#822)
- docs(validation): add activation guide for validation orchestrator (#824)
- docs(registration): update stale workaround comment now that topic routing is fixed (#814)
- test(registration): add regression test asserting ModelNodeRegistrationAccepted routes to correct topic (#813)
- ci: add published_events consistency checker to pre-commit and CI (#816)
- ci(standards): add version pin compliance check (#803)
- test(registration): integration test for liveness expiry -> heartbeat race (#800)

## v0.19.0 (2026-03-13)

### Features
- feat(ci): add placeholder topic denylist to prevent stub topic names (#802)
- feat(deploy): k8s-pod-readiness-check, verify-omnidash-health, VirtioFS gate, fatal health check (#789)
- feat(deploy): add preflight-check.sh with env var and bus tunnel gates (#788)

### Bug Fixes
- fix(monitoring): alert on terminal-state heartbeat warning in monitor_logs.py (#801)
- fix(types): modernize pre-PEP604 Union type annotation to X | Y syntax (#798)
- fix(ci): extend kafka-no-hardcoded-fallback to catch private-IP endpoints (#796)
- fix(docker): remove overlayfs-incompatible nodes bind mount (#787)
- fix(migrations): backfill NULL checksums + enforce NOT NULL on schema_migrations (#790)
- fix(migrations): create role_omniweb with DML grants for omniweb tables (#791)

### Other Changes
- test(registration): failing tests for decide_heartbeat terminal-state guard (#794)
- chore(version): add version source sync check script (#792)
- chore(topics): add placeholder topic denylist script (#793)
- test(ci): add x-runtime-env anchor regression tests (#795)

## v0.18.0 (2026-03-12)

### Features
- feat(topics): wire TopicProvisioner to ContractTopicExtractor (transitional union) (#780)
- feat(topics): add --skills-root flag to create_kafka_topics.py (#781)
- feat(topics): add contract topic parity gate CI script (#783)
- feat(runtime): wire OMNICLAUDE_SKILLS_ROOT to TopicProvisioner at startup (#784)
- feat(topics): add extract_from_skill_manifests and extend extract_all (#779)
- feat(topics): provision onex.evt.omniclaude.fix-transition.v1 topic (#777)

### Bug Fixes
- fix(health): mark skill-lifecycle-consumer healthy when lag=0 and polls current (#775)
- fix(hygiene): block operational artifact commits (#776)

### Other Changes
- test(topics): replace count-based assertions with structural guards (#782)
- docs(env): document OMNICLAUDE_SKILLS_ROOT in env-example-full.txt (#785)

# Changelog

All notable changes to the ONEX Infrastructure (omnibase_infra) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.16.1] - 2026-03-09

### Added
- HandlerResourceManager stub for httpx client lifecycle (#730)
- NodeMergeGateEffect + migration (#659)
- Full consul removal (#723)
- Degraded status detailed diagnostics in /health endpoint (#664)
- Artifact reconciliation ORCHESTRATOR node (#710)
- coerce_message_category boundary normalization helper (#708)
- Artifact reconcile CLI command (#703)
- CI guard against duplicate shared enum definitions (#707)
- ONEX handler classification rules design doc (#705)
- Structural move of non-node dirs out of nodes/ (#700)
- Update Plan REDUCER Node — FSM + HandlerCreatePlan (#704)
- GitHub Action workflow and PR webhook publisher script (#702)
- Change Detector EFFECT node with three handlers (#692)
- Domain event models for artifact reconciliation (#688)
- Artifact Registry Models + Loader (#687)
- NodeDeltaBundleEffect + NodeDeltaMetricsEffect + migrations (#660)
- Comprehensive contract schema validation (#667)
- Impact Analyzer COMPUTE Node (#691)
- RetryWorker for subscription notification delivery (#669)
- ServiceEffectMockRegistry with thread-local utility (#670)
- Seed artifact registry with 15 real omnibase_infra artifacts (#690)
- Shared Enum Ownership Rule architecture docs (#714)
- Regression tests for enum class-identity split (#715)
- config_prefetch_status exposed in /health endpoint (#686)
- CI check for x-runtime-env completeness (#684)
- CI invariant for node contract discoverability (#685)
- Observability and documentation for operation bindings (#671)
- Parameterize reducer purity tests for all reducers (#674)
- Observability tests for performance metrics (#668)
- Decouple prefetch contract scan from handler contract paths (#683)
- USE_EVENT_ROUTING added to docker-compose passthrough (#682)
- Close topic-constants-vs-generated-enums contract drift (#676)
- add infisical_folders check to system_health_check.sh (#694)

### Fixed
- schema-tolerant parsing for legacy fixture messages in skill-lifecycle-consumer (#729)
- Trigger idempotency and forward migration runner for warm Postgres volumes (#728)
- Restore _get_route_dispatcher_id shim for handler_id compat (#727)
- Cloud env runtime fixes: contracts in Docker, ProtocolEventBusPublisher, snappy, dispatch shim (#718)
- Create 041_create_agent_trace_tables.sql, fix sequence validator (#725)
- Wire omniintelligence migration runner to docker-compose (#724)
- Add 038_placeholder.sql to document intentional gap (#722)
- Override OMNIBASE_INFRA_DB_URL to Docker-internal hostname for containers (#721)
- Coerce category input in get_dispatchers() for foreign-enum safety (#720)
- Skip test_health_endpoint_accessible when Postgres unavailable in CI (#697)
- make provision-infisical.py folder creation idempotent (#698)
- Remove duplicate EnumMessageCategory from omnibase_infra (#701)
- Replace sleep-based wait with deterministic signal in E2E tests (#673)
- Replace hardcoded __version__ with importlib.metadata (#679)

### Changed
- Extract coerce_message_category to _enum_coercion to break circular import (#719)
- Coerce EnumMessageCategory at RegistryDispatcher boundary call sites (#716)
- ServiceTopicCatalogPostgres renamed to HandlerTopicCatalogPostgres (#717)
- 3.2b classification: projector mixins KEEP AS MIXIN (#712)
- 3.3 classification: MixinLlmHttpTransport scores 5/5, deferred (#711)
- 3.2a classification: MixinAsyncCircuitBreaker + MixinRetryExecution KEEP AS MIXIN (#709)
- Rename architecture_validator and contract_registry_reducer to node_ prefix (#695)
- Wire contract validation gate into omnibase_infra CI (#699)
- Mark docker-integration-tests as continue-on-error in CI (#696)
- Split release.yml into critical publish + advisory post-release checks (#677)
- Refactor CI workflow to use composite action for Python/uv setup (#675)
- Add validate-string-versions pre-commit hook (#680)
- Investigate non-node dirs under nodes/ (#693)
- POC 3.1 outcome: postgres mixins classified as KEEP AS MIXIN (#706)
- skill lifecycle writer: tolerate old-schema messages (#713)

## [0.16.0] - 2026-03-07

### Added
- Instrument runtime with Phoenix OTEL traces (#655)
- Registry-first startup topic assertions for event bus (#649)
- Canonical system health gate script (#650)
- Wire omnidash read-model migrations into bootstrap Step 1d (#646)
- Boot-order migration sentinel (#645)
- Handler pooling for parallel execution (#619)
- Batch response publishing to RuntimeHostProcess (#618)
- Parallel handler execution with asyncio concurrency (#617)
- Per-handler shutdown timeouts (#613)
- Enhanced error context with stack traces and suggestions (#615)
- AST-based cosmetic change filter for writer-migration gate (#623)
- Topic completeness check script (#620)
- WARNING_PATTERN alerting for known recurring warnings (#610)
- RestartWatcher thread for restart-loop detection (#609)
- Validate-kafka-schema-handshake CI gate with --changed-only (#602)
- Pre-commit validator blocks duplicate migration sequence numbers (#601)
- Writer-migration coupling gate (#598)
- Migration-integration job to CI gate (#592)
- NodeSetupOrchestrator — handler, node, contract, registry (#591)
- onex-setup.py interactive CLI with cloud gate output (#595)
- Provision cross-repo tables script + bootstrap wire-in (#594)
- Kafka-no-hardcoded-fallback pre-commit guard (#593)
- Health monitor with Slack alerts for runners (#641)
- Untagged image prune to Docker cron (#637)
- Cloud bus guard pre-commit hook (#652)
- No-planning-docs pre-commit hook (#611)

### Fixed
- Use venv Python for torch verification in Docker builder stage (#656)
- Idle-aware health check for skill-lifecycle-consumer (#653)
- Convert emitted_at to datetime for asyncpg (#648)
- Reduce Docker image size with CPU-only torch + cleanup (#640)
- Replace broken migration runner with fingerprint stamp in entrypoint (#643)
- restamp_fingerprint() calls installed module instead of missing script (#642)
- Remove SLACK_WEBHOOK_URL fallback, enforce Web API-only (#616)
- Run-loop diagnostics for worker exit-code-0 debugging (#614)
- Reorder shutdown to stop runtime before unsubscribing consumers (#612)
- Install all Docker plugins with --no-deps to prevent core version downgrade (#624)
- Pin qdrant-client>=1.16.0 and add grpcio>=1.62.0 lower bound (#580)
- Bump build-and-push-runtime timeout 45-60 min (#600)
- Upgrade protobuf to clear CVE-2026-0994 (#599)
- Pin Trivy to v0.69.3 (#596)
- Pin actions/checkout@v4 and actions/setup-python@v5 (#654)
- Pin torch CPU-only in Dockerfile.runtime (#636)

### Changed
- Remove Consul entirely from omnibase_infra runtime (#588)
- Purge cloud bus (29092) references from omnibase_infra (#647)
- Scale CI runners 5->10 + update Docker image threshold (#644)
- Migrate Docker workflows to self-hosted runners (#639)
- CI resilience fixes (#621)

### Chores
- Fix pre-existing AI-slop violations for --strict mode (#622)
- Remove dead _REAL_SCHEMA_FILE variable from test (#608)
- Bump docker/build-push-action from 5 to 6 (#625)
- Bump actions/upload-artifact from 4 to 7 (#628)
- Bump astral-sh/setup-uv from 3 to 7 (#626)
- Bump actions/checkout from 4 to 6 (#629)
- Bump actions/setup-python from 5 to 6 (#631)
- Update types-aiofiles requirement (#627)
- Update ruff requirement (#630)
- Update uvicorn requirement (#632)
- Update opentelemetry-instrumentation requirement (#633)
- Update textual requirement (#634)

### Tests
- E2E integration tests for NodeSetupOrchestrator (#597)
- Tighten warning assertion in test_unexpected_error_logged_as_warning (#607)
- Verify advisory lock executes inside transaction block (#606)
- Remove xfail from intent boundary pair (#604)
- Cover advisory lock failure path in schema init tests (#605)
- Unit tests for schema init advisory lock (#603)

## [0.14.0] - 2026-03-03

### Added
- Migrate event bus to local Docker Redpanda (#557)
- `ONEX_REGISTRATION_AUTO_ACK` direct-publish ack command (#558)
- `TCB_OUTCOME_REGISTRATION` topic (#560)
- Wire `NodeBaselinesBatchCompute` to daily scheduler (#564)
- Autoheal sidecar for aiokafka stuck-state recovery (#553)
- Keycloak service + postgres init script (#534)
- `provision-keycloak.py` + `update_env_file` utility (#537)
- Keycloak step 3.5 in `bootstrap-infisical.sh` (#538)
- Infisical auth transport folder and Keycloak vars documentation (#536)
- Kafka consumer → Linear ticket reporter in container log monitor (#539)
- PostgreSQL error event emitter in monitor (#535)
- Container log monitor with Slack alerts (#524)
- Golden-path fixture for `node_ledger_projection_compute` (#540)
- Golden-path fixture for `node_validation_ledger_projection_compute` (#541)
- Golden-path fixture for `node_validation_orchestrator` (#543)
- Arch-invariants CI gate for raw topic literals (#533)
- Hardcode broker addresses in docker-compose (#549)
- `sync-omnibase-env.py` with 5-guard TDD implementation (#512)
- Topic naming lint extended to Python enum files (#514)
- `update-plugin-pins.py` to pin omninode plugins to latest PyPI versions (#523)
- Kafka broker allowlist validator at `ServiceKernel.bootstrap()` (#530)
- Topic naming linter pre-commit + CI gate (#503)
- Self-hosted GitHub Actions runners — Dockerfile, compose, deploy script (#521)
- Conditional self-hosted runner routing in CI workflows (#522)
- Cross-repo Kafka boundary compat test (#515)

### Fixed
- **Remove `reconnect_backoff_ms`/`reconnect_backoff_max_ms` kwargs unsupported by aiokafka==0.11.0** (#508, #511) — resolves emit daemon crash on startup
- Update `last_poll_at` on `TimeoutError` to prevent false 503 after Kafka reconnect (#554)
- Wire `TimeoutCoordinator` into `HandlerRuntimeTick` (#556)
- Redesign health rule 5 — distinguish idle vs failing consumer (#552)
- Kafka topic drift — register missing publish_topics in `node_registration_orchestrator` (#546)
- Kafka topic drift — remove orphaned subscribe_topic from `node_baselines_batch_compute` (#544)
- Kafka topic drift — register cross-repo validation publish_topics (#548)
- Permissive ingest model for routing-decision schema drift (#551)
- Omniweb Keycloak client redirect URI (#550)
- `provision-keycloak.py` compatibility for Keycloak 26 (#563)
- Monitor: VALKEY_PASSWORD forwarded to Redis dedup auth (#562)
- Monitor: CodeRabbit review findings addressed (#542)
- Isolate `config_prefetcher` from host env in integration tests (#559)
- Handlers: consul contract, db DSN resolution, MCP `kafka_enabled` default (#509)
- Handlers: graph handler signature mismatch and filesystem `allowed_paths` (#510)
- TUI: push `ScreenStatus` via `push_screen()` instead of `compose()` (#505)
- Compose: remove nested variable expansion, add CI guard (#513)
- Docker: bump `omninode-intelligence` pin to 0.9.0 (#532)
- Remove stale local Redpanda from compose + fix `ONEX_ENVIRONMENT` default (#525)
- Fix `invalid_blocks` Slack API error in `monitor_logs` (#526)
- Extend `RUNTIME_SERVICES` to all 7 services (#519)
- Self-hosted runner routing labels (#531)
- Switch omninode-claude/memory to range pins with `--no-deps` (#506)
- Guard cross-repo dispatch steps against missing `CROSS_REPO_PAT` (#504)

### Changed
- Gate local Redpanda behind `local-redpanda` compose profile (#528)
- Cross-repo schema handshake gate for `routing-decision.v1` (#555)
- Automate `version_compatibility.py` matrix updates (#507)
- Expose `ONEX_REGISTRATION_AUTO_ACK` to runtime containers (#561)

### Reverted
- Remove wrong k3s manifests accidentally merged in #569 (#570)

## [0.13.0] - 2026-02-28

### Added
- Contract-driven Kafka topic creator script `create_kafka_topics.py` (#488)
- `TopicEnumGenerator` — per-producer enum rendering (#487)
- `ContractTopicExtractor` for contract-driven topic parsing (#486)
- `generate_topic_enums.py` script and initial generated enum files (#490)
- AI-slop checker Phase 2 rollout (#491)
- Catalog responder for `topic-catalog-request.v1` (#469)
- `NodeBaselinesBatchCompute` EFFECT node (#497)
- Skill lifecycle consumer and topic provisioning (#475)
- Consumer for manifest injection lifecycle events (#481)
- Decision-recorded topics to intelligence provisioning registry (#477)
- Reconnect backoff kwargs wired to AIOKafkaProducer/Consumer sites (#467)
- `reconnect_backoff_ms`/`max_ms` to `ModelKafkaEventBusConfig` (#466)
- Configurable Redpanda memory and connection limits (#465)
- Canonical `ModelRewardAssignedEvent` with policy signal fields (#470)
- `omninode-claude` plugin install in `Dockerfile.runtime` (#498)
- E2E automated regression for contract-driven topic enum pipeline (#499)

### Fixed
- Add 21 missing omnimemory topics to provisioning registry (#480)
- Add `agent-observability` DLQ topic to provisioning registry (#484)
- Gate omnimemory topic provisioning behind `OMNIMEMORY_ENABLED` flag (#479)
- Resolve 503 health check and DLQ validation failures in agent-actions (#494)
- Extend contract discovery to find `contract_*.yaml` files (#496)
- CAS-atomic topic subscriber index writes in Consul (#483)
- Correct `TOPIC_SESSION_OUTCOME_CANONICAL` producer segment to `omniclaude` (#476)
- Correct gmail-archive-purged topic name hyphen in producer segment (#473)
- Retire orphan `policy-state-updated` topic constant (#474)
- Retire orphan `run-evaluated` topic and stale model (#471)
- Remove stale `run_evaluated` capability from registry (#472)
- Replace stub `ModelScoreVector` with canonical omnibase_core model (#468)
- Update handler count assertions for `HandlerCatalogRequest` (#485)
- Show full correlation UUID in Slack context block (#489)
- Remove stale omninode_bridge comment from docker-build workflow (#482)
- Tune AI-slop checker v1.0 — scope `step_narration` to markdown only (#500)

### Changed
- Renamed `PYPI_PRIVATE_*` secrets to `PYPI_*` for public PyPI (#495)

### Dependencies
- `omnibase-core` bumped to >=0.22.0,<0.23.0 (was >=0.21.0,<0.22.0); git source override removed
- `omnibase-spi` bumped to >=0.15.0,<0.16.0 (was >=0.14.0,<0.15.0)

## [0.12.0] - 2026-02-27

### Changed
- Version bump as part of coordinated OmniNode platform release run release-20260227-eceed7

### Dependencies
- omnibase-core bumped to >=0.21.0,<0.22.0 (was >=0.20.0,<0.21.0); git source override removed
- omnibase-spi bumped to >=0.14.0,<0.15.0 (was >=0.13.0,<0.14.0)

## [0.11.0] - 2026-02-25

### Added

#### Event Bus Registry

- **Replace Consul discovery with event bus registry queries**: `HandlerMcpRegistryEffect` now queries the event bus registry instead of Consul for service discovery, removing Consul as a hard dependency for MCP-04 discovery flows (#421)

#### Topic Catalog PostgreSQL Backend

- **Replace `ServiceTopicCatalog` Consul KV backend with PostgreSQL**: Topic catalog persistence migrated from Consul KV to PostgreSQL, eliminating Consul as a runtime dependency for topic catalog operations (#422)

#### Runtime Observability

- **Runtime source-hash and compose-project startup banner**: Services now log a structured startup banner including source hash, compose project name, and environment at boot time for improved traceability (#412)

#### Deployment Safety

- **Detect compose project name collisions in `deploy-runtime.sh`**: The deploy script now detects and rejects duplicate compose project names before starting services, preventing silent container conflicts (#413)

### Changed

- Bumped version to 0.11.0

### Tests

- **Adversarial fingerprint CI twins**: Tests that prove fingerprint CI twins catch drift between runtime and test environments (#414)

### Documentation

- **ADR: two-handler-system architecture**: Decision record documenting the dual-handler pattern for protocol binding separation (#411)

## [0.10.0] - 2026-02-23

### Added

#### Zero-Repo-Env Policy

- **`scripts/register-repo.py`** — central Infisical onboarding CLI with `seed-shared` and `onboard-repo` subcommands; dry-run by default, `--execute` required to write; replaces ~80 lines of hardcoded secret declarations with YAML-driven loading (#387, #400)
- **`config/shared_key_registry.yaml`** — versioned authoritative registry of 39 shared platform keys across 8 transport folders (`db`, `kafka`, `consul`, `vault`, `llm`, `auth`, `valkey`, `env`); single source of truth replacing the hardcoded `SHARED_PLATFORM_SECRETS` dict (#393, #400)
- **`contract_config_extractor.py`**: extended `_TRANSPORT_ALIASES` to cover 13 previously unmapped keys (#387)
- **Pre-commit hook**: rejects `.env` files anywhere in the repo tree; `.env` removed from the allowed root file list, enforcing the zero-repo-env policy (#388, #389)

#### LLM Metrics Observability

- **`ServiceLlmMetricsPublisher`** — service-layer wrapper around `HandlerLlmOpenaiCompatible` that reads `last_call_metrics` after each inference call and publishes to `onex.evt.omniintelligence.llm-call-completed.v1`; fixes zero-data `/cost-trends` dashboard (#390)
- **`register_openai_compatible_with_metrics()`** and **`register_ollama_with_metrics()`** factory methods on `RegistryInfraLlmInferenceEffect` for wiring the publisher at container bootstrap time (#390)

### Fixed

- **`ConfigSessionStorage`** (session): removed `env_prefix="OMNIBASE_INFRA_SESSION_STORAGE_"` so the config reads standard `POSTGRES_*` vars rather than the non-existent prefixed variants (#391)
- **`config_store.py`**: set `env_file=None` to prevent stale `.env` file reads after zero-repo-env migration (#400)

### Changed

#### Dependencies

- **Bump `omnibase-core`** from `>=0.18.1,<0.19.0` → `>=0.19.0,<0.20.0` (DecisionRecord, NodeReducer projection effect)
- **Bump `omnibase-spi`** from `>=0.10.0,<0.11.0` → `>=0.12.0,<0.13.0` (ProtocolEffect, ProtocolNodeProjectionEffect, ContractProjectionResult)
- **Bump `omniintelligence`** from `0.4.0` → `0.5.0` in `docker/Dockerfile.runtime` (#386)
- Bumped version to 0.10.0

## [0.9.0] - 2026-02-20

### Added

#### OmniMemory Topics

- `platform_topic_suffixes` — OmniMemory Kafka topic suffix constants (`store`, `retrieve`, `retrieved`, `delete`, `deleted`, `search`, `search_results`, `error`) with package exports and full unit test coverage (#383)

#### Topic Catalog

- Topic catalog change notification emission with CAS (compare-and-swap) versioning — catalog mutations now emit `TopicCatalogChangedEvent` with a version vector for optimistic concurrency control (#379)
- Topic catalog response warnings channel — catalog query responses now carry a `warnings` field for non-fatal advisory messages (e.g. deprecated topic references, schema drift) (#377)

#### LLM-Driven Code Generation Handlers

- `HandlerCodeReviewAnalysis` — code review analysis handler via Coder-14B LLM, producing structured review results from git diff input (#376)
- `HandlerTestBoilerplateGeneration` — test boilerplate generation handler via Coder-14B LLM, scaffolding pytest unit tests from source signatures (#375)

#### Tests

- Unit tests for `NodeLedgerWriteEffect` handlers — full coverage of ledger write effect handler behaviour including error paths (#382)
- Topic catalog multi-client no-cross-talk E2E test — validates that concurrent catalog clients do not observe each other's in-flight mutations (#378)

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

- `MixinLlmHttpTransport` for structured LLM HTTP calls with sanitized response bodies, case-insensitive content-type handling, and locked client teardown (#320, #322)
- `HandlerLlmOpenaiCompatible` for OpenAI wire-format inference (chat completions, embeddings) against local vLLM/Ollama-compatible servers (#325)
- `HandlerLlmOllama` with node scaffold for Ollama-native inference (#328)
- `node_llm_embedding_effect` with models, handlers, node, contract, and registry for embedding extraction (#327)
- `ModelLlmInferenceRequest` and `ModelLlmMessage` for typed LLM request construction (#321)
- `ModelLlmInferenceResponse` with `text XOR tool_calls` invariant enforcement (#324)
- `ModelLlmShared` — shared LLM models for inference and embedding nodes (#318)
- Inference node assembly with contract, registry, and operation validation (#335)
- LLM endpoint health checker service with per-endpoint liveness probes (#352)
- LLM endpoint SLO profiling and load test scaffolding (#347)
- CIDR allowlist and HMAC request signing on LLM HTTP transport (#350)
- Inference handler unit tests (#331)
- Inference model validation tests (#334)
- Embedding node unit tests (#329)
- `MixinLlmHttpTransport` unit tests (#337)

#### LLM Cost Tracking

- Token usage extraction and normalization from LLM API responses (#346)
- LLM cost aggregation service with per-session and per-call rollups (#348)
- Static context token cost attribution for system prompt overhead (#361)
- `ModelPricingTable` with YAML manifest and cost estimation utilities (#360)
- `llm_call_metrics` and `llm_cost_aggregates` database migration 031 (#343)
- LLM cost tracking input validation and edge case tests (#358)
- Integrate SPI 0.9.0 LLM cost tracking contracts (#345)
- SPI LLM protocol adapters for `ProtocolLlmCostTracker` and `ProtocolLlmPricingTable` (#353)

#### Enrichment Handlers

- `HandlerCodeAnalysisEnrichment` for git diff analysis via Coder-14B LLM (#363)
- Embedding similarity enrichment handler for vector-based context relevance scoring (#366)
- Context summarization enrichment handler for token-efficient context compression (#367)
- Documentation generation handler via Qwen-72B for automated doc synthesis (#371)

#### Topic Catalog

- Topic Catalog model and suffix foundation (#357)
- `ServiceTopicCatalog` with KV (Valkey) precedence and in-memory caching (#370)
- Topic catalog query handler, dispatcher, and contract wiring (#372)

#### Baselines and Effectiveness Metrics

- A/B baseline comparison compute node with delta scoring (#332)
- Batch compute effectiveness metrics and cache invalidation notifier (#362)
- Baselines tables and batch compute service with Postgres persistence (#369)

#### Secret Management — Infisical Backend

- Infisical secret backend: adapter, handler, and config resolution layer (#355)
- Contract-driven config discovery, Infisical seed script, and bootstrap orchestration (#359)
- Remove Vault handler; migrate all secret resolution references to Infisical (#368)

#### Schema and Event Registry Integrity

- Schema fingerprint manifest with startup assertion gate (#317)
- Event registry fingerprint with startup assertion gate (#326)
- CI twins for schema and event registry fingerprint drift detection (#338)
- Full check catalog, artifact storage, and flake detection (#330)

#### Runtime and Bootstrap

- Bootstrap attestation gate in kernel handshake phase (#336)
- Runtime contract routing verification tests and demo (#312)
- Install `omniintelligence` in runtime Docker image (#323)
- Stable runtime deployment script for repeatable container launches (#340)
- Intelligence topic provisioning; bump omniintelligence to 0.2.0 (#342)
- Set `OMNIINTELLIGENCE_PUBLISH_INTROSPECTION` on `omninode-runtime` only (#364)

#### Demo and Test Tooling

- Demo loop assertion gate for canonical event loop validation (#349)
- Demo reset scoped command for safe environment reset between runs (#354)

#### Error Taxonomy

- `InfraRateLimitedError` exception class added to infrastructure error hierarchy (#315)

#### Registration

- Implement `reduce_confirmation()` for registration reducer (#319)
- Reducer-authoritative registration with E2E integration follow-ups (#316)

### Fixed

- Consumer group instance discriminator for multi-container dev environments (#351)
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

- Required status checks added to branch protection rules (#333)
- Extract duplicated rules from CLAUDE.md to shared config (#339)

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

#### Slack Webhook Handler
- **HandlerSlackWebhook**: Async handler with Block Kit formatting, retry with exponential backoff, and 429 rate limit handling
- **NodeSlackAlerterEffect**: Pure declarative effect node for Slack alerts
- **EnumAlertSeverity**: Severity levels (critical/error/warning/info)
- **ModelSlackAlert/ModelSlackAlertResult**: Type-safe frozen Pydantic models
- Features: Correlation ID tracking, exponential backoff retry (1s → 2s → 4s), 429 rate limit handling

#### Contract Dependency Resolution
- **ContractDependencyResolver**: Reads protocol dependencies from `contract.yaml` and resolves from container
- **ModelResolvedDependencies**: Pydantic model for resolved protocol instances
- **ProtocolDependencyResolutionError**: Fail-fast error for missing protocols
- **RuntimeHostProcess integration**: Automatic dependency resolution during node discovery
- Zero-code nodes can now receive injected dependencies via constructor

#### Event Ledger Integration Tests
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

- Fix ORDER BY injection position when LIMIT clause exists in `PostgresRepositoryRuntime` (#229)
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
- Update `omnibase-core` from `^0.9.9` to `^0.9.10` for contract-driven topics

## [0.2.6] - 2026-01-30

### Added

#### Contract Registry System
- `KafkaContractSource` for cache-based contract discovery from Kafka topics (#213)
- Contract registry reducer with Postgres projection for persistent contract storage (#212)

#### Event Ledger Persistence
- `NodeLedgerProjectionCompute` for event ledger persistence with compute node pattern (#211)
- PostgreSQL handlers for event ledger persistence operations (#209)
- Event ledger schema and models for tracking event processing state (#208)

#### Declarative Configuration & Routing
- `RuntimeContractConfigLoader` for declarative operation bindings from contract.yaml (#210)
- Declarative topic→operation→handler routing with contract-driven dispatch (#198)
- Contract-driven event bus subscription wiring for automatic topic binding (#200)

#### Emit Daemon
- Emit daemon for persistent Kafka connections with connection pooling (#207)

#### Kafka & Event Bus Improvements
- Event bus topic storage in registry for dynamic topic routing (#199)
- Derived Kafka consumer group IDs with deterministic naming (#197)
- Replace hardcoded topics with validated suffix constants (#206)

#### Handler & Intent Improvements
- Intent storage effect node with integration tests (#195)
- `execute()` dispatcher to `HandlerGraph` for contract discovery (#193)
- Canonical publish interface ADR and test adapter (#201)

### Changed

#### Dependencies
- **omnibase-core**: Updated from ^0.9.6 to ^0.9.9 (baseline topic constants export)
- **omnibase-spi**: Updated from ^0.6.2 to ^0.6.3

## [0.2.3] - 2026-01-25

### Added

#### Infrastructure Primitives for Atomic Operations
- `write_atomic_bytes()` / `write_atomic_bytes_async()` for crash-safe file writes with temp file + rename pattern
- `transaction_context()` async context manager with configurable isolation levels, read-only/deferrable options, and per-transaction timeouts
- `retry_on_optimistic_conflict()` decorator/helper with exponential backoff, jitter, and attempt tracking
- Comprehensive test coverage (103 unit tests) for all new utilities

#### Intent Handler Routing (Demo)
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

#### File and Class Naming Standardization

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

#### MixinNodeIntrospection API

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

#### Error Code for Unhandled node_kind
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

#### Dependency Updates
- **omnibase_core upgraded to 0.7.0**: Breaking changes in core dependency
- **omnibase_spi upgraded to 0.5.0**: Breaking changes in SPI dependency
- **pytest-asyncio 0.25+ compatibility**: Test framework compatibility updates, requires `asyncio_mode = "auto"` in pyproject.toml
- **Infrastructure IP defaults changed to localhost**: Default infrastructure IPs changed from remote server to localhost for local development

#### Error Handling
- **RuntimeError replaced with structured domain errors**: All generic `RuntimeError` raises have been replaced with specific domain errors from the error taxonomy. If you were catching `RuntimeError`, update to catch the specific error types:
  - `ProtocolConfigurationError` for configuration issues
  - `InfraConnectionError` for connection failures
  - `InfraTimeoutError` for timeout issues
  - `InfraUnavailableError` for unavailable resources

### Added

#### Node Introspection
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
- **Protocol Patterns Documentation**: Added comprehensive documentation for protocol design patterns, cross-mixin composition, and TYPE_CHECKING patterns in `docs/patterns/protocol_patterns.md`

#### Testing
- **Correlation ID Integration Tests**: Added integration tests for correlation ID propagation across service boundaries

#### Handlers
- **HttpHandler**: HTTP REST protocol handler for MVP
  - GET and POST operations using httpx async client
  - Fixed 30s timeout (configurable timeout deferred to Beta)
  - Returns `EnumHandlerType.HTTP`
  - Error handling mapping to infrastructure errors (`InfraTimeoutError`, `InfraConnectionError`)
  - Full lifecycle support (initialize, shutdown, health_check, describe)
  - 46 unit tests with 97.93% coverage

#### Event Bus
- **InMemoryEventBus**: In-memory event bus for local development and testing
  - Implements `ProtocolEventBus` from omnibase_core
  - Topic-based pub/sub with `asyncio.Queue` per topic
  - Thread-safe subscription management
  - Automatic cleanup on unsubscribe
  - Consumer groups with load balancing
  - Graceful shutdown with message draining
  - Comprehensive error handling
  - 1336+ lines of test coverage

#### Runtime
- **ProtocolBindingRegistry**: Handler and event bus registration system
  - Single source of truth for handler registration
  - Thread-safe registration operations
  - Support for handler type constants (HTTP, DATABASE, KAFKA, etc.)
  - Event bus registry (InMemory, Kafka)
  - Protocol resolution utilities

#### Errors
- **Infrastructure Error Taxonomy**: Structured error hierarchy
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
- **Directory Structure**: Initial MVP directory structure
  - `handlers/`: Protocol handler implementations
  - `event_bus/`: Event bus implementations
  - `runtime/`: Runtime host components
  - `errors/`: Infrastructure error classes
  - `enums/`: Infrastructure enumerations
  - `validation/`: Contract validation utilities

### Changed

#### Handler to Dispatcher Terminology Migration

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
