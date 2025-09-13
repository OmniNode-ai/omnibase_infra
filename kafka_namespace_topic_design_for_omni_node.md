# Kafka Namespace & Topic Design for OmniNode

## Objective

Define a rigorous namespace and topic scheme for OmniNode that supports multi tenant operation, strict governance, replayability, privacy controls, and clean separation of concerns across products such as ONEX, CAIA, PRISM, INTELCRAWLER, and the Validator stack.

---

## Design Principles

1. Stable contracts first. Topics are an interface. Treat each like an API surface with versioned schemas and explicit SLAs.
2. Single responsibility per topic class. Event streams, commands, queries, and telemetry do not mix.
3. Deterministic partitioning. Keys carry domain identity. No random keys.
4. Safe replay. Compaction and retention tuned by stream class. Snapshots support backfills.
5. Tenant and environment isolation by namespace. No cross tenant topics.
6. Privacy and compliance by default. PII in value only, scoped encryption, redacted headers, and field level policies.
7. Explicit lifecycle. Every topic has an owner, SLA, retention, schema subject, and deprecation plan.

---

## Namespace Taxonomy

Use a five tier hierarchy to encode environment, tenant, bounded context, stream class, and version. This is compact, queryable, and future proof.

```
<env>.<tenant>.<context>.<class>.<topic>.<v>
```

**Fields**

- `env`: dev | stage | prod | sand | test
- `tenant`: omnibase | customer slug, lowercase kebab
- `context`: bounded context, lowercase kebab. Examples: onex, caia, prism, intelcrawler, validator, registry, memory, kb, security, gateway
- `class`: evt | cmd | qrs | ctl | dlt | rty | cdc | met | aud | log
- `topic`: domain aggregate or function, lowercase kebab
- `v`: vN semantic major version. Example: v1, v2

**Rationale**

- Wildcard friendly. `prod.*.onex.evt.*.*` matches all ONEX events across tenants.
- Clear separation of stream classes. `evt` is immutable fact, `cmd` is intent, `qrs` is request reply, `ctl` is control plane, `dlt` and `rty` are failure handling, `met` metrics, `aud` audit, `log` structured logs, `cdc` change data capture.

---

## Topic Classes and Policies

### 1) Event streams `evt`

Immutable facts. Append only. Use compaction only for latest by key snapshots if needed.

Retention: 7 to 30 days for hot, archive to tiered storage after. Critical ledgers keep 90 days.

Example topics

```
prod.omnibase.onex.evt.contract-created.v1
prod.omnibase.validator.evt.test-run-completed.v1
prod.omnibase.prism.evt.policy-updated.v1
prod.omnibase.intelcrawler.evt.crawl-finding.v1
prod.omnibase.memory.evt.context-stamped.v1
```

Partition key: aggregate id. Examples: `contract_id`, `tuple_id`, `policy_id`, `crawl_id`, `memory_key`.

### 2) Command streams `cmd`

Intent to change state. One partitioning rule per aggregate. Consumers are authoritative services.

Retention: short, 3 days. No compaction.

Example topics

```
prod.omnibase.onex.cmd.generate-node.v1
prod.omnibase.validator.cmd.run-suite.v1
prod.omnibase.registry.cmd.register-asset.v1
```

### 3) Query request response `qrs`

Synchronous over Kafka with correlation. Request and reply topics paired per context.

Retention: 24 hours. No compaction.

Example topics

```
prod.omnibase.kb.qrs.search-requests.v1
prod.omnibase.kb.qrs.search-replies.v1
```

Headers: `x-correlation-id`, `reply-to`, `timeout-ms`, `actor`, `scope`.

### 4) Control plane `ctl`

Cluster control and orchestration signals for agents and schedulers.

Retention: 48 hours. Compaction allowed.

Examples

```
prod.omnibase.orchestrator.ctl.lease-heartbeat.v1
prod.omnibase.orchestrator.ctl.task-schedule.v1
```

### 5) Retry `rty` and Dead letter `dlt`

One retry topic per source class and context with bounded backoff tiers. One DLT per context with strict triage policy.

Retention: `rty` 7 days, `dlt` 30 to 90 days.

Examples

```
prod.omnibase.validator.rty.test-run-completed.v1
prod.omnibase.validator.dlt.test-run-completed.v1
```

Message headers: `x-failure-class`, `x-failure-reason`, `x-attempt`, `x-first-seen-ts`, `x-last-seen-ts`.

### 6) CDC `cdc`

Database change streams per service. Schema bound to storage model, not domain model.

Examples

```
prod.omnibase.registry.cdc.assets.v1
prod.omnibase.memory.cdc.embeddings.v1
```

### 7) Metrics `met`, Audit `aud`, Logs `log`

Telemetry separation for privacy and volume control.

Examples

```
prod.omnibase.security.met.pii-redaction.v1
prod.omnibase.gateway.aud.request.v1
prod.omnibase.gateway.log.access.v1
```

---

## Schema Strategy

**Registry**: Confluent Schema Registry or compatible. Subject naming: `<topic>-value` and `<topic>-key` with version pinning. Avro or Protobuf preferred. JSON allowed for external bridges only.

**Versioning**

- Backward compatible changes increment minor in schema, topic version unchanged.
- Breaking changes create new topic version `vN+1`. Old topic in read only deprecation until backlog drains and consumers migrate.

**Headers**

- Minimum: `x-schema-version`, `x-correlation-id`, `x-trace-id`, `x-actor`, `x-tenant`, `x-scope`, `x-origin`, `x-encryption-scheme`, `x-pii-level`.

**Contracts**

- Each topic paired with a contract spec file under repo `contracts/<context>/<class>/<topic>/vN/` containing schema, ownership, SLA, retention, compaction, partitioning, key rules, retry, DLT policy, and sample payloads.

---

## Partitioning and Keys

Rules

1. Choose a key that preserves per aggregate ordering. Never use random UUID without domain meaning.
2. High cardinality keys for hot paths to avoid leader hotspots. Use shard keys if needed: `hash(domain_id) % N` in header, while keeping real id in value.
3. For fan out, publish parent fact once and derive downstream by processor, not by duplicating on multiple topics.

Examples

- `onex.evt.contract-created`: key `contract_id`
- `validator.evt.test-run-completed`: key `run_id`
- `prism.evt.policy-updated`: key `policy_id`

---

## Retention and Compaction Matrix

| Class | Retention | Compaction | Use                            |
| ----- | --------- | ---------- | ------------------------------ |
| evt   | 7 to 30d  | Optional   | Domain facts, backfill, replay |
| cmd   | 3d        | None       | Intent, idempotent handlers    |
| qrs   | 24h       | None       | Request reply                  |
| ctl   | 48h       | Optional   | Heartbeats, schedules          |
| rty   | 7d        | None       | Backoff queues                 |
| dlt   | 30 to 90d | None       | Triage, forensics              |
| cdc   | 7 to 14d  | Optional   | Storage changes                |
| met   | 7d        | Optional   | Counters, timings              |
| aud   | 90d       | Optional   | Security and compliance        |
| log   | 3 to 7d   | None       | High volume logs               |

Tiered storage enabled for `evt`, `aud`, and critical `ctl` streams.

---

## Security and Privacy

- ACLs by namespace level. Producers and consumers scoped to `<env>.<tenant>.<context>.<class>.*.*` at minimum.
- Field level PII classification in schema with `pii:{none|low|high}` annotations. High PII requires envelope encryption with KMS and redaction in `met`, `aud`, and `log`.
- Token bound headers: `x-actor`, `x-scope`, `x-tenant` validated by Gateway. Reject on mismatch.
- Non repudiation for critical streams with detached signatures in headers `x-sig-alg`, `x-sig`, and hash in `x-payload-sha256`.

---

## Failure Handling

- Three tier retry strategy per topic: `rty-1m`, `rty-15m`, `rty-2h` implemented as time wheel or delayed topics. Use a single `rty` topic with `x-next-at` for scheduler based delay if broker does not support delayed produce.
- DLT analysis workflow publishes summary reports to `aud` and metrics to `met`.
- Strict poison message policy. After N attempts, route to DLT with full envelope.

Example retry topics

```
prod.omnibase.validator.rty.test-run-completed.v1
prod.omnibase.validator.dlt.test-run-completed.v1
```

---

## Observability

- Tracing headers propagated end to end. Map to OpenTelemetry.
- Heartbeats on `ctl.lease-heartbeat`. Liveness SLOs per consumer group.
- Topic level dashboards for lag, throughput, error rate, and reprocessing count.
- Periodic canaries that publish schema compliant test messages to each `evt` topic. Validate consumer health without production data.

---

## Multi region and DR

- Active active across regions with MirrorMaker 2 or Cluster Linking. Namespace includes region code in header `x-region` rather than topic name to avoid combinatorial explosion.
- Conflict strategy: last writer wins per aggregate sequence with monotonic version in value. Cross region producers must reserve sequence ranges or use CRDT style merges for idempotent facts.

---

## Governance

- Topic RFC required for any new topic. Stored with the contract. Includes owner, PagerDuty rotation, data classification, SLOs, and deprecation gates.
- Linting CI enforces namespace pattern, schema subject naming, owner label, and retention tags before merge.
- Quarterly review to prune unused `cmd` and `qrs` topics and rotate keys for encrypted fields.

---

## Concrete Example Set for OmniNode

```
prod.omnibase.onex.evt.contract-created.v1
prod.omnibase.onex.evt.contract-updated.v1
prod.omnibase.onex.cmd.generate-node.v1
prod.omnibase.validator.evt.test-run-started.v1
prod.omnabase.validator.evt.test-run-completed.v1   # spelling check: tenant is omnibase
prod.omnibase.validator.cmd.run-suite.v1
prod.omnibase.registry.evt.asset-registered.v1
prod.omnibase.registry.cmd.register-asset.v1
prod.omnibase.prism.evt.policy-updated.v1
prod.omnibase.intelcrawler.evt.crawl-finding.v1
prod.omnibase.gateway.aud.request.v1
prod.omnibase.security.met.pii-redaction.v1
prod.omnibase.kb.qrs.search-requests.v1
prod.omnibase.kb.qrs.search-replies.v1
```

Note: fix the `omnabase` typo before provisioning.

---

## Migration and Deprecation

- Breaking schema change creates `vN+1` topic. Mirror events to both topics during migration window. Consumers must dual read until cutover.
- Deprecation checklist: zero lag, no producers, audit freeze, delete after archive.

---

## Starter Provisioning Checklist

1. Bootstrap registry for `onex`, `validator`, `registry`, `prism`, `intelcrawler`, `gateway`, `kb`, `memory` contexts.
2. Create baseline topics per context for classes `evt`, `cmd`, `ctl`, `aud`, `met`, and paired `qrs` when needed.
3. Set ACLs by namespace patterns and service accounts.
4. Register key and value schemas with annotation for PII.
5. Enable tiered storage for `evt` and `aud` families.
6. Wire CI lints for namespace, schema, retention, and ownership.
7. Ship canary publishers and consumer health checks.

---

## Open Questions

- Do we want per tenant Schema Registry or shared with subject isolation. Default to shared with strict subject naming.
- Should we include `org` above `tenant` for marketplace. Likely yes later. Omit for now.
- Do we n
