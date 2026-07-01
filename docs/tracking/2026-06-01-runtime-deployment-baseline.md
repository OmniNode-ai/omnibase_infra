# Phase 1 — Live Runtime Deployment Baseline & Evidence

- Date captured: 2026-06-01
- Mode: STRICTLY READ-ONLY. No runtime mutation, no deploy/build/restart, no runtime host mutation, no empty commits.
- Design doc: `omni_home/docs/plans/2026-06-01-node-based-runtime-deployment-occ-tdd.md`
- Live host: `<onex-host>` (reachable from this machine; `ssh <user>@<onex-host>` → `omninode-pc`).
- Probe method: read-only `docker ps`, `docker inspect`, `docker exec <broker> rpk group list`, and `curl /health` / `/v1/introspection/manifest` per lane.

Each item below is structured as **claim / evidence (command + output excerpt) / source**, tagged
`[live-verified]` (probed on the runtime host) or `[from-source]` (read from the repo working tree).

---

## Item 1 — `node_redeploy` manifest (contract, handlers, FSM, topics)

### Claim
`omnimarket/src/omnimarket/nodes/node_redeploy` is a live FSM-driven WorkflowPackage
(`node_type: workflow`, contract v2.0.0) that owns the deploy lifecycle and reaches the
deploy agent only over Kafka. It is the only deployment orchestrator and must be *extended*,
not duplicated.

### Evidence — contract `[from-source]`
`omnimarket/src/omnimarket/nodes/node_redeploy/contract.yaml`:
- `name: node_redeploy`, `contract_version: {major: 2, minor: 0, patch: 0}`, `node_type: workflow`,
  `node_version: {major: 2, minor: 0, patch: 0}`.
- `descriptor.node_archetype: workflow`, `purity: side_effect`, `idempotent: false`,
  `timeout_ms: 660000`, `runtime_profiles: [effects]`.
- `terminal_event: onex.evt.omnimarket.redeploy-completed.v1`.

`event_bus` block (verbatim topics):
```text
subscribe_topics:
  - onex.cmd.omnimarket.redeploy-start.v1
  - onex.evt.deploy.rebuild-completed.v1          # cross-service: deploy agent completion
publish_topics:
  - onex.evt.omnimarket.redeploy-phase-transition.v1
  - onex.evt.omnimarket.redeploy-completed.v1
  - onex.cmd.deploy.rebuild-requested.v1          # cross-service: deploy agent rebuild command
```

Inputs (contract): `scope` (default `full`), `git_ref` (default `origin/main`), `services`,
`versions`, `skip_sync`, `verify_only`, `dry_run`, `requested_by`.

### Evidence — three handlers `[from-source]`
`omnimarket/src/omnimarket/nodes/node_redeploy/handlers/` (verbatim contract `handler_routing`):
- `handler_workflow_runner.py` → `HandlerRedeployWorkflowRunner`, operation `run_workflow`:
  "drives FSM through all phases, invoking Kafka rebuild at REBUILD phase."
- `handler_redeploy.py` → `HandlerRedeploy`, operation `fsm_transition`:
  "FSM state machine — phase transitions, circuit breaker, terminal detection." Pure logic, no I/O.
  Circuit breaker trips at `max_consecutive_failures` (default 3) → `FAILED`.
- `handler_redeploy_kafka.py` → `HandlerRedeployKafka`, operation `kafka_rebuild`:
  EFFECT publish-monitor. Publishes `onex.cmd.deploy.rebuild-requested.v1`, subscribes
  `onex.evt.deploy.rebuild-completed.v1`, matches on `correlation_id`, default timeout 600s,
  poll interval 2.0s. Module docstring: "never SSHes, never calls rpk directly, no subprocess —
  pure event bus publish-monitor." The deploy-agent boundary is therefore already Kafka-only.

### Evidence — FSM phases `[from-source]`
`omnimarket/src/omnimarket/nodes/node_redeploy/models/model_redeploy_state.py`,
`EnumRedeployPhase` + `_PHASE_SEQUENCE`:
```text
IDLE -> SYNC_CLONES -> UPDATE_PINS -> REBUILD -> SEED_INFISICAL -> VERIFY_HEALTH -> DONE
TERMINAL_PHASES = {DONE, FAILED}
```
`handler_workflow_runner.py`: only the `REBUILD` phase invokes `HandlerRedeployKafka`; the deploy-agent
phases (`SYNC_CLONES`, `UPDATE_PINS`, `SEED_INFISICAL`, `VERIFY_HEALTH`) advance with `success=True`
because the deploy agent reports failure via `rebuild_result`, not the FSM circuit breaker.

### Evidence — live consumer groups `[live-verified]`
`docker exec omnibase-infra-redpanda rpk group list` (dev-lane broker) shows both
`node_redeploy` subscriptions `Stable`:
```text
local.omnimarket.node_redeploy.consume.2.0.0.__t.onex.cmd.omnimarket.redeploy-start.v1            Stable
local.omnimarket.node_redeploy.consume.2.0.0.__t.onex.evt.deploy.rebuild-completed.v1             Stable
```
Stability-test broker (`omnibase-infra-stability-test-redpanda`) and prod broker
(`omnibase-infra-prod-redpanda`) each carry the same two `node_redeploy` groups `Stable`:
```text
stability-test.omnimarket.node_redeploy.consume.2.0.0.__i.stability-test-effects.__t.onex.cmd.omnimarket.redeploy-start.v1   Stable
stability-test.omnimarket.node_redeploy.consume.2.0.0.__i.stability-test-effects.__t.onex.evt.deploy.rebuild-completed.v1    Stable
prod.omnimarket.node_redeploy.consume.2.0.0.__i.prod-effects.__t.onex.cmd.omnimarket.redeploy-start.v1                       Stable
prod.omnimarket.node_redeploy.consume.2.0.0.__i.prod-effects.__t.onex.evt.deploy.rebuild-completed.v1                        Stable
```

### Source
`omnimarket/src/omnimarket/nodes/node_redeploy/{contract.yaml,handlers/*,models/model_redeploy_state.py}`;
live runtime host redpanda brokers (dev/stability/prod).

**Verdict: CONFIRMED.** `node_redeploy` is the live, single deployment orchestrator on all three
lanes; deploy-agent reach is already Kafka-only via `handler_redeploy_kafka`.

---

## Item 2 — Live consumer groups for the reused nodes

### Claim (from design doc)
`node_runtime_sweep`, `node_runtime_source_attestor_effect`, and `node_runtime_manifest_reducer`
are live (Stable). `node_deployment_evidence_reducer`, `node_occ_pr_writer_effect`,
`node_readiness_gate_orchestrator`, and `node_evidence_pipeline_orchestrator` have **no live
consumer group**; only `node_evidence_dashboard_effect` observes their topics.

### Evidence — present / Stable (dev-lane broker) `[live-verified]`
`docker exec omnibase-infra-redpanda rpk group list` (420 groups total; 351 `Stable`, 68 `Empty`):
```text
local.omnimarket.runtime_sweep.consume.1.0.0.__t.onex.cmd.omnimarket.runtime-sweep-start.v1                               Stable
local.omnibase_infra.node_runtime_source_attestor_effect.consume.1.0.0.__t.onex.evt.omnibase-infra.runtime-booted.v1      Stable
local.omnibase_infra.node_runtime_manifest_reducer.consume.1.0.0.__t.onex.evt.omnibase-infra.runtime-manifest-published.v1 Stable
local.omnibase_infra.node_runtime_error_triage_effect.consume.1.0.0.__t.onex.evt.omnibase-infra.runtime-error.v1          Stable
```

### Evidence — absent / not actively consuming `[live-verified]`
On the **dev-lane** broker, a targeted grep for `deployment_evidence`, `occ_pr_writer`,
`readiness_gate`, and `evidence_pipeline` returns **no `node_*` consumer groups** for those four
nodes. The only consumer observing their topics is `node_evidence_dashboard_effect` (all `Stable`):
```text
local.omnimarket.node_evidence_dashboard_effect.consume.1.0.0.__t.onex.cmd.omnimarket.evidence-pipeline-start.v1     Stable
local.omnimarket.node_evidence_dashboard_effect.consume.1.0.0.__t.onex.cmd.omnimarket.readiness-gate-start.v1        Stable
local.omnimarket.node_evidence_dashboard_effect.consume.1.0.0.__t.onex.evt.omnimarket.evidence-collected.v1          Stable
local.omnimarket.node_evidence_dashboard_effect.consume.1.0.0.__t.onex.evt.omnimarket.evidence-extracted.v1          Stable
local.omnimarket.node_evidence_dashboard_effect.consume.1.0.0.__t.onex.evt.omnimarket.evidence-pipeline-completed.v1 Stable
local.omnimarket.node_evidence_dashboard_effect.consume.1.0.0.__t.onex.evt.omnimarket.evidence-validated.v1          Stable
local.omnimarket.node_evidence_dashboard_effect.consume.1.0.0.__t.onex.evt.omnimarket.occ-pr-created.v1              Stable
local.omnimarket.node_evidence_dashboard_effect.consume.1.0.0.__t.onex.evt.omnimarket.readiness-gate-blocked.v1      Stable
local.omnimarket.node_evidence_dashboard_effect.consume.1.0.0.__t.onex.evt.omnimarket.readiness-gate-completed.v1    Stable
```

### Evidence — REFINEMENT: those nodes ARE registered on the stability-test broker, but `Empty` `[live-verified]`
`docker exec omnibase-infra-stability-test-redpanda rpk group list`:
```text
stability-test.omnimarket.node_evidence_pipeline_orchestrator.consume.1.0.0.__i.stability-test-effects.__t.onex.cmd.omnimarket.evidence-pipeline-start.v1   Empty
stability-test.omnimarket.node_occ_pr_writer_effect.consume.1.0.0.__i.stability-test-effects.__t.onex.evt.omnimarket.evidence-validated.v1                  Empty
stability-test.omnimarket.node_readiness_gate_orchestrator.consume.1.0.0.__i.stability-test-effects.__t.onex.cmd.omnimarket.readiness-gate-start.v1          Empty
stability-test.omnimarket.node_readiness_gate_orchestrator.consume.1.0.0.__i.stability-test-effects.__t.onex.evt.omnimarket.readiness-scored.v1             Empty
```
`Empty` = the consumer group exists (the node registered its subscription) but has no active member
currently joined / consuming. These are **registered but not actively consuming**, which is a
materially stronger statement than "no consumer group at all," but they are still NOT live
(no Stable member, no inbound traffic). On the **prod** broker, none of these four appear at all
(only `node_redeploy` and `runtime_sweep` consumer groups exist).

### Evidence — `node_deployment_evidence_reducer` absent on ALL three brokers `[live-verified]`
```text
for b in omnibase-infra-redpanda omnibase-infra-stability-test-redpanda omnibase-infra-prod-redpanda; do
  docker exec "$b" rpk group list | grep -i deployment_evidence  -> (none)
done
```
No consumer group for `node_deployment_evidence_reducer` on dev, stability-test, or prod.

### Source
Live runtime host redpanda brokers (dev/stability/prod), `2026-06-01`.

**Verdict on the design doc's B3 claim: CONFIRMED with one refinement.**
- Live (Stable): `node_runtime_sweep`, `node_runtime_source_attestor_effect`,
  `node_runtime_manifest_reducer` (and `node_runtime_error_triage_effect`). ✅
- NOT live: `node_occ_pr_writer_effect`, `node_readiness_gate_orchestrator`,
  `node_evidence_pipeline_orchestrator` — confirmed no Stable consumer; the only Stable observer
  of their topics is `node_evidence_dashboard_effect`. ✅
- **Refinement:** the three OCC/readiness/evidence nodes ARE registered on the stability-test
  broker (state `Empty`), not entirely absent. They are absent from dev and prod brokers.
- `node_deployment_evidence_reducer` — confirmed absent on all three brokers (a node to *wire*,
  not reuse). ✅

---

## Item 3 — Prod / stability / dev runtime versions + image DIGEST drift

### Claim (design doc, the motivating gap)
prod is on runtime `0.36.1`; dev and stability-test are on `0.37.0` — production runs a different
artifact than the one proven in stability.

### Evidence — `/health` version per lane `[live-verified]`
```text
curl http://localhost:8085/health   (dev)            -> "version":"0.37.0", environment "local",          subscriber_count 231, topic_count 204
curl http://localhost:18085/health  (stability-test) -> "version":"0.37.0", environment "stability-test",  subscriber_count 249, topic_count 214
curl http://localhost:28085/health  (prod)           -> "version":"0.36.1", environment "prod",            subscriber_count 222, topic_count 197
```
All three report `"status":"healthy"`, `event_bus.healthy:true`, `circuit_state:"closed"`.

### Evidence — image DIGEST drift (the authoritative proof) `[live-verified]`
`docker inspect --format '{{.Config.Image}} | ImageID={{.Image}}'` on the running runtime containers:
```text
omninode-runtime (dev)                                              ImageID=sha256:e3ea08e10957ec9e01dba8ee616b76971fbc8a75dee518e3688f84cb80296611
omninode-stability-test-runtime.pre-final-redeploy-...20260529...   ImageID=sha256:e3ea08e10957ec9e01dba8ee616b76971fbc8a75dee518e3688f84cb80296611
omninode-prod-runtime                                               ImageID=sha256:ed93a1337ebe03a6334303c17243a226f8c870c4399810bbb59532b1ee54d856
```
`docker inspect` RepoDigests corroborate:
```text
runtime:latest                          -> @sha256:e3ea08e10957ec9e01dba8ee616b76971fbc8a75dee518e3688f84cb80296611
omnibase-infra-prod-omninode-runtime    -> @sha256:ed93a1337ebe03a6334303c17243a226f8c870c4399810bbb59532b1ee54d856
```

### Evidence — container ages `[live-verified]`
`docker ps`: prod runtime containers `Up 7 days`; dev runtime `Up 34 hours`; the stability-test
runtime containers are the `.pre-final-redeploy-20260529T091011Z` pair `Up 22 hours`.

### Evidence — `/v1/introspection/manifest` per lane `[live-verified]`
All three lanes respond (no 404). Excerpts:
- dev 8085 + stability 18085: first contract `ab_compare_reducer` (`node_type: reducer`).
- prod 28085: first contract `node_architecture_validator` (`node_type: COMPUTE_GENERIC` — the
  legacy `*_GENERIC` drift the design doc flags on older contracts; prod runs the older 0.36.1
  contract set).

### Source
Live runtime host containers + `/health` + `/v1/introspection/manifest`, `2026-06-01`.

**Verdict: CONFIRMED and strengthened.** Not only do version strings differ (prod 0.36.1 vs
dev/stability 0.37.0), the **image digests differ**: dev and stability share
`sha256:e3ea08e1…`, while prod runs `sha256:ed93a133…`. This is first-class live evidence for the
"production can run a different artifact than the one proven in stability" gap and justifies the
hard same-digest FSM gate. Note: the stability-test runtime is currently the
`.pre-final-redeploy-20260529T091011Z` container, not a freshly-named one — re-verify the canonical
stability container name before any future deploy.

---

## Item 4 — Runtime attestation surfaces

### Claim (design doc)
A live runtime-attestation pipeline already exists in `omnibase_infra`:
`node_runtime_source_attestor_effect` consumes `onex.evt.omnibase-infra.runtime-booted.v1`;
`node_runtime_manifest_reducer` consumes `onex.evt.omnibase-infra.runtime-manifest-published.v1`.
The proposed probe-effect proof fields overlap these; consume them rather than re-collect.

### Evidence — node sources exist `[from-source]`
```text
omnibase_infra/src/omnibase_infra/nodes/node_runtime_source_attestor_effect/  -> __init__.py contract.yaml handlers node.py
omnibase_infra/src/omnibase_infra/nodes/node_runtime_manifest_reducer/        -> __init__.py contract.yaml handlers models node.py
```

### Evidence — both consume their attestation topics, Stable `[live-verified]`
(dev-lane broker `rpk group list`):
```text
local.omnibase_infra.node_runtime_source_attestor_effect.consume.1.0.0.__t.onex.evt.omnibase-infra.runtime-booted.v1       Stable
local.omnibase_infra.node_runtime_manifest_reducer.consume.1.0.0.__t.onex.evt.omnibase-infra.runtime-manifest-published.v1 Stable
local.omnibase_infra.node_runtime_error_triage_effect.consume.1.0.0.__t.onex.evt.omnibase-infra.runtime-error.v1           Stable
```

### Source
`omnibase_infra/src/omnibase_infra/nodes/{node_runtime_source_attestor_effect,node_runtime_manifest_reducer}/`;
live runtime host dev-lane broker, `2026-06-01`.

**Verdict: CONFIRMED.** Both attestation consumers are live (Stable) on the dev-lane broker,
consuming `runtime-booted.v1` and `runtime-manifest-published.v1` respectively. The runtime-error
triage consumer is also live. Future probe work should consume these, not re-collect.

---

## Corroborating evidence for downstream blockers (read-only, from source)

These are not Phase-1 deliverables but were confirmed in passing and pin down blockers B1/B2
referenced by the design doc.

### B2 — deploy-agent boundary cannot express lane or digest `[from-source]`
`omnibase_infra/scripts/deploy-agent/deploy_agent/events.py`, `ModelRebuildRequested` (lines 85-102)
carries only: `correlation_id`, `requested_by`, `scope`, `build_source`, `services` (default `[]`),
`git_ref` (default `origin/main`). **No `runtime_lane`, no `image_digest`.** `ModelRebuildCompleted`
similarly carries no lane/digest fields. Topic constants: `onex.cmd.deploy.rebuild-requested.v1`,
`onex.evt.deploy.rebuild-completed.v1`, `onex.evt.deploy.rebuild-rejected.v1`.

`omnibase_infra/scripts/deploy-agent/deploy_agent/executor.py` hardcodes the dev lane:
```text
COMPOSE_FILE = f"{REPO_DIR}/docker/docker-compose.infra.yml"          # executor.py:39
COMPOSE_PROJECT = "omnibase-infra"                                    # executor.py:40
RUNTIME_HEALTH_TARGETS = (("omninode-runtime", 8085), ("runtime-effects", 8086))  # executor.py:54-56
```
The agent cannot target stability-test (18085 / `-stability-test`) or prod (28085 / `-prod`) and
cannot deploy a pinned digest. **Verdict: B2 CONFIRMED — lane/digest truth cannot be enforced until
this boundary carries them.**

### Note on CI bypass
Not independently re-probed in this read-only pass; the design doc's reference to
`runtime-rebuild-trigger.yml` / `trigger_rebuild_on_merge.py` hardcoding `origin/main` is recorded
here as **from-doc, re-verify in Phase 2** rather than asserted live.

---

## Summary table

| Surface | State | Source |
|---|---|---|
| `node_redeploy` (orchestrator) | Live, Stable, all 3 lanes; contract v2.0.0; FSM IDLE→…→DONE | live + source |
| deploy-agent boundary | Kafka-only via `handler_redeploy_kafka`; NO lane/digest fields (B2) | live + source |
| `node_runtime_sweep` | Live, Stable (dev/stability/prod) | live |
| `node_runtime_source_attestor_effect` | Live, Stable (runtime-booted.v1) | live |
| `node_runtime_manifest_reducer` | Live, Stable (runtime-manifest-published.v1) | live |
| `node_runtime_error_triage_effect` | Live, Stable (runtime-error.v1) | live |
| `node_evidence_pipeline_orchestrator` | NOT live — registered `Empty` on stability-test only | live |
| `node_occ_pr_writer_effect` | NOT live — registered `Empty` on stability-test only | live |
| `node_readiness_gate_orchestrator` | NOT live — registered `Empty` on stability-test only | live |
| `node_deployment_evidence_reducer` | NOT present on any broker — node to *wire* | live |
| `node_evidence_dashboard_effect` | Live, Stable; sole observer of OCC/readiness/evidence topics | live |
| Runtime versions | dev 0.37.0 / stability 0.37.0 / prod 0.36.1 | live |
| Image digests | dev+stability `sha256:e3ea08e1…`; prod `sha256:ed93a133…` (DRIFT) | live |
