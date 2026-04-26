# Market Node Deployment Runbook

**Status:** documents current ad-hoc path as of 2026-04-25; gaps identified for closure
**Source plan:** docs/plans/2026-04-25-platform-hardening-and-correctness.md (Track F/F2)
**Source matrix:** docs/tracking/2026-04-25-market-node-health-matrix.md (OMN-9718)
**Linear ticket:** OMN-9727

---

## Overview

A "market node" is any node in `omnimarket/src/omnimarket/nodes/`. There are 135 nodes
as of the F0 audit. This runbook traces the full path from PR merge to a node actively
consuming from Kafka. Each step documents what actually happens today, who owns the
trigger, and where the code lives.

---

## Current Path (7 phases, evidence-cited)

### Phase 1 — PR merge to omnimarket main

**What happens:** A PR targeting `main` on `OmniNode-ai/omnimarket` is squash-merged
via the GitHub merge queue.

**Trigger owner:** GitHub merge queue (automated). Manual merges are prohibited by
branch protection.

**Code location:**
- Branch protection: `gh api repos/OmniNode-ai/omnimarket/branches/main/protection`
- Merge queue policy: documented in the shared CLAUDE.md (Merge Queue Policy section)

**Relevant CI jobs that run before merge (on every PR):**
- `ci.yml` — pytest, ruff, mypy
- `contract-validation.yml` — validates `contract.yaml` files
- `omni-standards-compliance.yml` — runs compliance sweep
- `deploy-gate.yml` — receipt-gate: requires OMN-XXXX ticket citation + dod_evidence
- `cr-thread-gate-caller.yml` — blocks merge if CodeRabbit threads unresolved

**Gap:** None at this phase. Merge is gated.

---

### Phase 2 — Image rebuild trigger

**What happens:** On a merged PR, the `runtime-rebuild-trigger.yml` workflow fires.
It evaluates whether the PR touched runtime-relevant files and, if so, publishes a
signed Kafka event to the cloud data-plane bus.

**Trigger owner:** GitHub Actions — `on: pull_request: types: [closed]` on the
`omnimarket` repo, gated by `if: github.event.pull_request.merged == true`.

**Code location:**
- Workflow: `omnimarket/.github/workflows/runtime-rebuild-trigger.yml`
- Decision script: `omnimarket/scripts/trigger_rebuild_on_merge.py`
- Topic published: `onex.cmd.deploy.rebuild-requested.v1` (via cloud SASL/SSL Kafka)
- Required secrets: `KAFKA_BOOTSTRAP_SERVERS`, `KAFKA_SASL_USERNAME`,
  `KAFKA_SASL_PASSWORD`, `DEPLOY_AGENT_HMAC_SECRET`

**Trigger conditions** (`trigger_rebuild_on_merge.py::should_trigger`):
- PR had the `runtime_change` label, **OR**
- Any changed file matches `src/omnimarket/*` or `src/omnibase_infra/nodes/*`

**Payload:** `ModelRebuildRequested` (from `deploy_agent/events.py`):
```text
correlation_id, requested_by, scope="runtime", git_ref="origin/main"
```

**HMAC signing:** the payload is signed with `DEPLOY_AGENT_HMAC_SECRET` before
publishing. The deploy-agent verifies the signature on receipt.

**GAP — trigger fires on Kafka cloud bus, not local Docker bus:**
The trigger script publishes to `KAFKA_BOOTSTRAP_SERVERS` (cloud Confluent endpoint).
The deploy-agent on .201 must be configured to consume from the same cloud bus
(or a bridge). There is no evidence in the codebase that the deploy-agent is
configured to consume from the cloud bus vs the local Docker Redpanda. This is
a known issue: OMN-9713 tracks "deploy-agent localhost" — the deploy-agent was
consuming from `localhost:19092` (Docker Redpanda) while the trigger was publishing
to the cloud bus. Any PR touching `src/omnimarket/**` would silently never reach
the deploy-agent.

---

### Phase 3 — Image deploy to .201

**What happens:** The deploy-agent on `.201` receives the `rebuild-requested` event,
runs `git pull` on `/data/omninode/omnibase_infra`, regenerates the compose file,
builds the Docker image, and brings runtime containers up.

**Trigger owner:** deploy-agent — a Python service running as a systemd user service
on `.201` (`scripts/deploy-agent/deploy/deploy-agent.service`).

**Code location:**
- Service definition: `omnibase_infra/scripts/deploy-agent/deploy/deploy-agent.service`
- Consumer: `scripts/deploy-agent/deploy_agent/consumer.py` — polls
  `onex.cmd.deploy.rebuild-requested.v1`, verifies HMAC, validates schema, deduplicates
- Executor: `scripts/deploy-agent/deploy_agent/executor.py`
- Hardcoded repo path on .201: `REPO_DIR = "/data/omninode/omnibase_infra"` (executor.py:33)

**Executor rebuild phases** (Scope.RUNTIME path):
1. **Self-update:** pulls latest deploy-agent source from git; re-execs itself if behind
2. **Preflight:** verifies git remote is reachable + Docker is available
3. **Git:** `git fetch --all --prune` + `git reset --hard origin/main` on `REPO_DIR`
4. **Compose gen:** regenerates `docker-compose.infra.yml` from the catalog CLI:
   `uv run python -m omnibase_infra.docker.catalog.cli generate core runtime`
5. **Seed:** runs `seed-infisical.py` to populate Infisical before containers start
6. **Build:** `docker compose build --build-arg GIT_SHA=<sha>` (invalidates COPY src/ layer)
7. **Up:** `docker compose up -d --force-recreate --no-deps <runtime-services>`

**Runtime services restarted** (from `events.py::SCOPE_SERVICES[Scope.RUNTIME]`):
```text
omninode-runtime, runtime-effects, runtime-worker, agent-actions-consumer,
skill-lifecycle-consumer, context-audit-consumer, intelligence-migration,
intelligence-api, omninode-contract-resolver, autoheal
```

**GAP — omnimarket is installed from git@main, not from local REPO_DIR:**
The Dockerfile.runtime installs omnimarket with:
```bash
uv pip install --no-deps "git+https://github.com/OmniNode-ai/omnimarket.git@main"
```
The executor pulls `omnibase_infra` from `/data/omninode/omnibase_infra`, but omnimarket
nodes are fetched fresh from GitHub at image build time. This means:
- An omnimarket PR merged → triggers rebuild → omnimarket code is re-fetched from
  GitHub `main` at build time. This is correct when the PR has landed on main.
- However, there is no explicit wait between the PR merge webhook and the rebuild
  trigger. If GitHub's internal propagation is slow, the Dockerfile `git+...@main`
  fetch could grab the pre-merge commit.

**GAP — no confirmation that rebuild actually completed:**
The rebuild trigger publishes the event and exits. There is no feedback loop from
the deploy-agent back to the PR or CI. An operator must check deploy-agent logs
manually to confirm a successful rebuild.

---

### Phase 4 — Container restart

**What happens:** `docker compose up -d --force-recreate --no-deps <runtime-services>`
recreates the runtime containers using the freshly built image.

**Trigger owner:** deploy-agent executor (`_compose_up` in `executor.py`).

**Verification:** The executor polls `docker ps` for up to 120s to confirm containers
reach `running` state. If containers are stuck in `Created`, it attempts `docker start`
recovery. A second poll confirms recovery.

**Code location:** `executor.py::verify_containers_up`, `_compose_up`

**Health check on container:** The `omninode-runtime` service manifest
(`docker/catalog/services/omninode-runtime.yaml`) declares:
```yaml
healthcheck:
  test: curl -sf http://localhost:8085/health
  interval_s: 30
  timeout_s: 10
  retries: 3
  start_period_s: 90
```

**GAP — deploy-agent verification checks wrong ports:**
`executor.py::verify()` checks `http://localhost:8000/health` (LLM port),
`http://localhost:8001/health` (LLM port), and `http://localhost:8002/health`.
The real runtime health endpoints are `8085` (omninode-runtime) and
`8086` (runtime-effects). This is a documented gap from the E3 addendum in the
platform plan. The verification phase may report "pass" while the actual runtime
is unhealthy.

---

### Phase 5 — Handler registration

**What happens:** When `omninode-runtime` starts, it calls `onex-runtime` (the
`omnibase_infra.runtime.kernel:main` entrypoint). The kernel runs the auto-wiring
engine which:
1. Scans all installed packages for `onex.nodes` entry points
   (`importlib.metadata.entry_points(group="onex.nodes")`)
2. Locates the sibling `contract.yaml` for each entry point
3. Wires handlers declared in the contract into the DI container

**Trigger owner:** Runtime startup — automatic on container start.

**Code location:**
- Discovery: `src/omnibase_infra/runtime/auto_wiring/discovery.py`
  (`ENTRY_POINT_GROUP = "onex.nodes"`, `discover_contracts()`)
- Wiring: `src/omnibase_infra/runtime/auto_wiring/wiring.py` (`wire_from_manifest`)
- Kernel boot: `src/omnibase_infra/runtime/service_kernel.py` (calls `wire_from_manifest`
  around line 2400)

**Entry point prerequisite:** For a market node to be discovered, it MUST be registered
in `omnimarket/pyproject.toml` under `[project.entry-points."onex.nodes"]`. The F0
audit found 5 BROKEN-NEW nodes missing this registration (repaired by F1).

**GAP — no registration verification after startup:**
The deploy-agent's `verify()` phase checks `handler_registry` row count in Postgres
(`SELECT count(*) FROM handler_registry`). It only checks `count > 0`, not that
the expected set of nodes is registered. A node whose entry point fails to import
(e.g., due to a missing dependency) is silently dropped; the count check still passes
because other nodes registered successfully.

---

### Phase 6 — Consumer group subscription

**What happens:** Nodes with `event_bus.subscribe_topics` declared in their
`contract.yaml` automatically subscribe to those Kafka topics when the runtime
auto-wiring registers them. The Kafka consumer group ID is derived from the node's
contract (typically `local.omnimarket.<node-name>`).

**Trigger owner:** Runtime auto-wiring + Kafka consumer group creation — automatic
on handler registration.

**Code location:**
- Contract field: `contract.yaml::event_bus.subscribe_topics`
- Consumer group ID: convention `local.omnimarket.<node-name>` (from contract or
  `ONEX_GROUP_ID` env var override)
- Runtime profile env: `ONEX_GROUP_ID=onex-runtime-main` (global fallback in
  `omninode-runtime.yaml`)

**GAP — no expected consumer group registry:**
There is no authoritative list of which consumer groups should exist after a
successful deploy. The E1a baseline check (Track E) proposes reading from an
`expected_consumer_groups.yaml` file, but this file does not exist yet. Without
it, there is no automated way to diff `rpk group list` output against expectations.
The only current verification is `rpk group list | grep local.omnimarket` — which
requires SSH to .201 and manual inspection.

**GAP — topic pattern match triggers trigger but not deploy:**
The trigger script matches `src/omnimarket/*` (single-level glob). A node added
at `src/omnimarket/nodes/new_node/handler.py` matches `src/omnimarket/*` (because
`fnmatch` with `*` matches path separators when the prefix matches). This works
for the trigger condition. However, there is no guarantee the new node's Kafka
topics were pre-created on Redpanda before the runtime tries to subscribe. Redpanda
auto-creates topics by default, but if `auto.create.topics.enable=false`, the
subscription silently fails.

---

### Phase 7 — Round-trip verification

**What happens:** After the runtime is up and handlers are registered, there is
**no automated round-trip verification** that any specific market node is actually
consuming correctly. The deploy-agent's `verify()` phase checks:
1. No unhealthy Docker containers
2. No restarting Docker containers
3. `handler_registry` row count > 0 in Postgres
4. HTTP health on ports 8000, 8001, 8002 (wrong ports — see Step 4 gap)

None of these checks send a synthetic event and await a node-specific response.

**Trigger owner:** None for per-node round-trip. Manual only.

**GAP — no per-node smoke test:**
There is no mechanism (CI, launchd, or deploy-agent) that sends a synthetic event
tagged `synthetic=true` to each node's subscribe topic and waits for the expected
publish event. This is the core gap that allowed OMN-9695 (15 broken nodes) to go
undetected for 6+ days after merges. F4 (Track F, Wave 3) proposes a 30m canary,
but it is not yet implemented.

---

## Gaps Identified

| # | Gap | Severity | Evidence | Recommended fix |
|---|-----|----------|----------|-----------------|
| 1 | Deploy-agent consuming from wrong Kafka bus (localhost vs cloud) | HIGH | OMN-9713; executor.py hardcodes `localhost:19092`; trigger publishes to cloud SASL endpoint | Fix deploy-agent Kafka config to match trigger bus; add preflight broker-reachability check |
| 2 | No per-node round-trip verification after deploy | HIGH | executor.py verify() checks only count>0 and wrong ports; OMN-9695 undetected 6 days | Implement F4 canary: synthetic event → expected response per node |
| 3 | Deploy-agent health check probes wrong ports (8000/8001/8002 vs 8085/8086) | HIGH | executor.py:636-654 hardcodes LLM ports; runtime health is on 8085/8086 | Fix executor.py verify() to probe 8085 and 8086; check `details.config_prefetch_status=ok` |
| 4 | No expected consumer group registry | HIGH | No `expected_consumer_groups.yaml` exists; rpk group list diff is manual | Create registry from F0 matrix; wire into E1a baseline check |
| 5 | No feedback loop from deploy-agent to PR/CI | MED | trigger_rebuild_on_merge.py exits after publish; no status returned | Publish `rebuild-completed` event back; add GHA job that polls for completion |
| 6 | handler_registry count check is not node-specific | MED | executor.py verify() only checks count>0; a single registered node passes | Query expected set from contract.yaml discovery; diff against actual registry |
| 7 | Redpanda topic auto-creation not verified | LOW | No `auto.create.topics.enable` check in trigger or deploy; silent subscribe failure possible | Add topic-existence pre-check to deploy executor or create topics in deploy pipeline |

---

## Highest-leverage gap (proposed for immediate closure)

**Gap #3 — Deploy-agent health check probes wrong ports** is the most actionable
immediate fix because it is a single-file code change with high impact: it makes
the deploy-agent's `verify()` phase actually test the real runtime health surface
instead of the LLM inference ports. This closes the scenario where a deploy
completes with a green verify result while `omninode-runtime` is actually down or
in `config_prefetch_status=degraded_error`.

**Proposed fix scope:**
- `scripts/deploy-agent/deploy_agent/executor.py`: change the health check loop at
  lines 636-654 to probe `http://localhost:8085/health` and `http://localhost:8086/health`
  instead of ports 8000/8001/8002
- Add a check that the health response body contains `config_prefetch_status` equal
  to `ok` (not just HTTP 200), matching the E3b spec in the platform plan
- Add a test in `scripts/deploy-agent/tests/` that mocks the health endpoints and
  asserts the verifier correctly distinguishes `config_prefetch_status=ok` from
  `config_prefetch_status=degraded_error`

**Why not Gap #1 (bus mismatch)?** Gap #1 (OMN-9713) is already tracked by the
runtime team and gated on their deliverable. It cannot be fixed here without
SSH access to .201 and changes to the live deploy-agent configuration.

**Why not Gap #2 (no round-trip canary)?** Gap #2 (F4 canary) is Wave 3 work
gated on Track B completion. It is the right long-term fix but is not immediately
actionable from a static code-only PR.

---

## Manual deployment path (current fallback)

When the deploy-agent is unreachable (e.g., OMN-9713 bus mismatch), the current
manual path on `.201` is (requires `INFRA_HOST` and `INFRA_USER` from `~/.omnibase/.env`):

```bash
# 1. SSH to the infra host
ssh ${INFRA_USER}@${INFRA_HOST}

# 2. Pull latest omnibase_infra (path is hardcoded in deploy-agent executor.py)
git -C /data/omninode/omnibase_infra pull --ff-only

# 3. Run deploy-runtime.sh from the infra clone
cd /data/omninode/omnibase_infra
./scripts/deploy-runtime.sh --execute --restart

# 4. Verify runtime health (ports 8085/8086 — NOT 8000/8001 which are LLM ports)
curl -sf http://localhost:8085/health | jq .details.config_prefetch_status
curl -sf http://localhost:8086/health | jq .details.config_prefetch_status

# 5. Verify consumer groups (broker address from KAFKA_BOOTSTRAP_SERVERS in .env)
rpk group list --brokers ${KAFKA_BOOTSTRAP_SERVERS:-localhost:19092} | grep local.omnimarket
```

`deploy-runtime.sh` rsyncs the repo to `~/.omnibase/infra/deployed/<version>/`,
builds the Docker image with VCS_REF label, and optionally restarts runtime containers.
It does NOT install omnimarket from the local clone — the Dockerfile always fetches
omnimarket from `git+https://github.com/OmniNode-ai/omnimarket.git@main` at build time.
