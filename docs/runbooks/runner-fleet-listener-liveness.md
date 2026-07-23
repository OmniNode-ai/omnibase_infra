# Runner fleet listener liveness (OMN-13915)

**Status:** active runbook
**Ticket:** OMN-13915 (incident 2026-07-03) — related: OMN-12433 (egress healthcheck), OMN-13109 (silent wedge / crash loop monitor)

## The rule that changed

> **"All runner containers are `Up (healthy)`" is NOT sufficient evidence that the fleet is serving jobs.**
> The GitHub org runner registry (`GET /orgs/OmniNode-ai/actions/runners`) is the authoritative signal, and the `runner-fleet-canary` scheduled workflow is the enforced surface that watches it.

## Incident summary (2026-07-03)

- GitHub org runners API: **37 offline / 11 online** across the 48-runner self-hosted fleet on `.201`.
- `docker ps` on `.201`: **all 48** runner containers `Up X days (healthy)`.
- `omninode-runner-1` logs: listener ran jobs normally until 2026-06-29 22:47Z, then went silent — the `Runner.Listener` process died inside the container, the `run.sh` wrapper tree stayed alive, the entrypoint never saw an exit code, and the container-level healthcheck stayed green for four days.
- Org-wide CI backlog reached 150+ queued runs once volume outran the 11 survivors.

## Failure mode

A point-in-time process/container check cannot prove a runner is serving jobs:

1. **Container liveness ≠ listener liveness.** Containers created before the healthcheck stanza existed keep their creation-time (or absent) healthcheck. The healthcheck *definition* is captured at container creation; syncing `healthcheck.sh` on disk changes behavior only for containers whose definition already invokes it.
2. **Listener process presence ≠ listener registration.** A hung/zombied listener, or a dead listener under a still-alive wrapper tree, passes loose `pgrep` checks.
3. **Host-side monitors share fate with the host.** `runner-monitor.sh` (cron on `.201`) detects Docker-healthy-vs-GitHub-offline divergence, but if its cron, env, or Slack path is broken, nothing notices the gap (same class as OMN-13909).

## Detection layers (after OMN-13915)

| Layer | Surface | What it proves | Latency |
|-------|---------|----------------|---------|
| 1 | `docker/runners/healthcheck.sh` (in-container) | `bin/Runner.Listener` process alive AND `_diag` heartbeat fresh (`RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS`, default 900s) AND github.com egress | ≤ ~17 min (staleness + 3×30s retries) |
| 2 | `docker/runners/entrypoint.sh` watchdog | listener process exists while `run.sh` runs; recycles the wrapper tree after 5×60s consecutive misses (bounded by `LISTENER_RESTART_MAX=50`, then container exit → restart policy). **OMN-14564:** also recycles (with an explicit listener kill) when the listener process is alive but its `_diag` heartbeat is stale for `LISTENER_HEARTBEAT_MISSES` (3) consecutive 60s ticks — never while a `Runner.Worker` job is executing | ≤ ~6 min (dead) / ≤ ~18 min (hung: 900s staleness + 3×60s) |
| 3 | `runner-monitor.sh` cron on `.201` (OMN-13109) | Docker vs GitHub divergence, silent wedge, crash loop → Slack | 3 min cadence, shares fate with host |
| 4 | **`runner-fleet-canary` GHA workflow (authoritative)** | GitHub org registry online count vs `config/runner_fleet.yaml` `expected_count`; fails the run when offline+missing > `RUNNER_CANARY_MAX_OFFLINE` (5) | 15 min cadence, GitHub-hosted — survives total `.201` loss |

## Hung-listener mode (OMN-14564, incident 2026-07-16..23)

A second zombie variant the OMN-13915 process-existence watchdog cannot catch:
the `Runner.Listener` process stays **alive** but deadlocks inside its
AAD/OAuth token-refresh HTTP call while acknowledging a broker job assignment
(runner v2.334.0, `disableUpdate: true`; terminal `_diag` line is
`AAD Correlation ID for this token request: Unknown` after
`BrokerServer SocketException(125)` long-poll churn). 11/64 runners sat
GitHub-offline for ~6 days: `pgrep` green, `_diag` silent, no exit code, no
respawn. Docker health (layer 1) correctly flagged all 11 — but detection
without remediation left them zombied until a manual idle-only restart.

The layer-2 watchdog now treats *listener-alive-but-heartbeat-stale* as a
listener death: same `_diag` staleness condition as the healthcheck (shared
tunable `RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS`, default 900s), guarded by a
`Runner.Worker` check so an executing job is never killed, with an explicit
`pkill` of the listener binary on recycle (a hung listener ignores the
wrapper-tree TERM and would collide with the respawned listener's session).
Restarts remain bounded by `LISTENER_RESTART_MAX`.

## Operator response to a canary failure

1. Read the failed `runner-fleet-canary` run summary — it lists offline runner names.
2. Do **NOT** `docker restart` runners (crash-loops: cached creds + expired baked token — OMN-13109).
3. Safe bounce, named services only, fresh token, detached:
   ```bash
   # on .201, from ~/.omnibase/runners
   TOKEN=$(gh api --method POST /orgs/OmniNode-ai/actions/runners/registration-token --jq .token)
   RUNNER_TOKEN="$TOKEN" timeout 120 \
     docker compose -f docker/docker-compose.runners.yml up -d --force-recreate --no-deps <omninode-runner-N ...>
   ```
4. Confirm recovery via the org API (`gh api /orgs/OmniNode-ai/actions/runners --jq '[.runners[]|select(.status=="online")]|length'`), **not** via `docker ps`.

## Verifying the healthcheck catches a dead listener (synthetic kill)

Run against a **test** container (never the live fleet mid-use):

```bash
docker exec <runner> pkill -f 'bin/Runner\.Listener'
# within ~6 min the entrypoint watchdog logs "WATCHDOG: listener dead-in-container" and recycles;
# if the watchdog is disabled, within ~16-17 min the container flips to (unhealthy).
docker inspect --format '{{.State.Health.Status}}' <runner>
```

The offline equivalent (no container needed) is covered by
`tests/ci/test_runner_listener_liveness.py`, which spawns a synthetic
`bin/Runner.Listener` process, kills it, and asserts `healthcheck.sh` flips
from exit 0 to exit 1.

## Rollout notes

- `healthcheck.sh` and `entrypoint.sh` are bind-mounted `:ro` from the compose dir on `.201` (`~/.omnibase/runners/docker/runners/`). Syncing files updates healthcheck *script behavior* immediately (it is re-exec'd each interval) but entrypoint changes and healthcheck *definition* changes require a force-recreate of each service.
- Recreate with the safe bounce recipe above (fresh token, named services, never `docker restart`).
- The canary needs the `RUNNER_FLEET_STATUS_TOKEN` repo/org secret (classic PAT `admin:org` read, or fine-grained org "Self-hosted runners: read"); until it is seeded the canary falls back to `CROSS_REPO_PAT` and fails loudly if that token lacks the scope.
