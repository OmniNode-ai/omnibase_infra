# Runner fleet listener liveness (OMN-13915)

**Status:** active runbook
**Ticket:** OMN-13915 (incident 2026-07-03) — related: OMN-12433 (egress healthcheck), OMN-13109 (silent wedge / crash loop monitor), OMN-15233 (2026-07-27 threshold recalibration + orphan/crash-loop detection), OMN-15255 (composite readiness + quarantine gate), OMN-15776 (broker-dispatch/reconnect race — a distinct GitHub-side failure class, targeted rerun remediation)

## The rule that changed

> **"All runner containers are `Up (healthy)`" is NOT sufficient evidence that the fleet is serving jobs.**
> The GitHub org runner registry (`GET /orgs/OmniNode-ai/actions/runners`) is the authoritative signal, and the `runner-fleet-canary` scheduled workflow is the enforced surface that watches it.
>
> **The converse is equally true (OMN-15233): "Docker says unhealthy" is NOT sufficient evidence that a runner is degraded.**
> **Cross-check the GitHub runner registry before ANY restart sweep. If the runners are online, the flag is the bug — do not restart.**
> ```bash
> gh api /orgs/OmniNode-ai/actions/runners --jq \
>   '[.runners[]|select(.status=="online")]|length'
> ```
> Never restart-sweep off the Docker-unhealthy flag alone. On 2026-07-27 that count went 13 → 37 → 59 while the registry reported **64/64 online throughout**; 59 → 4 resolved with only 8 restarts and the untouched control group self-healed. The "growth" was measurement phase, not fleet degradation.

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
| 1 | `docker/runners/healthcheck.sh` (in-container) | `bin/Runner.Listener` process alive AND exactly one listener with a non-1 PPID (OMN-15233) AND `_diag` heartbeat fresh (`RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS`, default **4500s** since OMN-15233) AND ≤ `RUNNER_HEALTH_MAX_LOG_STARTS_PER_HOUR` (6) listener starts in the last hour AND the GitHub **broker session** is not persistently broken (`RUNNER_HEALTH_MAX_SESSION_BROKEN_SECONDS`, default **900s** since OMN-15311) AND github.com egress | ≤ ~77 min (staleness + 3×30s retries); orphan/crash-loop layers fire in ≤ ~2 min; broken-session layer in ≤ ~17 min |
| 2 | `docker/runners/entrypoint.sh` watchdog | listener process exists while `run.sh` runs; recycles the wrapper tree after 5×60s consecutive misses (bounded by `LISTENER_RESTART_MAX=50`, then container exit → restart policy). **OMN-14564:** also recycles (with an explicit listener kill) when the listener process is alive but its `_diag` heartbeat is older than `LISTENER_HEARTBEAT_MAX_AGE_SECONDS` (3600s) for `LISTENER_HEARTBEAT_MISSES` (3) consecutive 60s ticks — never while a `Runner.Worker` job is executing. **OMN-15233:** reaps any surviving listener (TERM → KILL after `LISTENER_REAP_TIMEOUT_SECONDS`) BEFORE spawning a replacement | ≤ ~6 min (dead) / ≤ ~63 min (hung: 3600s staleness + 3×60s) |
| 3 | `runner-monitor.sh` cron on `.201` (OMN-13109) | Docker vs GitHub divergence, silent wedge, crash loop → Slack | 3 min cadence, shares fate with host |
| 4 | **`runner-fleet-canary` GHA workflow (authoritative)** | GitHub org registry online count vs `config/runner_fleet.yaml` `expected_count`; fails the run when offline+missing > `RUNNER_CANARY_MAX_OFFLINE` (5) | 15 min cadence, GitHub-hosted — survives total `.201` loss |

## Composite readiness — the adjudicating surface (OMN-15255)

Layers 1–4 above each answer one question and none of them adjudicates when two
disagree. On 2026-07-27T16:40Z the registry read `{total: 64, online: 64, busy: 0}`
while **53 of 64 containers read docker-unhealthy**. Deciding which surface was
right meant a human diffing three outputs by hand — `gh api .../actions/runners`,
`docker ps`, and per-container `_diag` mtimes.

`node_runner_fleet_health_compute` now emits one composite verdict per runner.
Readiness is a **conjunction** over six independently-probed signals; a runner is
`READY` only when every one PASSes:

| Signal | PASS when |
|---|---|
| `github_registration` | registry reports `online` |
| `docker_health` | container state `running` and health `healthy` (or `none` — image declares no healthcheck) |
| `diag_heartbeat` | newest `_diag/*.log` age ≤ `RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS` (**4500s**, brackets the ~50-min token-refresh cycle — OMN-15233) |
| `listener_topology` | exactly one `Runner.Listener`, zero at PPID 1 |
| `container_stability` | `RestartCount` ≤ `CRASHLOOP_RESTART_THRESHOLD` (5) |
| `disk_capacity` | runner-host disk used < 90% (module constant, deliberately not a new env read — see OMN-15234) |

Read the verdict, not the individual surfaces:

- `ready_count` — usable capacity. **This, not `online_count`.**
- `quarantined_runners` — a signal was probed and FAILed.
- `bounce_eligible_runners` — the strict subset a force-recreate can actually fix.
- `readiness_signal_rollups` — which surface disagrees, and about how many runners.

**Two fail directions, on purpose.** Readiness fails CLOSED: `UNKNOWN` is not
`READY`, so an unprobeable runner is not counted as capacity. The bounce gate
fails SAFE: no restart on an indeterminate source, never on a busy runner, and
never for a cause a recreate cannot fix. Concretely — a GitHub-offline runner
whose local listener is single and non-orphaned with a fresh heartbeat is
quarantined but **never bounced** (OMN-14057 status-lag corroboration), and a
full host disk is quarantined but never bounced (a recreate frees no disk).

`state` (the precedence classification) and `readiness` legitimately disagree, and
neither is wrong: `state` answers "what is the most severe single thing wrong",
first-match-wins; `readiness` answers "may this runner take work". A runner that
is GitHub-online with a fresh heartbeat, an unhealthy container and two listeners
is `state=HEALTHY` and `readiness=NOT_READY`.

**Retired by this:** the four state-keyed `RESTART_RUNNER` branches
(`CRASH_LOOPING` 0.9 / `LISTENER_ZOMBIE` 0.85 / `OFFLINE_IDLE` 0.6 / `WEDGED` 0.5)
are **deleted** — bounce-eligibility is now the single producer of a restart
recommendation. Four independently-tunable confidence heuristics over the same
facts is how one misread threshold (the retired 900s heartbeat window) became a
fleet-wide restart storm with nothing able to veto it.

**Not yet retired — rollout-gated.** The manual three-surface cross-check in
"The rule that changed" above remains the operator procedure until the extended
probe is deployed: `docker_health`, listener topology and host disk are gathered
by `node_runner_health_snapshot_effect`, and until that runs against the fleet
those signals report `UNKNOWN`. UNKNOWN quarantines nothing and bounces nothing,
so the code is inert before rollout by construction — verify `ready_count` is
non-zero before trusting the view.

## Broker-dispatch/reconnect race — a DIFFERENT failure class (OMN-15776)

Layers 1–4 and the composite readiness verdict above all detect state the
*local runner process* can observe: a dead listener, a hung listener, a
container that is unhealthy or offline. **This class is invisible to every one
of them**, because the failure happens on the GitHub side of the wire before
any local process this repo controls ever runs.

**Mechanism (proven, 2026-08-09 — direct evidence on 4/4 independently-checked
runners: omninode-runner-2/17/35/48):** GitHub's Actions broker dispatches a
job to a self-hosted runner within 2-7s of that SAME runner finishing its
*previous, unrelated* job — exactly while the runner's `Runner.Listener` is
mid-reconnect on its broker long-poll (every job completion triggers a
`TaskCanceledException`/`IOException`/`SocketException(125)` retry storm on
that connection, with an observed 5-12s exponential backoff). The new dispatch
lands in that reconnect gap and is **never delivered** to the runner's active
message loop — no `Runner.Worker` process is ever spawned locally, so the
runner's own `_diag/Runner_*.log` has **zero** `"Running job: <name>"` entry
for that job (this is not a crashed step 1 — it is a dispatch that never
arrived) — while GitHub's server side records the assignment, sets
`started_at`, and independently times the orphaned assignment out at a
**fixed ~10m0-1s**, unrelated to any declared `timeout-minutes` and unrelated
to this repo's `LISTENER_HEARTBEAT_MAX_AGE_SECONDS` (3600s) watchdog.

**Ruled out** (2026-08-09 investigation): host resource contention (`docker
events` = zero container-level events in every kill window, `RestartCount=0`
throughout), kernel/NIC faults (`journalctl -k` shows only routine veth
churn), host cron/`runner-monitor.sh` auto-bounce (zero bounces on any
implicated runner in any kill window), and the OMN-14564 heartbeat watchdog
(fires on these runners routinely but always 50min-3h offset from the actual
kill windows — its detection surface, idle-listener silence, has zero overlap
with this failure's signature: an *actively chattering* listener silently
dropping one specific dispatch, with no Worker for the watchdog's
Worker-running guard to ever observe).

**Why no entrypoint.sh/watchdog fix applies.** The drop occurs in the GitHub
Actions client/broker protocol path, strictly before local process state
diverges from normal — there is nothing to `pgrep`, no heartbeat to go stale,
no wrapper tree to recycle. A local fix cannot close this gap.

**Remediation layer 5 — targeted, signature-keyed rerun
(`runner-broker-dispatch-wedge-rerun` GHA workflow, 10-min cadence,
GitHub-hosted — same isolation rationale as layer 4):**
`scripts/ci/runner_broker_dispatch_wedge_rerun.sh` queries the Jobs API
(never log-text grepping — there is no log content to grep) for jobs matching
the exact structural fingerprint and reissues only the matched job:

| Signal | Match condition |
|---|---|
| `runner_name` | set (self-hosted only — GitHub-hosted jobs are never touched) |
| `conclusion` | `failure` or `cancelled` |
| `steps` | empty array (no Worker ever spawned) |
| duration | `completed_at - started_at` within a tight band (default 595-605s) around the proven fixed ~10m0-1s server-side timeout |

This is deliberately narrow and additive to the existing
`scripts/infra-signature-rerun.sh` (OMN-13040), which matches known infra
**log-content** signatures (disk casualty, network wedge) — this class has no
log content to match against, so it needed a structural (Jobs-API-shape)
matcher instead. A job outside the duration band, or with any recorded steps,
is a genuine failure and is never rerun by this layer.

Coverage: `tests/unit/nodes/node_runner_fleet_maintain/test_runner_readiness_composite_omn15255.py`
and `..._facts_effect_omn15255.py`.

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
listener death: same `_diag` `find -mmin` staleness condition as the
healthcheck, guarded by a `Runner.Worker` check so an executing job is never
killed, with an explicit `pkill` of the listener binary on recycle (a hung
listener ignores the wrapper-tree TERM and would collide with the respawned
listener's session). Restarts remain bounded by `LISTENER_RESTART_MAX`.

**The kill threshold (`LISTENER_HEARTBEAT_MAX_AGE_SECONDS`, 3600s) is
deliberately DECOUPLED from the healthcheck alert threshold
(`RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS`, 900s at the time of this readback; 4500s
since OMN-15233 — see the orphan section below, which explains why the alert
threshold now sits ABOVE the kill threshold).** Live readback
2026-07-23T05:25–06:02Z: a fleet-wide broker-quiet window silenced `_diag` on
53/64 listeners for 35–50 min while GitHub kept all of them online — two
docker-"unhealthy" runners were actively executing jobs, and runners 2 and 45
resumed on their own after ~37 min blocked in the same token-refresh path
that hangs the true zombies forever. Docker health going red during such a
window is expected and benign (alerting stays sensitive); the watchdog only
kills once staleness clears the observed benign ceiling with margin, which
still recovers a true AAD-deadlock zombie in ~1 h instead of the 6 days the
2026-07-16..23 incident took.

## Orphan / session-conflict mode (OMN-15233, incident 2026-07-27)

The zombie shape the layer-1 heartbeat check was built to catch was **scoring
HEALTHY**, and the same check was **manufacturing false unhealthy on idle
runners**. Both defects were in the same layer.

**(a) False positive by arithmetic.** `RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS=900`
sat far below the **~50-minute IDLE `_diag` write cadence** — when a runner has
no job, only the OAuth/AAD token refresh writes `_diag`; the minutes-scale
cadence holds only while jobs run. An idle runner therefore read unhealthy for
**~35 of every 50 minutes** with nothing degraded. The threshold is now **4500s**
(75 min), which clears the observed idle cadence with 50% margin and is
justified inline in `healthcheck.sh`. The watchdog KILL threshold
(`LISTENER_HEARTBEAT_MAX_AGE_SECONDS`, 3600s) stays decoupled and is now BELOW
the alert threshold — remediation is bounded by `LISTENER_HEARTBEAT_MISSES`
(3×60s) and guarded by the `Runner.Worker` job check, so it is the narrower,
better-guarded signal.

**(b) Inversion.** An orphaned `Runner.Listener` reparented to **PPID 1** keeps
holding the GitHub broker session. The watchdog spawns a replacement, which
crash-loops every ~5 min on `TaskAgentSessionConflictException` because the
orphan still owns the session — and **every crash mints a fresh
`Runner_*.log`**, which keeps the `_diag` mtime fresh, so the check read HEALTHY
forever. Four such zombies (runners **1, 43, 55, 57**; **88–234** `Runner_*.log`
files vs **3–7** on normal runners) were found only by process scan.

Three fixes, all in `docker/runners/`:

1. **Process topology** — `healthcheck.sh` fails on **duplicate**
   `Runner.Listener` processes and on any listener with **PPID 1**. A healthy
   listener's chain is `entrypoint.sh(PID 1) → run.sh → run-helper.sh →
   Runner.Listener`, so PPID 1 is unambiguously an orphan.
2. **Rate-based crash-loop signal** — `healthcheck.sh` counts `Runner_*.log`
   files touched inside `RUNNER_HEALTH_LOG_RATE_WINDOW_MINUTES` (60) and fails
   above `RUNNER_HEALTH_MAX_LOG_STARTS_PER_HOUR` (6). A ~5-min crash cadence is
   ~12/hour. The threshold is a **rate**, normalized to the window before the
   comparison (`ceil(per_hour × window_minutes / 60)`), so retuning the window
   alone does not silently retune the threshold — a 30m window allows 3 starts,
   not 6. Both tunables fail closed if set to anything but a non-negative
   integer. **This is deliberately NOT cumulative:** a cumulative count grows
   monotonically with container uptime, so any long-lived healthy container
   would eventually red-line forever, and a permanently-red check is a disabled
   check.
3. **Reap before respawn** — `entrypoint.sh` kills any surviving listener
   (TERM, then KILL after `LISTENER_REAP_TIMEOUT_SECONDS`) and confirms it is
   gone before spawning a replacement. Spawn-without-reap is what manufactures
   the `TaskAgentSessionConflictException` in the first place; if a listener
   survives SIGKILL the entrypoint exits so the restart policy replaces the whole
   PID namespace rather than looping silently.

Coverage: `tests/ci/test_runner_listener_liveness.py` —
`TestHealthcheckIdleThresholdRecalibration`, `TestHealthcheckOrphanInversion`,
`TestHealthcheckCrashLoopRate`, `TestEntrypointOrphanReap`.

## Broken broker session — the FOURTH state (OMN-15311, measured 2026-07-27)

Three container-local failure states were already modelled: **listener dead**
(layer 1 `pgrep`), **PPID-1 orphan / duplicate listener** (OMN-15233 topology),
**listener permanently silent** (heartbeat staleness). During the OMN-15233
fan-out a fourth appeared, and every existing surface called it healthy.

A transient host↔GitHub network fault hit at ~17:26Z. The registry-offline spike
mostly self-healed, but runners **36, 38 and 56 stayed registry-OFFLINE for ~20
minutes** with, at the same time:

- a single, non-orphaned, live `Runner.Listener` (PPID chain intact);
- `_diag` **fresh** — the listener's own reconnect **retry traffic** is itself a
  `_diag` write, so staleness never accrues;
- a normal listener start rate;
- `github.com` reachable;
- and `healthcheck.sh` returning **exit 0**.

The layers above assert that a listener exists, is singular, is parented, is
writing, and has egress. None of them asserts the property that actually decides
whether a runner can take work: **that it holds a live GitHub broker session.**
A state-4 runner is counted as capacity by Docker *and* is suppressed from
`runner-monitor.sh` auto-bounce by the local-listener evidence rule, so it
silently absorbs zero jobs until something else restarts it. All three cleared on
restart.

**Detection (layer 3b in `healthcheck.sh`).** The container has no GitHub
credential to query the registry, and 64 unauthenticated pollers would
rate-limit themselves. The listener's own newest `Runner_*.log` is the local
projection of the same fact:

- the session is **broken** when the **last** session marker in that log is an
  error (`Runner connect error`, `TaskAgentSessionConflictException`,
  `A session for this runner already exists`, `Unable to connect to the server`,
  `Failed to create session`) with **no** re-establish
  (`Listening for Jobs`, `Runner reconnected`, `Job message received`) after it.
  Marker **order**, not presence — a healthy long-lived runner has connect errors
  somewhere in its log, and a presence check would red-line the fleet.
- **`SocketException` is deliberately NOT a broken marker** (removed 2026-07-28,
  adversarial fleet probe). `BrokerServer` writes
  `System.Net.Sockets.SocketException (125): Operation canceled` ~45–150x per
  listener log as ordinary long-poll cancellation, immediately followed by
  `Get messages has been cancelled using local token source. Continue to get
  messages with new status.` — the session is still up. Ordering does **not**
  rescue this one, because the connected markers only fire at session
  establishment / job assignment, so on any runner idle >15 min the retry noise
  is the last marker. Measured across all 64 live listeners on `omninode-pc`
  (all `Up (healthy)`, all registry-**online**): with `SocketException` in the
  set, **64/64** classified broken; without it, **0/64**. Shipping it would have
  flipped the whole fleet Docker-unhealthy 15 min after the bind-mount swap —
  a permanently-red check is a disabled check.
  `tests/ci/fixtures/runner_diag_real_tail.log.gz` (a byte-faithful contiguous
  tail of `omninode-runner-10`'s live `Runner_20260727-170542-utc.log`, captured
  2026-07-28: last connected marker at line 9, 45 `SocketException` lines after
  it) pins the vocabulary against real data — the synthetic 3-line fixtures
  exercise the artifact that runs but not the input distribution that runs.
- reconnects are routine and fast, so the layer gates on **persistence**:
  `${RUNNER_HOME}/_diag/.session_broken_since` is stamped on first observation
  and the check only fails once that stamp is older than
  `RUNNER_HEALTH_MAX_SESSION_BROKEN_SECONDS` (**900s**). Recovery deletes the
  stamp, so a later blip restarts the clock instead of inheriting an old one.
  900s sits above every recovery observed in the fault and below the ~20-minute
  dwell of the cohort that never recovered.
- a live listener with **no** `Runner_*.log` at all fails closed (a registered
  listener always mints one) — same class as the missing-`_diag` branch.
- `RUNNER_HEALTH_SESSION_STATE_CHECK=0` disarms the layer fleet-wide by env,
  without a bind-mount file swap, if it ever misfires.

**Operator reading.** `unhealthy: GitHub broker session broken for more than …`
means the registry almost certainly reports this runner OFFLINE while the
container looks fine. This is the one docker-unhealthy signature that is **not**
covered by the OMN-15233 interim rule ("if the registry says online, the flag is
the bug") — here the flag and the registry agree. A bounce is the known
remediation; all three affected runners cleared on restart.

Coverage: `tests/ci/test_runner_listener_liveness.py` —
`TestHealthcheckBrokerSessionState`.

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
