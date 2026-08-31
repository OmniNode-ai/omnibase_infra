# The `.201` gate-runner entry point (OMN-17317)

`scripts/ci/run_on_gate_runner.sh` is **the** way to run governed pre-push work
or a bounded ad-hoc test run inside `omninode-gate-runner`. A raw `docker exec`
is not a supported route and has four separately-ticketed failure modes; this
page is the executable recipe the pre-push hook's refusal text points at, via
[`lab-prepush-host-table.md`](lab-prepush-host-table.md).

Works identically for any account in the host's `docker` group — nothing here
reads `~/push-lanes`, which is mode `0750` on `.201` and therefore
single-account by construction (OMN-17280 contractor-access chain).

## The two commands

```bash
# 1. Governed pre-push. Detached, 4h wall-clock bound, exclusive heavy slot.
scripts/ci/run_on_gate_runner.sh --sync --detached --label omn1234-prepush \
  "$OMNI_HOME/omni_worktrees/OMN-1234/omnimarket" git push

# 2. Bounded ad-hoc test run. No slot, so a <=30min bound is enforced.
scripts/ci/run_on_gate_runner.sh --detached --no-slot --timeout 1200 \
  --label omn1234-unit "$OMNI_HOME/omni_worktrees/OMN-1234/omnimarket" \
  uv run pytest tests/unit -q
```

Both print a run-id, a run directory, a tail command and a poll command, then
exit. **Poll the receipt; never hold the pipe.**

```bash
scripts/ci/run_on_gate_runner.sh --status <run-dir>          # one-shot verdict
scripts/ci/run_on_gate_runner.sh --wait   <run-dir>          # block until terminal
docker exec omninode-gate-runner tail -f <run-dir>/run.log   # live output
docker exec omninode-gate-runner cat     <run-dir>/heartbeat # alive? progressing?
```

Driving `.201`'s container from another host is the same command with
`GATE_RUNNER_SSH_TARGET=jonah@192.168.86.201` exported. <!-- onex-allow-internal-ip OMN-17317 reason="runbook recipe needs the real .201 target" -->

## Why detached is not optional

A governed pre-push driven by an **attached** `docker exec` deadlocks forever,
silently, **after the suite has already finished and passed**, if the exec
client detaches at any point during the run.

`pre_commit._run_single_hook` buffers a hook's entire output and writes it out
in one burst after the hook exits. A fail-closed full-suite escalation produces
megabytes of per-test lines, so that burst crosses 64 KiB immediately. The exec
session's stdout is a kernel pipe whose only reader is the containerd shim; once
the client goes away — a dockerd bounce, an ssh drop, an agent session ending —
the shim holds the read end open and nothing drains it. Every subsequent
`write(2)` blocks forever.

Measured live 2026-08-31 (omnibase_core PR #1629, OMN-16619 lane): 3 h 45 m of
tests ran green and exited 0, then `pre-commit` parked in `anon_pipe_write` and
`git push` parked in `do_wait` behind a hook that could never exit. No log line,
no timeout, no diagnosis. Recovery meant draining the shim's read fd by hand —
exactly 65 538 bytes, the 64 KiB buffer.

`--detached` removes the dependency by construction: `docker exec -d` attaches no
stdio at all, `setsid` puts the run in its own session, and
`scripts/ci/gate_runner_supervisor.sh` re-points fds 1 and 2 at a log file on the
bind-mounted worktree before the payload produces its first byte.

## What a run leaves behind

Everything lives under the target worktree, so it survives the session, the
container restart policy, and any monitoring you attach later:

```
<worktree>/.onex_state/gate-runner/<run-id>/
├── run.log       # every byte of stdout+stderr, durable evidence
├── heartbeat     # "<utc-timestamp> log_bytes=<n> supervisor_pid=<pid>", every 15s
└── receipt.json  # schema onex.gate_runner.receipt.v1
```

`heartbeat` is what makes a slow run readable. Timestamp advancing **and**
`log_bytes` growing is healthy. Timestamp advancing, `log_bytes` frozen is
alive-but-stalled. Timestamp frozen is dead. The absence of that distinction is
why a healthy 3 h 45 m run was nearly killed during the incident above.

`receipt.json` is the only thing a caller should read to decide anything:

| `status` | meaning | `exit_code` |
|---|---|---|
| `running` | started, slot taken, payload in flight | `0` (not yet meaningful) |
| `passed` | payload exited 0 | `0` |
| `failed` | payload exited non-zero | the payload's own status |
| `timeout` | exceeded its wall-clock bound; terminated **loudly** | `124` |
| `refused` | never started — see `reason` | `4` slot held, `5` capability missing |

## Refusals are typed and immediate

Nothing is ever queued silently. The launcher exits with a named code:

| exit | code | meaning |
|---:|---|---|
| `2` | — | setup failure (fails closed) |
| `3` | `REFUSED_LOAD` | the container is over its admission threshold |
| `4` | `REFUSED_SLOT` | the exclusive heavy-suite slot is held, or a foreign heavy pre-push is running |
| `5` | `REFUSED_PROBE` | admission could not be measured — fails closed |

**Admission reads the container's own cgroup, not the bare host's loadavg.**
`.201` has 32 cores; this container is capped at 4 CPUs / 8 GiB. A host-level
`/proc/loadavg` read can report "HAS capacity" while the container being
recommended is saturated — the documented container-starvation trap
(OMN-16446 finding 5). The probe takes a one-second `cpu.stat` delta against the
`cpu.max` quota and compares `memory.current` to `memory.max`, using the same
`PREPUSH_LOAD_THRESHOLD` knob and the same busy/limit ratio semantics the
governed selector's `host_is_fit()` uses.

**The heavy-suite slot is exclusive and is taken on this path too.** The host
queue's gate-1 (`~/push-lanes/queue-runner.sh`) is only ever called from inside
the queue loop, so a container run that never enqueued consulted nothing — which
is how two heavy `omnibase_core` suites ran concurrently on 2026-08-30
(OMN-17221). This entry point takes an exclusive `flock` inside the container
*and* checks the same host-side `pgrep -f prepush_smart_tests\.sh` signal gate-1
uses, so the two paths now see each other. The lock is held by the supervisor
process itself, so the kernel releases it on death — there is no stale-slot path.

`--no-slot` exists for genuinely bounded ad-hoc work and mechanically forces
`--timeout <= 1800`. It is recorded in the receipt as `"slot":"skipped"`; it is
never silent, and it cannot become the way a heavy suite skips serialization.

## What this does NOT close

* **The container still cannot push to GitHub on its own credentials.** It ships
  no `ssh` client (`docker/Dockerfile.gate-runner` installs no `openssh-client`,
  OMN-16446 gap 3), and while it *does* ship the `gh` binary — OMN-16752 added it
  at a pinned `2.68.1` so `shutil.which("gh")` resolves for the DoD evidence
  guard — the compose file mounts no `gh` config and no git credential helper, so
  `gh` is present but unauthenticated (OMN-16446 gap 6: the gap is the credential,
  not the binary). Provisioning credentials into a shared container is a security
  decision and is deliberately left open, not defaulted. Today the push runs from
  inside the container using a credential the operator supplies for that one push.
* **The host `~/push-lanes` queue is unchanged.** It is host state outside every
  repo. This entry point makes the container path consult the same signal; it
  does not modify the queue-runner.
* **An ad-hoc interactive `docker exec` typed at a `.201` terminal is still
  possible.** This is a sanctioned-path gate, not a runtime interdiction.
