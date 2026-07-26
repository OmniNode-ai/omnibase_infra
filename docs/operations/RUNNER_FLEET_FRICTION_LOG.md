# Runner Fleet Friction Log

This log captures operational friction encountered while repairing the
OmniNode self-hosted GitHub Actions runner fleet. Treat each item as a fix
candidate, not as normal operator burden.

## 2026-07-04

### SSH Access To AI PC Is Unreliable

- Symptom: `omninode-pc.tail75df5e.ts.net` accepts TCP/22, but SSH often times
  out during banner exchange or while running simple Docker commands.
- Impact: repair commands are slow, partial, and hard to prove; long-running
  Docker/compose operations can lose their controlling SSH session.
- Evidence: repeated `Connection timed out during banner exchange`, `Timeout,
  server omninode-pc.tail75df5e.ts.net not responding`, and successful `nc`
  checks at the same time.
- Fix direction: add host-level sshd saturation monitoring, journal capture,
  MaxStartups/UseDNS/login-shell checks, and alert when SSH banner latency is
  high.

### GitHub Runner State Can Be Stale

- Symptom: GitHub reports a runner `offline` while Docker reports the container
  healthy and `Runner.Listener` is present.
- Impact: naive Docker health checks say the fleet is healthy while Actions
  capacity is missing.
- Evidence: runner 44 had healthy Docker state and a live listener but remained
  GitHub `offline` until its stale GitHub runner registration was deleted.
- Fix direction: keep the GitHub API status as an authoritative repair signal,
  and add stale registration deletion for persistent offline-idle runners after
  a grace period.

### Cached Runner Credentials Can Be Stale

- Symptom: after force-recreate, a runner restores cached credentials and then
  fails session creation because GitHub deleted the server-side registration.
- Impact: force-recreate without clearing the specific credential volume may
  bring the container back but not restore capacity.
- Evidence: runner 37 logged `runner registration has been deleted from the
  server`, cleared cache, then required fresh registration.
- Fix direction: teach repair automation to remove the named runner credential
  volume when the log contains deleted-registration text.

### Runner Self-Update Still Appeared In Live Logs

- Symptom: live logs showed GitHub runner self-update attempts followed by
  `Runner.Listener: No such file or directory`.
- Impact: runners flap offline/unhealthy after jobs complete.
- Evidence: multiple runners logged `Runner update process finished` and then
  missing `Runner.Listener`.
- Fix direction: ensure every live container is using the entrypoint with
  `--disableupdate`; add a monitor assertion that samples all containers and
  alerts if the mounted entrypoint lacks it.

### Repair Timer Needed Different Policy Than Alert Timer

- Symptom: a frequent monitor pass is useful for Slack alerts, but frequent
  auto-bounce risks killing work if GitHub busy state lags.
- Impact: operators were asked to repeatedly intervene manually.
- Fix direction implemented: deploy script now installs a 3-minute alert-only
  monitor and a separate 10-minute bounded repair pass with
  `MONITOR_AUTO_BOUNCE=1` and `OFFLINE_IDLE_RECREATE_AGE_SECONDS=600`.

### Docker Compose Recreate Is Slow On The AI PC

- Symptom: targeted `docker compose up -d --force-recreate --no-deps` for one
  or two runners can take several minutes.
- Impact: repair sessions look hung and may be interrupted too early.
- Evidence: runner 32/37 recreate took multiple minutes between stop,
  recreate, and start phases.
- Fix direction: add per-phase timing to repair logs and alert on slow Docker
  operations separately from runner health.

### Busy-Offline Is Ambiguous

- Symptom: GitHub can report runners as both `offline` and `busy`.
- Impact: blindly restarting them can kill active CI jobs, but leaving them
  alone can hide stuck capacity.
- Evidence: multiple snapshots had all remaining offline runners marked busy.
- Fix direction: keep busy-offline protected by default, but add local log
  corroboration and age thresholds so truly stale busy-offline runners can be
  escalated.

### GitHub-Hosted Runner Leakage Was Caused By Variable Drift

- Symptom: trusted workflows using the standard selector still ran on
  `ubuntu-latest` because repo-level variables overrode org defaults.
- Impact: GitHub Actions minutes were consumed despite a self-hosted fleet.
- Evidence: repo variables for several repos had
  `OMNI_TRUSTED_CI_RUNS_ON_JSON=["ubuntu-latest"]`.
- Fix direction implemented: runner routing audit now checks trusted runner
  variables and bare hosted workflow usage.

## Open Fix Candidates

- Add stale-registration deletion to `runner-monitor.sh` for persistent
  offline-idle runners whose local logs prove deleted registration.
- Add all-container `--disableupdate` drift check to the monitor.
- Add sshd health/bannertime check for `omninode-pc.tail75df5e.ts.net`.
- Add Docker compose phase timing and slow-operation alerts.
- Add a stricter busy-offline age detector backed by local runner logs.
