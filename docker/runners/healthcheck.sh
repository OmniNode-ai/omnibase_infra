#!/usr/bin/env bash
# Docker healthcheck for the GitHub Actions runner container.
# Tickets: OMN-12433 (egress), OMN-13915 (listener liveness + heartbeat
#          freshness), OMN-15233 (threshold recalibration + orphan/crash-loop
#          detection).
#
# History of what each layer catches:
#   - The original check only asserted container liveness — 37/48 runners sat
#     "Up (healthy)" for four days with a dead Runner.Listener (OMN-13915).
#   - OMN-12433 added `pgrep Runner.Listener` + github.com egress. A point-in-time
#     pgrep still passes when the listener is hung/zombied or when a wrapper
#     process keeps the tree "alive-looking" while no work flows.
#   - OMN-13915 adds a HEARTBEAT FRESHNESS check: a live, registered listener
#     appends to ${RUNNER_HOME}/_diag continuously. If the newest _diag file is
#     older than RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS the listener is not actually
#     talking to GitHub, whatever the process table says.
#   - OMN-15233 fixes the two defects the OMN-13915 layer shipped with:
#       (a) FALSE POSITIVE by arithmetic — the 900s threshold sat far below the
#           ~50-minute IDLE _diag write cadence, so an idle runner read unhealthy
#           for ~35 of every 50 minutes with nothing wrong. See the threshold
#           comment below.
#       (b) INVERSION — the zombie shape this check exists to catch scored
#           HEALTHY. An orphaned listener reparented to PPID 1 keeps the GitHub
#           session; the watchdog's replacement crash-loops on
#           TaskAgentSessionConflictException every ~5 min, and each crash mints
#           a fresh Runner_*.log — which keeps _diag "fresh" forever. Layers 2
#           (process topology) and 4 (crash-loop rate) catch that shape.
#
# Tunables (env, defaults chosen for the 64-runner .201 fleet):
#   RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS      heartbeat staleness threshold (4500)
#   RUNNER_HEALTH_MAX_LOG_STARTS_PER_HOUR   listener starts/hour before the
#                                           crash-loop layer fails (6)
#   RUNNER_HEALTH_LOG_RATE_WINDOW_MINUTES   crash-loop rate window (60)
#   RUNNER_HEALTH_EGRESS_CHECK              set to 0 to skip the github.com egress
#                                           probe (used by offline CI tests only;
#                                           production compose leaves it enabled)
set -u

RUNNER_HOME="${RUNNER_HOME:-/home/runner/actions-runner}"
# OMN-15233 (a): 4500s (75 min), NOT the original 900s. When a runner is IDLE the
# only thing that writes _diag is the OAuth/AAD token refresh, on a ~50-minute
# cadence; the minutes-scale cadence only holds while jobs are running. 900s
# therefore flagged every idle runner for ~35 of every 50 minutes purely by
# threshold arithmetic — that is what produced the 13 -> 37 -> 59 "unhealthy
# growth" on 2026-07-27 while the GitHub registry reported 64/64 online
# throughout (59 -> 4 resolved with only 8 restarts; the untouched control group
# self-healed). 4500s clears the observed ~50-min idle write cadence with 50%
# margin. It is deliberately NOT the liveness signal of record — the GitHub
# registry (runner-fleet-canary, layer 4 of the runbook) is; this threshold only
# has to avoid manufacturing false unhealthy on an idle fleet while still
# surfacing a listener that has gone permanently silent.
RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS="${RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS:-4500}"
RUNNER_HEALTH_MAX_LOG_STARTS_PER_HOUR="${RUNNER_HEALTH_MAX_LOG_STARTS_PER_HOUR:-6}"
RUNNER_HEALTH_LOG_RATE_WINDOW_MINUTES="${RUNNER_HEALTH_LOG_RATE_WINDOW_MINUTES:-60}"
RUNNER_HEALTH_EGRESS_CHECK="${RUNNER_HEALTH_EGRESS_CHECK:-1}"

# 1. Listener process must be alive. Match THIS runner home's listener BINARY
#    path (${RUNNER_HOME}/bin/Runner.Listener), not a loose substring: wrapper
#    scripts, log paths, or another runner's listener must never satisfy the
#    liveness assertion. Dots in the path are escaped for pgrep's ERE matching.
listener_pattern="${RUNNER_HOME//./\\.}/bin/Runner\.Listener"
listener_pids=$(pgrep -f "${listener_pattern}" 2>/dev/null)
if [[ -z "${listener_pids}" ]]; then
  echo "unhealthy: Runner.Listener not running"
  exit 1
fi

# 2. Listener process TOPOLOGY must be sane (OMN-15233 b). Process EXISTENCE is
#    not the assertion — process SINGULARITY and PARENTAGE are.
#
#    - Duplicate listeners: exactly one Runner.Listener may hold this runner's
#      GitHub broker session. Two means the session is contested, which is the
#      TaskAgentSessionConflictException crash-loop from the inside.
#    - PPID 1: a healthy listener's parent chain is
#      entrypoint.sh(PID 1) -> run.sh -> run-helper.sh -> Runner.Listener, so a
#      healthy listener is NEVER a direct child of PID 1. PPID 1 means its
#      parent tree died and it was reparented — an orphan still holding the
#      session while the entrypoint spawns replacements that cannot register.
#      This is exactly the shape found on runners 1/43/55/57 (88-234 Runner_*.log
#      files vs 3-7 on normal runners), which the mtime-only check scored
#      HEALTHY because the crash-looping replacement kept _diag fresh.
listener_count=$(printf '%s\n' "${listener_pids}" | grep -c '^[0-9][0-9]*$')
if [[ "${listener_count}" -gt 1 ]]; then
  echo "unhealthy: ${listener_count} Runner.Listener processes (pids: ${listener_pids//$'\n'/ }) — duplicate listeners contest the GitHub session (OMN-15233 orphan/conflict mode)"
  exit 1
fi
for pid in ${listener_pids}; do
  listener_ppid=$(ps -o ppid= -p "${pid}" 2>/dev/null | tr -d '[:space:]')
  if [[ "${listener_ppid}" == "1" ]]; then
    echo "unhealthy: Runner.Listener pid ${pid} has PPID 1 — orphaned listener reparented after its wrapper tree died (OMN-15233 orphan mode)"
    exit 1
  fi
done

# 3. Listener heartbeat must be FRESH (OMN-13915, threshold recalibrated by
#    OMN-15233 a). The listener writes to _diag on every long-poll cycle while
#    busy and on every token refresh while idle; a listener that stopped talking
#    to GitHub entirely stops writing. Fail when no _diag file was modified
#    within the threshold. A missing _diag directory with a "live" listener
#    process is the same divergence — fail closed (compose start_period covers
#    first registration).
diag_dir="${RUNNER_HOME}/_diag"
max_age_minutes=$(( (RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS + 59) / 60 ))
if [[ ! -d "${diag_dir}" ]]; then
  echo "unhealthy: listener process present but ${diag_dir} does not exist"
  exit 1
fi
fresh_file=$(find "${diag_dir}" -type f -name '*.log' -mmin "-${max_age_minutes}" -print 2>/dev/null | head -n 1)
if [[ -z "${fresh_file}" ]]; then
  echo "unhealthy: no _diag heartbeat within ${RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS}s (listener silent — OMN-13915 zombie mode)"
  exit 1
fi

# 4. Listener restart RATE must be sane (OMN-15233 b). The runner mints one
#    Runner_<timestamp>-utc.log per listener process start, so listener starts
#    inside a bounded window are directly countable. A replacement listener
#    crash-looping against an orphan that owns the session restarts every ~5 min
#    (~12/hour); a normal runner restarts a handful of times over its entire
#    lifetime.
#
#    This is deliberately RATE-BASED, NOT CUMULATIVE. A cumulative Runner_*.log
#    count persists across restarts and grows monotonically with container
#    uptime, so any long-lived healthy container would eventually cross a fixed
#    total and flag forever — a permanently-red check is a disabled check. Only
#    logs touched inside RUNNER_HEALTH_LOG_RATE_WINDOW_MINUTES are counted; a
#    runner with 234 historical logs and one active log reads 1.
#
#    -mmin (not birth time) is the portable proxy: a Runner_*.log is only
#    appended to while its listener process lives, so a closed log's mtime is
#    within seconds of that listener's death and an active log's mtime is now.
#    -maxdepth 1 keeps the per-job page logs under _diag/pages/ out of the count.
window_minutes="${RUNNER_HEALTH_LOG_RATE_WINDOW_MINUTES}"
recent_starts=$(find "${diag_dir}" -maxdepth 1 -type f -name 'Runner_*.log' -mmin "-${window_minutes}" -print 2>/dev/null | wc -l | tr -d '[:space:]')
if [[ "${recent_starts}" -gt "${RUNNER_HEALTH_MAX_LOG_STARTS_PER_HOUR}" ]]; then
  echo "unhealthy: ${recent_starts} listener starts in the last ${window_minutes}m (> ${RUNNER_HEALTH_MAX_LOG_STARTS_PER_HOUR}/hour) — listener crash-looping (OMN-15233 crash-loop rate)"
  exit 1
fi

# 5. github.com must be reachable (OMN-12433). A connected listener with no
#    in-flight job is expected; an egress fault that drops the GitHub
#    connection is what we catch. Use a bounded HEAD request instead of the
#    unauthenticated API rate_limit endpoint so shared-IP API limits cannot
#    create false unhealthy flaps.
if [[ "${RUNNER_HEALTH_EGRESS_CHECK}" != "0" ]]; then
  if ! curl -fsSI --connect-timeout 3 --max-time 8 -o /dev/null https://github.com/; then
    echo "unhealthy: github.com egress unreachable"
    exit 1
  fi
fi

echo "healthy: single non-orphaned listener, heartbeat fresh, ${recent_starts} start(s) in ${window_minutes}m, github.com reachable"
exit 0
