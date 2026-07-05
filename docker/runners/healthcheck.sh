#!/usr/bin/env bash
# Docker healthcheck for the GitHub Actions runner container.
# Tickets: OMN-12433 (egress), OMN-13915 (listener liveness + heartbeat freshness).
#
# History of what each layer catches:
#   - The original check only asserted container liveness — 37/48 runners sat
#     "Up (healthy)" for four days with a dead Runner.Listener (OMN-13915).
#   - OMN-12433 added `pgrep Runner.Listener` + github.com egress. A point-in-time
#     pgrep still passes when the listener is hung/zombied or when a wrapper
#     process keeps the tree "alive-looking" while no work flows.
#   - OMN-13915 adds a HEARTBEAT FRESHNESS check: a live, registered listener
#     appends to ${RUNNER_HOME}/_diag continuously (long-poll cycle ~50s). If the
#     newest _diag file is older than RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS the
#     listener is not actually talking to GitHub, whatever the process table says.
#
# Tunables (env, defaults chosen for the 48-runner .201 fleet):
#   RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS  heartbeat staleness threshold (default 900)
#   RUNNER_HEALTH_EGRESS_CHECK          set to 0 to skip the github.com egress
#                                       probe (used by offline CI tests only;
#                                       production compose leaves it enabled)
set -u

RUNNER_HOME="${RUNNER_HOME:-/home/runner/actions-runner}"
RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS="${RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS:-900}"
RUNNER_HEALTH_EGRESS_CHECK="${RUNNER_HEALTH_EGRESS_CHECK:-1}"

# 1. Listener process must be alive. Match THIS runner home's listener BINARY
#    path (${RUNNER_HOME}/bin/Runner.Listener), not a loose substring: wrapper
#    scripts, log paths, or another runner's listener must never satisfy the
#    liveness assertion. Dots in the path are escaped for pgrep's ERE matching.
listener_pattern="${RUNNER_HOME//./\\.}/bin/Runner\.Listener"
if ! pgrep -f "${listener_pattern}" >/dev/null 2>&1; then
  echo "unhealthy: Runner.Listener not running"
  exit 1
fi

# 2. Listener heartbeat must be FRESH (OMN-13915). The listener writes to
#    _diag on every long-poll cycle; a listener that stopped talking to GitHub
#    stops writing. Fail when no _diag file was modified within the threshold.
#    A missing _diag directory with a "live" listener process is the same
#    divergence — fail closed (compose start_period covers first registration).
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

# 3. github.com must be reachable (OMN-12433). A connected listener with no
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

echo "healthy: listener up, heartbeat fresh, github.com reachable"
exit 0
