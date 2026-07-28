#!/usr/bin/env bash
# Docker healthcheck for the GitHub Actions runner container.
# Tickets: OMN-12433 (egress), OMN-13915 (listener liveness + heartbeat
#          freshness), OMN-15233 (threshold recalibration + orphan/crash-loop
#          detection), OMN-15311 (broker session state — the fourth state).
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
#   - OMN-15311 adds a BROKER SESSION STATE check (layer 3b) for the FOURTH
#     state, measured live on 2026-07-27 during the OMN-15233 fan-out: runners
#     36, 38 and 56 sat GitHub-registry-OFFLINE for ~20 minutes while every
#     local layer above passed — one live non-orphaned listener, _diag kept
#     FRESH by the listener's own reconnect RETRY traffic, normal start rate,
#     github.com reachable. The layers above assert that a listener process
#     exists, is singular, is parented, is writing, and can reach github.com.
#     None of them assert the thing that actually matters: that the listener
#     HOLDS A LIVE BROKER SESSION and can therefore be handed a job. A
#     state-4 runner is counted as capacity by Docker and is suppressed from
#     auto-bounce by runner-monitor.sh's local-listener evidence rule, so it
#     silently absorbs zero jobs until something else restarts it.
#
# Tunables (env, defaults chosen for the 64-runner .201 fleet):
#   RUNNER_HEALTH_MAX_DIAG_AGE_SECONDS      heartbeat staleness threshold (4500)
#   RUNNER_HEALTH_MAX_LOG_STARTS_PER_HOUR   listener starts/hour before the
#                                           crash-loop layer fails (6). A RATE,
#                                           normalized to the window below — not
#                                           a raw count of files in the window.
#   RUNNER_HEALTH_LOG_RATE_WINDOW_MINUTES   crash-loop rate window (60)
#   RUNNER_HEALTH_MAX_SESSION_BROKEN_SECONDS how long the broker session may stay
#                                           broken before the runner fails (900).
#                                           A GRACE, not a threshold on a
#                                           measurement — see layer 3b.
#   RUNNER_HEALTH_SESSION_STATE_CHECK       set to 0 to skip the broker-session
#                                           layer entirely (fleet-wide kill
#                                           switch by env, no file swap needed)
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
# OMN-15311: 900s (15 min). This is a GRACE on a state that is already known to
# be bad, not a threshold on a noisy measurement. A listener that loses its
# broker session normally re-establishes it in well under a minute (the
# 2026-07-27 network fault's registry-offline spike mostly self-healed within
# one poll); the cohort that did NOT recover held the broken state for ~20 min
# and only cleared on restart. 900s therefore sits above every recovery
# observed and below the shortest unrecovered case.
RUNNER_HEALTH_MAX_SESSION_BROKEN_SECONDS="${RUNNER_HEALTH_MAX_SESSION_BROKEN_SECONDS:-900}"
RUNNER_HEALTH_SESSION_STATE_CHECK="${RUNNER_HEALTH_SESSION_STATE_CHECK:-1}"
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

# 3b. Broker SESSION state must be CONNECTED (OMN-15311 — the FOURTH state).
#
#     Every layer above is satisfied by a runner that GitHub considers OFFLINE.
#     That is not hypothetical: on 2026-07-27 a transient host<->GitHub network
#     fault left runners 36/38/56 registry-offline for ~20 min with a single
#     non-orphaned live listener, a FRESH _diag (the reconnect retries are
#     themselves _diag writes), a normal listener start rate, and github.com
#     reachable. Exit 0 on all five layers; zero jobs accepted.
#
#     The registry is the authoritative liveness surface, but the container has
#     no GitHub credential to query it (and an unauthenticated poll from 64
#     containers would rate-limit itself). The listener's OWN log is the local
#     projection of that same fact: it records when the broker session is
#     established and when it drops.
#
#     STATE, NOT PRESENCE. What matters is which marker class appears LAST in
#     the newest Runner_*.log. "A connect error appears somewhere in the log" is
#     true of essentially every long-lived healthy runner and would be a
#     fleet-wide false positive; "the last session marker is an error, with no
#     re-establish after it" is the actual broken state.
#
#     PERSISTENCE, NOT INSTANT. Reconnects are normal and fast. A single check
#     that fails the moment a drop is observed would flap the whole fleet on
#     every blip. The broken state is therefore stamped on first observation and
#     only fails once the stamp is older than the grace window; recovery deletes
#     the stamp, so a later blip restarts the clock instead of inheriting an
#     ancient one. Age is measured with find -mmin — the same idiom layers 3/4
#     already use — deliberately NOT by parsing the log's "[YYYY-MM-DD HH:MM:SSZ]"
#     prefix, which needs GNU `date -d` and would make the layer unexercisable
#     on the BSD-date gate host.
if [[ "${RUNNER_HEALTH_SESSION_STATE_CHECK}" != "0" ]]; then
  if ! [[ "${RUNNER_HEALTH_MAX_SESSION_BROKEN_SECONDS}" =~ ^[0-9]+$ ]] ||
    [[ "${RUNNER_HEALTH_MAX_SESSION_BROKEN_SECONDS}" -lt 1 ]]; then
    echo "unhealthy: RUNNER_HEALTH_MAX_SESSION_BROKEN_SECONDS='${RUNNER_HEALTH_MAX_SESSION_BROKEN_SECONDS}' is not a positive integer — refusing to guess a broker-session grace (fail closed)"
    exit 1
  fi

  # Markers the runner itself writes. CONNECTED means the broker session is
  # established and the runner is reachable for job assignment; BROKEN means the
  # session was lost, refused, or is contested.
  #
  # SocketException is DELIBERATELY ABSENT from the broken set (removed
  # 2026-07-28 by adversarial fleet probe, OMN-15311). BrokerServer emits
  # "System.Net.Sockets.SocketException (125): Operation canceled" ~45-150x per
  # listener log as ORDINARY long-poll cancellation - the very next line is
  # "Get messages has been cancelled using local token source. Continue to get
  # messages with new status." and the session is still up. Ordering does not
  # rescue it: the connected markers only fire at session establishment / job
  # assignment, so on any runner idle for >15 min that retry noise IS the last
  # marker. Measured against all 64 live listeners on omninode-pc, every one of
  # them Up-healthy and registry-online: WITH SocketException 64/64 classified
  # broken; WITHOUT it 0/64. A permanently-red check is a disabled check - the
  # same reasoning as the rate-vs-cumulative note in layer 4 below.
  # tests/ci/fixtures/runner_diag_real_tail.log.gz pins this to a real log tail.
  session_connected_patterns='Listening for Jobs|Runner reconnected|Job message received'
  session_broken_patterns='Runner connect error|TaskAgentSessionConflictException|A session for this runner already exists|Unable to connect to the server|Failed to create session'

  # SC2012: `ls -t` is the portable newest-first ordering here; find -printf
  # '%T@' is GNU-only and this script must also run under the BSD find on the
  # gate host. Runner_*.log names are runner-generated and contain no spaces.
  # shellcheck disable=SC2012
  newest_runner_log=$(ls -1t "${diag_dir}"/Runner_*.log 2>/dev/null | head -n 1)
  if [[ -z "${newest_runner_log}" ]]; then
    # Same divergence class as a missing _diag directory: a live listener always
    # mints a Runner_<timestamp>-utc.log at start, so its absence means the
    # process we matched is not a listener that ever registered. Fail closed.
    echo "unhealthy: listener process present but no ${diag_dir}/Runner_*.log exists — broker session state is unreadable (OMN-15311 fail-closed)"
    exit 1
  fi

  session_stamp="${diag_dir}/.session_broken_since"
  last_connected_line=$(grep -nE "${session_connected_patterns}" "${newest_runner_log}" 2>/dev/null | tail -n 1 | cut -d: -f1)
  last_broken_line=$(grep -nE "${session_broken_patterns}" "${newest_runner_log}" 2>/dev/null | tail -n 1 | cut -d: -f1)

  session_is_broken=0
  if [[ -n "${last_broken_line}" ]]; then
    if [[ -z "${last_connected_line}" ]] || [[ "${last_broken_line}" -gt "${last_connected_line}" ]]; then
      session_is_broken=1
    fi
  fi

  if [[ "${session_is_broken}" -eq 1 ]]; then
    if [[ ! -e "${session_stamp}" ]]; then
      if ! : >"${session_stamp}" 2>/dev/null; then
        # Without the stamp the grace cannot be measured, so "transient" and
        # "stuck for an hour" become indistinguishable. Indeterminate is not
        # health — and a non-writable _diag is itself a real fault, since the
        # listener writes there continuously.
        echo "unhealthy: broker session is broken and ${session_stamp} is not writable — cannot measure how long it has been broken (OMN-15311 fail-closed)"
        exit 1
      fi
    fi
    session_grace_minutes=$(((RUNNER_HEALTH_MAX_SESSION_BROKEN_SECONDS + 59) / 60))
    if [[ -z "$(find "${session_stamp}" -mmin "-${session_grace_minutes}" -print 2>/dev/null)" ]]; then
      echo "unhealthy: GitHub broker session broken for more than ${RUNNER_HEALTH_MAX_SESSION_BROKEN_SECONDS}s — listener alive and writing _diag, but its last session marker in $(basename "${newest_runner_log}") is an error with no re-establish after it (OMN-15311 broken-session mode; GitHub reports this runner OFFLINE)"
      exit 1
    fi
    session_state="reconnecting (broken, inside the ${RUNNER_HEALTH_MAX_SESSION_BROKEN_SECONDS}s grace)"
  else
    # Recovered (or never broken): drop the stamp so the grace clock restarts
    # from the NEXT drop rather than from an old one.
    rm -f "${session_stamp}" 2>/dev/null || true
    session_state="connected"
  fi
else
  session_state="unchecked"
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
#
#    NORMALIZATION (the two tunables are in DIFFERENT units): the count is taken
#    over RUNNER_HEALTH_LOG_RATE_WINDOW_MINUTES but the threshold is expressed
#    PER HOUR, so comparing them directly is only correct at the 60-minute
#    default. A 30m window would silently halve the effective threshold (6/hour
#    enforced as 6-per-30m = 12/hour) and a 120m window would double it
#    (6-per-120m = 3/hour). The per-hour budget is therefore scaled to the window
#    actually measured before the comparison. Integer arithmetic only (no bc in
#    the runner image): the +59 numerator rounds the allowance UP, so a
#    fractional budget (6/hour over a 5m window = 0.5) never floors to 0 — a
#    zero allowance would fail on the first legitimate listener start.
window_minutes="${RUNNER_HEALTH_LOG_RATE_WINDOW_MINUTES}"
if ! [[ "${window_minutes}" =~ ^[0-9]+$ ]] || [[ "${window_minutes}" -lt 1 ]]; then
  echo "unhealthy: RUNNER_HEALTH_LOG_RATE_WINDOW_MINUTES='${window_minutes}' is not a positive integer — refusing to guess a crash-loop window (fail closed)"
  exit 1
fi
if ! [[ "${RUNNER_HEALTH_MAX_LOG_STARTS_PER_HOUR}" =~ ^[0-9]+$ ]]; then
  echo "unhealthy: RUNNER_HEALTH_MAX_LOG_STARTS_PER_HOUR='${RUNNER_HEALTH_MAX_LOG_STARTS_PER_HOUR}' is not a non-negative integer — refusing to guess a crash-loop threshold (fail closed)"
  exit 1
fi
max_starts_in_window=$(( (RUNNER_HEALTH_MAX_LOG_STARTS_PER_HOUR * window_minutes + 59) / 60 ))
recent_starts=$(find "${diag_dir}" -maxdepth 1 -type f -name 'Runner_*.log' -mmin "-${window_minutes}" -print 2>/dev/null | wc -l | tr -d '[:space:]')
if [[ "${recent_starts}" -gt "${max_starts_in_window}" ]]; then
  echo "unhealthy: ${recent_starts} listener starts in the last ${window_minutes}m (> ${max_starts_in_window} allowed — ${RUNNER_HEALTH_MAX_LOG_STARTS_PER_HOUR}/hour normalized to a ${window_minutes}m window) — listener crash-looping (OMN-15233 crash-loop rate)"
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

echo "healthy: single non-orphaned listener, heartbeat fresh, broker session ${session_state}, ${recent_starts}/${max_starts_in_window} allowed start(s) in ${window_minutes}m, github.com reachable"
exit 0
