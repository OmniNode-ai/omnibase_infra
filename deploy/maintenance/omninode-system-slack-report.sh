#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# omninode-system-slack-report.sh — the .201 host system-health Slack reporter.
#
# WHAT THIS IS
#   This is the producer behind the Slack messages "*OmniNode system alert*",
#   "*OmniNode morning system digest*" and "*[OmniNode alert resolved]*".
#   It is executed as root on the `.201` host (omninode-pc) by
#   /etc/cron.d/omninode-system-slack-report:
#       5 8  * * *  root  <this script> --mode digest
#       */15 * * * * root <this script> --mode alert
#   The deployed copy lives at /data/maintenance/bin/omninode-system-slack-report.sh.
#   `deploy/maintenance/cron.d/omninode-system-slack-report` in this repo is the
#   as-deployed cron unit.
#
# WHY IT IS IN THIS REPO NOW (OMN-15509)
#   Until 2026-07-30 this script existed ONLY on the host, untracked by any
#   repository. That is why the defect below survived: there was no diff to
#   review, no test to fail, and no grep in any repo that could find it. The
#   file is version-controlled here so a change to what the platform alarms on
#   is a reviewable diff with a test attached.
#
# THE DEFECT THIS FIXES (OMN-15509)
#   The as-deployed `collect()` probed exactly five endpoints:
#       runtime-18085, runtime-28085, projection-api-13002,
#       deploy-agent-8099, web-3003
#   The dev/lab lane's main runtime port (:8085) was ABSENT. On 2026-07-30 the
#   dev runtime sat at Docker `health: starting` with :8085 returning 503, zero
#   registered handlers, and dependent containers stuck in `Created`, for at
#   least 26 minutes — and the 16:30:02Z Slack message reported every listed
#   runtime endpoint as `HTTP 200 (OK)`. The monitor was structurally incapable
#   of observing the thing that broke.
#
#   Four separate false-green mechanisms are closed here:
#     1. Lane coverage. Every lane's MAIN runtime health endpoint is probed,
#        and the lane->port map is read from `docker/runtime-policy.env` (the
#        rendered runtime-policy contract that `deploy-runtime.sh` and
#        `scripts/system_health_check.sh` already read) rather than being
#        hardcoded per call site. A lane cannot be silently dropped.
#     2. Body honesty. A runtime `/health` can return 200 with a body that says
#        it is NOT healthy. The old check used `grep -Ei 'healthy|ok'` against
#        the body, which MATCHES `{"healthy": false}` — the substring is there.
#        Runtime endpoints now resolve a real status from the JSON body and go
#        CRITICAL when it is not healthy, or when no status can be resolved.
#     3. `health: starting`. A container whose healthcheck has never once passed
#        reports neither `unhealthy` nor `Exited`, so it was invisible. A
#        container still `starting` past its own `start_period` is now CRITICAL.
#     4. Exit accounting. `Exited` was reported but excluded from the CRITICAL
#        condition, so a crashed container was silent. Non-zero exits are now
#        CRITICAL; Exit(0) one-shots (migration/init containers) are not — they
#        are expected to finish.
#
#   Fail-closed throughout: an endpoint that cannot be probed (connection
#   refused, DNS failure, timeout -> code 000) is CRITICAL, never skipped; a
#   docker query that fails is CRITICAL, never treated as "nothing wrong".
#
# WHAT OMN-16789 FIXES ON TOP (found by the operator, in Slack, as noise)
#   OMN-15509 taught this reporter to SEE a dead lane and OMN-15525 taught it to
#   SAY so. Neither addressed how often it says it. On 2026-08-27 the operator
#   reported #omninode-notifications receiving the same three criticals over and
#   over. Measured from /data/maintenance/logs/ across 39 consecutive ticks
#   (09:30Z-18:30Z): 22 posted, 17 suppressed.
#
#   The de-duplication was not missing — it was defeated. It hashed the WHOLE
#   issue set and compared to the previous hash, which is sound only against a
#   stable input. `runtime-stability-test-18085` bounced CRITICAL(000)/OK(200)
#   on nearly every tick, so the set alternated {c,s,d}/{c,d}, the hash changed
#   every tick, and the comparison never once suppressed anything. And because
#   `$issues` stayed non-empty on the shrink, the "one critical cleared" tick
#   re-posted the entire alert digest instead of reading as a recovery.
#
#   The bounce itself was self-inflicted: a hardcoded `curl --max-time 4`
#   against a lane whose first response after idle measures ~3.2s under host
#   load average ~9-10. See PROBE_TIMEOUT_SECONDS.
#
#   Three changes, all env-tunable, no literals in the decision path:
#     1. Per-key state (see the state machine below) instead of one set hash, so
#        one key changing cannot re-page every other standing key.
#     2. Hysteresis. CONFIRM_TICKS before a key may page; CLEAR_TICKS of
#        continuous absence before it is called recovered, with the notified
#        flag RETAINED across a short absence so a bounce cannot re-arm it.
#     3. RENOTIFY_SECONDS, so a standing critical is re-surfaced on a long
#        cadence rather than going silent forever after its first alert.
#   The daily 08:05 digest remains the only unconditional post — a clean fleet
#   still produces nothing on the */15 cadence.
#
# WHAT OMN-15525 FIXES ON TOP (found by actually deploying the above)
#   The OMN-15509 revision was installed on .201 on 2026-07-30T18:12Z and
#   immediately reported CRITICAL for all three lanes against a demonstrably
#   HEALTHY fleet (all three /health returned 200 with "healthy":true). It was
#   rolled back at 18:13:29Z, before the 18:15 cron fired. Two defects:
#
#     A. False-RED from a truncated parse. `check_runtime_lane` cut the body to
#        180 bytes for display and then handed THAT to jq. A real runtime
#        /health body on .201 is 2644 bytes, so jq always failed with
#        "Unfinished string at EOF", the verdict was always "unresolvable", and
#        fail-closed correctly turned an unparseable input into CRITICAL. The
#        rule was right; the input was mutilated before it got there. The body
#        is now parsed whole and truncated only for the reported excerpt.
#        This is why the unit fixtures did not catch it: HEALTHY_BODY was 63
#        bytes, comfortably under the cut. Fixtures now carry a realistic
#        >180-byte body so the truncation boundary is inside test coverage.
#
#     C. The alert could not fire at all. `$issues`, `$issue_keys`,
#        `$critical_count` and `$warning_count` were all selected with
#        `$2=="CRITICAL"`, but endpoint rows carry their status in `$1` (only
#        disk/docker rows use `$2`). So no endpoint failure — no runtime lane,
#        projection-api, deploy-agent or web — ever reached `$issues`, and
#        `--mode alert` took the "clean" branch and posted nothing. The digest
#        text rendered the CRITICAL lines because `format_digest` happened to
#        handle both shapes, which is why the .201 run printed three CRITICAL
#        lanes directly under `Issues: *0 critical*`. OMN-15509 taught this
#        script to SEE a dead lane; defect C is what kept it from SAYING so.
#        See `row_status` / `row_key` near the bottom of this file.
#
#     B. Silent hardcoded port fallback (rule 8). `policy_env_value` returned
#        SUCCESS when the policy file did not exist, and `lane_main_port`
#        substituted a literal 8085/18085/28085 for an empty value. A renamed
#        key or an unrendered runtime-policy.env degraded to probing guessed
#        ports with no signal. An unresolvable lane port is now CRITICAL.
#
#   A monitor has two failure directions and both are fatal to it: blind (the
#   OMN-15509 defect) and crying wolf (defect A). A permanent CRITICAL on a
#   healthy fleet gets the channel muted, after which the blind spot is back
#   with extra steps.
#
# PROD IS READ-ONLY
#   The prod lane is probed with a plain GET against /health and nothing else.
#   This script never mutates any lane.

set -euo pipefail

PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV_FILE=${OMNINODE_ALERT_ENV_FILE:-/data/omninode/omnibase_infra/.env}
# Repo root that carries the rendered runtime policy. Defaults to the parent of
# ENV_FILE's directory-of-record so the deployed layout needs no extra config.
INFRA_REPO_ROOT=${OMNINODE_INFRA_REPO_ROOT:-/data/omninode/omnibase_infra}
STATE_DIR=${OMNINODE_ALERT_STATE_DIR:-/data/maintenance/state}
LOG_DIR=${OMNINODE_ALERT_LOG_DIR:-/data/maintenance/logs}
LOCK_FILE=${OMNINODE_ALERT_LOCK_FILE:-/run/omninode-system-slack-report.lock}
PROBE_HOST=${OMNINODE_ALERT_PROBE_HOST:-127.0.0.1} # fallback-ok: this reporter runs ON the .201 host as root via cron and probes that host's OWN published lane ports, so the loopback IS the target, not a stand-in for a remote address; same posture as LANE_PROBE_HOST in scripts/system_health_check.sh
# Grace applied to a `health: starting` container whose image declares no
# start_period. Fail-closed: a finite grace, not "never alarm".
STARTING_GRACE_SECONDS=${OMNINODE_ALERT_STARTING_GRACE_SECONDS:-180}
# How much of a response body is quoted into the Slack message / log line. This
# bounds DISPLAY ONLY. It must never be applied before a body is parsed or
# pattern-matched (OMN-15525) — see check_runtime_lane.
BODY_EXCERPT_BYTES=${OMNINODE_ALERT_BODY_EXCERPT_BYTES:-180}
# How long an endpoint probe may take before it is scored unreachable (000).
#
# OMN-16789. This was a hardcoded `--max-time 4` at both probe call sites, and 4
# seconds is BELOW the measured legitimate response time of a warm-but-loaded
# lane on .201. Six probes of :18085/health at 2026-08-27T18:34Z, 3s apart:
# 000/000/000/000 then 200 at t=3.196s then 200 at t=0.013s — the lane was up
# the entire time; the first response after idle simply takes ~3.2s while the
# host sits at load average ~9-10. At a 4s ceiling that is a coin flip, so the
# monitor manufactured its own flapping input and then faithfully alerted on it.
# Fail-closed on 000 is correct and unchanged; feeding it a timeout that a
# healthy lane cannot reliably beat is what was wrong.
PROBE_TIMEOUT_SECONDS=${OMNINODE_ALERT_PROBE_TIMEOUT_SECONDS:-15}
# --- alert cadence (OMN-16789) -------------------------------------------
# All three are env-overridable and carry no literal anywhere else in the file.
#
# CONFIRM_TICKS  consecutive ticks a key must hold the SAME status before it is
#                allowed to page. Trades time-to-alert (2 ticks = 30 min on the
#                */15 cron) for immunity to single-tick blips. This is a host
#                digest, not a customer-facing pager; the operator's complaint
#                was noise, and a one-tick transient is not news.
# CLEAR_TICKS    consecutive ticks a key must be ABSENT before it is reported
#                recovered and dropped from state. This is the half that kills
#                flap spam: while a key is briefly absent its notified flag is
#                RETAINED, so a key that comes back does not re-page. Without
#                it, every OK->CRITICAL bounce re-arms the alert.
# RENOTIFY_SECONDS  how often a still-standing, already-notified key is
#                re-surfaced, so a permanent critical cannot go permanently
#                silent. Set to 0 to disable re-notification entirely.
CONFIRM_TICKS=${OMNINODE_ALERT_CONFIRM_TICKS:-2}
CLEAR_TICKS=${OMNINODE_ALERT_CLEAR_TICKS:-8}
RENOTIFY_SECONDS=${OMNINODE_ALERT_RENOTIFY_SECONDS:-21600}
MODE=digest

if [[ "${1:-}" == "--mode" ]]; then
  MODE="${2:-digest}"
elif [[ -n "${1:-}" ]]; then
  MODE="$1"
fi

mkdir -p "$STATE_DIR" "$LOG_DIR"
LOG_FILE="$LOG_DIR/omninode-system-slack-report-$(date -u +%Y%m%dT%H%M%SZ).log"
# dry-run is the human/CI inspection mode: keep its report on stdout instead of
# burying it in a log file, so it can be read (and asserted on) directly.
if [[ "$MODE" != "dry-run" ]]; then
  exec >>"$LOG_FILE" 2>&1
fi
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) another system report is already running"
  exit 0
fi

if [[ -f "$ENV_FILE" ]]; then
  set -a
  set +u
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set -u
  set +a
fi

if [[ "$MODE" != "dry-run" ]]; then
  : "${SLACK_BOT_TOKEN:?SLACK_BOT_TOKEN must be set in $ENV_FILE}"
  SLACK_CHANNEL_ID="${SLACK_CHANNEL_ID:-${SLACK_DEFAULT_CHANNEL:-}}"
  : "${SLACK_CHANNEL_ID:?SLACK_CHANNEL_ID or SLACK_DEFAULT_CHANNEL must be set in $ENV_FILE}"
else
  SLACK_CHANNEL_ID="${SLACK_CHANNEL_ID:-${SLACK_DEFAULT_CHANNEL:-dry-run}}"
fi

DATA_WARN_PCT=${DATA_WARN_PCT:-95}
DATA_CRIT_PCT=${DATA_CRIT_PCT:-98}
DATA_WARN_FREE_GB=${DATA_WARN_FREE_GB:-200}
DATA_CRIT_FREE_GB=${DATA_CRIT_FREE_GB:-50}
ROOT_WARN_PCT=${ROOT_WARN_PCT:-85}
ROOT_CRIT_PCT=${ROOT_CRIT_PCT:-92}
ROOT_WARN_FREE_GB=${ROOT_WARN_FREE_GB:-50}
ROOT_CRIT_FREE_GB=${ROOT_CRIT_FREE_GB:-20}

# ---------------------------------------------------------------------------
# Lane -> main runtime port map (OMN-15509 AC2)
#
# Read by targeted key extraction from the rendered runtime policy, NOT by
# `source`: that file is a generated artifact whose key set changes when the
# contract is re-rendered, and sourcing it would let a future render silently
# redefine this script's other variables. Same idiom, same file, and the same
# reasoning as scripts/system_health_check.sh.
#
# There are NO literal fallback ports (OMN-15525). The original revision of this
# block fell back to hardcoded 8085/18085/28085 whenever the policy file was
# missing or a key was renamed, and `policy_env_value` even returned SUCCESS on
# a missing file. That is a silent default masquerading as a resolved contract —
# CLAUDE.md rule 8 forbids exactly this, and it is the same false-green family
# OMN-15509 closed: the probe would keep reporting on a guessed port and the
# operator would never learn the lane map had stopped resolving. An unresolvable
# lane port is now a CRITICAL fact, reported the same as an unreachable endpoint.
# ---------------------------------------------------------------------------
LANE_PORT_UNRESOLVED='__unresolved__'

runtime_policy_file() {
  printf '%s' "${OMNINODE_RUNTIME_POLICY_ENV:-${INFRA_REPO_ROOT}/docker/runtime-policy.env}"
}

# Echo the value for `key`, or return non-zero when the policy file is absent or
# the key is not present in it. Fail-fast: never returns a substitute value.
policy_env_value() {
  local key="$1" file value
  file="$(runtime_policy_file)"
  [[ -f "$file" ]] || return 1
  value=$(sed -n "s/^${key}=//p" "$file" | tail -n 1 | tr -d "\"'")
  [[ -n "$value" ]] || return 1
  printf '%s' "$value"
}

# label|policy-key -- a pure data table, no comments inside the array (the
# fallback-port guard parses every line in it as lane|key).
#
# Must cover EVERY lane declaring a *_RUNTIME_MAIN_PORT in runtime-policy.env.
# Held in two-way parity by
# tests/unit/scripts/test_omninode_system_slack_report.py, which DERIVES the
# lane set from the policy instead of restating it: a row dropped here, or a
# lane added to the policy and not here, fails that module.
#
# judge (:48085) is read-only and NOT authorized for mutation, but it runs
# seven containers on .201 and was omitted from this table through OMN-15509
# and OMN-15525, so a dead judge runtime paged nobody (OMN-15556). Probing it
# is a plain GET /health, which is a read.
RUNTIME_LANE_SPECS=(
  "dev|DEV_RUNTIME_MAIN_PORT"
  "stability-test|STABILITY_TEST_RUNTIME_MAIN_PORT"
  "prod|PROD_RUNTIME_MAIN_PORT"
  "judge|JUDGE_RUNTIME_MAIN_PORT"
)

# Resolve a lane's main runtime port, or emit $LANE_PORT_UNRESOLVED. A value that
# is not a bare integer is also unresolved — probing "http://host:garbage/health"
# would just produce a confusing connection error instead of naming the real
# problem (the lane map).
lane_main_port() {
  local key="$1" value
  if ! value="$(policy_env_value "$key")" || [[ ! "$value" =~ ^[0-9]+$ ]]; then
    printf '%s' "$LANE_PORT_UNRESOLVED"
    return 0
  fi
  printf '%s' "$value"
}

post_slack() {
  local text="$1"
  local color="${2:-#439FE0}"
  local payload
  payload=$(jq -n \
    --arg channel "$SLACK_CHANNEL_ID" \
    --arg text "$text" \
    --arg color "$color" \
    '{channel:$channel,text:$text,attachments:[{color:$color,text:$text,mrkdwn_in:["text"]}]}')
  curl -fsS --retry 2 --max-time 10 \
    -H "Authorization: Bearer ${SLACK_BOT_TOKEN}" \
    -H 'Content-Type: application/json; charset=utf-8' \
    -d "$payload" \
    https://slack.com/api/chat.postMessage | jq -e '.ok == true' >/dev/null
}

df_line() {
  local mount="$1"
  df -BG --output=target,size,used,avail,pcent "$mount" | awk 'NR==2 {gsub("G","",$4); gsub("%","",$5); print $1"|"$2"|"$3"|"$4"|"$5}'
}

classify_disk() {
  local mount="$1" pct="$2" avail_gb="$3" warn_pct="$4" crit_pct="$5" warn_free="$6" crit_free="$7"
  if (( pct >= crit_pct || avail_gb <= crit_free )); then
    echo "CRITICAL"
  elif (( pct >= warn_pct || avail_gb <= warn_free )); then
    echo "WARNING"
  else
    echo "OK"
  fi
}

check_http() {
  local label="$1" url="$2" expect_regex="${3:-}"
  local tmp code status body body_excerpt
  tmp=$(mktemp)
  code=$(curl -sS --max-time "$PROBE_TIMEOUT_SECONDS" -o "$tmp" -w '%{http_code}' "$url" 2>/dev/null || true)
  # Match against the WHOLE body; truncate only the excerpt that gets reported.
  # See BODY_EXCERPT_BYTES — matching an excerpt makes the verdict depend on
  # where the payload happens to be cut.
  body=$(tr '\n' ' ' <"$tmp")
  body_excerpt=$(printf '%s' "$body" | head -c "$BODY_EXCERPT_BYTES")
  rm -f "$tmp"
  status="OK"
  if [[ ! "$code" =~ ^2 ]]; then
    status="CRITICAL"
  elif [[ -n "$expect_regex" ]] && ! grep -Eiq "$expect_regex" <<<"$body"; then
    status="WARNING"
  fi
  printf '%s|%s|%s|%s\n' "$status" "$label" "${code:-000}" "$body_excerpt"
}

# Resolve a runtime /health body to healthy / unhealthy / unresolvable.
#
# Substring matching is NOT usable here: `{"healthy": false}` contains both
# "healthy" and "false", and the old `grep -Ei 'healthy|ok'` scored it OK. The
# body is parsed as JSON and an explicit boolean/status field is required.
# Anything that cannot be resolved to an affirmative healthy signal is
# unresolvable, and unresolvable is CRITICAL (fail-closed) — a monitor that
# cannot tell is not allowed to say green.
runtime_body_verdict() {
  local body="$1" verdict
  verdict=$(jq -r '
    def norm: if type == "string" then ascii_downcase else . end;
    if type != "object" then "unresolvable"
    elif (.healthy? != null and (.healthy | type) == "boolean")
      then (if .healthy then "healthy" else "unhealthy" end)
    elif (.details?.healthy? != null and (.details.healthy | type) == "boolean")
      then (if .details.healthy then "healthy" else "unhealthy" end)
    elif (.status? != null)
      then (if (.status | norm) == "healthy" or (.status | norm) == "ok" or (.status | norm) == "pass"
            then "healthy" else "unhealthy" end)
    else "unresolvable"
    end
  ' <<<"$body" 2>/dev/null) || verdict="unresolvable"
  [[ -n "$verdict" ]] || verdict="unresolvable"
  printf '%s' "$verdict"
}

# Probe one lane's MAIN runtime health endpoint. Read-only GET on every lane,
# prod included.
check_runtime_lane() {
  local lane="$1" port="$2"
  local label="runtime-${lane}-${port}"
  local tmp code body body_excerpt status detail verdict
  tmp=$(mktemp)
  code=$(curl -sS -X GET --max-time "$PROBE_TIMEOUT_SECONDS" -o "$tmp" -w '%{http_code}' "http://${PROBE_HOST}:${port}/health" 2>/dev/null || true)
  # The verdict is computed from the FULL body; only the reported excerpt is
  # truncated. Truncating BEFORE the verdict was OMN-15525: a real .201 runtime
  # /health body is ~2.6 KB, so the 180-byte excerpt handed to jq was always
  # invalid JSON ("Unfinished string at EOF"), every lane resolved to
  # "unresolvable", and fail-closed turned that into CRITICAL on a fully healthy
  # fleet. Fail-closed is right; feeding it a mutilated input is not.
  body=$(tr '\n' ' ' <"$tmp")
  body_excerpt=$(printf '%s' "$body" | head -c "$BODY_EXCERPT_BYTES")
  rm -f "$tmp"
  code="${code:-000}"

  if [[ ! "$code" =~ ^2 ]]; then
    # Covers 5xx AND connection-refused/timeout (code 000). Never skipped.
    status="CRITICAL"
    detail="$body_excerpt"
  else
    verdict=$(runtime_body_verdict "$body")
    case "$verdict" in
      healthy)      status="OK";       detail="$body_excerpt" ;;
      unhealthy)    status="CRITICAL"; detail="HTTP 200 but health body is NOT healthy: $body_excerpt" ;;
      *)            status="CRITICAL"; detail="HTTP 200 but health status could not be resolved from body (fail-closed): $body_excerpt" ;;
    esac
  fi
  printf '%s|%s|%s|%s\n' "$status" "$label" "$code" "$detail"
}

# ISO-8601 -> epoch seconds. GNU `date -d` on the deploy host; BSD `date -j -f`
# fallback so the hermetic test can drive this same artifact on macOS (rule 11a
# runs gates on .200). Returns 0 when the timestamp cannot be parsed, which the
# caller treats as "cannot age this container" -> fail-closed.
epoch_from_iso() {
  local ts="$1" out
  [[ -n "$ts" ]] || { printf '0'; return 0; }
  out=$(date -u -d "$ts" +%s 2>/dev/null || true)
  if [[ ! "$out" =~ ^[0-9]+$ ]]; then
    # Docker emits e.g. 2026-07-30T16:19:02.123456789Z — trim to whole seconds.
    local trimmed="${ts%%.*}"
    trimmed="${trimmed%Z}"
    out=$(date -u -j -f '%Y-%m-%dT%H:%M:%S' "$trimmed" +%s 2>/dev/null || true)
  fi
  [[ "$out" =~ ^[0-9]+$ ]] || out=0
  printf '%s' "$out"
}

# Containers stuck in `health: starting` past their own start_period.
#
# This is the state that produced the silent 26-minute outage: a healthcheck
# that has never once passed reports neither `unhealthy` nor `Exited`, so every
# count the old script kept was zero.
starting_past_start_period() {
  local names name started start_period_ns grace_s started_epoch now_epoch age_s
  now_epoch=$(date -u +%s)
  names=$(docker ps --filter 'health=starting' --format '{{.Names}}' 2>/dev/null || echo "__DOCKER_QUERY_FAILED__")
  if [[ "$names" == "__DOCKER_QUERY_FAILED__" ]]; then
    printf '%s' "__DOCKER_QUERY_FAILED__"
    return 0
  fi
  local out=()
  while IFS= read -r name; do
    [[ -n "$name" ]] || continue
    started=$(docker inspect -f '{{.State.StartedAt}}' "$name" 2>/dev/null || true)
    start_period_ns=$(docker inspect -f '{{if .Config.Healthcheck}}{{.Config.Healthcheck.StartPeriod}}{{else}}0{{end}}' "$name" 2>/dev/null || true)
    [[ "$start_period_ns" =~ ^[0-9]+$ ]] || start_period_ns=0
    grace_s=$(( start_period_ns / 1000000000 ))
    (( grace_s > 0 )) || grace_s="$STARTING_GRACE_SECONDS"
    started_epoch=$(epoch_from_iso "$started")
    if (( started_epoch == 0 )); then
      # Cannot age the container -> cannot prove it is still inside its grace.
      out+=("${name}(age-unknown)")
      continue
    fi
    age_s=$(( now_epoch - started_epoch ))
    if (( age_s > grace_s )); then
      out+=("${name}(${age_s}s>${grace_s}s)")
    fi
  done <<<"$names"
  printf '%s' "${out[*]:-}"
}

# Containers that exited non-zero. Exit(0) one-shots (migration/init) are
# expected to finish and must not alarm.
exited_nonzero() {
  local raw
  raw=$(docker ps -a --format '{{.Names}}\t{{.Status}}' 2>/dev/null || echo "__DOCKER_QUERY_FAILED__")
  if [[ "$raw" == "__DOCKER_QUERY_FAILED__" ]]; then
    printf '%s' "__DOCKER_QUERY_FAILED__"
    return 0
  fi
  awk -F'\t' '$2 ~ /Exited \([1-9][0-9]*\)/ {printf "%s ", $1}' <<<"$raw" | sed 's/ $//'
}

# Required GitHub status contexts that never reported (OMN-15550).
#
# WHY THIS LIVES HERE AND NOT IN GITHUB ACTIONS
#   A required check that never reports is ABSENT, not RED. Branch protection
#   blocks the PR identically, but an absent context has no row in any list, so
#   `gh pr checks` reads all-green while every PR in the repo is unmergeable.
#   On 2026-07-30 (OMN-15536) `omnibase_infra`'s ci.yml failed to assemble;
#   `CI Summary` is that repo's SOLE required context, so all 7 open PRs wedged
#   silently for ~2.5h until a human noticed. A detector living inside the CI
#   system it watches would have failed to assemble with it -- so it runs here,
#   on a host that does not depend on GitHub Actions.
#
#   Folding it into this reporter rather than building a second alerter is the
#   net-negative-surface rule: it inherits this script's Slack poster, its
#   state-change de-duplication, its resolved-notification and its */15 cron.
#   No new cron unit, no second Slack integration.
#
# The probe emits `ci|STATUS|key|detail` rows, which `row_status()` reads at
# column 2 and `row_key()` de-duplicates as `ci|<key>`. A probe failure is a
# WARNING row, never silence: "could not look" must not render as "nothing
# wrong". It is deliberately not CRITICAL -- an unreachable API is not evidence
# that PRs are stranded, and paging on every network blip mutes the channel.
check_ci_required_contexts() {
  # Both artifacts are installed side by side in /data/maintenance/bin by the
  # host maintenance sync, so `dirname $0` resolves the probe on the host. In
  # the repo the probe lives under scripts/ (the env-read gate's approved
  # location for operational Python), not next to this file.
  local probe="${OMNINODE_CI_PROBE_SCRIPT:-$(dirname "$0")/omninode-ci-required-context-probe.py}"
  local python_bin="${OMNINODE_CI_PROBE_PYTHON:-python3}"

  if [[ "${OMNINODE_CI_PROBE_ENABLED:-1}" != "1" ]]; then
    return 0
  fi
  if [[ ! -r "$probe" ]]; then
    printf 'ci|WARNING|required-contexts|probe script missing or unreadable at %s\n' "$probe"
    return 0
  fi
  if ! command -v "$python_bin" >/dev/null 2>&1; then
    printf 'ci|WARNING|required-contexts|%s not found; required-context probe did not run\n' "$python_bin"
    return 0
  fi

  local out
  # A hung probe must not wedge the whole 15-minute health tick, so it is
  # bounded and a timeout is reported as a WARNING row like any other
  # "we could not look" outcome.
  if ! out=$(timeout "${OMNINODE_CI_PROBE_TIMEOUT:-120}" "$python_bin" "$probe" 2>/dev/null); then
    printf 'ci|WARNING|required-contexts|probe exited non-zero or timed out; required-context state unknown\n'
    return 0
  fi
  if [[ -z "$out" ]]; then
    printf 'ci|WARNING|required-contexts|probe produced no rows; required-context state unknown\n'
    return 0
  fi
  printf '%s\n' "$out"
}

collect() {
  local now host root data root_status data_status running unhealthy restarting dead created
  local dangling named_dangling anonymous_dangling docker_status docker_detail
  local starting_stuck exited_bad lane spec key port
  now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  host=$(hostname)
  root=$(df_line /)
  data=$(df_line /data)
  IFS='|' read -r _ root_size root_used root_avail root_pct <<<"$root"
  IFS='|' read -r _ data_size data_used data_avail data_pct <<<"$data"
  root_status=$(classify_disk / "$root_pct" "$root_avail" "$ROOT_WARN_PCT" "$ROOT_CRIT_PCT" "$ROOT_WARN_FREE_GB" "$ROOT_CRIT_FREE_GB")
  data_status=$(classify_disk /data "$data_pct" "$data_avail" "$DATA_WARN_PCT" "$DATA_CRIT_PCT" "$DATA_WARN_FREE_GB" "$DATA_CRIT_FREE_GB")

  running=$(docker ps --format '{{.Names}}' | wc -l | tr -d ' ')
  unhealthy=$(docker ps -a --format '{{.Names}}\t{{.Status}}' | grep -Eci 'unhealthy' || true)
  restarting=$(docker ps -a --format '{{.Names}}\t{{.Status}}' | grep -Eci 'Restarting' || true)
  dead=$(docker ps -a --format '{{.Names}}\t{{.Status}}' | grep -Eci 'Dead' || true)
  created=$(docker ps -a --format '{{.Names}}\t{{.Status}}' | grep -Eci 'Created' || true)
  dangling=$(docker volume ls -qf dangling=true | wc -l | tr -d ' ')
  anonymous_dangling=$(docker volume ls -qf dangling=true | grep -Ec '^[0-9a-f]{64}$' || true)
  named_dangling=$(docker volume ls -qf dangling=true | grep -Evc '^[0-9a-f]{64}$' || true)

  starting_stuck=$(starting_past_start_period)
  exited_bad=$(exited_nonzero)

  docker_status=OK
  docker_detail="unhealthy=$unhealthy restarting=$restarting dead=$dead created=$created"
  if [[ "$starting_stuck" == "__DOCKER_QUERY_FAILED__" || "$exited_bad" == "__DOCKER_QUERY_FAILED__" ]]; then
    # Fail-closed: a docker query that did not run is not evidence of health.
    docker_status=CRITICAL
    docker_detail="$docker_detail docker_query=FAILED"
  else
    docker_detail="$docker_detail starting_past_start_period=${starting_stuck:-none} exited_nonzero=${exited_bad:-none}"
    if [[ "$unhealthy" != 0 || "$restarting" != 0 || "$dead" != 0 || "$created" != 0 \
          || -n "$starting_stuck" || -n "$exited_bad" ]]; then
      docker_status=CRITICAL
    fi
  fi

  {
    echo "timestamp|$now"
    echo "host|$host"
    echo "disk|$root_status|/|${root_used}/${root_size}|${root_avail}G free|${root_pct}%"
    echo "disk|$data_status|/data|${data_used}/${data_size}|${data_avail}G free|${data_pct}%"
    echo "docker|OK|running_containers|$running"
    echo "docker|$docker_status|container_issues|$docker_detail"
    echo "docker|OK|dangling_volumes|total=$dangling anonymous=$anonymous_dangling named=$named_dangling"
    # Every lane's MAIN runtime health endpoint, from the lane->port map.
    for spec in "${RUNTIME_LANE_SPECS[@]}"; do
      IFS='|' read -r lane key <<<"$spec"
      port=$(lane_main_port "$key")
      if [[ "$port" == "$LANE_PORT_UNRESOLVED" ]]; then
        # Rule 8 / OMN-15525: refuse to probe a guessed port. An unresolvable
        # lane map is itself the outage-shaped fact worth alarming on.
        printf 'CRITICAL|runtime-%s-unresolved|000|lane main port unresolvable: key %s missing from %s\n' \
          "$lane" "$key" "$(runtime_policy_file)"
        continue
      fi
      check_runtime_lane "$lane" "$port"
    done
    check_http projection-api-13002 "http://${PROBE_HOST}:13002/health" 'ok|healthy'
    # OMN-16789 follow-up: `deploy-agent-8099` is REMOVED, not disabled.
    #
    # It probed a service that does not exist on this host. Verified on `.201`
    # 2026-08-27: no listener on 8099 (`ss -lntp`), no container publishing it
    # (`docker ps -a`), `curl :8099/health` -> 000. The real deploy runner
    # (`omninode-deploy-runner`, healthy) publishes NO ports --
    # `NetworkSettings.Ports` is `{}` -- because it is HMAC-command driven
    # (DEPLOY_AGENT_HMAC_SECRET), not an HTTP service. Port 8099 is allocated by
    # the service catalog to an unrelated profile-gated fixture:
    # `docker/catalog/services/fault-inject-fixture.yaml` -> ports.external 8099.
    #
    # So this probe could only ever return 000. Fail-closed on 000 is correct
    # for a real endpoint and is unchanged everywhere else; the defect was that
    # this endpoint was never real. It was CRITICAL on all 39 ticks measured in
    # OMN-16789 and one of only two standing criticals on the host -- permanent
    # false-RED, the "crying wolf" direction this file's header calls fatal to a
    # monitor. A monitor nobody believes is a monitor that is not running.
    #
    # If a deploy agent ever does expose HTTP health, add it back with its real
    # port from the service catalog rather than restoring this literal.
    check_http web-3003 "http://${PROBE_HOST}:3003/" ''
    check_ci_required_contexts
  }
}

snapshot=$(collect)
if [[ "$MODE" != "dry-run" ]]; then
  echo "$snapshot"
fi

host=$(awk -F'|' '$1=="host"{print $2}' <<<"$snapshot")

# The snapshot carries TWO row shapes and the status lives in a different
# column in each:
#
#   disk|STATUS|name|...            <- resource rows, status in $2
#   docker|STATUS|name|...
#   STATUS|label|code|detail        <- endpoint rows, status in $1
#
# OMN-15525: every selector below used to test `$2` only, so NO endpoint row
# could ever land in `$issues` / the counters. `format_digest` handled both
# shapes, which is why the rendered text listed `CRITICAL runtime-dev-8085`
# under *Active issues* while the header said `0 critical` — and, far worse,
# why `--mode alert` computed an EMPTY `$issues`, took the "clean" branch, and
# paged nobody. A dead runtime lane could not raise an alert even after the
# probe was fixed: OMN-15509 taught the reporter to SEE the lane, and this is
# what stopped it from SAYING anything. Verified live on .201 — the merged
# revision printed three CRITICAL lanes above `Issues: *0 critical*`.
#
# `row_status` is the single definition of "this row's status" and everything
# downstream keys off it.
row_status='function row_status() { return ($1=="OK" || $1=="WARNING" || $1=="CRITICAL") ? $1 : $2 }'
# Stable identity for alert de-duplication: label + status only. Volatile
# fields (HTTP code, body excerpt, free-GB) are deliberately excluded so a
# flapping 000/503 on one dead lane is one alert, not one per tick.
row_key='function row_key() { return ($1=="OK" || $1=="WARNING" || $1=="CRITICAL") ? $2 : $1 "|" $3 }'

issues=$(awk -F'|' "$row_status"'{ s=row_status() } s=="WARNING" || s=="CRITICAL" {print}' <<<"$snapshot" || true)
issue_keys=$(awk -F'|' "$row_status$row_key"'{ s=row_status() } s=="WARNING" || s=="CRITICAL" {print row_key() "|" s}' <<<"$snapshot" || true)
critical_count=$(awk -F'|' "$row_status"'{ s=row_status() } s=="CRITICAL" {c++} END {print c+0}' <<<"$snapshot")
warning_count=$(awk -F'|' "$row_status"'{ s=row_status() } s=="WARNING" {c++} END {print c+0}' <<<"$snapshot")
# --- per-key alert state machine (OMN-16789) -------------------------------
#
# WHAT WAS WRONG WITH THE SINGLE HASH
#   The previous revision reduced the whole issue set to one sha256 and posted
#   whenever that hash changed. That is correct only if the input is stable.
#   It was not: `runtime-stability-test-18085` alternated CRITICAL(000) /
#   OK(200) on almost every tick (a 4s probe timeout against a ~3.2s response —
#   see PROBE_TIMEOUT_SECONDS), so the set flipped between {c,s,d} and {c,d},
#   the hash changed every tick, and the de-duplication never engaged once.
#   Measured 2026-08-27 09:30Z-18:30Z: 39 ticks, 22 posted, 17 suppressed, and
#   every post traced to that one key bouncing.
#
#   Worse, the shrink direction did not read as a recovery: with `$issues` still
#   non-empty the alert branch simply re-posted the entire digest minus one
#   line. The operator saw "the same alert again", because it was.
#
# WHAT REPLACES IT
#   State is per key, not per set:
#       key <TAB> status <TAB> present_streak <TAB> absent_streak <TAB> notified <TAB> last_notified_epoch
#
#   A key pages once when it has held the same status for CONFIRM_TICKS. While
#   it is briefly absent (fewer than CLEAR_TICKS ticks) its row and its notified
#   flag are RETAINED, so a key that bounces back does not re-arm — this is what
#   actually absorbs the flap. Only after CLEAR_TICKS consecutive absences is it
#   reported recovered and dropped. A key that is still standing is re-surfaced
#   every RENOTIFY_SECONDS so a permanent critical cannot go permanently quiet.
#
#   A status CHANGE on a present key (WARNING <-> CRITICAL) resets the streak
#   and re-arms notification: an escalation is a real state change, not noise.
#
# FAIL-CLOSED: an unreadable or malformed state file is treated as empty, which
# re-pages standing issues rather than silencing them. Losing state must never
# be the quiet outcome.
state_file="$STATE_DIR/omninode-system-alert-keys.tsv"
now_epoch=$(date -u +%s)
current_keys_file=$(mktemp)
prev_state_file=$(mktemp)
printf '%s\n' "$issue_keys" | awk -F'|' 'NF>1 { st=$NF; sub(/\|[^|]*$/, "", $0); print $0 "\t" st }' >"$current_keys_file"
# A row with the wrong field count is dropped rather than trusted (fail-closed
# to "unknown key" -> re-page), and a missing file is simply empty.
awk -F'\t' 'NF==6' "$state_file" 2>/dev/null >"$prev_state_file" || true

# The two-file pass keys off FILENAME, NOT the usual `NR==FNR` idiom. On the
# very first run the previous-state file is EMPTY, and `NR==FNR` is then true
# for the first record of the SECOND file too — so every current issue would be
# swallowed by the prior-state branch, no key would ever reach the decision
# loop, and the reporter would go permanently silent. That is a worse failure
# than the spam this replaces, and it is invisible until state happens to be
# empty. Matching on FILENAME has no such edge.
decisions=$(awk -F'\t' -v OFS='\t' \
  -v prevfile="$prev_state_file" \
  -v now="$now_epoch" \
  -v confirm="$CONFIRM_TICKS" \
  -v clear="$CLEAR_TICKS" \
  -v renotify="$RENOTIFY_SECONDS" '
  FILENAME==prevfile {
    k=$1
    p_status[k]=$2; p_present[k]=$3; p_absent[k]=$4; p_notified[k]=$5; p_last[k]=$6
    if (!(k in seen)) { seen[k]=1; p_order[++pn]=k }
    next
  }
  { k=$1; if (!(k in cur)) { cur[k]=$2; c_order[++cn]=k } }
  END {
    for (i=1; i<=cn; i++) {
      k=c_order[i]; s=cur[k]
      if ((k in seen) && p_status[k]==s) {
        present=p_present[k]+1; notified=p_notified[k]; last=p_last[k]
      } else {
        # New key, or a real status change: re-arm.
        present=1; notified=0; last=0
      }
      if (present>=confirm) {
        if (notified==0)                                   { print "NEW", k, s; notified=1; last=now }
        else if (renotify>0 && (now-last)>=renotify)       { print "RENOTIFY", k, s; last=now }
      }
      print "STATE", k, s, present, 0, notified, last
      handled[k]=1
    }
    for (i=1; i<=pn; i++) {
      k=p_order[i]
      if (k in handled) continue
      absent=p_absent[k]+1
      if (absent>=clear) {
        if (p_notified[k]==1) print "RECOVERED", k, p_status[k]
        continue  # dropped from state
      }
      # Below the clear threshold: keep the row, the notified flag AND the
      # present streak.
      #
      # Preserving the streak matters. Zeroing it looked tidier but meant a key
      # only ever needed one clean tick between failures to keep its streak at
      # 1 forever — so an endpoint failing every OTHER probe (a 50%-broken
      # service, the single most likely real degradation) could never reach
      # CONFIRM_TICKS and would never alert at all. Caught by the measured-flap
      # replay test, which contains no two consecutive failures anywhere in it.
      # The streak is still bounded: CLEAR_TICKS of continuous absence drops the
      # row entirely, so this accumulates evidence within a window rather than
      # forever.
      print "STATE", k, p_status[k], p_present[k], absent, p_notified[k], p_last[k]
    }
  }
' "$prev_state_file" "$current_keys_file")
rm -f "$current_keys_file" "$prev_state_file"

# Atomic swap: a crash mid-write must not leave a truncated state file that
# reads as "no issues known" and re-pages everything.
printf '%s\n' "$decisions" | awk -F'\t' -v OFS='\t' '$1=="STATE" { $1=""; sub(/^\t/, ""); print }' >"${state_file}.tmp"
mv -f "${state_file}.tmp" "$state_file"

new_keys=$(awk -F'\t' '$1=="NEW"       { print $2 " (" $3 ")" }' <<<"$decisions")
renotify_keys=$(awk -F'\t' '$1=="RENOTIFY"  { print $2 " (" $3 ")" }' <<<"$decisions")
recovered_keys=$(awk -F'\t' '$1=="RECOVERED" { print $2 }' <<<"$decisions")

format_digest() {
  local title="$1"
  local lines endpoint_lines ci_lines issue_lines
  lines=$(awk -F'|' '$1=="disk" {printf "- `%s`: %s, %s, %s (%s)\n", $3, $4, $5, $6, $2} $1=="docker" {printf "- Docker `%s`: %s (%s)\n", $3, $4, $2}' <<<"$snapshot")
  endpoint_lines=$(awk -F'|' '$1=="OK" || $1=="WARNING" || $1=="CRITICAL" {printf "- `%s`: HTTP %s (%s)\n", $2, $3, $1}' <<<"$snapshot")
  # OMN-15550. The heartbeat row renders here even when clean, so a reader can
  # tell "scanned N repos, found nothing" apart from "did not scan" -- the
  # detection-shelf blindness where a silent section reads as healthy.
  ci_lines=$(awk -F'|' '$1=="ci" {printf "- `%s`: %s (%s)\n", $3, $4, $2}' <<<"$snapshot")
  [[ -n "$ci_lines" ]] || ci_lines="- No required-context probe rows this tick"
  # `next` keeps the three row shapes mutually exclusive so a `ci` row cannot
  # also be rendered by the generic column-2 branch below it.
  issue_lines=$(awk -F'|' '$1=="ci" && ($2=="WARNING" || $2=="CRITICAL") {printf "- %s `%s`: %s\n", $2, $3, $4; next} $2=="WARNING" || $2=="CRITICAL" {printf "- %s `%s`: %s %s %s\n", $2, $3, $4, $5, $6; next} $1=="WARNING" || $1=="CRITICAL" {printf "- %s `%s`: HTTP %s %s\n", $1, $2, $3, $4}' <<<"$snapshot")
  if [[ -z "$issue_lines" ]]; then
    issue_lines="- No active warning/critical checks"
  fi
  cat <<MSG
$title
Host: $host
Issues: *$critical_count critical*, *$warning_count warning*

*Disk / Docker*
$lines
*Runtime endpoints*
$endpoint_lines
*CI required contexts*
$ci_lines
*Active issues*
$issue_lines
MSG
}

case "$MODE" in
  digest)
    post_slack "$(format_digest '*OmniNode morning system digest*')" '#439FE0'
    ;;
  alert)
    # Order matters: a tick that both escalates and recovers is an alert, and
    # the recovered keys ride along in the digest's own *Active issues* delta
    # rather than as a second message.
    if [[ -n "$new_keys" || -n "$renotify_keys" ]]; then
      color='warning'
      [[ "$critical_count" != 0 ]] && color='danger'
      post_slack "$(format_digest '*OmniNode system alert*')" "$color"
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) alert posted; new=[${new_keys//$'\n'/, }] renotify=[${renotify_keys//$'\n'/, }]"
    elif [[ -n "$recovered_keys" ]]; then
      # Per-key recovery. The old code could not express this: with any issue
      # still standing it re-posted the whole alert digest, so "one thing got
      # better" was indistinguishable from "here is the same alert again".
      recovered_list="${recovered_keys//$'\n'/, }"
      if [[ -z "$issues" ]]; then
        post_slack "*[OmniNode alert resolved]* .201 system checks are clean again. Recovered: ${recovered_list}." 'good'
      else
        post_slack "*[OmniNode alert resolved]* Recovered: ${recovered_list}. Still open: *${critical_count} critical*, *${warning_count} warning*." 'good'
      fi
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) recovery posted; recovered=[${recovered_list}]"
    else
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) no alert state change; Slack suppressed"
    fi
    ;;
  dry-run)
    format_digest '*OmniNode system dry run*'
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 2
    ;;
esac
