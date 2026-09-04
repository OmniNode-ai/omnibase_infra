#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# system_health_check.sh -- Canonical system health gate for ONEX infrastructure
#
# Composes individual service health checks into a single pass/fail gate.
# Designed for CI, pre-deploy verification, and local diagnostics.
#
# Usage:
#   bash scripts/system_health_check.sh [OPTIONS]
#
# Options:
#   --json          Output results as JSON
#   --ci            Non-interactive mode (implies --json, sets exit codes for CI)
#   --cross-repo    Enable cross-repo checks (env audit, cloud bus refs)
#   --lane          Lane-liveness subset ONLY (dev_lane_liveness, redpanda,
#                   runtime_containers). Runs from a containerized CI runner
#                   with only docker.sock + network access — deliberately
#                   skips the checks that need POSTGRES_HOST / VALKEY_HOST /
#                   INFISICAL_* credentials. This is the mode the scheduled
#                   enforcement surface runs (OMN-15190).
#   --verbose       Show detailed output for each check
#   --help          Show this help message
#
# Exit codes:
#   0  All checks green or yellow (advisory warnings only)
#   1  One or more checks red (hard failure)
#
# Checks performed:
#   1.  postgres          - PostgreSQL connectivity and omnibase_infra DB
#   2.  redpanda          - Redpanda/Kafka broker health
#   3.  valkey            - Valkey (Redis-compatible) connectivity
#   4.  infra_containers  - Core infra containers running
#   5.  keycloak          - Keycloak auth (yellow if not running)
#   6.  runtime_containers - Runtime profile containers (RED under keep-alive)
#   7.  required_topics   - Required Kafka topics exist
#   8.  migration_parity  - Docker and src migration directories in sync
#   9.  env_audit         - No rogue .env files (--cross-repo only)
#  10.  cloud_bus_refs    - No unsuppressed 29092 references (--cross-repo only)  # cloud-bus-ok OMN-4922
#  11.  bus_endpoint      - KAFKA_BOOTSTRAP_SERVERS must not contain 29092  # cloud-bus-ok OMN-4922
#  12.  infisical_folders - /shared/<transport>/ folders exist in Infisical (when INFISICAL_ADDR is set)
#  13.  dev_lane_liveness - The lab/dev compose lane is UP and reachable on the
#                           exact path CI publishers use (OMN-15190)
#
# LANE KEEP-ALIVE (ONEX_LANE_KEEPALIVE, default 1 — operator ruling 2026-07-29)
#
#   The lab/dev lane used to be documented as ephemeral-by-design (OMN-13414):
#   GC/idle-reclaimed to zero containers between uses, rediscovered reactively
#   by whichever PR happened to hit the resulting CI cascade. That posture is
#   REVERSED by operator ruling (WS-4): the lab lane is KEEP-ALIVE, because
#   testing things live is the entire point of the lab. Lane-down is therefore
#   a DEFECT, not an expected state.
#
#   This file is where that ruling changes behavior. Before it, a fully
#   torn-down lane scored `runtime_containers: yellow ("none running — runtime
#   profile not active")`, and yellow exits 0 — i.e. the canonical health gate
#   reported SUCCESS on a lane that was stranding every repo's
#   occ-autobind / occ-companion-effect publish with connection-refused.
#   Under keep-alive those states are RED. Set ONEX_LANE_KEEPALIVE=0 to restore
#   the advisory posture for a lane that genuinely is ephemeral.
#
# OMN-3772 OMN-3903 OMN-15190

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Resolve OMNI_HOME for cross-repo checks
OMNI_HOME="${OMNI_HOME:-/Volumes/PRO-G40/Code/omni_home}"

# ----- Flags -----
FLAG_JSON=false
FLAG_CI=false
FLAG_CROSS_REPO=false
FLAG_VERBOSE=false
FLAG_LANE=false

# ----- Lane keep-alive posture (OMN-15190) -----
# 1 = the lab/dev lane is expected to be UP at all times (operator ruling
# 2026-07-29). 0 = pre-ruling ephemeral posture, lane-absent is advisory only.
LANE_KEEPALIVE="${ONEX_LANE_KEEPALIVE:-1}"

# Lane identity + probe targets come from the rendered service contract
# (docker/runtime-policy.env, generated from contracts/services/
# runtime_policy.contract.yaml) — the same source refresh_dev_lane.sh reads,
# so the lane this check watches cannot drift from the lane the refresh
# script deploys.
#
# Read by targeted key extraction, NOT `source`: that file is a generated
# artifact whose key set changes when the contract is re-rendered, and
# sourcing it into this script's global scope would let a future render
# silently redefine POSTGRES_* / VALKEY_* / KAFKA_BOOTSTRAP_SERVERS and
# change what the OTHER checks in this file are asserting. Two keys are
# wanted; two keys are read.
policy_env_value() {
    local key="$1" file="${REPO_ROOT}/docker/runtime-policy.env"
    [[ -f "$file" ]] || return 0
    sed -n "s/^${key}=//p" "$file" | tail -n 1 | tr -d "\"'"
}
DEV_LANE_COMPOSE_PROJECT="${DEV_LANE_COMPOSE_PROJECT:-$(policy_env_value DEV_COMPOSE_PROJECT)}"
DEV_LANE_COMPOSE_PROJECT="${DEV_LANE_COMPOSE_PROJECT:-omnibase-infra}"
DEV_LANE_MAIN_PORT="${DEV_LANE_MAIN_PORT:-$(policy_env_value DEV_RUNTIME_MAIN_PORT)}"
DEV_LANE_MAIN_PORT="${DEV_LANE_MAIN_PORT:-8085}"
# The broker host-port CI publishers connect to. No contract var exists for it
# (the contract renders in-cluster addresses); this is the PUBLISHED host port
# that occ-autobind / occ-companion-effect dial.
DEV_LANE_BROKER_PORT="${DEV_LANE_BROKER_PORT:-19092}"
# fallback-ok: localhost IS the lane host in the documented primary context
# (this script runs ON the lane host); the containerized deploy runner
# overrides via compose env LANE_PROBE_HOST=host.docker.internal (OMN-14958),
# exactly as refresh_dev_lane.sh / refresh_stability_lane.sh do.
LANE_PROBE_HOST="${LANE_PROBE_HOST:-localhost}" # fallback-ok: localhost IS the lane host when this script runs ON the lane host (its documented primary context); the containerized deploy runner overrides via LANE_PROBE_HOST=host.docker.internal (OMN-14958), same as refresh_dev_lane.sh

# ----- State -----
OVERALL_STATUS="green"   # green | yellow | red
declare -a CHECK_NAMES=()
declare -a CHECK_STATUSES=()
declare -a CHECK_DETAILS=()

# ----- Helpers -----

json_escape() {
    local str="$1"
    str="${str//\\/\\\\}"
    str="${str//\"/\\\"}"
    str="${str//$'\n'/\\n}"
    str="${str//$'\r'/\\r}"
    str="${str//$'\t'/\\t}"
    printf '%s' "$str"
}

log_check() {
    local name="$1" status="$2" detail="$3"
    CHECK_NAMES+=("$name")
    CHECK_STATUSES+=("$status")
    CHECK_DETAILS+=("$detail")

    # Promote overall status (skip does not affect overall)
    case "$status" in
        red)    OVERALL_STATUS="red" ;;
        yellow) [[ "$OVERALL_STATUS" != "red" ]] && OVERALL_STATUS="yellow" ;;
    esac

    if [[ "$FLAG_JSON" == "false" ]]; then
        local icon
        case "$status" in
            green)  icon="[GREEN]" ;;
            yellow) icon="[YELLOW]" ;;
            red)    icon="[RED]" ;;
            skip)   icon="[SKIP]" ;;
        esac
        printf "  %-8s %-22s %s\n" "$icon" "$name" "$detail"
    fi
}

show_help() {
    sed -n '2,40p' "${BASH_SOURCE[0]}" | grep '^#' | sed 's/^# \?//'
    exit 0
}

# ----- Parse arguments -----

while [[ $# -gt 0 ]]; do
    case "$1" in
        --json)       FLAG_JSON=true; shift ;;
        --ci)         FLAG_CI=true; FLAG_JSON=true; shift ;;
        --cross-repo) FLAG_CROSS_REPO=true; shift ;;
        --lane)       FLAG_LANE=true; shift ;;
        --verbose)    FLAG_VERBOSE=true; shift ;;
        --help|-h)    show_help ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

# =====================================================================
# Check functions
# =====================================================================

check_postgres() {
    local name="postgres"
    # Try connecting via psql
    if ! command -v psql >/dev/null 2>&1; then
        log_check "$name" "red" "psql not found in PATH"
        return
    fi

    local pg_host="${POSTGRES_HOST:?POSTGRES_HOST required}"
    local pg_port="${POSTGRES_PORT:-5436}"
    local pg_user="${POSTGRES_USER:-postgres}"
    local pg_db="${POSTGRES_DB:-omnibase_infra}"

    local result
    if result=$(PGPASSWORD="${POSTGRES_PASSWORD:-}" psql -h "$pg_host" -p "$pg_port" -U "$pg_user" -d "$pg_db" -c "SELECT 1" -t -A 2>&1); then
        if [[ "$result" == *"1"* ]]; then
            log_check "$name" "green" "connected to ${pg_db} on ${pg_host}:${pg_port}"
        else
            log_check "$name" "red" "unexpected query result: $(json_escape "$result")"
        fi
    else
        log_check "$name" "red" "connection failed: $(json_escape "$result")"
    fi
}

check_redpanda() {
    local name="redpanda"

    # Check via rpk inside container first
    local result
    if result=$(docker exec omnibase-infra-redpanda rpk cluster health 2>&1); then
        if echo "$result" | grep -qi "healthy"; then
            log_check "$name" "green" "cluster healthy"
        else
            log_check "$name" "yellow" "cluster response: $(json_escape "$result")"
        fi
    else
        # Fallback: check if container is running
        if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "omnibase-infra-redpanda"; then
            log_check "$name" "yellow" "container running but rpk health failed"
        else
            log_check "$name" "red" "container not running"
        fi
    fi
}

check_valkey() {
    local name="valkey"
    local vk_host="${VALKEY_HOST:?VALKEY_HOST required}"
    local vk_port="${VALKEY_PORT:-16379}"
    local vk_pass="${VALKEY_PASSWORD:-${REDIS_PASSWORD:-}}"

    # Build auth args for CLI
    local auth_args=()
    if [[ -n "$vk_pass" ]]; then
        auth_args=(-a "$vk_pass")
    fi

    # Try docker exec first
    local result
    if result=$(docker exec omnibase-infra-valkey valkey-cli "${auth_args[@]}" ping 2>&1); then
        if [[ "$result" == *"PONG"* ]]; then
            log_check "$name" "green" "PONG on container"
        elif [[ "$result" == *"NOAUTH"* ]]; then
            log_check "$name" "red" "auth required (set VALKEY_PASSWORD)"
        else
            log_check "$name" "red" "unexpected response: $(json_escape "$result")"
        fi
    elif command -v redis-cli >/dev/null 2>&1; then
        if result=$(redis-cli -h "$vk_host" -p "$vk_port" "${auth_args[@]}" ping 2>&1); then
            if [[ "$result" == *"PONG"* ]]; then
                log_check "$name" "green" "PONG on ${vk_host}:${vk_port}"
            else
                log_check "$name" "red" "unexpected response: $(json_escape "$result")"
            fi
        else
            log_check "$name" "red" "connection failed: $(json_escape "$result")"
        fi
    else
        # Check container status as last resort
        if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "omnibase-infra-valkey"; then
            log_check "$name" "yellow" "container running but cannot verify (no valkey-cli or redis-cli)"
        else
            log_check "$name" "red" "container not running and no CLI available"
        fi
    fi
}

check_infra_containers() {
    local name="infra_containers"
    local required=("omnibase-infra-postgres" "omnibase-infra-redpanda" "omnibase-infra-valkey")
    local running
    running=$(docker ps --format '{{.Names}}' 2>/dev/null) || true

    local missing=()
    for c in "${required[@]}"; do
        if ! echo "$running" | grep -q "^${c}$"; then
            missing+=("$c")
        fi
    done

    if [[ ${#missing[@]} -eq 0 ]]; then
        log_check "$name" "green" "all core containers running (${#required[@]}/${#required[@]})"
    else
        log_check "$name" "red" "missing: ${missing[*]}"
    fi
}

check_keycloak() {
    local name="keycloak"
    local running
    running=$(docker ps --format '{{.Names}}' 2>/dev/null) || true

    if echo "$running" | grep -q "omnibase-infra-keycloak"; then
        log_check "$name" "green" "container running"
    else
        log_check "$name" "yellow" "not running (auth profile not active)"
    fi
}

check_runtime_containers() {
    local name="runtime_containers"
    local expected=("omninode-runtime" "omninode-runtime-effects" "omnibase-intelligence-api")
    local running
    running=$(docker ps --format '{{.Names}}' 2>/dev/null) || true

    local found=0
    local missing=()
    for c in "${expected[@]}"; do
        if echo "$running" | grep -q "^${c}$"; then
            ((found++)) || true
        else
            missing+=("$c")
        fi
    done

    # OMN-15190: under the keep-alive ruling a missing runtime service is a
    # DEFECT, not "profile not active". Yellow exits 0, so the pre-ruling
    # severity made this gate report success on a lane that was down.
    local absent_severity="yellow"
    local absent_note=" (runtime profile not active)"
    if [[ "$LANE_KEEPALIVE" == "1" ]]; then
        absent_severity="red"
        absent_note=" — lab lane is KEEP-ALIVE (ONEX_LANE_KEEPALIVE=1), a missing runtime service is a defect"
    fi

    if [[ ${#missing[@]} -eq 0 ]]; then
        log_check "$name" "green" "all runtime containers running (${found}/${#expected[@]})"
    elif [[ $found -gt 0 ]]; then
        log_check "$name" "$absent_severity" "partial: ${found}/${#expected[@]} running, missing: ${missing[*]}${absent_note}"
    else
        log_check "$name" "$absent_severity" "none running${absent_note}"
    fi
}

# Bounded TCP connect probe.
#
# Deliberately NOT `nc -w`: that flag bounds READS, not the connect itself, so
# an `nc -w 5` against a black-holed host can hang far past the timeout
# (memory `reference_nc_w_flag_does_not_bound_connect`). bash's /dev/tcp does
# the connect; the bound comes from coreutils `timeout` where present and from
# an explicit watchdog otherwise, so this works on the BSD-userland gate host
# where `timeout` is absent.
#
# Returns 0 when the port accepts a connection, 1 otherwise.
tcp_probe() {
    local host="$1" port="$2" secs="${3:-5}"

    if command -v timeout >/dev/null 2>&1; then
        timeout "$secs" bash -c "exec 3<>/dev/tcp/${host}/${port}" 2>/dev/null
        return $?
    fi

    ( exec 3<>/dev/tcp/"${host}"/"${port}" ) 2>/dev/null &
    local probe_pid=$!
    local waited=0
    while kill -0 "$probe_pid" 2>/dev/null && [[ "$waited" -lt "$secs" ]]; do
        sleep 1
        waited=$((waited + 1))
    done
    if kill -0 "$probe_pid" 2>/dev/null; then
        kill -9 "$probe_pid" 2>/dev/null || true
        wait "$probe_pid" 2>/dev/null || true
        return 1
    fi
    wait "$probe_pid"
}

check_dev_lane_liveness() {
    local name="dev_lane_liveness"
    local project="$DEV_LANE_COMPOSE_PROJECT"

    # Indeterminate is not health. A check that cannot see the lane must not
    # report the lane as fine — that is the exact inversion OMN-13915 shipped.
    if ! command -v docker >/dev/null 2>&1; then
        log_check "$name" "red" "docker CLI unavailable — lane liveness is unprovable (fail-closed)"
        return
    fi

    local rows
    if ! rows=$(docker ps -a --filter "label=com.docker.compose.project=${project}" \
        --format '{{.Names}}|{{.State}}|{{.Status}}' 2>&1); then
        log_check "$name" "red" "docker daemon unreachable: $(json_escape "$rows") (fail-closed)"
        return
    fi

    # The compose PROJECT LABEL is the membership oracle, not a hardcoded
    # container-name list: the dev lane's compose file leaves some services
    # without an explicit container_name (compose-assigned), so a name map
    # silently under-counts — the same trap verify_dev_refresh.py documents.
    local total=0
    [[ -n "$rows" ]] && total=$(printf '%s\n' "$rows" | grep -c '|')

    if [[ "$total" -eq 0 ]]; then
        if [[ "$LANE_KEEPALIVE" == "1" ]]; then
            log_check "$name" "red" \
                "compose project '${project}' has ZERO containers — the lab lane is fully GC/idle-reclaimed (OMN-15190). Under the keep-alive ruling this is a defect: while it is down, every repo's occ-autobind / occ-companion-effect publish fails connection-refused and cascades into occ-preflight / Receipt Gate org-wide. Recovery: knowledge-base:runbooks/cold-lane-full-bringup.md"
        else
            log_check "$name" "yellow" \
                "compose project '${project}' has ZERO containers (ONEX_LANE_KEEPALIVE=0 — lane treated as ephemeral per OMN-13414)"
        fi
        return
    fi

    local running=0
    local exited_nonzero=() not_running=() unhealthy=()
    local cname cstate cstatus
    while IFS='|' read -r cname cstate cstatus; do
        [[ -z "$cname" ]] && continue
        case "$cstate" in
            running)
                ((running++)) || true
                # Docker health is used here ONLY as a secondary signal. It is
                # never the sole verdict (OMN-13915: 37/48 runners sat
                # "Up (healthy)" with a dead listener; OMN-15233: 59/64 read
                # unhealthy while the registry said 64/64 online).
                [[ "$cstatus" == *"(unhealthy)"* ]] && unhealthy+=("$cname")
                ;;
            exited)
                # One-shots (migrations, provisioners) legitimately exit 0. A
                # NONZERO exit is a real failure that leaves the lane serving
                # with an unapplied schema — the OMN-15312 class, which is
                # invisible to every "is it up?" probe.
                if [[ "$cstatus" =~ Exited\ \(([0-9]+)\) ]]; then
                    [[ "${BASH_REMATCH[1]}" != "0" ]] && exited_nonzero+=("${cname}(exit ${BASH_REMATCH[1]})")
                fi
                ;;
            *)
                # created / restarting / dead / paused — a container stuck in
                # any of these in a keep-alive lane is a defect, and
                # `restarting` specifically is a crash loop.
                not_running+=("${cname}(${cstate})")
                ;;
        esac
    done <<< "$rows"

    # Reachability on the EXACT path the CI publishers use. This is the signal
    # that actually decides whether the org's receipt path works; container
    # state alone does not (a running broker with an unpublished/unroutable
    # host port strands CI just as completely as a torn-down lane).
    local red_reasons=()
    if ! tcp_probe "$LANE_PROBE_HOST" "$DEV_LANE_BROKER_PORT" 5; then
        red_reasons+=("broker ${LANE_PROBE_HOST}:${DEV_LANE_BROKER_PORT} refused connection (the occ-autobind / occ-companion-effect publish path)")
    fi
    local http_code
    http_code=$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 3 --max-time 8 \
        "http://${LANE_PROBE_HOST}:${DEV_LANE_MAIN_PORT}/health" 2>/dev/null) || http_code="000"
    if [[ "$http_code" != "200" ]]; then
        red_reasons+=("runtime /health on ${LANE_PROBE_HOST}:${DEV_LANE_MAIN_PORT} returned ${http_code}")
    fi
    [[ ${#exited_nonzero[@]} -gt 0 ]] && red_reasons+=("nonzero-exit container(s): ${exited_nonzero[*]}")
    [[ ${#not_running[@]} -gt 0 ]] && red_reasons+=("not-running container(s): ${not_running[*]}")

    local census="${total} container(s), ${running} running, probe host ${LANE_PROBE_HOST}"
    if [[ ${#red_reasons[@]} -gt 0 ]]; then
        local joined="${red_reasons[0]}"
        local i
        for ((i = 1; i < ${#red_reasons[@]}; i++)); do
            joined="${joined}; ${red_reasons[$i]}"
        done
        log_check "$name" "red" "lane '${project}' DEGRADED — ${joined} [${census}]"
        return
    fi

    if [[ ${#unhealthy[@]} -gt 0 ]]; then
        # Advisory, not red: a running-but-unhealthy sidecar does not strand
        # the CI publish path, and a permanently-red check is a disabled check.
        log_check "$name" "yellow" "lane '${project}' serving, but docker-unhealthy: ${unhealthy[*]} [${census}]"
        return
    fi

    log_check "$name" "green" "lane '${project}' up and reachable — broker :${DEV_LANE_BROKER_PORT} open, /health 200 [${census}]"
}

check_required_topics() {
    local name="required_topics"

    # Core topics that should always exist when Redpanda is healthy
    local required_topics=(
        "agent-actions"
        "agent-transformation-events"
    )

    local result
    if ! result=$(docker exec omnibase-infra-redpanda rpk topic list 2>&1); then
        log_check "$name" "yellow" "cannot list topics (rpk failed)"
        return
    fi

    local missing=()
    for topic in "${required_topics[@]}"; do
        # Match topic name at start of line (rpk tabular output)
        if ! echo "$result" | grep -qE "^${topic}[[:space:]]"; then
            missing+=("$topic")
        fi
    done

    if [[ ${#missing[@]} -eq 0 ]]; then
        log_check "$name" "green" "all required topics present (${#required_topics[@]})"
    else
        log_check "$name" "yellow" "missing topics: ${missing[*]}"
    fi
}

check_migration_parity() {
    local name="migration_parity"
    local docker_dir="${REPO_ROOT}/docker/migrations/forward"
    local src_dir="${REPO_ROOT}/src/omnibase_infra/migrations/forward"

    if [[ ! -d "$docker_dir" ]] && [[ ! -d "$src_dir" ]]; then
        log_check "$name" "skip" "Migration directory not set up"
        return
    fi
    if [[ ! -d "$docker_dir" ]]; then
        log_check "$name" "skip" "docker migrations dir not set up"
        return
    fi
    if [[ ! -d "$src_dir" ]]; then
        log_check "$name" "skip" "src migrations dir not set up"
        return
    fi

    local docker_count src_count
    docker_count=$(find "$docker_dir" -maxdepth 1 -type f \( -name '*.sql' -o -name '*.sh' \) | wc -l | tr -d ' ')
    src_count=$(find "$src_dir" -maxdepth 1 -type f \( -name '*.sql' -o -name '*.sh' \) | wc -l | tr -d ' ')

    if [[ "$FLAG_VERBOSE" == "true" ]]; then
        log_check "$name" "green" "docker=${docker_count} src=${src_count} migration files"
    else
        log_check "$name" "green" "docker=${docker_count} src=${src_count} migration files"
    fi
}

check_env_audit() {
    local name="env_audit"
    if [[ "$FLAG_CROSS_REPO" == "false" ]]; then
        log_check "$name" "green" "skipped (use --cross-repo to enable)"
        return
    fi

    local audit_script="${OMNI_HOME}/scripts/audit-env-files.sh"
    if [[ ! -f "$audit_script" ]]; then
        log_check "$name" "yellow" "audit script not found at ${audit_script}"
        return
    fi

    local result
    if result=$(bash "$audit_script" 2>&1); then
        log_check "$name" "green" "no rogue .env files found"
    else
        local count
        count=$(echo "$result" | grep -c 'COMMITTED\|UNTRACKED' || true)
        log_check "$name" "red" "${count} rogue .env file(s) found"
    fi
}

check_cloud_bus_refs() {
    local name="cloud_bus_refs"
    if [[ "$FLAG_CROSS_REPO" == "false" ]]; then
        log_check "$name" "green" "skipped (use --cross-repo to enable)"
        return
    fi

    local guard_script="${OMNI_HOME}/scripts/check_no_cloud_bus.sh"
    if [[ ! -f "$guard_script" ]]; then
        log_check "$name" "yellow" "cloud bus guard not found at ${guard_script}"
        return
    fi

    local result
    if result=$(bash "$guard_script" "${REPO_ROOT}" 2>&1); then
        log_check "$name" "green" "no unsuppressed 29092 references"  # cloud-bus-ok OMN-4922
    else
        local count
        count=$(echo "$result" | grep -c '^VIOLATION' || true)
        log_check "$name" "red" "${count} unsuppressed cloud bus reference(s)"
    fi
}

check_bus_endpoint() {
    local name="bus_endpoint"
    local bootstrap="${KAFKA_BOOTSTRAP_SERVERS:-}"

    if [[ -z "$bootstrap" ]]; then
        log_check "$name" "yellow" "KAFKA_BOOTSTRAP_SERVERS not set"
        return
    fi

    if echo "$bootstrap" | grep -q "29092"; then  # cloud-bus-ok OMN-4922
        log_check "$name" "red" "KAFKA_BOOTSTRAP_SERVERS contains 29092 (cloud bus): ${bootstrap}"  # cloud-bus-ok OMN-4922
    else
        log_check "$name" "green" "endpoint OK: ${bootstrap}"
    fi
}

check_infisical_folders() {
    local name="infisical_folders"

    # Skip entirely when INFISICAL_ADDR is not configured (secrets profile not active)
    local infisical_addr="${INFISICAL_ADDR:-}"
    if [[ -z "$infisical_addr" ]]; then
        log_check "$name" "skip" "INFISICAL_ADDR not set (secrets profile not active)"
        return
    fi

    # Probe reachability with a 5-second timeout (matches existing check timeouts)
    if ! curl -sf --max-time 5 "${infisical_addr}/api/status" >/dev/null 2>&1; then
        log_check "$name" "yellow" "Infisical not reachable at ${infisical_addr}"
        return
    fi

    # Obtain a short-lived access token via universal-auth login
    local client_id="${INFISICAL_CLIENT_ID:-}"
    local client_secret="${INFISICAL_CLIENT_SECRET:-}"
    local project_id="${INFISICAL_PROJECT_ID:-}"

    if [[ -z "$client_id" || -z "$client_secret" || -z "$project_id" ]]; then
        log_check "$name" "yellow" "INFISICAL_CLIENT_ID/SECRET/PROJECT_ID not set; cannot verify folders"
        return
    fi

    local token_response
    token_response=$(curl -sf --max-time 5 \
        -X POST "${infisical_addr}/api/v1/auth/universal-auth/login" \
        -H "Content-Type: application/json" \
        -d "{\"clientId\":\"${client_id}\",\"clientSecret\":\"${client_secret}\"}" 2>&1) || true

    local access_token
    access_token=$(printf '%s' "$token_response" | grep -o '"accessToken":"[^"]*"' | sed 's/"accessToken":"//;s/"//') || true

    if [[ -z "$access_token" ]]; then
        log_check "$name" "yellow" "could not obtain Infisical access token (auth failed or API changed)"
        return
    fi

    # Query /shared/ folders for the prod environment
    local folder_response
    folder_response=$(curl -sf --max-time 5 \
        -H "Authorization: Bearer ${access_token}" \
        "${infisical_addr}/api/v1/folders?workspaceId=${project_id}&environment=prod&path=%2Fshared%2F" 2>&1) || true

    # Extract folder names from the response (simple grep-based parse — no jq dependency)
    local existing_folders
    existing_folders=$(printf '%s' "$folder_response" | grep -o '"name":"[^"]*"' | sed 's/"name":"//;s/"//g') || true

    # Core folder: RED if missing while Infisical is running
    local core_folders=("db")

    # Advisory folders: YELLOW if missing (optional transport types)
    local advisory_folders=("kafka" "http" "filesystem" "graph" "mcp")

    local missing_core=()
    local missing_advisory=()

    for folder in "${core_folders[@]}"; do
        if ! printf '%s\n' "$existing_folders" | grep -qx "$folder"; then
            missing_core+=("$folder")
        fi
    done

    for folder in "${advisory_folders[@]}"; do
        if ! printf '%s\n' "$existing_folders" | grep -qx "$folder"; then
            missing_advisory+=("$folder")
        fi
    done

    if [[ ${#missing_core[@]} -gt 0 ]]; then
        log_check "$name" "red" "core /shared/ folder(s) missing: ${missing_core[*]} (re-run seed-infisical.py)"
        return
    fi

    if [[ ${#missing_advisory[@]} -gt 0 ]]; then
        log_check "$name" "yellow" "advisory /shared/ folder(s) missing: ${missing_advisory[*]} (run seed-infisical.py to populate)"
        return
    fi

    local total=$(( ${#core_folders[@]} + ${#advisory_folders[@]} ))
    log_check "$name" "green" "all /shared/<transport>/ folders present (${total}/${total})"
}

# =====================================================================
# Run all checks
# =====================================================================

if [[ "$FLAG_JSON" == "false" ]]; then
    echo ""
    echo "ONEX System Health Gate"
    echo "======================"
    echo ""
fi

if [[ "$FLAG_LANE" == "true" ]]; then
    # Lane-liveness subset (OMN-15190). Runs from the containerized deploy
    # runner, which has docker.sock + host-gateway network reachability but
    # none of the POSTGRES_/VALKEY_/INFISICAL_ credentials the full gate
    # needs — so the full gate cannot BE the scheduled surface, and a
    # credential-less full run would die on `${POSTGRES_HOST:?}` instead of
    # reporting on the lane.
    check_dev_lane_liveness
    check_redpanda
    check_runtime_containers
else
    check_postgres
    check_redpanda
    check_valkey
    check_infra_containers
    check_keycloak
    check_runtime_containers
    check_required_topics
    check_migration_parity
    check_env_audit
    check_cloud_bus_refs
    check_bus_endpoint
    check_infisical_folders
    check_dev_lane_liveness
fi

# =====================================================================
# Output
# =====================================================================

if [[ "$FLAG_JSON" == "true" ]]; then
    # Build JSON output
    checks_json=""
    for i in "${!CHECK_NAMES[@]}"; do
        escaped_name=$(json_escape "${CHECK_NAMES[$i]}")
        escaped_status=$(json_escape "${CHECK_STATUSES[$i]}")
        escaped_detail=$(json_escape "${CHECK_DETAILS[$i]}")
        entry="{\"name\":\"${escaped_name}\",\"status\":\"${escaped_status}\",\"detail\":\"${escaped_detail}\"}"
        if [[ -n "$checks_json" ]]; then
            checks_json="${checks_json},${entry}"
        else
            checks_json="${entry}"
        fi
    done

    cat <<EOF
{
  "overall": "${OVERALL_STATUS}",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "checks": [${checks_json}],
  "flags": {
    "cross_repo": ${FLAG_CROSS_REPO},
    "ci": ${FLAG_CI},
    "lane": ${FLAG_LANE},
    "verbose": ${FLAG_VERBOSE},
    "lane_keepalive": "${LANE_KEEPALIVE}"
  }
}
EOF
else
    echo ""
    echo "----------------------"
    printf "  Overall: %s\n" "$OVERALL_STATUS"
    echo "----------------------"
    echo ""
fi

# Exit code: 0 for green/yellow, 1 for red
if [[ "$OVERALL_STATUS" == "red" ]]; then
    exit 1
fi
exit 0
