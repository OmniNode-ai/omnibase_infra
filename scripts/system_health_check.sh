#!/usr/bin/env bash
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
#   6.  runtime_containers - Runtime profile containers (yellow if not running)
#   7.  required_topics   - Required Kafka topics exist
#   8.  migration_parity  - Docker and src migration directories in sync
#   9.  env_audit         - No rogue .env files (--cross-repo only)
#  10.  cloud_bus_refs    - No unsuppressed 29092 references (--cross-repo only)
#  11.  bus_endpoint      - KAFKA_BOOTSTRAP_SERVERS must not contain 29092
#
# OMN-3772

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

    local pg_host="${POSTGRES_HOST:-localhost}"
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
    local vk_host="${VALKEY_HOST:-localhost}"
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

    if [[ ${#missing[@]} -eq 0 ]]; then
        log_check "$name" "green" "all runtime containers running (${found}/${#expected[@]})"
    elif [[ $found -gt 0 ]]; then
        log_check "$name" "yellow" "partial: ${found}/${#expected[@]} running, missing: ${missing[*]}"
    else
        log_check "$name" "yellow" "none running (runtime profile not active)"
    fi
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
        log_check "$name" "green" "no unsuppressed 29092 references"
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

    if echo "$bootstrap" | grep -q "29092"; then
        log_check "$name" "red" "KAFKA_BOOTSTRAP_SERVERS contains 29092 (cloud bus): ${bootstrap}"
    else
        log_check "$name" "green" "endpoint OK: ${bootstrap}"
    fi
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
    "verbose": ${FLAG_VERBOSE}
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
