#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# refresh_dev_lane.sh -- OMN-14889: health-gated refresh of the .201 `dev`
# lab lane to a named ref, driven by a release-train tag push.
#
# Dev-lane analog of refresh_stability_lane.sh (OMN-14873). Mirrors its
# health-gate / receipt shape but is COLD-AWARE (OMN-13414): the dev lane is
# the one lane in the lane table explicitly marked ephemeral ("2 desired, not
# in last census") -- it is routinely GC/idle-reclaimed down to zero core
# containers between uses, unlike the always-warm stability-test lane. This
# script probes live container state instead of assuming a hardcoded
# container map or a permanently-warm lane:
#
#   - resolves each of the 4 core services' running container ID via
#     `docker compose ps -q` (NOT a hardcoded name map -- the dev lane's
#     compose file does not set an explicit container_name for
#     runtime-worker, so its name is compose-assigned; resolving live avoids
#     hardcoding a name that could silently drift),
#   - if ALL 4 core containers are already running: WARM path, identical
#     shape to refresh_stability_lane.sh (pre-image capture, preflight
#     rollback tag, ancestry assertion, scoped --restart, rollback-on-fail),
#   - if ANY core container is missing: COLD-AWARE path -- there is no
#     baseline image for at least one service, so digest-changed / ancestry /
#     image-retag-rollback are not meaningful. Uses deploy-runtime.sh's
#     --cold full bring-up if the baseline deps (postgres) are ALSO down,
#     else its scoped --restart (which creates-or-recreates via
#     `up -d --no-deps --force-recreate`, fine whether or not the target
#     already exists). No rollback is attempted in this branch -- a failure
#     is reported as FAILED (STOP AND REPORT), matching the "never mask a
#     failure as success" rule; there is nothing prior to roll back to.
#
# Intended to run ON the host where the lane's containers live (.201) from
# the canonical omnibase_infra clone -- same constraint as
# refresh_stability_lane.sh: NOT from a worktree, NOT over ssh wrapping.
#
# Usage:
#   refresh_dev_lane.sh [--ref <ref>] [--min-contracts <n>] [--execute]
#
# Exit codes:
#   0  plan printed (dry-run) or refresh SUCCEEDED (health-gate PASS)
#   1  refresh FAILED (warm path) and rollback restored a healthy lane
#      (FAILED_ROLLED_BACK)
#   2  refresh FAILED and could not be confirmed healthy (warm-rollback also
#      unhealthy, OR cold-aware path failed with nothing to roll back to) --
#      STOP AND REPORT, do not retry-until-green
#   64 usage / precondition error

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOY_RUNTIME="${DEPLOY_RUNTIME:-${REPO_ROOT}/scripts/deploy-runtime.sh}"
VERIFY_SCRIPT="${SCRIPT_DIR}/verify_dev_refresh.py"

# Same runtime-policy + operator-env sourcing as refresh_stability_lane.sh --
# this script also issues direct `docker compose` calls (container
# resolution, rollback recreate) that need the same vars in scope.
_OPERATOR_OMNI_HOME="${OMNI_HOME:-}"
if [[ -f "${REPO_ROOT}/docker/runtime-policy.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/docker/runtime-policy.env"
    set +a
fi
if [[ -f "${HOME}/.omnibase/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "${HOME}/.omnibase/.env"
    set +a
fi
if [[ -n "${_OPERATOR_OMNI_HOME}" ]]; then
    export OMNI_HOME="${_OPERATOR_OMNI_HOME}"
fi

# --- hardcoded lane identity (no --lane flag; dev ONLY) ---------------------
readonly LANE="dev"
readonly COMPOSE_PROJECT="omnibase-infra"
readonly REDPANDA_CONTAINER="omnibase-infra-redpanda"
readonly POSTGRES_CONTAINER="omnibase-infra-postgres"
readonly CORE_SERVICES=(omninode-runtime runtime-effects runtime-worker projection-api)
readonly ALL_TRACKED_REPOS=(omnibase_infra omnibase_core omnibase_compat onex_change_control omnimarket)

REF="origin/dev"
MIN_CONTRACTS=288
MANIFEST_URL="http://localhost:8085/v1/introspection/manifest"
HEALTH_URL="http://localhost:8085/health"
MODE="plan"
TRIGGERING_TAG=""

usage() {
    sed -n '4,42p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

log() { printf '[refresh-dev-lane] %s\n' "$*" >&2; }
err() { printf '[refresh-dev-lane] ERROR: %s\n' "$*" >&2; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ref)
            [[ -n "${2:-}" ]] || { err "--ref requires a value"; exit 64; }
            REF="$2"; shift 2 ;;
        --min-contracts)
            [[ -n "${2:-}" ]] || { err "--min-contracts requires a value"; exit 64; }
            MIN_CONTRACTS="$2"; shift 2 ;;
        --manifest-url)
            [[ -n "${2:-}" ]] || { err "--manifest-url requires a value"; exit 64; }
            MANIFEST_URL="$2"; shift 2 ;;
        --health-url)
            [[ -n "${2:-}" ]] || { err "--health-url requires a value"; exit 64; }
            HEALTH_URL="$2"; shift 2 ;;
        --triggering-tag)
            [[ -n "${2:-}" ]] || { err "--triggering-tag requires a value"; exit 64; }
            TRIGGERING_TAG="$2"; shift 2 ;;
        --execute)
            MODE="execute"; shift ;;
        --help|-h)
            usage 0 ;;
        *)
            err "unknown option: $1"; usage 64 ;;
    esac
done

OMNI_HOME="${OMNI_HOME:-}"
if [[ -z "${OMNI_HOME}" ]]; then
    err "OMNI_HOME must be set (the ambient clones under it are the deploy source)."
    exit 64
fi

if [[ -n "${OMNIBASE_INFRA_COMPOSE_PROJECT:-}" && "${OMNIBASE_INFRA_COMPOSE_PROJECT}" != "${COMPOSE_PROJECT}" ]]; then
    err "OMNIBASE_INFRA_COMPOSE_PROJECT='${OMNIBASE_INFRA_COMPOSE_PROJECT}' is set but this script"
    err "  ONLY drives '${COMPOSE_PROJECT}' (dev). Unset it before calling this script."
    exit 64
fi

for cmd in docker git curl jq; do
    command -v "${cmd}" >/dev/null 2>&1 || { err "'${cmd}' is required but not found in PATH."; exit 64; }
done
if [[ ! -x "${DEPLOY_RUNTIME}" && ! -f "${DEPLOY_RUNTIME}" ]]; then
    err "deploy-runtime.sh not found at ${DEPLOY_RUNTIME}"; exit 64
fi
if [[ ! -f "${VERIFY_SCRIPT}" ]]; then
    err "verify_dev_refresh.py not found at ${VERIFY_SCRIPT}"; exit 64
fi

PYTHON_BIN=""
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
elif command -v uv &>/dev/null; then
    PYTHON_BIN="uv-run"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
else
    err "No Python interpreter available to run the health-gate."; exit 64
fi

run_verify() {
    if [[ "${PYTHON_BIN}" == "uv-run" ]]; then
        uv run --project "${REPO_ROOT}" python "${VERIFY_SCRIPT}" "$@"
    else
        "${PYTHON_BIN}" "${VERIFY_SCRIPT}" "$@"
    fi
}

INFRA_CLONE="${OMNI_HOME}/omnibase_infra"

# --- ambient-clone git wrapper ----------------------------------------------
# The ambient clones under OMNI_HOME are owned by the uid that provisioned them
# (deploy-agent), but this script can run as a DIFFERENT uid -- e.g. inside the
# self-hosted runner container. git then refuses every operation on them with
# "fatal: detected dubious ownership in repository at '<clone>'" and exits 128.
# Run 29799976126 died exactly there, on ${OMNI_HOME}/omnibase_infra, before a
# single fetch happened.
#
# That was relieved out-of-band on the host with an ad hoc `git config --global
# --add safe.directory ...`, but that relief is NOT DURABLE: the runner's
# /home/runner is not a named volume, so the global gitconfig -- and with it the
# exemption -- is lost on any `--force-recreate` of the runner container, and the
# next tag push fails the same way. Carry the exemption in-repo instead, scoped
# per invocation: `-c safe.directory=<clone>` is protected configuration git
# honours from the command line, applies to exactly one git process and one
# path, requires no host change, writes nothing to the user's global gitconfig,
# and is never the blanket `safe.directory=*`.
git_clone() {
    local clone="$1"
    shift
    git -c "safe.directory=${clone}" -C "${clone}" "$@"
}

compose_ps_q() {
    # Resolve a service's running container ID (empty string if not running).
    # Never hardcode a container name for this lane -- see file header.
    docker compose -p "${COMPOSE_PROJECT}" \
        -f "${INFRA_CLONE}/docker/docker-compose.infra.yml" \
        --profile runtime \
        ps -q "$1" 2>/dev/null || true
}

STATE_DIR="${HOME}/.omnibase/state/dev_lane_refresh"
HISTORY_DIR="${STATE_DIR}/history"
mkdir -p "${HISTORY_DIR}"
UTC_NOW="$(date -u +%Y%m%dT%H%M%SZ)"
WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/refresh-dev-lane.XXXXXX")"
trap 'rm -rf "${WORKDIR}"' EXIT

log "lane            : ${LANE} (compose project ${COMPOSE_PROJECT})"
log "ref             : ${REF}"
log "min contracts   : ${MIN_CONTRACTS}"
log "core services   : ${CORE_SERVICES[*]}"
log "mode            : ${MODE}"

# --- probe live state to decide WARM vs COLD-AWARE --------------------------
declare -A PRE_CONTAINER_IDS
CORE_ALL_RUNNING=true
for svc in "${CORE_SERVICES[@]}"; do
    cid="$(compose_ps_q "${svc}")"
    PRE_CONTAINER_IDS["${svc}"]="${cid}"
    if [[ -z "${cid}" ]]; then
        CORE_ALL_RUNNING=false
        log "  ${svc}: NOT running"
    else
        log "  ${svc}: running (${cid:0:12})"
    fi
done

DEPS_RUNNING=true
if [[ -z "$(docker inspect "${POSTGRES_CONTAINER}" --format '{{.Id}}' 2>/dev/null || true)" ]]; then
    DEPS_RUNNING=false
fi
log "baseline deps (${POSTGRES_CONTAINER}) running: ${DEPS_RUNNING}"

if [[ "${CORE_ALL_RUNNING}" == true ]]; then
    BRANCH="warm"
elif [[ "${DEPS_RUNNING}" == true ]]; then
    BRANCH="cold-aware-restart"   # deps up, core down: scoped restart still applies
else
    BRANCH="cold-aware-full"      # whole lane reclaimed (OMN-13414): full --cold bring-up
fi
log "branch selected : ${BRANCH}"

if [[ "${MODE}" != "execute" ]]; then
    log "dry-run: no fetch/checkout/build/restart performed. Re-run with --execute."
    case "${BRANCH}" in
        warm)
            log "would run (warm): OMNI_HOME=${OMNI_HOME} OMNIBASE_INFRA_COMPOSE_PROJECT=${COMPOSE_PROJECT} \\"
            log "  BUILD_SOURCE=workspace DEPLOY_REF=${REF} RUNTIME_BUILD_SERVICES_OVERRIDE=\"${CORE_SERVICES[*]}\" \\"
            log "  ${DEPLOY_RUNTIME} --execute --force --restart" ;;
        cold-aware-restart)
            log "would run (cold-aware, deps up): same as warm but no pre-image/ancestry/rollback capture" ;;
        cold-aware-full)
            log "would run (cold-aware, whole lane down): OMNI_HOME=${OMNI_HOME} OMNIBASE_INFRA_COMPOSE_PROJECT=${COMPOSE_PROJECT} \\"
            log "  DEPLOY_REF=${REF} ${DEPLOY_RUNTIME} --execute --force --cold" ;;
    esac
    exit 0
fi

# OMN-14900: ensure the private runner clones exist before any git operation.
# Idempotent: clones any of the 5 repos missing under OMNI_HOME (the deploy
# runner's PRIVATE bind mount) and asserts each is operable by this euid.
OMNI_HOME="${OMNI_HOME}" bash "${SCRIPT_DIR}/ensure_runner_clones.sh"

declare -A PRE_IMAGE_IDS
OLD_INFRA_REVISION=""
declare -A PRIOR_REFS
ANCESTRY_OK=true
ANCESTRY_COMMANDS=()
ROLLBACK_TRIGGERED=false
GATE2_JSON=""

if [[ "${BRANCH}" == "warm" ]]; then
    # ============================================================
    # 1. Capture PRE-STATE (identical shape to refresh_stability_lane.sh)
    # ============================================================
    log "=== Capture pre-state ==="
    for svc in "${CORE_SERVICES[@]}"; do
        image_id="$(docker inspect "${PRE_CONTAINER_IDS[${svc}]}" --format '{{.Image}}' 2>/dev/null || true)"
        if [[ -z "${image_id}" ]]; then
            err "Could not resolve running image ID for ${svc}. Is the lane up?"
            exit 64
        fi
        PRE_IMAGE_IDS["${svc}"]="${image_id}"
        log "  ${svc}: pre_image_id=${image_id}"
    done
    OLD_INFRA_REVISION="$(docker inspect "${PRE_CONTAINER_IDS[omninode-runtime]}" \
        --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' 2>/dev/null || true)"
    log "  current org.opencontainers.image.revision=${OLD_INFRA_REVISION:-<none>}"

    for repo in "${ALL_TRACKED_REPOS[@]}"; do
        clone="${OMNI_HOME}/${repo}"
        [[ -d "${clone}/.git" ]] || { err "Expected clone not found: ${clone}"; exit 64; }
        PRIOR_REFS["${repo}"]="$(git_clone "${clone}" rev-parse HEAD)"
        log "  ${repo} prior HEAD: ${PRIOR_REFS[${repo}]:0:12}"
    done

    log "=== Tag preflight rollback anchor (${UTC_NOW}) ==="
    for svc in "${CORE_SERVICES[@]}"; do
        image_tag="${COMPOSE_PROJECT}-${svc}"
        docker tag "${image_tag}:latest" "${image_tag}:preflight-${UTC_NOW}"
        log "  tagged ${image_tag}:latest -> ${image_tag}:preflight-${UTC_NOW}"
    done
else
    log "=== Cold-aware branch: skipping pre-image capture / preflight rollback tag ==="
    log "  (at least one core container is not running -- no baseline image to snapshot)"
    for repo in "${ALL_TRACKED_REPOS[@]}"; do
        clone="${OMNI_HOME}/${repo}"
        [[ -d "${clone}/.git" ]] || { err "Expected clone not found: ${clone}"; exit 64; }
        PRIOR_REFS["${repo}"]="$(git_clone "${clone}" rev-parse HEAD)"
    done
fi

# Refresh the omnibase_infra ambient clone itself to --ref.
# (deploy-runtime.sh reads its own git_sha from THIS clone's HEAD; DEPLOY_REF
# below only pins the SIBLING repos staged by stage_workspace.sh -- it does not
# touch omnibase_infra's own tree.)
#
# This clone's local `dev` branch may already be checked out in a SIBLING
# worktree on the same host (deploy-agent's
# runtime-sync-worktrees/OMN-12618/omni_home/omnibase_infra) -- git refuses to
# check that branch out a second time ("fatal: 'dev' is already checked out at
# <path>"), hard-failing this whole script on an environment collision that has
# nothing to do with lane health. Runs 29800684577 and 29800954943 died exactly
# there. refresh_stability_lane.sh already retired the checkout+`reset --hard`
# pair for this reason (OMN-12618); the dev lane mirrors it rather than
# reintroducing the retired pattern. Resolve --ref to a commit SHA first and
# check it out DETACHED: a detached HEAD carries no branch identity, so it can
# never collide with another worktree regardless of what branch that worktree
# holds. --force discards any local modifications the same way the previous
# checkout+`reset --hard` pair did.
log "=== Refresh omnibase_infra ambient clone to ${REF} ==="
git_clone "${INFRA_CLONE}" fetch origin --prune
RESOLVED_REF_SHA="$(git_clone "${INFRA_CLONE}" rev-parse "${REF}^{commit}")"
git_clone "${INFRA_CLONE}" checkout --force --detach "${RESOLVED_REF_SHA}"
NEW_INFRA_SHA="$(git_clone "${INFRA_CLONE}" rev-parse HEAD)"
NEW_INFRA_SHA_SHORT="$(git_clone "${INFRA_CLONE}" rev-parse --short=12 HEAD)"
log "  omnibase_infra now at ${NEW_INFRA_SHA_SHORT} (full: ${NEW_INFRA_SHA})"

log "=== Refresh tracked sibling ambient clones ==="
for repo in "${ALL_TRACKED_REPOS[@]}"; do
    [[ "${repo}" == "omnibase_infra" ]] && continue
    clone="${OMNI_HOME}/${repo}"
    [[ -d "${clone}/.git" ]] || { err "Expected clone not found: ${clone}"; exit 64; }
    git_clone "${clone}" fetch origin --prune
    sibling_ref="${REF}"
    if ! sibling_sha="$(git_clone "${clone}" rev-parse "${sibling_ref}^{commit}" 2>/dev/null)"; then
        sibling_ref="origin/dev"
        sibling_sha="$(git_clone "${clone}" rev-parse "${sibling_ref}^{commit}")"
    fi
    git_clone "${clone}" checkout --force --detach "${sibling_sha}"
    log "  ${repo} now at $(git_clone "${clone}" rev-parse --short=12 HEAD) via ${sibling_ref}"
done

log "=== Build + bring-up (branch: ${BRANCH}) ==="
DEPLOY_EXIT=0
if [[ "${BRANCH}" == "cold-aware-full" ]]; then
    (
        export OMNI_HOME
        # shellcheck disable=SC2030,SC2031 # deliberately subshell-local; only
        # this bash "${DEPLOY_RUNTIME}" invocation ever reads these two exports.
        export OMNIBASE_INFRA_COMPOSE_PROJECT="${COMPOSE_PROJECT}"
        # shellcheck disable=SC2030,SC2031
        export DEPLOY_REF="${REF}"
        bash "${DEPLOY_RUNTIME}" --execute --force --cold
    ) || DEPLOY_EXIT=$?
else
    (
        export OMNI_HOME
        # shellcheck disable=SC2030,SC2031 # deliberately subshell-local; only
        # this bash "${DEPLOY_RUNTIME}" invocation ever reads these exports.
        export OMNIBASE_INFRA_COMPOSE_PROJECT="${COMPOSE_PROJECT}"
        export BUILD_SOURCE="workspace"
        # shellcheck disable=SC2030,SC2031
        export DEPLOY_REF="${REF}"
        export RUNTIME_BUILD_SERVICES_OVERRIDE="${CORE_SERVICES[*]}"
        bash "${DEPLOY_RUNTIME}" --execute --force --restart
    ) || DEPLOY_EXIT=$?
fi
if [[ "${DEPLOY_EXIT}" -ne 0 ]]; then
    log "deploy-runtime.sh exited ${DEPLOY_EXIT} -- proceeding to health-gate anyway"
fi

declare -A NEW_REFS
if [[ "${BRANCH}" == "warm" ]]; then
    for repo in "${ALL_TRACKED_REPOS[@]}"; do
        clone="${OMNI_HOME}/${repo}"
        NEW_REFS["${repo}"]="$(git_clone "${clone}" rev-parse HEAD)"
        cmd="git -c safe.directory=${clone} -C ${clone} merge-base --is-ancestor ${PRIOR_REFS[${repo}]} ${NEW_REFS[${repo}]}"
        ANCESTRY_COMMANDS+=("${cmd}")
        if git_clone "${clone}" merge-base --is-ancestor "${PRIOR_REFS[${repo}]}" "${NEW_REFS[${repo}]}"; then
            log "  ${repo}: ${PRIOR_REFS[${repo}]:0:12} -> ${NEW_REFS[${repo}]:0:12} (forward progress OK)"
        else
            err "  ${repo}: ${PRIOR_REFS[${repo}]:0:12} -> ${NEW_REFS[${repo}]:0:12} is NOT forward progress!"
            ANCESTRY_OK=false
        fi
    done
    if [[ "${ANCESTRY_OK}" != true ]]; then
        err "Forward-progress assertion FAILED for at least one repo. Refusing to certify success."
    fi
else
    for repo in "${ALL_TRACKED_REPOS[@]}"; do
        NEW_REFS["${repo}"]="$(git_clone "${OMNI_HOME}/${repo}" rev-parse HEAD)"
    done
    log "cold-aware branch: no prior running state to compare -- ancestry check N/A"
fi

log "=== Health-gate ==="
PRE_IMAGE_IDS_JSON="{}"
REQUIRE_DIGEST_CHANGE="--no-require-digest-change"
if [[ "${BRANCH}" == "warm" ]]; then
    PRE_IMAGE_IDS_JSON="$(
        printf '%s\n' "${!PRE_IMAGE_IDS[@]}" | while read -r k; do
            printf '%s\t%s\n' "${k}" "${PRE_IMAGE_IDS[${k}]}"
        done | jq -Rn '[inputs | split("\t") | {(.[0]): .[1]}] | add'
    )"
    REQUIRE_DIGEST_CHANGE=""
fi

declare -A NEW_CONTAINER_IDS
for svc in "${CORE_SERVICES[@]}"; do
    NEW_CONTAINER_IDS["${svc}"]="$(compose_ps_q "${svc}")"
done
CONTAINER_IDS_JSON="$(
    printf '%s\n' "${!NEW_CONTAINER_IDS[@]}" | while read -r k; do
        printf '%s\t%s\n' "${k}" "${NEW_CONTAINER_IDS[${k}]}"
    done | jq -Rn '[inputs | split("\t") | {(.[0]): .[1]}] | add'
)"

GATE1_JSON="${WORKDIR}/gate1.json"
GATE_EXIT=0
# shellcheck disable=SC2086
run_verify \
    --lane "${LANE}" \
    --expected-revision "${NEW_INFRA_SHA_SHORT}" \
    --pre-image-ids "${PRE_IMAGE_IDS_JSON}" \
    --container-ids "${CONTAINER_IDS_JSON}" \
    --manifest-url "${MANIFEST_URL}" \
    --health-url "${HEALTH_URL}" \
    --broker-container "${REDPANDA_CONTAINER}" \
    --min-contracts "${MIN_CONTRACTS}" \
    ${REQUIRE_DIGEST_CHANGE} \
    --json > "${GATE1_JSON}" || GATE_EXIT=$?

if ! jq -e . "${GATE1_JSON}" >/dev/null 2>&1; then
    err "health-gate did not produce valid JSON (exit ${GATE_EXIT}) -- writing INFRA_ERROR placeholder"
    jq -n --arg lane "${LANE}" '{lane: $lane, overall: "INFRA_ERROR", errors: ["health-gate produced no valid JSON output"]}' > "${GATE1_JSON}"
fi

GATE1_OVERALL="$(jq -r '.overall // "INFRA_ERROR"' "${GATE1_JSON}" 2>/dev/null || echo "INFRA_ERROR")"
log "health-gate result: ${GATE1_OVERALL} (exit ${GATE_EXIT})"
cat "${GATE1_JSON}" >&2

RESULT="FAILED"
if [[ "${GATE1_OVERALL}" == "PASS" && "${ANCESTRY_OK}" == true ]]; then
    RESULT="SUCCESS"
    log "=== SUCCESS: health-gate PASS ==="
    if [[ "${BRANCH}" == "warm" ]]; then
        for svc in "${CORE_SERVICES[@]}"; do
            image_tag="${COMPOSE_PROJECT}-${svc}"
            old_tags="$(docker images --format '{{.Tag}}' "${image_tag}" | grep '^preflight-' | sort -r | tail -n +4 || true)"
            for t in ${old_tags}; do
                docker rmi "${image_tag}:${t}" >/dev/null 2>&1 || true
            done
        done
    fi
elif [[ "${BRANCH}" == "warm" ]]; then
    log "=== FAILURE: triggering rollback ==="
    ROLLBACK_TRIGGERED=true
    for svc in "${CORE_SERVICES[@]}"; do
        image_tag="${COMPOSE_PROJECT}-${svc}"
        docker tag "${image_tag}:preflight-${UTC_NOW}" "${image_tag}:latest"
        log "  rolled back ${image_tag}:latest <- ${image_tag}:preflight-${UTC_NOW}"
    done
    ROLLBACK_RECREATE_EXIT=0
    docker compose -p "${COMPOSE_PROJECT}" \
        -f "${INFRA_CLONE}/docker/docker-compose.infra.yml" \
        --profile runtime \
        up -d --no-deps --no-build --force-recreate \
        "${CORE_SERVICES[@]}" || ROLLBACK_RECREATE_EXIT=$?
    if [[ "${ROLLBACK_RECREATE_EXIT}" -ne 0 ]]; then
        err "rollback targeted-recreate exited ${ROLLBACK_RECREATE_EXIT} -- proceeding to health-gate anyway"
    fi

    log "=== Re-verifying health after rollback ==="
    ROLLBACK_CONTAINER_IDS_JSON="$(
        for svc in "${CORE_SERVICES[@]}"; do
            printf '%s\t%s\n' "${svc}" "$(compose_ps_q "${svc}")"
        done | jq -Rn '[inputs | split("\t") | {(.[0]): .[1]}] | add'
    )"
    GATE2_JSON="${WORKDIR}/gate2.json"
    GATE2_EXIT=0
    run_verify \
        --lane "${LANE}" \
        --expected-revision "${OLD_INFRA_REVISION:-unknown}" \
        --pre-image-ids '{}' \
        --container-ids "${ROLLBACK_CONTAINER_IDS_JSON}" \
        --manifest-url "${MANIFEST_URL}" \
        --health-url "${HEALTH_URL}" \
        --broker-container "${REDPANDA_CONTAINER}" \
        --min-contracts "${MIN_CONTRACTS}" \
        --no-require-digest-change \
        --json > "${GATE2_JSON}" || GATE2_EXIT=$?

    if ! jq -e . "${GATE2_JSON}" >/dev/null 2>&1; then
        err "post-rollback health-gate did not produce valid JSON (exit ${GATE2_EXIT}) -- writing INFRA_ERROR placeholder"
        jq -n --arg lane "${LANE}" '{lane: $lane, overall: "INFRA_ERROR", errors: ["post-rollback health-gate produced no valid JSON output"]}' > "${GATE2_JSON}"
    fi
    GATE2_OVERALL="$(jq -r '.overall // "INFRA_ERROR"' "${GATE2_JSON}" 2>/dev/null || echo "INFRA_ERROR")"
    log "post-rollback health-gate result: ${GATE2_OVERALL} (exit ${GATE2_EXIT})"
    cat "${GATE2_JSON}" >&2

    if [[ "${GATE2_OVERALL}" == "PASS" ]]; then
        RESULT="FAILED_ROLLED_BACK"
        log "Rollback restored a healthy lane. Refresh FAILED but the lane is HEALTHY."
    else
        RESULT="FAILED"
        err "=============================================================="
        err "STOP AND REPORT: rollback did NOT restore a healthy lane."
        err "The dev lane may be UNHEALTHY right now. Do not retry automatically."
        err "=============================================================="
    fi
else
    RESULT="FAILED"
    err "=============================================================="
    err "STOP AND REPORT: cold-aware refresh FAILED health-gate and there is"
    err "no prior running state to roll back to. Manual intervention required."
    err "=============================================================="
fi

# --- emit receipt ------------------------------------------------------------
PRIOR_REFS_JSON="$(for r in "${ALL_TRACKED_REPOS[@]}"; do printf '%s\t%s\n' "${r}" "${PRIOR_REFS[${r}]}"; done | jq -Rn '[inputs | split("\t") | {(.[0]): .[1]}] | add')"
NEW_REFS_JSON="$(for r in "${ALL_TRACKED_REPOS[@]}"; do printf '%s\t%s\n' "${r}" "${NEW_REFS[${r}]:-${PRIOR_REFS[${r}]}}"; done | jq -Rn '[inputs | split("\t") | {(.[0]): .[1]}] | add')"
if [[ "${#ANCESTRY_COMMANDS[@]}" -gt 0 ]]; then
    ANCESTRY_COMMANDS_JSON="$(printf '%s\n' "${ANCESTRY_COMMANDS[@]}" | jq -Rn '[inputs]')"
else
    ANCESTRY_COMMANDS_JSON='[]'
fi
BUILD_SCOPE_JSON="$(printf '%s\n' "${CORE_SERVICES[@]}" | jq -Rn '[inputs]')"
ROLLBACK_GATE_JSON="null"
if [[ -n "${GATE2_JSON}" && -f "${GATE2_JSON}" ]]; then
    ROLLBACK_GATE_JSON="$(cat "${GATE2_JSON}")"
fi

RECEIPT_PATH="${HISTORY_DIR}/${UTC_NOW}-${NEW_INFRA_SHA_SHORT}.json"
jq -n \
    --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg lane "${LANE}" \
    --arg branch "${BRANCH}" \
    --arg triggering_tag "${TRIGGERING_TAG:-}" \
    --argjson prior_refs "${PRIOR_REFS_JSON}" \
    --argjson new_refs "${NEW_REFS_JSON}" \
    --argjson ancestry_ok "${ANCESTRY_OK}" \
    --argjson ancestry_cmds "${ANCESTRY_COMMANDS_JSON}" \
    --argjson build_scope "${BUILD_SCOPE_JSON}" \
    --slurpfile health_gate "${GATE1_JSON}" \
    --argjson rollback_triggered "${ROLLBACK_TRIGGERED}" \
    --argjson rollback_gate "${ROLLBACK_GATE_JSON}" \
    --arg result "${RESULT}" \
    '{
        ts_utc: $ts,
        lane: $lane,
        triggering_tag: (if $triggering_tag == "" then null else $triggering_tag end),
        branch: $branch,
        prior_refs: $prior_refs,
        new_refs: $new_refs,
        ancestry_proof: {merge_base_is_ancestor: $ancestry_ok, commands: $ancestry_cmds},
        build_scope: $build_scope,
        health_gate: $health_gate[0],
        rollback: {triggered: $rollback_triggered, gate: $rollback_gate},
        result: $result
    }' > "${RECEIPT_PATH}"

cp "${RECEIPT_PATH}" "${STATE_DIR}/latest.json"
log "=== Receipt written: ${RECEIPT_PATH} (and ${STATE_DIR}/latest.json) ==="
log "result: ${RESULT}"

case "${RESULT}" in
    SUCCESS) exit 0 ;;
    FAILED_ROLLED_BACK) exit 1 ;;
    *) exit 2 ;;
esac
