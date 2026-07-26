#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# refresh_stability_lane.sh -- OMN-14263/OMN-14873: health-gated, rollback-on-
# failure refresh of the .201 `.201 stability-test` proof lane to a named ref.
#
# The .201 stability-test lane (compose project omnibase-infra-stability-test,
# ports 18085/18086) is the "preferred proof lane for synthetic integration
# evidence" (CLAUDE.md), but has no refresh cadence -- it silently runs
# whatever code happened to be there the last time someone manually rebuilt
# it (OMN-14263). This script wraps the PROVEN manual recipe (workspace-mode
# build scoped to the 4 known-good core services + targeted recreate) into a
# single command with:
#
#   - pre-state capture (image IDs + a preflight rollback tag) BEFORE any
#     mutation,
#   - a forward-progress ancestry assertion (refuses to "refresh" backwards),
#   - a build SCOPED to the 4 core services only (omninode-runtime,
#     runtime-effects, runtime-worker, projection-api) via the new
#     RUNTIME_BUILD_SERVICES_OVERRIDE knob in deploy-runtime.sh -- this routes
#     around the still-open BUILD_SOURCE selector-mismatch defect on the 4
#     release-only services (agent-actions-consumer, skill-lifecycle-consumer,
#     intelligence-api, omninode-contract-resolver; OMN-14262 residual) as a
#     CONTROLLED decision, not a side effect of a partial build failure,
#   - a health-gate (verify_stability_refresh.py): digest changed, manifest
#     contract-count floor, /health, rpk cluster health, declared consumer
#     groups Stable, and image-revision readback,
#   - automatic rollback-and-re-verify on ANY health-gate failure, and
#   - a durable JSON receipt (~/.omnibase/state/stability_lane_refresh/) so a
#     future session can trust freshness from one cheap file read instead of
#     re-deriving the whole forensic chain by hand.
#
# This script is hardcoded to the stability-test lane ONLY. It does not accept
# a --lane flag and refuses to run against any other compose project (mirrors
# cut-lab-ref.sh's lane refusal for prod/judge) -- the stability-test lane is
# the only lane this session is authorized to mutate.
#
# Intended to run ON the host where the lane's containers live (.201) from
# the canonical omnibase_infra clone, the same way deploy-runtime.sh and
# cut-lab-ref.sh already do -- NOT from a worktree, NOT over ssh wrapping.
#
# Usage:
#   refresh_stability_lane.sh [--ref <ref>] [--min-contracts <n>] [--execute]
#
# Exit codes:
#   0  plan printed (dry-run) or refresh SUCCEEDED (health-gate PASS)
#   1  refresh FAILED and rollback restored a healthy lane (FAILED_ROLLED_BACK)
#   2  refresh FAILED and rollback ALSO could not restore health -- STOP AND
#      REPORT, do not retry-until-green
#   64 usage / precondition error

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOY_RUNTIME="${DEPLOY_RUNTIME:-${REPO_ROOT}/scripts/deploy-runtime.sh}"
VERIFY_SCRIPT="${SCRIPT_DIR}/verify_stability_refresh.py"
CONSUMER_GROUPS_FILE="${SCRIPT_DIR}/consumer_groups_stability.yaml"

log() { printf '[refresh-stability-lane] %s\n' "$*" >&2; }
err() { printf '[refresh-stability-lane] ERROR: %s\n' "$*" >&2; }
# OMN-14984: fail-fast-by-name helper for values this script expects the
# already-sourced docker/runtime-policy.env (contracts/services/
# runtime_policy.contract.yaml) to have provided, instead of re-declaring a
# hardcoded default that silently drifts from the contract.
require_contract_var() {
    local var_name="$1"
    if [[ -z "${!var_name:-}" ]]; then
        err "${var_name} is not set -- expected from docker/runtime-policy.env"
        err "  (generated from contracts/services/runtime_policy.contract.yaml)."
        err "  Verify docker/runtime-policy.env exists and was sourced above, and"
        err "  that the contract still renders ${var_name} for the stability_test lane."
        exit 64
    fi
}

# Source the same contract-rendered runtime policy + operator env that
# deploy-runtime.sh sources at its own top (docker/runtime-policy.env, then
# ~/.omnibase/.env). deploy-runtime.sh's OWN docker compose calls already do
# this internally, but this script ALSO issues a direct `docker compose`
# invocation for the rollback targeted-recreate (step 6 below) -- without
# these vars in scope, that compose invocation fails at config-interpolation
# time (e.g. BIFROST_VERIFY_ENDPOINTS) before it can even attempt the
# recreate. Preserve any operator-set OMNI_HOME/HEALTH_CHECK_URL exactly the
# way deploy-runtime.sh's own header does.
_OPERATOR_OMNI_HOME="${OMNI_HOME:-}"
# OMN-14958: operator env path is parameterized (same knob as deploy-runtime.sh)
# so the containerized deploy runner can point at its provisioned read-only
# mount instead of a ${HOME} that carries no operator env.
OMNIBASE_OPERATOR_ENV_FILE="${OMNIBASE_OPERATOR_ENV_FILE:-${HOME}/.omnibase/.env}"
if [[ -f "${REPO_ROOT}/docker/runtime-policy.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/docker/runtime-policy.env"
    # OMN-14983: a bare `-f` (exists) check silently treats an EXISTING but
    # UNREADABLE file the same as an absent one -- the operator env file is
    # then just skipped with no error, and the real failure surfaces later
    # and misleadingly (e.g. a missing POSTGRES_PASSWORD during compose
    # interpolation) instead of here, at the actual root cause. Distinguish
    # "missing" (still optional -- unchanged behavior) from "present but
    # unreadable" (always a fail-fast error, never silent).
    if [[ -e "${OMNIBASE_OPERATOR_ENV_FILE}" ]]; then
        if [[ ! -r "${OMNIBASE_OPERATOR_ENV_FILE}" ]]; then
            {
                echo "[refresh-stability-lane] ERROR: OPERATOR_ENV_UNREADABLE -- operator env file exists but this process cannot read it:"
                echo "  ${OMNIBASE_OPERATOR_ENV_FILE}"
                echo "  effective uid=$(id -u) ($(id -un 2>/dev/null || echo unknown))"
                echo "  Check the file's ownership/permissions. A silent skip here would let the"
                echo "  refresh proceed without the operator env (POSTGRES_PASSWORD etc.) and fail"
                echo "  later at a less obvious point (OMN-14983)."
            } >&2
            exit 64
        fi
        # shellcheck source=/dev/null
        source "${OMNIBASE_OPERATOR_ENV_FILE}"
    fi
    set +a
fi
if [[ -n "${_OPERATOR_OMNI_HOME}" ]]; then
    export OMNI_HOME="${_OPERATOR_OMNI_HOME}"
fi

# --- hardcoded lane identity (no --lane flag; stability-test ONLY) ---------
# OMN-14984: COMPOSE_PROJECT duplicated the contract-rendered
# STABILITY_TEST_COMPOSE_PROJECT (docker/runtime-policy.env, generated from
# contracts/services/runtime_policy.contract.yaml) as a second, independently
# hand-maintained bash literal -- re-render the contract and this script goes
# stale silently. Read it from the file this script already sources above
# instead; fail fast BY NAME (rule 8) if the sourced file didn't provide it,
# rather than falling back to a hardcoded default.
require_contract_var STABILITY_TEST_COMPOSE_PROJECT
readonly LANE="stability-test"
readonly COMPOSE_PROJECT="${STABILITY_TEST_COMPOSE_PROJECT}"
# REDPANDA_CONTAINER / CORE_SERVICES / CORE_CONTAINERS / MIN_CONTRACTS below
# are NOT currently rendered into docker/runtime-policy.env -- verified
# against contracts/services/runtime_policy.contract.yaml (OMN-14984 recon):
# only compose_project/main_port/effects_port are declared per lane there.
# These stay hardcoded here (no contract-rendered source to read from without
# fabricating one); a future lane-descriptor YAML analogous to
# consumer_groups_stability.yaml, or a contract schema extension, is the
# correct home for them, out of scope for this config-duplication kill.
readonly REDPANDA_CONTAINER="omnibase-infra-stability-test-redpanda"
readonly CORE_SERVICES=(omninode-runtime runtime-effects runtime-worker projection-api)
# service -> lane-scoped container_name (docker-compose.stability-test.yml).
# projection-api's container_name is prefixed "omnimarket-", not "omninode-" --
# resolve by explicit map, never assume a pattern (OMN-13826-class lesson).
declare -A CORE_CONTAINERS=(
    [omninode-runtime]="omninode-stability-test-runtime"
    [runtime-effects]="omninode-stability-test-runtime-effects"
    [runtime-worker]="omninode-stability-test-runtime-worker"
    [projection-api]="omnimarket-stability-test-projection-api"
)
# Siblings pinned to --ref via deploy-runtime.sh's DEPLOY_REF (RT-1, OMN-14438)
# during workspace staging, PLUS the omnibase_infra ambient clone itself (which
# is NOT part of that sibling set -- deploy-runtime.sh reads git_sha from
# wherever it is invoked FROM, i.e. this clone's own HEAD).
readonly ALL_TRACKED_REPOS=(omnibase_infra omnibase_core omnibase_compat onex_change_control omnimarket)

# --- defaults ---------------------------------------------------------------
# OMN-14958: the health-gate probe host is parameterized. `localhost` is only
# correct when this script runs ON the lane host itself; inside the
# containerized deploy runner (bridge network) localhost is the runner
# container, both probes die ECONNREFUSED, and a healthy lane gets reported
# as a false "lane unhealthy" STOP (deploy run 29977968728). The runner
# compose sets LANE_PROBE_HOST=host.docker.internal (host-gateway alias);
# --manifest-url/--health-url flags still override entirely.
# OMN-14984: the manifest/health default port duplicated the contract-rendered
# STABILITY_TEST_RUNTIME_MAIN_PORT as a hardcoded 18085 literal in these two
# URLs. Read it from the sourced runtime-policy.env instead; fail fast BY NAME
# if absent. --manifest-url/--health-url still fully override these defaults.
require_contract_var STABILITY_TEST_RUNTIME_MAIN_PORT
LANE_PROBE_HOST="${LANE_PROBE_HOST:-localhost}" # fallback-ok: localhost IS the lane host in the documented primary context (script runs ON .201); containerized runner overrides via compose env (OMN-14958)
REF="origin/dev"
MIN_CONTRACTS=288
MANIFEST_URL="http://${LANE_PROBE_HOST}:${STABILITY_TEST_RUNTIME_MAIN_PORT}/v1/introspection/manifest"
HEALTH_URL="http://${LANE_PROBE_HOST}:${STABILITY_TEST_RUNTIME_MAIN_PORT}/health"
MODE="plan"

usage() {
    sed -n '4,52p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

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
        --execute)
            MODE="execute"; shift ;;
        --help|-h)
            usage 0 ;;
        *)
            err "unknown option: $1"; usage 64 ;;
    esac
done

# --- preconditions -----------------------------------------------------------
OMNI_HOME="${OMNI_HOME:-}"
if [[ -z "${OMNI_HOME}" ]]; then
    err "OMNI_HOME must be set (the ambient clones under it are the deploy source)."
    exit 64
fi

# Safety: refuse to run if the operator env has repointed the compose project
# at a non-stability-test target (e.g. a sourced prod/judge env file). This
# script has no --lane flag by design; COMPOSE_PROJECT is hardcoded above, but
# deploy-runtime.sh resolves ITS OWN compose project from
# OMNIBASE_INFRA_COMPOSE_PROJECT at call time -- this script always exports the
# correct value explicitly below, so this guard is a defense against a caller
# who exported a conflicting value expecting it to leak through.
if [[ -n "${OMNIBASE_INFRA_COMPOSE_PROJECT:-}" && "${OMNIBASE_INFRA_COMPOSE_PROJECT}" != "${COMPOSE_PROJECT}" ]]; then
    err "OMNIBASE_INFRA_COMPOSE_PROJECT='${OMNIBASE_INFRA_COMPOSE_PROJECT}' is set in the environment but"
    err "  this script ONLY drives '${COMPOSE_PROJECT}' (stability-test). Refusing to run against a"
    err "  possibly-different target -- unset OMNIBASE_INFRA_COMPOSE_PROJECT before calling this script."
    exit 64
fi

for cmd in docker git curl jq; do
    command -v "${cmd}" >/dev/null 2>&1 || { err "'${cmd}' is required but not found in PATH."; exit 64; }
done

if [[ ! -x "${DEPLOY_RUNTIME}" && ! -f "${DEPLOY_RUNTIME}" ]]; then
    err "deploy-runtime.sh not found at ${DEPLOY_RUNTIME}"
    exit 64
fi
if [[ ! -f "${VERIFY_SCRIPT}" ]]; then
    err "verify_stability_refresh.py not found at ${VERIFY_SCRIPT}"
    exit 64
fi

PYTHON_BIN=""
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
elif command -v uv &>/dev/null; then
    PYTHON_BIN="uv-run"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
else
    err "No Python interpreter available to run the health-gate."
    exit 64
fi

run_verify() {
    # Run verify_stability_refresh.py with the given extra args, print JSON to stdout.
    if [[ "${PYTHON_BIN}" == "uv-run" ]]; then
        uv run --project "${REPO_ROOT}" python "${VERIFY_SCRIPT}" "$@"
    else
        "${PYTHON_BIN}" "${VERIFY_SCRIPT}" "$@"
    fi
}

# --- ambient-clone git wrapper (OMN-14900) ----------------------------------
# The clones under OMNI_HOME are owned by the uid that provisioned them, but
# this script can run as a DIFFERENT uid -- e.g. inside the self-hosted deploy
# runner container. git then refuses every operation on them with "fatal:
# detected dubious ownership in repository at '<clone>'" and exits 128. Five
# release-train-lab.yml runs on 2026-07-21 died exactly there, before a single
# fetch happened.
#
# That was relieved out-of-band on the host with an ad hoc `git config
# --global --add safe.directory ...`, but that relief is NOT DURABLE: the
# runner's /home/runner is not a named volume, so the global gitconfig -- and
# with it the exemption -- is lost on any `--force-recreate` of the runner
# container, and the next tag push fails the same way. Carry the exemption
# in-repo instead, scoped per invocation: `-c safe.directory=<clone>` is
# protected configuration git honours from the command line, applies to
# exactly one git process and one path, requires no host change, writes
# nothing to the user's global gitconfig, and is never the blanket
# `safe.directory=*`. (Ported verbatim from refresh_dev_lane.sh.)
git_clone() {
    local clone="$1"
    shift
    git -c "safe.directory=${clone}" -C "${clone}" "$@"
}

# --- state / receipt paths ---------------------------------------------------
STATE_DIR="${HOME}/.omnibase/state/stability_lane_refresh"
HISTORY_DIR="${STATE_DIR}/history"
mkdir -p "${HISTORY_DIR}"
UTC_NOW="$(date -u +%Y%m%dT%H%M%SZ)"
WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/refresh-stability-lane.XXXXXX")"
trap 'rm -rf "${WORKDIR}"' EXIT

# --- plan / logging -----------------------------------------------------------
log "lane            : ${LANE} (compose project ${COMPOSE_PROJECT})"
log "ref             : ${REF}"
log "min contracts   : ${MIN_CONTRACTS}"
log "core services   : ${CORE_SERVICES[*]}"
log "mode            : ${MODE}"

if [[ "${MODE}" != "execute" ]]; then
    log "dry-run: no fetch/checkout/build/restart performed. Re-run with --execute."
    log "would run: OMNI_HOME=${OMNI_HOME} OMNIBASE_INFRA_COMPOSE_PROJECT=${COMPOSE_PROJECT} \\"
    log "  BUILD_SOURCE=workspace DEPLOY_REF=${REF} \\"
    log "  RUNTIME_BUILD_SERVICES_OVERRIDE=\"${CORE_SERVICES[*]}\" \\"
    log "  ${DEPLOY_RUNTIME} --execute --force --restart"
    exit 0
fi

# =============================================================================
# 0. Ensure the private runner clones exist (OMN-14900)
#    Idempotent: clones any of the 5 repos missing under OMNI_HOME (the deploy
#    runner's PRIVATE bind mount) and asserts each is operable by this euid.
#    Committed provisioning, never a remembered manual host step.
# =============================================================================
OMNI_HOME="${OMNI_HOME}" bash "${SCRIPT_DIR}/ensure_runner_clones.sh"

# =============================================================================
# 1. Capture PRE-STATE (before touching anything)
# =============================================================================
log "=== Capture pre-state ==="

declare -A PRE_IMAGE_IDS
for svc in "${CORE_SERVICES[@]}"; do
    container="${CORE_CONTAINERS[${svc}]}"
    image_id="$(docker inspect "${container}" --format '{{.Image}}' 2>/dev/null || true)"
    if [[ -z "${image_id}" ]]; then
        err "Could not resolve running image ID for ${container} (${svc}). Is the lane up?"
        exit 64
    fi
    PRE_IMAGE_IDS["${svc}"]="${image_id}"
    log "  ${svc} (${container}): pre_image_id=${image_id}"
done

OLD_INFRA_REVISION="$(docker inspect "${CORE_CONTAINERS[omninode-runtime]}" \
    --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' 2>/dev/null || true)"
log "  current org.opencontainers.image.revision=${OLD_INFRA_REVISION:-<none>}"

declare -A PRIOR_REFS
for repo in "${ALL_TRACKED_REPOS[@]}"; do
    clone="${OMNI_HOME}/${repo}"
    if [[ ! -d "${clone}/.git" ]]; then
        err "Expected clone not found: ${clone}"
        exit 64
    fi
    PRIOR_REFS["${repo}"]="$(git_clone "${clone}" rev-parse HEAD)"
    log "  ${repo} prior HEAD: ${PRIOR_REFS[${repo}]:0:12}"
done

# =============================================================================
# 2. Preflight rollback tag (BEFORE any build can overwrite :latest in place)
# =============================================================================
log "=== Tag preflight rollback anchor (${UTC_NOW}) ==="
for svc in "${CORE_SERVICES[@]}"; do
    image_tag="${COMPOSE_PROJECT}-${svc}"
    docker tag "${image_tag}:latest" "${image_tag}:preflight-${UTC_NOW}"
    log "  tagged ${image_tag}:latest -> ${image_tag}:preflight-${UTC_NOW}"
done

# =============================================================================
# 3. Refresh the omnibase_infra ambient clone itself to --ref
#    (deploy-runtime.sh reads its own git_sha from THIS clone's HEAD; DEPLOY_REF
#    below only pins the 4 SIBLING repos staged by stage_workspace.sh -- it
#    does not touch omnibase_infra's own tree.)
#
#    OMN-14889 canary finding: this clone's local `dev` branch may already be
#    checked out in a SIBLING worktree on the same host (e.g. deploy-agent's
#    runtime-sync-worktrees/OMN-12618) -- git refuses to check that branch out
#    a second time ("already checked out at <path>"), hard-failing this whole script on
#    an environment collision that has nothing to do with lane health. Resolve
#    --ref to a commit SHA first and check it out DETACHED: a detached HEAD
#    carries no branch identity, so it can never collide with another
#    worktree regardless of what branch that worktree holds. --force discards
#    any local modifications the same way the previous checkout+reset --hard
#    pair did.
# =============================================================================
log "=== Refresh omnibase_infra ambient clone to ${REF} ==="
INFRA_CLONE="${OMNI_HOME}/omnibase_infra"
git_clone "${INFRA_CLONE}" fetch origin --prune
RESOLVED_REF_SHA="$(git_clone "${INFRA_CLONE}" rev-parse "${REF}^{commit}")"
git_clone "${INFRA_CLONE}" checkout --force --detach "${RESOLVED_REF_SHA}"
NEW_INFRA_SHA="$(git_clone "${INFRA_CLONE}" rev-parse HEAD)"
NEW_INFRA_SHA_SHORT="$(git_clone "${INFRA_CLONE}" rev-parse --short=12 HEAD)"
log "  omnibase_infra now at ${NEW_INFRA_SHA_SHORT} (full: ${NEW_INFRA_SHA})"

# =============================================================================
# 4. Build + restart (scoped to the 4 core services) via deploy-runtime.sh
# =============================================================================
log "=== Build + restart (scoped to: ${CORE_SERVICES[*]}) ==="
DEPLOY_EXIT=0
(
    export OMNI_HOME
    export OMNIBASE_INFRA_COMPOSE_PROJECT="${COMPOSE_PROJECT}"
    export BUILD_SOURCE="workspace"
    export DEPLOY_REF="${REF}"
    export RUNTIME_BUILD_SERVICES_OVERRIDE="${CORE_SERVICES[*]}"
    bash "${DEPLOY_RUNTIME}" --execute --force --restart
) || DEPLOY_EXIT=$?

if [[ "${DEPLOY_EXIT}" -ne 0 ]]; then
    log "deploy-runtime.sh exited ${DEPLOY_EXIT} -- proceeding to health-gate anyway"
    log "  (it may have partially succeeded; the health-gate below is the source of truth)"
fi

# =============================================================================
# 5. Capture NEW refs (post-refresh) for every tracked repo + ancestry proof
# =============================================================================
declare -A NEW_REFS
ANCESTRY_OK=true
ANCESTRY_COMMANDS=()
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
    err "Forward-progress assertion FAILED for at least one repo. This refresh would REGRESS"
    err "the lane. Refusing to certify success regardless of health-gate outcome."
fi

# =============================================================================
# 6. Health-gate
# =============================================================================
log "=== Health-gate ==="
PRE_IMAGE_IDS_JSON="$(
    printf '%s\n' "${!PRE_IMAGE_IDS[@]}" | while read -r k; do
        printf '%s\t%s\n' "${k}" "${PRE_IMAGE_IDS[${k}]}"
    done | jq -Rn '[inputs | split("\t") | {(.[0]): .[1]}] | add'
)"

GATE1_JSON="${WORKDIR}/gate1.json"
GATE_EXIT=0
run_verify \
    --lane "${LANE}" \
    --expected-revision "${NEW_INFRA_SHA_SHORT}" \
    --pre-image-ids "${PRE_IMAGE_IDS_JSON}" \
    --manifest-url "${MANIFEST_URL}" \
    --health-url "${HEALTH_URL}" \
    --broker-container "${REDPANDA_CONTAINER}" \
    --min-contracts "${MIN_CONTRACTS}" \
    --consumer-groups-file "${CONSUMER_GROUPS_FILE}" \
    --json > "${GATE1_JSON}" || GATE_EXIT=$?

# Defensive: if the health-gate crashed before printing valid JSON (should not
# happen -- run_verify's own main() always emits JSON on every documented exit
# path -- but a receipt-writing script must never itself crash on a malformed
# upstream artifact), fall back to a minimal INFRA_ERROR placeholder so the
# jq reads below (and the final receipt assembly) always have valid JSON.
if ! jq -e . "${GATE1_JSON}" >/dev/null 2>&1; then
    err "health-gate did not produce valid JSON (exit ${GATE_EXIT}) -- writing INFRA_ERROR placeholder"
    jq -n --arg lane "${LANE}" '{lane: $lane, overall: "INFRA_ERROR", errors: ["health-gate produced no valid JSON output"]}' > "${GATE1_JSON}"
fi

GATE1_OVERALL="$(jq -r '.overall // "INFRA_ERROR"' "${GATE1_JSON}" 2>/dev/null || echo "INFRA_ERROR")"
log "health-gate result: ${GATE1_OVERALL} (exit ${GATE_EXIT})"
cat "${GATE1_JSON}" >&2

ROLLBACK_TRIGGERED=false
GATE2_JSON=""
RESULT="FAILED"

if [[ "${GATE1_OVERALL}" == "PASS" && "${ANCESTRY_OK}" == true ]]; then
    RESULT="SUCCESS"
    log "=== SUCCESS: health-gate PASS, ancestry OK ==="
    # Prune old preflight tags (keep last 3) -- bounded local rollback history.
    for svc in "${CORE_SERVICES[@]}"; do
        image_tag="${COMPOSE_PROJECT}-${svc}"
        old_tags="$(docker images --format '{{.Tag}}' "${image_tag}" | grep '^preflight-' | sort -r | tail -n +4 || true)"
        for t in ${old_tags}; do
            docker rmi "${image_tag}:${t}" >/dev/null 2>&1 || true
        done
    done
else
    log "=== FAILURE: triggering rollback ==="
    ROLLBACK_TRIGGERED=true

    for svc in "${CORE_SERVICES[@]}"; do
        image_tag="${COMPOSE_PROJECT}-${svc}"
        docker tag "${image_tag}:preflight-${UTC_NOW}" "${image_tag}:latest"
        log "  rolled back ${image_tag}:latest <- ${image_tag}:preflight-${UTC_NOW}"
    done

    # Targeted recreate against the rolled-back images (same command shape as
    # deploy-runtime.sh's restart_services(), scoped to the 4 core services).
    # Guarded (never let a compose failure here abort the script under set -e
    # -- a rollback-recreate failure must still reach the health-gate +
    # receipt below, not crash silently with no receipt at all).
    ROLLBACK_RECREATE_EXIT=0
    docker compose -p "${COMPOSE_PROJECT}" \
        -f "${INFRA_CLONE}/docker/docker-compose.infra.yml" \
        -f "${INFRA_CLONE}/docker/docker-compose.stability-test.yml" \
        --profile runtime \
        up -d --no-deps --no-build --force-recreate \
        "${CORE_SERVICES[@]}" || ROLLBACK_RECREATE_EXIT=$?
    if [[ "${ROLLBACK_RECREATE_EXIT}" -ne 0 ]]; then
        err "rollback targeted-recreate exited ${ROLLBACK_RECREATE_EXIT} -- proceeding to health-gate anyway (it will report the true state)"
    fi

    log "=== Re-verifying health after rollback ==="
    GATE2_JSON="${WORKDIR}/gate2.json"
    GATE2_EXIT=0
    run_verify \
        --lane "${LANE}" \
        --expected-revision "${OLD_INFRA_REVISION:-unknown}" \
        --pre-image-ids '{}' \
        --manifest-url "${MANIFEST_URL}" \
        --health-url "${HEALTH_URL}" \
        --broker-container "${REDPANDA_CONTAINER}" \
        --min-contracts "${MIN_CONTRACTS}" \
        --consumer-groups-file "${CONSUMER_GROUPS_FILE}" \
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
    elif [[ "${GATE2_OVERALL}" == "INFRA_ERROR" ]]; then
        # OMN-14958: an INFRA_ERROR gate means the probes could not RUN (e.g.
        # ${HEALTH_URL} unreachable from THIS execution context, docker/rpk
        # missing) -- the lane was never actually observed unhealthy. Do NOT
        # claim lane damage the gate never saw; that false alarm is exactly
        # what deploy run 29977968728 emitted from inside the runner container.
        RESULT="FAILED"
        err "=============================================================="
        err "STOP AND REPORT: post-rollback health-gate is INFRA_ERROR -- the"
        err "lane is UNREACHABLE FROM THIS PROBE CONTEXT, not proven unhealthy."
        err "Probes attempted: ${HEALTH_URL} / ${MANIFEST_URL}."
        err "If this ran inside the containerized deploy runner, localhost is"
        err "the runner container, not the lane host -- set LANE_PROBE_HOST"
        err "(or --health-url/--manifest-url) to a host reachable from here,"
        err "then verify the lane's true state from a host that can reach it"
        err "(e.g. curl the lane's /health on the lane host) before treating"
        err "this as lane damage. Do not retry automatically."
        err "=============================================================="
    else
        RESULT="FAILED"
        err "=============================================================="
        err "STOP AND REPORT: rollback did NOT restore a healthy lane."
        err "The stability-test lane may be UNHEALTHY right now (health-gate"
        err "readback FAILED -- this is an observed-unhealthy verdict, not a"
        err "probe-context error). Do not retry automatically -- this is a"
        err "STOP condition per operating rules."
        err "Suggested Linear ticket:"
        err "  title: stability-test lane unhealthy after refresh+rollback (OMN-14263 recurrence)"
        err "  body: refresh to ${REF} failed health-gate; rollback to preflight-${UTC_NOW}"
        err "        images ALSO failed health-gate. Manual intervention required."
        err "=============================================================="
    fi
fi

# =============================================================================
# 7. Emit receipt
# =============================================================================
PRIOR_REFS_JSON="$(for r in "${ALL_TRACKED_REPOS[@]}"; do printf '%s\t%s\n' "${r}" "${PRIOR_REFS[${r}]}"; done | jq -Rn '[inputs | split("\t") | {(.[0]): .[1]}] | add')"
NEW_REFS_JSON="$(for r in "${ALL_TRACKED_REPOS[@]}"; do printf '%s\t%s\n' "${r}" "${NEW_REFS[${r}]:-${PRIOR_REFS[${r}]}}"; done | jq -Rn '[inputs | split("\t") | {(.[0]): .[1]}] | add')"
ANCESTRY_COMMANDS_JSON="$(printf '%s\n' "${ANCESTRY_COMMANDS[@]}" | jq -Rn '[inputs]')"
BUILD_SCOPE_JSON="$(printf '%s\n' "${CORE_SERVICES[@]}" | jq -Rn '[inputs]')"
ROLLBACK_GATE_JSON="null"
if [[ -n "${GATE2_JSON}" && -f "${GATE2_JSON}" ]]; then
    ROLLBACK_GATE_JSON="$(cat "${GATE2_JSON}")"
fi

RECEIPT_PATH="${HISTORY_DIR}/${UTC_NOW}-${NEW_INFRA_SHA_SHORT}.json"
jq -n \
    --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg lane "${LANE}" \
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
