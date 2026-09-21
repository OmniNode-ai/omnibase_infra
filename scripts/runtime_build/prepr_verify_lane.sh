#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# =============================================================================
# prepr_verify_lane.sh -- bring up ONE ephemeral pre-PR verify slot
# (OMN-18893, Task 4 of epic OMN-18888)
# =============================================================================
#
# This is the single entrypoint rule 24(e) sanctions by name. Read that rule
# before changing anything here: the sanction is bound to THIS path, and it is
# bound to it precisely because the refusals below are the point.
#
# WHAT IT DOES
#   claim -> lock -> clean-tree check -> slot-private source snapshot ->
#   provision the slot's databases and principals -> workspace build from the
#   snapshot -> render-and-verify -> migrate -> up -> readiness ->
#   print the slot descriptor.
#
# WHAT IT REFUSES, ALWAYS
#   * every declared lane, BY NAME (scripts/runtime_build/prepr_slot_policy.py)
#   * any compose project that is neither a declared lane nor a pool slot
#   * a dirty target worktree
#   * a build with no stated reason (the attribution preflight)
#   * a second concurrent build (the pool-wide build lock)
#   * a rendered configuration that does not match the slot policy
#
# IT ACCEPTS NO LANE ARGUMENT AT ALL. There is no --lane, no --compose-project,
# no --force and no --skip. The compose project is DERIVED from the slot
# number, so there is no argument through which a caller could aim this script
# at a governed lane. That is the whole reason this is a separate entrypoint
# rather than a flag on scripts/deploy-runtime.sh: a pool arm there would put a
# branch workspace build one argument away from every governed lane.
#
# WHY NOT stage_workspace.sh
#   The existing workspace staging helper rsyncs each sibling from the AMBIENT
#   canonical clone on the host and, when a deploy ref is set, brings those
#   clones to a clean checkout of it. Two things follow, and both are
#   disqualifying here. A slot calling it would test CANONICAL sibling code
#   while believing it tested the branch, and it would MUTATE the shared clones
#   that the dev lane's own builds and a second slot read. The build lock
#   bounds the second problem; only an isolated snapshot removes it. So this
#   script stages into a slot-private root and never writes to a canonical
#   clone. A test asserts the canonical clones' commit and working-tree state
#   are unchanged across a slot build.
#
# WHAT A SLOT IS NOT
#   A premise for anything. Outside GOVERNED_LANES and GRANT_INTERLOCK_LANES,
#   sourcing no `stability-proven` digest, never part of a promotion's evidence
#   chain. The attribution preflight is extended to the pool for its REASON
#   requirement only; the live-grant interlock is deliberately NOT extended.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
POLICY_PY="${SCRIPT_DIR}/prepr_slot_policy.py"
LANE_LOCK_PY="${SCRIPT_DIR}/lane_lock.py"
PROVISION_DB="${REPO_ROOT}/scripts/provision_db_slot.sh"
ATTRIBUTION_PY="${REPO_ROOT}/scripts/preflight_lane_deploy_attribution.py"

# Exit codes, mirrored from prepr_slot_policy.py. Duplicated as literals here
# because a shell cannot import a Python constant, and pinned equal to the
# module by tests/unit/scripts/test_prepr_verify_lane_entrypoint_omn18893.py so
# the duplication cannot drift.
EXIT_USAGE=2
# These two are never spelled at an `exit` in this script, and that is correct:
# the declared-lane and unknown-project refusals are raised by
# prepr_slot_policy.py and this script propagates its code verbatim with
# `|| exit $?`. They are declared here anyway because the test module reads
# every EXIT_* literal back against the Python constants, so a renumbering in
# one place is a red test rather than a caller misreading which refusal fired.
# shellcheck disable=SC2034
EXIT_REFUSED_DECLARED_LANE=3
# shellcheck disable=SC2034
EXIT_REFUSED_UNKNOWN_PROJECT=4
EXIT_REFUSED_DIRTY_WORKTREE=5
EXIT_REFUSED_ATTRIBUTION=6
EXIT_BUILD_LOCK_CONTENDED=7
EXIT_SLOT_LOCK_CONTENDED=8
EXIT_PROVISION_FAILED=9
EXIT_BUILD_FAILED=10
EXIT_BOOT_FAILED=11
EXIT_PROVENANCE_MISMATCH=12

# The siblings a workspace build vendors, in the order the Dockerfile needs
# them installed (core first -- OMN-13405).
SIBLING_REPOS=(omnibase_core omnibase_compat omnimarket)

log()  { printf '[prepr-verify-lane] %s\n' "$*" >&2; }
fail() { local code="$1"; shift; printf '[prepr-verify-lane] REFUSED/FAILED: %s\n' "$*" >&2; exit "${code}"; }

python_bin() {
    if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
        printf '%s\n' "${REPO_ROOT}/.venv/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        printf 'python3\n'
    else
        return 1
    fi
}

usage() {
    cat >&2 <<'USAGE'
usage: prepr_verify_lane.sh --slot <1|2> --worktree <path> --reason <text> [options]

required:
  --slot <n>          the pool slot to claim. The compose project, ports, topic
                      namespace, database suffix and Valkey index are all
                      DERIVED from it. There is no way to name them directly.
  --worktree <path>   the git worktree whose HEAD is built. Must be clean.
  --reason <text>     the attribution reason. Also readable from
                      ONEX_DEPLOY_REASON. A slot build with no stated reason is
                      refused exactly as a stability-test build is.

options:
  --with-gateway      also start the slot's onex-api container. OFF by default:
                      onex-api is image-referenced rather than lane-built, and a
                      tag predating OMN-18891 carries no topic-namespace surface,
                      so starting it would publish UNPREFIXED into the dev lane's
                      topics. Pinning that image is Task 5 (OMN-18894).
  --descriptor-out <path>
                      where to write the slot descriptor JSON. Default:
                      .onex_state/prepr/slot-<n>/descriptor.json under the repo.
  --staging-root <path>
                      the slot-private source snapshot root. Default:
                      /var/tmp/onex-prepr/slot-<n>.
  --build-timeout <s> seconds to allow the image build. Default 3600.
  --boot-timeout <s>  seconds to allow the slot to become ready. Default 2400.
  --lock-timeout <s>  seconds to wait for the slot and pool-build locks.
                      Default 5400, which must exceed one full build.
  --plan-only         do every check and refusal, print what WOULD be built,
                      and open no connection, build nothing and start nothing.
  -h, --help          this text.

There is deliberately NO --lane, --compose-project, --force or --skip option.
USAGE
    exit "${EXIT_USAGE}"
}

SLOT=""
WORKTREE=""
REASON="${ONEX_DEPLOY_REASON:-}"
WITH_GATEWAY=0
DESCRIPTOR_OUT=""
STAGING_ROOT=""
BUILD_TIMEOUT=3600
BOOT_TIMEOUT=2400
LOCK_TIMEOUT=5400
PLAN_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --slot)            shift || usage; [[ $# -gt 0 ]] || usage; SLOT="$1" ;;
        --worktree)        shift || usage; [[ $# -gt 0 ]] || usage; WORKTREE="$1" ;;
        --reason)          shift || usage; [[ $# -gt 0 ]] || usage; REASON="$1" ;;
        --descriptor-out)  shift || usage; [[ $# -gt 0 ]] || usage; DESCRIPTOR_OUT="$1" ;;
        --staging-root)    shift || usage; [[ $# -gt 0 ]] || usage; STAGING_ROOT="$1" ;;
        --build-timeout)   shift || usage; [[ $# -gt 0 ]] || usage; BUILD_TIMEOUT="$1" ;;
        --boot-timeout)    shift || usage; [[ $# -gt 0 ]] || usage; BOOT_TIMEOUT="$1" ;;
        --lock-timeout)    shift || usage; [[ $# -gt 0 ]] || usage; LOCK_TIMEOUT="$1" ;;
        --with-gateway)    WITH_GATEWAY=1 ;;
        --plan-only)       PLAN_ONLY=1 ;;
        -h|--help)         usage ;;
        *) printf '[prepr-verify-lane] unknown option: %s\n' "$1" >&2; usage ;;
    esac
    shift
done

[[ -n "${SLOT}" ]] || usage
[[ -n "${WORKTREE}" ]] || usage

PY="$(python_bin)" || fail "${EXIT_USAGE}" "no python3 available."

# -----------------------------------------------------------------------------
# 1. POLICY. Resolve the slot, which is also the refusal of everything else.
# -----------------------------------------------------------------------------
SLOT_JSON="$("${PY}" "${POLICY_PY}" --slot "${SLOT}")" || exit $?
read_policy() { printf '%s' "${SLOT_JSON}" | "${PY}" -c "import json,sys;print(json.load(sys.stdin)['$1'])"; }

COMPOSE_PROJECT="$(read_policy compose_project)"
DB_SLOT="$(read_policy db_slot)"
TOPIC_NAMESPACE="$(read_policy topic_namespace)"
KAFKA_ENV_TOKEN="$(read_policy kafka_environment)"
VALKEY_INDEX="$(read_policy valkey_db_index)"
MAIN_PORT="$(read_policy runtime_main_port)"
EFFECTS_PORT="$(read_policy runtime_effects_port)"
GATEWAY_PORT="$(read_policy gateway_port)"
PROJECTION_API_PORT="$(read_policy projection_api_port)"
POOL_BUILD_LOCK_PROJECT="$("${PY}" -c "
import sys; sys.path.insert(0, '${SCRIPT_DIR}')
from prepr_slot_policy import POOL_BUILD_LOCK_PROJECT; print(POOL_BUILD_LOCK_PROJECT)")"

# The derived project is run back through the refusal table. This looks
# redundant -- we just derived it from the slot -- and it is not: it is the
# assertion that the derivation can never produce a governed lane, and it is
# what a future edit to the policy table trips over rather than silently
# widening. A gate that only checks the caller's input cannot catch a table
# that started producing the wrong output.
"${PY}" "${POLICY_PY}" --assert-pool-slot "${COMPOSE_PROJECT}" >/dev/null || exit $?

LANE_NAME="${COMPOSE_PROJECT#omnibase-infra-}"

log "slot ${SLOT} -> project ${COMPOSE_PROJECT}, namespace '${TOPIC_NAMESPACE}', db suffix '_${DB_SLOT}', ports ${MAIN_PORT}/${EFFECTS_PORT}/${GATEWAY_PORT}/${PROJECTION_API_PORT}"

STAGING_ROOT="${STAGING_ROOT:-/var/tmp/onex-prepr/slot-${SLOT}}"
DESCRIPTOR_OUT="${DESCRIPTOR_OUT:-${REPO_ROOT}/.onex_state/prepr/slot-${SLOT}/descriptor.json}"
SLOT_ENV_DIR="${STAGING_ROOT}/.slot-env"
PROVISIONER_ENV="${SLOT_ENV_DIR}/provisioner.env"
COMPOSE_ENV="${SLOT_ENV_DIR}/compose.env"
TENANT_STATE_DIR="${STAGING_ROOT}/tenant-state"

# -----------------------------------------------------------------------------
# 2. ATTRIBUTION. A slot build with no stated reason is refused.
#
# The pool lanes are in REASON_REQUIRED_LANES and deliberately NOT in
# GRANT_INTERLOCK_LANES: a slot promotes nothing, so a live prod grant's
# premise cannot be eroded by building one, and making a slot wait on a grant
# resolution would gate a branch build on the change-control repository.
# -----------------------------------------------------------------------------
if [[ -z "${REASON}" ]]; then
    fail "${EXIT_REFUSED_ATTRIBUTION}" \
        "no reason given. Pass --reason or set ONEX_DEPLOY_REASON. A build
  nobody stated a purpose for is an unattributed mutation of a shared host,
  which is the OMN-15218 failure this preflight exists for."
fi
ONEX_DEPLOY_REASON="${REASON}" \
    "${PY}" "${ATTRIBUTION_PY}" \
        --compose-project "${COMPOSE_PROJECT}" \
        --source prepr_verify_lane.sh \
        --invoking-command "slot ${SLOT} <- ${WORKTREE}" \
    || fail "${EXIT_REFUSED_ATTRIBUTION}" "the lane-deploy attribution preflight refused this build."

# -----------------------------------------------------------------------------
# 3. CLEAN TREE. A build from uncommitted content describes a commit that does
# not contain it. The pull-request gate keys on a commit, so an unreproducible
# image is a false pass waiting to happen -- refuse rather than label it.
# -----------------------------------------------------------------------------
[[ -d "${WORKTREE}" ]] || fail "${EXIT_USAGE}" "worktree '${WORKTREE}' does not exist."
git -C "${WORKTREE}" rev-parse --git-dir >/dev/null 2>&1 \
    || fail "${EXIT_USAGE}" "'${WORKTREE}' is not a git worktree."

TARGET_DIRT="$(git -C "${WORKTREE}" status --porcelain)"
if [[ -n "${TARGET_DIRT}" ]]; then
    fail "${EXIT_REFUSED_DIRTY_WORKTREE}" \
        "the target worktree is DIRTY and will not be built. The receipt this
  slot produces names a commit; an image built from uncommitted content does
  not correspond to that commit and nobody could reproduce it. Commit the work.
  Uncommitted paths:
$(printf '%s\n' "${TARGET_DIRT}" | sed 's/^/    /')"
fi

TARGET_COMMIT="$(git -C "${WORKTREE}" rev-parse HEAD)"
TARGET_BRANCH="$(git -C "${WORKTREE}" rev-parse --abbrev-ref HEAD)"
TARGET_REPO="$(basename "${WORKTREE}")"
log "target ${TARGET_REPO} @ ${TARGET_COMMIT} (${TARGET_BRANCH}), clean."

# Sibling sources. A sibling worktree beside the target under the same ticket
# directory wins over the canonical clone, because a multi-repo branch is the
# normal shape here and building the target's branch against canonical siblings
# would test a combination that exists nowhere.
OMNI_HOME_RESOLVED="${OMNI_HOME:-}"
[[ -n "${OMNI_HOME_RESOLVED}" ]] || fail "${EXIT_USAGE}" \
    "OMNI_HOME must be set; the sibling repositories are resolved from it."
TICKET_DIR="$(dirname "${WORKTREE}")"

declare -A SIBLING_SRC=()
declare -A SIBLING_COMMIT=()
declare -A SIBLING_ORIGIN=()
for repo in "${SIBLING_REPOS[@]}"; do
    if [[ -d "${TICKET_DIR}/${repo}/.git" || -f "${TICKET_DIR}/${repo}/.git" ]]; then
        SIBLING_SRC["${repo}"]="${TICKET_DIR}/${repo}"
        SIBLING_ORIGIN["${repo}"]="ticket-worktree"
    elif [[ -d "${OMNI_HOME_RESOLVED}/${repo}/.git" ]]; then
        SIBLING_SRC["${repo}"]="${OMNI_HOME_RESOLVED}/${repo}"
        SIBLING_ORIGIN["${repo}"]="canonical-clone"
    else
        fail "${EXIT_USAGE}" "sibling repository '${repo}' found neither at ${TICKET_DIR}/${repo} nor at ${OMNI_HOME_RESOLVED}/${repo}."
    fi
    src="${SIBLING_SRC[${repo}]}"
    SIBLING_COMMIT["${repo}"]="$(git -C "${src}" rev-parse HEAD)"
    # A dirty SIBLING is recorded, not refused. The receipt's clean-tree
    # assertion is about the TARGET, whose commit the gate keys on; a sibling
    # is vendored content and its dirt is a provenance fact the descriptor must
    # carry rather than a reason to block a branch that did not cause it.
    if [[ -n "$(git -C "${src}" status --porcelain)" ]]; then
        SIBLING_ORIGIN["${repo}"]="${SIBLING_ORIGIN[${repo}]}+dirty"
        log "WARNING: sibling ${repo} at ${src} is DIRTY; recording it as such on the descriptor."
    fi
    log "sibling ${repo} <- ${src} @ ${SIBLING_COMMIT[${repo}]} (${SIBLING_ORIGIN[${repo}]})"
done

if [[ "${PLAN_ONLY}" == "1" ]]; then
    log "--plan-only: every refusal passed. Nothing was built, started or connected to."
    "${PY}" - "$@" <<PYPLAN
import json, sys
print(json.dumps({
    "plan_only": True,
    "slot": ${SLOT},
    "compose_project": "${COMPOSE_PROJECT}",
    "target_repo": "${TARGET_REPO}",
    "target_commit": "${TARGET_COMMIT}",
    "target_branch": "${TARGET_BRANCH}",
    "staging_root": "${STAGING_ROOT}",
    "topic_namespace": "${TOPIC_NAMESPACE}",
    "db_slot": "${DB_SLOT}",
    "with_gateway": bool(${WITH_GATEWAY}),
}, indent=2))
PYPLAN
    exit 0
fi

# -----------------------------------------------------------------------------
# 4. LOCKS. The slot lock covers the whole critical section; the pool-wide
# build lock covers the BUILD only and is released the moment it finishes, so
# two slots overlap freely in boot, probe and teardown -- which is where most
# of the wall clock that is not the build actually goes.
# -----------------------------------------------------------------------------
# shellcheck source=scripts/runtime_build/lane_lock.sh
source "${SCRIPT_DIR}/lane_lock.sh"

POOL_LOCK_OWNED=0
pool_build_lock_acquire() {
    local lock_file
    lock_file="$("${PY}" "${LANE_LOCK_PY}" path --compose-project "${POOL_BUILD_LOCK_PROJECT}")" \
        || return 3
    mkdir -p "$(dirname "${lock_file}")"
    exec 8>"${lock_file}" || return 3
    if ! "${PY}" "${LANE_LOCK_PY}" acquire \
            --compose-project "${POOL_BUILD_LOCK_PROJECT}" \
            --fd 8 --timeout "${LOCK_TIMEOUT}" \
            --lane "prepr-pool-build" --ref "${TARGET_COMMIT}" \
            --argv "slot ${SLOT} ${TARGET_REPO}@${TARGET_COMMIT}"; then
        exec 8>&- 2>/dev/null || true
        return 2
    fi
    POOL_LOCK_OWNED=1
    return 0
}
pool_build_lock_release() {
    [[ "${POOL_LOCK_OWNED}" == "1" ]] || return 0
    "${PY}" "${LANE_LOCK_PY}" release --compose-project "${POOL_BUILD_LOCK_PROJECT}" >/dev/null 2>&1 || true
    exec 8>&- 2>/dev/null || true
    POOL_LOCK_OWNED=0
    log "released the pool build lock."
}

cleanup() {
    pool_build_lock_release
    lane_lock_release
}
trap cleanup EXIT

if ! lane_lock_acquire "${COMPOSE_PROJECT}" "${LANE_NAME}" "${TARGET_COMMIT}" "${LOCK_TIMEOUT}" "$0" "$@"; then
    fail "${EXIT_SLOT_LOCK_CONTENDED}" "slot ${SLOT} is held by another process; the holder was named above."
fi

# -----------------------------------------------------------------------------
# 5. SLOT-PRIVATE SOURCE SNAPSHOT.
#
# Everything the build reads is copied here first. No canonical clone is
# checked out, fetched into, reset or cleaned at any point -- this script only
# ever READS them, and a test asserts their commit and working-tree state are
# unchanged across a build.
#
# --delete so a re-run of the same slot cannot build a union of two branches.
# The .git directory is excluded: it is large, the build does not read it, and
# in a worktree it is a file pointing back at the canonical clone, so copying
# it would make `git` inside the snapshot answer about the clone rather than
# about the snapshot -- which is exactly the unreliable-narrator property the
# isolated snapshot exists to remove.
# -----------------------------------------------------------------------------
log "staging a slot-private snapshot into ${STAGING_ROOT} ..."
rm -rf "${STAGING_ROOT}"
mkdir -p "${STAGING_ROOT}" "${SLOT_ENV_DIR}" "${TENANT_STATE_DIR}"
chmod 700 "${SLOT_ENV_DIR}"

rsync -a --delete \
    --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
    --exclude '.onex_state' --exclude 'node_modules' \
    "${WORKTREE}/" "${STAGING_ROOT}/repo/"

mkdir -p "${STAGING_ROOT}/repo/workspace/sibling-repos"
for repo in "${SIBLING_REPOS[@]}"; do
    rsync -a --delete \
        --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
        --exclude '.onex_state' --exclude 'node_modules' \
        "${SIBLING_SRC[${repo}]}/" "${STAGING_ROOT}/repo/workspace/sibling-repos/${repo}/"
done

# Content digests of the snapshot itself. The commit above is read from the
# source at copy time; this is read from the BYTES that will be built, so the
# two together answer a question neither answers alone -- "which commit" and
# "and is this really it". The image digest recorded after the build binds both
# to the artifact.
snapshot_digest() {
    ( cd "$1" && find . -type f -not -path './workspace/sibling-repos/*' -print0 \
        | sort -z | xargs -0 sha256sum 2>/dev/null | sha256sum | cut -d' ' -f1 )
}
TARGET_SNAPSHOT_DIGEST="$(snapshot_digest "${STAGING_ROOT}/repo")"
declare -A SIBLING_SNAPSHOT_DIGEST=()
for repo in "${SIBLING_REPOS[@]}"; do
    SIBLING_SNAPSHOT_DIGEST["${repo}"]="$(snapshot_digest "${STAGING_ROOT}/repo/workspace/sibling-repos/${repo}")"
done
log "snapshot staged; target content digest ${TARGET_SNAPSHOT_DIGEST:0:16}..."

# -----------------------------------------------------------------------------
# 6. THE OPERATOR ENVIRONMENT, SOURCED BEFORE BOTH THE PROVISIONER AND THE SLOT.
#
# Same two files scripts/deploy-runtime.sh and refresh_dev_lane.sh source, in
# the same order, under `set -a`: the rendered runtime policy and the operator
# env file. Compose then reads the process environment, and no `--env-file` is
# passed -- the stale-snapshot copy that used to live at docker/.env was
# removed for good reasons and is not reintroduced here.
#
# ORDER IS LOAD-BEARING IN BOTH DIRECTIONS, and both were live defects.
#
# It must come BEFORE the provisioner: provision_db_slot.sh opens a superuser
# connection using POSTGRES_USER / POSTGRES_PASSWORD / POSTGRES_HOST, which
# live in this file and nowhere else. Sourced after it, the provisioner fails
# on an unset credential.
#
# It must come BEFORE the slot's own exports: the operator env file on the lab
# host sets KAFKA_ENVIRONMENT to the dev lane's own consumer-group token, so
# sourcing it afterwards would overwrite the slot's namespace with the dev
# lane's and put every slot service in the dev lane's consumer groups. The
# render gate re-reads the result rather than trusting this ordering, because
# an ordering argument in a comment is not a control.
# -----------------------------------------------------------------------------
OMNIBASE_OPERATOR_ENV_FILE="${OMNIBASE_OPERATOR_ENV_FILE:-${HOME}/.omnibase/.env}"
if [[ ! -r "${OMNIBASE_OPERATOR_ENV_FILE}" ]]; then
    fail "${EXIT_USAGE}" \
        "the operator env file is missing or unreadable at
  ${OMNIBASE_OPERATOR_ENV_FILE}
  The compose overlay resolves the shared broker, database and Keycloak
  credentials from it and every one of them is spelled fail-closed, so a slot
  cannot be brought up without it. Set OMNIBASE_OPERATOR_ENV_FILE to a readable
  path, as the deploy path does."
fi
RUNTIME_POLICY_ENV="${STAGING_ROOT}/repo/docker/runtime-policy.env"
set -a
if [[ -r "${RUNTIME_POLICY_ENV}" ]]; then
    # shellcheck disable=SC1090
    source "${RUNTIME_POLICY_ENV}"
fi
# shellcheck disable=SC1090
source "${OMNIBASE_OPERATOR_ENV_FILE}"
set +a

# -----------------------------------------------------------------------------
# 6b. THE SLOT'S DATABASES AND PRINCIPALS (OMN-18892).
#
# provision_db_slot.sh creates the suffixed databases and the slot's own login
# principals behind a name fence and an ownership fence, and refuses to report
# success until it has proven from the catalog that each principal reaches the
# slot's databases and nothing else. It never touches an existing shared role,
# which is the whole reason it exists rather than the initialiser.
# -----------------------------------------------------------------------------
log "provisioning slot databases and principals (ONEX_DB_SLOT=${DB_SLOT}) ..."
ONEX_DB_SLOT="${DB_SLOT}" bash "${PROVISION_DB}" --apply --env-file "${PROVISIONER_ENV}" \
    || fail "${EXIT_PROVISION_FAILED}" "slot database provisioning failed; nothing was built or started."

# Remap the provisioner's DERIVED password variable names onto the canonical,
# unsuffixed spellings the compose overlay resolves. This is what lets ONE
# overlay serve both slots: a file spelling ROLE_OMNIDASH_PREPR1_PASSWORD could
# not also serve slot 2. Credentials are moved between two 0600 files and are
# never placed on a command line, in an environment listing or in a log.
"${PY}" - "${PROVISIONER_ENV}" "${COMPOSE_ENV}" "${DB_SLOT}" <<'PYREMAP'
import os, sys, pathlib
src, dst, slot = sys.argv[1], sys.argv[2], sys.argv[3]
suffix = "_" + slot.upper() + "_PASSWORD"
out = []
for line in pathlib.Path(src).read_text(encoding="utf-8").splitlines():
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, _, value = line.partition("=")
    if key.endswith(suffix):
        # role_omnidash_prepr1 -> ROLE_OMNIDASH_PREPR1_PASSWORD -> ROLE_OMNIDASH_PASSWORD
        out.append(f"{key[: -len(suffix)]}_PASSWORD={value}")
p = pathlib.Path(dst)
p.write_text("\n".join(out) + "\n", encoding="utf-8")
os.chmod(p, 0o600)
print(f"remapped {len(out)} slot credentials onto their canonical names", file=sys.stderr)
PYREMAP

# -----------------------------------------------------------------------------
# 7. THE SLOT ENVIRONMENT, PASSED AS REAL PROCESS ENVIRONMENT.
#
# NOT through --env-file alone, and this is a live defect rather than a
# preference. Compose resolves an interpolation from the OS environment FIRST
# and from --env-file only as a fallback, so an ambient KAFKA_ENVIRONMENT in
# the invoking shell silently wins over the slot's value. The lab host's
# interactive shell exports exactly that variable, set to `local` -- the dev
# lane's own token -- so a slot brought up from a terminal would have joined the
# dev lane's consumer groups while every file on disk said it was isolated.
# Measured 2026-09-21 during this build. Exporting the slot's values puts them
# in the OS environment, where they outrank the ambient ones, and step 9
# verifies the rendered result rather than trusting this.
# -----------------------------------------------------------------------------
export ONEX_PREPR_SLOT="${SLOT}"
export ONEX_DB_SLOT="${DB_SLOT}"
export KAFKA_TOPIC_NAMESPACE="${TOPIC_NAMESPACE}"
export KAFKA_ENVIRONMENT="${KAFKA_ENV_TOKEN}"
export PREPR_VALKEY_DB_INDEX="${VALKEY_INDEX}"
export PREPR_RUNTIME_MAIN_PORT="${MAIN_PORT}"
export PREPR_RUNTIME_EFFECTS_PORT="${EFFECTS_PORT}"
export PREPR_GATEWAY_PORT="${GATEWAY_PORT}"
export PREPR_PROJECTION_API_PORT="${PROJECTION_API_PORT}"
export ONEX_PREPR_TENANT_STATE_DIR="${TENANT_STATE_DIR}"
export BUILD_SOURCE=workspace
export EXPECTED_BUILD_SOURCE=workspace
export OMNI_HOME="${OMNI_HOME_RESOLVED}"
export GIT_SHA="${TARGET_COMMIT}"

# The slot's own credentials, exported for the same precedence reason. Read
# from the 0600 file rather than passed on any command line.
set -a
# shellcheck disable=SC1090
source "${COMPOSE_ENV}"
set +a

SNAPSHOT_DOCKER="${STAGING_ROOT}/repo/docker"
COMPOSE_FILES=(
    -f "${SNAPSHOT_DOCKER}/docker-compose.infra.yml"
    -f "${SNAPSHOT_DOCKER}/docker-compose.dev-lane.yml"
    -f "${SNAPSHOT_DOCKER}/docker-compose.prepr.yml"
)
compose() { docker compose -p "${COMPOSE_PROJECT}" "${COMPOSE_FILES[@]}" "$@"; }

SLOT_SERVICES=(
    omninode-runtime runtime-effects runtime-worker projection-api
    tenant-projection-writer
    projection-tenant-registry-writer projection-delegation-writer
    projection-registration-writer projection-savings-writer
    projection-tenant-credentials-writer projection-live-events-writer
)
PROFILES=(--profile prepr)
if [[ "${WITH_GATEWAY}" == "1" ]]; then
    PROFILES+=(--profile prepr-gateway)
    SLOT_SERVICES+=(onex-api)
fi

# -----------------------------------------------------------------------------
# 8. BUILD, under the pool-wide build lock.
# -----------------------------------------------------------------------------
log "waiting for the pool build lock (one build at a time across the pool) ..."
if ! pool_build_lock_acquire; then
    fail "${EXIT_BUILD_LOCK_CONTENDED}" \
        "another slot holds the pool build lock; the holder was named above.
  Two 20-25 minute image builds at once on a host already at load average 7
  with 66 GiB swapped out is the contention this lock exists to prevent."
fi
log "building the runtime image from the snapshot (timeout ${BUILD_TIMEOUT}s) ..."
if ! timeout "${BUILD_TIMEOUT}" env -C "${STAGING_ROOT}/repo" \
        docker compose -p "${COMPOSE_PROJECT}" "${COMPOSE_FILES[@]}" "${PROFILES[@]}" \
        build omninode-runtime; then
    fail "${EXIT_BUILD_FAILED}" "the workspace build failed or timed out."
fi
pool_build_lock_release

# -----------------------------------------------------------------------------
# 9. RENDER AND VERIFY, before anything starts.
#
# The rendered configuration is read back and checked against the policy table
# for every slot service: container name, published ports, topic namespace,
# consumer-group environment token, every DSN's database suffix and principal,
# and every named volume. A mismatch refuses the bring-up.
#
# This is the step that makes the isolation a property of the SLOT rather than
# of the file: an ambient environment variable, a merge that did not replace a
# list, a base-file edit that added an unsuffixed DSN -- all of them show up
# here, and none of them would show up in a review of this script.
# -----------------------------------------------------------------------------
log "rendering and verifying the slot configuration ..."
RENDERED="${SLOT_ENV_DIR}/rendered.json"
compose "${PROFILES[@]}" config --format json > "${RENDERED}" 2>/dev/null \
    || fail "${EXIT_PROVENANCE_MISMATCH}" "the slot configuration did not render."

VERIFY_ARGS=(--rendered "${RENDERED}" --slot "${SLOT}")
[[ "${WITH_GATEWAY}" == "1" ]] && VERIFY_ARGS+=(--expect-gateway)
"${PY}" "${SCRIPT_DIR}/prepr_verify_rendered_slot.py" "${VERIFY_ARGS[@]}" \
    || fail "${EXIT_PROVENANCE_MISMATCH}" \
        "the rendered slot configuration does not match the slot policy. Nothing
  was started. The mismatches are listed above; each one is a way this slot
  would have reached the dev lane's own namespace."

# -----------------------------------------------------------------------------
# 10. MIGRATE the slot's databases, then bring the slot up.
# -----------------------------------------------------------------------------
log "running the forward migration against the slot's databases ..."
if ! compose --profile prepr-migrate run --rm --no-deps forward-migration; then
    fail "${EXIT_PROVISION_FAILED}" \
        "the forward migration failed against the slot's fresh databases.
  On a branch under test this is a FINDING, not a pool defect: a migration that
  only applies to an already-migrated database is a real defect that today
  reaches staging. Record it as the slot's verdict."
fi

log "starting the slot's services ..."
compose "${PROFILES[@]}" up -d --no-deps "${SLOT_SERVICES[@]}" \
    || fail "${EXIT_BOOT_FAILED}" "the slot did not start."

# -----------------------------------------------------------------------------
# 11. READINESS.
# -----------------------------------------------------------------------------
log "waiting up to ${BOOT_TIMEOUT}s for the slot runtime to answer on ${MAIN_PORT} ..."
deadline=$(( $(date +%s) + BOOT_TIMEOUT ))
ready=0
while [[ $(date +%s) -lt ${deadline} ]]; do
    if curl -sf "http://localhost:${MAIN_PORT}/health" >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 15
done
[[ "${ready}" == "1" ]] || fail "${EXIT_BOOT_FAILED}" \
    "the slot runtime did not become ready on port ${MAIN_PORT} within ${BOOT_TIMEOUT}s."

RUNNING_IMAGE_DIGEST="$(docker inspect --format '{{index .Image}}' \
    "omninode-prepr-${SLOT}-runtime" 2>/dev/null || echo "")"

# -----------------------------------------------------------------------------
# 12. THE SLOT DESCRIPTOR.
#
# Every provenance fact is RECORDED here rather than asserted anywhere: the
# target commit, each sibling's commit and where it came from, the clean-tree
# result, the snapshot content digests and the digest of the image that is
# actually running. An image label cannot prove what was built; these can be
# checked against each other.
# -----------------------------------------------------------------------------
mkdir -p "$(dirname "${DESCRIPTOR_OUT}")"
{
    printf '{\n'
    printf '  "schema": "onex.prepr.slot-descriptor.v1",\n'
    printf '  "slot": %s,\n' "${SLOT}"
    printf '  "compose_project": "%s",\n' "${COMPOSE_PROJECT}"
    printf '  "created_at": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '  "target": {"repo": "%s", "branch": "%s", "commit": "%s", "clean_tree": true, "snapshot_digest": "%s"},\n' \
        "${TARGET_REPO}" "${TARGET_BRANCH}" "${TARGET_COMMIT}" "${TARGET_SNAPSHOT_DIGEST}"
    printf '  "siblings": {'
    sep=""
    for repo in "${SIBLING_REPOS[@]}"; do
        printf '%s"%s": {"commit": "%s", "origin": "%s", "snapshot_digest": "%s"}' \
            "${sep}" "${repo}" "${SIBLING_COMMIT[${repo}]}" "${SIBLING_ORIGIN[${repo}]}" "${SIBLING_SNAPSHOT_DIGEST[${repo}]}"
        sep=", "
    done
    printf '},\n'
    printf '  "image_digest": "%s",\n' "${RUNNING_IMAGE_DIGEST}"
    printf '  "isolation": {"topic_namespace": "%s", "db_slot": "%s", "kafka_environment": "%s", "valkey_db_index": %s},\n' \
        "${TOPIC_NAMESPACE}" "${DB_SLOT}" "${KAFKA_ENV_TOKEN}" "${VALKEY_INDEX}"
    printf '  "ports": {"runtime_main": %s, "runtime_effects": %s, "gateway": %s, "projection_api": %s},\n' \
        "${MAIN_PORT}" "${EFFECTS_PORT}" "${GATEWAY_PORT}" "${PROJECTION_API_PORT}"
    printf '  "gateway_started": %s,\n' "$([[ "${WITH_GATEWAY}" == "1" ]] && echo true || echo false)"
    printf '  "staging_root": "%s",\n' "${STAGING_ROOT}"
    printf '  "reason": %s\n' "$("${PY}" -c 'import json,sys;print(json.dumps(sys.argv[1]))' "${REASON}")"
    printf '}\n'
} > "${DESCRIPTOR_OUT}"

log "slot ${SLOT} is READY. Descriptor: ${DESCRIPTOR_OUT}"
cat "${DESCRIPTOR_OUT}"
