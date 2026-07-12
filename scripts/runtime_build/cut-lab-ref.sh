#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# cut-lab-ref.sh -- RT-2 (OMN-14438): one-command git-ref deploy to a lab lane.
#
# The lab fast lane (Train 1, mechanical-release-trains plan §3) needs a single
# mechanical command that gets an EXACT named ref of code running in the dev /
# stability lane and fails loudly when it deploys nothing. This wrapper:
#
#   (a) optionally cuts a lightweight dev tag  lab/<lane>/<utc>-<shortsha>  at the
#       chosen ref in each sibling clone (a reproducible marker, NOT a v* release
#       tag -- Train 1 never touches PyPI),
#   (b) engages RT-1's clean-ref checkout + vendored-SHA assertion by exporting
#       DEPLOY_REF (or DEPLOY_HOTPATCH) into the workspace build, and
#   (c) builds + deploys the resulting image to the target lab lane via
#       deploy-runtime.sh.
#
# Supports --ref <branch|tag|sha> and --hotpatch (deploy a dirty tree
# deliberately, LABELLED as such in the manifest -- a hot-patch is labelled, never
# laundered). This wrapper drives the lab lanes (dev / stability-test) only; the
# grant-gated prod lane is Train 2 and is refused here.
#
# Default is a DRY-RUN plan; pass --execute to actually build + deploy.
#
# Usage:
#   cut-lab-ref.sh [--ref <ref>] [--lane dev|stability-test] [--hotpatch]
#                  [--cut-tag] [--cold|--restart] [--execute]
#
# Exit codes:
#   0  plan printed (dry-run) or deploy succeeded
#   1  usage / precondition error
#   2  unknown / unsupported lane (e.g. prod)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# Overridable so an operator can point at a relocated deploy-runtime.sh (and so
# the execute path is exercisable in tests without a real Docker deploy).
DEPLOY_RUNTIME="${DEPLOY_RUNTIME:-${REPO_ROOT}/scripts/deploy-runtime.sh}"

# Siblings whose clones are checked out to <ref> and (optionally) lab-tagged.
# Mirrors SIBLING_REPOS in stage_workspace.sh plus the infra build-context repo.
LAB_REF_REPOS=(
    "omnibase_infra"
    "omnibase_core"
    "omnibase_compat"
    "onex_change_control"
    "omnimarket"
)

# --- defaults -------------------------------------------------------------
REF="origin/dev"
LANE="dev"
HOTPATCH=false
CUT_TAG=false
BRINGUP="--restart"   # warm refresh by default; --cold for a full bring-up
MODE="plan"           # plan | execute

usage() {
    sed -n '4,33p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

log() { printf '[cut-lab-ref] %s\n' "$*" >&2; }
err() { printf '[cut-lab-ref] ERROR: %s\n' "$*" >&2; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ref)
            [[ -n "${2:-}" ]] || { err "--ref requires a value"; exit 1; }
            REF="$2"; shift 2 ;;
        --lane)
            [[ -n "${2:-}" ]] || { err "--lane requires a value"; exit 1; }
            LANE="$2"; shift 2 ;;
        --hotpatch)
            HOTPATCH=true; shift ;;
        --cut-tag)
            CUT_TAG=true; shift ;;
        --cold)
            BRINGUP="--cold"; shift ;;
        --restart)
            BRINGUP="--restart"; shift ;;
        --execute)
            MODE="execute"; shift ;;
        --help|-h)
            usage 0 ;;
        *)
            err "unknown option: $1"; usage 1 ;;
    esac
done

# --- lane -> compose project (mirrors deploy-runtime.sh lane mapping) ------
case "${LANE}" in
    dev)
        COMPOSE_PROJECT="omnibase-infra" ;;
    stability-test)
        COMPOSE_PROJECT="omnibase-infra-stability-test" ;;
    prod|judge)
        err "lane '${LANE}' is not a lab fast-lane target."
        err "  prod is Train 2 (grant-gated, PyPI-backed); judge is read-only."
        err "  cut-lab-ref drives dev / stability-test only."
        exit 2 ;;
    *)
        err "unknown lane '${LANE}'; expected dev or stability-test."
        exit 2 ;;
esac

# --- OMNI_HOME (required: the sibling clones are the build source) ---------
OMNI_HOME="${OMNI_HOME:-}"
if [[ -z "${OMNI_HOME}" ]]; then
    err "OMNI_HOME must be set (the sibling clones under it are the build source)."
    exit 1
fi

if [[ "${HOTPATCH}" == true ]]; then
    DEPLOY_HOTPATCH_VAL="1"
else
    DEPLOY_HOTPATCH_VAL="0"
fi

# --- optional lab tag: lab/<lane>/<utc>-<shortsha> ------------------------
# A lightweight, local (un-pushed) marker cut at <ref> in each sibling clone so
# this exact lab deploy is redeployable later via --ref <that-tag>. The tag name
# carries the infra repo's short SHA + a UTC stamp; the same name is cut in every
# sibling at that sibling's own resolved <ref> SHA.
LAB_TAG=""
compute_lab_tag_name() {
    local infra_clone="${OMNI_HOME}/omnibase_infra"
    local utc short
    utc="$(date -u +%Y%m%dT%H%M%SZ)"
    if [[ "${HOTPATCH}" == true ]]; then
        short="$(git -C "${infra_clone}" rev-parse --short=12 HEAD 2>/dev/null || echo "HEAD")"
    else
        short="$(git -C "${infra_clone}" rev-parse --short=12 "${REF}^{commit}" 2>/dev/null || echo "unknown")"
    fi
    echo "lab/${LANE}/${utc}-${short}"
}

cut_lab_tags() {
    local tag="$1"
    local repo clone sha
    for repo in "${LAB_REF_REPOS[@]}"; do
        clone="${OMNI_HOME}/${repo}"
        if [[ ! -d "${clone}/.git" ]] && ! git -C "${clone}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
            log "skip tag ${repo}: not a git clone at ${clone}"
            continue
        fi
        if [[ "${HOTPATCH}" == true ]]; then
            sha="$(git -C "${clone}" rev-parse HEAD)"
        else
            sha="$(git -C "${clone}" rev-parse "${REF}^{commit}")"
        fi
        git -C "${clone}" tag -f "${tag}" "${sha}" >/dev/null
        log "tagged ${repo}: ${tag} -> ${sha:0:12}"
    done
}

# --- build the plan -------------------------------------------------------
log "lane            : ${LANE} (compose project ${COMPOSE_PROJECT})"
if [[ "${HOTPATCH}" == true ]]; then
    log "ref             : <hotpatch: current HEAD, dirty tree deployed AS-IS>"
else
    log "ref             : ${REF}"
fi
log "bring-up        : ${BRINGUP}"
log "cut lab tag     : ${CUT_TAG}"
log "mode            : ${MODE}"

if [[ "${CUT_TAG}" == true ]]; then
    LAB_TAG="$(compute_lab_tag_name)"
    log "lab tag         : ${LAB_TAG}"
fi

# Environment RT-1 needs: workspace build + the ref that engages clean-checkout
# + assertion in stage_workspace.sh, plus the lane's compose project.
PLAN_ENV=(
    "OMNI_HOME=${OMNI_HOME}"
    "OMNIBASE_INFRA_COMPOSE_PROJECT=${COMPOSE_PROJECT}"
    "BUILD_SOURCE=workspace"
    "DEPLOY_HOTPATCH=${DEPLOY_HOTPATCH_VAL}"
)
if [[ "${HOTPATCH}" != true ]]; then
    PLAN_ENV+=("DEPLOY_REF=${REF}")
fi

log "deploy command  :"
log "  ${PLAN_ENV[*]} \\"
log "    ${DEPLOY_RUNTIME} --execute ${BRINGUP}"

if [[ "${MODE}" != "execute" ]]; then
    log "dry-run: no tags cut, no build/deploy performed. Re-run with --execute."
    exit 0
fi

# --- execute --------------------------------------------------------------
if [[ ! -x "${DEPLOY_RUNTIME}" && ! -f "${DEPLOY_RUNTIME}" ]]; then
    err "deploy-runtime.sh not found at ${DEPLOY_RUNTIME}"
    exit 1
fi

if [[ "${CUT_TAG}" == true ]]; then
    cut_lab_tags "${LAB_TAG}"
fi

export OMNI_HOME
export OMNIBASE_INFRA_COMPOSE_PROJECT="${COMPOSE_PROJECT}"
export BUILD_SOURCE="workspace"
export DEPLOY_HOTPATCH="${DEPLOY_HOTPATCH_VAL}"
if [[ "${HOTPATCH}" != true ]]; then
    export DEPLOY_REF="${REF}"
fi

log "executing deploy-runtime.sh ..."
exec bash "${DEPLOY_RUNTIME}" --execute "${BRINGUP}"
