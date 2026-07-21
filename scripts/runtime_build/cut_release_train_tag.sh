#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# cut_release_train_tag.sh -- OMN-14889: cut + push a git-tag release-train
# marker for the dev/stability lab lanes.
#
# Runtime updates go through release trains; deploying to a lab lane is
# "push a tag" (operator directive, 2026-07-20). This script is the CUT
# half of that mechanism -- it is the tag-cutting block of cut-lab-ref.sh's
# compute_lab_tag_name()/cut_lab_tags() (OMN-14438/RT-2), extracted verbatim,
# extended with exactly ONE new capability cut-lab-ref.sh does not have:
# pushing the tag to GitHub so a `push: tags:` workflow trigger can fire.
#
# Tag scheme: lab/<lane>/<utc>-<shortsha>, lane in {dev, stability}. The tag
# is cut LOCALLY on all 5 sibling clones at each clone's own resolved SHA for
# <ref> (so a single tag name resolves in all 5 repos even though the ref
# string itself never existed in the other 4) but PUSHED TO GITHUB ON THE
# ANCHOR REPO ONLY (omnibase_infra) -- the deploy step's own checkout only
# needs the other 4 repos' tags to exist locally (git checkout <tag> works
# against a local-only tag), and pushing to all 5 GitHub repos would add
# write surface with no behavior difference (OMN-14889 Fork 2).
#
# This script does NOT deploy anything -- it only tags + pushes. The deploy
# step is a separate workflow job that reacts to the tag push (see
# .github/workflows/release-train-lab.yml) and calls refresh_stability_lane.sh
# (stability; unmodified) or refresh_dev_lane.sh (dev; OMN-14889).
#
# Default is a DRY-RUN plan; pass --execute to actually cut + push.
#
# Usage:
#   cut_release_train_tag.sh --lane dev|stability [--ref <branch|tag|sha>] [--execute]
#
# Exit codes:
#   0  plan printed (dry-run) or tag cut + pushed
#   1  usage / precondition error
#   2  unknown / unsupported lane

set -euo pipefail

# Siblings whose clones get the LOCAL tag. Mirrors LAB_REF_REPOS in
# cut-lab-ref.sh and SIBLING_REPOS in stage_workspace.sh.
TAG_REPOS=(
    "omnibase_infra"
    "omnibase_core"
    "omnibase_compat"
    "onex_change_control"
    "omnimarket"
)

# Only this repo's tag is pushed to GitHub (Fork 2 decision above).
readonly ANCHOR_REPO="omnibase_infra"

REF="origin/dev"
LANE=""
MODE="plan"

usage() {
    sed -n '4,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

log() { printf '[cut-release-train-tag] %s\n' "$*" >&2; }
err() { printf '[cut-release-train-tag] ERROR: %s\n' "$*" >&2; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lane)
            [[ -n "${2:-}" ]] || { err "--lane requires a value"; exit 1; }
            LANE="$2"; shift 2 ;;
        --ref)
            [[ -n "${2:-}" ]] || { err "--ref requires a value"; exit 1; }
            REF="$2"; shift 2 ;;
        --execute)
            MODE="execute"; shift ;;
        --help|-h)
            usage 0 ;;
        *)
            err "unknown option: $1"; usage 1 ;;
    esac
done

case "${LANE}" in
    dev|stability) ;;
    "")
        err "--lane is required (dev or stability)."
        exit 1 ;;
    *)
        err "unknown lane '${LANE}'; expected dev or stability."
        err "  (prod is Train 2, PyPI-backed, grant-gated -- never a lab tag.)"
        exit 2 ;;
esac

OMNI_HOME="${OMNI_HOME:-}"
if [[ -z "${OMNI_HOME}" ]]; then
    err "OMNI_HOME must be set (the sibling clones under it are tagged)."
    exit 1
fi

# --- compute the tag name: lab/<lane>/<utc>-<shortsha> ----------------------
# Resolved against the ANCHOR repo's <ref> SHA (matches cut-lab-ref.sh's
# compute_lab_tag_name(), which always short-shas off the omnibase_infra
# clone regardless of which sibling triggered the cut).
compute_tag_name() {
    local infra_clone="${OMNI_HOME}/${ANCHOR_REPO}"
    local utc short
    utc="$(date -u +%Y%m%dT%H%M%SZ)"
    short="$(git -C "${infra_clone}" rev-parse --short=12 "${REF}^{commit}" 2>/dev/null || echo "unknown")"
    echo "lab/${LANE}/${utc}-${short}"
}

TAG="$(compute_tag_name)"

log "lane   : ${LANE}"
log "ref    : ${REF}"
log "tag    : ${TAG}"
log "mode   : ${MODE}"

if [[ "${MODE}" != "execute" ]]; then
    log "dry-run: no tags cut, nothing pushed. Re-run with --execute."
    log "would cut this tag locally on: ${TAG_REPOS[*]}"
    log "would push to GitHub on: ${ANCHOR_REPO} only"
    exit 0
fi

# --- execute: cut locally on all 5, push on the anchor only -----------------
for repo in "${TAG_REPOS[@]}"; do
    clone="${OMNI_HOME}/${repo}"
    if ! git -C "${clone}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        log "skip tag ${repo}: not a git clone at ${clone}"
        continue
    fi
    sha="$(git -C "${clone}" rev-parse "${REF}^{commit}")"
    git -C "${clone}" tag -f "${TAG}" "${sha}" >/dev/null
    log "tagged ${repo}: ${TAG} -> ${sha:0:12}"
done

anchor_clone="${OMNI_HOME}/${ANCHOR_REPO}"
log "pushing ${TAG} to origin (${ANCHOR_REPO} only)..."
git -C "${anchor_clone}" push origin "refs/tags/${TAG}"
log "pushed. This push is what fires .github/workflows/release-train-lab.yml's deploy job."
