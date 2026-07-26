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
# creating the tag ref on GitHub so a `push: tags:` workflow trigger can fire.
#
# Tag scheme: lab/<lane>/<utc>-<shortsha>, lane in {dev, stability}. The tag
# is cut LOCALLY on all 5 sibling clones (so a single tag name resolves in
# all 5 repos) but CREATED ON GITHUB ON THE ANCHOR REPO ONLY
# (omnibase_infra) -- the deploy step's own checkout only needs the other 4
# repos' tags to exist locally (git checkout <tag> works against a
# local-only tag), and pushing to all 5 GitHub repos would add write surface
# with no behavior difference (OMN-14889 Fork 2).
#
# REF RESOLUTION (OMN-14956): <ref> is resolved in the ANCHOR clone
# (omnibase_infra); it is the only repo whose commit space <ref> is
# guaranteed to live in. Each SIBLING clone tags at its own resolution of
# <ref> when that resolves there (branch/tag names like origin/dev do), and
# otherwise FALLS BACK, loudly, to the sibling's own origin/dev -- a raw
# omnibase_infra commit SHA can never resolve in the other 4 repos, and the
# pre-fix per-sibling `rev-parse <ref>` died exit 128 at omnibase_core
# despite usage advertising `--ref <branch|tag|sha>` (run 29977699589).
#
# OMN-14900: the GitHub-side tag is created via `gh api .../git/refs` (the
# workflow token, contents:write), NOT `git push` from a clone -- the deploy
# runner's private OMNI_HOME clones are cloned anonymously from public repos
# and carry NO push credentials, and the old ambient-clone push only ever
# worked from an operator workstation. All local git operations are scoped
# with `-c safe.directory=<clone>` (see git_clone below).
#
# This script does NOT deploy anything -- it only tags + creates the GitHub
# ref, and (when $GITHUB_OUTPUT is set) publishes the cut tag name as a step
# output (`tag=<TAG>`) for the calling workflow.
#
# TRIGGER REALITY (OMN-14957): a ref created with the workflow's own
# GITHUB_TOKEN does NOT fire `push: tags:` workflows -- GitHub suppresses
# event delivery for default-token mutations (documented anti-recursion
# behavior; OMN-9426 class). release-train-lab.yml therefore CHAINS its
# deploy job onto the cut-tag job in the SAME run (needs: cut-tag, keyed off
# the `tag` output above). The `push: tags:` trigger remains ONLY for tags
# cut/pushed with non-GITHUB_TOKEN credentials (e.g. an operator
# workstation). The deploy job calls refresh_stability_lane.sh (stability)
# or refresh_dev_lane.sh (dev).
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Siblings whose clones get the LOCAL tag. Mirrors LAB_REF_REPOS in
# cut-lab-ref.sh and SIBLING_REPOS in stage_workspace.sh.
TAG_REPOS=(
    "omnibase_infra"
    "omnibase_core"
    "omnibase_compat"
    "onex_change_control"
    "omnimarket"
)

# Only this repo's tag is created on GitHub (Fork 2 decision above).
readonly ANCHOR_REPO="omnibase_infra"
readonly ANCHOR_GITHUB_REPO="OmniNode-ai/omnibase_infra"

REF="origin/dev"
LANE=""
MODE="plan"

usage() {
    sed -n '4,61p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
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

# --- scoped git wrapper (OMN-14900) -----------------------------------------
# The clones under OMNI_HOME may be owned by a different uid than the one
# running this script (the deploy runner container case). An unscoped git
# invocation dies with "detected dubious ownership"; `-c
# safe.directory=<clone>` scopes the exemption to one process + one path with
# no global gitconfig write -- durable across container recreation because it
# is committed here, not exec-applied container state. (Same wrapper as
# refresh_dev_lane.sh.)
git_clone() {
    local clone="$1"
    shift
    git -c "safe.directory=${clone}" -C "${clone}" "$@"
}

# --- execute-mode preflight: provision + refresh the private clones ---------
# Runs BEFORE computing the tag name so the tag's <shortsha> is minted from a
# freshly-fetched origin/<ref>, never a stale private clone. Dry-run stays
# read-only (no clone/fetch) -- the printed tag name is an estimate.
if [[ "${MODE}" == "execute" ]]; then
    command -v gh >/dev/null 2>&1 || {
        err "'gh' is required to create the GitHub tag ref (repos/${ANCHOR_GITHUB_REPO}/git/refs)."
        exit 1
    }
    OMNI_HOME="${OMNI_HOME}" bash "${SCRIPT_DIR}/ensure_runner_clones.sh"
    for repo in "${TAG_REPOS[@]}"; do
        git_clone "${OMNI_HOME}/${repo}" fetch origin --prune --tags
    done
fi

# --- compute the tag name: lab/<lane>/<utc>-<shortsha> ----------------------
# Resolved against the ANCHOR repo's <ref> SHA (matches cut-lab-ref.sh's
# compute_lab_tag_name(), which always short-shas off the omnibase_infra
# clone regardless of which sibling triggered the cut).
compute_tag_name() {
    local infra_clone="${OMNI_HOME}/${ANCHOR_REPO}"
    local utc short
    utc="$(date -u +%Y%m%dT%H%M%SZ)"
    short="$(git_clone "${infra_clone}" rev-parse --short=12 "${REF}^{commit}" 2>/dev/null || echo "unknown")"
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
    log "would create the GitHub ref on: ${ANCHOR_GITHUB_REPO} only (via gh api)"
    exit 0
fi

# --- execute: cut locally on all 5, create the GitHub ref on the anchor -----
# OMN-14956: resolve --ref in the ANCHOR clone FIRST, with a named error --
# the anchor is the only repo whose commit space --ref is guaranteed to live
# in (usage advertises <branch|tag|sha>, and a raw sha only exists here).
ANCHOR_CLONE="${OMNI_HOME}/${ANCHOR_REPO}"
if ! ANCHOR_SHA="$(git_clone "${ANCHOR_CLONE}" rev-parse "${REF}^{commit}" 2>/dev/null)"; then
    err "--ref '${REF}' does not resolve to a commit in the anchor clone (${ANCHOR_CLONE})."
    err "  Pass a ref that exists in ${ANCHOR_REPO}: a branch (origin/dev), a tag, or a"
    err "  full/short ${ANCHOR_REPO} commit SHA reachable from a fetched ref."
    exit 1
fi

# Sibling fallback ref used when --ref does not resolve in a sibling clone
# (always the case for a raw anchor SHA). Freshly fetched by the execute-mode
# preflight above.
readonly SIBLING_FALLBACK_REF="origin/dev"

for repo in "${TAG_REPOS[@]}"; do
    clone="${OMNI_HOME}/${repo}"
    if ! git_clone "${clone}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        if [[ "${repo}" == "${ANCHOR_REPO}" ]]; then
            err "anchor clone at ${clone} is not a git work tree; refusing to continue."
            exit 1
        fi
        log "skip tag ${repo}: not a git clone at ${clone}"
        continue
    fi
    if [[ "${repo}" == "${ANCHOR_REPO}" ]]; then
        sha="${ANCHOR_SHA}"
    elif sha="$(git_clone "${clone}" rev-parse "${REF}^{commit}" 2>/dev/null)"; then
        : # --ref resolves in this sibling (branch/tag name) -- use it as-is.
    elif sha="$(git_clone "${clone}" rev-parse "${SIBLING_FALLBACK_REF}^{commit}" 2>/dev/null)"; then
        log "NOTE ${repo}: --ref '${REF}' does not resolve here (anchor-only ref, e.g. a raw"
        log "  ${ANCHOR_REPO} SHA) -- falling back to this sibling's own ${SIBLING_FALLBACK_REF} (${sha:0:12})."
    else
        err "${repo}: neither --ref '${REF}' nor fallback '${SIBLING_FALLBACK_REF}' resolves in ${clone}."
        err "  The clone is present but has no usable ref -- re-provision it (ensure_runner_clones.sh)"
        err "  or fetch its origin before cutting a release-train tag."
        exit 1
    fi
    git_clone "${clone}" tag -f "${TAG}" "${sha}" >/dev/null
    log "tagged ${repo}: ${TAG} -> ${sha:0:12}"
done

log "creating refs/tags/${TAG} on GitHub (${ANCHOR_GITHUB_REPO}) at ${ANCHOR_SHA:0:12} via gh api..."
gh api "repos/${ANCHOR_GITHUB_REPO}/git/refs" \
    -f ref="refs/tags/${TAG}" \
    -f sha="${ANCHOR_SHA}" >/dev/null
# OMN-14957: this GITHUB_TOKEN-created ref does NOT fire `push: tags:`
# workflows (GitHub suppresses default-token event delivery). The deploy job
# is chained in-run by release-train-lab.yml via the step output below; the
# push trigger only fires for refs created with non-GITHUB_TOKEN credentials.
log "created. NOTE: a GITHUB_TOKEN-created ref fires NO push:tags workflow -- the"
log "  release-train-lab.yml deploy job runs chained in this same workflow run."
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    echo "tag=${TAG}" >> "${GITHUB_OUTPUT}"
fi
