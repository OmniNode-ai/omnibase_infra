#!/usr/bin/env bash
# OMN-14027 C2 — serialized refresh of the local bare git mirrors on the runner host.
#
# WHY THIS EXISTS
# 72 self-hosted runners NAT through one home uplink. Before this, every job's
# `actions/checkout` cold-cloned the full repo from github.com, because
# runner-job-started.sh `rm -rf`s the workspace before each job (a correctness
# fix for stale sparse-checkout state) and therefore guarantees a from-scratch
# clone every single time. Under wave load that is 72 simultaneous full clones
# of the same repos across one uplink -> `RPC failed; curl 56 GnuTLS recv
# error` / `fatal: early EOF` (the C2 failure class in the OMN-14027 design).
#
# This script performs exactly ONE upstream fetch per repo per interval. The
# runner-side pre-seed (runner-job-started.sh) then hydrates each job's
# workspace from the local mirror over the docker bridge, so the subsequent
# authenticated `actions/checkout` fetch from github.com is a small delta
# instead of the whole object graph.
#
# SERIALIZATION is the whole point and is enforced twice:
#   1. flock on ${MIRROR_ROOT}/.refresh.lock  -- no two refreshes ever overlap.
#   2. systemd `Type=oneshot` -- the timer will not start a new run while the
#      previous one is still active.
#
# Run as the runner-host user (jonah) via omninode-git-mirror-refresh.timer.
# Auth: the host's global git credential helper (`gh auth git-credential`)
# supplies the token for the private repos; no secret is stored here.

set -euo pipefail

MIRROR_ROOT="${OMNI_GIT_MIRROR_ROOT:-${HOME}/.omnibase/runners/git-mirror}"
GITHUB_ORG="${OMNI_GIT_MIRROR_ORG:-OmniNode-ai}"
FETCH_TIMEOUT_SECONDS="${OMNI_GIT_MIRROR_FETCH_TIMEOUT:-600}"

# High-traffic repos, in descending order of CI job volume. Adding a repo here
# is the only change needed to mirror it -- the runner-side pre-seed keys off
# GITHUB_REPOSITORY and silently no-ops for repos that have no mirror.
MIRROR_REPOS=(
    onex_change_control
    omnibase_infra
    omnibase_core
    omnimarket
    omniclaude
    omniweb
    omnimemory
    omnibase_compat
    knowledge-base
)

mkdir -p "${MIRROR_ROOT}"

lock_file="${MIRROR_ROOT}/.refresh.lock"
exec 9>"${lock_file}"
if ! flock -n 9; then
    echo "[git-mirror-refresh] another refresh holds the lock; exiting (this is the serialization guarantee, not an error)."
    exit 0
fi

started_at="$(date -Is)"
echo "[git-mirror-refresh] start ${started_at} root=${MIRROR_ROOT}"

rc=0
# OMN-16063 C2b -- serving settings every mirror must carry.
#
# allowFilter: the runner-side rewrite gate (wire_uv_git_mirror_rewrite in
#   runner-job-started.sh) proves a pinned commit is servable with a
#   `--filter=tree:0 --depth=1` fetch before it redirects uv at this daemon.
#   Without allowFilter that probe fails, the gate reads "absent", and every
#   job silently forgoes the mirror and clones from github.com instead.
# allowAnySHA1InWant: uv fetches an exact pinned SHA, which is ordinarily a
#   mid-history commit rather than a ref tip. Without this, upload-pack
#   refuses the request as "not our ref" even though the object is right here.
#
# Applied on every pass, not just at clone time: these are cheap idempotent
# writes, and applying them unconditionally means a mirror that was cloned
# before this existed -- or re-cloned by the branch below -- cannot end up
# quietly unable to serve the thing the whole component depends on.
apply_mirror_serving_config() {
    git -C "$1" config uploadpack.allowFilter true
    git -C "$1" config uploadpack.allowAnySHA1InWant true
}

for repo in "${MIRROR_REPOS[@]}"; do
    mirror_dir="${MIRROR_ROOT}/${repo}.git"
    upstream="https://github.com/${GITHUB_ORG}/${repo}.git"

    if [[ ! -d "${mirror_dir}" ]]; then
        echo "[git-mirror-refresh] ${repo}: no mirror yet -- cloning"
        if timeout "${FETCH_TIMEOUT_SECONDS}" git clone --mirror "${upstream}" "${mirror_dir}.tmp"; then
            mv "${mirror_dir}.tmp" "${mirror_dir}"
            git -C "${mirror_dir}" config remote.origin.fetch '+refs/*:refs/*'
            # Repacking 72 job-serving mirrors on an unpredictable schedule would
            # spike host IO mid-wave; `git gc` is run explicitly by the runbook.
            git -C "${mirror_dir}" config gc.auto 0
            apply_mirror_serving_config "${mirror_dir}"
        else
            echo "[git-mirror-refresh] ${repo}: CLONE FAILED" >&2
            rm -rf "${mirror_dir}.tmp"
            rc=1
        fi
        continue
    fi

    apply_mirror_serving_config "${mirror_dir}"

    # --prune keeps deleted branches/PR refs from accumulating forever.
    if timeout "${FETCH_TIMEOUT_SECONDS}" git -C "${mirror_dir}" fetch --quiet --prune origin '+refs/*:refs/*'; then
        head_sha="$(git -C "${mirror_dir}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
        echo "[git-mirror-refresh] ${repo}: ok head=${head_sha}"
    else
        # A failed refresh is NOT fatal to CI: the runner-side pre-seed is
        # fail-open and a stale mirror still supplies almost every object.
        echo "[git-mirror-refresh] ${repo}: FETCH FAILED (mirror left stale; CI degrades, does not fail)" >&2
        rc=1
    fi
done

echo "[git-mirror-refresh] end $(date -Is) rc=${rc}"
exit "${rc}"
