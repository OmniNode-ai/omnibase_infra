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
    # OMN-18802 -- write a pack, never loose objects.
    #
    # git's default `fetch.unpackLimit` is 100: a fetch bringing fewer than
    # 100 objects is UNPACKED into individual loose object files instead of
    # being kept as a pack. This timer fires every ~2 minutes and almost
    # every pass brings a handful of objects, so the default meant this
    # mirror grew loose objects forever and packed almost nothing.
    #
    # By 2026-09-19 the onex_change_control mirror held 153,451 loose objects
    # / 4.52 GiB against 385 MiB of actual packed content, and serving a
    # clone out of that took 47-264s where omnimarket took 5s -- which is
    # what cancelled both OCC publisher jobs at their 5-minute budget
    # (OMN-18802). Setting the limit to 1 makes every fetch write a pack.
    git -C "$1" config fetch.unpackLimit 1
    git -C "$1" config transfer.unpackLimit 1
}

# OMN-18802 -- consolidate packs when they have accumulated.
#
# WHY THIS IS HERE AND NOT IN A RUNBOOK. The clone branch below sets
# `gc.auto 0` with the comment "`git gc` is run explicitly by the runbook",
# and the runbook it defers to -- docs/runbooks/c2-git-mirror-egress-rollout.md,
# also the `Documentation=` target of omninode-git-mirror-daemon.service --
# has never existed in this repository. So nothing has ever repacked these
# mirrors, and "the runbook does it" read as coverage for four months. A
# mechanism that exists only as a reference to an absent document is not a
# mechanism.
#
# `gc.auto 0` stays, and this is deliberately NOT `git gc`: the original
# concern was real, that repacking 72 job-serving mirrors on an unpredictable
# schedule would spike host IO mid-wave. What runs instead is bounded and
# conditional:
#
#   * it runs INSIDE the flock this script already holds, so it can never
#     overlap a refresh or another maintenance pass;
#   * it runs only when the pack count crosses a threshold, so the steady
#     state is a no-op;
#   * `nice`/`ionice` keep it behind live CI for the host's IO;
#   * `repack -a -d` plus `prune-packed` only MOVE reachable objects from
#     loose files into a pack. Nothing unreachable is expired and no history
#     is dropped, so a concurrent `upload-pack` reader cannot lose an object
#     out from under it -- unlike `git gc --prune`, which is why that is not
#     what this runs.
MAINTENANCE_PACK_THRESHOLD="${OMNI_GIT_MIRROR_PACK_THRESHOLD:-12}"

maintain_mirror_packs() {
    local mirror_dir="$1" repo="$2"
    local pack_count

    pack_count="$(find "${mirror_dir}/objects/pack" -maxdepth 1 -name '*.pack' 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "${pack_count}" -lt "${MAINTENANCE_PACK_THRESHOLD}" ]]; then
        return 0
    fi

    local loose_before loose_after
    loose_before="$(git -C "${mirror_dir}" count-objects | awk '{print $1}')"
    echo "[git-mirror-refresh] ${repo}: ${pack_count} packs >= ${MAINTENANCE_PACK_THRESHOLD}, ${loose_before} loose -- repacking"

    # Resolved rather than assumed: a missing `ionice` would otherwise make
    # every repack look like a repack FAILURE, which reads as a corrupt
    # mirror rather than a missing binary.
    local -a niced=()
    command -v nice >/dev/null 2>&1 && niced+=(nice -n 19)
    if command -v ionice >/dev/null 2>&1; then
        niced+=(ionice -c3)
    else
        echo "[git-mirror-refresh] ${repo}: ionice absent -- repacking without IO nicing" >&2
    fi

    if ! "${niced[@]}" git -C "${mirror_dir}" repack -a -d -q; then
        # A failed repack leaves the mirror exactly as it was -- repack writes
        # the new pack before dropping the old ones -- so this degrades, it
        # does not corrupt. Report and carry on rather than failing the timer.
        echo "[git-mirror-refresh] ${repo}: REPACK FAILED (mirror unchanged and still serving)" >&2
        return 1
    fi
    "${niced[@]}" git -C "${mirror_dir}" prune-packed -q || true

    loose_after="$(git -C "${mirror_dir}" count-objects | awk '{print $1}')"
    echo "[git-mirror-refresh] ${repo}: repacked -- loose ${loose_before} -> ${loose_after}, packs $(find "${mirror_dir}/objects/pack" -maxdepth 1 -name '*.pack' 2>/dev/null | wc -l | tr -d ' ')"
    return 0
}

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
        maintain_mirror_packs "${mirror_dir}" "${repo}" || rc=1
    else
        # A failed refresh is NOT fatal to CI: the runner-side pre-seed is
        # fail-open and a stale mirror still supplies almost every object.
        echo "[git-mirror-refresh] ${repo}: FETCH FAILED (mirror left stale; CI degrades, does not fail)" >&2
        rc=1
    fi
done

echo "[git-mirror-refresh] end $(date -Is) rc=${rc}"
exit "${rc}"
