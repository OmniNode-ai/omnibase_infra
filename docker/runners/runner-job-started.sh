#!/usr/bin/env bash
# Reset the current repository workspace before each self-hosted runner job.
#
# Stateful self-hosted runners can inherit sparse-checkout and partial worktree
# state from earlier jobs on the same runner. GitHub-hosted runners avoid this
# by starting each job on a fresh VM; this hook gives the Docker runner fleet
# the same repository-workspace invariant before actions/checkout runs.
#
# OMN-15134: this hook runs as the unprivileged `runner` user (the same
# identity every job step executes as -- entrypoint.sh gosu-drops from root
# before run.sh starts). A plain `rm -rf` on its own workspace tree normally
# suffices. It can still fail with EACCES if a PRIOR, OUT-OF-BAND root-
# privileged mutation of this container (verified root cause: a manual
# `docker exec omninode-deploy-runner ...` issued without `-u runner` --
# this image's ENTRYPOINT legitimately starts as root for the docker-socket
# GID fix, so the container's default exec identity is root, and any ad hoc
# bare `docker exec` silently inherits it) left root-owned files under the
# job workspace that `runner` cannot delete. Rather than hand-clean via
# another ad hoc root `docker exec` every time this recurs, fail loudly with
# the exact offending paths, then self-heal via the narrowly scoped NOPASSWD
# sudo rule installed in the Dockerfile (one command, one argument pattern,
# confined to this runner's own `_work` tree -- never a general root shell).
# If even that fails, this hook still exits non-zero: it never silently
# leaves a job to run against an unreset workspace.

set -euo pipefail

# ---------------------------------------------------------------------------
# OMN-14027 C2 -- local git-mirror pre-seed (fail-open by construction)
# ---------------------------------------------------------------------------
#
# THE PROBLEM THIS SOLVES. The workspace reset above is correct but it has a
# cost: because the workspace is destroyed before every job, `actions/checkout`
# can never reuse an existing object store, so EVERY job cold-clones the entire
# repository from github.com. With 72 runners on one home uplink that is 72
# simultaneous full clones of the same five repos, which is what produces the
# `RPC failed; curl 56 GnuTLS recv error (-54)` / `fatal: early EOF` kills.
#
# THE FIX. Re-create the "warm workspace" that checkout knows how to exploit,
# but hydrate it from a bare mirror served over the docker bridge instead of
# from stale on-disk state. After this function runs the workspace is a real
# git repo whose `remote.origin.url` is exactly the URL checkout expects, with
# HEAD detached at the mirror's default-branch tip.
#
# WHY THIS CANNOT FAIL A JOB, even with a stale mirror. The mirror is never a
# source of truth. checkout still fetches the exact requested SHA from
# github.com over its own authenticated remote; the only thing the pre-seed
# changes is that git's fetch negotiation now has a local "have" (the detached
# HEAD commit) to offer, so the server sends a small delta instead of the whole
# object graph. A mirror that is minutes behind costs a slightly larger delta.
# A mirror that is missing, unreachable, or corrupt costs nothing: every step
# below is guarded and returns 0, leaving the workspace exactly as this hook
# would have left it before this change.
#
# The detached checkout is load-bearing, not cosmetic: checkout's
# `prepareExistingDirectory` runs `git checkout --detach` on the existing repo
# and wipes the directory if that fails, which it would on an unborn HEAD.
_C2_MIRROR_HOST="${OMNI_GIT_MIRROR_HOST:-172.18.0.1}"
_C2_MIRROR_PORT="${OMNI_GIT_MIRROR_PORT:-9418}"
# Space-separated RUNNER_NAME allowlist, or ALL. Canary rollouts narrow this on
# the host copy of this file (it is bind-mounted, so no container recreate is
# needed); the committed default is the post-canary fleet-wide value.
_C2_MIRROR_RUNNERS="${OMNI_GIT_MIRROR_RUNNERS:-ALL}"
_C2_SEED_TIMEOUT="${OMNI_GIT_MIRROR_SEED_TIMEOUT:-180}"

seed_workspace_from_mirror() {
    local workspace_dir="$1"

    # NOTE: written as `if`, not `[[ ... ]] && return 0`. This script runs under
    # `set -e`; a bare AND-list whose left side is false returns non-zero and
    # would kill the hook -- i.e. fail every job -- the moment the kill switch
    # was NOT set. The kill switch must never be able to break the thing it is
    # there to protect.
    if [[ "${OMNI_GIT_MIRROR_DISABLE:-0}" == "1" ]]; then
        return 0
    fi
    [[ -n "${GITHUB_REPOSITORY:-}" ]] || return 0
    command -v git >/dev/null 2>&1 || return 0

    if [[ "${_C2_MIRROR_RUNNERS}" != "ALL" ]]; then
        case " ${_C2_MIRROR_RUNNERS} " in
            *" ${RUNNER_NAME:-} "*) ;;
            *) return 0 ;;
        esac
    fi

    # The runner invokes this hook with cwd == GITHUB_WORKSPACE, and the reset
    # above `rm -rf`s exactly that directory. `mkdir -p` then recreates the
    # PATH, but this shell's cwd is still the old, now-deleted inode -- so every
    # subsequent git invocation dies with
    #   fatal: Unable to read current working directory: No such file or directory
    # before it ever opens a connection. That is what produced the field
    # "no mirror" report on omnibase_infra job 94699283129 (07:07:05Z); it was
    # never a transient mirror outage. Re-enter the recreated directory (falling
    # back to / so this can never itself be the thing that fails).
    cd "${workspace_dir}" 2>/dev/null || cd / || return 0

    local repo_name="${GITHUB_REPOSITORY##*/}"
    local mirror_url="git://${_C2_MIRROR_HOST}:${_C2_MIRROR_PORT}/${repo_name}.git"
    local origin_url="${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY}"

    # Probe first so an unreachable/absent mirror costs ~1s, not a fetch timeout.
    # Two attempts: a single probe failure was observed in the field
    # (omnibase_infra, 2026-08-14T06:48:00Z) that was not reproducible seconds
    # later from the same container, so treat one miss as transient. The probe's
    # stderr is echoed rather than swallowed -- a silent "no mirror" line makes
    # the difference between "mirror is down" and "this repo is not mirrored"
    # undiagnosable from the job log, which is the only surface that matters
    # once the fleet is running unattended.
    local probe_err probe_ok=0 attempt
    probe_err="$(mktemp)"
    for attempt in 1 2; do
        if timeout 10 git ls-remote --heads "${mirror_url}" >/dev/null 2>"${probe_err}"; then
            probe_ok=1
            break
        fi
        sleep 1
    done
    if [[ "${probe_ok}" -ne 1 ]]; then
        echo "[c2-mirror] no usable mirror for ${repo_name} at ${mirror_url} after 2 probes; leaving workspace cold (fail-open path, job is unaffected). git said: $(tr '\n' ' ' <"${probe_err}")"
        rm -f "${probe_err}"
        return 0
    fi
    rm -f "${probe_err}"

    local seed_start seed_end
    seed_start="$(date +%s)"

    (
        set +e
        git init --quiet "${workspace_dir}" || exit 0
        # `origin` must match byte-for-byte what actions/checkout computes
        # (`${GITHUB_SERVER_URL}/${owner}/${repo}`, no .git suffix) or checkout
        # decides the directory belongs to a different repo and deletes it.
        git -C "${workspace_dir}" remote add origin "${origin_url}" 2>/dev/null \
            || git -C "${workspace_dir}" remote set-url origin "${origin_url}"
        # Branch heads + tags only. refs/pull/* is deliberately NOT fetched:
        # the objects are shared with the branch graph anyway, and the ref
        # explosion would cost more than it saves.
        timeout "${_C2_SEED_TIMEOUT}" git -C "${workspace_dir}" fetch --quiet --prune "${mirror_url}" \
            '+refs/heads/*:refs/remotes/origin/*' '+refs/tags/*:refs/tags/*' || exit 0

        # Anchor HEAD on the mirror's default branch so checkout's detach
        # succeeds and so fetch negotiation has a "have" to offer.
        local head_ref head_sha
        head_ref="$(timeout 10 git ls-remote --symref "${mirror_url}" HEAD 2>/dev/null | awk '/^ref:/ {print $2; exit}')"
        head_sha=""
        if [[ -n "${head_ref}" ]]; then
            head_sha="$(git -C "${workspace_dir}" rev-parse --verify --quiet "refs/remotes/origin/${head_ref##refs/heads/}")"
        fi
        if [[ -z "${head_sha}" ]]; then
            for candidate in dev main master; do
                head_sha="$(git -C "${workspace_dir}" rev-parse --verify --quiet "refs/remotes/origin/${candidate}")"
                [[ -n "${head_sha}" ]] && break
            done
        fi
        [[ -n "${head_sha}" ]] || exit 0
        timeout "${_C2_SEED_TIMEOUT}" git -C "${workspace_dir}" checkout --quiet --detach "${head_sha}" || exit 0
        exit 0
    ) || true

    seed_end="$(date +%s)"
    if git -C "${workspace_dir}" rev-parse --verify --quiet HEAD >/dev/null 2>&1; then
        echo "[c2-mirror] pre-seeded ${repo_name} from ${mirror_url} at $(git -C "${workspace_dir}" rev-parse --short HEAD) in $((seed_end - seed_start))s -- checkout will fetch a delta, not a full clone."
    else
        echo "[c2-mirror] pre-seed of ${repo_name} did not complete; leaving workspace cold (fail-open)."
    fi
    return 0
}

# ---------------------------------------------------------------------------
# OMN-16063 C2b -- route uv's git-dependency clone at the local mirror
# ---------------------------------------------------------------------------
#
# THE GAP THIS CLOSES. The C2 pre-seed above only warms the workspace for the
# job's OWN repository, so it only accelerates actions/checkout. It does
# nothing for the OTHER full clone every job performs: every `uv sync
# --no-cache` re-resolves
#   onex-change-control @ git+https://github.com/OmniNode-ai/onex_change_control.git@<sha>
# from github.com. That is a ~93MB object graph per invocation, ci.yml runs
# `uv sync` in 9 jobs, and `--no-cache` means uv resolves into a throwaway
# cache dir every time -- so there is nothing on the runner to reuse and no
# uv-level cache to pre-seed. Under wave load that is the dominant remaining
# source of the concurrent-clone/GnuTLS churn the C2 component exists to kill.
#
# THE MECHANISM: a fetch-only `url.<mirror>.insteadOf`, exported through
# GITHUB_ENV as GIT_CONFIG_* rather than written to any file. This matters:
#   - Nothing is written inside the 72 containers. The rewrite lives only in
#     the job's own environment and evaporates when the job ends, so reverting
#     is restoring this file -- there is no per-container residue to chase.
#   - It survives into uv, because uv shells out to `git` and git honours
#     GIT_CONFIG_COUNT/KEY/VALUE from the environment.
#
# WHY ONLY THE `.git`-SUFFIXED URL IS REWRITTEN. This is the load-bearing
# scoping decision, not an accident. uv fetches the URL exactly as pinned in
# uv.lock, which carries the `.git` suffix (verified from uv 0.6.14's own error
# output: `git fetch ... 'https://github.com/OmniNode-ai/onex_change_control.git'`).
# actions/checkout computes `${GITHUB_SERVER_URL}/${owner}/${repo}` with NO
# `.git` suffix (the same fact the pre-seed above depends on). Rewriting only
# the suffixed form therefore hits uv's clone and CANNOT touch any
# actions/checkout -- including the ci.yml steps that check out
# onex_change_control as a sibling repo, which keep their authenticated
# github.com fetch and their exact-ref semantics. There is no blanket
# github.com rewrite and no pushInsteadOf, so pushes, `gh` API calls and the
# checkout action's token path are all untouched.
#
# WHY THIS IS GATED ON THE EXACT PIN -- MEASURED, NOT ASSUMED. `insteadOf` has
# no fallback: if the rewritten remote cannot serve the requested object, git
# fails and uv propagates it. Verified 2026-08-14 in a throwaway container off
# omninode-runner:latest by pointing the rewrite at a mirror that lacks the
# pinned commit:
#     fatal: remote error: upload-pack: not our ref 2dd26ade...
#     -> uv exits 1. It does NOT retry against github.com.
# So an unconditional rewrite would convert "mirror is stale" into "job fails",
# which is strictly worse than the problem being solved. Instead this function
# resolves the pin the job will actually use (uv.lock at GITHUB_SHA, read via a
# delta fetch against the already-seeded workspace) and installs the rewrite
# ONLY if the mirror currently advertises that exact commit. Every other
# outcome -- no uv.lock, no pin, pin not advertised, mirror unreachable, any
# unexpected error -- installs nothing and leaves the job on the github.com
# path it uses today. The check is deliberately conservative: it matches
# advertised refs only, so a pinned commit that is real but not a ref tip is
# treated as a miss and simply forgoes the speedup.
_C2B_REWRITE_REPOS="${OMNI_GIT_MIRROR_REWRITE_REPOS:-onex_change_control}"

# ---------------------------------------------------------------------------
# Shared rewrite accumulator (OMN-16063 C2b + OMN-16114 C2c)
# ---------------------------------------------------------------------------
#
# GIT_CONFIG_COUNT is a single flat index namespace in the job's environment.
# The two mechanisms below (the uv git-dependency rewrite, and the OMN-16114
# sibling actions/checkout rewrite further down) each discover their own set
# of fetch-only insteadOf pairs to install. If each wrote its own
# `GIT_CONFIG_COUNT=N` line to GITHUB_ENV independently, the SECOND write
# would win -- GITHUB_ENV lines just set env vars, a later same-named write
# replaces rather than merges, silently dropping the first mechanism's
# GIT_CONFIG_KEY_0/VALUE_0 entries even though both lines are present in the
# file. Both mechanisms append to this shared accumulator instead; exactly
# one flush, after both have run, writes the combined set.
declare -a _C2_REWRITE_ENV_LINES=()
_C2_REWRITE_COUNT=0

# Adds one fetch-only redirect: fetches of $2 (upstream) resolve to $1
# (mirror); pushes to $2 stay pinned at $2 via an identity pushInsteadOf
# (git resolves pushInsteadOf before insteadOf, so this pins the push path
# back on github.com without touching the fetch redirect -- see the "Push is
# pinned back on github.com" note in docker/runners/README-c2b-uv-git-mirror.md
# for the field verification this relies on).
_c2_rewrite_add_pair() {
    local mirror_url="$1" upstream_url="$2"
    _C2_REWRITE_ENV_LINES+=("GIT_CONFIG_KEY_${_C2_REWRITE_COUNT}=url.${mirror_url}.insteadOf")
    _C2_REWRITE_ENV_LINES+=("GIT_CONFIG_VALUE_${_C2_REWRITE_COUNT}=${upstream_url}")
    _C2_REWRITE_COUNT=$((_C2_REWRITE_COUNT + 1))
    _C2_REWRITE_ENV_LINES+=("GIT_CONFIG_KEY_${_C2_REWRITE_COUNT}=url.${upstream_url}.pushInsteadOf")
    _C2_REWRITE_ENV_LINES+=("GIT_CONFIG_VALUE_${_C2_REWRITE_COUNT}=${upstream_url}")
    _C2_REWRITE_COUNT=$((_C2_REWRITE_COUNT + 1))
}

_c2_rewrite_flush() {
    if [[ "${_C2_REWRITE_COUNT}" -eq 0 ]]; then
        return 0
    fi
    if [[ -z "${GITHUB_ENV:-}" || ! -w "${GITHUB_ENV}" ]]; then
        echo "[c2-mirror-rewrite] GITHUB_ENV unwritable; ${_C2_REWRITE_COUNT} discovered rewrite pair(s) not applied (fail-open)."
        return 0
    fi
    {
        printf '%s\n' "${_C2_REWRITE_ENV_LINES[@]}"
        echo "GIT_CONFIG_COUNT=${_C2_REWRITE_COUNT}"
    } >>"${GITHUB_ENV}" 2>/dev/null || {
        echo "[c2-mirror-rewrite] GITHUB_ENV write failed; job unaffected (fail-open)."
        return 0
    }
    return 0
}

# Fetches GITHUB_SHA into the pre-seeded own-repo workspace, once, so both
# mechanisms below can read files (uv.lock, workflow YAML) EXACTLY as they
# exist at the commit this job builds -- not the mirror's default-branch tip,
# which a PR may have bumped away from. Idempotent: a second call with the
# object already present is a fast no-op fetch, not a re-clone.
_c2_head_sha_fetched=0
_c2_ensure_head_sha_fetched() {
    local workspace_dir="$1" infra_mirror="$2"
    if [[ "${_c2_head_sha_fetched}" -eq 1 ]]; then
        return 0
    fi
    _c2_head_sha_fetched=1
    [[ -n "${GITHUB_SHA:-}" ]] || return 0
    timeout 60 git -C "${workspace_dir}" fetch --quiet "${infra_mirror}" "${GITHUB_SHA}" 2>/dev/null || true
}

wire_uv_git_mirror_rewrite() {
    local workspace_dir="$1"

    # `if`, not an AND-list: under `set -e` a false AND-list would return
    # non-zero from the function. Same reasoning as the pre-seed kill switch.
    if [[ "${OMNI_GIT_MIRROR_REWRITE_DISABLE:-0}" == "1" ]]; then
        return 0
    fi
    if [[ "${OMNI_GIT_MIRROR_DISABLE:-0}" == "1" ]]; then
        return 0
    fi
    if [[ -z "${GITHUB_ENV:-}" || ! -w "${GITHUB_ENV}" ]]; then
        return 0
    fi
    command -v git >/dev/null 2>&1 || return 0
    [[ -d "${workspace_dir}/.git" ]] || return 0

    local own_repo="${GITHUB_REPOSITORY##*/}"
    local infra_mirror="git://${_C2_MIRROR_HOST}:${_C2_MIRROR_PORT}/${own_repo}.git"

    # The pin must come from the commit THIS job builds, not from the mirror's
    # default-branch tip: a PR that bumps the pin would otherwise be verified
    # against the wrong SHA. The pre-seed left full default-branch history in
    # the workspace, so fetching GITHUB_SHA is a small delta, not a clone.
    _c2_ensure_head_sha_fetched "${workspace_dir}" "${infra_mirror}"
    local lock_blob=""
    if [[ -n "${GITHUB_SHA:-}" ]]; then
        lock_blob="$(git -C "${workspace_dir}" cat-file -p "${GITHUB_SHA}:uv.lock" 2>/dev/null || true)"
    fi
    if [[ -z "${lock_blob}" ]]; then
        # Deliberately NOT falling back to the seeded default-branch uv.lock.
        # Verifying the wrong pin is worse than verifying none: it would
        # install a rewrite for a commit uv never asks for, and uv would then
        # miss on its real pin against a mirror it has been redirected to --
        # exactly the hard failure this gating exists to prevent.
        echo "[c2-mirror-rewrite] could not read uv.lock at ${GITHUB_SHA:-<unset>}; leaving uv on github.com (fail-open)."
        return 0
    fi

    local repo pin
    for repo in ${_C2B_REWRITE_REPOS}; do
        # Never rewrite the job's own repository -- that is checkout's remote.
        if [[ "${repo}" == "${own_repo}" ]]; then
            continue
        fi

        pin="$(printf '%s\n' "${lock_blob}" \
            | grep -oE "${repo}\.git\?rev=[0-9a-f]{40}" \
            | head -1 \
            | grep -oE '[0-9a-f]{40}' || true)"
        if [[ -z "${pin}" ]]; then
            continue
        fi

        local repo_mirror="git://${_C2_MIRROR_HOST}:${_C2_MIRROR_PORT}/${repo}.git"
        # Ask the mirror the exact question that matters: "can you serve this
        # commit?" -- by performing the same kind of by-SHA fetch uv will, but
        # with `--filter=tree:0 --depth=1`, so the server sends the commit
        # object and nothing else. Measured 2026-08-14 against these mirrors:
        # present pin 104ms/132K, wrong repo 46ms, nonexistent SHA 52ms, both
        # correctly reported absent. An `ls-remote` advertised-refs check is
        # NOT sufficient and was tried first: uv pins are ordinarily
        # mid-history commits, not ref tips, so it reported the real pin
        # missing and the rewrite never engaged.
        #
        # This probe is also self-limiting in the right direction: it depends
        # on `uploadpack.allowFilter` being set on the mirror, so a mirror that
        # has not been prepared for this simply answers "absent" and the job
        # stays on github.com.
        local probe_dir probe_ok=0
        probe_dir="$(mktemp -d)"
        if git -C "${probe_dir}" init --quiet 2>/dev/null \
            && timeout 20 git -C "${probe_dir}" fetch --quiet --depth=1 --filter=tree:0 \
                   "${repo_mirror}" "${pin}" >/dev/null 2>&1; then
            probe_ok=1
        fi
        rm -rf "${probe_dir}"
        if [[ "${probe_ok}" -ne 1 ]]; then
            echo "[c2-mirror-rewrite] ${repo} pin ${pin:0:12} not served by ${repo_mirror}; leaving uv on github.com (fail-open)."
            continue
        fi

        # Only the `.git` form -- see the scoping note above. FETCH-ONLY: see
        # _c2_rewrite_add_pair for why an identity pushInsteadOf is also
        # installed (`insteadOf` on its own ALSO rewrites the push URL -- this
        # is documented git behaviour, not an edge case; verified in
        # docker/runners/README-c2b-uv-git-mirror.md that an unpinned push
        # dies with "access denied or repository not exported" against the
        # daemon, which deliberately serves no receive-pack).
        local upstream_url="https://github.com/${GITHUB_REPOSITORY%%/*}/${repo}.git"
        _c2_rewrite_add_pair "${repo_mirror}" "${upstream_url}"
        echo "[c2-mirror-rewrite] ${repo} pin ${pin:0:12} present on mirror; uv git fetch -> ${repo_mirror} (fetch-only; push and actions/checkout stay on github.com)."
    done

    return 0
}

# ---------------------------------------------------------------------------
# OMN-16114 C2c -- extend the fetch-only rewrite to sibling actions/checkout
# steps
# ---------------------------------------------------------------------------
#
# THE GAP THIS CLOSES. C2's pre-seed (seed_workspace_from_mirror) only warms
# the workspace for the job's OWN repository -- it is keyed off
# GITHUB_REPOSITORY alone and never looks at a step's `repository:` input.
# C2b's rewrite above only rewrites the `.git`-suffixed URL form uv's git
# dependencies use. Neither mechanism ever touches a SIBLING `actions/checkout`
# step -- a second checkout in the same job with an explicit `repository:`
# different from GITHUB_REPOSITORY, e.g. dispatcher-route-coverage.yml's
# "Checkout omnimarket (sibling)" step. Those go straight to github.com with
# zero acceleration even when the target repo is already mirrored. Confirmed
# root cause of ~27% of `dispatcher-route-coverage` job runs timing out at the
# 30-minute budget (RPC 408 / GnuTLS -110 / empty-reply, retried unbounded
# until the job timeout kills it) -- see OMN-16114.
#
# WHY THIS CANNOT BE A SINGLE UNCONDITIONAL insteadOf. Unlike C2b, which
# rewrites ONE well-known dependency with a pin resolvable from uv.lock,
# sibling checkouts are numerous, per-workflow, and fetch a mix of branch
# names (`ref: dev`), historical exact SHAs, and refs computed by an earlier
# step in the SAME job (unresolvable here, before any step has run).
# `insteadOf` has NO server fallback (verified in
# docker/runners/README-c2b-uv-git-mirror.md): once a URL is rewritten, a
# fetch the mirror cannot serve fails outright -- git does not then retry the
# un-rewritten URL. A blanket per-repo rewrite would therefore convert "mirror
# is stale for this one ref" into a hard failure, worse than the flakiness
# this component exists to remove. So this function does not trust the mirror
# by default -- it discovers exactly which (repo, ref) pairs THIS job's steps
# will request and proves the mirror can serve every one of them first.
#
# DISCOVERY MECHANISM. GITHUB_WORKFLOW_REF (a standard Actions env var, e.g.
# "OmniNode-ai/omnibase_infra/.github/workflows/ci.yml@refs/heads/dev") names
# the exact workflow file for this run; GITHUB_JOB names the exact job. Both
# are read from the OWN repo's checkout at GITHUB_SHA (the delta fetch shared
# with C2b via _c2_ensure_head_sha_fetched, so a PR that edits its own
# workflow file is scanned as it exists on THIS commit, not the mirror's
# stale default-branch copy). The job's YAML block is scanned for
# `repository: OmniNode-ai/<repo>` / `ref: <value>` pairs with plain text
# matching, not a YAML parser -- this repo's workflow files use a regular,
# predictable indentation shape, and a text scan avoids adding a new
# interpreter dependency to a hook that runs before every job on the fleet.
#
# CONJUNCTIVE PER-REPO GATING. `insteadOf` operates on the URL, not on a
# specific ref -- once a repo's rewrite is installed, it applies to EVERY
# fetch against that URL for the rest of the job, including ones this
# function never saw. So a repo is only made rewrite-eligible when EVERY
# (repo, ref) pair discovered for it in this job's block is individually
# proven servable:
#   - a 40-hex-char ref is treated as an exact SHA and probed exactly like
#     C2b's uv pin (`fetch --depth=1 --filter=tree:0`). This is the empirical
#     answer to "does an exact-SHA fetch fall through to origin if the mirror
#     lacks it" the ticket asked to verify, not assume: it does not -- so
#     this must be proven true before any rewrite is installed. A miss here
#     disqualifies the WHOLE repo for this job, not just that one occurrence
#     (ci.yml's application-database-domain-enforcement job checks out
#     omnibase_infra twice at two different pinned SHAs; both must be
#     servable or neither checkout is redirected).
#   - any other literal (branch/tag name, e.g. `dev`) is probed with
#     `git ls-remote --exit-code`, which the 120s-refreshed mirror
#     (config/runner_fleet.yaml git_mirror.refresh_interval_seconds) almost
#     always answers.
#   - a ref containing a GitHub Actions expression (`${{ ... }}`) or with no
#     `ref:` line at all (defaults to the target repo's default branch,
#     unknowable here) is unresolvable and disqualifies the repo outright --
#     never installed on a guess.
#
# NEVER THE JOB'S OWN REPOSITORY. `own_repo` is excluded unconditionally, by
# construction, even where it appears under an explicit `repository:` key
# (ci.yml's `.proof-dependencies` job checks out omnibase_infra-the-repo
# twice more, at older pinned SHAs, from an omnibase_infra job). Both the
# primary checkout and these self-referential sibling checkouts share the
# identical no-`.git`-suffix URL form actions/checkout always computes
# (`${GITHUB_SERVER_URL}/${owner}/${repo}`), so a rewrite keyed on that URL
# cannot tell them apart. The primary checkout fetches GITHUB_SHA -- the PR's
# own, possibly seconds-old head commit -- which the 120s-refresh mirror can
# easily not have yet; redirecting it with no fallback would break every job
# on every fresh push. Losing acceleration on the rare same-repo sibling
# checkout is the accepted, deliberate cost of that safety margin.
_C2C_SIBLING_PROBE_TIMEOUT="${OMNI_GIT_MIRROR_CHECKOUT_PROBE_TIMEOUT:-20}"

wire_sibling_checkout_mirror_rewrite() {
    local workspace_dir="$1"

    # `if`, not an AND-list: same `set -e` reasoning as every kill switch
    # above -- a false AND-list would return non-zero and kill the hook.
    if [[ "${OMNI_GIT_MIRROR_CHECKOUT_REWRITE_DISABLE:-0}" == "1" ]]; then
        return 0
    fi
    if [[ "${OMNI_GIT_MIRROR_DISABLE:-0}" == "1" ]]; then
        return 0
    fi
    if [[ -z "${GITHUB_ENV:-}" || ! -w "${GITHUB_ENV}" ]]; then
        return 0
    fi
    command -v git >/dev/null 2>&1 || return 0
    [[ -d "${workspace_dir}/.git" ]] || return 0
    [[ -n "${GITHUB_JOB:-}" ]] || return 0

    local workflow_path="${GITHUB_WORKFLOW_REF:-}"
    workflow_path="${workflow_path%%@*}"
    case "${workflow_path}" in
        "${GITHUB_REPOSITORY:-}"/*)
            workflow_path="${workflow_path#"${GITHUB_REPOSITORY}"/}"
            ;;
        *)
            return 0
            ;;
    esac

    local own_repo="${GITHUB_REPOSITORY##*/}"
    local infra_mirror="git://${_C2_MIRROR_HOST}:${_C2_MIRROR_PORT}/${own_repo}.git"

    _c2_ensure_head_sha_fetched "${workspace_dir}" "${infra_mirror}"
    local workflow_blob=""
    if [[ -n "${GITHUB_SHA:-}" ]]; then
        workflow_blob="$(git -C "${workspace_dir}" cat-file -p "${GITHUB_SHA}:${workflow_path}" 2>/dev/null || true)"
    fi
    if [[ -z "${workflow_blob}" ]]; then
        echo "[c2-mirror-rewrite] could not read ${workflow_path} at ${GITHUB_SHA:-<unset>}; no sibling checkout rewrite this job (fail-open)."
        return 0
    fi

    # Isolate this job's own block: a top-level (2-space-indented) key
    # matching GITHUB_JOB up to the next top-level key. Workflow files with
    # many jobs (ci.yml has 40+) would otherwise pay a probe round-trip per
    # job on every single run, most of which have no sibling checkout at all.
    local job_block
    job_block="$(printf '%s\n' "${workflow_blob}" | awk -v job="${GITHUB_JOB}" '
        $0 ~ "^  " job ":[[:space:]]*$" { injob=1; next }
        injob && /^  [A-Za-z0-9_.-]+:[[:space:]]*$/ { injob=0 }
        injob { print }
    ')"
    [[ -n "${job_block}" ]] || return 0

    # (repo, ref) pairs in encounter order, TSV. A `repository:` line starts
    # a pending pair; the next `ref:` line completes it; hitting the next
    # step boundary (`- name:` / `- uses:`) with no `ref:` seen completes it
    # with an empty ref (unresolved -- defaults to the repo's default branch).
    local pins
    pins="$(printf '%s\n' "${job_block}" | awk '
        /repository:[[:space:]]*OmniNode-ai\// {
            repo = $0
            sub(/.*OmniNode-ai\//, "", repo)
            gsub(/[[:space:]]*#.*/, "", repo)
            gsub(/[[:space:]]+$/, "", repo)
            pending_repo = repo
            next
        }
        pending_repo != "" && /^[[:space:]]*ref:/ {
            r = $0
            sub(/^[[:space:]]*ref:[[:space:]]*/, "", r)
            gsub(/[[:space:]]*#.*$/, "", r)
            gsub(/[[:space:]]+$/, "", r)
            print pending_repo "\t" r
            pending_repo = ""
            next
        }
        pending_repo != "" && /^[[:space:]]*-[[:space:]]*(name|uses):/ {
            print pending_repo "\t"
            pending_repo = ""
        }
        END {
            if (pending_repo != "") print pending_repo "\t"
        }
    ')"
    [[ -n "${pins}" ]] || return 0

    local pins_file
    pins_file="$(mktemp)"
    printf '%s\n' "${pins}" >"${pins_file}"

    local repo
    for repo in $(cut -f1 "${pins_file}" | sort -u); do
        if [[ "${repo}" == "${own_repo}" ]]; then
            continue
        fi

        local repo_mirror="git://${_C2_MIRROR_HOST}:${_C2_MIRROR_PORT}/${repo}.git"
        local all_ok=1 ref
        while IFS= read -r ref; do
            if [[ -z "${ref}" ]]; then
                all_ok=0
                break
            fi
            case "${ref}" in
                *'${{'*)
                    all_ok=0
                    break
                    ;;
            esac

            if [[ "${ref}" =~ ^[0-9a-f]{40}$ ]]; then
                # Exact SHA -- same by-SHA servability probe C2b uses for the
                # uv pin (see the long comment above this function for why
                # this specific check exists).
                local probe_dir probe_ok=0
                probe_dir="$(mktemp -d)"
                if git -C "${probe_dir}" init --quiet 2>/dev/null \
                    && timeout "${_C2C_SIBLING_PROBE_TIMEOUT}" git -C "${probe_dir}" fetch --quiet --depth=1 --filter=tree:0 \
                           "${repo_mirror}" "${ref}" >/dev/null 2>&1; then
                    probe_ok=1
                fi
                rm -rf "${probe_dir}"
                if [[ "${probe_ok}" -ne 1 ]]; then
                    echo "[c2-mirror-rewrite] ${repo}@${ref:0:12} (exact SHA) not served by ${repo_mirror}; leaving ${repo} sibling checkout(s) on github.com (fail-open)."
                    all_ok=0
                    break
                fi
            else
                # Branch/tag literal -- confirm the mirror currently
                # advertises it.
                if ! timeout "${_C2C_SIBLING_PROBE_TIMEOUT}" git ls-remote --exit-code "${repo_mirror}" "${ref}" >/dev/null 2>&1; then
                    echo "[c2-mirror-rewrite] ${repo}@${ref} not advertised by ${repo_mirror}; leaving ${repo} sibling checkout(s) on github.com (fail-open)."
                    all_ok=0
                    break
                fi
            fi
        done < <(awk -F'\t' -v r="${repo}" '$1==r{print $2}' "${pins_file}")

        if [[ "${all_ok}" -eq 1 ]]; then
            # No `.git` suffix -- see the scoping note in the header comment
            # above this function; this is the exact URL form actions/checkout
            # computes (`${GITHUB_SERVER_URL}/${owner}/${repo}`).
            local upstream_url="${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY%%/*}/${repo}"
            _c2_rewrite_add_pair "${repo_mirror}" "${upstream_url}"
            echo "[c2-mirror-rewrite] ${repo}: every sibling-checkout ref for job ${GITHUB_JOB} served by ${repo_mirror}; actions/checkout -> mirror (fetch-only)."
        fi
    done

    rm -f "${pins_file}"
    return 0
}

# ---------------------------------------------------------------------------
# OMN-16363 -- pre-job disk-admission gate (write-amplification loop breaker)
# ---------------------------------------------------------------------------
#
# THE MECHANISM THIS BREAKS. Per OMN-16360: when /data runs low, a job
# assigned to a runner fails almost instantly with ENOSPC -- but not before
# actions/checkout, uv sync, and docker build/layer writes have already landed
# a partial write on disk (ENOSPC is only raised once a write actually cannot
# complete; every byte written before that point is real, already-committed
# disk consumption). With dozens of runners cycling through repeated
# instant-fail-and-reassign, the AGGREGATE partial-write throughput across the
# fleet outpaces docker builder/image prune, turning a recoverable low-disk
# condition into a self-perpetuating write-amplification loop that drives free
# space to literal zero bytes within roughly 15-20 minutes (confirmed
# 2026-08-21, recurred twice more on 2026-08-22).
#
# THE FIX. Check free disk BEFORE any of the write-heavy steps in this hook's
# main body (workspace rm -rf + mirror reseed, uv/checkout mirror-rewrite
# discovery, wire_pypi_cache). Below the floor, fail the job here -- the SAME
# "instant fail" outcome ENOSPC already produces, but capped at a single `df`
# call's worth of I/O instead of however many megabytes a partial
# checkout/cache-write burns before the kernel actually returns ENOSPC. This
# is fix direction #1 from OMN-16363: "a minimum-free-disk gate before
# accepting a new job -- refuse/defer job acceptance below some threshold ...
# rather than accepting and immediately ENOSPC-failing."
#
# WHAT THIS DOES NOT CLAIM TO FIX. A self-hosted runner has no API to
# decline/requeue a job GitHub has already dispatched to it (there is no such
# endpoint in the Actions runner protocol), so GitHub will still reassign this
# runner a new job immediately after this one fails. The rapid-reassignment
# CYCLING is therefore not eliminated by this gate alone -- only the WRITE
# COST of each cycle is, which is exactly the amplification variable the
# incident evidence identifies (write throughput outpacing reclamation, not
# reassignment frequency by itself). disk_admission_self_pause() below is the
# secondary mechanism that also reduces cycling frequency, per fix direction
# #2 ("backoff-on-repeated-instant-setup-failure guard").
#
# THRESHOLD CONSISTENCY (OMN-16363 AC3). Same default (5 GB) as
# .github/workflows/runner-disk-preflight.yml's RUNNER_DISK_WARN_GB. The two
# checks run in different execution contexts (this one host-side pre-job on
# .201; that one inside the GH Actions job on whichever runner picked up the
# work) and cannot share a single config file, so the default is kept in sync
# by comment cross-reference, not by import -- the same pattern already used
# for the WEDGE_QUEUE_AGE_SECONDS-style thresholds in runner-monitor.sh.
_C3_DISK_ADMISSION_MIN_FREE_GB="${RUNNER_DISK_ADMISSION_MIN_FREE_GB:-5}"
# Consecutive-admission-failure backoff (fix direction #2). State is a plain
# counter file under RUNNER_HOME -- container-local, NOT a host bind mount --
# so it survives across jobs on the SAME persistent container (this fleet
# never recreates containers per-job) and resets naturally on a container
# recreate, which is correct: a freshly recreated runner has observed no disk
# pressure yet.
_C3_DISK_ADMISSION_BACKOFF_N="${RUNNER_DISK_ADMISSION_BACKOFF_N:-3}"
_C3_DISK_ADMISSION_STATE_DIR="${RUNNER_DISK_ADMISSION_STATE_DIR:-${RUNNER_HOME:-/home/runner/actions-runner}/.onex-disk-admission}"
# Durable pause-marker directory. Bind-mounted host-side
# (docker-compose.runners.yml) so a host-side companion
# (scripts/runner-disk-admission-restore.sh) can see which runners this gate
# paused and safely bring them back once disk recovers. Fails open (self-pause
# skipped, the per-job gate above still blocks each individual job) when the
# mount is absent -- e.g. before the fleet is recreated to pick up the new
# compose volume; see the rollout note in docs/runbooks/runner-disk-admission-gate.md.
_C3_DISK_ADMISSION_PAUSE_DIR="${RUNNER_DISK_ADMISSION_PAUSE_DIR:-/home/runner/.onex-disk-admission-pause}"
# The runner's own workspace/tool-cache mount -- the same volume actions/checkout,
# uv, and docker builds all write to inside this container.
_C3_DISK_ADMISSION_MOUNT="${RUNNER_DISK_ADMISSION_MOUNT:-/home/runner/actions-runner}"

# _c3_avail_kb -- test seam: DISK_ADMISSION_DF_OVERRIDE_KB lets tests inject an
# exact avail-KB reading without needing a real near-full filesystem.
_c3_avail_kb() {
    if [[ -n "${DISK_ADMISSION_DF_OVERRIDE_KB:-}" ]]; then
        echo "${DISK_ADMISSION_DF_OVERRIDE_KB}"
        return 0
    fi
    df -Pk "${_C3_DISK_ADMISSION_MOUNT}" 2>/dev/null | awk 'NR==2 {print $4}'
}

# disk_admission_gate -- returns 1 (job must fail) when free space on
# _C3_DISK_ADMISSION_MOUNT is below the admission floor; returns 0 (job
# proceeds) otherwise, INCLUDING when disk usage cannot be read at all (fail
# OPEN -- a broken `df` must never itself become a fleet-wide outage; the
# existing ENOSPC failure mode remains the backstop in that case).
disk_admission_gate() {
    local avail_kb avail_gb_frac min_free_kb runner="${RUNNER_NAME:-unknown}"
    avail_kb="$(_c3_avail_kb)"
    if ! [[ "${avail_kb}" =~ ^[0-9]+$ ]]; then
        echo "[disk-admission] could not read free space on ${_C3_DISK_ADMISSION_MOUNT}; gate fails open, job proceeds."
        return 0
    fi

    min_free_kb=$(( _C3_DISK_ADMISSION_MIN_FREE_GB * 1024 * 1024 ))
    if [[ "${avail_kb}" -ge "${min_free_kb}" ]]; then
        # Healthy -- clear any prior consecutive-failure streak.
        rm -f "${_C3_DISK_ADMISSION_STATE_DIR}/consecutive_failures" 2>/dev/null || true
        return 0
    fi

    avail_gb_frac="$(awk "BEGIN {printf \"%.2f\", ${avail_kb}/1024/1024}")"
    echo "::error title=RUNNER-DISK-ADMISSION:${avail_gb_frac}GB::Runner '${runner}' has only ${avail_gb_frac} GB free on ${_C3_DISK_ADMISSION_MOUNT} (below the ${_C3_DISK_ADMISSION_MIN_FREE_GB} GB admission floor). Refusing this job BEFORE workspace reset/checkout/dependency writes to avoid contributing to the OMN-16360 write-amplification loop. This is an infra failure, not a domain defect."

    mkdir -p "${_C3_DISK_ADMISSION_STATE_DIR}" 2>/dev/null || true
    local count_file="${_C3_DISK_ADMISSION_STATE_DIR}/consecutive_failures"
    local prev_count=0
    [[ -f "${count_file}" ]] && prev_count="$(cat "${count_file}" 2>/dev/null || echo 0)"
    [[ "${prev_count}" =~ ^[0-9]+$ ]] || prev_count=0
    local new_count=$(( prev_count + 1 ))
    echo "${new_count}" > "${count_file}" 2>/dev/null || true
    echo "[disk-admission] consecutive admission failures on ${runner}: ${new_count} (backoff threshold ${_C3_DISK_ADMISSION_BACKOFF_N})"

    if [[ "${new_count}" -ge "${_C3_DISK_ADMISSION_BACKOFF_N}" ]]; then
        disk_admission_self_pause "${runner}" "${avail_gb_frac}"
    fi

    return 1
}

# disk_admission_self_pause -- fix direction #2 ("backoff-on-repeated-instant-
# setup-failure guard"). Stops THIS container via the already-bind-mounted
# docker socket (docker-compose.runners.yml mounts /var/run/docker.sock into
# every runner) so its Runner.Listener stops polling GitHub entirely -- the
# only way a self-hosted runner can actually "not accept new job assignments"
# (there is no requeue/decline API). `docker stop` (never `docker restart`/
# recreate) on a container whose compose restart policy is `unless-stopped`
# does NOT auto-restart -- that policy means "restart on unexpected exit", not
# "always running"; an explicit stop is honored. Backgrounded with a short
# delay so it runs AFTER this hook (and therefore this job's failure) has
# fully exited, never mid-hook.
#
# Fails open by construction: without the pause-dir bind mount (pre-rollout,
# or any container not yet recreated with the new compose volume), this is a
# silent no-op and ONLY the per-job admission gate above is active -- which is
# already the primary, immediately-effective mechanism and requires no
# recreate to roll out fleet-wide (this whole file is bind-mounted read-only
# from the host; editing the host copy changes behaviour on every runner at
# its NEXT job start, same as every other mechanism in this file).
disk_admission_self_pause() {
    local runner="$1" avail_gb_frac="$2"
    if [[ ! -d "${_C3_DISK_ADMISSION_PAUSE_DIR}" ]]; then
        echo "[disk-admission] self-pause skipped: ${_C3_DISK_ADMISSION_PAUSE_DIR} not mounted (fleet not yet rolled out to this container)."
        return 0
    fi
    if ! command -v docker >/dev/null 2>&1; then
        echo "[disk-admission] self-pause skipped: docker CLI unavailable in container."
        return 0
    fi

    local marker="${_C3_DISK_ADMISSION_PAUSE_DIR}/${runner}"
    if [[ -f "${marker}" ]]; then
        echo "[disk-admission] self-pause skipped: ${runner} already has a pause marker (previous pause not yet restored)."
        return 0
    fi

    {
        echo "runner=${runner}"
        echo "avail_gb=${avail_gb_frac}"
        echo "paused_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "reason=consecutive_disk_admission_failures"
    } > "${marker}.tmp" 2>/dev/null || return 0
    mv "${marker}.tmp" "${marker}" 2>/dev/null || return 0

    echo "[disk-admission] PAUSING ${runner}: ${_C3_DISK_ADMISSION_BACKOFF_N} consecutive disk-admission failures. Stopping this container so its listener stops polling GitHub; scripts/runner-disk-admission-restore.sh will restart it once /data recovers (slope-plus-canary)."
    nohup sh -c "sleep 2; docker stop '${runner}' >/dev/null 2>&1" >/dev/null 2>&1 &
    disown 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# OMN-14027 C1 -- devpi PyPI pull-through cache wiring (host layer, fail-open)
# ---------------------------------------------------------------------------
#
# THE PROBLEM. 72 runners NAT through one home uplink and each one independently
# cold-downloads the same wheels from pypi.org. That redundant egress is what
# trips `uv sync` download timeouts under concurrent load. A devpi pull-through
# cache (omninode-pypi-cache, 22G warm corpus) fetches each wheel from PyPI once
# and serves the rest of the fleet from the LAN.
#
# WHY THIS LIVES IN THE JOB-STARTED HOOK AND NOT IN COMPOSE. The canary
# (runner-2/4/5) carries the index as CONTAINER env, which can only be applied by
# recreating the container. Recreating a runner kills any in-flight job and wipes
# its warm tool cache, so it is not an acceptable way to roll 69 more runners.
# This hook is bind-mounted read-only from the host
# (docker/runners/runner-job-started.sh), so editing the host copy changes
# behaviour on EVERY runner at its NEXT job start -- zero container recreates,
# zero in-flight jobs killed. The compose override
# (docker/docker-compose.pypi-canary.yml) stays as the recreate-time path so the
# two mechanisms agree; this one is what makes the rollout non-disruptive.
#
# FALLBACK SEMANTICS -- MEASURED, NOT ASSUMED. `uv` has NO index-level fallback.
# Verified 2026-08-14 in a throwaway container off the runner image (uv 0.6.14):
#   * UV_DEFAULT_INDEX=<dead> + PIP_EXTRA_INDEX_URL=https://pypi.org/simple/
#     (the canary's exact config shape) -> uv HARD-FAILS:
#     "Failed to fetch: .../six/ ... tcp connect error: Connection refused".
#     PIP_* vars are invisible to uv.
#   * UV_DEFAULT_INDEX=<dead> + UV_INDEX=https://pypi.org/simple +
#     UV_INDEX_STRATEGY=unsafe-best-match -> STILL hard-fails. A second index
#     does not rescue an unreachable first index; index strategy only arbitrates
#     between indexes that answer.
#   * pip with PIP_INDEX_URL=<dead> + PIP_EXTRA_INDEX_URL=pypi.org -> DEGRADES
#     cleanly and installs from pypi.org.
# So the fleet-wide risk of pointing uv at devpi is that a devpi outage would
# fail EVERY job closed. The mitigation is the liveness probe below: the index is
# exported only when the cache answers at job start, so a dead cache means the
# job runs on direct pypi.org egress exactly as it does today. This narrows the
# hard-fail window to a cache that dies mid-job, which is the same exposure a job
# already has to pypi.org itself.
#
# EFFECTIVENESS BEACON. Presence of an env var in a config file proves nothing --
# the previous canary recipe was a silent no-op that passed review. This hook also
# exports UV_HTTP_TIMEOUT=601 (semantically identical to the fleet default of 600)
# purely as a beacon: the hardened setup-python-uv composite echoes
# "UV_HTTP_TIMEOUT=<value>" in its Install dependencies step, so any job log that
# prints 601 PROVES the hook's env reached the job's steps on that runner, and any
# job log that prints 600 proves it did not. Do not "tidy" this to 600.
#
# WHY UV_INDEX AND NOT UV_DEFAULT_INDEX -- MEASURED, NOT ASSUMED. Every uv.lock in
# this org records `source = { registry = "https://pypi.org/simple" }` per package
# (194 entries in omnibase_infra uv.lock) alongside absolute
# files.pythonhosted.org wheel URLs. Substituting the DEFAULT index invalidates
# that lock, so every `uv sync --locked` / `uv lock --check` gate fails closed.
# Reproduced on omnibase_infra pyproject+uv.lock @ f15a491c6 in a throwaway
# container off omninode-runner:latest (uv 0.6.14):
#   UV_DEFAULT_INDEX=https://pypi.org/simple -> rc=0, "Resolved 196 packages in 5ms"
#   UV_DEFAULT_INDEX=<devpi>                 -> rc=2 after a 24.74s re-resolve,
#                                               "lockfile ... needs to be updated"
#   UV_INDEX=<devpi> (extra index; default stays pypi.org) -> rc=0, 5ms / 0.72ms
# It also cost a real job: omniintelligence 94804218001 failed `uv sync --locked`
# on 2026-08-14T14:55Z while wired with UV_DEFAULT_INDEX. So devpi is added as an
# EXTRA index: locked/frozen jobs keep passing untouched, while the re-resolving
# `uv sync` path (the hardened composite default, and where the canary measured 618
# wheels served from LAN) still prefers the cache. UV_INDEX_STRATEGY is
# unsafe-best-match so a devpi index that is stale for one package loses to
# pypi.org on version rather than pinning the job to the stale one.
# Do not "simplify" this back to UV_DEFAULT_INDEX.
#
# ROLLOUT CONTROL. _C1_PYPI_RUNNERS is a space-separated RUNNER_NAME allowlist or
# ALL or NONE. Staged rollouts edit it on the host copy of this file; no container
# is touched. OMNI_PYPI_CACHE_DISABLE=1 is the kill switch.
_C1_PYPI_INDEX="${OMNI_PYPI_CACHE_INDEX:-http://omninode-pc.tail75df5e.ts.net:3141/root/pypi/+simple/}"
_C1_PYPI_FALLBACK="${OMNI_PYPI_CACHE_FALLBACK:-https://pypi.org/simple/}"
_C1_PYPI_RUNNERS="${OMNI_PYPI_CACHE_RUNNERS:-ALL}"
_C1_PYPI_PROBE_TIMEOUT="${OMNI_PYPI_CACHE_PROBE_TIMEOUT:-5}"
_C1_PYPI_AUDIT="${OMNI_PYPI_CACHE_AUDIT:-/tmp/omni-c1-pypi-cache.log}"

_c1_audit() {
    # Job log line (diagnosable from GitHub) + container-local audit trail
    # (readable per-runner with `docker exec ... cat`, which is how this rollout
    # is verified runner-by-runner without downloading 72 job logs).
    echo "[c1-pypi] $*"
    printf '%s %s\n' "$(date -u +%FT%TZ 2>/dev/null || echo unknown)" "$*" \
        >>"${_C1_PYPI_AUDIT}" 2>/dev/null || true
}

wire_pypi_cache() {
    # Written as `if`, not `[[ ... ]] && return 0`: this script runs under
    # `set -e` and a false AND-list would kill the hook -- i.e. fail every job.
    if [[ "${OMNI_PYPI_CACHE_DISABLE:-0}" == "1" ]]; then
        return 0
    fi

    local runner="${RUNNER_NAME:-unknown}"
    local ctx="runner=${runner} run=${GITHUB_RUN_ID:-?} job=${GITHUB_JOB:-?}"

    if [[ "${_C1_PYPI_RUNNERS}" == "NONE" ]]; then
        return 0
    fi
    if [[ "${_C1_PYPI_RUNNERS}" != "ALL" ]]; then
        case " ${_C1_PYPI_RUNNERS} " in
            *" ${runner} "*) ;;
            *) return 0 ;;
        esac
    fi

    # The runner-2/4/5 canary already carries the index in CONTAINER env, which
    # outranks nothing here but means there is nothing to add. Report, don't
    # duplicate.
    if [[ -n "${UV_DEFAULT_INDEX:-}" ]]; then
        _c1_audit "${ctx} SKIP already-wired-at-container-env index=${UV_DEFAULT_INDEX}"
        return 0
    fi

    if [[ -z "${GITHUB_ENV:-}" ]]; then
        _c1_audit "${ctx} NOOP GITHUB_ENV-unset -- job-started hooks cannot export env on this runner version; job unaffected (direct pypi.org egress)"
        return 0
    fi
    if [[ ! -f "${GITHUB_ENV}" || ! -w "${GITHUB_ENV}" ]]; then
        _c1_audit "${ctx} NOOP GITHUB_ENV=${GITHUB_ENV} not-writable; job unaffected (direct pypi.org egress)"
        return 0
    fi
    if ! command -v curl >/dev/null 2>&1; then
        _c1_audit "${ctx} NOOP curl-missing; cannot prove cache liveness, staying on direct egress"
        return 0
    fi

    # Liveness probe. HEAD, not GET: a GET of the index root is a 41MB body and
    # would cost more than it saves, 72x per job.
    if ! curl -fsS -I --max-time "${_C1_PYPI_PROBE_TIMEOUT}" -o /dev/null "${_C1_PYPI_INDEX}" 2>/dev/null; then
        _c1_audit "${ctx} DEGRADE cache-probe-failed index=${_C1_PYPI_INDEX} -- job runs on direct pypi.org egress (fail-open)"
        return 0
    fi

    {
        echo "UV_INDEX=${_C1_PYPI_INDEX}"
        echo "UV_INDEX_STRATEGY=unsafe-best-match"
        echo "PIP_INDEX_URL=${_C1_PYPI_INDEX}"
        echo "PIP_EXTRA_INDEX_URL=${_C1_PYPI_FALLBACK}"
        echo "UV_HTTP_TIMEOUT=601"
    } >>"${GITHUB_ENV}" 2>/dev/null || {
        _c1_audit "${ctx} NOOP GITHUB_ENV-write-failed; job unaffected"
        return 0
    }

    _c1_audit "${ctx} WIRED index=${_C1_PYPI_INDEX} pip-fallback=${_C1_PYPI_FALLBACK} beacon=UV_HTTP_TIMEOUT:601"
    return 0
}

# OMN-16363 -- disk-admission gate runs FIRST, before every other step in this
# hook (including wire_pypi_cache and the workspace reset below). This is the
# one call site in this file that is allowed to fail the job: every other
# mechanism here is `|| true` fail-open by design, but the whole point of this
# gate is to fail fast, before any write-heavy step, when disk is critically
# low. See the OMN-16363 comment block above disk_admission_gate() for why this
# ordering is load-bearing.
disk_admission_gate || exit 1

# Deliberately called before the workspace-reset logic below, and guarded with
# `|| true`, so neither an unset GITHUB_WORKSPACE nor a workspace-reset failure
# can change whether the cache is wired, and a defect here can never fail a job.
wire_pypi_cache || true

RUNNER_HOME="${RUNNER_HOME:-/home/runner/actions-runner}"
RUNNER_WORK_DIR="${RUNNER_WORK_DIR:-_work}"
WORK_ROOT="${RUNNER_HOME}/${RUNNER_WORK_DIR}"

workspace="${GITHUB_WORKSPACE:-}"
if [[ -z "${workspace}" && -n "${GITHUB_REPOSITORY:-}" ]]; then
    repo_name="${GITHUB_REPOSITORY##*/}"
    workspace="${WORK_ROOT}/${repo_name}/${repo_name}"
fi

if [[ -z "${workspace}" ]]; then
    echo "[runner-job-started] GITHUB_WORKSPACE is unset; no workspace cleanup performed."
    exit 0
fi

canonical_work_root="$(realpath -m -- "${WORK_ROOT}")"
canonical_workspace="$(realpath -m -- "${workspace}")"

case "${canonical_workspace}" in
    "${canonical_work_root}/"*/*) ;;
    *)
        echo "[runner-job-started] Refusing to clean workspace outside ${canonical_work_root}: ${canonical_workspace}" >&2
        exit 1
        ;;
esac

if [[ "${canonical_workspace}" == "/" || "${canonical_workspace}" == "${canonical_work_root}" ]]; then
    echo "[runner-job-started] Refusing unsafe workspace path: ${canonical_workspace}" >&2
    exit 1
fi

echo "[runner-job-started] Resetting workspace: ${canonical_workspace}"

err_file="$(mktemp)"
trap 'rm -f "${err_file}"' EXIT

if rm -rf -- "${canonical_workspace}" 2>"${err_file}"; then
    mkdir -p -- "${canonical_workspace}"
    seed_workspace_from_mirror "${canonical_workspace}"
    wire_uv_git_mirror_rewrite "${canonical_workspace}" || true
    wire_sibling_checkout_mirror_rewrite "${canonical_workspace}" || true
    _c2_rewrite_flush || true
    exit 0
fi

# Plain rm -rf failed. Fail loud FIRST: name the offending paths and the
# owning uid so this is diagnosable from the job log, not just "exit 1".
echo "[runner-job-started] ERROR: rm -rf failed on ${canonical_workspace}:" >&2
cat "${err_file}" >&2
echo "[runner-job-started] Non-runner-owned paths under ${canonical_workspace}:" >&2
find "${canonical_workspace}" \! -user "$(id -u)" -printf '%u:%g %m %p\n' 2>/dev/null >&2 || true

# Self-heal fallback (OMN-15134): a narrowly scoped NOPASSWD sudo rule
# (Dockerfile, /etc/sudoers.d/runner-workspace-reset) allows exactly this one
# rm -rf under this runner's own _work tree as root. Try it; if sudo itself
# is unavailable or the rule doesn't match (defense in depth -- this must
# never silently grant a broader escalation), fail the hook loudly rather
# than let the job proceed against an unreset workspace.
echo "[runner-job-started] Attempting scoped root cleanup fallback..." >&2
: >"${err_file}"
if ! command -v sudo >/dev/null 2>&1; then
    echo "[runner-job-started] ERROR: sudo is not installed in this image; cannot self-heal." >&2
elif sudo -n /bin/rm -rf -- "${canonical_workspace}" 2>"${err_file}"; then
    echo "[runner-job-started] Root-owned debris removed via scoped sudo fallback." >&2
    mkdir -p -- "${canonical_workspace}"
    seed_workspace_from_mirror "${canonical_workspace}"
    wire_uv_git_mirror_rewrite "${canonical_workspace}" || true
    wire_sibling_checkout_mirror_rewrite "${canonical_workspace}" || true
    _c2_rewrite_flush || true
    exit 0
else
    echo "[runner-job-started] ERROR: scoped sudo fallback also failed:" >&2
    cat "${err_file}" >&2
fi

echo "[runner-job-started] Refusing to run the job against an unreset workspace." >&2
echo "[runner-job-started] Manual remediation: docker exec -u root omninode-deploy-runner rm -rf -- '${canonical_workspace}'" >&2
exit 1
