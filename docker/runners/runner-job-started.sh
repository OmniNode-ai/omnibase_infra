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
    local lock_blob=""
    if [[ -n "${GITHUB_SHA:-}" ]]; then
        if timeout 60 git -C "${workspace_dir}" fetch --quiet "${infra_mirror}" "${GITHUB_SHA}" 2>/dev/null; then
            lock_blob="$(git -C "${workspace_dir}" cat-file -p "${GITHUB_SHA}:uv.lock" 2>/dev/null || true)"
        fi
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

    local repo pin rewrite_count=0
    local -a env_lines=()
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

        # Only the `.git` form -- see the scoping note above.
        local upstream_url="https://github.com/${GITHUB_REPOSITORY%%/*}/${repo}.git"
        env_lines+=("GIT_CONFIG_KEY_${rewrite_count}=url.${repo_mirror}.insteadOf")
        env_lines+=("GIT_CONFIG_VALUE_${rewrite_count}=${upstream_url}")
        rewrite_count=$((rewrite_count + 1))

        # Make the redirect FETCH-ONLY. `insteadOf` on its own ALSO rewrites the
        # push URL -- this is documented git behaviour, not an edge case, and it
        # was verified here: with only the insteadOf above installed,
        # `git remote -v` reported
        #     origin  git://172.18.0.1:9418/onex_change_control.git (push)
        # and a push died with "access denied or repository not exported"
        # because the daemon deliberately serves no receive-pack. Any job that
        # pushed to the `.git`-suffixed URL would have failed CLOSED -- exactly
        # the class of outcome the rest of this function is gated to prevent.
        #
        # git resolves a push URL against pushInsteadOf rules FIRST and only
        # falls back to insteadOf when none match, so an IDENTITY pushInsteadOf
        # (upstream -> itself) pins pushes back on github.com while leaving the
        # fetch redirect fully intact. Verified in a throwaway container off
        # omninode-runner:latest: fetch -> git://172.18.0.1:9418/..., push ->
        # https://github.com/..., and the by-SHA fetch uv performs still rc=0.
        env_lines+=("GIT_CONFIG_KEY_${rewrite_count}=url.${upstream_url}.pushInsteadOf")
        env_lines+=("GIT_CONFIG_VALUE_${rewrite_count}=${upstream_url}")
        rewrite_count=$((rewrite_count + 1))

        echo "[c2-mirror-rewrite] ${repo} pin ${pin:0:12} present on mirror; uv git fetch -> ${repo_mirror} (fetch-only; push and actions/checkout stay on github.com)."
    done

    if [[ "${rewrite_count}" -eq 0 ]]; then
        return 0
    fi

    {
        printf '%s\n' "${env_lines[@]}"
        echo "GIT_CONFIG_COUNT=${rewrite_count}"
    } >>"${GITHUB_ENV}" 2>/dev/null || {
        echo "[c2-mirror-rewrite] GITHUB_ENV write failed; job unaffected (fail-open)."
        return 0
    }
    return 0
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
    exit 0
else
    echo "[runner-job-started] ERROR: scoped sudo fallback also failed:" >&2
    cat "${err_file}" >&2
fi

echo "[runner-job-started] Refusing to run the job against an unreset workspace." >&2
echo "[runner-job-started] Manual remediation: docker exec -u root omninode-deploy-runner rm -rf -- '${canonical_workspace}'" >&2
exit 1
