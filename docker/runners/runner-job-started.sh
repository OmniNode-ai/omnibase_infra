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
    exit 0
else
    echo "[runner-job-started] ERROR: scoped sudo fallback also failed:" >&2
    cat "${err_file}" >&2
fi

echo "[runner-job-started] Refusing to run the job against an unreset workspace." >&2
echo "[runner-job-started] Manual remediation: docker exec -u root omninode-deploy-runner rm -rf -- '${canonical_workspace}'" >&2
exit 1
