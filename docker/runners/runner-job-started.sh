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
    exit 0
else
    echo "[runner-job-started] ERROR: scoped sudo fallback also failed:" >&2
    cat "${err_file}" >&2
fi

echo "[runner-job-started] Refusing to run the job against an unreset workspace." >&2
echo "[runner-job-started] Manual remediation: docker exec -u root omninode-deploy-runner rm -rf -- '${canonical_workspace}'" >&2
exit 1
