#!/usr/bin/env bash
# Reset the current repository workspace before each self-hosted runner job.
#
# Stateful self-hosted runners can inherit sparse-checkout and partial worktree
# state from earlier jobs on the same runner. GitHub-hosted runners avoid this
# by starting each job on a fresh VM; this hook gives the Docker runner fleet
# the same repository-workspace invariant before actions/checkout runs.

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

case "${workspace}" in
    "${WORK_ROOT}/"*/*) ;;
    *)
        echo "[runner-job-started] Refusing to clean workspace outside ${WORK_ROOT}: ${workspace}" >&2
        exit 1
        ;;
esac

if [[ "${workspace}" == "/" || "${workspace}" == "${WORK_ROOT}" ]]; then
    echo "[runner-job-started] Refusing unsafe workspace path: ${workspace}" >&2
    exit 1
fi

echo "[runner-job-started] Resetting workspace: ${workspace}"
rm -rf -- "${workspace}"
mkdir -p -- "${workspace}"
