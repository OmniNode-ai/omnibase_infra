#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# Tell tools inside a runner container their real CPU count (OMN-19960).
#
# A runner runs under a CFS quota (`cpus: "2.0"` in the runner compose files),
# but `nproc`, `os.cpu_count()` and `os.sched_getaffinity()` inside it report
# every host core (32 on .201 and .202). Tools that size their fan-out from that
# count start 32 workers inside a 2-CPU, 6 GiB container: pre-commit partitions
# every hook 32 ways (the .202 `validate-spdx-headers` memcg OOM kills), and
# `pytest -n auto` starts 32 xdist workers (the OMN-19209 per-job pins).
#
# `omni_cgroup_cpu_env [cpu.max path]` reads the container's own cgroup v2
# `cpu.max` (default /sys/fs/cgroup/cpu.max). When the quota is not `max`, it
# computes N = ceil(quota / period) and exports, each only if not already set:
#
#   OMNI_CGROUP_CPUS=N               the count itself, for scripts to read
#   PYTEST_XDIST_AUTO_NUM_WORKERS=N  what `pytest -n auto` starts
#   OMP_NUM_THREADS=N                OpenMP pools; GNU `nproc` honours it too
#   PRE_COMMIT_NO_CONCURRENCY=1      only when N <= 2; pre-commit has no
#                                    numeric knob, and at 2 CPUs serial costs at
#                                    most half of one hook's wall time
#
# An unlimited quota (`max`), a missing or unreadable file (cgroup v1, or no
# cgroup mount), or a malformed line exports nothing: no guess.
#
# entrypoint.sh sources this file and calls the function before it spawns
# run.sh. gosu keeps the environment, so every job step inherits the values.
# Two cases do NOT inherit them:
#   - a job that runs in its own `container:` gets that container's
#     environment, not the runner's;
#   - `docker exec <runner> ...` starts from the image's Config.Env, not from
#     the entrypoint's process tree. Read the live values from the listener's
#     /proc/<pid>/environ, or from a job step.
#
# Sourcing the file and calling the function twice changes nothing: every
# export is guarded by "only if unset".

omni_cgroup_cpu_env() {
    local cpu_max_path="${1:-/sys/fs/cgroup/cpu.max}"
    local quota="" period="" cpus

    [[ -r "${cpu_max_path}" ]] || return 0
    read -r quota period < "${cpu_max_path}" || true

    # `max` (no quota) or anything that is not two positive integers.
    [[ "${quota}" =~ ^[0-9]+$ && "${period}" =~ ^[0-9]+$ ]] || return 0
    (( quota > 0 && period > 0 )) || return 0

    cpus=$(( (quota + period - 1) / period ))

    export OMNI_CGROUP_CPUS="${OMNI_CGROUP_CPUS:-${cpus}}"
    export PYTEST_XDIST_AUTO_NUM_WORKERS="${PYTEST_XDIST_AUTO_NUM_WORKERS:-${cpus}}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${cpus}}"
    if (( cpus <= 2 )); then
        export PRE_COMMIT_NO_CONCURRENCY="${PRE_COMMIT_NO_CONCURRENCY:-1}"
    fi
    return 0
}
