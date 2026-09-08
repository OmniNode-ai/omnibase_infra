#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# lane_lock.sh -- shell front end for the per-compose-project lane lock
# (OMN-16729). Sourced by scripts/deploy-runtime.sh,
# scripts/runtime_build/refresh_dev_lane.sh and
# scripts/runtime_build/refresh_stability_lane.sh.
#
# Contract:
#   lane_lock_acquire <compose_project> <lane> <ref> <timeout_seconds> [argv...]
#       0  acquired, or already held by an outer process (re-entrant no-op)
#       2  contended -- the bounded wait expired, the holder was named
#       3  precondition failure (no python, helper missing)
#   lane_lock_release
#       releases only a lock THIS shell acquired; a no-op otherwise.
#
# Re-entrancy: the acquiring shell exports ONEX_LANE_LOCK_HELD with the compose
# project appended. A nested call (refresh_*_lane.sh -> deploy-runtime.sh) sees
# its own project already listed and does NOT try to acquire again, so the
# nested call cannot deadlock against its own parent. That token is inherited
# only by real children of the holder, which is exactly the set of processes
# covered by the parent's lock.
#
# The lock itself is an fcntl lock on a numbered fd this shell opens. Because
# the fd stays open in this shell, the lock is held for the whole critical
# section and is released by the kernel however this shell exits -- including
# SIGKILL. Nothing here ever steals a lock.
#
# Residual: children inherit the descriptor, so a surviving child of a killed
# holder keeps the lane locked. That is usually the correct answer (a live
# `docker compose up` orphaned by a killed deploy is exactly when a second
# refresh must not start) and it is diagnosable rather than silent -- the holder
# sidecar names the shell pid and `lane_lock.py describe` reports it NOT RUNNING
# while the lock is still held. See lane_lock.py's module docstring.

# Numbered descriptor the lock file is opened on. Overridable only if a caller
# already uses fd 9 for something else.
ONEX_LANE_LOCK_FD="${ONEX_LANE_LOCK_FD:-9}"
# Set to the compose project when THIS shell owns the lock; empty otherwise.
ONEX_LANE_LOCK_OWNED=""

_LANE_LOCK_DIR_SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LANE_LOCK_PY="${_LANE_LOCK_DIR_SELF}/lane_lock.py"

lane_lock_log() { printf '[lane-lock] %s\n' "$*" >&2; }

lane_lock_python_bin() {
    # lane_lock.py is pure stdlib -- any python3 will do. Prefer the repo venv
    # when present so the interpreter matches the rest of the deploy path.
    local repo_venv="${_LANE_LOCK_DIR_SELF}/../../.venv/bin/python"
    if [[ -x "${repo_venv}" ]]; then
        printf '%s\n' "${repo_venv}"
        return 0
    fi
    if command -v python3 >/dev/null 2>&1; then
        printf 'python3\n'
        return 0
    fi
    return 1
}

lane_lock_is_held() {
    # True when an OUTER process in this ancestry already holds the lane.
    local project="$1"
    case " ${ONEX_LANE_LOCK_HELD:-} " in
        *" ${project} "*) return 0 ;;
    esac
    return 1
}

lane_lock_acquire() {
    local project="$1"
    local lane="$2"
    local ref="$3"
    local timeout="$4"
    shift 4 || true

    if [[ -z "${project}" ]]; then
        lane_lock_log "ERROR: lane_lock_acquire requires a compose project name."
        return 3
    fi

    if lane_lock_is_held "${project}"; then
        lane_lock_log "lane '${project}' is already held by an outer process in this ancestry (ONEX_LANE_LOCK_HELD='${ONEX_LANE_LOCK_HELD}'); re-entrant, not re-acquiring."
        return 0
    fi

    local py
    if ! py="$(lane_lock_python_bin)"; then
        lane_lock_log "ERROR: no python3 available to take the lane lock. Refusing to mutate lane '${project}' unserialised."
        return 3
    fi
    if [[ ! -f "${LANE_LOCK_PY}" ]]; then
        lane_lock_log "ERROR: lane lock helper not found at ${LANE_LOCK_PY}. Refusing to mutate lane '${project}' unserialised."
        return 3
    fi

    local lock_file
    if ! lock_file="$("${py}" "${LANE_LOCK_PY}" path --compose-project "${project}")"; then
        lane_lock_log "ERROR: could not resolve the lock path for lane '${project}'."
        return 3
    fi
    mkdir -p "$(dirname "${lock_file}")" || return 3

    # Open the lock file on the numbered fd IN THIS SHELL. The child below
    # flocks the inherited descriptor; because this shell keeps it open, the
    # lock is held for the whole critical section rather than for the lifetime
    # of the child.
    eval "exec ${ONEX_LANE_LOCK_FD}>\"\${lock_file}\"" || {
        lane_lock_log "ERROR: could not open ${lock_file} on fd ${ONEX_LANE_LOCK_FD}."
        return 3
    }

    if ! "${py}" "${LANE_LOCK_PY}" acquire \
        --compose-project "${project}" \
        --fd "${ONEX_LANE_LOCK_FD}" \
        --timeout "${timeout}" \
        --lane "${lane}" \
        --ref "${ref}" \
        --argv "$*"; then
        eval "exec ${ONEX_LANE_LOCK_FD}>&-" 2>/dev/null || true
        return 2
    fi

    ONEX_LANE_LOCK_OWNED="${project}"
    export ONEX_LANE_LOCK_HELD="${ONEX_LANE_LOCK_HELD:+${ONEX_LANE_LOCK_HELD} }${project}"
    return 0
}

lane_lock_release() {
    [[ -n "${ONEX_LANE_LOCK_OWNED:-}" ]] || return 0
    local py
    if py="$(lane_lock_python_bin)"; then
        "${py}" "${LANE_LOCK_PY}" release --compose-project "${ONEX_LANE_LOCK_OWNED}" \
            >/dev/null 2>&1 || true
    fi
    eval "exec ${ONEX_LANE_LOCK_FD}>&-" 2>/dev/null || true
    lane_lock_log "released lane '${ONEX_LANE_LOCK_OWNED}'."
    ONEX_LANE_LOCK_OWNED=""
}
