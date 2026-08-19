#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# compose_wait_timeout.sh -- OMN-15718: bounded-deadline + stranded-container
# reconciliation helpers shared by deploy-runtime.sh and
# refresh_stability_lane.sh.
#
# Background: refresh_stability_lane.sh --ref origin/dev --execute (2026-08-05,
# .201) hit a forward-migration failure (OMN-15717), then its own failure-path
# rollback retagged images back to the prior known-good digest but the
# subsequent `docker compose up -d --no-deps --force-recreate` for the core
# services left runtime-effects/runtime-worker stranded in `Created` state --
# compose still honors `depends_on: migration-gate: condition: service_healthy`
# for those services even under `--no-deps`, and migration-gate can never
# become healthy once forward-migration has already failed. That `up` call had
# no bounded deadline, so it hung indefinitely instead of failing fast: it
# never reached the health-gate/receipt stage, and the operator had to recover
# manually with `docker start`.
#
# This file is SOURCED, not executed. It must not be invoked directly.
#
# Provides:
#   RUNTIME_COMPOSE_WAIT_TIMEOUT_SECONDS  Bounded deadline (seconds) applied to
#                                         every wrapped `docker compose ... up`
#                                         call. Operator-overridable; defaults
#                                         to 300s. Set before sourcing this
#                                         file to change it.
#   compose_up_bounded <timeout_s> <cmd...>
#                                         Run <cmd...> (a full `docker compose
#                                         ... up ...` argv) under a `timeout`
#                                         deadline. Propagates the wrapped
#                                         command's exit code verbatim; on
#                                         expiry returns exactly 124 (GNU
#                                         `timeout`'s own signal for "killed at
#                                         deadline"), never remapped onto a
#                                         reused/generic code, so callers can
#                                         test for it explicitly.
#   reconcile_container_running_state <container_ref> [<label>]
#                                         Ensure one container ends this call
#                                         either 'running' (recovered via
#                                         `docker start` if it was not) or torn
#                                         down (`docker rm -f`) if it cannot be
#                                         brought to running -- never leaves it
#                                         stranded in an ambiguous
#                                         Created/Exited state. Returns 0 if
#                                         running (already, or recovered), 1 if
#                                         it had to be torn down.

# Include guard -- this file may be sourced by more than one caller in the
# same process (e.g. a test harness sourcing both deploy-runtime.sh helpers
# and this file directly). This file is SOURCED ONLY (see header); it is
# never executed directly, so a plain `return` here is always valid.
if [[ -n "${__OMNIBASE_COMPOSE_WAIT_TIMEOUT_SH_SOURCED:-}" ]]; then
    return 0
fi
__OMNIBASE_COMPOSE_WAIT_TIMEOUT_SH_SOURCED=1

# Parameter-default assignment (safe under `set -u`): honors a caller- or
# operator-exported value, otherwise defaults to 300s. Deliberately NOT
# `readonly` -- a caller sourcing this file after setting its own default
# would otherwise collide.
: "${RUNTIME_COMPOSE_WAIT_TIMEOUT_SECONDS:=300}"

_cwt_log() { printf '[compose-wait-timeout] %s\n' "$*" >&2; }
_cwt_err() { printf '[compose-wait-timeout] ERROR: %s\n' "$*" >&2; }

compose_up_bounded() {
    # Usage: compose_up_bounded <timeout_seconds> <cmd...>
    #
    # Wraps a `docker compose ... up ...` argv in a bounded wall-clock
    # deadline via GNU `timeout`. This is the correct backstop even for
    # commands that already carry compose's own `--wait --wait-timeout`:
    # that flag only bounds the CLI's POST-start wait for the named target
    # service(s) to report healthy. It does NOT bound compose's PRE-start
    # wait for a `depends_on: condition: service_healthy` dependency that a
    # target service declares in the compose file -- that wait is honored
    # even under `--no-deps`, and has no deadline of its own if the
    # dependency can never resolve (the migration-gate case above). `timeout`
    # bounds the whole `docker compose up` process regardless of which
    # internal phase is stuck.
    local timeout_seconds="$1"
    shift
    local -a cmd=("$@")

    if [[ ${#cmd[@]} -eq 0 ]]; then
        _cwt_err "compose_up_bounded called with no command to run."
        return 64
    fi
    if ! command -v timeout >/dev/null 2>&1; then
        _cwt_err "'timeout' (GNU coreutils) not found in PATH -- cannot bound this compose call."
        _cwt_err "  Refusing to run it unbounded rather than silently reintroducing the hang."
        return 64
    fi

    local exit_code=0
    # --kill-after: if the wrapped process ignores SIGTERM (docker compose can,
    # while blocked deep in a health-poll loop), escalate to SIGKILL 15s later
    # so the deadline is a real deadline, not a suggestion.
    timeout --kill-after=15 "${timeout_seconds}" "${cmd[@]}" || exit_code=$?

    if [[ "${exit_code}" -eq 124 ]]; then
        _cwt_err "COMPOSE_UP_TIMEOUT: '${cmd[*]}' did not complete within ${timeout_seconds}s -- killed."
        _cwt_err "  Typed timeout, not a compose failure: the process was still running"
        _cwt_err "  (most likely blocked on a depends_on:condition:service_healthy dependency"
        _cwt_err "  that can never resolve) when the deadline hit. Increase"
        _cwt_err "  RUNTIME_COMPOSE_WAIT_TIMEOUT_SECONDS only after confirming the dependency"
        _cwt_err "  itself is not permanently stuck."
    fi
    return "${exit_code}"
}

reconcile_container_running_state() {
    # Usage: reconcile_container_running_state <container_ref> [<label>]
    #
    # <container_ref> is a container name or ID. Ensures it ends this call
    # either 'running' or torn down -- never left stranded in 'Created' or any
    # other ambiguous non-running state that requires manual `docker
    # start`/`docker ps` forensics to diagnose.
    local container_ref="$1"
    local label="${2:-${container_ref}}"

    if ! docker inspect "${container_ref}" >/dev/null 2>&1; then
        _cwt_log "  ${label}: no container found -- nothing to reconcile."
        return 0
    fi

    local status
    status="$(docker inspect "${container_ref}" --format '{{.State.Status}}' 2>/dev/null || echo unknown)"
    if [[ "${status}" == "running" ]]; then
        _cwt_log "  ${label}: already running."
        return 0
    fi

    _cwt_log "  ${label}: found in '${status}' state -- attempting docker start."
    docker start "${container_ref}" >/dev/null 2>&1 || true
    status="$(docker inspect "${container_ref}" --format '{{.State.Status}}' 2>/dev/null || echo unknown)"
    if [[ "${status}" == "running" ]]; then
        _cwt_log "  ${label}: recovered to running via docker start."
        return 0
    fi

    _cwt_err "  ${label}: STRANDED_CONTAINER -- state='${status}' after docker start attempt."
    _cwt_err "  ${label}: tearing down (docker rm -f) rather than leaving it stranded."
    docker rm -f "${container_ref}" >/dev/null 2>&1 || true
    return 1
}

container_status() {
    # Usage: container_status <container_ref>
    # Prints the container's .State.Status, or "absent" if it does not exist.
    # Used to build pre/post-rollback census comparisons.
    local container_ref="$1"
    if ! docker inspect "${container_ref}" >/dev/null 2>&1; then
        printf 'absent\n'
        return 0
    fi
    docker inspect "${container_ref}" --format '{{.State.Status}}' 2>/dev/null || printf 'unknown\n'
}
