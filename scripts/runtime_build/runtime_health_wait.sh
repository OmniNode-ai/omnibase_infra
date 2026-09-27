#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# runtime_health_wait.sh -- OMN-18349: the deploy health wait honours the
# runtime container's OWN declared health budget.
#
# Measured defect (.201, 2026-09-23T21:33Z): a governed
# `refresh_stability_lane.sh --ref origin/dev --execute` recreated the
# stability-test runtime on 0.38.57. deploy-runtime.sh then polled /health 15
# times 4 s apart, about 60 s, while the runtime took about 73 s to boot
# (/health 503 at 21:34:13Z, docker `healthy` at 21:34:33Z). The deploy
# declared the lane dead, re-tagged every image to its pre-build id and wrote
# no refresh receipt, leaving healthy containers on the new build under tags
# naming the old one.
#
# The runtime container declares how long it may take to start (its compose
# healthcheck: start_period, interval, timeout, retries). A deploy that gives
# up earlier than that is judging the container against a budget nobody
# declared. These helpers let the verify loop keep waiting past its fixed
# floor ONLY while docker itself still reports the container `starting`, and
# only within the container's declared budget. A container that is unhealthy,
# not running or absent stops the wait at once, so a crash still fails fast.
#
# This file is SOURCED, not executed.
#
# Provides:
#   runtime_health_budget_seconds <container>
#       Print start_period + retries x (interval + timeout), in whole seconds,
#       from the container's declared healthcheck. Prints 0 when the container
#       declares none or cannot be inspected.
#   runtime_started_at <container>
#       Print the container's State.StartedAt, or nothing if it cannot be read.
#   runtime_health_state <container> [<started_at_baseline>]
#       Print one of: starting | healthy | unhealthy | no-healthcheck |
#       not-running | restarted | absent. `restarted` means State.StartedAt no
#       longer equals the baseline: a crash-looping container re-enters its
#       start period on every restart, so without this it would read
#       `starting` for the whole declared budget.
#   runtime_health_keep_waiting <elapsed_s> <budget_s> <state>
#       Return 0 to keep polling: the state is `starting` or `healthy` (docker's
#       own check passed; the host probe gets the rest of the budget) and the
#       elapsed time is below the budget. Return 1 otherwise.

runtime_health_budget_seconds() {
    # The durations go through printf "%d": a bare {{.Config.Healthcheck.X}}
    # renders a Go duration ("30m0s"), not nanoseconds, and the budget would
    # silently read 0 (caught by the live readback on .201, 2026-09-23).
    local container="$1"
    local declared start_ns interval_ns retries timeout_ns
    if ! declared="$(docker inspect --format \
        '{{if .Config.Healthcheck}}{{printf "%d %d %d %d" .Config.Healthcheck.StartPeriod .Config.Healthcheck.Interval .Config.Healthcheck.Retries .Config.Healthcheck.Timeout}}{{else}}0 0 0 0{{end}}' \
        "${container}" 2>/dev/null)"; then
        echo 0
        return 0
    fi
    read -r start_ns interval_ns retries timeout_ns <<<"${declared}"
    if ! [[ "${start_ns:-}" =~ ^[0-9]+$ && "${interval_ns:-}" =~ ^[0-9]+$ \
        && "${retries:-}" =~ ^[0-9]+$ && "${timeout_ns:-}" =~ ^[0-9]+$ ]]; then
        echo 0
        return 0
    fi
    echo $(( (start_ns + retries * (interval_ns + timeout_ns)) / 1000000000 ))
}

runtime_started_at() {
    docker inspect --format '{{.State.StartedAt}}' "$1" 2>/dev/null || true
}

runtime_health_state() {
    local container="$1"
    local baseline="${2:-}"
    local reported status health started_at
    if ! reported="$(docker inspect --format \
        '{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}} {{.State.StartedAt}}' \
        "${container}" 2>/dev/null)"; then
        echo absent
        return 0
    fi
    read -r status health started_at <<<"${reported}"
    if [[ -z "${status:-}" ]]; then
        echo absent
    elif [[ "${status}" != "running" ]]; then
        echo not-running
    elif [[ -n "${baseline}" && "${started_at:-}" != "${baseline}" ]]; then
        echo restarted
    else
        case "${health:-none}" in
            starting | healthy | unhealthy) echo "${health}" ;;
            *) echo no-healthcheck ;;
        esac
    fi
}

runtime_health_keep_waiting() {
    local elapsed="$1" budget="$2" state="$3"
    case "${state}" in
        starting | healthy) (( elapsed < budget )) ;;
        *) return 1 ;;
    esac
}
