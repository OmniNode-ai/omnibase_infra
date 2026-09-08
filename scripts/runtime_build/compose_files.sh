#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# compose_files.sh -- the ONE derivation of a lane's `docker compose -f ...`
# token sequence [OMN-16729, extending OMN-13581 / OMN-15379].
#
# Sourced by scripts/deploy-runtime.sh, scripts/runtime_build/refresh_dev_lane.sh
# and scripts/runtime_build/refresh_stability_lane.sh. Same shared-helper shape
# as compose_wait_timeout.sh and lane_lock.sh, which those three already share.
#
# WHY THIS FILE EXISTS, measured on the .201 dev lane 2026-09-08T18:48:59Z:
# deploy-runtime.sh resolved both compose files correctly, but
# refresh_dev_lane.sh carried its OWN, SEPARATE compose invocations -- the
# service-id resolver and the failure ROLLBACK recreate -- and each of those
# spelled a single `-f docker/docker-compose.infra.yml` by hand. The dev lane's
# overlay, docker/docker-compose.dev-lane.yml, is the sole declaration of the
# four KAFKA_SASL_* / KAFKA_SECURITY_PROTOCOL vars for the runtime family. The
# rollback recreated all four core containers WITHOUT it, on a broker that had
# required SASL since 18:16:19Z, so omninode-runtime crash-looped on
# KafkaConnectionError and compose never started runtime-effects or
# runtime-worker behind `depends_on: omninode-runtime: service_healthy`. Both
# sat at State=created with zero log lines for 53 minutes.
#
# The recreate path is the one that MUST be right: it runs when something has
# already gone wrong, on a lane nobody is watching closely. A second hand-spelled
# copy of the file list is how that path silently loses an overlay, so there is
# exactly one copy and every caller reads it.
#
# This file defines functions only. It sets no options and mutates nothing at
# source time, so it is safe to source from a script that has already set
# `set -euo pipefail`.

# Emit an error through the caller's own logger when it has one, so a message
# from this lib is indistinguishable from the host script's own. Resolved at
# CALL time, not source time: deploy-runtime.sh sources this file above its own
# log_error definition.
_compose_files_log_error() {
    if declare -F log_error >/dev/null 2>&1; then
        log_error "$@"
    else
        printf '[compose-files] ERROR: %s\n' "$*" >&2
    fi
}

# Compose project -> lane (overlay) mapping. EVERY lane layers exactly two
# files: docker-compose.infra.yml plus its own overlay. A non-dev lane layers
# docker-compose.<lane>.yml so the overlay's container_name + project name +
# lane network win; the bare omnibase-infra project layers
# docker-compose.dev-lane.yml (OMN-15379, OMN-18012 phase B). The older
# "the dev project gets no overlay" reading of this mapping is WRONG and is the
# sentence the 2026-09-08 rollback was written against.
#
# OMN-13581: deploy-runtime.sh historically passed ONLY `-f infra.yml` on every
# `docker compose` call, including warm_broker_topic_provisioning's `up redpanda`
# step. The base infra compose hardcodes `container_name: omnibase-infra-redpanda`
# (the DEV name) and the dev network, so running the warmup against a non-dev
# project (e.g. omnibase-infra-stability-test) makes compose try to (re)create
# redpanda as the DEV-named container, which collides with the live dev broker,
# gets a Docker hash prefix, and lands in 'created' -- DESTROYING the lane's own
# correctly-named broker. That left the stability lane broker-less for ~3 days.
# Layering the matching overlay gives redpanda the lane-prefixed container_name +
# lane network, so the lane's broker is targeted and never displaced.
#
# This mirrors the authoritative, tested lane->compose-file mapping in
# scripts/deploy-agent/deploy_agent/executor.py (_LANE_CONFIGS): stability-test
# layers docker-compose.stability-test.yml, prod layers docker-compose.prod.yml,
# judge layers docker-compose.judge.yml.
resolve_lane_name() {
    # Echo the LANE name derived from a compose project (OMN-15218).
    #   omnibase-infra                -> dev
    #   omnibase-infra-stability-test -> stability-test
    #   omnibase-infra-prod           -> prod
    #   omnibase-infra-judge          -> judge
    # Single derivation shared by the hot-patch preflight and the lane-deploy
    # attribution guard, so one deploy can never be recorded under two different
    # lane names. Unknown suffixes echo through unchanged; the callers that must
    # fail closed on an unknown lane (resolve_lane_overlay_filename) do their own
    # allowlist check.
    local compose_project="$1"
    local lane="${compose_project#omnibase-infra}"
    lane="${lane#-}"
    if [[ -z "${lane}" ]]; then
        lane="dev"
    fi
    echo "${lane}"
}

resolve_lane_overlay_filename() {
    # Echo the overlay compose FILENAME (relative to docker/) for a compose
    # project, or nothing for the bare dev project. Fails closed: an unknown
    # non-dev project aborts rather than silently running on the dev config (the
    # exact failure mode that displaced the lane broker).
    local compose_project="$1"

    # Lane = compose project suffix after the canonical "omnibase-infra" prefix.
    # omnibase-infra                -> "" (dev, no lane-named overlay)
    # omnibase-infra-stability-test -> "stability-test"
    # omnibase-infra-prod           -> "prod"
    # omnibase-infra-judge          -> "judge"
    local lane="${compose_project#omnibase-infra}"
    lane="${lane#-}"

    case "${lane}" in
        "")
            # Dev lane: no docker-compose.dev.yml exists. The dev overlay is
            # docker-compose.dev-lane.yml and is appended by
            # resolve_compose_file_args' else-branch, not named here.
            return 0
            ;;
        stability-test|prod|judge)
            echo "docker-compose.${lane}.yml"
            return 0
            ;;
        *)
            _compose_files_log_error "Unknown lane '${lane}' derived from compose project '${compose_project}'."
            _compose_files_log_error "  Only the dev / stability-test / prod / judge lanes are known."
            _compose_files_log_error "  Refusing to deploy: running a non-dev lane on the bare infra.yml config"
            _compose_files_log_error "  would recreate the DEV-named redpanda and displace this lane's broker"
            _compose_files_log_error "  (OMN-13581). Add the lane's overlay mapping before deploying it."
            exit 1
            ;;
    esac
}

resolve_compose_file_args() {
    # Populate a caller-provided array (passed by name) with the full
    # `-f <file>` token sequence for a lane: always docker-compose.infra.yml,
    # plus that lane's overlay -- docker-compose.<lane>.yml for a non-dev
    # project, docker-compose.dev-lane.yml for the bare dev project.
    #
    # EVERY compose invocation against a lane goes through this, including the
    # read-only ones and especially the rollback recreate. A lane's overlay is
    # not optional decoration: on dev it carries the runtime family's whole
    # broker-auth environment, and on stability/prod/judge it carries the
    # lane-prefixed container names and network.
    #
    # Usage:
    #   local -a compose_args
    #   resolve_compose_file_args compose_args "${deploy_target}" "${compose_project}"
    #   docker compose -p "${compose_project}" "${compose_args[@]}" ...
    #
    # `deploy_target` is the directory that CONTAINS docker/ -- the versioned
    # deploy root for deploy-runtime.sh, the ambient clone for the refresh
    # wrappers. Both spell the same files.
    local _out_args_name="$1"
    local deploy_target="$2"
    local compose_project="$3"

    local docker_dir="${deploy_target}/docker"
    eval "${_out_args_name}=(-f $(printf '%q' "${docker_dir}/docker-compose.infra.yml"))"

    local overlay_filename
    overlay_filename="$(resolve_lane_overlay_filename "${compose_project}")"
    if [[ -n "${overlay_filename}" ]]; then
        eval "${_out_args_name}+=( -f $(printf '%q' "${docker_dir}/${overlay_filename}") )"
    else
        # Dev/lab lane (bare omnibase-infra project). OMN-15379: layer the
        # dev-lane overlay, whose content includes ONEX_MIGRATION_LANE=dev for
        # forward-migration -- the lane indicator that releases the
        # node_projection_registration trio (operator ruling 15, lab lane is the
        # FORCE proving ground) -- and, since OMN-18012 phase B, the runtime
        # family's KAFKA_SECURITY_PROTOCOL / KAFKA_SASL_* environment. It is a
        # separate file precisely so no non-dev lane can inherit it from the
        # base: see the header of docker/docker-compose.dev-lane.yml.
        eval "${_out_args_name}+=( -f $(printf '%q' "${docker_dir}/docker-compose.dev-lane.yml") )"
    fi
}
