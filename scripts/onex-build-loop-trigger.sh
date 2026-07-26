#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# onex-build-loop-trigger.sh — Publish a build-loop-orchestrator-start command
# into the target lane's Redpanda/Kafka broker via `docker exec ... rpk topic
# produce`.
#
# OMN-15179: the previous copy of this script (unversioned host state at
# ~/.local/bin/onex-build-loop-trigger.sh on omninode-pc, invoked daily by the
# onex-build-loop.service/.timer systemd user units) had two defects:
#
#   1. It hardcoded the dev-lane container name (`omnibase-infra-redpanda`),
#      which is not running on omninode-pc -- only stability-test/judge/prod
#      lanes are up there. The consumers this trigger is meant to reach
#      (node_build_loop_write_effect, node_build_loop_projection_compute)
#      live on the stability-test lane (verified via `rpk group list` against
#      omnibase-infra-stability-test-redpanda, 2026-07-26).
#   2. It piped into `docker exec` without checking the exit status, so a
#      "No such container" failure was immediately followed by an
#      unconditional "Build loop triggered" success line -- a false-success
#      control (§4a class: reports success and does nothing).
#
# This copy fixes both: the target container is a REQUIRED env var (fail-fast,
# no silent default -- per the no-silent-defaults rule), and the docker exec
# exit status is checked and treated as fatal.
#
# ENVIRONMENT (required, no defaults)
#   ONEX_BUILD_LOOP_REDPANDA_CONTAINER   Docker container name of the target
#                                        lane's Redpanda broker, e.g.
#                                        omnibase-infra-stability-test-redpanda
#
# USAGE
#   onex-build-loop-trigger.sh [mode]
#
#   mode   Build-loop mode string forwarded in the command payload
#          (default: "build")

set -euo pipefail

SCRIPT_NAME="$(basename "$0")"

# Fail-fast: no silent default. An absent/empty value here means the daily
# trigger has no idea which lane to reach -- better to die loudly than emit a
# false "triggered" line into the wrong container (or none at all).
if [[ -z "${ONEX_BUILD_LOOP_REDPANDA_CONTAINER:-}" ]]; then
    printf '%s: ERROR: ONEX_BUILD_LOOP_REDPANDA_CONTAINER is required and not set.\n' "${SCRIPT_NAME}" >&2
    printf '%s: Set it to the target lane redpanda container, e.g. omnibase-infra-stability-test-redpanda\n' "${SCRIPT_NAME}" >&2
    exit 1
fi

TARGET_CONTAINER="${ONEX_BUILD_LOOP_REDPANDA_CONTAINER}"
MODE="${1:-build}"

CORR_ID=$(python3 -c 'import uuid;print(uuid.uuid4())')
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
PAYLOAD=$(printf '{"event_type":"omnimarket.build-loop-orchestrator-start","correlation_id":"%s","payload":{"correlation_id":"%s","mode":"%s","max_cycles":1,"dry_run":false,"requested_at":"%s"}}' "$CORR_ID" "$CORR_ID" "$MODE" "$TS")

# `set -o pipefail` makes the pipeline's exit status the rightmost failing
# command's status (docker exec, not echo). Disable `set -e` around this one
# statement so a non-zero status is handled explicitly below instead of
# aborting the script before the loud error message prints.
set +e
echo "$PAYLOAD" | docker exec -i "${TARGET_CONTAINER}" rpk topic produce onex.cmd.omnimarket.build-loop-orchestrator-start.v1
STATUS=$?
set -e

if [[ "${STATUS}" -ne 0 ]]; then
    printf '%s: ERROR: docker exec into %s failed (exit %s) -- build loop was NOT triggered.\n' "${SCRIPT_NAME}" "${TARGET_CONTAINER}" "${STATUS}" >&2
    exit "${STATUS}"
fi

printf 'Build loop triggered: %s (mode=%s, container=%s)\n' "$CORR_ID" "$MODE" "$TARGET_CONTAINER"
