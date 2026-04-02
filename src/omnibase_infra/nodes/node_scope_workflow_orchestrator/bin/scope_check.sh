#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# Emit a scope-check command event to the ONEX event bus.
# Usage: scope_check.sh <plan-file-path> [--output <manifest-path>] [--confirm]
#
# This is the bin-shell dispatcher for the scope-check skill-to-node canary.
# It emits a command event to Kafka that triggers the scope_workflow_orchestrator.

set -euo pipefail

PLAN_FILE="${1:?Usage: scope_check.sh <plan-file-path> [--output <path>] [--confirm]}"
OUTPUT_PATH="${SCOPE_MANIFEST_PATH:-$HOME/.claude/scope-manifest.json}"
CONFIRM="false"

shift
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) OUTPUT_PATH="$2"; shift 2 ;;
    --confirm) CONFIRM="true"; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

CORRELATION_ID="$(uuidgen | tr '[:upper:]' '[:lower:]')"
TOPIC="onex.cmd.skill.scope-check.v1"

PAYLOAD=$(cat <<EOF
{
  "correlation_id": "${CORRELATION_ID}",
  "plan_file_path": "${PLAN_FILE}",
  "output_path": "${OUTPUT_PATH}",
  "auto_confirm": ${CONFIRM}
}
EOF
)

echo "${PAYLOAD}" | kcat -P -b "${KAFKA_BOOTSTRAP_SERVERS:-localhost:19092}" -t "${TOPIC}"

echo "scope-check command emitted"
echo "  correlation_id: ${CORRELATION_ID}"
echo "  topic: ${TOPIC}"
echo "  plan_file: ${PLAN_FILE}"
echo "  output: ${OUTPUT_PATH}"
