#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# deploy-agent-trigger.sh — publish a signed rebuild-requested command to the
# deploy-agent Kafka topic (onex.cmd.deploy.rebuild-requested.v1).
#
# USAGE:
#   DEPLOY_AGENT_HMAC_SECRET=<secret> \
#   KAFKA_BOOTSTRAP_SERVERS=<host:port> \
#     ./deploy-agent-trigger.sh \
#       --git-ref origin/main \
#       --reason "manual trigger by operator" \
#       [--requested-by claude] \
#       [--correlation-id <uuid>] \
#       [--dry-run]
#
# REQUIRED ENV VARS:
#   DEPLOY_AGENT_HMAC_SECRET   HMAC-SHA256 key — from ~/.omnibase/.env on .201
#   KAFKA_BOOTSTRAP_SERVERS    e.g. 192.168.86.201:19092 (local) or localhost:29092 (tunnel)  # onex-allow-internal-ip # cloud-bus-ok OMN-9411
#
# OPTIONAL ENV VARS:
#   KAFKA_SASL_USERNAME        SASL username (omit for PLAINTEXT connections)
#   KAFKA_SASL_PASSWORD        SASL password
#
# WARNING: Unsigned triggers are silently dropped by auth.py.
# NEVER construct the command JSON manually — always use this script so the
# HMAC-SHA256 _signature field is computed correctly.
#
# The signature is HMAC-SHA256 over the JSON-serialised envelope (sort_keys,
# no spaces) with the _signature field itself excluded — matching
# deploy_agent/auth.py::verify_command() byte-for-byte.

set -euo pipefail

TOPIC="onex.cmd.deploy.rebuild-requested.v1"

# ── defaults ─────────────────────────────────────────────────────────────────
GIT_REF="origin/main"
REASON=""
REQUESTED_BY="operator-manual"
CORRELATION_ID=""
DRY_RUN=0

usage() {
    grep '^# ' "$0" | sed 's/^# //'
    exit 1
}

# ── arg parsing ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --git-ref)         GIT_REF="$2";         shift 2 ;;
        --reason)          REASON="$2";           shift 2 ;;
        --requested-by)    REQUESTED_BY="$2";     shift 2 ;;
        --correlation-id)  CORRELATION_ID="$2";   shift 2 ;;
        --dry-run)         DRY_RUN=1;             shift   ;;
        -h|--help)         usage ;;
        *) echo "Unknown arg: $1" >&2; usage ;;
    esac
done

# ── pre-flight ───────────────────────────────────────────────────────────────
if [[ -z "${DEPLOY_AGENT_HMAC_SECRET:-}" ]]; then
    echo "ERROR: DEPLOY_AGENT_HMAC_SECRET is not set." >&2
    echo "       Source it with: source ~/.omnibase/.env" >&2
    exit 1
fi

if [[ $DRY_RUN -eq 0 && -z "${KAFKA_BOOTSTRAP_SERVERS:-}" ]]; then
    echo "ERROR: KAFKA_BOOTSTRAP_SERVERS is not set." >&2
    exit 1
fi

if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 is required to compute the HMAC signature." >&2
    exit 1
fi

# kcat is used for publish; rpk is accepted as fallback on .201
_publish_tool=""
if command -v kcat &>/dev/null; then
    _publish_tool="kcat"
elif command -v rpk &>/dev/null; then
    _publish_tool="rpk"
elif [[ $DRY_RUN -eq 0 ]]; then
    echo "ERROR: kcat (or rpk) not found. Install kcat: brew install kcat" >&2
    exit 1
fi

# ── build envelope + compute signature ───────────────────────────────────────
if [[ -z "$CORRELATION_ID" ]]; then
    CORRELATION_ID="$(python3 -c 'import uuid; print(uuid.uuid4())')"
fi

# All user-supplied values are passed via environment variables — never
# interpolated into Python source — so special characters cannot break
# JSON structure or inject code.
SIGNED_JSON="$(
    _TRIGGER_GIT_REF="$GIT_REF" \
    _TRIGGER_REASON="$REASON" \
    _TRIGGER_REQUESTED_BY="$REQUESTED_BY" \
    _TRIGGER_CORRELATION_ID="$CORRELATION_ID" \
    python3 - <<'PYEOF'
import hashlib
import hmac
import json
import os
from datetime import UTC, datetime

secret         = os.environ["DEPLOY_AGENT_HMAC_SECRET"]
correlation_id = os.environ["_TRIGGER_CORRELATION_ID"]
git_ref        = os.environ["_TRIGGER_GIT_REF"]
reason         = os.environ["_TRIGGER_REASON"]
requested_by   = os.environ["_TRIGGER_REQUESTED_BY"]

envelope = {
    "correlation_id": correlation_id,
    "git_ref":        git_ref,
    "reason":         reason,
    "requested_at":   datetime.now(UTC).isoformat(),
    "requested_by":   requested_by,
    "scope":          "runtime",
    "services":       [],
}
body = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode()
sig  = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
signed = {**envelope, "_signature": sig}
print(json.dumps(signed, separators=(",", ":")))
PYEOF
)"

# ── audit log (signature masked) ─────────────────────────────────────────────
# SIGNED_JSON is passed via environment variable — never interpolated into
# Python source — so embedded quotes or special characters cannot break syntax.
MASKED_JSON="$(
    _TRIGGER_SIGNED_JSON="${SIGNED_JSON}" python3 - <<'PYEOF'
import json, os
d = json.loads(os.environ["_TRIGGER_SIGNED_JSON"])
d["_signature"] = d["_signature"][:8] + "...<masked>"
print(json.dumps(d, indent=2))
PYEOF
)"

echo "=== deploy-agent-trigger ==="
echo "topic:          ${TOPIC}"
echo "git_ref:        ${GIT_REF}"
echo "correlation_id: ${CORRELATION_ID}"
echo "requested_by:   ${REQUESTED_BY}"
echo "payload (sig masked):"
echo "${MASKED_JSON}"

if [[ $DRY_RUN -eq 1 ]]; then
    echo ""
    echo "(dry-run: skipping Kafka publish)"
    exit 0
fi

# ── publish ──────────────────────────────────────────────────────────────────
echo ""
echo "Publishing to ${KAFKA_BOOTSTRAP_SERVERS} ..."

if [[ "$_publish_tool" == "kcat" ]]; then
    KCAT_ARGS=(-P -b "${KAFKA_BOOTSTRAP_SERVERS}" -t "${TOPIC}" -K /)
    if [[ -n "${KAFKA_SASL_USERNAME:-}" && -n "${KAFKA_SASL_PASSWORD:-}" ]]; then
        KCAT_ARGS+=(
            -X security.protocol=SASL_SSL
            -X sasl.mechanisms=PLAIN
            -X "sasl.username=${KAFKA_SASL_USERNAME}"
            -X "sasl.password=${KAFKA_SASL_PASSWORD}"
        )
    fi
    # SIGNED_JSON piped via stdin — not passed as a shell argument
    echo "manual-${CORRELATION_ID}/${SIGNED_JSON}" | kcat "${KCAT_ARGS[@]}"
else
    # rpk fallback — used on .201 where rpk is available inside/outside containers
    RPK_ARGS=(--brokers "${KAFKA_BOOTSTRAP_SERVERS}" --key "manual-${CORRELATION_ID}")
    if [[ -n "${KAFKA_SASL_USERNAME:-}" && -n "${KAFKA_SASL_PASSWORD:-}" ]]; then
        RPK_ARGS+=(
            --tls-enabled
            --sasl-mechanism PLAIN
            --sasl-username "${KAFKA_SASL_USERNAME}"
            --sasl-password "${KAFKA_SASL_PASSWORD}"
        )
    fi
    rpk topic produce "${TOPIC}" "${RPK_ARGS[@]}" <<< "${SIGNED_JSON}"
fi

echo "Published. correlation_id=${CORRELATION_ID}"
