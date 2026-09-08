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
#       --runtime-lane dev \
#       --git-ref origin/dev \
#       [--build-source release|workspace] \
#       [--image-digest sha256:...] \
#       [--reason "manual trigger by operator"] \
#       [--requested-by claude] \
#       [--correlation-id <uuid>] \
#       [--dry-run]
#
# REQUIRED ARGS:
#   --runtime-lane   dev | stability-test | prod. There is NO default: the lane
#                    selects the compose overlay, compose project and health
#                    ports the agent will act on, so guessing it is the one
#                    mistake this script must never make. Mirrors the required
#                    `runtime_lane` field of ModelRebuildRequested.
#
# NOTE ON --reason (OMN-16442): `reason` is printed in this script's own audit
# output and is deliberately NOT part of the signed envelope.
# `ModelRebuildRequested` is declared `extra="forbid"`, so an envelope carrying
# `reason` is rejected wholesale at `consumer.poll_and_accept` — before
# `self_update`, before the env-contract validator, before anything runs. The
# envelope this script signs must contain exactly the model's own fields and
# nothing else; `tests/unit/test_trigger_payload_matches_model_omn16442.py`
# validates this script's real `--dry-run` output against the real model so the
# two cannot drift apart again unnoticed.
#
# REQUIRED ENV VARS:
#   DEPLOY_AGENT_HMAC_SECRET   HMAC-SHA256 key — from ~/.omnibase/.env on .201
#   KAFKA_BOOTSTRAP_SERVERS    e.g. 192.168.86.201:19092 (local) or localhost:29092 (tunnel)  # onex-allow-internal-ip # cloud-bus-ok OMN-9411
#   DEPLOY_AGENT_TRACKING_REF  branch this lane deploys, e.g. `dev`. REQUIRED
#                              only when --git-ref is omitted; it supplies the
#                              default as origin/<branch>. There is no built-in
#                              default (OMN-16442): the literal `origin/main`
#                              that used to sit here published deploy commands
#                              resetting the deploy-source clone onto a
#                              release-synced branch the lane was never on.
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
# No default here: resolved from DEPLOY_AGENT_TRACKING_REF after arg parsing,
# and only when --git-ref was not supplied (OMN-16442).
GIT_REF=""
# No default lane, by design (rule 8): an undeclared lane must abort naming the
# flag, never silently pick one.
RUNTIME_LANE=""
BUILD_SOURCE=""
IMAGE_DIGEST=""
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
        --runtime-lane)    RUNTIME_LANE="$2";     shift 2 ;;
        --build-source)    BUILD_SOURCE="$2";     shift 2 ;;
        --image-digest)    IMAGE_DIGEST="$2";     shift 2 ;;
        --reason)          REASON="$2";           shift 2 ;;
        --requested-by)    REQUESTED_BY="$2";     shift 2 ;;
        --correlation-id)  CORRELATION_ID="$2";   shift 2 ;;
        --dry-run)         DRY_RUN=1;             shift   ;;
        -h|--help)         usage ;;
        *) echo "Unknown arg: $1" >&2; usage ;;
    esac
done

# ── resolve the runtime lane ─────────────────────────────────────────────────
# Fail-fast, no default. `runtime_lane` is a REQUIRED field of
# ModelRebuildRequested (deploy_agent/events.py) and selects the lane the agent
# will actually mutate; a wrong or guessed value is a deploy against the wrong
# compose project. The accepted set is EnumRuntimeLane's own members.
if [[ -z "$RUNTIME_LANE" ]]; then
    echo "ERROR: --runtime-lane is required (dev | stability-test | prod)." >&2
    echo "       It has no default: the lane selects the compose overlay," >&2
    echo "       compose project and health ports the deploy acts on." >&2
    exit 1
fi

case "$RUNTIME_LANE" in
    dev|stability-test|prod) ;;
    *)
        echo "ERROR: unknown --runtime-lane '${RUNTIME_LANE}'." >&2
        echo "       Accepted: dev, stability-test, prod (EnumRuntimeLane)." >&2
        exit 1 ;;
esac

if [[ -n "$BUILD_SOURCE" ]]; then
    case "$BUILD_SOURCE" in
        workspace|release) ;;
        *)
            echo "ERROR: unknown --build-source '${BUILD_SOURCE}'." >&2
            echo "       Accepted: workspace, release (BuildSource)." >&2
            exit 1 ;;
    esac
fi

# Prod deploys a pinned, stability-proven digest and never rebuilds from a ref;
# the model enforces this too, but refusing here means the operator is told
# before a command is signed and published rather than after it is rejected.
if [[ "$RUNTIME_LANE" == "prod" && -z "$IMAGE_DIGEST" ]]; then
    echo "ERROR: --runtime-lane prod requires --image-digest sha256:..." >&2
    echo "       Production deploys the exact stability-proven digest." >&2
    exit 1
fi

# ── resolve the deploy ref ───────────────────────────────────────────────────
# Fail-fast rather than defaulting: an undeclared tracking ref is exactly the
# defect this removes (operator ruling 2026-09-08 — track the lane deploy
# branch, never `main`).
if [[ -z "$GIT_REF" ]]; then
    if [[ -z "${DEPLOY_AGENT_TRACKING_REF:-}" ]]; then
        echo "ERROR: no --git-ref given and DEPLOY_AGENT_TRACKING_REF is not set." >&2
        echo "       Pass --git-ref origin/<branch>, or export" >&2
        echo "       DEPLOY_AGENT_TRACKING_REF=<branch> (e.g. dev) to supply the default." >&2
        exit 1
    fi
    GIT_REF="origin/${DEPLOY_AGENT_TRACKING_REF}"
fi

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
elif command -v docker &>/dev/null && docker exec omnibase-infra-redpanda rpk version &>/dev/null; then
    _publish_tool="docker-rpk"
elif [[ $DRY_RUN -eq 0 ]]; then
    echo "ERROR: kcat, rpk, or dockerized redpanda rpk not found." >&2
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
    _TRIGGER_RUNTIME_LANE="$RUNTIME_LANE" \
    _TRIGGER_BUILD_SOURCE="$BUILD_SOURCE" \
    _TRIGGER_IMAGE_DIGEST="$IMAGE_DIGEST" \
    _TRIGGER_REQUESTED_BY="$REQUESTED_BY" \
    _TRIGGER_CORRELATION_ID="$CORRELATION_ID" \
    python3 - <<'PYEOF'
import hashlib
import hmac
import json
import os
secret         = os.environ["DEPLOY_AGENT_HMAC_SECRET"]
correlation_id = os.environ["_TRIGGER_CORRELATION_ID"]
git_ref        = os.environ["_TRIGGER_GIT_REF"]
requested_by   = os.environ["_TRIGGER_REQUESTED_BY"]
runtime_lane   = os.environ["_TRIGGER_RUNTIME_LANE"]
build_source   = os.environ["_TRIGGER_BUILD_SOURCE"]
image_digest   = os.environ["_TRIGGER_IMAGE_DIGEST"]

# Exactly the fields ModelRebuildRequested declares, and nothing else: the
# model is extra="forbid", so one stray key rejects the whole command. The
# optional fields are omitted rather than sent empty so the model's own
# defaults apply.
envelope = {
    "correlation_id": correlation_id,
    "git_ref":        git_ref,
    "requested_by":   requested_by,
    "runtime_lane":   runtime_lane,
    "scope":          "runtime",
    "services":       [],
}
if build_source:
    envelope["build_source"] = build_source
if image_digest:
    envelope["image_digest"] = image_digest
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
echo "runtime_lane:   ${RUNTIME_LANE}"
echo "git_ref:        ${GIT_REF}"
echo "correlation_id: ${CORRELATION_ID}"
echo "requested_by:   ${REQUESTED_BY}"
# Printed, not signed: see the NOTE ON --reason in the header.
echo "reason:         ${REASON:-(none)}  [local audit only, not in envelope]"
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
elif [[ "$_publish_tool" == "rpk" ]]; then
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
else
    docker exec -i omnibase-infra-redpanda \
        rpk topic produce "${TOPIC}" --brokers localhost:9092 \
        --key "manual-${CORRELATION_ID}" <<< "${SIGNED_JSON}"
fi

echo "Published. correlation_id=${CORRELATION_ID}"
