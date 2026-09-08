#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# deploy-agent-trigger.sh — publish a signed rebuild-requested command to the
# deploy-agent control bus (onex.cmd.deploy.rebuild-requested.v1).
#
# This file is a THIN WRAPPER. It resolves an interpreter that can import the
# deploy_agent package and hands every argument to `python -m deploy_agent.trigger`.
# It builds no JSON, computes no signature, and knows nothing about the bus.
#
# WHY (OMN-16442). This script used to do all three of those things in embedded
# snippets, and every one of them had drifted from the agent it drives:
#
#   * it hand-wrote the command JSON with a `reason` field and no
#     `runtime_lane`, while the agent's ModelRebuildRequested forbids the first
#     and requires the second — measured on the .201 dev lane, verbatim:
#     "2 validation errors for ModelRebuildRequested / runtime_lane / Field
#     required / reason / Extra inputs are not permitted";
#   * it read unprefixed KAFKA_SASL_USERNAME/PASSWORD while the lane env file
#     carries DEV_-prefixed names, then hardcoded SASL_SSL + PLAIN against a
#     broker running SASL_PLAINTEXT + SCRAM-SHA-256;
#   * with neither kcat nor rpk on PATH it fell through to a `docker exec ...
#     rpk --brokers localhost:9092` branch carrying no authentication at all,
#     which also defaults to snappy compression — a codec the agent's client
#     cannot decode, which crash-looped it until systemd gave up.
#
# The first of those three was closed by #3323, which corrected the hand-written
# envelope in place. This change removes the hand-writing instead: the envelope
# is built FROM ModelRebuildRequested and serialised, so the shape cannot drift
# again in the first place, and the same module resolves the transport through
# the loader the agent itself starts from. There is no unauthenticated fallback:
# a missing prerequisite is a refusal naming what is missing.
#
# THE OPERATOR-FACING CONTRACT OF #3323 IS PRESERVED — same flags, same
# refusals, same refusal wording — and is pinned by that PR's own end-to-end
# binding test (tests/unit/test_trigger_payload_matches_model_omn16442.py),
# which drives THIS script and validates its real --dry-run output against the
# real model.
#
# USAGE:
#   DEPLOY_AGENT_HMAC_SECRET=<secret> \
#   KAFKA_BOOTSTRAP_SERVERS=<host:port> \
#     ./deploy-agent-trigger.sh \
#       --runtime-lane dev \
#       --git-ref origin/dev \
#       [--scope runtime|core|full] \
#       [--build-source release|workspace] \
#       [--image-digest sha256:...] \
#       [--service <name>]... \
#       [--reason "manual trigger by operator"] \
#       [--requested-by claude] \
#       [--correlation-id <uuid>] \
#       [--dry-run]
#
# REQUIRED ARGS:
#   --runtime-lane   dev | stability-test | prod. The lane selects the compose
#                    overlay, compose project and health ports the agent will
#                    act on, so guessing it is the one mistake this script must
#                    never make. It resolves only from an explicit declaration:
#                    the flag, else a single-lane DEPLOY_AGENT_ALLOWED_LANES,
#                    else DEPLOY_AGENT_TRACKING_REF when that branch name is
#                    also a lane name. Nothing else resolves it and there is no
#                    literal default — an undeclared lane refuses, naming the
#                    flag.
#
# NOTE ON --reason (OMN-16442): `reason` is printed in this script's own audit
# output and is deliberately NOT part of the signed envelope.
# `ModelRebuildRequested` is declared `extra="forbid"`, so an envelope carrying
# `reason` is rejected wholesale at `consumer.poll_and_accept` — before
# `self_update`, before the env-contract validator, before anything runs.
#
# REQUIRED ENV VARS:
#   DEPLOY_AGENT_HMAC_SECRET   HMAC-SHA256 key — from the operator env file on
#                              .201. An unsigned command is silently dropped.
#   KAFKA_BOOTSTRAP_SERVERS    e.g. 192.168.86.201:19092 (local) or localhost:29092 (tunnel)  # onex-allow-internal-ip # cloud-bus-ok OMN-9411
#   KAFKA_SECURITY_PROTOCOL    PLAINTEXT | SSL | SASL_PLAINTEXT | SASL_SSL —
#                              declared, never inferred from whether credentials
#                              happen to be present (OMN-18012)
#   KAFKA_SASL_MECHANISM       required when the protocol is SASL_*
#   DEPLOY_AGENT_TRACKING_REF  branch this lane deploys, e.g. `dev`. REQUIRED
#                              only when --git-ref is omitted; it supplies the
#                              default as origin/<branch>. There is no built-in
#                              default (OMN-16442): the literal `origin/main`
#                              that used to sit here published deploy commands
#                              resetting the deploy-source clone onto a
#                              release-synced branch the lane was never on.
#
# OPTIONAL ENV VARS:
#   KAFKA_SASL_ENV_PREFIX      prefix the SASL principal is declared under on
#                              this host, e.g. DEV_ — the same variable the
#                              systemd unit sets, read by the same loader, so a
#                              lane whose credentials are prefixed resolves here
#                              exactly as it does in the agent
#   DEPLOY_AGENT_ALLOWED_LANES when it names exactly one lane, that lane is the
#                              default --runtime-lane
#   DEPLOY_AGENT_PYTHON        explicit interpreter; otherwise the agent venv
#                              beside this script, then $VIRTUAL_ENV, then python3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── resolve an interpreter that can import deploy_agent ──────────────────────
# Fail closed naming the missing prerequisite. The old script's third publish
# branch is exactly what "carry on with whatever is available" produced.
#
# The probe requires only `deploy_agent.trigger`, NOT `kafka`: building and
# signing a command is useful without a client (--dry-run), and a missing
# kafka-python is reported at publish time by the module, naming the package.
_candidates=()
if [[ -n "${DEPLOY_AGENT_PYTHON:-}" ]]; then
    _candidates+=("${DEPLOY_AGENT_PYTHON}")
fi
_candidates+=("${SCRIPT_DIR}/.venv/bin/python")
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    _candidates+=("${VIRTUAL_ENV}/bin/python")
fi
if command -v python3 &>/dev/null; then
    _candidates+=("$(command -v python3)")
fi

PYTHON=""
for _candidate in "${_candidates[@]}"; do
    [[ -x "$_candidate" ]] || continue
    if PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
        "$_candidate" -c 'import deploy_agent.trigger' &>/dev/null; then
        PYTHON="$_candidate"
        break
    fi
done

if [[ -z "$PYTHON" ]]; then
    echo "ERROR: no interpreter found that can import 'deploy_agent.trigger'." >&2
    echo "       Tried: ${_candidates[*]}" >&2
    echo "       Create the agent venv beside this script" >&2
    echo "       (${SCRIPT_DIR}/.venv), or point DEPLOY_AGENT_PYTHON at an" >&2
    echo "       interpreter that has this package and its dependencies." >&2
    echo "       Refusing to publish through an unauthenticated fallback." >&2
    exit 1
fi

exec env PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
    "$PYTHON" -m deploy_agent.trigger "$@"
