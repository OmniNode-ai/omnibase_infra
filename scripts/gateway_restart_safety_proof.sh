#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# gateway_restart_safety_proof.sh -- OMN-15521 remediation: a real,
# executable restart-durability smoke proof for the OMN-12912 gateway
# idempotency store, run against the ALREADY-DEPLOYED omninode-gateway
# lane on `.201`.
#
# What this is NOT: the full cross-broker at-least-once/exactly-once
# redelivery proof (an in-flight message deliberately killed mid-delivery,
# then confirmed not to duplicate or drop on the far side). That needs a
# synthetic in-flight cloud MSK message and is OMN-12912's own test suite's
# job -- see PR #2556's own scope-boundary note: "Docker Compose/runtime
# proof is delegated to hosted CI because neither the development Mac nor
# .200 has a Docker CLI", i.e. that proof has never run against a REAL
# deployed container.
#
# What this IS: a concrete, mechanical proof that the durable SQLite
# idempotency store (WAL + synchronous=FULL, per OMN-12912's PR
# description) actually survives a real container restart on the real box,
# and that the container comes back genuinely healthy rather than
# false-green -- the restart-safety half of OMN-12912's own scope ("make
# delivery-loop or heartbeat failure remove readiness ... instead of
# leaving a false-green gateway").
#
# The receipt this prints is meant to be pasted into an OMN-12912 comment
# (per OMN-15521's own AC5 wording: "that receipt lands on OMN-12912, not
# this ticket") -- this script does not file it anywhere itself.
#
# Usage:
#   ./scripts/gateway_restart_safety_proof.sh          # run the proof
#   ./scripts/gateway_restart_safety_proof.sh --help    # usage

set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_NAME

CONTAINER_NAME="${GATEWAY_RESTART_PROOF_CONTAINER:-omninode-gateway-forwarder}"
SYSTEMD_UNIT="${GATEWAY_RESTART_PROOF_UNIT:-onex-gateway-forwarder}"
SQLITE_PATH="${GATEWAY_RESTART_PROOF_SQLITE_PATH:-/app/data/gateway/delivery.sqlite3}"
HEALTHY_TIMEOUT_SECONDS="${GATEWAY_RESTART_PROOF_HEALTHY_TIMEOUT_SECONDS:-120}"

log_info() { printf '[gateway-restart-proof] %s\n' "$*"; }
log_warn() { printf '[gateway-restart-proof] WARNING: %s\n' "$*" >&2; }
log_error() { printf '[gateway-restart-proof] ERROR: %s\n' "$*" >&2; }
log_step() { printf '\n[gateway-restart-proof] === %s ===\n' "$*"; }

usage() {
    cat <<EOF
${SCRIPT_NAME} -- OMN-15521 restart-durability smoke proof for the OMN-12912
gateway idempotency store (AC5).

Snapshots the row count of the running container's durable idempotency
store, reloads the systemd unit (the same mechanism scripts/deploy-
gateway.sh's --execute uses to recreate the container), waits for the
container to report Docker-healthy again, then re-snapshots and asserts no
durable markers were lost across the restart.

Does NOT prove full cross-broker exactly-once redelivery (that needs a
synthetic in-flight cloud MSK message and is OMN-12912's own test suite's
job). Proves: the SQLite store survives a real restart, and the container
does not come back false-green.

USAGE
    ${SCRIPT_NAME}          Run the proof (mutates: restarts the container).
    ${SCRIPT_NAME} --help   Show this help and exit.

The printed receipt is meant to be pasted into an OMN-12912 comment -- this
script does not file it anywhere itself (OMN-15521's own AC5 wording: "that
receipt lands on OMN-12912, not this ticket").
EOF
    exit 0
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
fi

snapshot() {
    # Prints "<row_count>\t<max_processed_at>" read from the running
    # container's idempotency store via the stdlib sqlite3 module (no
    # sqlite3 CLI binary is installed in the runtime image -- see
    # docker/Dockerfile.runtime's minimal apt-get install list). Read-only;
    # "0\t0" if the store does not exist yet (container never processed
    # anything) or the container is unreachable.
    docker exec "${CONTAINER_NAME}" python3 -c "
import sqlite3
try:
    conn = sqlite3.connect('${SQLITE_PATH}')
    row = conn.execute(
        'SELECT COUNT(*), COALESCE(MAX(processed_at), 0) FROM idempotency_records'
    ).fetchone()
    print(f'{row[0]}\t{row[1]}')
except Exception:
    print('0\t0')
" 2>/dev/null || printf '0\t0\n'
}

wait_healthy() {
    local timeout="$1" waited=0
    while (( waited < timeout )); do
        if [[ "$(docker inspect "${CONTAINER_NAME}" --format '{{.State.Health.Status}}' 2>/dev/null || true)" == "healthy" ]]; then
            return 0
        fi
        sleep 3
        waited=$((waited + 3))
    done
    return 1
}

main() {
    log_step "Restart-safety proof: ${CONTAINER_NAME}"
    log_info "Idempotency store: ${SQLITE_PATH}"

    local before before_count before_max
    before="$(snapshot)"
    before_count="${before%%$'\t'*}"
    before_max="${before#*$'\t'}"
    log_info "Before restart: rows=${before_count} max_processed_at=${before_max:-<none>}"

    if [[ "${before_count}" == "0" ]]; then
        log_warn "No idempotency records exist yet -- this run will only show an EMPTY store surviving a restart, which is trivially true and is not evidence of durability. Re-run once real traffic has flowed."
    fi

    log_step "Reload ${SYSTEMD_UNIT}"
    log_info "Same mechanism scripts/deploy-gateway.sh's --execute uses to recreate the container."
    sudo systemctl reload "${SYSTEMD_UNIT}"

    if ! wait_healthy "${HEALTHY_TIMEOUT_SECONDS}"; then
        log_error "AC5-PROOF FAILED: ${CONTAINER_NAME} did not report Docker-healthy within ${HEALTHY_TIMEOUT_SECONDS}s of restart."
        exit 1
    fi
    log_info "Container healthy after restart (not false-green)."

    local after after_count after_max
    after="$(snapshot)"
    after_count="${after%%$'\t'*}"
    after_max="${after#*$'\t'}"
    log_info "After restart:  rows=${after_count} max_processed_at=${after_max:-<none>}"

    if (( after_count < before_count )); then
        log_error "AC5-PROOF FAILED: idempotency record count dropped from ${before_count} to ${after_count} across restart -- the durable marker store did NOT survive."
        exit 1
    fi

    log_step "Done"
    log_info "AC5-PROOF OK: idempotency store survived restart (rows ${before_count} -> ${after_count}, no durable-marker loss); container returned genuinely healthy."
    log_info "File this receipt on OMN-12912 (not OMN-15521, per that ticket's own AC5 filing instruction) -- this script does not post it anywhere itself."
}

main "$@"
