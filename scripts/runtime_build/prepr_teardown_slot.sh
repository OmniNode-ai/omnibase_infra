#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# =============================================================================
# prepr_teardown_slot.sh -- destroy ONE pre-PR verify slot and prove it gone
# (OMN-18896, the teardown half of Task 7 of epic OMN-18888)
# =============================================================================
#
# The counterpart of prepr_verify_lane.sh. That entrypoint leaves a slot
# running so a lane can exercise its branch on it; this one removes everything
# the slot put on the lab host and the dev lane's shared servers, then reads
# every axis back beside a positive control. The selection and the readback
# live in prepr_teardown_slot.py, where a test runs them against listings that
# carry the dev lane's names beside the slot's.
#
# Like the bring-up it accepts NO lane argument. The compose project, the name
# prefixes, the Valkey index and the database suffix are all derived from the
# slot number through prepr_slot_policy.py, and a derived project is re-checked
# against the declared-lane refusal table.
#
# It takes the same per-slot lane lock the bring-up takes, so a teardown cannot
# run underneath a bring-up of the same slot, and two teardowns cannot overlap.
#
# Not here: the reaper and the heartbeat lease (the rest of Task 7), and tenant
# revoke (Task 5 has not minted one yet).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
POLICY_PY="${SCRIPT_DIR}/prepr_slot_policy.py"
TEARDOWN_PY="${SCRIPT_DIR}/prepr_teardown_slot.py"

# Mirrored from prepr_slot_policy.py and pinned equal to it by
# tests/unit/scripts/test_prepr_teardown_slot_omn18896.py.
EXIT_USAGE=2
EXIT_REFUSED_ATTRIBUTION=6
EXIT_SLOT_LOCK_CONTENDED=8
EXIT_TEARDOWN_INCOMPLETE=13

log()  { printf '[prepr-teardown] %s\n' "$*" >&2; }
fail() { local code="$1"; shift; printf '[prepr-teardown] REFUSED/FAILED: %s\n' "$*" >&2; exit "${code}"; }

usage() {
    cat >&2 <<'USAGE'
usage: prepr_teardown_slot.sh --slot <1|2> --reason <text> [options]

required:
  --slot <n>          the pool slot to destroy.
  --reason <text>     why (also ONEX_DEPLOY_REASON). A mutation of the shared
                      host nobody stated a purpose for is refused.

options:
  --staging-root <p>  the slot's staging root; must sit strictly below
                      /var/tmp/onex-prepr. Default /var/tmp/onex-prepr/slot-<n>.
  --report-out <p>    also write the JSON report here.
  --lock-timeout <s>  seconds to wait for the slot lock. Default 600.
  --plan-only         list what would be removed; remove nothing.
  -h, --help          this text.

There is deliberately NO --lane, --compose-project, --force or --skip option.
USAGE
    exit "${EXIT_USAGE}"
}

SLOT=""
REASON="${ONEX_DEPLOY_REASON:-}"
STAGING_ROOT=""
REPORT_OUT=""
LOCK_TIMEOUT=600
PLAN_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --slot)          shift || usage; [[ $# -gt 0 ]] || usage; SLOT="$1" ;;
        --reason)        shift || usage; [[ $# -gt 0 ]] || usage; REASON="$1" ;;
        --staging-root)  shift || usage; [[ $# -gt 0 ]] || usage; STAGING_ROOT="$1" ;;
        --report-out)    shift || usage; [[ $# -gt 0 ]] || usage; REPORT_OUT="$1" ;;
        --lock-timeout)  shift || usage; [[ $# -gt 0 ]] || usage; LOCK_TIMEOUT="$1" ;;
        --plan-only)     PLAN_ONLY=1 ;;
        -h|--help)       usage ;;
        *) printf '[prepr-teardown] unknown option: %s\n' "$1" >&2; usage ;;
    esac
    shift
done

[[ -n "${SLOT}" ]] || usage
[[ -n "${REASON}" ]] || fail "${EXIT_REFUSED_ATTRIBUTION}" \
    "no reason given. Pass --reason or set ONEX_DEPLOY_REASON."

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PY="${REPO_ROOT}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
else
    fail "${EXIT_USAGE}" "no python3 available."
fi

# Resolve the slot, then run the derived project back through the refusal
# table, exactly as the bring-up does.
SLOT_JSON="$("${PY}" "${POLICY_PY}" --slot "${SLOT}")" || exit $?
COMPOSE_PROJECT="$(printf '%s' "${SLOT_JSON}" | "${PY}" -c 'import json,sys;print(json.load(sys.stdin)["compose_project"])')"
"${PY}" "${POLICY_PY}" --assert-pool-slot "${COMPOSE_PROJECT}" >/dev/null || exit $?
LANE_NAME="${COMPOSE_PROJECT#omnibase-infra-}"
log "slot ${SLOT} -> ${COMPOSE_PROJECT}; reason: ${REASON}"

# The operator env carries the dev lane's broker, Valkey and Postgres
# credentials the slot borrowed. Read into this process only; each reaches its
# tool by environment (docker exec -e NAME), never on a command line.
OMNIBASE_OPERATOR_ENV_FILE="${OMNIBASE_OPERATOR_ENV_FILE:-${HOME}/.omnibase/.env}"
[[ -r "${OMNIBASE_OPERATOR_ENV_FILE}" ]] || fail "${EXIT_USAGE}" \
    "the operator env file is missing or unreadable at ${OMNIBASE_OPERATOR_ENV_FILE}."
set -a
# shellcheck disable=SC1090
source "${OMNIBASE_OPERATOR_ENV_FILE}"
set +a
: "${DEV_KAFKA_SASL_USERNAME:?DEV_KAFKA_SASL_USERNAME must be set in the operator env}"
: "${DEV_KAFKA_SASL_PASSWORD:?DEV_KAFKA_SASL_PASSWORD must be set in the operator env}"
: "${VALKEY_PASSWORD:?VALKEY_PASSWORD must be set in the operator env}"
: "${POSTGRES_USER:?POSTGRES_USER must be set in the operator env}"
export RPK_USER="${DEV_KAFKA_SASL_USERNAME}"
export RPK_PASS="${DEV_KAFKA_SASL_PASSWORD}"
export RPK_SASL_MECHANISM="SCRAM-SHA-256"
export REDISCLI_AUTH="${VALKEY_PASSWORD}"

# shellcheck source=scripts/runtime_build/lane_lock.sh
source "${SCRIPT_DIR}/lane_lock.sh"
# shellcheck disable=SC2329  # invoked by the EXIT trap below
cleanup() { lane_lock_release; }
trap cleanup EXIT

if ! lane_lock_acquire "${COMPOSE_PROJECT}" "${LANE_NAME}" "teardown" "${LOCK_TIMEOUT}" "$0" --slot "${SLOT}"; then
    fail "${EXIT_SLOT_LOCK_CONTENDED}" "slot ${SLOT} is held by another process; the holder was named above."
fi

ARGS=(--slot "${SLOT}")
[[ -n "${STAGING_ROOT}" ]] && ARGS+=(--staging-root "${STAGING_ROOT}")
[[ -n "${REPORT_OUT}" ]] && ARGS+=(--report-out "${REPORT_OUT}")
[[ "${PLAN_ONLY}" == "1" ]] && ARGS+=(--plan-only)

rc=0
"${PY}" "${TEARDOWN_PY}" "${ARGS[@]}" || rc=$?
if [[ "${rc}" -eq 0 ]]; then
    log "slot ${SLOT}: $([[ "${PLAN_ONLY}" == "1" ]] && echo "plan printed, nothing removed" || echo "CLEAN, every axis read back zero beside a non-zero control")."
elif [[ "${rc}" -eq "${EXIT_TEARDOWN_INCOMPLETE}" ]]; then
    log "slot ${SLOT}: NOT clean. The report above names the residue or the unproven control."
fi
exit "${rc}"
