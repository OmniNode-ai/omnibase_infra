#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# render_dev_lane_tenant_path_env.sh -- provision the LANE-LOCAL sentinels the
# dev-lane tenant-scoped control plane fails closed on (OMN-17530).
#
# WHAT THIS IS AND IS NOT.
#
# It is the compose analogue of omninode_infra k8s/onex-lab/render_ci_secrets.py:
# the k3s lab lane generates its own per-run values for exactly these names,
# because on onex-dev they come from Secrets provisioned out-of-band and an
# out-of-band Secret is not a manifest defect a lab lane can inherit -- it is a
# value that simply is not there.
#
# It is NOT a credential rotation, re-issue, revoke or key mint (Operating Rule
# 22, unchanged). Every value it writes:
#
#   * authorises exactly one thing, inside one compose project, on one host;
#   * is compared constant-time by onex-api against a request header and
#     authenticates nothing to any third party;
#   * in the Stripe case is a SENTINEL that reaches no Stripe endpoint -- the
#     lane's three billing redirect URLs name a reserved never-resolvable host
#     (RFC 6761 `.invalid`).
#
# It touches no provider key, no broker principal, no Infisical identity and no
# database password that any other lane, cluster or person holds.
#
# IDEMPOTENT BY CONSTRUCTION. A name that already has a non-empty value in the
# target env file is LEFT ALONE and reported as `kept`. Re-running this script
# can therefore never change a live lane's admin secret out from under a running
# onex-api -- which would be indistinguishable, at the first 401, from the
# secret having been wrong all along.
#
# NOTHING IS PRINTED BUT NAMES. Each written value is reported as
# `<NAME> written (len=<n> sha256-12=<prefix>)`. No value reaches stdout, the
# process list or this script's own arguments.
set -euo pipefail

ENV_FILE="${OMNIBASE_OPERATOR_ENV_FILE:-${HOME}/.omnibase/.env}"

# ROLE_OMNINODE_PASSWORD is FIRST and is the one name here that is not an
# onex-api admin gate. docker/migrations/forward/000_create_multiple_databases.sh
# maps omninode_cloud -> role_omninode -> this variable, and treats an EMPTY
# value as "skip this role" -- which is why the lane has an omninode_cloud
# database with zero tables and no owning login. docker/catalog/services/
# postgres.yaml defaults it to '' deliberately (a lane that has not provisioned
# the credential must not get a half-configured role); this script is the
# deployment-owned step that provisions it. 32 hex characters because
# validate_password() in that init script rejects anything non-hex.
NAMES=(
  ROLE_OMNINODE_PASSWORD
  TENANT_BOOTSTRAP_ADMIN_SECRET
  TENANT_TOPICS_ADMIN_SECRET
  TENANT_CLIENTS_ADMIN_SECRET
  TENANT_OFFBOARD_ADMIN_SECRET
  ALPHA_INVITE_ADMIN_SECRET
  STRIPE_API_KEY
  STRIPE_WEBHOOK_SECRET
)

usage() {
  cat >&2 <<'USAGE'
usage: render_dev_lane_tenant_path_env.sh [--check]

  (no flag)  provision any missing name in the operator env file, keep the rest
  --check    report which names are present or missing and exit 1 if any is
             missing; writes nothing

The target file is $OMNIBASE_OPERATOR_ENV_FILE, default ~/.omnibase/.env.
USAGE
}

MODE="write"
case "${1:-}" in
  --check) MODE="check" ;;
  "") ;;
  -h|--help) usage; exit 0 ;;
  *) usage; exit 2 ;;
esac

[ -f "$ENV_FILE" ] || {
  echo "FATAL: ${ENV_FILE} does not exist. This script provisions lane-local values into" >&2
  echo "       the operator env file the compose project already reads; it does not create" >&2
  echo "       that file, because doing so on the wrong host is how a lane ends up with two." >&2
  exit 1
}

current_value() {
  # Last assignment wins, matching shell sourcing semantics. The value is
  # captured into a variable and never echoed.
  sed -n "s/^${1}=//p" "$ENV_FILE" | tail -1
}

fingerprint() {
  # sha256-12 + length. The only value-derived facts this script ever emits.
  local value="$1" digest
  if command -v sha256sum >/dev/null 2>&1; then
    digest="$(printf '%s' "$value" | sha256sum | cut -c1-12)"
  else
    digest="$(printf '%s' "$value" | shasum -a 256 | cut -c1-12)"
  fi
  printf 'len=%s sha256-12=%s' "${#value}" "$digest"
}

generate() {
  case "$1" in
    # Sentinels that LOOK like the real thing, so a value that accidentally
    # reached a real Stripe call would be rejected by Stripe as malformed
    # rather than being tried. `sk_test_` is Stripe's own test prefix.
    STRIPE_API_KEY) printf 'sk_test_onexlab%s' "$(openssl rand -hex 16)" ;;
    STRIPE_WEBHOOK_SECRET) printf 'whsec_onexlab%s' "$(openssl rand -hex 16)" ;;
    # Hex only: 000_create_multiple_databases.sh validate_password() rejects a
    # non-hex password outright.
    *) openssl rand -hex 32 ;;
  esac
}

# ONEX_LAB_TENANT_STATE_DIR is a PATH, not a secret, so it is handled apart from
# the generated names above: it gets a resolved default rather than random hex,
# and the directory is created with mode 0700. It exists as a variable at all
# because docker/docker-compose.dev-lane.yml may not spell a ${HOME}-derived
# default -- a nested expansion resolves against whichever HOME the compose
# invocation carries, and on the lane host an interactive shell, the deploy
# agent's systemd unit and a CI runner carry three different ones.
tenant_state_dir="$(current_value ONEX_LAB_TENANT_STATE_DIR)"
if [ -z "$tenant_state_dir" ]; then
  tenant_state_dir="${HOME}/.omnibase/state/dev-lane-tenant"
  if [ "$MODE" = check ]; then
    echo "ONEX_LAB_TENANT_STATE_DIR MISSING"
  else
    printf 'ONEX_LAB_TENANT_STATE_DIR=%s\n' "$tenant_state_dir" >> "$ENV_FILE"
    echo "ONEX_LAB_TENANT_STATE_DIR written (${tenant_state_dir})"
  fi
else
  echo "ONEX_LAB_TENANT_STATE_DIR kept (${tenant_state_dir})"
fi
[ "$MODE" = check ] || { mkdir -p "$tenant_state_dir"; chmod 700 "$tenant_state_dir"; }

missing=0
for name in "${NAMES[@]}"; do
  value="$(current_value "$name")"
  if [ -n "$value" ]; then
    echo "${name} kept ($(fingerprint "$value"))"
    continue
  fi
  if [ "$MODE" = check ]; then
    echo "${name} MISSING"
    missing=$((missing + 1))
    continue
  fi
  value="$(generate "$name")"
  # umask before the append, so a file created by a future edit cannot be
  # world-readable even for an instant.
  ( umask 077; printf '%s=%s\n' "$name" "$value" >> "$ENV_FILE" )
  echo "${name} written ($(fingerprint "$value"))"
done
chmod 600 "$ENV_FILE"

if [ "$MODE" = check ] && [ "$missing" -ne 0 ]; then
  echo "FATAL: ${missing} lane-local value(s) missing from ${ENV_FILE}; run this script with no flag" >&2
  exit 1
fi
