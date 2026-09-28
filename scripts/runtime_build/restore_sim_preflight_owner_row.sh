#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Restore exactly one captured analytics owner row into the disposable sim lane.
set -euo pipefail

readonly TARGET_CONTAINER="omnibase-infra-sim-preflight-postgres"
readonly TARGET_PROJECT="omnibase-infra-sim-preflight"
readonly ANALYTICS_DB="omnidash_analytics"
readonly OWNER_TABLE="public.delegation_events"

require() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "${name} is required" >&2
    exit 64
  fi
}

is_uuid() {
  [[ "$1" =~ ^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$ ]]
}

for name in SIM_ARCHIVE_OWNER_ROW_PATH SIM_ARCHIVE_OWNER_METADATA_PATH; do
  require "${name}"
done
if [[ ! -f "${SIM_ARCHIVE_OWNER_ROW_PATH}" || ! -f "${SIM_ARCHIVE_OWNER_METADATA_PATH}" ]]; then
  echo "private owner archive inputs must be regular files" >&2
  exit 64
fi

correlation_id=""
tenant_id=""
schema_sha256=""
row_sha256=""
source_db=""
source_table=""
while IFS='=' read -r key value; do
  case "${key}" in
    correlation_id) correlation_id="${value}" ;;
    tenant_id) tenant_id="${value}" ;;
    schema_sha256) schema_sha256="${value}" ;;
    row_sha256) row_sha256="${value}" ;;
    source_db) source_db="${value}" ;;
    source_table) source_table="${value}" ;;
    *) echo "owner metadata contains an unknown field" >&2; exit 65 ;;
  esac
done <"${SIM_ARCHIVE_OWNER_METADATA_PATH}"
for value_name in correlation_id tenant_id schema_sha256 row_sha256 source_db source_table; do
  if [[ -z "${!value_name}" ]]; then
    echo "owner metadata is incomplete" >&2
    exit 65
  fi
done
if [[ "${source_db}" != "${ANALYTICS_DB}" || "${source_table}" != "${OWNER_TABLE}" ]]; then
  echo "owner metadata is not for the declared analytics relation" >&2
  exit 65
fi
if ! is_uuid "${correlation_id}" || ! is_uuid "${tenant_id}"; then
  echo "owner metadata coordinates must be UUIDs" >&2
  exit 65
fi
if [[ ! "${schema_sha256}" =~ ^[0-9a-f]{64}$ || ! "${row_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "owner metadata checksums must be lowercase SHA-256" >&2
  exit 65
fi
if [[ "$(shasum -a 256 "${SIM_ARCHIVE_OWNER_ROW_PATH}" | awk '{print $1}')" != "${row_sha256}" ]]; then
  echo "owner archive checksum differs from metadata" >&2
  exit 65
fi
if [[ "$(head -c 11 "${SIM_ARCHIVE_OWNER_ROW_PATH}" | LC_ALL=C od -An -t x1 | tr -d '[:space:]')" != "5047434f50590aff0d0a00" ]]; then
  echo "owner archive is not PostgreSQL binary COPY data" >&2
  exit 65
fi
if [[ "$(docker inspect -f '{{ index .Config.Labels "com.docker.compose.project" }}' "${TARGET_CONTAINER}")" != "${TARGET_PROJECT}" ]]; then
  echo "target is not the disposable sim-preflight postgres container" >&2
  exit 65
fi

target_psql() {
  docker exec -i "${TARGET_CONTAINER}" sh -lc \
    "exec psql -X -q -A -t -v ON_ERROR_STOP=1 -U \"\$POSTGRES_USER\" -d ${ANALYTICS_DB}"
}

readonly SCHEMA_SQL="SELECT string_agg(attname || ':' || atttypid::regtype::text || ':' || attnotnull::text, ',' ORDER BY attnum) FROM pg_attribute WHERE attrelid = '${OWNER_TABLE}'::regclass AND attnum > 0 AND NOT attisdropped;"
target_schema="$(printf '%s\n' "${SCHEMA_SQL}" | target_psql)"
if [[ -z "${target_schema}" || "$(printf '%s' "${target_schema}" | shasum -a 256 | awk '{print $1}')" != "${schema_sha256}" ]]; then
  echo "target analytics schema does not match captured owner row" >&2
  exit 65
fi
before_count="$(printf 'SELECT count(*) FROM %s;\n' "${OWNER_TABLE}" | target_psql | tr -d '[:space:]')"
if [[ "${before_count}" != "0" ]]; then
  echo "target owner relation must be empty before narrow restore" >&2
  exit 65
fi
if ! docker exec -i "${TARGET_CONTAINER}" sh -lc \
  "exec psql -X -q -v ON_ERROR_STOP=1 -U \"\$POSTGRES_USER\" -d ${ANALYTICS_DB} -c 'COPY ${OWNER_TABLE} FROM STDIN WITH (FORMAT binary);'" <"${SIM_ARCHIVE_OWNER_ROW_PATH}"; then
  echo "target owner restore failed" >&2
  exit 65
fi
readonly EXACT_OWNER_SQL="SELECT count(*) FROM ${OWNER_TABLE} WHERE correlation_id = '${correlation_id}'::text AND tenant_id = '${tenant_id}'::uuid;"
exact_count="$(printf '%s\n' "${EXACT_OWNER_SQL}" | target_psql | tr -d '[:space:]')"
total_count="$(printf 'SELECT count(*) FROM %s;\n' "${OWNER_TABLE}" | target_psql | tr -d '[:space:]')"
if [[ "${exact_count}" != "1" || "${total_count}" != "1" ]]; then
  echo "target owner readback is not exactly the captured owner row" >&2
  exit 65
fi
printf '%s\n' "owner-row-restored row_sha256=${row_sha256} schema_sha256=${schema_sha256}"
