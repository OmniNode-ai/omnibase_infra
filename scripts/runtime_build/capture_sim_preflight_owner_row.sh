#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Capture one declared analytics owner row from the read-only source lane.
set -euo pipefail

readonly SOURCE_CONTAINER="omnibase-infra-postgres"
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

for name in SIM_SOURCE_SSH_HOST SIM_ARCHIVE_CAPTURE_DIR SIM_ARCHIVE_CORRELATION_ID SIM_ARCHIVE_TENANT_ID; do
  require "${name}"
done
if ! is_uuid "${SIM_ARCHIVE_CORRELATION_ID}" || ! is_uuid "${SIM_ARCHIVE_TENANT_ID}"; then
  echo "archive owner coordinates must be UUIDs" >&2
  exit 64
fi
if [[ ! -d "${SIM_ARCHIVE_CAPTURE_DIR}" || ! -w "${SIM_ARCHIVE_CAPTURE_DIR}" ]]; then
  echo "capture directory must already exist and be writable" >&2
  exit 64
fi

readonly ROW_PATH="${SIM_ARCHIVE_CAPTURE_DIR}/delegation-events-owner-row.copy"
readonly METADATA_PATH="${SIM_ARCHIVE_CAPTURE_DIR}/delegation-events-owner-row.metadata"
if [[ -e "${ROW_PATH}" || -e "${METADATA_PATH}" ]]; then
  echo "refusing to overwrite an existing private archive" >&2
  exit 64
fi

source_psql() {
  ssh -o BatchMode=yes -o ConnectTimeout=10 "${SIM_SOURCE_SSH_HOST}" \
    "docker exec -i ${SOURCE_CONTAINER} sh -lc 'exec psql -X -q -A -t -v ON_ERROR_STOP=1 -U \"\$POSTGRES_USER\" -d ${ANALYTICS_DB}'"
}

readonly OWNER_COUNT_SQL="SELECT count(*) FROM ${OWNER_TABLE} WHERE correlation_id = '${SIM_ARCHIVE_CORRELATION_ID}'::text AND tenant_id = '${SIM_ARCHIVE_TENANT_ID}'::uuid;"
owner_count="$(printf '%s\n' "${OWNER_COUNT_SQL}" | source_psql | tr -d '[:space:]')"
if [[ "${owner_count}" != "1" ]]; then
  echo "source owner predicate must select exactly one row" >&2
  exit 65
fi

readonly SCHEMA_SQL="SELECT string_agg(attname || ':' || atttypid::regtype::text || ':' || attnotnull::text, ',' ORDER BY attnum) FROM pg_attribute WHERE attrelid = '${OWNER_TABLE}'::regclass AND attnum > 0 AND NOT attisdropped;"
schema="$(printf '%s\n' "${SCHEMA_SQL}" | source_psql)"
if [[ -z "${schema}" ]]; then
  echo "source owner schema is absent" >&2
  exit 65
fi
schema_sha256="$(printf '%s' "${schema}" | shasum -a 256 | awk '{print $1}')"

readonly COPY_SQL="COPY (SELECT * FROM ${OWNER_TABLE} WHERE correlation_id = '${SIM_ARCHIVE_CORRELATION_ID}'::text AND tenant_id = '${SIM_ARCHIVE_TENANT_ID}'::uuid) TO STDOUT WITH (FORMAT binary);"
if ! printf '%s\n' "${COPY_SQL}" | source_psql >"${ROW_PATH}"; then
  rm -f "${ROW_PATH}"
  echo "source owner copy failed" >&2
  exit 65
fi
if [[ "$(head -c 11 "${ROW_PATH}" | LC_ALL=C od -An -t x1 | tr -d '[:space:]')" != "5047434f50590aff0d0a00" ]]; then
  rm -f "${ROW_PATH}"
  echo "source owner copy is not PostgreSQL binary COPY data" >&2
  exit 65
fi
row_sha256="$(shasum -a 256 "${ROW_PATH}" | awk '{print $1}')"
umask 077
printf '%s\n' \
  "correlation_id=${SIM_ARCHIVE_CORRELATION_ID}" \
  "tenant_id=${SIM_ARCHIVE_TENANT_ID}" \
  "schema_sha256=${schema_sha256}" \
  "row_sha256=${row_sha256}" \
  "source_db=${ANALYTICS_DB}" \
  "source_table=${OWNER_TABLE}" >"${METADATA_PATH}"
chmod 600 "${ROW_PATH}" "${METADATA_PATH}"
printf '%s\n' "owner-row-captured row_sha256=${row_sha256} schema_sha256=${schema_sha256}"
