#!/bin/sh
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

set -eu

FRESH_HOST="${FRESH_HOST:-fresh-postgres}"
LEGACY_HOST="${LEGACY_HOST:-legacy-postgres}"
FRESH_PORT="${FRESH_PORT:-5432}"
LEGACY_PORT="${LEGACY_PORT:-5432}"
MIGRATIONS_DIR="${MIGRATIONS_DIR:-/migrations/forward}"
RUNNER="${RUNNER:-/opt/omn15422/run-forward-migrations.sh}"
LEDGER_BLOCKER='column "migration_id" of relation "schema_migrations" does not exist'

fail() {
  echo "fixture_status=FAIL detail=$1" >&2
  exit 1
}

sql_value() {
  host="$1"
  database="$2"
  statement="$3"
  if [ "$host" = "$FRESH_HOST" ]; then
    port="$FRESH_PORT"
  else
    port="$LEGACY_PORT"
  fi
  psql -X -qAt -h "$host" -p "$port" -U postgres -d "$database" \
    -v ON_ERROR_STOP=1 -c "$statement"
}

assert_pair() {
  case_id="$1"
  positive_sql="$2"
  red_sql="$3"
  red_signature="$4"

  positive_count="$(sql_value "$LEGACY_HOST" omnidash_analytics "$positive_sql")"
  [ "$positive_count" = "0" ] || fail "$case_id positive control reported $positive_count defect(s)"

  red_count="$(sql_value "$LEGACY_HOST" omnidash_analytics "$red_sql")"
  [ "$red_count" -gt 0 ] || fail "$case_id RED control was not discriminated"
  echo "fixture_case=$case_id positive=PASS red=DETECTED red_signature=$red_signature red_count=$red_count"
}

assert_pair \
  mapping_ambiguity \
  "SELECT count(*) FROM (SELECT legacy_tenant_value FROM omn15422_fixture.mapping_positive GROUP BY legacy_tenant_value HAVING count(DISTINCT tenant_uuid) <> 1) AS defects" \
  "SELECT count(*) FROM (SELECT legacy_tenant_value FROM omn15422_fixture.mapping_red GROUP BY legacy_tenant_value HAVING count(DISTINCT tenant_uuid) <> 1) AS defects" \
  ambiguous_mapping

assert_pair \
  checksum_conflict \
  "SELECT count(*) FROM (SELECT migration_id FROM omn15422_fixture.checksum_positive GROUP BY migration_id HAVING count(DISTINCT checksum) <> 1) AS defects" \
  "SELECT count(*) FROM (SELECT migration_id FROM omn15422_fixture.checksum_red GROUP BY migration_id HAVING count(DISTINCT checksum) <> 1) AS defects" \
  checksum_conflict

assert_pair \
  owner_drift \
  "SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace JOIN pg_roles r ON r.oid=c.relowner WHERE n.nspname='omn15422_fixture' AND c.relname='owner_positive' AND r.rolname <> 'role_omnidash'" \
  "SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace JOIN pg_roles r ON r.oid=c.relowner WHERE n.nspname='omn15422_fixture' AND c.relname='owner_red' AND r.rolname <> 'role_omnidash'" \
  owner_mismatch

assert_pair \
  unsafe_rls_policy \
  "SELECT (CASE WHEN c.relrowsecurity AND c.relforcerowsecurity THEN 0 ELSE 1 END) + (SELECT count(*) FROM information_schema.columns WHERE table_schema='omn15422_fixture' AND table_name='tenant_usage_safe' AND column_name='tenant_id' AND (data_type <> 'uuid' OR is_nullable <> 'NO')) FROM pg_class c WHERE c.oid='omn15422_fixture.tenant_usage_safe'::regclass" \
  "SELECT (CASE WHEN c.relrowsecurity AND c.relforcerowsecurity THEN 0 ELSE 1 END) + (SELECT count(*) FROM information_schema.columns WHERE table_schema='public' AND table_name='tenant_usage_legacy' AND column_name='tenant_id' AND (data_type <> 'uuid' OR is_nullable <> 'NO')) FROM pg_class c WHERE c.oid='public.tenant_usage_legacy'::regclass" \
  legacy_varchar_enable_only

assert_pair \
  unsafe_view \
  "SELECT CASE WHEN coalesce(reloptions, ARRAY[]::text[]) @> ARRAY['security_invoker=true'] THEN 0 ELSE 1 END FROM pg_class WHERE oid='omn15422_fixture.tenant_usage_safe_view'::regclass" \
  "SELECT CASE WHEN coalesce(reloptions, ARRAY[]::text[]) @> ARRAY['security_invoker=true'] THEN 0 ELSE 1 END FROM pg_class WHERE oid='omn15422_fixture.tenant_usage_red_view'::regclass" \
  missing_security_invoker

assert_pair \
  unsafe_function \
  "SELECT (CASE WHEN p.prosecdef THEN 1 ELSE 0 END) + (CASE WHEN EXISTS (SELECT 1 FROM aclexplode(coalesce(p.proacl, acldefault('f', p.proowner))) acl WHERE acl.grantee=0 AND acl.privilege_type='EXECUTE') THEN 1 ELSE 0 END) FROM pg_proc p WHERE p.oid='omn15422_fixture.tenant_usage_safe_count()'::regprocedure" \
  "SELECT (CASE WHEN p.prosecdef THEN 1 ELSE 0 END) + (CASE WHEN EXISTS (SELECT 1 FROM aclexplode(coalesce(p.proacl, acldefault('f', p.proowner))) acl WHERE acl.grantee=0 AND acl.privilege_type='EXECUTE') THEN 1 ELSE 0 END) FROM pg_proc p WHERE p.oid='omn15422_fixture.tenant_usage_red_count()'::regprocedure" \
  definer_or_public_execute

assert_pair \
  transformation_collision \
  "SELECT count(*) FROM (SELECT target_key FROM omn15422_fixture.transform_positive GROUP BY target_key HAVING count(*) > 1) AS defects" \
  "SELECT count(*) FROM (SELECT target_key FROM omn15422_fixture.transform_red GROUP BY target_key HAVING count(*) > 1) AS defects" \
  duplicate_target_key

signature() {
  host="$1"
  database="$2"
  relation="$3"
  sql_value "$host" "$database" "SELECT string_agg(column_name || ':' || data_type || ':' || is_nullable, ',' ORDER BY ordinal_position) FROM information_schema.columns WHERE table_schema='public' AND table_name='$relation'"
}

flat_control="$(signature "$LEGACY_HOST" omnibase_infra flat_node_parity_control)"
node_control="$(signature "$LEGACY_HOST" omnidash_analytics flat_node_parity_control)"
[ "$flat_control" = "$node_control" ] || fail "flat/node positive shape control diverged"
flat_red="$(signature "$LEGACY_HOST" omnibase_infra llm_cost_aggregates)"
node_red="$(signature "$LEGACY_HOST" omnidash_analytics llm_cost_aggregates)"
[ "$flat_red" != "$node_red" ] || fail "flat/node RED shape collision was not discriminated"
echo "fixture_case=flat_node_shape_collision positive=PASS red=DETECTED red_signature=column_signature_mismatch"

ledger_positive="$(signature "$LEGACY_HOST" omnibase_infra schema_migrations)"
ledger_red="$(signature "$LEGACY_HOST" omnidash_analytics schema_migrations)"
ledger_version="$(signature "$LEGACY_HOST" omninode_cloud schema_migrations)"
node_ledger="$(signature "$LEGACY_HOST" omnidash_analytics node_schema_migrations)"
case "$ledger_positive" in
  migration_id:*checksum:*source_set:*) ;;
  *) fail "checksum-capable positive ledger shape missing: $ledger_positive" ;;
esac
case "$ledger_red" in
  filename:*applied_at:*) ;;
  *) fail "filename/applied_at legacy ledger shape missing: $ledger_red" ;;
esac
case "$ledger_version" in
  version:*applied_at:*checksum:*) ;;
  *) fail "version/nullable-checksum legacy ledger shape missing: $ledger_version" ;;
esac
case "$node_ledger" in
  version:*applied_at:*checksum:*) ;;
  *) fail "node ledger shape missing: $node_ledger" ;;
esac
echo "fixture_case=legacy_shape_collision positive=PASS red=SEE_REAL_RUNNER"

dependency_count="$(sql_value "$LEGACY_HOST" omnidash_analytics "SELECT (SELECT count(*) FROM pg_constraint WHERE conrelid='public.tenant_usage_legacy'::regclass AND contype='f') + (SELECT count(*) FROM pg_indexes WHERE schemaname='public' AND tablename='tenant_usage_legacy') + (SELECT count(*) FROM pg_views WHERE schemaname='public' AND viewname='tenant_usage_legacy_view') + (SELECT count(*) FROM pg_proc WHERE oid='public.tenant_usage_legacy_count()'::regprocedure)")"
[ "$dependency_count" -ge 4 ] || fail "dependent FK/index/view/function catalog is incomplete"
sentinel_count="$(sql_value "$LEGACY_HOST" omnidash_analytics "SELECT count(*) FROM public.tenants_legacy WHERE tenant_id IN ('', 'omninode', '00000000-0000-0000-0000-000000000000')")"
[ "$sentinel_count" = "3" ] || fail "synthetic sentinel corpus is incomplete"
acl_count="$(sql_value "$LEGACY_HOST" omnidash_analytics "SELECT (SELECT count(*) FROM information_schema.role_table_grants WHERE table_schema='public' AND table_name='tenant_usage_legacy' AND grantee='role_omnidash') + (SELECT count(*) FROM pg_default_acl d CROSS JOIN LATERAL aclexplode(d.defaclacl) acl JOIN pg_roles r ON r.oid=acl.grantee WHERE r.rolname='app_dashboard' AND acl.privilege_type='SELECT')")"
[ "$acl_count" -ge 2 ] || fail "legacy grants/default privileges are incomplete"
policy_count="$(sql_value "$LEGACY_HOST" omnidash_analytics "SELECT count(*) FROM pg_policies WHERE schemaname='public' AND tablename='tenant_usage_legacy' AND policyname='tenant_usage_legacy_policy'")"
[ "$policy_count" = "1" ] || fail "legacy RLS policy is missing"
echo "fixture_case=dependencies_acl_and_sentinels status=PASS dependency_count=$dependency_count acl_count=$acl_count policy_count=$policy_count sentinel_count=$sentinel_count"

# Reproduce OMN-15335 with the real migration as the non-owner role. The same
# file then runs twice as postgres to prove the OMN-15376 shape reconciliation
# is idempotent independently of the earlier ledger wall.
owner_log="$(mktemp)"
if psql -X -h "$LEGACY_HOST" -p "$LEGACY_PORT" -U role_omnidash -d omnidash_analytics \
  -v ON_ERROR_STOP=1 \
  -f "$MIGRATIONS_DIR/nodes/node_projection_cost_summary/0001_create_llm_cost_aggregates.sql" \
  >"$owner_log" 2>&1; then
  fail "owner-drift RED migration unexpectedly succeeded"
fi
grep -F 'must be owner of table llm_cost_aggregates' "$owner_log" >/dev/null \
  || { sed -n '1,160p' "$owner_log"; fail "owner-drift failure signature moved"; }
echo "fixture_case=owner_drift_real_migration red=DETECTED red_signature=must_be_owner"

for pass in 1 2; do
  psql -X -q -h "$LEGACY_HOST" -p "$LEGACY_PORT" -U postgres -d omnidash_analytics \
    -v ON_ERROR_STOP=1 \
    -f "$MIGRATIONS_DIR/nodes/node_projection_cost_summary/0001_create_llm_cost_aggregates.sql"
  psql -X -q -h "$LEGACY_HOST" -p "$LEGACY_PORT" -U postgres -d omnidash_analytics \
    -v ON_ERROR_STOP=1 \
    -f "$MIGRATIONS_DIR/nodes/node_projection_baselines/0001_create_baselines_tables.sql"
  echo "fixture_case=legacy_shape_reconciliation pass=$pass status=PASS"
done
echo "fixture_case=owner_drift_real_migration positive=PASS red=DETECTED passes=2"

run_forward() {
  host="$1"
  port="$2"
  database="$3"
  log="$4"
  POSTGRES_HOST="$host" \
  POSTGRES_PORT="$port" \
  POSTGRES_USER=postgres \
  POSTGRES_PASSWORD='' \
  POSTGRES_DB="$database" \
  NODE_POSTGRES_DB=omnidash_analytics \
  MIGRATIONS_DIR="$MIGRATIONS_DIR" \
  sh "$RUNNER" >"$log" 2>&1
}

for pass in 1 2; do
  fresh_log="$(mktemp)"
  if ! run_forward "$FRESH_HOST" "$FRESH_PORT" omnibase_infra "$fresh_log"; then
    sed -n '1,240p' "$fresh_log"
    fail "fresh real migration pass $pass failed"
  fi
  grep -F 'Sentinel set. Migration gate will report HEALTHY.' "$fresh_log" >/dev/null \
    || fail "fresh pass $pass omitted terminal sentinel proof"
  if [ "$pass" = "2" ]; then
    grep -E 'Complete: 0 infra applied, [0-9]+ infra skipped; 0 node applied, [0-9]+ node skipped' "$fresh_log" >/dev/null \
      || { tail -n 80 "$fresh_log"; fail "fresh second pass was not idempotent"; }
  fi
  echo "fixture_case=fresh_install pass=$pass status=PASS"
done

# The fixture executes the real legacy-upgrade path twice on the same synthetic
# state. Until OMN-15413 converges/imports the filename ledger, the current
# runner must fail at this exact catalog boundary. Treating another error (or a
# surprise success) as equivalent would make the fixture a false green.
for pass in 1 2; do
  legacy_log="$(mktemp)"
  if run_forward "$LEGACY_HOST" "$LEGACY_PORT" omnidash_analytics "$legacy_log"; then
    fail "legacy upgrade pass $pass unexpectedly crossed the OMN-15413 boundary"
  fi
  grep -F "$LEDGER_BLOCKER" "$legacy_log" >/dev/null \
    || { sed -n '1,240p' "$legacy_log"; fail "legacy upgrade blocker signature moved"; }
  echo "fixture_case=legacy_upgrade status=BLOCKED pass=$pass blocker=OMN-15413 signature=migration_id_missing"
done

echo "fixture_status=PASS_WITH_EXPECTED_BLOCKER blocker=OMN-15413"
