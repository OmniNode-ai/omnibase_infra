-- onex-create-database: omnidash_analytics
-- =============================================================================
-- MIGRATION: physically create omninode_internal.live_events + transform-copy
--            the existing public.live_events rows (OMN-15359)
-- =============================================================================
-- Ticket: OMN-15359 (P2-P4 build classified schemas and migrate internal,
--         control-plane, catalog, and tenant targets)
-- Related: OMN-15421 (Projection Domain Adapter Proof), OMN-15423 (relation
--          inventory/ownership), OMN-15425 (tenant projection authority --
--          separate, unrelated gap)
-- Version: 1.0.0
--
-- WHAT THIS FILE DOES
--   098_create_omninode_internal_schema.sql created the (empty) omninode_internal
--   schema and a temporary physical bridge (physical_schema_mapping.py's
--   INTERNAL_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359) that let
--   omninode_internal-domain tables keep resolving against `public` where they
--   physically still live. This migration performs the first real family
--   cutover out of that bridge: `live_events`.
--
--   1. CREATE TABLE omninode_internal.live_events, identical shape to
--      public.live_events (docker/migrations/forward/nodes/node_projection_live_events/
--      0000_create_live_events.sql + 0001's type-classification data fixups).
--   2. Transform-copy every row from public.live_events into
--      omninode_internal.live_events (idempotent: ON CONFLICT (event_id) DO
--      NOTHING, safe to re-run).
--   3. Reconciliation: fail loud (RAISE EXCEPTION, migration Job aborts) unless
--      every source row's key is present in the destination AND a row-level
--      content hash over the full shared key set matches exactly. A partial or
--      corrupted copy blocks the migration Job instead of silently landing.
--   4. public.live_events is PRESERVED -- not dropped, not ALTER TABLE ... SET
--      SCHEMA'd. This ticket's own scope text requires the source relation to
--      survive until parity is independently reproven post-deploy (see the
--      ground-phase live readback cited in the companion PR body).
--
-- WHY THIS TABLE, WHY NOW
--   node_projection_live_events/contract.yaml has declared
--   db_io.db_tables[0].schema: omninode_internal since before this migration
--   existed. handler_wiring.py's _resolve_projection_database_target uses that
--   contract-declared schema literally as the SQL write target (it does not
--   consult the physical bridge for query building, only for the grant-
--   privilege check) -- so the runtime has been issuing
--   `INSERT INTO omninode_internal.live_events` since before 098 merged, and
--   failing with "relation does not exist" because no migration had ever
--   physically created that relation. 098 fixed the schema-level half of that
--   gap; this migration fixes the table-level half for this one family.
--
-- WHY live_events IS ALSO REMOVED FROM THE BRIDGE (companion change, same PR)
--   src/omnibase_infra/topology/physical_schema_mapping.py drops "live_events"
--   from INTERNAL_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359 in this same PR.
--   Post-migration, physical_grant_schema_for_table('omninode_internal',
--   'live_events') must agree with the INSERT-target schema
--   (omninode_internal) -- leaving it in the bridge after the physical table
--   exists would make the grant-privilege check assert against `public` while
--   every write already lands in `omninode_internal`, silently reintroducing
--   the same class of drift this migration exists to close. The shipped
--   topology TABLE grants (src/omnibase_infra/topology/instances/*.yaml,
--   docker/catalog/database-topology/*.yaml) are regenerated in the same PR
--   via scripts/generate_application_database_table_grants.py --write so the
--   omninode_runtime principal's live_events grant moves from schema: public
--   to schema: omninode_internal, matching physical reality.
--
-- DATABASE CONTEXT
--   Same pattern as 083/096/097/098: the forward runner applies this against
--   POSTGRES_DB, so this file switches with `\connect omnidash_analytics`.
--
-- IDEMPOTENCY
--   CREATE TABLE IF NOT EXISTS + INSERT ... ON CONFLICT DO NOTHING are both
--   safe to re-run. The reconciliation DO block re-derives its counts/hashes
--   from live catalog state on every apply, so a re-run after a partial prior
--   apply re-validates rather than trusting a cached result.
--
-- ROLLBACK
--   See rollback/rollback_099_create_omninode_internal_live_events.sql.
--   RESTRICT, not CASCADE, and refuses (by construction -- DROP TABLE with no
--   CASCADE on a table nothing else yet depends on) once any downstream
--   object has been built against omninode_internal.live_events. Because
--   public.live_events is untouched, rollback of this migration alone never
--   loses data: the source of truth remains intact throughout.
-- =============================================================================

\connect omnidash_analytics

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS omninode_internal.live_events (
  id             UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  event_id       TEXT        UNIQUE NOT NULL,
  type           TEXT        NOT NULL DEFAULT 'ACTION',
  timestamp      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  source         TEXT        NOT NULL DEFAULT 'platform',
  topic          TEXT        NOT NULL DEFAULT '',
  summary        TEXT        NOT NULL DEFAULT '',
  payload        TEXT        NOT NULL DEFAULT '{}',
  correlation_id TEXT,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Transform-copy + reconciliation, guarded by public.live_events actually
-- existing. That table is node-owned DDL
-- (docker/migrations/forward/nodes/node_projection_live_events/0000_create_live_events.sql,
-- vendored from omnimarket) applied by a SEPARATE migration stream
-- (scripts/sync-node-migrations.sh + run-forward-migrations.sh) from this
-- file's own top-level docker/migrations/forward/*.sql stream. The two
-- streams are not guaranteed ordered relative to each other in every
-- consumer: the standalone "Migration Integration Test" CI gate applies only
-- the numbered top-level files (024...098, no nodes/ subtree) against a
-- fresh database, so public.live_events genuinely does not exist in that
-- scope. Guarding on `to_regclass` (not EXECUTE/dynamic SQL -- an ordinary
-- plpgsql IF branch is compiled but not planned/resolved until control flow
-- actually reaches it) makes this file correct in both scopes: the full
-- production apply (where node migrations already ran and populated real
-- rows) and the standalone top-level-only gate (empty destination, no
-- reconciliation to perform, migration still succeeds).
DO $$
DECLARE
  v_src_count     BIGINT;
  v_dst_count     BIGINT;
  v_missing_keys  BIGINT;
  v_src_hash      TEXT;
  v_dst_hash      TEXT;
BEGIN
  IF to_regclass('public.live_events') IS NULL THEN
    RAISE NOTICE
      'OMN-15359: public.live_events does not exist in this migration scope '
      '(node-owned migration not applied here) -- omninode_internal.live_events '
      'created empty; transform-copy and reconciliation skipped.';
    RETURN;
  END IF;

  INSERT INTO omninode_internal.live_events
    (id, event_id, type, timestamp, source, topic, summary, payload, correlation_id, created_at)
  SELECT
    src.id, src.event_id, src.type, src.timestamp, src.source, src.topic,
    src.summary, src.payload, src.correlation_id, src.created_at
  FROM public.live_events AS src
  ON CONFLICT (event_id) DO NOTHING;

  SELECT count(*) INTO v_src_count FROM public.live_events;
  SELECT count(*) INTO v_dst_count FROM omninode_internal.live_events;

  -- Exact parity, not "at least as many": a destination with MORE rows than
  -- the source (e.g. a stray row from an unrelated write) is exactly as much
  -- a reconciliation failure as one with fewer.
  IF v_dst_count <> v_src_count THEN
    RAISE EXCEPTION
      'OMN-15359: omninode_internal.live_events has % row(s), public.live_events '
      'has % -- row counts must match exactly', v_dst_count, v_src_count;
  END IF;

  SELECT count(*) INTO v_missing_keys
    FROM public.live_events AS src
   WHERE NOT EXISTS (
     SELECT 1 FROM omninode_internal.live_events AS dst
      WHERE dst.event_id = src.event_id
   );
  IF v_missing_keys > 0 THEN
    RAISE EXCEPTION
      'OMN-15359: % source event_id key(s) present in public.live_events but '
      'missing from omninode_internal.live_events', v_missing_keys;
  END IF;

  -- Row-content hash over the shared key set, including `id` (the primary
  -- key, not just event_id the dedup key). Deterministic ordering (ORDER BY
  -- event_id) makes the aggregate hash reproducible across the source and
  -- destination scans. Including `id` matters: without it, a destination row
  -- whose event_id matches but whose id differs (e.g. a stale row from a
  -- prior partial/corrupted copy that ON CONFLICT (event_id) DO NOTHING left
  -- in place) would hash identically to the correct row and this check would
  -- pass over a genuine primary-key divergence.
  SELECT md5(string_agg(row_hash, '' ORDER BY event_id)) INTO v_src_hash
    FROM (
      SELECT
        event_id,
        md5(
          id::text || '|' || event_id || '|' || type || '|' || source || '|' ||
          topic || '|' || summary || '|' || payload || '|' ||
          coalesce(correlation_id, '') || '|' || timestamp::text || '|' ||
          created_at::text
        ) AS row_hash
      FROM public.live_events
    ) AS s;

  SELECT md5(string_agg(row_hash, '' ORDER BY event_id)) INTO v_dst_hash
    FROM (
      SELECT
        event_id,
        md5(
          id::text || '|' || event_id || '|' || type || '|' || source || '|' ||
          topic || '|' || summary || '|' || payload || '|' ||
          coalesce(correlation_id, '') || '|' || timestamp::text || '|' ||
          created_at::text
        ) AS row_hash
      FROM omninode_internal.live_events
      WHERE event_id IN (SELECT event_id FROM public.live_events)
    ) AS d;

  IF v_src_hash IS DISTINCT FROM v_dst_hash THEN
    RAISE EXCEPTION
      'OMN-15359: content hash mismatch between public.live_events and '
      'omninode_internal.live_events over the shared key set (src=%, dst=%)',
      v_src_hash, v_dst_hash;
  END IF;
END$$;

-- Indexes matching public.live_events's shape
-- (docker/migrations/forward/nodes/node_projection_live_events/0000_create_live_events.sql).
CREATE INDEX IF NOT EXISTS idx_omninode_internal_live_events_created_at
  ON omninode_internal.live_events (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_omninode_internal_live_events_topic
  ON omninode_internal.live_events (topic);

CREATE INDEX IF NOT EXISTS idx_omninode_internal_live_events_source
  ON omninode_internal.live_events (source);

CREATE INDEX IF NOT EXISTS idx_omninode_internal_live_events_correlation_id
  ON omninode_internal.live_events (correlation_id)
  WHERE correlation_id IS NOT NULL;

-- Post-condition. Statically provable (no DO/RAISE), matching the
-- OMN-15361 application-database gate's requirement for deployable SQL.
SELECT 1 / count(*) AS omninode_internal_live_events_exists_assertion
  FROM information_schema.tables
 WHERE table_schema = 'omninode_internal' AND table_name = 'live_events';
