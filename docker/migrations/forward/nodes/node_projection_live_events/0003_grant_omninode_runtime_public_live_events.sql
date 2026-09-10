-- OMN-18109: the omninode_runtime TABLE grant on the relation the STANDALONE
--            live-events writer actually writes.
-- Target DB: omnidash_analytics (NODE_POSTGRES_DB)
-- Node: node_projection_live_events
--
-- ============================================================================
-- WHAT IS BROKEN
-- ============================================================================
--   OMN-18109 moves the dev lane's standalone live-events writer off
--   `role_omnidash` and onto `omninode_runtime`, the principal the topology
--   declares for this node's `omninode_internal` domain. That fixes the live
--   refusal -- `Failed to update watermark: permission denied for schema
--   omninode_internal`, 411 occurrences in thirty minutes on the .201 dev lane
--   to 2026-09-10T00:56:40Z -- and, without this file, would immediately trade
--   it for a different one.
--
--   The reason is a split between what the contract DECLARES and what the
--   handler WRITES. `node_projection_live_events/contract.yaml` declares
--   `db_io.db_tables: [{name: live_events, schema: omninode_internal}]`, but
--   `HandlerLiveEventsProjectionRunner.__init__` builds its SQL from the bare
--   `name` (`_by_role = {t["role"]: t["name"] ...}`) and never reads `schema`.
--   The statement it issues is therefore unqualified and resolves through
--   `search_path` into `public`. Both relations exist -- read live on that
--   lane, `information_schema.tables` returns `omninode_internal.live_events`
--   AND `public.live_events` -- and the standalone writer writes the second
--   one while 099 and this lineage's 0002 granted only the first.
--
--   Live grants on `public.live_events`, read read-only 2026-09-10, names only:
--
--       role_omnidash    : DELETE, INSERT, SELECT, UPDATE
--       omninode_runtime : -- none --
--
--   `role_omnidash` holds them by migration 096's blanket
--   `GRANT ... ON ALL TABLES IN SCHEMA public`, which is exactly the
--   undeclared mechanism the writers have been resting on.
--
-- ============================================================================
-- WHAT THIS FILE DOES AND DOES NOT DO
-- ============================================================================
--   DOES grant the same three privileges this principal already holds on the
--   `omninode_internal` twin, at the PHYSICAL location the handler writes. A
--   grant must name the physical schema, so it says `public` -- the convention
--   `node_contract_registry/0001` and `node_projection_tenant_registry/0001`
--   state in their own words.
--
--   DOES NOT converge the two relations, move the write, or change the
--   contract. The contract-declares-`omninode_internal`-but-the-handler-writes
--   -`public` split is a real defect and it is NOT this file's: it is recorded
--   as out of scope on OMN-18109 and belongs with the OMN-15359 schema cutover
--   that owns every other unqualified relation in this corpus.
--
--   DOES NOT revoke anything from `role_omnidash`. Removing the blanket grant
--   is its own decision with its own blast radius; this file only makes the
--   DECLARED principal able to do what it is declared to do.
--
--   DOES NOT grant DELETE. A projection writer upserts; it does not reshape
--   the table. Same invariant 096 states for `role_omnidash`, 099 for this
--   principal on `omninode_internal.live_events`, and
--   `node_projection_registration/0006` for it on `node_service_registry`.
--   SELECT is required alongside INSERT/UPDATE because the adapter's write is
--   `INSERT ... ON CONFLICT DO UPDATE`, which reads back.
--
--   NO SEQUENCE HALF IS NEEDED, and that is measured rather than assumed:
--   `public.live_events` has no `nextval` column default and no owned
--   sequence -- a catalog probe for sequences named after this relation
--   returned zero rows on the same lane, unlike the sequence-backed relations
--   OMN-17447 covers.
--
-- Idempotency: GRANT is idempotent; re-running is a no-op. Nothing here
-- touches RLS, ownership, or any role attribute. `public.live_events` carries
-- `relrowsecurity = f`, so this changes no row-visibility behaviour.

-- ---------------------------------------------------------------------------
-- 1. Schema USAGE, mirroring topology
--    `principals.omninode_runtime.grants[object_type: SCHEMA, schema: public]`.
--    Idempotent, and re-asserted here for the same reason 099 re-asserts the
--    omninode_internal one: a migration must not assume a sibling file ran.
-- ---------------------------------------------------------------------------
GRANT USAGE ON SCHEMA public TO omninode_runtime;

-- ---------------------------------------------------------------------------
-- 2. Table grant
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE ON public.live_events TO omninode_runtime;

-- ---------------------------------------------------------------------------
-- 3. Assertions: fail the migration if a grant did not take. Division by zero
--    when the grant is absent -- the fail-loud shape 099,
--    node_projection_registration/0006 and node_contract_registry/0001 use.
--
--    All three directions are asserted, not just INSERT. A lane where the
--    INSERT landed and the SELECT did not would accept writes and fail every
--    `ON CONFLICT DO UPDATE` read-back, which is the harder failure to
--    diagnose.
-- ---------------------------------------------------------------------------
SELECT 1 / count(*) AS public_live_events_insert_grant_assertion
FROM information_schema.role_table_grants
WHERE table_schema = 'public'
  AND table_name = 'live_events'
  AND grantee = 'omninode_runtime'
  AND privilege_type = 'INSERT';

SELECT 1 / count(*) AS public_live_events_select_grant_assertion
FROM information_schema.role_table_grants
WHERE table_schema = 'public'
  AND table_name = 'live_events'
  AND grantee = 'omninode_runtime'
  AND privilege_type = 'SELECT';

SELECT 1 / count(*) AS public_live_events_update_grant_assertion
FROM information_schema.role_table_grants
WHERE table_schema = 'public'
  AND table_name = 'live_events'
  AND grantee = 'omninode_runtime'
  AND privilege_type = 'UPDATE';
