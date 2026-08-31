-- OMN-17298: supersedes 0003. Same end state, but the tenant_isolation drop
-- and recreate share ONE transaction, so no path can commit with RLS enabled
-- and no policy on public.capability_scores.
--
-- ===========================================================================
-- HOW THIS FILE WAS FOUND
-- ===========================================================================
-- Not by review. By the gate this ticket added:
-- scripts/validation/check_migration_rls_policy_atomicity.py, RULE B. Run
-- against the checked-in forward tree at dev tip it reported exactly one
-- un-superseded violation, this one -- the THIRD instance of a defect class
-- whose first two were migration 0032 (OMN-17288, superseded by 0033) and the
-- hypothesis that opened OMN-17298. Detection that is not a gate does not
-- hold (Operating Rule 5); the gate found a live one on its first run.
--
-- ===========================================================================
-- WHAT 0003 GOT WRONG -- byte-identical to the 0032 shape
-- ===========================================================================
-- 0003 drops the policy INSIDE its DO block (line 127):
--
--     DROP POLICY IF EXISTS tenant_isolation ON public.capability_scores;
--
-- and recreates it in a standalone statement AFTER `END$$` (lines 163-164).
-- The forward runner is `psql -v ON_ERROR_STOP=1 -f <file>` with NO
-- --single-transaction, so `END$$` COMMITS. Between that commit and the
-- standalone CREATE POLICY there is a real window in which
-- public.capability_scores is ENABLE + FORCE ROW LEVEL SECURITY with zero
-- policies -- which denies every row to every non-owner, non-BYPASSRLS
-- principal. An interruption inside the window (operator ^C, pod eviction,
-- connection reset, OOM) leaves the relation permanently in that state, and
-- the resulting failure is SQLSTATE 42501, which reads at the call site as a
-- missing GRANT rather than as a missing policy.
--
-- THIS ONE ACTUALLY RAN. Unlike 0032, which is applied on no lane, 0003 is
-- recorded in the .201 dev lane's platform_catalog.schema_migrations, applied
-- 2026-08-17 02:30:59.157734+00 (read-only probe as `postgres`, 2026-08-31).
-- The window was traversed on a real database and happened not to be
-- interrupted. Live readback the same day confirms the lane landed in the
-- intended end state -- pg_class shows relrowsecurity=t, relforcerowsecurity=t
-- and exactly one policy on public.capability_scores -- so this file is a
-- CONVERGENCE and a WINDOW CLOSURE, not a repair of observed damage.
--
-- ===========================================================================
-- WHY A NEW FILE INSTEAD OF AN EDIT TO 0003
-- ===========================================================================
-- 0003 is declared in docker/migrations/forward/_ledger/
-- application-migrations.tsv, and scripts/validation/check_migration_append_only.py
-- (OMN-16705) refuses any modification to a file declared at the base ref
-- unless the same diff lands a strictly-higher-ordinal successor and records
-- the supersession. Independently, 0003 IS applied on the dev lane with a
-- recorded content_sha256, so an in-place edit would make bootstrap.sql raise
-- 'conflicting migration checksum in canonical node history' on every later
-- run there -- the OMN-16705 failure class. Supersession is the only path.
-- 0003's bytes are untouched by this change.
--
-- ===========================================================================
-- WHAT THIS FILE DOES -- and deliberately does not do
-- ===========================================================================
-- It restates the policy and the OMN-14894 app_dashboard GRANT, both inside
-- one guarded DO block with nothing following it. It does NOT re-run any part
-- of the tenant_id TEXT->UUID conversion: that conversion is 0003's, it is
-- idempotent there, and re-expressing it here would duplicate a closed
-- three-value literal map this platform has since ruled against (OMN-16930).
-- The predicate below is carried over from 0003 verbatim, including the
-- explicit ::uuid cast on the GUC, and matches CANONICAL_TENANT_PREDICATE in
-- src/omnibase_infra/validation/application_database_domain_enforcement.py.
--
-- Exactly one PERMISSIVE ALL policy, roles PUBLIC, is the canonical shape that
-- module enforces; this file preserves it and adds nothing beside it. A second
-- permissive policy scoped to the writer role would be rejected there as
-- 'extra permissive policy ... widens access', which is why the fix for a
-- policy-less FORCE-RLS relation is an atomic restatement and never an extra
-- admitting policy.
--
-- Idempotent on every path: the leading DROP makes the CREATE safe on a lane
-- where the policy already exists (CREATE POLICY on an existing name is an
-- error, not a no-op), and GRANT is idempotent in PostgreSQL. Re-running this
-- file changes nothing.

DO $$
DECLARE
    v_owner         NAME;
    v_assumed_owner BOOLEAN := FALSE;
BEGIN
    -- A true no-op. Nothing follows this block, so RETURN here ends the file.
    -- This is the other half of the 0032 lesson (OMN-17288 finding 1a): a
    -- RETURN leaves the block, not the file, so a guard is only a no-op when
    -- nothing trails the block it guards.
    IF to_regclass('public.capability_scores') IS NULL THEN
        RAISE NOTICE
            'OMN-17298: public.capability_scores does not exist on this lane; '
            'nothing to restate';
        RETURN;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'app_dashboard'
    ) THEN
        RAISE EXCEPTION
            'OMN-17298: app_dashboard role missing -- apply omnibase_infra '
            'forward migration 094_create_app_dashboard_role.sql (OMN-14899) '
            'before this file. The OMN-14894 ratchet requires the GRANT below '
            'to live in the same file as the policy, so a missing read role '
            'must fail loudly here rather than leave the ratchet unsatisfied.';
    END IF;

    -- ---------------------------------------------------------------------
    -- Ownership. CREATE POLICY and GRANT are owner-only, and 0033 established
    -- that a block which owns its policy statements must hold ownership
    -- itself rather than inheriting whatever identity psql connected as.
    -- Failing here is not a new refusal: a non-owner identity that reached
    -- 0003's standalone CREATE POLICY got `must be owner of table
    -- capability_scores` from PostgreSQL a moment later. This says the same
    -- thing earlier and names the reason.
    -- ---------------------------------------------------------------------
    v_owner := (
        SELECT pg_get_userbyid(relowner)
        FROM pg_catalog.pg_class
        WHERE oid = 'public.capability_scores'::regclass);

    IF NOT pg_has_role(current_user, v_owner, 'USAGE') THEN
        RAISE EXCEPTION
            'OMN-17298: the migrate identity % is not a member of '
            'public.capability_scores'' owner role % -- it can neither restate '
            'the tenant_isolation policy nor, with row-level enforcement '
            'forced on, read the rows any guard would inspect. Refusing to '
            'half-apply.',
            current_user, v_owner;
    END IF;
    -- set_config('role', <name>, is_local => true) is exactly `SET LOCAL ROLE
    -- <name>` and takes the owner as a VALUE, so no SQL text is composed at
    -- runtime and the OMN-15361 gate's dynamic-SQL rejection does not apply.
    -- PL/pgSQL's own `SET` cannot take a variable, which is why the naive
    -- spelling would have had to compose one.
    PERFORM set_config('role', v_owner::text, true);
    v_assumed_owner := TRUE;

    -- ---------------------------------------------------------------------
    -- THE FIX. Drop and recreate in the SAME transaction, so the relation is
    -- never visible to another session with RLS on and no policy -- it never
    -- COMMITS in that state. An abort between these two statements rolls the
    -- drop back with it.
    -- ---------------------------------------------------------------------
    DROP POLICY IF EXISTS tenant_isolation ON public.capability_scores;
    CREATE POLICY tenant_isolation ON public.capability_scores
      FOR ALL
      USING (tenant_id = current_setting('app.tenant_id', true)::uuid)
      WITH CHECK (tenant_id = current_setting('app.tenant_id', true)::uuid);

    -- OMN-14894 ratchet: every file that (re)creates this policy must grant
    -- app_dashboard SELECT in the same file. Idempotent; already granted by
    -- migrations 0002 and 0003, restated here so this file alone satisfies
    -- the ratchet.
    GRANT SELECT ON public.capability_scores TO app_dashboard;

    IF v_assumed_owner THEN
        RESET ROLE;
    END IF;
END$$;
