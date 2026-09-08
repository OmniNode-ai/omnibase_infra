-- OMN-15683: supersedes 0034. Same registry-resolved, single-transaction
-- conversion, with TWO defects repaired -- both of them MEASURED on the live
-- onex-dev table on 2026-09-08, not reasoned about.
--
-- ===========================================================================
-- DEFECT 1 -- 0034 CANNOT CONVERT A MIXED-REPRESENTATION COLUMN
-- ===========================================================================
-- 0034 resolves identity with a SINGLE predicate:
--
--     LEFT JOIN tenant_registry_mirror m ON m.tenant_slug = d.tenant_id
--
-- and has NO branch for a tenant_id that is ALREADY the canonical UUID. That
-- was true when it was written and is false now. Write-time UUID stamping
-- (OMN-16804) is LIVE and writing canonical UUIDs into the still-text column,
-- so delegation_events.tenant_id is a MIXED-REPRESENTATION text column: older
-- rows hold slugs, newer rows hold the canonical UUID as text.
--
-- Measured on onex-dev (DEV-SYSTEM cluster, RDS omnidash_analytics), read-only,
-- 2026-09-08T18:40Z -- full enumeration, reconciled against
-- pg_stat_user_tables.n_live_tup:
--
--     representation   distinct values   rows
--     slug                          5     196
--     canonical UUID                3      26     <-- 0034 cannot resolve these
--     seed fixture                  2       6     <-- deleted below by id
--                                             ---
--                                             229  = n_live_tup
--
-- All THREE canonical-UUID values ARE in tenant_registry_mirror -- as
-- tenant_uuid, which 0034 never looks at. Two independent halves of 0034 then
-- agree the column cannot convert:
--
--   * its fail-closed pre-guard RAISEs, naming those 26 rows' values, and
--   * its resolving UPDATE leaves the scratch column NULL for the same rows,
--     so the 0022 NOT NULL aborts the ALTER even with the guard removed.
--
-- 0036 resolves on BOTH forms:
--
--     ON m.tenant_slug = d.tenant_id OR m.tenant_uuid::text = d.tenant_id
--
-- A row that already holds the canonical UUID therefore resolves TO ITSELF and
-- passes through unchanged -- by resolution, not by a bypass branch, so it is
-- still proven against the registry rather than trusted. A value matching
-- NEITHER form is STILL fail-closed. Nothing is loosened.
--
-- ===========================================================================
-- DEFECT 2 -- 0034's EXCEPTION TEXT MISDIRECTS ON PRESENT DATA
-- ===========================================================================
-- 0034's guard says, of every unresolvable value, that the tenant-registry
-- projection "HAS NOT CAUGHT UP". On onex-dev that sentence was FALSE for all
-- 26 rows: the projection had those tenants, under tenant_uuid. 0034 was
-- written to replace `contains null values`, a message that described a
-- symptom instead of a cause -- and then produced a message that asserts a
-- cause it has not established.
--
-- The guard below asserts nothing it did not test. It names, per value: the
-- value, its row count, and WHICH of the two lookups failed. It then states
-- both possible dispositions -- projection lag, or genuine debris -- and
-- declines to choose between them, because a migration cannot tell them apart.
--
-- ===========================================================================
-- DEFECT 3 -- 0034's HEADER RECIPE INVERTS ITS OWN ANSWER (superseded here)
-- ===========================================================================
-- 0034's deployment-ordering block tells the operator to "Verify with this
-- file's own pre-guard query, not pod status". THAT INSTRUCTION IS UNSAFE AND
-- IS SUPERSEDED BY THIS FILE. Measured, same session:
--
--   Run standalone as the migrate identity against onex-dev, 0034's pre-guard
--   LEFT JOIN returned ZERO unresolvable values -- the exact result that means
--   PASS -- while pg_stat_user_tables.n_live_tup for the same relation read
--   229. delegation_events carries relforcerowsecurity = t, and its live policy
--   is USING (tenant_id = current_setting('app.tenant_id', true)); with the GUC
--   unset that predicate is NULL for every row, so the probe saw an EMPTY TABLE
--   and reported the guard's own success string. FORCE means even the owner is
--   not exempt: SET ROLE to the owner still returned 0, and SET row_security=off
--   returned `ERROR: query would be affected by row-level security policy`
--   rather than a count.
--
-- The migration itself is not blind -- it reaches its guard AFTER SET ROLE to
-- the owner and AFTER `NO FORCE ROW LEVEL SECURITY`. A read-only probe cannot
-- take that DDL step, which is exactly why quoting the migration's SQL as a
-- standalone readiness check inverts the answer.
--
-- THE OPERATOR-SAFE FORM OF THAT CHECK IS A SCRIPT, NOT A QUERY:
--
--     python3 scripts/ci/check_delegation_tenant_conversion_readiness.py --dsn <dsn>
--
-- It is read-only, it reconstructs visibility per-identity when the table is
-- FORCE-blinded, and it REFUSES to print PASS unless its enumerated total
-- reconciles against pg_stat_user_tables.n_live_tup. Do not hand-run the SQL
-- below as a readiness check.
--
-- This file carries the same defence INSIDE the migration (see the BLINDNESS
-- RECONCILIATION block), so that a future runner cannot reach the pre-guard on
-- a half-visible table and pass it.
--
-- ===========================================================================
-- WHY THIS FILE EXISTS INSTEAD OF AN EDIT TO 0034
-- ===========================================================================
-- Identical to the ground recorded for 0032 -> 0033 and 0033 -> 0034:
-- scripts/validation/check_migration_append_only.py (OMN-16705) keys on
-- MANIFEST DECLARATION, not on lane application, and 0034 has been declared in
-- _ledger/application-migrations.tsv since #3062. The only escape the gate
-- accepts for an EDIT is a supersession row whose successor is ADDED by the
-- same diff -- so the repair is a new file either way. 0034's bytes are NOT
-- edited here; the gate passes on the ADD alone, and the supersession row in
-- _ledger/migration-supersessions.tsv is the ledger of 0034's RETIREMENT and
-- the home of the evidence for it.
--
-- 0035 is not in this chain. It is an unrelated GRANT on generation_events and
-- is untouched; 0036 is simply the next free ordinal.
--
-- ===========================================================================
-- UNCHANGED FROM 0034, DELIBERATELY
-- ===========================================================================
-- The TWO-predicate ownership guard (pg_has_role USAGE and SET, OMN-17316) is
-- carried over verbatim, messages included. So are: the debris DELETE by exact
-- correlation_id, the scratch-column resolution (PostgreSQL rejects a subquery
-- in a transform expression), the DROP DEFAULT before the type change, the
-- second NOT NULL guard, the house-tenant DEFAULT, the FORCE restore, and the
-- policy recreate + app_dashboard GRANT INSIDE the guarded block so no path can
-- commit with RLS on and zero policies (OMN-17288). The already-uuid branch
-- still falls THROUGH to that restatement rather than RETURNing past it.
-- Nothing follows END$$. No dynamic SQL is introduced -- the OMN-15361 gate
-- still sees only static statements, and set_config('role', <value>, true)
-- still takes the owner as a VALUE. The OMN-14894 ratchet is satisfied in this
-- file by the GRANT below.
--
-- ===========================================================================
-- DEPLOYMENT ORDERING -- MANDATORY. THIS MIGRATION IS FENCED.
-- ===========================================================================
-- 0036 takes 0034's place in the OMN-15349 baseline fence
-- (fenced-node-migrations.yaml) and in the compose dev/lab lane release
-- (run-forward-migrations.sh). 0031, 0032, 0033 and 0034 stay fenced: they are
-- RETIRED, and leaving a retired id fenced is what keeps it retired.
--
-- It stays in the BASELINE rather than leaving it because it enables FORCE ROW
-- LEVEL SECURITY, which the OMN-15336 item-4 guard rejects for any id absent
-- from the fence manifest ("NOTHING was applied by this migration"). The
-- guard's own message names this remedy: keep the fence entry, add a lane
-- release authorized by an operator ruling.
--
-- Before this file is released on any further lane, ALL of the following must
-- hold:
--
--   1. node_projection_tenant_registry is DEPLOYED and CAUGHT UP -- that is,
--      tenant_registry_mirror resolves every distinct tenant_id present in
--      delegation_events, by tenant_slug OR by tenant_uuid. VERIFY WITH
--      scripts/ci/check_delegation_tenant_conversion_readiness.py, NOT by
--      hand-running the SQL below and NOT by pod status. See DEFECT 3.
--   2. Write-time UUID stamping (OMN-16804) is LIVE. Note what this precondition
--      actually buys: it stops NEW slugs, and it is also what PRODUCED the mixed
--      column this file exists to convert. It makes the conversion durable, not
--      easier.
--   3. The OPERATOR un-gates. Steps 1 and 2 are independent and may proceed in
--      parallel; step 3 requires both.

DO $$
DECLARE
    v_current_type   TEXT;
    v_owner          NAME;
    v_forced         BOOLEAN := FALSE;
    v_assumed_owner  BOOLEAN := FALSE;
    v_convert        BOOLEAN := TRUE;
    v_mirror         REGCLASS;
    v_row_count      BIGINT;
    v_live_tup       BIGINT;
    v_debris_deleted BIGINT;
    v_ambiguous      TEXT;
    v_unresolved     TEXT;
BEGIN
    -- A true no-op. Nothing follows this block, so RETURN here ends the file.
    IF to_regclass('delegation_events') IS NULL THEN
        RAISE NOTICE
            'OMN-15683: delegation_events does not exist on this lane; '
            'nothing to convert';
        RETURN;
    END IF;

    -- Assignment from a scalar subquery, not `SELECT ... INTO`: the OMN-15361
    -- gate parses a top-level `SELECT ... INTO <name>` as PostgreSQL's SELECT
    -- INTO table-creation form and reports the PL/pgSQL variable as an
    -- unqualified relation target. Same statement, same semantics, and the
    -- catalog reference stays visible to the gate.
    v_current_type := (
        SELECT atttypid::regtype::text
        FROM pg_catalog.pg_attribute
        WHERE attrelid = 'delegation_events'::regclass
          AND attname = 'tenant_id'
          AND NOT attisdropped);

    IF v_current_type IS NULL THEN
        RAISE EXCEPTION
            'OMN-15683: delegation_events.tenant_id column not found -- '
            'expected migration 0022 to have already landed it';
    ELSIF v_current_type = 'uuid' THEN
        -- NOT a RETURN (the 0032 defect). The conversion is skipped, but the
        -- policy and grant restatement below still has to run: on a lane where
        -- the column is already uuid, this file is what leaves the relation in
        -- the intended end state.
        RAISE NOTICE
            'OMN-15683: delegation_events.tenant_id is already uuid; '
            'skipping conversion and restating the policy and grant only';
        v_convert := FALSE;
    ELSIF v_current_type <> 'text' THEN
        RAISE EXCEPTION
            'OMN-15683: delegation_events.tenant_id has unexpected type %, '
            'expected text or uuid -- operator schema ruling required',
            v_current_type;
    END IF;

    -- ---------------------------------------------------------------------
    -- Ownership, once, for every path that gets here. Carried over from 0034
    -- verbatim, including both membership predicates (OMN-17316).
    -- ---------------------------------------------------------------------
    v_owner := (
        SELECT pg_get_userbyid(relowner)
        FROM pg_catalog.pg_class
        WHERE oid = 'delegation_events'::regclass);
    v_forced := (
        SELECT relforcerowsecurity
        FROM pg_catalog.pg_class
        WHERE oid = 'delegation_events'::regclass);

    -- TWO predicates, because since PostgreSQL 16 INHERIT and SET are
    -- INDEPENDENT membership options (OMN-17316). 0033 tested only the first
    -- and then exercised the second, so a membership created
    -- `WITH INHERIT TRUE, SET FALSE` passed its guard and aborted at the
    -- set_config below on a bare `permission denied to set role "<owner>"`.
    -- Neither predicate substitutes for the other, and 'MEMBER' substitutes
    -- for neither: it is also true under SET FALSE.
    IF NOT pg_has_role(current_user, v_owner, 'USAGE') THEN
        RAISE EXCEPTION
            'OMN-16930: the migrate identity % holds no INHERITED privilege '
            'from delegation_events'' owner role % (pg_has_role USAGE is '
            'false) -- it can neither restate the tenant_isolation policy '
            'nor, under FORCE ROW LEVEL SECURITY, see the rows its guards '
            'inspect (every one would be RLS-blinded and silently return zero '
            'rows, the OMN-16493 failure mode). Refusing to convert '
            'half-blind. Fix: GRANT % TO % WITH INHERIT TRUE.',
            current_user, v_owner, v_owner, current_user;
    END IF;

    IF NOT pg_has_role(current_user, v_owner, 'SET') THEN
        RAISE EXCEPTION
            'OMN-17316: the migrate identity % inherits from '
            'delegation_events'' owner role % but is NOT permitted to SET '
            'ROLE to it (pg_has_role SET is false -- a PostgreSQL 16 '
            'membership granted WITH SET FALSE). The next statement is '
            'set_config(''role'', ...), which is SET LOCAL ROLE and would '
            'abort on a bare `permission denied to set role "%"`; without it '
            'this block cannot hold ownership for FORCE ROW LEVEL SECURITY, '
            'CREATE POLICY or GRANT. Refusing to start a conversion that '
            'cannot finish. Fix: GRANT % TO % WITH SET TRUE.',
            current_user, v_owner, v_owner, v_owner, current_user;
    END IF;
    -- set_config('role', <name>, is_local => true) is exactly `SET LOCAL ROLE
    -- <name>` and takes the owner as a VALUE, so no SQL text is composed at
    -- runtime and the OMN-15361 gate's dynamic-SQL rejection does not apply.
    PERFORM set_config('role', v_owner::text, true);
    v_assumed_owner := TRUE;

    IF v_convert THEN
        IF v_forced THEN
            ALTER TABLE delegation_events NO FORCE ROW LEVEL SECURITY;
        END IF;

        -- -----------------------------------------------------------------
        -- BLINDNESS RECONCILIATION -- runs BEFORE anything is enumerated or
        -- deleted, under this file's own SET ROLE + NO FORCE.
        --
        -- FRICTION, measured 2026-09-08T18:40:02Z: 0034's pre-guard query,
        -- run standalone under FORCE ROW LEVEL SECURITY with app.tenant_id
        -- unset, returned the EXACT RESULT THAT MEANS PASS on a table holding
        -- 229 rows. An RLS-blinded read and a genuinely resolvable table are
        -- indistinguishable to the guard itself. This block makes them
        -- distinguishable, so that no future runner reaching this point on a
        -- half-visible table can pass.
        --
        -- Two checks, exact first:
        --
        --   (a) row_security_active() is an EXACT fact, not an estimate. After
        --       the SET ROLE and the NO FORCE above it must be false. If it is
        --       true, every guard below is reading a filtered table and its
        --       answers are worthless.
        --   (b) pg_stat_user_tables.n_live_tup is corroborating and is an
        --       ESTIMATE -- stated plainly rather than implied, because an
        --       exact equality against it would abort this migration on
        --       ordinary autovacuum lag. It is therefore used ONLY for the
        --       one signature it can settle without exactness: a visible
        --       count of ZERO on a relation the statistics say holds rows.
        --       That is the blindness signature and nothing else produces it.
        --
        -- The read-only operator form of this check reconciles the FULL sum
        -- rather than just the zero case, because it enumerates per-identity
        -- and can afford to; see
        -- scripts/ci/check_delegation_tenant_conversion_readiness.py.
        -- -----------------------------------------------------------------
        v_row_count := (
            SELECT count(*)
            FROM delegation_events);
        v_live_tup := (
            SELECT n_live_tup
            FROM pg_catalog.pg_stat_user_tables
            WHERE relid = 'delegation_events'::regclass);

        IF row_security_active('delegation_events') THEN
            RAISE EXCEPTION
                'OMN-15683: row-level security is STILL ACTIVE for % on '
                'delegation_events after SET ROLE to the owner role % and '
                'ALTER TABLE ... NO FORCE ROW LEVEL SECURITY. Every guard '
                'below would read a POLICY-FILTERED table and could report '
                'success on rows it never saw -- the failure measured on '
                'onex-dev 2026-09-08, where this file''s predecessor returned '
                'its own PASS string against 229 unread rows. Refusing to '
                'convert on a half-visible table.',
                current_user, v_owner;
        END IF;

        IF v_row_count = 0 AND COALESCE(v_live_tup, 0) > 0 THEN
            RAISE EXCEPTION
                'OMN-15683: delegation_events reads as EMPTY to % (count(*) = '
                '0) while pg_stat_user_tables.n_live_tup for the same relation '
                'is %. n_live_tup is an estimate and is never asserted exactly '
                'here, but no amount of statistics drift turns a populated '
                'table into a zero count: this is the row-level-security '
                'blindness signature. Refusing to convert a table this session '
                'cannot see. Investigate visibility first -- relrowsecurity, '
                'relforcerowsecurity, the tenant_isolation policy, and '
                'app.tenant_id.',
                current_user, v_live_tup;
        END IF;

        RAISE NOTICE
            'OMN-15683: visibility reconciled -- count(*) = %, '
            'pg_stat_user_tables.n_live_tup = % (estimate), '
            'row_security_active = false',
            v_row_count, v_live_tup;

        -- -----------------------------------------------------------------
        -- Pre-tenancy debris with no canonical identity, by exact
        -- correlation_id. Carried over from 0034 verbatim.
        --
        -- This is NOT a tenant map and must never be extended into one. These
        -- six rows are the SEED-A/SEED-B fixtures under the literal tenant
        -- values 11111111-... and 22222222-..., neither of which appears in
        -- omninode_cloud.public.tenants -- they have no registry identity to
        -- resolve to, at any point in the future, by construction. Operator
        -- ruling of 2026-08-27 on the OMN-16493 enumeration: map-to-canonical
        -- for every value that HAS a registry identity -- which is the JOIN
        -- below, not a list -- and delete for the rows that have none.
        --
        -- Deleting by exact correlation_id rather than by tenant_id is
        -- deliberate: a future row landing under a reused literal must not be
        -- swept up by a predicate written today.
        -- -----------------------------------------------------------------
        DELETE FROM delegation_events
        WHERE correlation_id IN (
            'SEED-A-1', 'SEED-A-2', 'SEED-A-3',
            'SEED-B-1', 'SEED-B-2', 'SEED-B-3'
        );
        GET DIAGNOSTICS v_debris_deleted = ROW_COUNT;
        RAISE NOTICE
            'OMN-15683: removed % pre-tenancy debris row(s) with no registry '
            'identity', v_debris_deleted;

        v_row_count := (
            SELECT count(*)
            FROM delegation_events);

        v_mirror := to_regclass('tenant_registry_mirror');

        IF v_mirror IS NULL THEN
            IF v_row_count > 0 THEN
                RAISE EXCEPTION
                    'OMN-15683: tenant_registry_mirror does not exist on this '
                    'lane, but delegation_events holds % row(s) that need '
                    'their tenant identity resolved. This migration resolves '
                    'identity by JOINing that mirror -- it does not carry a '
                    'literal map. ORDERING VIOLATED: '
                    'node_projection_tenant_registry migration '
                    '0000_create_tenant_registry_mirror.sql must be applied, '
                    'and the projection must have caught up, BEFORE this file '
                    'runs. Node directories are applied in sort order and '
                    'node_projection_delegation sorts first, so on a lane '
                    'with pre-existing delegation rows the tenant-registry '
                    'node must be deployed first. See OMN-16930.',
                    v_row_count;
            END IF;
            -- Empty table: nothing to resolve, so the conversion is
            -- unambiguous. This is the first-ever-bootstrap path (0007 creates
            -- the table in the same run). No row evaluates the USING
            -- expression.
            RAISE NOTICE
                'OMN-15683: tenant_registry_mirror absent and '
                'delegation_events is empty -- converting with no rows to '
                'resolve (fresh bootstrap)';
        ELSE
            -- -------------------------------------------------------------
            -- DETERMINISM GUARD, new in 0036 and required BY the OR.
            --
            -- 0034's single-predicate JOIN could match at most one mirror row
            -- per value. A two-form JOIN can, in principle, match two -- and
            -- `UPDATE ... FROM` with two matching source rows picks ONE
            -- ARBITRARILY and reports no error, which would silently assign a
            -- tenant its neighbour's identity. It cannot happen with
            -- well-formed data (no tenant_slug is the text form of another
            -- tenant's uuid), and it is checked anyway, because "cannot
            -- happen" is the premise every entry in this file's supersession
            -- chain was written to retire.
            -- -------------------------------------------------------------
            -- Wrapped in leading/trailing newlines HERE rather than in the
            -- RAISE below, so the message needs exactly ONE placeholder.
            -- Adjacent placeholders cannot be written in a RAISE format string:
            -- `%%` is an escaped literal percent, so `%%%` is "literal percent
            -- then one placeholder", not three. string_agg returns NULL on zero
            -- rows and NULL || anything is NULL, so the IS NOT NULL test below
            -- is unaffected.
            v_ambiguous := (
                SELECT chr(10) || string_agg(line, chr(10) ORDER BY line) || chr(10)
                FROM (
                    SELECT format(
                               '  %s resolves to %s DIFFERENT registry tenants',
                               quote_literal(d.tenant_id),
                               count(DISTINCT m.tenant_uuid)) AS line
                    FROM delegation_events d
                    JOIN tenant_registry_mirror m
                      ON m.tenant_slug = d.tenant_id
                      OR m.tenant_uuid::text = d.tenant_id
                    GROUP BY d.tenant_id
                    HAVING count(DISTINCT m.tenant_uuid) > 1
                ) ambiguous);

            IF v_ambiguous IS NOT NULL THEN
                RAISE EXCEPTION
                    'OMN-15683: tenant_registry_mirror resolves these '
                    'delegation_events tenant values to MORE THAN ONE '
                    'registry tenant, so the conversion is not '
                    'deterministic:%'
                    'This means one tenant''s slug is the text form of another '
                    'tenant''s uuid, or the mirror holds duplicate rows. '
                    'Refusing to pick one arbitrarily. Resolution is an '
                    'operator ruling recorded on the ticket, never a literal '
                    'added to this file.',
                    v_ambiguous;
            END IF;

            -- -------------------------------------------------------------
            -- FAIL-CLOSED PRE-GUARD. A value that resolves under NEITHER form
            -- aborts. Still fail-closed; still no literal in the resolution
            -- path.
            --
            -- WHAT CHANGED FROM 0034, and why it matters more than it looks:
            -- 0034 asserted a CAUSE it had not established ("the projection
            -- HAS NOT CAUGHT UP"), and on onex-dev that sentence was false for
            -- every row it named -- the tenants were present, under
            -- tenant_uuid. This message states only what was tested: the
            -- value, how many rows carry it, and WHICH lookup failed. The two
            -- dispositions are both named and neither is asserted, because
            -- this migration cannot tell them apart and must not pretend to.
            -- -------------------------------------------------------------
            v_unresolved := (
                SELECT chr(10) || string_agg(line, chr(10) ORDER BY line) || chr(10)
                FROM (
                    SELECT format(
                               '  %s -- %s row(s) -- %s',
                               quote_literal(d.tenant_id),
                               count(*),
                               CASE
                                   WHEN d.tenant_id ~* '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
                                       THEN 'well-formed UUID, but no tenant_registry_mirror row has tenant_uuid = this value'
                                   ELSE 'not a UUID, and no tenant_registry_mirror row has tenant_slug = this value'
                               END) AS line
                    FROM delegation_events d
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM tenant_registry_mirror m
                        WHERE m.tenant_slug = d.tenant_id
                           OR m.tenant_uuid::text = d.tenant_id)
                    GROUP BY d.tenant_id
                ) unresolved);

            IF v_unresolved IS NOT NULL THEN
                RAISE EXCEPTION
                    'OMN-15683: delegation_events holds tenant values that '
                    'tenant_registry_mirror resolves under NEITHER form. Both '
                    'lookups were performed for each value below -- '
                    'tenant_slug = <value> AND tenant_uuid::text = <value> -- '
                    'and both returned no row:%'
                    'This migration does NOT claim to know why. Exactly two '
                    'dispositions are possible and they are distinguished off '
                    'this table, not on it: (a) the tenant-registry projection '
                    '(node_projection_tenant_registry, consuming '
                    'onex.tenant.events) has not yet materialised a tenant '
                    'that DOES exist in omninode_cloud.public.tenants -- '
                    'confirm the writer is running and consuming, then re-run; '
                    'or (b) the value is genuinely absent from the registry, '
                    'in which case it is debris and its disposition is an '
                    'operator ruling recorded on the ticket -- never a literal '
                    'added to this file. Refusing to invent, default, or drop '
                    'an identity. Read-only triage: '
                    'scripts/ci/check_delegation_tenant_conversion_readiness.py '
                    '--dsn <dsn>. See OMN-15683, OMN-16930.',
                    v_unresolved;
            END IF;

            -- -------------------------------------------------------------
            -- Resolve through a scratch column, NOT a subquery in the USING
            -- clause. PostgreSQL rejects `ALTER COLUMN ... TYPE ... USING
            -- (SELECT ...)` outright: `ERROR: cannot use subquery in transform
            -- expression`. A transform expression may only reference columns
            -- of the row being rewritten.
            --
            -- So the JOIN happens one statement earlier, as a real UPDATE ...
            -- FROM against the mirror. The predicate carries BOTH forms: a row
            -- whose tenant_id is already the canonical UUID matches on
            -- m.tenant_uuid::text and is assigned that same uuid -- it passes
            -- through UNCHANGED, but by resolution against the registry rather
            -- than by an unchecked bypass, so it is proven and not trusted.
            -- Still resolution-by-JOIN, still no literal; the scratch column
            -- exists only inside this transaction and is dropped below.
            -- -------------------------------------------------------------
            ALTER TABLE delegation_events
                ADD COLUMN IF NOT EXISTS omn16930_resolved_tenant_uuid UUID;
            UPDATE delegation_events d
            SET omn16930_resolved_tenant_uuid = m.tenant_uuid
            FROM tenant_registry_mirror m
            WHERE m.tenant_slug = d.tenant_id
               OR m.tenant_uuid::text = d.tenant_id;
        END IF;

        -- The pre-existing tenant_isolation POLICY (migration 0023) depends on
        -- this column -- PostgreSQL refuses ALTER COLUMN ... TYPE while any
        -- policy references it. Dropped here and recreated at the end of THIS
        -- block, inside the same transaction.
        DROP POLICY IF EXISTS tenant_isolation ON delegation_events;

        -- The TEXT DEFAULT ('omninode', migration 0022) is not castable to
        -- uuid; Postgres tries to cast the DEFAULT expression itself during the
        -- type change and 'omninode'::uuid is not a valid uuid literal.
        ALTER TABLE delegation_events ALTER COLUMN tenant_id DROP DEFAULT;

        -- Second, independent fail-closed guard, retained from 0031: the
        -- resolving UPDATE above leaves the scratch column NULL for any value
        -- the pre-guard did not catch (e.g. a row inserted between the guard
        -- and the ALTER inside this transaction), and NULL is rejected by the
        -- column's existing NOT NULL constraint from 0022 -- aborting the
        -- statement. No partial conversion, no invented UUID, no silent
        -- passthrough.
        IF v_mirror IS NULL THEN
            ALTER TABLE delegation_events
                ALTER COLUMN tenant_id TYPE UUID USING (NULL::uuid);
        ELSE
            ALTER TABLE delegation_events
                ALTER COLUMN tenant_id TYPE UUID
                USING (omn16930_resolved_tenant_uuid);
        END IF;

        -- The scratch column never outlives this transaction.
        ALTER TABLE delegation_events
            DROP COLUMN IF EXISTS omn16930_resolved_tenant_uuid;

        -- The house tenant UUID: uuid5 of house-tenant.omninode.ai, matching
        -- the DEFAULT 0031 would have set. This is a column DEFAULT for rows
        -- that omit tenant_id, not an identity map -- it resolves nothing and
        -- converts nothing.
        ALTER TABLE delegation_events
            ALTER COLUMN tenant_id SET DEFAULT '820272f9-4aaf-5add-a2df-0af942852ab2'::uuid;

        IF v_forced THEN
            ALTER TABLE delegation_events FORCE ROW LEVEL SECURITY;
        END IF;
    END IF;

    -- ---------------------------------------------------------------------
    -- Same transaction as the DROP POLICY above (OMN-17288).
    --
    -- On the conversion path this re-establishes the policy the type change
    -- required dropping; on the already-uuid path it restates it so this file
    -- alone leaves the relation in the intended end state. Either way the
    -- relation is never visible to another session with RLS on and no policy,
    -- because it never COMMITS in that state.
    --
    -- The leading DROP is what makes the restatement idempotent: on the
    -- already-uuid path nothing above dropped it, and CREATE POLICY on an
    -- existing name is an error, not a no-op.
    -- ---------------------------------------------------------------------
    DROP POLICY IF EXISTS tenant_isolation ON delegation_events;
    CREATE POLICY tenant_isolation ON delegation_events
      FOR ALL
      USING (tenant_id = current_setting('app.tenant_id', true)::uuid)
      WITH CHECK (tenant_id = current_setting('app.tenant_id', true)::uuid);

    -- OMN-14894 ratchet: every file that (re)creates this policy must grant
    -- app_dashboard SELECT in the same file. Idempotent; already granted by
    -- migration 0023, restated here so this file alone satisfies the ratchet.
    GRANT SELECT ON delegation_events TO app_dashboard;

    IF v_assumed_owner THEN
        RESET ROLE;
    END IF;
END$$;
