-- OMN-15683: supersedes 0036. The conversion 0036 describes, performed in an
-- order that survives the PRIVILEGE topology of the target database.
--
-- Everything 0036 does is done here. Nothing is loosened, nothing is skipped,
-- no guard is weakened. The ONE change is WHEN the tenant_registry_mirror
-- reads happen relative to the role switch -- and that one change is the
-- difference between a migration that converts and a migration that aborts.
--
-- ===========================================================================
-- THE DEFECT -- MEASURED ON onex-dev, NOT REASONED ABOUT
-- ===========================================================================
-- 0036 was applied to onex-dev (the DEV-SYSTEM cluster's RDS
-- omnidash_analytics) by staging deploy run 34281092205 on 2026-09-08. The
-- fence subtraction selected it, 0034 was skipped as retired, its own blindness
-- reconciliation fired correctly ("count(*) = 233, n_live_tup = 233 (estimate),
-- row_security_active = false") and its debris DELETE removed 6 rows.
--
-- It then ABORTED:
--
--     ERROR: permission denied for table tenant_registry_mirror
--     CONTEXT: PL/pgSQL inline_code_block line 262
--
-- Line 262 is the v_ambiguous determinism guard -- the first statement in 0036
-- that JOINs tenant_registry_mirror. The whole transaction rolled back:
-- delegation_events.tenant_id is still text, the six debris rows came back, and
-- no node_schema_migrations row records 0036 on any lane.
--
-- CAUSE, READ LIVE FROM THE CATALOG (not inferred):
--
--     delegation_events        owner  role_omninode_owner
--     tenant_registry_mirror   owner  role_omnidash
--     tenant_registry_mirror   ACL    {role_omnidash=arwdDxt/role_omnidash,
--                                      app_dashboard=r,
--                                      omninode_runtime=arw,
--                                      jake_ro=r}
--
--     has_table_privilege('role_omninode_owner',
--                         'tenant_registry_mirror', 'SELECT')  ->  f
--
-- role_omninode_owner is ABSENT from that ACL. 0036 does
-- set_config('role', v_owner, true) -- SET LOCAL ROLE to delegation_events'
-- owner -- so that it may issue ALTER TABLE ... NO FORCE ROW LEVEL SECURITY and
-- see its own rows. Every mirror read then happens as role_omninode_owner, and
-- in that role the mirror is unreadable.
--
-- The two relations this migration touches have DIFFERENT OWNERS and are read
-- in DIFFERENT ROLES. 0036 assumed one privilege set for the whole block.
--
-- ===========================================================================
-- WHY THE LAB PROOF WAS GREEN AND ONEX-DEV WAS NOT
-- ===========================================================================
-- 0036 was proven by execution on a scratch database on the .201 dev-lane
-- Postgres, seeded to the exact onex-dev ROW CENSUS: the same row count, the
-- same slug/canonical-UUID mixture, FORCE ROW LEVEL SECURITY on with the 0023
-- TEXT policy, mirror rows carrying the three canonical UUIDs under
-- tenant_uuid. Every leg passed, including the RED control on 0034.
--
-- That reconstruction reproduced DATA. It did not reproduce OWNERSHIP, GRANTS
-- or ROLE MEMBERSHIP: both tables were created under one scratch owner, so the
-- cross-owner read that fails in production was free there. A lab proof green
-- on every functional axis was silent on the privilege axis.
--
-- Recorded as a CLASS, not as one bug, in the omni_home rolling work ledger
-- (FRICTION 2026-09-08T22:25:10Z). The harness change that closes it ships
-- WITH this file: the hermetic PG16 integration test now seeds the ownership
-- and ACL topology alongside the census, and
-- scripts/ci/check_delegation_tenant_conversion_readiness.py grew a PRIVILEGE
-- leg that derives, from THIS FILE'S OWN BYTES, which relations are read in
-- which role, and asserts has_table_privilege for each (role, relation) pair.
-- Pointed at 0036's bytes against the onex-dev topology, that leg names
-- (role_omninode_owner, tenant_registry_mirror) -- which is the leg that would
-- have caught this before the deploy.
--
-- ===========================================================================
-- THE REPAIR -- READ THE MIRROR BEFORE THE ROLE SWITCH, NOT AFTER
-- ===========================================================================
-- The two reads have irreconcilable role requirements inside one statement:
--
--   * delegation_events must be read AS ITS OWNER and AFTER
--     ALTER TABLE ... NO FORCE ROW LEVEL SECURITY, or every guard reads a
--     policy-filtered table and reports success on rows it never saw (the
--     failure 0036's BLINDNESS RECONCILIATION block exists to catch).
--   * tenant_registry_mirror must be read AS THE MIGRATE IDENTITY, because
--     delegation_events' owner has no SELECT on it.
--
-- So the JOIN cannot be issued in either role. It is SPLIT:
--
--   PHASE A, as the migrate identity, BEFORE any SET ROLE -- copy the mirror's
--   two resolution columns into a session-local TEMP table. This is the only
--   statement in the file that reads tenant_registry_mirror.
--
--   PHASE B, after SET ROLE, exactly as 0036 -- every guard and the resolving
--   UPDATE run unchanged, joining the TEMP SNAPSHOT instead of the mirror.
--
-- The snapshot is taken inside the same transaction as the reads that consume
-- it, so it is the same MVCC snapshot 0036 would have joined against: this
-- reorders WHO reads, not WHAT is read.
--
-- WHY A SNAPSHOT AND NOT A GRANT. The alternative repair is
-- GRANT SELECT ON tenant_registry_mirror TO role_omninode_owner as the file's
-- first statement. It is rejected here on evidence and on blast radius:
--
--   * The migrate identity demonstrably CAN read the mirror. On 2026-09-08,
--     the same session that later measured the ACL above ran 0034's pre-guard
--     -- a LEFT JOIN against tenant_registry_mirror -- standalone as the
--     migrate identity against onex-dev, and it returned a RESULT SET (zero
--     unresolvable values, wrongly, because delegation_events was RLS-blinded
--     to that identity). A missing SELECT on the mirror would have raised
--     permission denied there instead of returning rows. So the privilege this
--     file needs already exists, and a GRANT would be adding a privilege to
--     work around not having used the one already held.
--   * A GRANT is PERSISTENT and CROSS-OWNER. It would permanently widen
--     role_omninode_owner's reach into a relation owned by role_omnidash, to
--     make one transaction succeed. The temp snapshot is ON COMMIT DROP: it
--     cannot outlive this transaction, and no privilege survives the file.
--   * A migration cannot GRANT on a table it does not own anyway. The grantor
--     would have to be role_omnidash or a superuser, and this file holds
--     neither -- it would abort on a different permission error.
--
-- If a future lane genuinely denies the mirror to the migrate identity too,
-- this file refuses BY NAME in Phase A rather than aborting opaquely 200 lines
-- later, and states the remedy. That refusal is the honest form of the branch
-- the brief offered as an alternative: the GRANT is an OPERATOR act recorded on
-- the ticket, not a statement this file issues on its own authority.
--
-- WHY THE TEMP TABLE IS GRANTED TO PUBLIC. It is created by the migrate
-- identity and read, one statement later, as role_omninode_owner -- which is
-- not a member of the migrate identity and therefore holds nothing on it. The
-- grant is scoped to a relation in this session's pg_temp schema, which
-- PostgreSQL forbids any other session from accessing at all ("cannot access
-- temporary tables of other sessions"), and which ON COMMIT DROP destroys when
-- this transaction ends. PUBLIC here is one role for one statement inside one
-- transaction, and is written that way because GRANT does not accept a
-- PL/pgSQL variable for its grantee and composing the role name into SQL text
-- would introduce the dynamic SQL the OMN-15361 gate rejects.
--
-- ===========================================================================
-- WHY THIS FILE EXISTS INSTEAD OF AN EDIT TO 0036
-- ===========================================================================
-- RESOLVED AGAINST THE GATE, not assumed. 0036 merged (omnibase_infra#3336,
-- squash c7b282d8b) and was never successfully applied anywhere -- so the
-- tempting reading is that it is not "landed" and may be edited. That reading
-- is FALSE, and the gate's own source settles it:
-- scripts/validation/check_migration_append_only.py keys on MANIFEST
-- DECLARATION AT THE BASE REF (`declared_artifacts()` over
-- _ledger/application-migrations.tsv), never on lane application. 0036 is
-- declared there at dev tip with content_sha256
-- ea1ce1abb52c9351a5bc2e37862fca244ca8fea3a146ca435fa443c4d6e6e562. An edit is
-- therefore refused whether or not any database ever ran it, and the only
-- escape the gate accepts is a supersession row whose successor is ADDED by the
-- same diff. That is exactly the shape 0034 -> 0036 took, and it is the shape
-- here. The claim is proven by falsification in
-- tests/ci/test_migration_append_only_guard_omn16705.py rather than asserted.
--
-- 0036's bytes are NOT touched. It is retired in place and KEEPS its baseline
-- fence entry -- leaving a retired id fenced is what keeps it retired -- and it
-- simply no longer appears in any lane release.
--
-- ===========================================================================
-- UNCHANGED FROM 0036, DELIBERATELY
-- ===========================================================================
-- Byte-equivalent in effect: the two-predicate ownership guard (pg_has_role
-- USAGE and SET, OMN-17316), the BLINDNESS RECONCILIATION, the debris DELETE by
-- exact correlation_id, the determinism guard, the fail-closed pre-guard naming
-- the value / its row count / which lookup failed, the scratch-column
-- resolution on BOTH forms, the DROP DEFAULT, the NOT NULL backstop, the
-- house-tenant DEFAULT, the NO FORCE / FORCE bracket, and the policy recreate
-- with ::uuid plus the app_dashboard GRANT inside the guarded block (OMN-17288,
-- OMN-14894). The already-uuid branch still falls THROUGH to that restatement
-- rather than RETURNing past it. Nothing follows END$$. No dynamic SQL is
-- introduced -- set_config('role', <value>, true) still takes the owner as a
-- VALUE, so the OMN-15361 gate still sees only static statements.
--
-- ===========================================================================
-- DEPLOYMENT ORDERING -- MANDATORY. THIS MIGRATION IS FENCED.
-- ===========================================================================
-- 0037 takes 0036's place in the compose dev/lab lane release
-- (run-forward-migrations.sh) and in the k8s Job's release array. 0031, 0032,
-- 0033, 0034 and 0036 all stay in the OMN-15349 baseline fence: they are
-- RETIRED, and leaving a retired id fenced is what keeps it retired.
--
-- 0037 stays in the BASELINE rather than leaving it because it enables FORCE
-- ROW LEVEL SECURITY, which the OMN-15336 item-4 guard rejects for any id
-- absent from the fence manifest ("NOTHING was applied by this migration").
-- The guard's own message names this remedy: keep the fence entry, add a lane
-- release authorized by an operator ruling.
--
-- Before this file is released on any further lane, ALL of the following must
-- hold:
--
--   1. node_projection_tenant_registry is DEPLOYED and CAUGHT UP -- that is,
--      tenant_registry_mirror resolves every distinct tenant_id present in
--      delegation_events, by tenant_slug OR by tenant_uuid. VERIFY WITH
--      scripts/ci/check_delegation_tenant_conversion_readiness.py, NOT by
--      hand-running the SQL below and NOT by pod status. Hand-running the SQL
--      inverts its own answer under FORCE ROW LEVEL SECURITY; that is
--      DEFECT 3 of 0034 and the reason that script exists.
--   2. The migrate identity holds SELECT on tenant_registry_mirror. This is a
--      NEW precondition, and it is the one 0036 discovered the hard way. The
--      readiness script's PRIVILEGE leg checks it, per (role, relation) pair,
--      derived from this file's own bytes.
--   3. Write-time UUID stamping (OMN-16804) is LIVE. Note what this
--      precondition actually buys: it stops NEW slugs, and it is also what
--      PRODUCED the mixed column this file exists to convert. It makes the
--      conversion durable, not easier.
--   4. The OPERATOR un-gates. Steps 1-3 are independent and may proceed in
--      parallel; step 4 requires all of them.

DO $$
DECLARE
    v_current_type   TEXT;
    v_owner          NAME;
    v_forced         BOOLEAN := FALSE;
    v_assumed_owner  BOOLEAN := FALSE;
    v_convert        BOOLEAN := TRUE;
    v_mirror         REGCLASS;
    v_mirror_rows    BIGINT;
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
    -- Ownership, once, for every path that gets here. Carried over from 0036
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

    -- =====================================================================
    -- PHASE A -- THE ONLY tenant_registry_mirror READ IN THIS FILE, AND THE
    -- ONLY REASON THIS FILE EXISTS. IT RUNS AS THE MIGRATE IDENTITY, BEFORE
    -- THE ROLE SWITCH.
    --
    -- 0036 issued its mirror reads AFTER set_config('role', ...), as
    -- delegation_events' owner, and aborted on onex-dev with
    -- `permission denied for table tenant_registry_mirror` because that owner
    -- role is absent from the mirror's ACL. The mirror is owned by a
    -- DIFFERENT role. See the header.
    --
    -- to_regclass() is a catalog lookup and needs no privilege on the
    -- relation, so the existence branch is settled here, before anything else.
    -- Moving it up from 0036's position (after the debris DELETE) changes
    -- nothing: nothing between the two positions creates or drops a relation.
    -- =====================================================================
    v_mirror := to_regclass('tenant_registry_mirror');

    IF v_mirror IS NOT NULL THEN
        -- Refuse BY NAME, here, rather than opaquely 200 lines below in a
        -- guard whose subject is delegation_events. This is the check whose
        -- absence cost a red staging deploy on 2026-09-08.
        IF NOT has_table_privilege(
                   current_user, 'tenant_registry_mirror', 'SELECT') THEN
            RAISE EXCEPTION
                'OMN-15683: the migrate identity % holds no SELECT on '
                'tenant_registry_mirror, which is the relation this migration '
                'resolves every tenant identity against. It is owned by a '
                'DIFFERENT role than delegation_events (owner %), and this '
                'file deliberately reads it BEFORE switching to that owner -- '
                'reading it AFTER the switch is what aborted this file''s '
                'predecessor 0036 on onex-dev with `permission denied for '
                'table tenant_registry_mirror`. Refusing to start a '
                'conversion that cannot resolve. This file does NOT grant '
                'itself the privilege: the remedy is an operator act recorded '
                'on the ticket -- GRANT SELECT ON tenant_registry_mirror TO % '
                '-- issued by the mirror''s owner. Read-only triage: '
                'scripts/ci/check_delegation_tenant_conversion_readiness.py '
                '--dsn <dsn>, whose PRIVILEGE leg reports the failing '
                '(role, relation) pair. See OMN-15683.',
                current_user, v_owner, current_user;
        END IF;

        -- The snapshot. Two columns, because two columns are all the
        -- resolution uses. ON COMMIT DROP: it cannot outlive this
        -- transaction, so nothing this file creates survives it -- unlike a
        -- GRANT, which would. Taken inside the same transaction as the reads
        -- that consume it, so it is the same MVCC snapshot 0036's JOIN would
        -- have seen.
        CREATE TEMP TABLE omn15683_mirror_snapshot
        ON COMMIT DROP
        AS SELECT tenant_slug, tenant_uuid FROM tenant_registry_mirror;

        v_mirror_rows := (
            SELECT count(*)
            FROM omn15683_mirror_snapshot);

        -- Read one statement later as delegation_events' owner, which is not
        -- a member of the migrate identity and so holds nothing on a table
        -- the migrate identity just created. GRANT takes no PL/pgSQL variable
        -- for its grantee, and composing the role name into SQL text would be
        -- the dynamic SQL the OMN-15361 gate rejects -- so the grantee is
        -- PUBLIC. The blast radius of PUBLIC on this relation is one session:
        -- pg_temp is inaccessible to every other session by construction
        -- ("cannot access temporary tables of other sessions"), and this
        -- relation ceases to exist at COMMIT.
        GRANT SELECT ON omn15683_mirror_snapshot TO PUBLIC;

        RAISE NOTICE
            'OMN-15683: tenant_registry_mirror snapshotted as % -- % row(s) '
            '-- BEFORE the role switch; every resolution below joins the '
            'snapshot, never the mirror',
            current_user, v_mirror_rows;
    END IF;

    -- set_config('role', <name>, is_local => true) is exactly `SET LOCAL ROLE
    -- <name>` and takes the owner as a VALUE, so no SQL text is composed at
    -- runtime and the OMN-15361 gate's dynamic-SQL rejection does not apply.
    --
    -- EVERYTHING BELOW THIS LINE RUNS AS delegation_events' OWNER. No
    -- statement below reads tenant_registry_mirror. That is the invariant this
    -- file is named for, and the readiness script's PRIVILEGE leg derives it
    -- from these bytes rather than trusting this comment.
    PERFORM set_config('role', v_owner::text, true);
    v_assumed_owner := TRUE;

    IF v_convert THEN
        IF v_forced THEN
            ALTER TABLE delegation_events NO FORCE ROW LEVEL SECURITY;
        END IF;

        -- -----------------------------------------------------------------
        -- BLINDNESS RECONCILIATION -- runs BEFORE anything is enumerated or
        -- deleted, under this file's own SET ROLE + NO FORCE. Carried over
        -- from 0036 verbatim.
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
        -- correlation_id. Carried over from 0036 verbatim.
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
            -- DETERMINISM GUARD, carried over from 0036 and required BY the
            -- OR. Reads omn15683_mirror_snapshot, which is Phase A's copy of
            -- the mirror; the predicate and the message are 0036's.
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
                    JOIN omn15683_mirror_snapshot m
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
            -- path. Carried over from 0036 verbatim except for the relation
            -- it reads, which is Phase A's snapshot.
            --
            -- The message states only what was tested: the value, how many
            -- rows carry it, and WHICH lookup failed. The two dispositions are
            -- both named and neither is asserted, because this migration
            -- cannot tell them apart and must not pretend to.
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
                        FROM omn15683_mirror_snapshot m
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
            -- FROM against Phase A's snapshot. The predicate carries BOTH
            -- forms: a row whose tenant_id is already the canonical UUID
            -- matches on m.tenant_uuid::text and is assigned that same uuid --
            -- it passes through UNCHANGED, but by resolution against the
            -- registry rather than by an unchecked bypass, so it is proven and
            -- not trusted. Still resolution-by-JOIN, still no literal; the
            -- scratch column exists only inside this transaction and is
            -- dropped below.
            -- -------------------------------------------------------------
            ALTER TABLE delegation_events
                ADD COLUMN IF NOT EXISTS omn16930_resolved_tenant_uuid UUID;
            UPDATE delegation_events d
            SET omn16930_resolved_tenant_uuid = m.tenant_uuid
            FROM omn15683_mirror_snapshot m
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
