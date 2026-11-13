-- =============================================================================
-- MIGRATION: explicit role_omnibase_infra grants on the omnibase_infra state_io tables
-- =============================================================================
-- Ticket: OMN-24037.
--
-- WHAT IS WRONG
--   The runtime database account role_omnibase_infra is declared in
--   src/omnibase_infra/topology/instances/*.yaml
--   (databases.omnibase_infra.principals.role_omnibase_infra, TABLE grant
--   INSERT, SELECT, UPDATE on the public tables code_eval_trial_state,
--   delegation_workflow_state, floor_fill_state, infra_routing_decisions,
--   pr_landing_workflow_state, session_phase_state), but no migration ever
--   issued that grant. The tables are created by the postgres superuser, so
--   on a warm dev volume the runtime account holds nothing and the floor fill
--   state node cannot write floor_fill_state (0 rows).
--
-- WHAT THIS FILE DOES
--   It GRANTs the topology-declared privileges on the four of those tables that
--   carry a public ownership declaration (code_eval_trial_state,
--   floor_fill_state, pr_landing_workflow_state, session_phase_state).
--   delegation_workflow_state and infra_routing_decisions have none under
--   public, so the application database SQL gate refuses a target on them;
--   they are left out until their ownership is declared. It re-asserts
--   them through a post-condition that fails the migration unless every
--   privilege is actually held. It widens nothing beyond the declaration: no
--   DELETE, no wildcard.
--
-- EXECUTING ROLE
--   GRANT by a role that is not the owner, a member of the owner role, a
--   superuser, or a holder of the privilege WITH GRANT OPTION grants nothing
--   and only raises a WARNING. So the file does not trust the GRANT: the
--   post-condition below reads has_table_privilege and fails the migration
--   unless all three privileges are held for role_omnibase_infra itself. On a
--   lane where the role is absent, nothing is granted and a NOTICE is raised.
--
-- Idempotent: GRANT of privileges already held is a no-op, and so is the read.
-- Class: expand-only (GRANT only).

DO $grant$
DECLARE
    tbl text;
    priv text;
BEGIN
    IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'role_omnibase_infra') THEN
        GRANT SELECT, INSERT, UPDATE ON TABLE public.code_eval_trial_state, public.floor_fill_state, public.pr_landing_workflow_state, public.session_phase_state TO role_omnibase_infra;

        FOREACH tbl IN ARRAY ARRAY[
            'code_eval_trial_state',
            'floor_fill_state',
            'pr_landing_workflow_state',
            'session_phase_state'
        ] LOOP
            FOREACH priv IN ARRAY ARRAY['SELECT', 'INSERT', 'UPDATE'] LOOP
                IF NOT has_table_privilege('role_omnibase_infra', 'public.' || tbl, priv) THEN
                    RAISE EXCEPTION 'role_omnibase_infra lacks % on public.%', priv, tbl;
                END IF;
            END LOOP;
        END LOOP;
    ELSE
        RAISE NOTICE 'role_omnibase_infra is absent on this lane so nothing is granted';
    END IF;
END
$grant$;
