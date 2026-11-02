-- OMN-21933: the tenant projection writer needs table grants on skill_executions,
-- which node_projection_skill_lifecycle writes. Flat migration 103 provisions the
-- cluster role before node migrations run. skill_executions has no BIGSERIAL
-- column, so no sequence grant is needed.
DO $require_tenant_projection_writer$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles
        WHERE rolname = 'tenant_projection_writer'
    ) THEN
        RAISE EXCEPTION
            'tenant_projection_writer role missing; apply flat migration 103 before node migrations';
    END IF;
END
$require_tenant_projection_writer$;

GRANT USAGE ON SCHEMA public TO tenant_projection_writer;
-- UPDATE is granted although the node only inserts (ON CONFLICT DO NOTHING): the
-- contract declares the table for write, and the topology derives SELECT, INSERT,
-- UPDATE from that. A narrower grant would leave the derived UPDATE undelivered.
GRANT SELECT, INSERT, UPDATE
    ON public.skill_executions
    TO tenant_projection_writer;

DO $assert_tenant_projection_writer_privileges$
DECLARE
    privilege text;
BEGIN
    FOREACH privilege IN ARRAY ARRAY['SELECT', 'INSERT', 'UPDATE'] LOOP
        IF NOT has_table_privilege('tenant_projection_writer', 'public.skill_executions', privilege) THEN
            RAISE EXCEPTION
                'missing required tenant_projection_writer privilege % on public.skill_executions',
                privilege;
        END IF;
    END LOOP;
END
$assert_tenant_projection_writer_privileges$;
