-- OMN-18693: deliver the carrier's routing-decision SIU grant after this
-- node's table lineage. Node directories run in lexical order, so a central
-- inference-response grant would execute before this relation exists.

SELECT 'tenant_projection_writer'::regrole;

DO $$
BEGIN
  IF to_regclass('public.agent_routing_decisions') IS NULL THEN
    RAISE EXCEPTION 'required carrier relation public.agent_routing_decisions is absent';
  END IF;

  GRANT SELECT, INSERT, UPDATE ON public.agent_routing_decisions TO tenant_projection_writer;

  IF NOT has_table_privilege('tenant_projection_writer', 'public.agent_routing_decisions', 'SELECT') THEN
    RAISE EXCEPTION 'missing required tenant_projection_writer privilege SELECT on public.agent_routing_decisions';
  END IF;
  IF NOT has_table_privilege('tenant_projection_writer', 'public.agent_routing_decisions', 'INSERT') THEN
    RAISE EXCEPTION 'missing required tenant_projection_writer privilege INSERT on public.agent_routing_decisions';
  END IF;
  IF NOT has_table_privilege('tenant_projection_writer', 'public.agent_routing_decisions', 'UPDATE') THEN
    RAISE EXCEPTION 'missing required tenant_projection_writer privilege UPDATE on public.agent_routing_decisions';
  END IF;
END
$$;
