-- OMN-18693: warm dogfood volumes may have recorded 0004 before five public
-- projection ACLs took effect. Keep the repair append-only: historical
-- migration checksums are immutable.
--
-- The topology's tenant_projection_writer table grant set maps the current
-- physical locations to public. The carrier needs agent_routing_decisions,
-- dep_health_findings, and pattern_learning_artifacts; the standalone writers
-- need savings_estimates and tenant_inference_credentials. Their handlers use
-- SELECT/INSERT/UPDATE and never DELETE. The tenant routing-overlay write
-- already has that grant in 0004 and is intentionally not widened.

SELECT 'tenant_projection_writer'::regrole;

DO $$
BEGIN
  IF to_regclass('public.agent_routing_decisions') IS NULL THEN
    RAISE EXCEPTION 'required carrier relation public.agent_routing_decisions is absent';
  END IF;
  GRANT SELECT, INSERT, UPDATE ON public.agent_routing_decisions TO tenant_projection_writer;

  IF to_regclass('public.dep_health_findings') IS NULL THEN
    RAISE EXCEPTION 'required carrier relation public.dep_health_findings is absent';
  END IF;
  GRANT SELECT, INSERT, UPDATE ON public.dep_health_findings TO tenant_projection_writer;

  IF to_regclass('public.pattern_learning_artifacts') IS NULL THEN
    RAISE EXCEPTION 'required carrier relation public.pattern_learning_artifacts is absent';
  END IF;
  GRANT SELECT, INSERT, UPDATE ON public.pattern_learning_artifacts TO tenant_projection_writer;

  IF to_regclass('public.savings_estimates') IS NULL THEN
    RAISE EXCEPTION 'required writer relation public.savings_estimates is absent';
  END IF;
  GRANT SELECT, INSERT, UPDATE ON public.savings_estimates TO tenant_projection_writer;

  IF to_regclass('public.tenant_inference_credentials') IS NULL THEN
    RAISE EXCEPTION 'required writer relation public.tenant_inference_credentials is absent';
  END IF;
  GRANT SELECT, INSERT, UPDATE ON public.tenant_inference_credentials TO tenant_projection_writer;

  IF NOT has_table_privilege('tenant_projection_writer', 'public.agent_routing_decisions', 'SELECT') THEN
    RAISE EXCEPTION 'missing required tenant_projection_writer privilege SELECT on public.agent_routing_decisions';
  END IF;
  IF NOT has_table_privilege('tenant_projection_writer', 'public.agent_routing_decisions', 'INSERT') THEN
    RAISE EXCEPTION 'missing required tenant_projection_writer privilege INSERT on public.agent_routing_decisions';
  END IF;
  IF NOT has_table_privilege('tenant_projection_writer', 'public.agent_routing_decisions', 'UPDATE') THEN
    RAISE EXCEPTION 'missing required tenant_projection_writer privilege UPDATE on public.agent_routing_decisions';
  END IF;

  IF NOT has_table_privilege('tenant_projection_writer', 'public.dep_health_findings', 'SELECT') THEN
    RAISE EXCEPTION 'missing required tenant_projection_writer privilege SELECT on public.dep_health_findings';
  END IF;
  IF NOT has_table_privilege('tenant_projection_writer', 'public.dep_health_findings', 'INSERT') THEN
    RAISE EXCEPTION 'missing required tenant_projection_writer privilege INSERT on public.dep_health_findings';
  END IF;
  IF NOT has_table_privilege('tenant_projection_writer', 'public.dep_health_findings', 'UPDATE') THEN
    RAISE EXCEPTION 'missing required tenant_projection_writer privilege UPDATE on public.dep_health_findings';
  END IF;

  IF NOT has_table_privilege('tenant_projection_writer', 'public.pattern_learning_artifacts', 'SELECT') THEN
    RAISE EXCEPTION 'missing required tenant_projection_writer privilege SELECT on public.pattern_learning_artifacts';
  END IF;
  IF NOT has_table_privilege('tenant_projection_writer', 'public.pattern_learning_artifacts', 'INSERT') THEN
    RAISE EXCEPTION 'missing required tenant_projection_writer privilege INSERT on public.pattern_learning_artifacts';
  END IF;
  IF NOT has_table_privilege('tenant_projection_writer', 'public.pattern_learning_artifacts', 'UPDATE') THEN
    RAISE EXCEPTION 'missing required tenant_projection_writer privilege UPDATE on public.pattern_learning_artifacts';
  END IF;

  IF NOT has_table_privilege('tenant_projection_writer', 'public.savings_estimates', 'SELECT') THEN
    RAISE EXCEPTION 'missing required tenant_projection_writer privilege SELECT on public.savings_estimates';
  END IF;
  IF NOT has_table_privilege('tenant_projection_writer', 'public.savings_estimates', 'INSERT') THEN
    RAISE EXCEPTION 'missing required tenant_projection_writer privilege INSERT on public.savings_estimates';
  END IF;
  IF NOT has_table_privilege('tenant_projection_writer', 'public.savings_estimates', 'UPDATE') THEN
    RAISE EXCEPTION 'missing required tenant_projection_writer privilege UPDATE on public.savings_estimates';
  END IF;

  IF NOT has_table_privilege('tenant_projection_writer', 'public.tenant_inference_credentials', 'SELECT') THEN
    RAISE EXCEPTION 'missing required tenant_projection_writer privilege SELECT on public.tenant_inference_credentials';
  END IF;
  IF NOT has_table_privilege('tenant_projection_writer', 'public.tenant_inference_credentials', 'INSERT') THEN
    RAISE EXCEPTION 'missing required tenant_projection_writer privilege INSERT on public.tenant_inference_credentials';
  END IF;
  IF NOT has_table_privilege('tenant_projection_writer', 'public.tenant_inference_credentials', 'UPDATE') THEN
    RAISE EXCEPTION 'missing required tenant_projection_writer privilege UPDATE on public.tenant_inference_credentials';
  END IF;
END
$$;
