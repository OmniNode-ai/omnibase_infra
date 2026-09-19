-- OMN-18693: deliver the carrier's dep-health SIU grant after this node owns
-- public.dep_health_findings. Sequence USAGE remains in the preceding 003.

SELECT 'tenant_projection_writer'::regrole;

DO $$
BEGIN
  IF to_regclass('public.dep_health_findings') IS NULL THEN
    RAISE EXCEPTION 'required carrier relation public.dep_health_findings is absent';
  END IF;

  GRANT SELECT, INSERT, UPDATE ON public.dep_health_findings TO tenant_projection_writer;

  IF NOT has_table_privilege('tenant_projection_writer', 'public.dep_health_findings', 'SELECT') THEN
    RAISE EXCEPTION 'missing required tenant_projection_writer privilege SELECT on public.dep_health_findings';
  END IF;
  IF NOT has_table_privilege('tenant_projection_writer', 'public.dep_health_findings', 'INSERT') THEN
    RAISE EXCEPTION 'missing required tenant_projection_writer privilege INSERT on public.dep_health_findings';
  END IF;
  IF NOT has_table_privilege('tenant_projection_writer', 'public.dep_health_findings', 'UPDATE') THEN
    RAISE EXCEPTION 'missing required tenant_projection_writer privilege UPDATE on public.dep_health_findings';
  END IF;
END
$$;
