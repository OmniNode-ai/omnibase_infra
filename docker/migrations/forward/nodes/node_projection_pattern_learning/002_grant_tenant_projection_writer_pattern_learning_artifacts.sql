-- OMN-18693: deliver the carrier's pattern-learning SIU grant after this
-- node's table lineage, not from an earlier cross-node migration.

SELECT 'tenant_projection_writer'::regrole;

DO $$
BEGIN
  IF to_regclass('public.pattern_learning_artifacts') IS NULL THEN
    RAISE EXCEPTION 'required carrier relation public.pattern_learning_artifacts is absent';
  END IF;

  GRANT SELECT, INSERT, UPDATE ON public.pattern_learning_artifacts TO tenant_projection_writer;

  IF NOT has_table_privilege('tenant_projection_writer', 'public.pattern_learning_artifacts', 'SELECT') THEN
    RAISE EXCEPTION 'missing required tenant_projection_writer privilege SELECT on public.pattern_learning_artifacts';
  END IF;
  IF NOT has_table_privilege('tenant_projection_writer', 'public.pattern_learning_artifacts', 'INSERT') THEN
    RAISE EXCEPTION 'missing required tenant_projection_writer privilege INSERT on public.pattern_learning_artifacts';
  END IF;
  IF NOT has_table_privilege('tenant_projection_writer', 'public.pattern_learning_artifacts', 'UPDATE') THEN
    RAISE EXCEPTION 'missing required tenant_projection_writer privilege UPDATE on public.pattern_learning_artifacts';
  END IF;
END
$$;
