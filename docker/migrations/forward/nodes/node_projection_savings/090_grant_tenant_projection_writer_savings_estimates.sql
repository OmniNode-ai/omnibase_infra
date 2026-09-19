-- OMN-18693: reassert the savings writer's SIU grant after this node's table
-- and RLS migrations have completed.

SELECT 'tenant_projection_writer'::regrole;

DO $$
BEGIN
  IF to_regclass('public.savings_estimates') IS NULL THEN
    RAISE EXCEPTION 'required writer relation public.savings_estimates is absent';
  END IF;

  GRANT SELECT, INSERT, UPDATE ON public.savings_estimates TO tenant_projection_writer;

  IF NOT has_table_privilege('tenant_projection_writer', 'public.savings_estimates', 'SELECT') THEN
    RAISE EXCEPTION 'missing required tenant_projection_writer privilege SELECT on public.savings_estimates';
  END IF;
  IF NOT has_table_privilege('tenant_projection_writer', 'public.savings_estimates', 'INSERT') THEN
    RAISE EXCEPTION 'missing required tenant_projection_writer privilege INSERT on public.savings_estimates';
  END IF;
  IF NOT has_table_privilege('tenant_projection_writer', 'public.savings_estimates', 'UPDATE') THEN
    RAISE EXCEPTION 'missing required tenant_projection_writer privilege UPDATE on public.savings_estimates';
  END IF;
END
$$;
