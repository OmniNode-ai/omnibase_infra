-- OMN-18693: reassert the credentials writer's SIU grant after this node owns
-- public.tenant_inference_credentials.

SELECT 'tenant_projection_writer'::regrole;

DO $$
BEGIN
  IF to_regclass('public.tenant_inference_credentials') IS NULL THEN
    RAISE EXCEPTION 'required writer relation public.tenant_inference_credentials is absent';
  END IF;

  GRANT SELECT, INSERT, UPDATE ON public.tenant_inference_credentials TO tenant_projection_writer;

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
