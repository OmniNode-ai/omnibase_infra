-- OMN-19808: restore the invoker rights set by 088 after the bare view
-- replacements in 089 and 090 reset projection_delegation_savings's reloptions.
-- Target DB: omnidash_analytics (NODE_POSTGRES_DB)
-- Node: node_projection_savings
--
-- CREATE OR REPLACE VIEW preserves grants but resets omitted view options;
-- 089's claim that security_invoker survives replacement is incorrect.
-- Applied migration bytes are immutable, so the correction is forward-only.
-- Reassert both views: SET is idempotent, and IF EXISTS guards absent views.

ALTER VIEW IF EXISTS public.projection_delegation_savings SET (security_invoker = true);
ALTER VIEW IF EXISTS public.projection_delegation_savings_series SET (security_invoker = true);
