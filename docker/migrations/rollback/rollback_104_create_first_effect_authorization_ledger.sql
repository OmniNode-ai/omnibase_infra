-- SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
--
-- Rollback: 104_create_first_effect_authorization_ledger
--
-- Operator-run only. RESTRICT is deliberate: do not erase the observation
-- recording scaffold when a downstream object depends on it. The forward
-- migration creates no runtime privilege, effect permission, or canonical
-- outbox intent.

DROP TABLE IF EXISTS public.first_effect_authorization_ledger RESTRICT;
DROP FUNCTION IF EXISTS public.enforce_first_effect_authorization_transition();
