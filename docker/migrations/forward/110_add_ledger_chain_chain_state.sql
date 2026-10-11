-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
-- =============================================================================
-- MIGRATION 110: Label ledger-chain completeness (OMN-17427)
-- =============================================================================
-- The in-process delegation path deliberately publishes only its terminal.
-- The writer labels it from the terminal producer and absent parent evidence;
-- a bus-routed chain missing a hop remains incomplete. This label is neither
-- a replay pass nor a chain fault. Every hop of a chain carries the same state.
-- The empty token means "written before this column existed"; legacy chains
-- retain the canary's existing completeness and replay checks.
-- The column grant lives in scripts/run-forward-migrations.sh, whose
-- LOGIN_ONLY_ROLE_GRANT_MAP reasserts the canary reader's column-scoped access.

ALTER TABLE public.ledger_chain
    ADD COLUMN IF NOT EXISTS chain_state TEXT NOT NULL DEFAULT '';
ALTER TABLE public.ledger_chain
    DROP CONSTRAINT IF EXISTS ck_ledger_chain_chain_state;
ALTER TABLE public.ledger_chain
    ADD CONSTRAINT ck_ledger_chain_chain_state
    CHECK (chain_state IN ('', 'complete', 'incomplete', 'in_process_terminal_only'));
COMMENT ON COLUMN public.ledger_chain.chain_state IS
    'OMN-17427: chain-level completeness label. complete means every declared hop '
    'was observed; incomplete means missing hops; in_process_terminal_only means '
    'only contract-declared in-process parentless terminals were observed, '
    'neither a pass nor a fault. Empty means written before this column existed.';
