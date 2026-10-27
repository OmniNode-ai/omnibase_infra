-- SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
CREATE TABLE IF NOT EXISTS omninode_internal.compaction_activations (
    activation_id UUID PRIMARY KEY,
    session_id TEXT NOT NULL,
    installed_messages_hash TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    published_at TIMESTAMPTZ NOT NULL
);
-- Shape reconciliation: guarded ADDs precede the index, as in the template.
ALTER TABLE omninode_internal.compaction_activations
    ADD COLUMN IF NOT EXISTS activation_id UUID;
ALTER TABLE omninode_internal.compaction_activations
    ADD COLUMN IF NOT EXISTS session_id TEXT;
ALTER TABLE omninode_internal.compaction_activations
    ADD COLUMN IF NOT EXISTS installed_messages_hash TEXT;
ALTER TABLE omninode_internal.compaction_activations
    ADD COLUMN IF NOT EXISTS payload_sha256 TEXT;
ALTER TABLE omninode_internal.compaction_activations
    ADD COLUMN IF NOT EXISTS published_at TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS compaction_activations_session_idx
    ON omninode_internal.compaction_activations (session_id, published_at);
