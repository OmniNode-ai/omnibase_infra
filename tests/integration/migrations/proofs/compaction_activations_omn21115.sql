-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
-- Apply the shipped migrations and exercise their runtime privileges in a
-- transaction. The hosted evidence runner supplies a disposable PG16 service.
BEGIN;
CREATE SCHEMA IF NOT EXISTS omninode_internal;
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'omninode_runtime') THEN
        CREATE ROLE omninode_runtime NOLOGIN;
    END IF;
END
$$;

\ir ../../../../docker/migrations/forward/nodes/node_context_install_confirm_trigger_effect/0000_create_compaction_activations.sql
\ir ../../../../docker/migrations/forward/nodes/node_context_install_confirm_trigger_effect/0001_grant_compaction_activations.sql

DO $$
DECLARE
    privilege TEXT;
BEGIN
    IF NOT has_schema_privilege('omninode_runtime', 'omninode_internal', 'USAGE') THEN
        RAISE EXCEPTION 'runtime schema USAGE is missing';
    END IF;
    FOREACH privilege IN ARRAY ARRAY['SELECT', 'INSERT', 'UPDATE'] LOOP
        IF NOT has_table_privilege('omninode_runtime', 'omninode_internal.compaction_activations', privilege) THEN
            RAISE EXCEPTION 'runtime privilege % is missing', privilege;
        END IF;
    END LOOP;
    FOREACH privilege IN ARRAY ARRAY['DELETE', 'TRUNCATE', 'REFERENCES', 'TRIGGER'] LOOP
        IF has_table_privilege('omninode_runtime', 'omninode_internal.compaction_activations', privilege) THEN
            RAISE EXCEPTION 'runtime privilege % exceeds the declaration', privilege;
        END IF;
    END LOOP;
END
$$;

SET LOCAL ROLE omninode_runtime;
INSERT INTO omninode_internal.compaction_activations
    (activation_id, session_id, installed_messages_hash, payload_sha256, published_at)
VALUES
    ('21115000-0000-0000-0000-000000000001', 'grant-proof', 'first', 'payload', CURRENT_TIMESTAMP);
-- The projection's ON CONFLICT path needs SELECT alongside INSERT and UPDATE.
INSERT INTO omninode_internal.compaction_activations
    (activation_id, session_id, installed_messages_hash, payload_sha256, published_at)
VALUES
    ('21115000-0000-0000-0000-000000000001', 'grant-proof', 'replayed', 'payload', CURRENT_TIMESTAMP)
ON CONFLICT (activation_id) DO UPDATE
SET installed_messages_hash = EXCLUDED.installed_messages_hash;
DO $$
BEGIN
    IF (SELECT installed_messages_hash FROM omninode_internal.compaction_activations
        WHERE activation_id = '21115000-0000-0000-0000-000000000001') IS DISTINCT FROM 'replayed' THEN
        RAISE EXCEPTION 'runtime activation upsert/read failed';
    END IF;
END
$$;
RESET ROLE;
ROLLBACK;
