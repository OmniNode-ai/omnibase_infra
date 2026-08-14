-- Migration: 026_create_code_entities.sql
-- Canonical DDL for the AST code-intelligence store: code_entities + code_relationships.
--
-- Ticket: OMN-15276 (supersedes the false-Done OMN-5765 "reconcile competing
--         code_entities migrations (025 vs 025_create)", closed 2026-03-21 with
--         both conflicting files still on omniintelligence dev four months later).
-- Origin:  omniintelligence/deployment/database/migrations/025_code_entities.sql (OMN-5661)
--        + omniintelligence/deployment/database/migrations/026_create_code_relationships.sql (OMN-5709) -- REJECTED shape, see below
--        + omniintelligence/deployment/database/migrations/027_code_entity_enrichment_part2.sql (OMN-5676) -- folded in
--
-- WHY THIS FILE LIVES HERE (DDL ownership, OMN-15276 scope item 2)
-- ---------------------------------------------------------------
-- The .201 docker lanes apply THIS directory, not omniintelligence's:
--   docker/docker-compose.infra.yml  -> intelligence-migration service
--                                       MIGRATIONS_DIR: /migrations/intelligence
--                                       ../docker/migrations/intelligence:/migrations/intelligence:ro
--   scripts/run-intelligence-migrations.sh -> applies ${MIGRATIONS_DIR}/*.sql in sorted
--                                       order, tracking basenames in omniintelligence.schema_migrations
-- The same binding appears in docker/docker-compose.judge.yml and
-- docker/catalog/services/intelligence-migration.yaml.
--
-- The two conflicting 025_* files lived in omniintelligence/deployment/database/migrations/,
-- which no .201 lane reads. That is why they were never applied: omniintelligence.schema_migrations
-- held 27 identical rows on the stability-test and prod lanes, ending at
-- 025_fix_llm_delegation_call_log_date_index (2026-06-11T09:37:18Z), and
-- code_entities/code_relationships were absent from 20/20 databases across both lanes
-- (read-only probe, 2026-07-27T23:46Z). A fix landed in the omniintelligence tree would
-- have read green in CI and changed nothing on any lane.
--
-- WHICH SCHEMA WON, AND WHY
-- -------------------------
-- The OMN-5661 shape (025_code_entities.sql). It is the shape both live consumers
-- actually query, column-for-column:
--   * omnimarket RepositoryCodeEntityPostgres (omnimarket#1923) projects
--     id/entity_name/entity_type/qualified_name/source_repo/source_path/docstring/
--     signature/classification/llm_description (embedding batch) and
--     .../bases/methods/fields/decorators (enrichment batch); it predicates on
--     last_embedded_at < last_extracted_at and classification IS NULL, and writes
--     architectural_pattern/classification_confidence/enrichment_version/last_enriched_at.
--   * omniintelligence RepositoryCodeEntity (node_ast_extraction_compute) upserts on
--     ON CONFLICT (qualified_name, source_repo) and selects WHERE source_path = $1.
--
-- The rejected OMN-5709 shape (025_create_code_entities.sql) named the same concepts
-- differently -- name/file_path/line_start/line_end, bases and decorators as JSONB
-- rather than TEXT[] -- and carried no qualified_name, no signature, no llm_description
-- and none of the last_*_at freshness stamps. Every consumer statement above would have
-- raised UndefinedColumn against it. Because both files were CREATE TABLE IF NOT EXISTS,
-- whichever sorted first would have silently won and the other's columns would never
-- have appeared: applying both did not converge, it just picked a winner quietly.
--
-- code_relationships takes the OMN-5661 shape for the same reason: the producer's
-- INSERT writes evidence, inject_into_context, source_repo and updated_at, none of
-- which exist in the OMN-5709 026_create_code_relationships.sql variant.

-- Latest-state entity table. One row per entity per repo.
-- Upsert key: (qualified_name, source_repo)
CREATE TABLE IF NOT EXISTS code_entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,  -- class, protocol, model, function, import, constant
    qualified_name TEXT NOT NULL,  -- module.ClassName.method_name
    source_repo TEXT NOT NULL,
    source_path TEXT NOT NULL,
    line_number INT,
    bases TEXT[],  -- base classes
    methods JSONB,  -- [{name, args, return_type, decorators}]
    fields JSONB,  -- for models: [{name, type, default}]
    decorators TEXT[],
    docstring TEXT,
    signature TEXT,  -- function signature string
    file_hash TEXT NOT NULL,  -- SHA256 for change detection
    -- LLM enrichment fields (NULL until enriched)
    classification TEXT,
    llm_description TEXT,
    architectural_pattern TEXT,
    classification_confidence FLOAT,
    enrichment_version TEXT,
    -- Deterministic classification (OMN-5674, folded from 027) -- fast, no LLM
    deterministic_node_type TEXT,
    deterministic_confidence FLOAT,
    deterministic_alternatives JSONB,
    -- Quality scoring (OMN-5675, folded from 027) -- multi-dimensional
    quality_score FLOAT,
    quality_dimensions JSONB,  -- {"complexity": 0.7, "maintainability": 0.8, ...}
    -- Config-aware idempotency metadata (folded from 027). Operational state kept
    -- separate from domain data; RepositoryCodeEntity merges into it with `||`,
    -- so it must default to '{}' and never be NULL.
    enrichment_metadata JSONB NOT NULL DEFAULT '{}',
    -- Multi-language support (OMN-5679, folded from 027)
    source_language TEXT DEFAULT 'python',
    -- Freshness timestamps for derived-store coordination
    last_extracted_at TIMESTAMPTZ DEFAULT NOW(),
    last_enriched_at TIMESTAMPTZ,
    last_embedded_at TIMESTAMPTZ,
    last_graph_synced_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(qualified_name, source_repo)
);

-- Latest-state relationship table.
-- Upsert key: (source_entity_id, target_entity_id, relationship_type)
CREATE TABLE IF NOT EXISTS code_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_entity_id UUID REFERENCES code_entities(id) ON DELETE CASCADE,
    target_entity_id UUID REFERENCES code_entities(id) ON DELETE CASCADE,
    relationship_type TEXT NOT NULL,
    trust_tier TEXT NOT NULL DEFAULT 'strong',
    confidence FLOAT DEFAULT 1.0,
    evidence TEXT[],
    inject_into_context BOOLEAN DEFAULT true,
    source_repo TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(source_entity_id, target_entity_id, relationship_type)
);

CREATE INDEX IF NOT EXISTS idx_code_entities_repo ON code_entities(source_repo);
CREATE INDEX IF NOT EXISTS idx_code_entities_type ON code_entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_code_entities_qualified ON code_entities(qualified_name);
CREATE INDEX IF NOT EXISTS idx_code_entities_classification ON code_entities(classification);
CREATE INDEX IF NOT EXISTS idx_code_entities_file_path ON code_entities(source_path);
CREATE INDEX IF NOT EXISTS idx_code_entities_det_node_type ON code_entities(deterministic_node_type);
CREATE INDEX IF NOT EXISTS idx_code_entities_quality ON code_entities(quality_score);
CREATE INDEX IF NOT EXISTS idx_code_entities_language ON code_entities(source_language);
CREATE INDEX IF NOT EXISTS idx_code_relationships_source ON code_relationships(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_code_relationships_target ON code_relationships(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_code_relationships_type ON code_relationships(relationship_type);
CREATE INDEX IF NOT EXISTS idx_code_relationships_injectable ON code_relationships(inject_into_context) WHERE inject_into_context = true;
