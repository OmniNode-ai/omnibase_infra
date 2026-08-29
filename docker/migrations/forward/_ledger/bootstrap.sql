-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
-- SPDX-License-Identifier: MIT
--
-- OMN-15413: deterministic, in-place application migration ledger upgrade.
--
-- This is a runner bootstrap, not a numbered migration: the legacy
-- filename-only schema cannot execute the numbered migration set until its
-- tracking relation has first been made readable.  The runner supplies
-- a validated temporary manifest table in the same psql session.
--
-- The selected relation is the existing checksum-capable ledger.  In the
-- application database that is public.node_schema_migrations.  The separate
-- service-owned infra ledger is out of scope and is never selected here.  The
-- application relation is moved (same OID, rows and owner) into
-- platform_catalog and extended in place.  The filename-only ledger is an
-- import source, never the selected canonical relation.
--
-- OMN-15695 (operator ruling 2026-08-04): the four-column
-- public.schema_migrations(migration_id, applied_at, checksum, source_set)
-- relation is ambiguous by column signature alone — it is the service-owned
-- ledger in the service database and the predecessor NODE ledger in the
-- application database.  It is partitioned on row content: source_set 'node'
-- rows with a node:<node>:<file>.sql identity are adopted as an import source
-- (source preserved, never moved), source_set 'docker' rows are ignored as
-- service-owned, and anything else aborts the transaction.  A relation with no
-- adoptable row is still refused outright.

\set ON_ERROR_STOP on

BEGIN;

CREATE SCHEMA IF NOT EXISTS platform_catalog;

DO $ledger_upgrade$
DECLARE
  public_ledger REGCLASS := to_regclass('public.schema_migrations');
  node_ledger REGCLASS := to_regclass('public.node_schema_migrations');
  canonical_ledger REGCLASS := to_regclass('platform_catalog.schema_migrations');
  column_count INTEGER;
  origin_shape TEXT := 'canonical';
  origin_source TEXT := 'platform_catalog.schema_migrations';
  public_shape TEXT := 'absent';
  -- OMN-15695: row-content sub-classification of the migration_id shape.
  -- 'none' when the shape is not migration_id, 'adopt' when the relation
  -- carries the historical runner's node rows, 'service' when it carries only
  -- the separate service-owned set.
  migration_id_disposition TEXT := 'none';
  adoptable_row_count INTEGER := 0;
  unrecognized_row_count INTEGER := 0;
  primary_key_name TEXT;
  primary_key_columns TEXT[];
BEGIN
  IF public_ledger IS NOT NULL THEN
    SELECT count(*) INTO column_count
    FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'schema_migrations';

    IF column_count = 2
       AND EXISTS (
         SELECT 1 FROM information_schema.columns
         WHERE table_schema = 'public' AND table_name = 'schema_migrations'
           AND column_name = 'filename'
       )
       AND EXISTS (
         SELECT 1 FROM information_schema.columns
         WHERE table_schema = 'public' AND table_name = 'schema_migrations'
           AND column_name = 'applied_at'
       )
       AND NOT EXISTS (
         SELECT 1 FROM information_schema.columns
         WHERE table_schema = 'public' AND table_name = 'schema_migrations'
           AND (
             (column_name = 'filename'
               AND (udt_name <> 'text' OR is_nullable <> 'NO'))
             OR (column_name = 'applied_at'
               AND (udt_name <> 'timestamptz' OR is_nullable <> 'NO'))
           )
       ) THEN
      public_shape := 'filename';
    ELSIF column_count = 4
       AND NOT EXISTS (
         SELECT 1 FROM information_schema.columns
         WHERE table_schema = 'public' AND table_name = 'schema_migrations'
           AND column_name NOT IN ('migration_id', 'applied_at', 'checksum', 'source_set')
       )
       AND EXISTS (
         SELECT 1 FROM information_schema.columns
         WHERE table_schema = 'public' AND table_name = 'schema_migrations'
           AND column_name = 'migration_id'
       )
       AND EXISTS (
         SELECT 1 FROM information_schema.columns
         WHERE table_schema = 'public' AND table_name = 'schema_migrations'
           AND column_name = 'source_set'
       )
       AND NOT EXISTS (
         SELECT 1 FROM information_schema.columns
         WHERE table_schema = 'public' AND table_name = 'schema_migrations'
           AND (
             (column_name IN ('migration_id', 'checksum', 'source_set')
               AND (udt_name <> 'text' OR is_nullable <> 'NO'))
             OR (column_name = 'applied_at'
               AND (udt_name <> 'timestamptz' OR is_nullable <> 'NO'))
           )
       ) THEN
      public_shape := 'migration_id';

      -- OMN-15695: the column signature alone cannot tell the service-owned
      -- ledger apart from the application database's predecessor NODE ledger.
      -- The pre-OMN-15413 runner wrote both shapes with these four columns:
      -- service rows as ('docker/<file>', ..., 'docker') and node rows as
      -- ('node:<node>:<file>.sql', ..., 'node').  Partition on row content and
      -- refuse anything that is neither.
      SELECT
        count(*) FILTER (
          WHERE source_set = 'node'
            AND migration_id ~
              '^node:[A-Za-z0-9_][A-Za-z0-9_.-]*:[A-Za-z0-9_][A-Za-z0-9_.-]*[.]sql$'
        ),
        count(*) FILTER (
          WHERE NOT (
            source_set = 'node'
            AND migration_id ~
              '^node:[A-Za-z0-9_][A-Za-z0-9_.-]*:[A-Za-z0-9_][A-Za-z0-9_.-]*[.]sql$'
          )
          AND NOT (source_set = 'docker' AND migration_id ~ '^docker/')
        )
      INTO adoptable_row_count, unrecognized_row_count
      FROM public.schema_migrations;

      IF unrecognized_row_count > 0 THEN
        RAISE EXCEPTION
          'unknown migration ledger shape: public.schema_migrations contains % unrecognized migration_id rows',
          unrecognized_row_count;
      END IF;
      IF adoptable_row_count > 0 THEN
        migration_id_disposition := 'adopt';
      ELSE
        migration_id_disposition := 'service';
      END IF;
    ELSIF column_count = 3
       AND NOT EXISTS (
         SELECT 1 FROM information_schema.columns
         WHERE table_schema = 'public' AND table_name = 'schema_migrations'
           AND column_name NOT IN ('version', 'applied_at', 'checksum')
       )
       AND EXISTS (
         SELECT 1 FROM information_schema.columns
         WHERE table_schema = 'public' AND table_name = 'schema_migrations'
           AND column_name = 'version'
       )
       AND NOT EXISTS (
         SELECT 1 FROM information_schema.columns
         WHERE table_schema = 'public' AND table_name = 'schema_migrations'
           AND (
             (column_name = 'version'
               AND (udt_name <> 'text' OR is_nullable <> 'NO'))
             OR (column_name = 'checksum' AND udt_name <> 'text')
             OR (column_name = 'applied_at'
               AND (udt_name <> 'timestamptz' OR is_nullable <> 'NO'))
           )
       ) THEN
      public_shape := 'version';
    ELSE
      RAISE EXCEPTION
        'unknown migration ledger shape: public.schema_migrations has % columns',
        column_count;
    END IF;
  END IF;

  IF canonical_ledger IS NOT NULL AND node_ledger IS NOT NULL THEN
    RAISE EXCEPTION
      'double migration declaration: both public.node_schema_migrations and platform_catalog.schema_migrations exist';
  END IF;
  -- OMN-15695: an adopted migration_id node ledger is deliberately preserved
  -- beside the canonical ledger (the same non-destructive contract the
  -- filename-only source has), so it is not a double declaration.  Every other
  -- checksum-capable public relation beside the canonical ledger still is.
  IF canonical_ledger IS NOT NULL
     AND (public_shape = 'version'
          OR (public_shape = 'migration_id'
              AND migration_id_disposition <> 'adopt')) THEN
    RAISE EXCEPTION
      'double migration declaration: checksum-capable public.schema_migrations exists beside the canonical ledger';
  END IF;

  -- A migration_id relation that carries no adoptable node row is the
  -- service-owned ledger.  It is never selected for the application database.
  IF canonical_ledger IS NULL
     AND public_shape = 'migration_id'
     AND migration_id_disposition <> 'adopt' THEN
    RAISE EXCEPTION
      'unknown migration stream: service-owned migration_id ledger cannot be selected for the application database';
  END IF;

  IF canonical_ledger IS NULL AND node_ledger IS NOT NULL THEN
    SELECT count(*) INTO column_count
    FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'node_schema_migrations';
    IF column_count <> 3 OR EXISTS (
      SELECT 1 FROM information_schema.columns
      WHERE table_schema = 'public' AND table_name = 'node_schema_migrations'
        AND (
          column_name NOT IN ('version', 'applied_at', 'checksum')
          OR (column_name IN ('version', 'checksum')
            AND (udt_name <> 'text' OR is_nullable <> 'NO'))
          OR (column_name = 'applied_at'
            AND (udt_name <> 'timestamptz' OR is_nullable <> 'NO'))
        )
    ) THEN
      RAISE EXCEPTION
        'unknown migration ledger shape: public.node_schema_migrations';
    END IF;

    ALTER TABLE public.node_schema_migrations SET SCHEMA platform_catalog;
    ALTER TABLE platform_catalog.node_schema_migrations
      RENAME TO schema_migrations;
    canonical_ledger := to_regclass('platform_catalog.schema_migrations');
    origin_shape := 'node';
    origin_source := 'public.node_schema_migrations';
  ELSIF canonical_ledger IS NULL
        AND public_shape = 'version' THEN
    ALTER TABLE public.schema_migrations SET SCHEMA platform_catalog;
    canonical_ledger := to_regclass('platform_catalog.schema_migrations');
    origin_shape := 'node';
    origin_source := 'public.schema_migrations';
    public_ledger := NULL;
    public_shape := 'absent';
  ELSIF canonical_ledger IS NULL THEN
    -- Fresh databases, filename-only databases, and the adopted migration_id
    -- node ledger have no checksum-capable candidate that can be moved: the
    -- migration_id relation cannot become the canonical shape without dropping
    -- source_set.  Create the selected canonical shape; that history is
    -- imported below without renaming or rewriting the source relation.
    CREATE TABLE platform_catalog.schema_migrations (
      migration_stream TEXT NOT NULL,
      owner TEXT NOT NULL,
      domain TEXT NOT NULL,
      version TEXT NOT NULL,
      checksum TEXT NOT NULL,
      checksum_kind TEXT NOT NULL,
      applied_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      provenance TEXT NOT NULL,
      PRIMARY KEY (migration_stream, domain, version)
    );
    canonical_ledger := to_regclass('platform_catalog.schema_migrations');
    origin_shape := 'fresh';
  END IF;

  ALTER TABLE platform_catalog.schema_migrations
    ADD COLUMN IF NOT EXISTS migration_stream TEXT,
    ADD COLUMN IF NOT EXISTS owner TEXT,
    ADD COLUMN IF NOT EXISTS domain TEXT,
    ADD COLUMN IF NOT EXISTS checksum TEXT,
    ADD COLUMN IF NOT EXISTS checksum_kind TEXT,
    ADD COLUMN IF NOT EXISTS provenance TEXT;

  IF origin_shape = 'node' THEN
    IF EXISTS (
      SELECT 1
      FROM platform_catalog.schema_migrations ledger
      LEFT JOIN onex_application_migration_manifest manifest
        ON manifest.version = ledger.version
      WHERE manifest.version IS NULL
    ) THEN
      RAISE EXCEPTION
        'unknown migration stream/domain: historical node version has no checked-in declaration';
    END IF;
    IF EXISTS (
      SELECT 1
      FROM platform_catalog.schema_migrations ledger
      JOIN onex_application_migration_manifest manifest
        ON manifest.version = ledger.version
      WHERE ledger.checksum IS NULL
         OR ledger.checksum !~ '^[0-9a-f]{64}$'
         OR ledger.checksum <> manifest.checksum
    ) THEN
      RAISE EXCEPTION
        'conflicting migration checksum in checksum-capable node history';
    END IF;

    UPDATE platform_catalog.schema_migrations ledger
    SET migration_stream = manifest.migration_stream,
        owner = manifest.owner,
        domain = manifest.domain,
        provenance = format(
          'legacy:%s:%s:version:%s:raw-checksum=%s',
          current_database(), origin_source, ledger.version,
          coalesce(ledger.checksum, '<NULL>')
        ),
        checksum_kind = 'content_sha256'
    FROM onex_application_migration_manifest manifest
    WHERE manifest.version = ledger.version;
  END IF;

  IF EXISTS (
    SELECT 1
    FROM platform_catalog.schema_migrations
    WHERE migration_stream IS NULL OR migration_stream = ''
       OR owner IS NULL OR owner = ''
       OR domain IS NULL OR domain = ''
       OR version IS NULL OR version = ''
       OR checksum IS NULL OR checksum !~ '^[0-9a-f]{64}$'
       OR checksum_kind NOT IN ('content_sha256', 'legacy_attestation')
       OR provenance IS NULL OR provenance = ''
  ) THEN
    RAISE EXCEPTION
      'canonical migration ledger contains null, empty, or malformed metadata';
  END IF;

  IF EXISTS (
    SELECT 1
    FROM platform_catalog.schema_migrations ledger
    LEFT JOIN onex_application_migration_manifest manifest
      ON manifest.migration_stream = ledger.migration_stream
     AND manifest.owner = ledger.owner
     AND manifest.domain = ledger.domain
     AND manifest.version = ledger.version
    LEFT JOIN onex_legacy_node_migration_declarations legacy
      ON legacy.migration_stream = ledger.migration_stream
     AND legacy.owner = ledger.owner
     AND legacy.domain = ledger.domain
     AND legacy.version = ledger.version
    WHERE manifest.version IS NULL AND legacy.version IS NULL
      AND NOT (
        ledger.migration_stream = 'legacy:filename-only'
        AND ledger.owner = 'legacy:filename-only'
        AND ledger.domain = 'legacy_unclassified'
      )
      AND NOT (
        ledger.migration_stream = 'omninode-cloud'
        AND ledger.owner = 'service:onex_api'
        AND ledger.domain = 'legacy_unclassified'
      )
  ) THEN
    RAISE EXCEPTION 'unknown migration stream/domain declaration';
  END IF;

  IF EXISTS (
    SELECT 1
    FROM platform_catalog.schema_migrations ledger
    JOIN onex_application_migration_manifest manifest
      ON manifest.migration_stream = ledger.migration_stream
     AND manifest.owner = ledger.owner
     AND manifest.domain = ledger.domain
     AND manifest.version = ledger.version
    WHERE ledger.checksum_kind = 'content_sha256'
      AND ledger.checksum <> manifest.checksum
  ) THEN
    RAISE EXCEPTION 'conflicting migration checksum in canonical node history';
  END IF;
  IF EXISTS (
    SELECT 1
    FROM platform_catalog.schema_migrations ledger
    JOIN onex_application_migration_manifest manifest
      ON manifest.migration_stream = ledger.migration_stream
     AND manifest.owner = ledger.owner
     AND manifest.domain = ledger.domain
     AND manifest.version = ledger.version
    WHERE ledger.checksum_kind <> 'content_sha256'
  ) THEN
    RAISE EXCEPTION
      'checksum-capable node history cannot be downgraded to legacy attestation';
  END IF;

  SELECT c.conname,
         array_agg(a.attname ORDER BY key_position.ordinality)
    INTO primary_key_name, primary_key_columns
  FROM pg_constraint c
  CROSS JOIN LATERAL unnest(c.conkey) WITH ORDINALITY AS key_position(attnum, ordinality)
  JOIN pg_attribute a
    ON a.attrelid = c.conrelid AND a.attnum = key_position.attnum
  WHERE c.conrelid = canonical_ledger AND c.contype = 'p'
  GROUP BY c.conname;

  IF primary_key_name IS NOT NULL
     AND primary_key_columns <> ARRAY['migration_stream', 'domain', 'version']::TEXT[] THEN
    EXECUTE format(
      'ALTER TABLE platform_catalog.schema_migrations DROP CONSTRAINT %I',
      primary_key_name
    );
    primary_key_name := NULL;
  END IF;

  IF primary_key_name IS NULL THEN
    IF EXISTS (
      SELECT 1
      FROM platform_catalog.schema_migrations
      GROUP BY migration_stream, domain, version
      HAVING count(*) > 1
    ) THEN
      RAISE EXCEPTION 'duplicate migration version in canonical ledger';
    END IF;
    ALTER TABLE platform_catalog.schema_migrations
      ADD PRIMARY KEY (migration_stream, domain, version);
  END IF;

  ALTER TABLE platform_catalog.schema_migrations
    ALTER COLUMN migration_stream SET NOT NULL,
    ALTER COLUMN owner SET NOT NULL,
    ALTER COLUMN domain SET NOT NULL,
    ALTER COLUMN version SET NOT NULL,
    ALTER COLUMN checksum SET NOT NULL,
    ALTER COLUMN checksum_kind SET NOT NULL,
    ALTER COLUMN applied_at SET NOT NULL,
    ALTER COLUMN provenance SET NOT NULL;

  SELECT count(*) INTO column_count
  FROM information_schema.columns
  WHERE table_schema = 'platform_catalog'
    AND table_name = 'schema_migrations';
  IF column_count <> 8 OR EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = 'platform_catalog'
      AND table_name = 'schema_migrations'
      AND (
        column_name NOT IN (
          'migration_stream', 'owner', 'domain', 'version', 'checksum',
          'checksum_kind', 'applied_at', 'provenance'
        )
        OR (column_name <> 'applied_at'
          AND (udt_name <> 'text' OR is_nullable <> 'NO'))
        OR (column_name = 'applied_at'
          AND (udt_name <> 'timestamptz' OR is_nullable <> 'NO'))
      )
  ) THEN
    RAISE EXCEPTION
      'unknown migration ledger shape: platform_catalog.schema_migrations';
  END IF;

  -- Recreate the named checks from the canonical predicates on every bootstrap.
  -- Accepting a constraint merely because its name matches would let a stale or
  -- unrelated expression disable the fail-closed boundary.
  ALTER TABLE platform_catalog.schema_migrations
    DROP CONSTRAINT IF EXISTS schema_migrations_stream_domain_check,
    DROP CONSTRAINT IF EXISTS schema_migrations_checksum_check,
    DROP CONSTRAINT IF EXISTS schema_migrations_checksum_kind_check;
  ALTER TABLE platform_catalog.schema_migrations
    ADD CONSTRAINT schema_migrations_stream_domain_check CHECK (
      (migration_stream ~ '^node:[A-Za-z0-9_][A-Za-z0-9_.-]*$'
        AND owner = migration_stream
        AND domain IN ('omninode_internal', 'tenant'))
      OR (migration_stream = 'legacy:filename-only'
        AND owner = 'legacy:filename-only'
        AND domain = 'legacy_unclassified')
      OR (migration_stream = 'omninode-cloud'
        AND owner = 'service:onex_api'
        AND domain = 'legacy_unclassified')
    ),
    ADD CONSTRAINT schema_migrations_checksum_check
      CHECK (checksum ~ '^[0-9a-f]{64}$'),
    ADD CONSTRAINT schema_migrations_checksum_kind_check
      CHECK (checksum_kind IN ('content_sha256', 'legacy_attestation'));
END
$ledger_upgrade$;

COMMENT ON TABLE platform_catalog.schema_migrations IS
  'Canonical application migration ledger selected by OMN-15413; evolved in place from the existing checksum-capable ledger.';
COMMENT ON COLUMN platform_catalog.schema_migrations.migration_stream IS
  'Contract-declared producer stream; legacy:filename-only is quarantined evidence.';
COMMENT ON COLUMN platform_catalog.schema_migrations.owner IS
  'Contract-declared migration producer owner; never inferred from the database role.';
COMMENT ON COLUMN platform_catalog.schema_migrations.domain IS
  'Declared target schema domain; legacy_unclassified is non-executable quarantine evidence.';
COMMENT ON COLUMN platform_catalog.schema_migrations.version IS
  'Migration identity within migration_stream; historical identity is preserved.';
COMMENT ON COLUMN platform_catalog.schema_migrations.checksum IS
  'Lowercase SHA-256: file bytes for content_sha256, or deterministic source-record attestation for legacy_attestation.';
COMMENT ON COLUMN platform_catalog.schema_migrations.provenance IS
  'Deterministic source record or checked-in file that produced this ledger row.';

-- Import filename-only omnidash history without renaming, updating, or
-- deleting the source rows.  The SHA-256 is explicitly a source-record
-- attestation because this ledger never captured migration file bytes.
DO $filename_import$
DECLARE
  source_row RECORD;
  existing_row RECORD;
  imported_checksum TEXT;
  imported_provenance TEXT;
  source_column_count INTEGER;
BEGIN
  IF to_regclass('public.schema_migrations') IS NULL THEN
    RETURN;
  END IF;

  SELECT count(*) INTO source_column_count
  FROM information_schema.columns
  WHERE table_schema = 'public' AND table_name = 'schema_migrations';

  -- OMN-15695: the migration_id-shaped predecessor node ledger is adopted by
  -- $migration_id_import$ below.  It is a deliberately preserved adoption
  -- source, not a filename-only source and not a double declaration.
  IF source_column_count = 4 AND NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'schema_migrations'
      AND column_name NOT IN
        ('migration_id', 'applied_at', 'checksum', 'source_set')
  ) THEN
    RETURN;
  END IF;

  IF source_column_count <> 2 OR EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'schema_migrations'
      AND column_name NOT IN ('filename', 'applied_at')
  ) THEN
    RAISE EXCEPTION
      'double migration declaration: checksum-capable public.schema_migrations remains beside the canonical ledger';
  END IF;

  FOR source_row IN
    SELECT filename, applied_at
    FROM public.schema_migrations
    ORDER BY filename
  LOOP
    IF source_row.filename IS NULL OR source_row.filename = '' THEN
      RAISE EXCEPTION 'duplicate migration version: empty filename';
    END IF;
    imported_checksum := encode(sha256(convert_to(
      'legacy:filename-only|legacy_unclassified|' || source_row.filename || '|filename-only',
      'UTF8'
    )), 'hex');
    imported_provenance := format(
      'legacy:%s:public.schema_migrations:filename:%s',
      current_database(), source_row.filename
    );

    SELECT * INTO existing_row
    FROM platform_catalog.schema_migrations
    WHERE migration_stream = 'legacy:filename-only'
      AND domain = 'legacy_unclassified'
      AND version = source_row.filename;
    IF FOUND THEN
      IF existing_row.checksum <> imported_checksum THEN
        RAISE EXCEPTION 'conflicting migration checksum for version %', source_row.filename;
      ELSIF existing_row.owner <> 'legacy:filename-only'
         OR existing_row.domain <> 'legacy_unclassified'
         OR existing_row.checksum_kind <> 'legacy_attestation'
         OR existing_row.applied_at IS DISTINCT FROM source_row.applied_at
         OR existing_row.provenance <> imported_provenance THEN
        RAISE EXCEPTION 'double migration declaration for version %', source_row.filename;
      END IF;
    ELSE
      INSERT INTO platform_catalog.schema_migrations (
        migration_stream, owner, domain, version, checksum, checksum_kind,
        applied_at, provenance
      ) VALUES (
        'legacy:filename-only', 'legacy:filename-only',
        'legacy_unclassified', source_row.filename,
        imported_checksum, 'legacy_attestation', source_row.applied_at,
        imported_provenance
      );
    END IF;
  END LOOP;
END
$filename_import$;

-- OMN-15695: adopt the application database's predecessor node ledger.
--
-- The pre-OMN-15413 runner created
-- public.schema_migrations(migration_id, applied_at, checksum, source_set)
-- in the NODE database and recorded each applied node migration as
-- ('node:<node>:<file>.sql', now(), 'applied-by-runner', 'node').  That
-- relation is the legitimate applied-history of this database, so the
-- migration_id arm's blanket refusal was a false negative for it.
--
-- Operator ruling 2026-08-04 (ADOPT/CONVERT): preserve the applied history,
-- never re-apply an already-applied migration, never delete or rewrite the
-- source rows.  The source relation is left byte-for-byte intact, exactly as
-- the filename-only import leaves its source.
--
-- Evidence class, stated plainly: 'applied-by-runner' is a sentinel, not a
-- hash — this ledger never captured file bytes.  Adoption writes the checked-in
-- manifest checksum under checksum_kind 'content_sha256', which ASSERTS that
-- the bytes applied historically equal today's checked-in bytes.  That
-- assertion is the operator ruling made mechanical; it is not derivable from
-- the database.  It is kept auditable and non-forgeable three ways: only the
-- exact 'applied-by-runner' literal is adoptable, a 64-hex source checksum that
-- disagrees with the manifest is still fatal, and provenance permanently
-- records the raw source checksum under an 'adopted:' prefix so an adopted row
-- can never be mistaken for a runner-verified 'file:nodes/...' row.
--
-- Service-owned rows (source_set 'docker') belong to the separate service
-- ledger and are ignored here.  Any row that is neither has already aborted the
-- transaction in the selection block above.
DO $migration_id_import$
DECLARE
  source_row RECORD;
  manifest_row RECORD;
  legacy_row RECORD;
  existing_row RECORD;
  -- OMN-15857.  A plpgsql RECORD keeps whatever the last SELECT INTO put in it,
  -- so this is reset at the top of every iteration: a verified adoption for one
  -- version must never leak its ticket/receipt into the provenance of the next.
  adoption_row RECORD;
  resolved_stream TEXT;
  resolved_owner TEXT;
  resolved_domain TEXT;
  imported_checksum TEXT;
  imported_checksum_kind TEXT;
  imported_provenance TEXT;
  source_column_count INTEGER;
BEGIN
  IF to_regclass('public.schema_migrations') IS NULL THEN
    RETURN;
  END IF;

  SELECT count(*) INTO source_column_count
  FROM information_schema.columns
  WHERE table_schema = 'public' AND table_name = 'schema_migrations';
  IF source_column_count <> 4 OR EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'schema_migrations'
      AND column_name NOT IN
        ('migration_id', 'applied_at', 'checksum', 'source_set')
  ) THEN
    RETURN;
  END IF;

  FOR source_row IN
    SELECT migration_id, applied_at, checksum
    FROM public.schema_migrations
    WHERE source_set = 'node'
      AND migration_id ~
        '^node:[A-Za-z0-9_][A-Za-z0-9_.-]*:[A-Za-z0-9_][A-Za-z0-9_.-]*[.]sql$'
    ORDER BY migration_id
  LOOP
    -- OMN-15857: clear the carried-over RECORD before each row.
    SELECT * INTO adoption_row
    FROM onex_verified_checksum_adoptions
    WHERE false;

    SELECT * INTO manifest_row
    FROM onex_application_migration_manifest
    WHERE version = source_row.migration_id;
    IF NOT FOUND THEN
      SELECT * INTO legacy_row
      FROM onex_legacy_node_migration_declarations
      WHERE version = source_row.migration_id;
      IF NOT FOUND THEN
        RAISE EXCEPTION
          'unknown migration stream/domain: adopted node version % has no checked-in declaration',
          source_row.migration_id;
      END IF;
      IF source_row.checksum <> legacy_row.source_checksum THEN
        RAISE EXCEPTION
          'conflicting migration checksum for version %', source_row.migration_id;
      END IF;
      resolved_stream := legacy_row.migration_stream;
      resolved_owner := legacy_row.owner;
      resolved_domain := legacy_row.domain;
      imported_checksum := encode(sha256(convert_to(
        'legacy-node-attestation|' || source_row.migration_id || '|' ||
        source_row.checksum || '|' || legacy_row.ticket,
        'UTF8'
      )), 'hex');
      imported_checksum_kind := 'legacy_attestation';
      imported_provenance := format(
        'legacy-adopted:%s:public.schema_migrations:migration_id:%s:raw-checksum=%s:ticket=%s',
        current_database(), source_row.migration_id, source_row.checksum,
        legacy_row.ticket
      );
    ELSE
      resolved_stream := manifest_row.migration_stream;
      resolved_owner := manifest_row.owner;
      resolved_domain := manifest_row.domain;
      IF source_row.checksum ~ '^[0-9a-f]{64}$' THEN
        IF source_row.checksum <> manifest_row.checksum THEN
          RAISE EXCEPTION
            'conflicting migration checksum for version %', source_row.migration_id;
        END IF;
        imported_checksum := source_row.checksum;
      ELSIF source_row.checksum = 'applied-by-runner' THEN
        imported_checksum := manifest_row.checksum;
      ELSE
        -- OMN-15857: a hand-written sentinel checksum
        -- ('hotfix-applied-by-codex', 'applied-manually-omn-11760', ...) is
        -- neither a content hash nor the runner literal, so it lands here and
        -- aborts every migration on the lane.  Widening the accepted spelling
        -- would be the wrong fix: the sentinel exists precisely to record that
        -- nobody proved the hand-applied SQL matches the checked-in file, and a
        -- blanket tolerance would launder that open question into a clean row.
        --
        -- The only accepted resolution is a committed, per-version declaration
        -- backed by a mechanical proof.  scripts/migrations/
        -- verify_migration_checksum_adoption.py replays the checked-in
        -- migration into a scratch database, derives the object surface that
        -- file is responsible for by executing it, and diffs that surface
        -- (columns, types, nullability, defaults, constraints, indexes, view
        -- definitions, enum labels) against the live database.  Only a proven
        -- `equivalent` verdict may be written into
        -- _ledger/verified-checksum-adoptions.tsv, and the row carries the
        -- sha256 of the receipt that proved it.
        --
        -- Three things must agree before the adoption is honoured here, so a
        -- stale declaration cannot outlive the fact it attested to:
        --   1. the version is declared,
        --   2. the declared source_checksum equals the sentinel actually on the
        --      row (a declaration cannot cover a different hand-edit), and
        --   3. the declared manifest_checksum equals the manifest checksum the
        --      proof ran against (rewriting the migration file after
        --      verification re-opens the question and fails closed).
        SELECT * INTO adoption_row
        FROM onex_verified_checksum_adoptions
        WHERE version = source_row.migration_id;
        IF NOT FOUND
           OR adoption_row.source_checksum <> source_row.checksum
           OR adoption_row.manifest_checksum <> manifest_row.checksum THEN
          RAISE EXCEPTION
            'conflicting migration checksum for version %', source_row.migration_id;
        END IF;
        imported_checksum := manifest_row.checksum;
      END IF;
      imported_checksum_kind := 'content_sha256';
      IF adoption_row.version IS NOT NULL THEN
        imported_provenance := format(
          'verified-adoption:%s:public.schema_migrations:migration_id:%s:raw-checksum=%s:ticket=%s:receipt=%s',
          current_database(), source_row.migration_id, source_row.checksum,
          adoption_row.ticket, adoption_row.receipt_sha256
        );
      ELSE
        imported_provenance := format(
          'adopted:%s:public.schema_migrations:migration_id:%s:raw-checksum=%s',
          current_database(), source_row.migration_id, source_row.checksum
        );
      END IF;
    END IF;

    SELECT * INTO existing_row
    FROM platform_catalog.schema_migrations
    WHERE migration_stream = resolved_stream
      AND domain = resolved_domain
      AND version = source_row.migration_id;
    IF FOUND THEN
      IF existing_row.checksum <> imported_checksum THEN
        RAISE EXCEPTION
          'conflicting migration checksum for version %', source_row.migration_id;
      ELSIF existing_row.owner <> resolved_owner
         OR existing_row.domain <> resolved_domain
         OR existing_row.checksum_kind <> imported_checksum_kind
         OR existing_row.applied_at IS DISTINCT FROM source_row.applied_at
         OR existing_row.provenance <> imported_provenance THEN
        RAISE EXCEPTION
          'double migration declaration for version %', source_row.migration_id;
      END IF;
    ELSE
      INSERT INTO platform_catalog.schema_migrations (
        migration_stream, owner, domain, version, checksum, checksum_kind,
        applied_at, provenance
      ) VALUES (
        resolved_stream, resolved_owner,
        resolved_domain, source_row.migration_id,
        imported_checksum, imported_checksum_kind, source_row.applied_at,
        imported_provenance
      );
    END IF;
  END LOOP;
END
$migration_id_import$;

-- Import omnimarket's specialized projection ledger when present.  Its
-- (node_name, filename) identity normalizes to the already-deployed
-- node:<node>:<filename> version grammar.  Two source rows normalizing to one
-- version are a double declaration and abort the transaction.
DO $omnimarket_import$
DECLARE
  source_row RECORD;
  manifest_row RECORD;
  existing_row RECORD;
  canonical_version TEXT;
  imported_checksum TEXT;
  imported_kind TEXT;
  imported_provenance TEXT;
  source_column_count INTEGER;
BEGIN
  IF to_regclass('public.omnimarket_schema_migrations') IS NULL THEN
    RETURN;
  END IF;

  SELECT count(*) INTO source_column_count
  FROM information_schema.columns
  WHERE table_schema = 'public' AND table_name = 'omnimarket_schema_migrations';
  IF source_column_count <> 6 OR EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'omnimarket_schema_migrations'
      AND (
        column_name NOT IN ('id', 'node_name', 'version', 'filename', 'checksum', 'applied_at')
        OR (column_name = 'id' AND (udt_name <> 'int4' OR is_nullable <> 'NO'))
        OR (column_name IN ('node_name', 'version', 'filename', 'checksum')
          AND (udt_name <> 'text' OR is_nullable <> 'NO'))
        OR (column_name = 'applied_at'
          AND (udt_name <> 'timestamptz' OR is_nullable <> 'NO'))
      )
  ) THEN
    RAISE EXCEPTION 'unknown migration ledger shape: public.omnimarket_schema_migrations';
  END IF;

  IF EXISTS (
    SELECT 1
    FROM public.omnimarket_schema_migrations
    GROUP BY node_name, filename
    HAVING count(*) > 1
  ) THEN
    RAISE EXCEPTION 'duplicate migration version in public.omnimarket_schema_migrations';
  END IF;

  FOR source_row IN
    SELECT node_name, version, filename, checksum, applied_at
    FROM public.omnimarket_schema_migrations
    ORDER BY node_name, version
  LOOP
    IF source_row.node_name !~ '^[A-Za-z0-9_][A-Za-z0-9_.-]*$'
       OR source_row.filename !~ '^[A-Za-z0-9_][A-Za-z0-9_.-]*[.]sql$' THEN
      RAISE EXCEPTION 'unknown migration stream identity in omnimarket source';
    END IF;
    IF source_row.version <> source_row.filename THEN
      RAISE EXCEPTION
        'unknown migration stream identity: omnimarket version % does not equal filename %',
        source_row.version, source_row.filename;
    END IF;
    canonical_version := format(
      'node:%s:%s', source_row.node_name, source_row.filename
    );
    SELECT * INTO manifest_row
    FROM onex_application_migration_manifest
    WHERE artifact_path = format(
      'nodes/%s/%s', source_row.node_name, source_row.filename
    );
    IF NOT FOUND OR manifest_row.version <> canonical_version THEN
      RAISE EXCEPTION
        'unknown migration stream/domain for omnimarket version %', canonical_version;
    END IF;
    IF source_row.checksum IS NULL
       OR source_row.checksum !~ '^[0-9a-f]{64}$'
       OR source_row.checksum <> manifest_row.checksum THEN
      RAISE EXCEPTION 'conflicting migration checksum for version %', canonical_version;
    END IF;
    imported_kind := 'content_sha256';
    imported_checksum := source_row.checksum;
    imported_provenance := format(
      'legacy:%s:public.omnimarket_schema_migrations:%s:%s:%s:raw-checksum=%s',
      current_database(), source_row.node_name, source_row.version,
      source_row.filename, coalesce(source_row.checksum, '<NULL>')
    );

    SELECT * INTO existing_row
    FROM platform_catalog.schema_migrations
    WHERE migration_stream = manifest_row.migration_stream
      AND domain = manifest_row.domain
      AND version = canonical_version;
    IF FOUND THEN
      IF existing_row.checksum <> imported_checksum THEN
        RAISE EXCEPTION 'conflicting migration checksum for version %', canonical_version;
      ELSIF existing_row.owner <> manifest_row.owner
         OR existing_row.domain <> manifest_row.domain
         OR existing_row.checksum_kind <> imported_kind
         OR existing_row.applied_at IS DISTINCT FROM source_row.applied_at
         OR existing_row.provenance <> imported_provenance THEN
        RAISE EXCEPTION 'double migration declaration for version %', canonical_version;
      END IF;
    ELSE
      INSERT INTO platform_catalog.schema_migrations (
        migration_stream, owner, domain, version, checksum, checksum_kind,
        applied_at, provenance
      ) VALUES (
        manifest_row.migration_stream, manifest_row.owner,
        manifest_row.domain, canonical_version,
        imported_checksum, imported_kind, source_row.applied_at,
        imported_provenance
      );
    END IF;
  END LOOP;
END
$omnimarket_import$;

COMMIT;
