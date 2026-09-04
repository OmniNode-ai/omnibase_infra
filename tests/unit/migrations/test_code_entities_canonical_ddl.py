# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Schema-shape contract for the canonical ``code_entities`` DDL (OMN-15276).

Why this file exists
--------------------
``code_entities`` was defined twice, incompatibly, in
``omniintelligence/deployment/database/migrations/`` — ``025_code_entities.sql``
(OMN-5661) and ``025_create_code_entities.sql`` (OMN-5709) — and *neither* was in
the set any lane applies. Both were ``CREATE TABLE IF NOT EXISTS``, so applying
both never converged: whichever sorted first silently won and the other's columns
never appeared. OMN-5765 closed "Done" on 2026-03-21 claiming this was reconciled;
both files were still on omniintelligence dev four months later.

These tests bind the surviving DDL to the columns its two live consumers actually
read, so the reconciliation cannot silently regress:

* **RED half** — :data:`REJECTED_OMN_5709_DDL` is the losing shape, kept verbatim.
  :func:`test_rejected_omn5709_shape_fails_the_consumer_column_contract` asserts it
  does *not* satisfy the contract. This is an exists-but-wrong RED: without it,
  "the DDL has the right columns" would pass against any superset and the choice
  between the two shapes would be untested.
* **Ownership half** — the canonical file must live in the directory the lanes
  apply. A fix landed in the omniintelligence tree reads green and changes
  nothing on any lane; see ``knowledge-base-internal:reference/omnibase-infra-omniintelligence-migration-set.md``.

Column provenance is recorded per constant. These lists are transcribed from the
consumers' SQL, not invented here; each constant names the file and statement it
came from so a reviewer can diff it against source.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

#: The directory the ``.201`` lanes actually apply. Proven by
#: ``docker/docker-compose.infra.yml`` (``MIGRATIONS_DIR: /migrations/intelligence``
#: + the ``../docker/migrations/intelligence`` bind mount), the identical pair in
#: ``docker/docker-compose.judge.yml`` and
#: ``docker/catalog/services/intelligence-migration.yaml``, and
#: ``scripts/run-intelligence-migrations.sh`` which globs ``${MIGRATIONS_DIR}/*.sql``.
INTELLIGENCE_MIGRATIONS_DIR = REPO_ROOT / "docker" / "migrations" / "intelligence"

CANONICAL_MIGRATION = INTELLIGENCE_MIGRATIONS_DIR / "026_create_code_entities.sql"

# ---------------------------------------------------------------------------
# Consumer column contracts (transcribed from the consumers' SQL)
# ---------------------------------------------------------------------------

#: Columns ``omnimarket``'s ``RepositoryCodeEntityPostgres`` reads or writes
#: (``src/omnimarket/repositories/repository_code_entity_postgres.py``, omnimarket#1923).
#: Sources, in order: ``_EMBEDDING_COLUMNS`` projection; ``_ENRICHMENT_COLUMNS``
#: projection; the ``last_embedded_at IS NULL OR last_embedded_at < last_extracted_at``
#: and ``classification IS NULL`` predicates plus ``ORDER BY last_extracted_at``;
#: the ``update_embedded_at`` and ``update_enrichment`` SET lists.
OMNIMARKET_CONSUMER_COLUMNS: frozenset[str] = frozenset(
    {
        # _EMBEDDING_COLUMNS
        "id",
        "entity_name",
        "entity_type",
        "qualified_name",
        "source_repo",
        "source_path",
        "docstring",
        "signature",
        "classification",
        "llm_description",
        # _ENRICHMENT_COLUMNS (adds these four)
        "bases",
        "methods",
        "fields",
        "decorators",
        # predicates / ordering
        "last_embedded_at",
        "last_extracted_at",
        # update_embedded_at / update_enrichment SET lists
        "updated_at",
        "architectural_pattern",
        "classification_confidence",
        "enrichment_version",
        "last_enriched_at",
    }
)

#: Columns the AST-extraction producer reads or writes
#: (``omniintelligence/src/omniintelligence/nodes/node_ast_extraction_compute/``
#: ``repository/repository_code_entity.py``). Everything in
#: :data:`OMNIMARKET_CONSUMER_COLUMNS` plus the extraction-side and part-2
#: enrichment columns: the ``upsert_entity`` INSERT column list (``line_number``,
#: ``file_hash``), ``update_graph_synced_at``, ``update_deterministic_classification``,
#: ``update_quality_score``, and ``get_entity_enrichment_metadata``.
OMNIINTELLIGENCE_PRODUCER_COLUMNS: frozenset[str] = (
    OMNIMARKET_CONSUMER_COLUMNS
    | frozenset(
        {
            "line_number",
            "file_hash",
            "last_graph_synced_at",
            "deterministic_node_type",
            "deterministic_confidence",
            "deterministic_alternatives",
            "quality_score",
            "quality_dimensions",
            "enrichment_metadata",
        }
    )
)

#: Columns the producer's ``upsert_relationship`` INSERT and the
#: ``get_all_entities_and_relationships`` projection touch on ``code_relationships``.
CODE_RELATIONSHIP_COLUMNS: frozenset[str] = frozenset(
    {
        "id",
        "source_entity_id",
        "target_entity_id",
        "relationship_type",
        "trust_tier",
        "confidence",
        "evidence",
        "inject_into_context",
        "source_repo",
        "updated_at",
    }
)

#: The losing shape, verbatim from the retired
#: ``omniintelligence/deployment/database/migrations/025_create_code_entities.sql``
#: (OMN-5709 / OMN-5720). Kept inline so the RED half survives the file's deletion.
REJECTED_OMN_5709_DDL = """
CREATE TABLE IF NOT EXISTS code_entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type TEXT NOT NULL,
    name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_hash VARCHAR(64) NOT NULL,
    source_repo TEXT NOT NULL,
    line_start INT,
    line_end INT,
    bases JSONB DEFAULT '[]',
    methods JSONB DEFAULT '[]',
    decorators JSONB DEFAULT '[]',
    docstring TEXT,
    source_code TEXT,
    confidence FLOAT DEFAULT 1.0,
    classification TEXT,
    embedding_id TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(source_repo, file_path, name, entity_type)
);
"""

#: Sequence prefixes allowed to appear more than once in the intelligence set.
#: ``023`` is grandfathered: both ``023_create_debug_intelligence_tables.sql`` and
#: ``023_create_dispatch_eval_results.sql`` are already recorded in
#: ``omniintelligence.schema_migrations`` on every lane, and the runner keys on the
#: file's basename — renumbering an applied file re-applies its SQL under a new id.
#: The duplicate exists because ``scripts/validation/validate_migration_sequence.py``
#: scans only ``docker/migrations/forward`` and ``src/omnibase_infra/migrations/forward``,
#: never this directory. This allowlist is a ratchet: it may shrink, never grow.
GRANDFATHERED_DUPLICATE_PREFIXES: frozenset[str] = frozenset({"023"})

# Statement-level table/constraint keywords that open a table-constraint clause
# rather than a column definition.
_NON_COLUMN_LEADERS = frozenset(
    {"UNIQUE", "PRIMARY", "FOREIGN", "CONSTRAINT", "CHECK", "EXCLUDE", "LIKE"}
)


def _strip_line_comments(sql: str) -> str:
    """Drop ``--`` comments so commas inside prose never split a column list."""
    return "\n".join(line.split("--", 1)[0] for line in sql.splitlines())


def _create_table_body(sql: str, table: str) -> str:
    """Return the parenthesised body of ``CREATE TABLE [IF NOT EXISTS] <table>``."""
    cleaned = _strip_line_comments(sql)
    match = re.search(
        rf"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?{re.escape(table)}\s*\(",
        cleaned,
        re.IGNORECASE,
    )
    if match is None:
        raise AssertionError(f"no CREATE TABLE statement for {table!r}")

    depth = 0
    for offset, char in enumerate(cleaned[match.end() - 1 :], start=match.end() - 1):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return cleaned[match.end() : offset]
    raise AssertionError(f"unbalanced parentheses in CREATE TABLE {table}")


def _split_top_level(body: str) -> list[str]:
    """Split a CREATE TABLE body on commas that are not inside parentheses."""
    items: list[str] = []
    depth = 0
    current: list[str] = []
    for char in body:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char == "," and depth == 0:
            items.append("".join(current))
            current = []
        else:
            current.append(char)
    items.append("".join(current))
    return [item.strip() for item in items if item.strip()]


def declared_columns(sql: str, table: str) -> frozenset[str]:
    """Column names declared by ``CREATE TABLE <table>`` in *sql*."""
    columns = set()
    for item in _split_top_level(_create_table_body(sql, table)):
        # The leading identifier, stopping at the first non-identifier character —
        # `UNIQUE(id, payload)` has no space after the keyword, so splitting on
        # whitespace alone would yield the bogus column name `UNIQUE(id,`.
        leader_match = re.match(r'\s*"?([A-Za-z_][A-Za-z0-9_]*)"?', item)
        if leader_match is None:
            continue
        leader = leader_match.group(1)
        if leader.upper() in _NON_COLUMN_LEADERS:
            continue
        columns.add(leader)
    return frozenset(columns)


@pytest.fixture(scope="module")
def canonical_sql() -> str:
    assert CANONICAL_MIGRATION.is_file(), (
        f"canonical code_entities DDL missing at {CANONICAL_MIGRATION.relative_to(REPO_ROOT)} — "
        "it must live in the directory the lanes apply (OMN-15276)"
    )
    return CANONICAL_MIGRATION.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Parser self-checks — a broken parser would make every assertion below vacuous.
# ---------------------------------------------------------------------------


def test_declared_columns_ignores_table_constraints_and_comments() -> None:
    sql = """
    CREATE TABLE IF NOT EXISTS sample (
        id UUID PRIMARY KEY,
        payload JSONB,  -- a comment, with a comma
        amount NUMERIC(14, 6),
        UNIQUE(id, payload)
    );
    """
    assert declared_columns(sql, "sample") == frozenset({"id", "payload", "amount"})


def test_declared_columns_raises_when_the_table_is_absent() -> None:
    with pytest.raises(AssertionError, match="no CREATE TABLE statement"):
        declared_columns("CREATE TABLE other (id UUID);", "sample")


# ---------------------------------------------------------------------------
# Ownership: the canonical DDL is in the directory the lanes apply, exactly once.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("table", ["code_entities", "code_relationships"])
def test_exactly_one_migration_creates_the_table(table: str) -> None:
    pattern = re.compile(
        rf"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?{table}\b", re.IGNORECASE
    )
    creators = sorted(
        path.name
        for path in REPO_ROOT.rglob("*.sql")
        if ".git" not in path.parts and pattern.search(path.read_text(encoding="utf-8"))
    )
    assert creators == [CANONICAL_MIGRATION.name], (
        f"{table} must be created by exactly one migration; found {creators}. "
        "Two CREATE TABLE IF NOT EXISTS definitions never converge — the first to "
        "sort wins and the other's columns silently never appear (OMN-15276)."
    )


def test_no_new_duplicate_sequence_prefixes_in_the_applied_set() -> None:
    by_prefix: defaultdict[str, list[str]] = defaultdict(list)
    for path in sorted(INTELLIGENCE_MIGRATIONS_DIR.glob("*.sql")):
        match = re.match(r"(\d+)_", path.name)
        assert match is not None, f"{path.name} does not start with a sequence number"
        by_prefix[match.group(1)].append(path.name)

    duplicates = {
        prefix: names for prefix, names in by_prefix.items() if len(names) > 1
    }
    unexpected = {
        prefix: names
        for prefix, names in duplicates.items()
        if prefix not in GRANDFATHERED_DUPLICATE_PREFIXES
    }
    assert not unexpected, (
        f"new duplicate migration prefixes in {INTELLIGENCE_MIGRATIONS_DIR.name}/: "
        f"{unexpected}. Sorted order is the apply order; a duplicated prefix makes it "
        "ambiguous. Take the next free number instead."
    )
    # Ratchet: the allowlist may shrink, never grow.
    assert set(GRANDFATHERED_DUPLICATE_PREFIXES) >= set(duplicates), (
        "GRANDFATHERED_DUPLICATE_PREFIXES lists a prefix that is no longer duplicated; "
        "remove the stale entry rather than leaving a dead exemption."
    )


# ---------------------------------------------------------------------------
# Schema shape: the DDL provides every column both consumers project.
# ---------------------------------------------------------------------------


def test_canonical_ddl_provides_every_column_omnimarket_projects(
    canonical_sql: str,
) -> None:
    missing = OMNIMARKET_CONSUMER_COLUMNS - declared_columns(
        canonical_sql, "code_entities"
    )
    assert not missing, (
        f"code_entities is missing columns RepositoryCodeEntityPostgres projects: "
        f"{sorted(missing)}. Each one is a runtime UndefinedColumn on the first "
        "dispatch of node_code_embedding_effect / node_code_enrichment_effect."
    )


def test_canonical_ddl_provides_every_column_the_ast_producer_writes(
    canonical_sql: str,
) -> None:
    missing = OMNIINTELLIGENCE_PRODUCER_COLUMNS - declared_columns(
        canonical_sql, "code_entities"
    )
    assert not missing, (
        f"code_entities is missing columns RepositoryCodeEntity writes: {sorted(missing)}"
    )


def test_canonical_ddl_provides_every_code_relationship_column(
    canonical_sql: str,
) -> None:
    missing = CODE_RELATIONSHIP_COLUMNS - declared_columns(
        canonical_sql, "code_relationships"
    )
    assert not missing, (
        f"code_relationships is missing columns upsert_relationship writes: "
        f"{sorted(missing)}"
    )


def test_canonical_ddl_keeps_the_upsert_key_both_repositories_conflict_on(
    canonical_sql: str,
) -> None:
    normalized = re.sub(r"\s+", " ", canonical_sql)
    assert "UNIQUE(qualified_name, source_repo)" in normalized, (
        "RepositoryCodeEntity.upsert_entity uses "
        "ON CONFLICT (qualified_name, source_repo); without the matching UNIQUE "
        "constraint every upsert raises InvalidColumnReference."
    )
    assert (
        "UNIQUE(source_entity_id, target_entity_id, relationship_type)" in normalized
    ), (
        "upsert_relationship uses ON CONFLICT "
        "(source_entity_id, target_entity_id, relationship_type)."
    )


def test_enrichment_metadata_is_non_null_so_jsonb_merge_cannot_erase_it(
    canonical_sql: str,
) -> None:
    # `enrichment_metadata = enrichment_metadata || $5::jsonb` yields NULL when the
    # left operand is NULL, so a nullable column silently drops every idempotency
    # stamp instead of accumulating them.
    assert re.search(
        r"enrichment_metadata\s+JSONB\s+NOT\s+NULL\s+DEFAULT\s+'\{\}'",
        canonical_sql,
    ), "enrichment_metadata must be NOT NULL DEFAULT '{}'"


# ---------------------------------------------------------------------------
# RED half: the rejected shape must fail the same contract.
# ---------------------------------------------------------------------------


def test_rejected_omn5709_shape_fails_the_consumer_column_contract() -> None:
    rejected = declared_columns(REJECTED_OMN_5709_DDL, "code_entities")
    missing = OMNIMARKET_CONSUMER_COLUMNS - rejected

    assert missing, (
        "the OMN-5709 shape now satisfies the consumer contract — either the "
        "contract was weakened or the wrong DDL was reinstated as canonical"
    )
    # Name the specific divergences so a partial revert cannot pass this test.
    assert {
        "entity_name",
        "qualified_name",
        "source_path",
        "signature",
        "llm_description",
        "architectural_pattern",
        "classification_confidence",
        "enrichment_version",
        "last_extracted_at",
        "last_enriched_at",
        "last_embedded_at",
    } <= missing
    # It also renames the same concepts, which is why "just add the missing
    # columns" was not an option.
    assert {"name", "file_path", "line_start", "line_end"} <= rejected


def test_rejected_and_canonical_shapes_disagree_on_the_dedup_key(
    canonical_sql: str,
) -> None:
    normalized_canonical = re.sub(r"\s+", " ", canonical_sql)
    normalized_rejected = re.sub(r"\s+", " ", REJECTED_OMN_5709_DDL)
    assert "UNIQUE(qualified_name, source_repo)" in normalized_canonical
    assert "UNIQUE(source_repo, file_path, name, entity_type)" in normalized_rejected, (
        "the rejected shape's entity identity differs; the two never converge"
    )
    assert "UNIQUE(qualified_name, source_repo)" not in normalized_rejected
