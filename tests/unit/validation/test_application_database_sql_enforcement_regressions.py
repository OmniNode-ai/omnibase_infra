# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression controls for PostgreSQL grammar/body SQL enforcement (OMN-15361)."""

from __future__ import annotations

import pytest

from omnibase_infra.topology.application_database import load_topology_profile
from omnibase_infra.topology.physical_schema_mapping import (
    INTERNAL_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359,
)
from omnibase_infra.validation.application_database_domain_enforcement import (
    application_database_created_catalog_identities,
    application_database_sql_target_requirements,
    lint_application_database_sql,
)

pytestmark = pytest.mark.unit

_TOPOLOGY = load_topology_profile("local")


@pytest.mark.parametrize(
    ("statement", "expected"),
    [
        ("CREATE INDEX ON events (payload);", "schema-qualified"),
        ("CREATE UNIQUE INDEX ON public.events (payload);", "public"),
        ("TABLE events;", "schema-qualified"),
        ("SELECT * FROM ONLY (events);", "schema-qualified"),
        ("COMMENT ON TABLE public.events IS 'owned';", "public"),
        (
            "CREATE FUNCTION tenant.safe_report() RETURNS bigint "
            "LANGUAGE sql AS $$ SELECT count(*) FROM events $$;",
            "schema-qualified",
        ),
        (
            "CREATE FUNCTION tenant.safe_report() RETURNS void "
            "LANGUAGE plpgsql AS 'BEGIN EXECUTE ''DROP TABLE public.events''; END';",
            "dynamic SQL",
        ),
        (
            "CREATE FUNCTION tenant.safe_report() RETURNS bigint LANGUAGE sql "
            "AS E'SELECT count(*) FROM events';",
            "cannot be proven statically",
        ),
        (
            "CREATE FUNCTION tenant.safe_report() RETURNS bigint LANGUAGE sql "
            "RETURN (SELECT count(*) FROM events);",
            "schema-qualified",
        ),
        ("LOCK TABLE events IN ACCESS EXCLUSIVE MODE;", "schema-qualified"),
        ("LOCK public.events IN ACCESS EXCLUSIVE MODE;", "public"),
        ("REINDEX TABLE events;", "schema-qualified"),
        ("VACUUM events;", "schema-qualified"),
        ("ANALYZE events;", "schema-qualified"),
        ("CLUSTER events;", "schema-qualified"),
        ("CLUSTER (VERBOSE) public.events;", "public"),
        (
            "SELECT * INTO public.events_copy FROM tenant.events;",
            "public",
        ),
        (
            "SELECT * INTO events_copy FROM tenant.events;",
            "schema-qualified",
        ),
    ],
)
def test_valid_postgresql_target_forms_cannot_bypass_lint(
    statement: str,
    expected: str,
) -> None:
    assert expected in "\n".join(lint_application_database_sql(statement, _TOPOLOGY))


@pytest.mark.parametrize(
    "statement",
    [
        "CREATE INDEX ON tenant.events (payload);",
        "CREATE UNIQUE INDEX ON tenant.events (payload);",
        "TABLE tenant.events;",
        "SELECT * FROM ONLY (tenant.events);",
        "COMMENT ON TABLE tenant.events IS 'owned';",
        (
            "CREATE FUNCTION tenant.safe_report() RETURNS bigint "
            "LANGUAGE sql AS $$ SELECT count(*) FROM tenant.events $$;"
        ),
        """
        DO $$
        BEGIN
          IF to_regclass('tenant.events') IS NOT NULL THEN
            GRANT SELECT ON tenant.events TO tenant_reader;
          END IF;
        END
        $$;
        """,
        "LOCK TABLE tenant.events IN ACCESS EXCLUSIVE MODE;",
        "LOCK tenant.events IN ACCESS EXCLUSIVE MODE;",
        "REINDEX TABLE tenant.events;",
        "VACUUM tenant.events;",
        "ANALYZE tenant.events;",
        "CLUSTER tenant.events;",
        "CLUSTER (VERBOSE) tenant.events;",
    ],
)
def test_valid_postgresql_target_forms_emit_ownership_requirements(
    statement: str,
) -> None:
    requirements = application_database_sql_target_requirements(statement, _TOPOLOGY)

    assert any(
        requirement.location == ("tenant", "events") for requirement in requirements
    )


def test_select_into_emits_an_exact_created_table_identity() -> None:
    sql = "SELECT * INTO tenant.events_copy FROM tenant.events;"

    assert tuple(
        (identity.schema, identity.name, identity.kind.value)
        for identity in application_database_created_catalog_identities(sql)
    ) == (("tenant", "events_copy", "table"),)


def test_view_case_expression_with_extract_from_is_not_a_relation_target() -> None:
    """A SELECT-list EXTRACT(... FROM ...) must not make CASE look like a table."""
    sql = """
    CREATE OR REPLACE VIEW omninode_internal.gateway_link_health_status AS
    SELECT
        EXTRACT(EPOCH FROM (NOW() - last_seen_at)) AS seconds_since_last_seen,
        CASE
            WHEN NOW() - last_seen_at > INTERVAL '60 seconds' THEN 'UNHEALTHY'
            ELSE 'HEALTHY'
        END AS health_status
    FROM omninode_internal.gateway_link_health;
    """

    violations = lint_application_database_sql(sql, _TOPOLOGY)

    assert "application relation target 'case' must be schema-qualified" not in (
        "\n".join(violations)
    )


@pytest.mark.parametrize(
    ("statement", "expected_locations"),
    [
        (
            "LOCK TABLE ONLY tenant.events, omninode_internal.runtime_state "
            "IN ACCESS EXCLUSIVE MODE;",
            {("tenant", "events"), ("omninode_internal", "runtime_state")},
        ),
        (
            "REINDEX (VERBOSE) TABLE CONCURRENTLY tenant.events;",
            {("tenant", "events")},
        ),
        (
            'REINDEX (TABLESPACE "fast)tier") TABLE tenant.events;',
            {("tenant", "events")},
        ),
        (
            "VACUUM (ANALYZE, VERBOSE) tenant.events (payload), "
            "omninode_internal.runtime_state;",
            {("tenant", "events"), ("omninode_internal", "runtime_state")},
        ),
        (
            "ANALYZE (SKIP_LOCKED TRUE) VERBOSE tenant.events;",
            {("tenant", "events")},
        ),
        (
            "CLUSTER VERBOSE tenant.events USING events_payload_idx;",
            {("tenant", "events")},
        ),
    ],
)
def test_maintenance_grammar_variants_emit_every_ownership_requirement(
    statement: str,
    expected_locations: set[tuple[str, str]],
) -> None:
    requirements = application_database_sql_target_requirements(statement, _TOPOLOGY)

    assert expected_locations.issubset(
        {requirement.location for requirement in requirements}
    )


@pytest.mark.parametrize(
    "statement",
    [
        (
            "SELECT 'INTO public.decoy' AS payload "
            "INTO tenant.events_copy FROM tenant.events;"
        ),
        (
            "WITH source AS (SELECT * FROM tenant.events) "
            "SELECT 'INTO public.decoy' AS payload "
            "INTO tenant.events_copy FROM source;"
        ),
    ],
)
def test_select_into_created_identity_ignores_literals_and_leading_ctes(
    statement: str,
) -> None:
    assert tuple(
        (identity.schema, identity.name, identity.kind.value)
        for identity in application_database_created_catalog_identities(statement)
    ) == (("tenant", "events_copy", "table"),)


@pytest.mark.parametrize(
    "statement",
    [
        "SELECT 'FROM ONLY (events)'::text;",
        "SELECT 'COMMENT ON TABLE public.events'::text;",
        "SELECT $$CREATE INDEX ON public.events (payload)$$::text;",
    ],
)
def test_new_target_keywords_inside_literals_remain_inert(statement: str) -> None:
    assert not lint_application_database_sql(statement, _TOPOLOGY)


@pytest.mark.parametrize(
    "statement",
    [
        "VACUUM;",
        "VACUUM (ANALYZE, VERBOSE);",
        "ANALYZE;",
        "ANALYZE VERBOSE;",
        "CLUSTER;",
        "CLUSTER VERBOSE;",
        "CLUSTER (VERBOSE);",
        "REINDEX INDEX tenant.events_payload_idx;",
        "REINDEX SCHEMA tenant;",
        "REINDEX DATABASE omnidash_analytics;",
        "REINDEX SYSTEM omnidash_analytics;",
        'REINDEX (TABLESPACE "fast)tier") INDEX tenant.events_payload_idx;',
    ],
)
def test_broad_or_non_table_maintenance_operations_fail_closed(
    statement: str,
) -> None:
    assert "exact schema-qualified table target" in "\n".join(
        lint_application_database_sql(statement, _TOPOLOGY)
    )


@pytest.mark.parametrize(
    "statement",
    [
        "GRANT CONNECT ON DATABASE omnidash_analytics TO onex_api;",
        "REVOKE CONNECT ON DATABASE omnidash_analytics FROM PUBLIC;",
        "GRANT USAGE ON SCHEMA tenant TO onex_api;",
        "REVOKE ALL ON FUNCTION tenant.safe_report() FROM PUBLIC;",
    ],
)
def test_database_schema_and_principal_privilege_tokens_are_not_relations(
    statement: str,
) -> None:
    assert not lint_application_database_sql(statement, _TOPOLOGY)


@pytest.mark.parametrize(
    "statement",
    [
        'VACUUM tenant."events""rogue";',
        'LOCK tenant."events""rogue" IN ACCESS EXCLUSIVE MODE;',
        'REINDEX TABLE tenant."events""rogue";',
        'ANALYZE tenant."events""rogue";',
        'CLUSTER tenant."events""rogue";',
    ],
)
def test_escaped_quoted_relation_identifiers_remain_exact(statement: str) -> None:
    requirements = application_database_sql_target_requirements(statement, _TOPOLOGY)

    assert any(
        requirement.location == ("tenant", 'events"rogue')
        for requirement in requirements
    )
    assert all(
        requirement.location != ("tenant", "events") for requirement in requirements
    )


def test_escaped_quoted_schema_identifier_cannot_alias_a_topology_schema() -> None:
    violations = lint_application_database_sql(
        'VACUUM "tenant""rogue".events;',
        _TOPOLOGY,
    )

    assert "unknown topology schema" in "\n".join(violations)


def test_select_into_preserves_escaped_quoted_created_identity() -> None:
    sql = 'SELECT * INTO tenant."events_copy""rogue" FROM tenant.events;'

    with pytest.raises(ValueError, match="name"):
        application_database_created_catalog_identities(sql)


@pytest.mark.parametrize(
    "statement",
    [
        "VACUUM tenant.eventsé;",
        "LOCK tenant.eventsé IN ACCESS EXCLUSIVE MODE;",
        "REINDEX TABLE tenant.eventsé;",
        "ANALYZE tenant.eventsé;",
        "CLUSTER tenant.eventsé;",
    ],
)
def test_unicode_unquoted_relation_identifiers_remain_exact(statement: str) -> None:
    requirements = application_database_sql_target_requirements(statement, _TOPOLOGY)

    assert any(
        requirement.location == ("tenant", "eventsé") for requirement in requirements
    )
    assert all(
        requirement.location != ("tenant", "events") for requirement in requirements
    )


def test_unicode_escape_identifiers_fail_closed_explicitly() -> None:
    violations = lint_application_database_sql(
        r'VACUUM U&"tenant".events;',
        _TOPOLOGY,
    )

    assert "Unicode-escaped identifiers" in "\n".join(violations)


def test_nested_block_comments_fail_closed_explicitly() -> None:
    violations = lint_application_database_sql(
        "REINDEX /* outer /* inner */ outer */ TABLE public.events;",
        _TOPOLOGY,
    )

    assert "nested block comments" in "\n".join(violations)


@pytest.mark.parametrize(
    "statement",
    [
        "VACUUM /* tenant.events /* nested */ tenant.events */ public.events;",
        "LOCK TABLE /* tenant.events /* nested */ tenant.events */ public.events;",
        "REINDEX /* tenant.events /* nested */ tenant.events */ TABLE public.events;",
        "ANALYZE /* tenant.events /* nested */ tenant.events */ public.events;",
        "CLUSTER /* tenant.events /* nested */ tenant.events */ public.events;",
        (
            "SELECT * INTO /* tenant.events_copy /* nested */ "
            "tenant.events_copy */ public.events_copy FROM tenant.events;"
        ),
    ],
)
def test_nested_block_comment_decoys_fail_closed_for_every_target_form(
    statement: str,
) -> None:
    assert "nested block comments" in "\n".join(
        lint_application_database_sql(statement, _TOPOLOGY)
    )


def test_unclosed_block_comments_fail_closed_explicitly() -> None:
    violations = lint_application_database_sql(
        "SELECT * FROM tenant.events /* unclosed",
        _TOPOLOGY,
    )

    assert "unterminated block comment" in "\n".join(violations)


def test_escape_string_quote_cannot_mask_a_real_public_target() -> None:
    violations = lint_application_database_sql(
        "SELECT E'foo\\'bar' FROM public.events;",
        _TOPOLOGY,
    )

    assert "public" in "\n".join(violations)


def test_quote_inside_delimited_identifier_cannot_mask_a_real_public_target() -> None:
    violations = lint_application_database_sql(
        'SELECT 1 AS "foo\'bar" FROM public.events;',
        _TOPOLOGY,
    )

    assert "public" in "\n".join(violations)


def test_select_into_ignores_into_keyword_inside_a_delimited_alias() -> None:
    sql = (
        'SELECT 1 AS "INTO tenant.events_copy" '
        "INTO public.events_copy FROM tenant.events;"
    )

    assert "public" in "\n".join(lint_application_database_sql(sql, _TOPOLOGY))
    assert tuple(
        (identity.schema, identity.name, identity.kind.value)
        for identity in application_database_created_catalog_identities(sql)
    ) == (("public", "events_copy", "table"),)


def test_escape_string_before_select_into_preserves_the_created_target() -> None:
    sql = "SELECT E'foo\\'bar' INTO tenant.events_copy FROM tenant.events;"

    assert tuple(
        (identity.schema, identity.name, identity.kind.value)
        for identity in application_database_created_catalog_identities(sql)
    ) == (("tenant", "events_copy", "table"),)


# ---------------------------------------------------------------------------
# OMN-15361: a view body may open its own WITH clause. The CTE names live past
# the `AS`, so a leading-WITH-only parse never collected them and every later
# reference to one was misread as an unqualified application relation.
# ---------------------------------------------------------------------------

_VIEW_BODY_CTE_STATEMENTS: tuple[tuple[str, str], ...] = (
    (
        "plain",
        "CREATE VIEW tenant.v AS WITH totals AS (SELECT 1 AS n) "
        "SELECT totals.n FROM totals;",
    ),
    (
        "or-replace",
        "CREATE OR REPLACE VIEW tenant.v AS WITH totals AS (SELECT 1 AS n) "
        "SELECT totals.n FROM totals;",
    ),
    (
        "materialized",
        "CREATE MATERIALIZED VIEW tenant.v AS WITH totals AS (SELECT 1 AS n) "
        "SELECT totals.n FROM totals;",
    ),
    (
        "recursive-cte",
        "CREATE VIEW tenant.v AS WITH RECURSIVE walk AS ("
        "SELECT 1 AS n UNION ALL SELECT n + 1 FROM walk WHERE n < 5) "
        "SELECT walk.n FROM walk;",
    ),
    (
        "column-list",
        "CREATE VIEW tenant.v (n) AS WITH totals AS (SELECT 1 AS n) "
        "SELECT totals.n FROM totals;",
    ),
    (
        "security-invoker-option-list",
        "CREATE OR REPLACE VIEW tenant.v WITH (security_invoker = true) AS "
        "WITH totals AS (SELECT 1 AS n) SELECT totals.n FROM totals;",
    ),
    (
        "chained-ctes-cross-joined",
        "CREATE OR REPLACE VIEW tenant.v AS WITH totals AS (SELECT 1 AS n), "
        "failure_categories AS (SELECT 2 AS rows), "
        "tokens_by_model AS (SELECT 3 AS tokens) "
        "SELECT totals.n, failure_categories.rows AS failure_categories, "
        "tokens_by_model.tokens FROM totals "
        "CROSS JOIN failure_categories CROSS JOIN tokens_by_model;",
    ),
)


@pytest.mark.parametrize(
    ("shape", "statement"),
    _VIEW_BODY_CTE_STATEMENTS,
    ids=[shape for shape, _ in _VIEW_BODY_CTE_STATEMENTS],
)
def test_view_body_cte_names_are_not_misread_as_relations(
    shape: str,
    statement: str,
) -> None:
    assert lint_application_database_sql(statement, _TOPOLOGY) == ()


def test_view_body_cte_parse_matches_the_observed_promotion_failure() -> None:
    """The exact shape that failed the gate on the dev->main promotion.

    Reduced from 0028_reconcile_delegation_observability_views.sql, keeping the
    parts that mattered: a CREATE OR REPLACE VIEW whose body opens a three-CTE
    WITH chain, a jsonb aggregate with its own nested parentheses and ORDER BY,
    and a trailing column alias that shadows one of the CTE names.
    """
    sql = """
CREATE OR REPLACE VIEW tenant.projection_delegation_quality_gate AS
WITH totals AS (
    SELECT COALESCE(AVG(actual_score), 0)::float AS avg_actual_score
    FROM tenant.delegation_events
),
failure_categories AS (
    SELECT COALESCE(
        jsonb_agg(
            jsonb_build_object('category', quality_gate_detail)
            ORDER BY quality_gate_detail DESC
        ),
        '[]'::jsonb
    ) AS rows
    FROM tenant.delegation_events
),
tokens_by_model AS (
    SELECT COALESCE(jsonb_agg(jsonb_build_object('model', model)), '[]'::jsonb) AS rows
    FROM tenant.delegation_events
)
SELECT
    totals.avg_actual_score,
    failure_categories.rows AS failure_categories,
    tokens_by_model.rows AS tokens_by_model
FROM totals
CROSS JOIN failure_categories
CROSS JOIN tokens_by_model;
"""

    assert lint_application_database_sql(sql, _TOPOLOGY) == ()


@pytest.mark.parametrize(
    ("statement", "expected"),
    [
        # An unqualified relation in the post-WITH tail still fails. Uses
        # unmapped_events (not delegation_events) -- OMN-16237 makes
        # delegation_events itself pass unqualified, since it is enumerated
        # in the shared physical-schema-mapping allowlist; this table name
        # is deliberately NOT in that allowlist so the CTE-scoping behavior
        # under test stays independent of the allowlist pass-through.
        (
            "CREATE VIEW tenant.v AS WITH totals AS (SELECT 1 AS n) "
            "SELECT * FROM totals CROSS JOIN unmapped_events;",
            "'unmapped_events' must be schema-qualified",
        ),
        # An unqualified relation inside a CTE body still fails.
        (
            "CREATE VIEW tenant.v AS WITH totals AS "
            "(SELECT * FROM unmapped_events) SELECT * FROM totals;",
            "'unmapped_events' must be schema-qualified",
        ),
        # The view's own name is still a real relation target.
        (
            "CREATE VIEW unqualified_view AS WITH totals AS (SELECT 1 AS n) "
            "SELECT totals.n FROM totals;",
            "'unqualified_view' must be schema-qualified",
        ),
        # A CTE name is only in scope for its own statement, never the next one.
        (
            "CREATE VIEW tenant.v AS WITH totals AS (SELECT 1 AS n) "
            "SELECT totals.n FROM totals; SELECT * FROM totals;",
            "'totals' must be schema-qualified",
        ),
        # A later CTE is not visible to an earlier sibling's body.
        (
            "CREATE VIEW tenant.v AS WITH first_cte AS (SELECT * FROM second_cte), "
            "second_cte AS (SELECT 1 AS n) SELECT * FROM first_cte;",
            "'second_cte' must be schema-qualified",
        ),
    ],
)
def test_view_body_cte_recognition_never_exempts_a_real_relation(
    statement: str,
    expected: str,
) -> None:
    assert expected in "\n".join(lint_application_database_sql(statement, _TOPOLOGY))


# ---------------------------------------------------------------------------
# OMN-16237: the static schema-qualification lint must consult the SAME
# physical-schema allowlist the runtime grants system already trusts
# (physical_grant_schema_for_table / INTERNAL_TABLES_PHYSICALLY_IN_PUBLIC_
# UNTIL_OMN15359, TENANT_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359) instead
# of unconditionally rejecting every unqualified or public-qualified
# application relation. A table only satisfies the allowlist pass-through by
# being enumerated there -- it must never become a blanket public exemption.
# ---------------------------------------------------------------------------


def test_intent_classification_events_is_enumerated_in_the_shared_allowlist() -> None:
    """Guard against the two allowlists drifting apart (import-shared, not copied)."""
    assert (
        "intent_classification_events"
        in INTERNAL_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359
    )


@pytest.mark.parametrize(
    "statement",
    [
        "ALTER TABLE intent_classification_events ADD COLUMN agent_source text;",
        "ALTER TABLE public.intent_classification_events ADD COLUMN agent_source text;",
    ],
)
def test_allowlisted_physically_public_table_passes_unqualified_or_public(
    statement: str,
) -> None:
    """OMN-16237: a table enumerated in the physical-schema-mapping allowlist
    satisfies schema-qualification whether referenced unqualified (matching
    its real physical location) or explicitly public-qualified. This is the
    exact shape OMN-14751's 0001_intent_classification_agent_source.sql
    ships -- an unqualified ALTER on a table whose contract is logically
    omninode_internal but physically public pending the OMN-15359 migration.
    """
    assert lint_application_database_sql(statement, _TOPOLOGY) == ()


@pytest.mark.parametrize(
    ("statement", "expected"),
    [
        (
            "ALTER TABLE not_an_allowlisted_table ADD COLUMN x text;",
            "'not_an_allowlisted_table' must be schema-qualified",
        ),
        (
            "ALTER TABLE public.not_an_allowlisted_table ADD COLUMN x text;",
            "'public.not_an_allowlisted_table' is prohibited in public",
        ),
    ],
)
def test_non_allowlisted_table_in_public_still_fails(
    statement: str,
    expected: str,
) -> None:
    """The allowlist is a narrow, enumerated pass-through, never a blanket
    public/unqualified exemption -- a table absent from the shared allowlist
    must still be rejected exactly as before."""
    assert expected in "\n".join(lint_application_database_sql(statement, _TOPOLOGY))


# ---------------------------------------------------------------------------
# OMN-17301: PL/pgSQL `SELECT ... INTO <var>` is variable assignment, never a
# relation target. PostgreSQL has no `SELECT INTO <table>` form inside
# PL/pgSQL — the create-table-as spelling is plain-SQL only — so treating a
# declared variable as an unqualified relation is a false positive that fires
# on any migration whose DO block reads a catalog value into a local.
# Reproduced by 103_create_tenant_projection_writer_role.sql, whose CONNECT
# readback declares db_present / explicit_grant / effective_connect.
# ---------------------------------------------------------------------------


def test_plpgsql_select_into_declared_variable_is_not_a_relation_target() -> None:
    sql = (
        "DO $$\n"
        "DECLARE\n"
        "  db_present        boolean;\n"
        "  explicit_grant    boolean;\n"
        "  effective_connect boolean;\n"
        "BEGIN\n"
        "  SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_database "
        "WHERE datname = 'omnidash_analytics')\n"
        "    INTO db_present;\n"
        "  SELECT true INTO explicit_grant;\n"
        "  SELECT has_database_privilege('r', 'omnidash_analytics', 'CONNECT')\n"
        "    INTO effective_connect;\n"
        "END\n"
        "$$;"
    )

    violations = lint_application_database_sql(sql, _TOPOLOGY)

    assert violations == (), violations
    # A declared variable is not a created relation either.
    assert application_database_created_catalog_identities(sql) == ()


def test_plpgsql_select_into_undeclared_name_still_fails_closed() -> None:
    """The narrowing is scoped to DECLARE'd names — it is not a blanket pass."""
    sql = (
        "DO $$\n"
        "DECLARE\n"
        "  db_present boolean;\n"
        "BEGIN\n"
        "  SELECT 1 INTO events_copy;\n"
        "END\n"
        "$$;"
    )

    assert "schema-qualified" in "\n".join(
        lint_application_database_sql(sql, _TOPOLOGY)
    )


def test_plpgsql_declared_variable_does_not_mask_a_real_relation_read() -> None:
    """Declaring a name must not license an unqualified FROM of the same name."""
    sql = (
        "DO $$\n"
        "DECLARE\n"
        "  events boolean;\n"
        "BEGIN\n"
        "  SELECT count(*) > 0 FROM events INTO events;\n"
        "END\n"
        "$$;"
    )

    assert "schema-qualified" in "\n".join(
        lint_application_database_sql(sql, _TOPOLOGY)
    )
