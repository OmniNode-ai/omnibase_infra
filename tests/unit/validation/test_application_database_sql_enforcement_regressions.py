# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression controls for PostgreSQL grammar/body SQL enforcement (OMN-15361)."""

from __future__ import annotations

import pytest

from omnibase_infra.topology.application_database import load_topology_profile
from omnibase_infra.topology.physical_schema_mapping import (
    APPLICATION_SEQUENCES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359,
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
        ("CREATE UNIQUE INDEX ON tenant.events (payload);", "unknown topology schema"),
        ("TABLE events;", "schema-qualified"),
        ("SELECT * FROM ONLY (events);", "schema-qualified"),
        ("COMMENT ON TABLE tenant.events IS 'owned';", "unknown topology schema"),
        (
            "CREATE FUNCTION omninode_internal.safe_report() RETURNS bigint "
            "LANGUAGE sql AS $$ SELECT count(*) FROM events $$;",
            "schema-qualified",
        ),
        (
            "CREATE FUNCTION omninode_internal.safe_report() RETURNS void "
            "LANGUAGE plpgsql AS 'BEGIN EXECUTE ''DROP TABLE public.events''; END';",
            "dynamic SQL",
        ),
        (
            "CREATE FUNCTION omninode_internal.safe_report() RETURNS bigint LANGUAGE sql "
            "AS E'SELECT count(*) FROM events';",
            "cannot be proven statically",
        ),
        (
            "CREATE FUNCTION omninode_internal.safe_report() RETURNS bigint LANGUAGE sql "
            "RETURN (SELECT count(*) FROM events);",
            "schema-qualified",
        ),
        ("LOCK TABLE events IN ACCESS EXCLUSIVE MODE;", "schema-qualified"),
        ("LOCK tenant.events IN ACCESS EXCLUSIVE MODE;", "unknown topology schema"),
        ("REINDEX TABLE events;", "schema-qualified"),
        ("VACUUM events;", "schema-qualified"),
        ("ANALYZE events;", "schema-qualified"),
        ("CLUSTER events;", "schema-qualified"),
        ("CLUSTER (VERBOSE) tenant.events;", "unknown topology schema"),
        (
            "SELECT * INTO tenant.events_copy FROM omninode_internal.events;",
            "unknown topology schema",
        ),
        (
            "SELECT * INTO events_copy FROM omninode_internal.events;",
            "schema-qualified",
        ),
    ],
)
def test_valid_postgresql_target_forms_cannot_bypass_lint(
    statement: str,
    expected: str,
) -> None:
    # OMN-17887: the schema-refusal rows name the retired `tenant` schema.
    # `public` is now the TENANT domain's declared schema, so a `public.<name>`
    # target is no longer a lint refusal; it is held to exact ownership instead
    # (see test_valid_postgresql_public_target_forms_are_held_to_ownership).
    assert expected in "\n".join(lint_application_database_sql(statement, _TOPOLOGY))


@pytest.mark.parametrize(
    "statement",
    [
        "CREATE UNIQUE INDEX ON public.events (payload);",
        "COMMENT ON TABLE public.events IS 'owned';",
        "LOCK public.events IN ACCESS EXCLUSIVE MODE;",
        "CLUSTER (VERBOSE) public.events;",
    ],
)
def test_valid_postgresql_public_target_forms_are_held_to_ownership(
    statement: str,
) -> None:
    """OMN-17887: a `public` target in every grammar form is still seen.

    The lint no longer refuses `public.<name>` outright, so the only thing
    between an undeclared public relation and deployment is the gate's
    exactly-one-ownership check -- which can only fire on a requirement the
    parser actually emitted.
    """
    requirements = application_database_sql_target_requirements(statement, _TOPOLOGY)

    assert any(
        requirement.location == ("public", "events") for requirement in requirements
    )


@pytest.mark.parametrize(
    "statement",
    [
        "CREATE INDEX ON omninode_internal.events (payload);",
        "CREATE UNIQUE INDEX ON omninode_internal.events (payload);",
        "TABLE omninode_internal.events;",
        "SELECT * FROM ONLY (omninode_internal.events);",
        "COMMENT ON TABLE omninode_internal.events IS 'owned';",
        (
            "CREATE FUNCTION omninode_internal.safe_report() RETURNS bigint "
            "LANGUAGE sql AS $$ SELECT count(*) FROM omninode_internal.events $$;"
        ),
        """
        DO $$
        BEGIN
          IF to_regclass('omninode_internal.events') IS NOT NULL THEN
            GRANT SELECT ON omninode_internal.events TO tenant_reader;
          END IF;
        END
        $$;
        """,
        "LOCK TABLE omninode_internal.events IN ACCESS EXCLUSIVE MODE;",
        "LOCK omninode_internal.events IN ACCESS EXCLUSIVE MODE;",
        "REINDEX TABLE omninode_internal.events;",
        "VACUUM omninode_internal.events;",
        "ANALYZE omninode_internal.events;",
        "CLUSTER omninode_internal.events;",
        "CLUSTER (VERBOSE) omninode_internal.events;",
    ],
)
def test_valid_postgresql_target_forms_emit_ownership_requirements(
    statement: str,
) -> None:
    requirements = application_database_sql_target_requirements(statement, _TOPOLOGY)

    assert any(
        requirement.location == ("omninode_internal", "events")
        for requirement in requirements
    )


def test_select_into_emits_an_exact_created_table_identity() -> None:
    sql = "SELECT * INTO omninode_internal.events_copy FROM omninode_internal.events;"

    assert tuple(
        (identity.schema, identity.name, identity.kind.value)
        for identity in application_database_created_catalog_identities(sql)
    ) == (("omninode_internal", "events_copy", "table"),)


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
            "LOCK TABLE ONLY omninode_internal.events, omninode_internal.runtime_state "
            "IN ACCESS EXCLUSIVE MODE;",
            {("omninode_internal", "events"), ("omninode_internal", "runtime_state")},
        ),
        (
            "REINDEX (VERBOSE) TABLE CONCURRENTLY omninode_internal.events;",
            {("omninode_internal", "events")},
        ),
        (
            'REINDEX (TABLESPACE "fast)tier") TABLE omninode_internal.events;',
            {("omninode_internal", "events")},
        ),
        (
            "VACUUM (ANALYZE, VERBOSE) omninode_internal.events (payload), "
            "omninode_internal.runtime_state;",
            {("omninode_internal", "events"), ("omninode_internal", "runtime_state")},
        ),
        (
            "ANALYZE (SKIP_LOCKED TRUE) VERBOSE omninode_internal.events;",
            {("omninode_internal", "events")},
        ),
        (
            "CLUSTER VERBOSE omninode_internal.events USING events_payload_idx;",
            {("omninode_internal", "events")},
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
            "INTO omninode_internal.events_copy FROM omninode_internal.events;"
        ),
        (
            "WITH source AS (SELECT * FROM omninode_internal.events) "
            "SELECT 'INTO public.decoy' AS payload "
            "INTO omninode_internal.events_copy FROM source;"
        ),
    ],
)
def test_select_into_created_identity_ignores_literals_and_leading_ctes(
    statement: str,
) -> None:
    assert tuple(
        (identity.schema, identity.name, identity.kind.value)
        for identity in application_database_created_catalog_identities(statement)
    ) == (("omninode_internal", "events_copy", "table"),)


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


def test_boolean_check_literals_are_not_relation_targets() -> None:
    sql = """
    CREATE TABLE action_authorization_claim.nonce_claims (
        execute_enabled BOOLEAN NOT NULL,
        one_time_use BOOLEAN NOT NULL,
        CONSTRAINT ck_execute_disabled CHECK (execute_enabled IS FALSE),
        CONSTRAINT ck_one_time_use CHECK (one_time_use IS TRUE)
    );
    """

    violations = lint_application_database_sql(sql, _TOPOLOGY)

    assert "application relation target 'false' must be schema-qualified" not in (
        "\n".join(violations)
    )
    assert "application relation target 'true' must be schema-qualified" not in (
        "\n".join(violations)
    )


def test_created_function_identity_strips_argument_names() -> None:
    sql = """
    CREATE FUNCTION action_authorization_claim.claim_action_authorization(
        p_authorization_id TEXT,
        p_execute_enabled BOOLEAN,
        p_issued_at TIMESTAMPTZ
    )
    RETURNS void
    LANGUAGE plpgsql
    AS $function$
    BEGIN
        RETURN;
    END;
    $function$;
    """

    identities = application_database_created_catalog_identities(sql)

    assert {
        identity.function_signature
        for identity in identities
        if identity.name == "claim_action_authorization"
    } == {"(TEXT, BOOLEAN, TIMESTAMPTZ)"}


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
        "REINDEX INDEX omninode_internal.events_payload_idx;",
        "REINDEX SCHEMA omninode_internal;",
        "REINDEX DATABASE omnidash_analytics;",
        "REINDEX SYSTEM omnidash_analytics;",
        'REINDEX (TABLESPACE "fast)tier") INDEX omninode_internal.events_payload_idx;',
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
        "GRANT USAGE ON SCHEMA omninode_internal TO onex_api;",
        "REVOKE ALL ON FUNCTION omninode_internal.safe_report() FROM PUBLIC;",
    ],
)
def test_database_schema_and_principal_privilege_tokens_are_not_relations(
    statement: str,
) -> None:
    assert not lint_application_database_sql(statement, _TOPOLOGY)


@pytest.mark.parametrize(
    "statement",
    [
        'VACUUM omninode_internal."events""rogue";',
        'LOCK omninode_internal."events""rogue" IN ACCESS EXCLUSIVE MODE;',
        'REINDEX TABLE omninode_internal."events""rogue";',
        'ANALYZE omninode_internal."events""rogue";',
        'CLUSTER omninode_internal."events""rogue";',
    ],
)
def test_escaped_quoted_relation_identifiers_remain_exact(statement: str) -> None:
    requirements = application_database_sql_target_requirements(statement, _TOPOLOGY)

    assert any(
        requirement.location == ("omninode_internal", 'events"rogue')
        for requirement in requirements
    )
    assert all(
        requirement.location != ("omninode_internal", "events")
        for requirement in requirements
    )


def test_escaped_quoted_schema_identifier_cannot_alias_a_topology_schema() -> None:
    violations = lint_application_database_sql(
        'VACUUM "omninode_internal""rogue".events;',
        _TOPOLOGY,
    )

    assert "unknown topology schema" in "\n".join(violations)


def test_select_into_preserves_escaped_quoted_created_identity() -> None:
    sql = 'SELECT * INTO omninode_internal."events_copy""rogue" FROM omninode_internal.events;'

    with pytest.raises(ValueError, match="name"):
        application_database_created_catalog_identities(sql)


@pytest.mark.parametrize(
    "statement",
    [
        "VACUUM omninode_internal.eventsé;",
        "LOCK omninode_internal.eventsé IN ACCESS EXCLUSIVE MODE;",
        "REINDEX TABLE omninode_internal.eventsé;",
        "ANALYZE omninode_internal.eventsé;",
        "CLUSTER omninode_internal.eventsé;",
    ],
)
def test_unicode_unquoted_relation_identifiers_remain_exact(statement: str) -> None:
    requirements = application_database_sql_target_requirements(statement, _TOPOLOGY)

    assert any(
        requirement.location == ("omninode_internal", "eventsé")
        for requirement in requirements
    )
    assert all(
        requirement.location != ("omninode_internal", "events")
        for requirement in requirements
    )


def test_unicode_escape_identifiers_fail_closed_explicitly() -> None:
    violations = lint_application_database_sql(
        r'VACUUM U&"omninode_internal".events;',
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
        "VACUUM /* omninode_internal.events /* nested */ omninode_internal.events */ public.events;",
        "LOCK TABLE /* omninode_internal.events /* nested */ omninode_internal.events */ public.events;",
        "REINDEX /* omninode_internal.events /* nested */ omninode_internal.events */ TABLE public.events;",
        "ANALYZE /* omninode_internal.events /* nested */ omninode_internal.events */ public.events;",
        "CLUSTER /* omninode_internal.events /* nested */ omninode_internal.events */ public.events;",
        (
            "SELECT * INTO /* omninode_internal.events_copy /* nested */ "
            "omninode_internal.events_copy */ public.events_copy FROM omninode_internal.events;"
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
        "SELECT * FROM omninode_internal.events /* unclosed",
        _TOPOLOGY,
    )

    assert "unterminated block comment" in "\n".join(violations)


def test_escape_string_quote_cannot_mask_a_real_public_target() -> None:
    # OMN-17887: a masked target must still be seen both where the lint refuses
    # it (the retired `tenant` schema) and where it is held to ownership
    # (`public`, the TENANT domain's schema).
    violations = lint_application_database_sql(
        "SELECT E'foo\\'bar' FROM tenant.events;",
        _TOPOLOGY,
    )
    requirements = application_database_sql_target_requirements(
        "SELECT E'foo\\'bar' FROM public.events;",
        _TOPOLOGY,
    )

    assert "'tenant.events' uses unknown topology schema" in "\n".join(violations)
    assert any(
        requirement.location == ("public", "events") for requirement in requirements
    )


def test_quote_inside_delimited_identifier_cannot_mask_a_real_public_target() -> None:
    # OMN-17887: see test_escape_string_quote_cannot_mask_a_real_public_target.
    violations = lint_application_database_sql(
        'SELECT 1 AS "foo\'bar" FROM tenant.events;',
        _TOPOLOGY,
    )
    requirements = application_database_sql_target_requirements(
        'SELECT 1 AS "foo\'bar" FROM public.events;',
        _TOPOLOGY,
    )

    assert "'tenant.events' uses unknown topology schema" in "\n".join(violations)
    assert any(
        requirement.location == ("public", "events") for requirement in requirements
    )


def test_select_into_ignores_into_keyword_inside_a_delimited_alias() -> None:
    # OMN-17887: the refused real target is now the retired `tenant` schema; a
    # `public` INTO target is a created identity held to the ownership census.
    sql = (
        'SELECT 1 AS "INTO omninode_internal.events_copy" '
        "INTO tenant.events_copy FROM omninode_internal.events;"
    )
    public_sql = (
        'SELECT 1 AS "INTO omninode_internal.events_copy" '
        "INTO public.events_copy FROM omninode_internal.events;"
    )

    assert "'tenant.events_copy' uses unknown topology schema" in "\n".join(
        lint_application_database_sql(sql, _TOPOLOGY)
    )
    assert tuple(
        (identity.schema, identity.name, identity.kind.value)
        for identity in application_database_created_catalog_identities(sql)
    ) == (("tenant", "events_copy", "table"),)
    assert tuple(
        (identity.schema, identity.name, identity.kind.value)
        for identity in application_database_created_catalog_identities(public_sql)
    ) == (("public", "events_copy", "table"),)


def test_escape_string_before_select_into_preserves_the_created_target() -> None:
    sql = "SELECT E'foo\\'bar' INTO omninode_internal.events_copy FROM omninode_internal.events;"

    assert tuple(
        (identity.schema, identity.name, identity.kind.value)
        for identity in application_database_created_catalog_identities(sql)
    ) == (("omninode_internal", "events_copy", "table"),)


# ---------------------------------------------------------------------------
# OMN-15361: a view body may open its own WITH clause. The CTE names live past
# the `AS`, so a leading-WITH-only parse never collected them and every later
# reference to one was misread as an unqualified application relation.
# ---------------------------------------------------------------------------

_VIEW_BODY_CTE_STATEMENTS: tuple[tuple[str, str], ...] = (
    (
        "plain",
        "CREATE VIEW omninode_internal.v AS WITH totals AS (SELECT 1 AS n) "
        "SELECT totals.n FROM totals;",
    ),
    (
        "or-replace",
        "CREATE OR REPLACE VIEW omninode_internal.v AS WITH totals AS (SELECT 1 AS n) "
        "SELECT totals.n FROM totals;",
    ),
    (
        "materialized",
        "CREATE MATERIALIZED VIEW omninode_internal.v AS WITH totals AS (SELECT 1 AS n) "
        "SELECT totals.n FROM totals;",
    ),
    (
        "recursive-cte",
        "CREATE VIEW omninode_internal.v AS WITH RECURSIVE walk AS ("
        "SELECT 1 AS n UNION ALL SELECT n + 1 FROM walk WHERE n < 5) "
        "SELECT walk.n FROM walk;",
    ),
    (
        "column-list",
        "CREATE VIEW omninode_internal.v (n) AS WITH totals AS (SELECT 1 AS n) "
        "SELECT totals.n FROM totals;",
    ),
    (
        "security-invoker-option-list",
        "CREATE OR REPLACE VIEW omninode_internal.v WITH (security_invoker = true) AS "
        "WITH totals AS (SELECT 1 AS n) SELECT totals.n FROM totals;",
    ),
    (
        "chained-ctes-cross-joined",
        "CREATE OR REPLACE VIEW omninode_internal.v AS WITH totals AS (SELECT 1 AS n), "
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

    OMN-17887: these are TENANT-domain relations, so they are qualified with
    `public` (the TENANT domain's schema); the retired `tenant` schema this
    reduction originally named would now fail as an unknown topology schema.
    """
    sql = """
CREATE OR REPLACE VIEW public.projection_delegation_quality_gate AS
WITH totals AS (
    SELECT COALESCE(AVG(actual_score), 0)::float AS avg_actual_score
    FROM public.delegation_events
),
failure_categories AS (
    SELECT COALESCE(
        jsonb_agg(
            jsonb_build_object('category', quality_gate_detail)
            ORDER BY quality_gate_detail DESC
        ),
        '[]'::jsonb
    ) AS rows
    FROM public.delegation_events
),
tokens_by_model AS (
    SELECT COALESCE(jsonb_agg(jsonb_build_object('model', model)), '[]'::jsonb) AS rows
    FROM public.delegation_events
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
        # unmapped_events, a name deliberately NOT in the shared
        # physical-schema-mapping allowlist, so the CTE-scoping behavior
        # under test stays independent of the allowlist pass-through.
        # (delegation_events was in that allowlist via the tenant bridge
        # until OMN-17887 deleted it.)
        (
            "CREATE VIEW omninode_internal.v AS WITH totals AS (SELECT 1 AS n) "
            "SELECT * FROM totals CROSS JOIN unmapped_events;",
            "'unmapped_events' must be schema-qualified",
        ),
        # An unqualified relation inside a CTE body still fails.
        (
            "CREATE VIEW omninode_internal.v AS WITH totals AS "
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
            "CREATE VIEW omninode_internal.v AS WITH totals AS (SELECT 1 AS n) "
            "SELECT totals.n FROM totals; SELECT * FROM totals;",
            "'totals' must be schema-qualified",
        ),
        # A later CTE is not visible to an earlier sibling's body.
        (
            "CREATE VIEW omninode_internal.v AS WITH first_cte AS (SELECT * FROM second_cte), "
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
# UNTIL_OMN15359; the tenant half was deleted by OMN-17887) instead
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


def test_ledger_chain_is_enumerated_in_the_shared_allowlist() -> None:
    """OMN-16964: ledger_chain is internal but physically public until OMN-15359."""
    assert "ledger_chain" in INTERNAL_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359


@pytest.mark.parametrize(
    "statement",
    [
        "ALTER TABLE intent_classification_events ADD COLUMN agent_source text;",
        "ALTER TABLE public.intent_classification_events ADD COLUMN agent_source text;",
        "CREATE TABLE IF NOT EXISTS public.ledger_chain (correlation_id text);",
        "COMMENT ON TABLE public.ledger_chain IS 'OMN-16964';",
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
            "ALTER TABLE tenant.not_an_allowlisted_table ADD COLUMN x text;",
            "'tenant.not_an_allowlisted_table' uses unknown topology schema",
        ),
    ],
)
def test_non_allowlisted_table_in_public_still_fails(
    statement: str,
    expected: str,
) -> None:
    """The allowlist is a narrow, enumerated pass-through, never a blanket
    public/unqualified exemption -- a table absent from the shared allowlist
    must still be rejected exactly as before.

    OMN-17887: the second row used to be ``public.not_an_allowlisted_table``
    refused as "prohibited in public". `public` is now the TENANT domain's
    declared schema, so that target is held to exact ownership instead
    (test_non_allowlisted_public_table_is_held_to_exact_ownership) and this row
    now refuses the retired `tenant` schema."""
    assert expected in "\n".join(lint_application_database_sql(statement, _TOPOLOGY))


def test_non_allowlisted_public_table_is_held_to_exact_ownership() -> None:
    """OMN-17887: a public table absent from the allowlist is not waved through.

    It emits an exact ownership requirement, which the SQL gate refuses unless
    exactly one ownership declaration answers for it.
    """
    requirements = application_database_sql_target_requirements(
        "ALTER TABLE public.not_an_allowlisted_table ADD COLUMN x text;",
        _TOPOLOGY,
    )

    assert any(
        requirement.location == ("public", "not_an_allowlisted_table")
        for requirement in requirements
    )


def test_known_physically_public_sequence_passes_static_lint() -> None:
    """OMN-17447: sequence grants follow their owning table's physical schema."""
    assert (
        "capability_scores_id_seq"
        in APPLICATION_SEQUENCES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359
    )
    assert (
        lint_application_database_sql(
            "GRANT USAGE ON SEQUENCE public.capability_scores_id_seq "
            "TO tenant_projection_writer;",
            _TOPOLOGY,
        )
        == ()
    )


def test_unlisted_public_sequence_is_held_to_ownership_not_passed_through() -> None:
    """OMN-17887: `public` is the TENANT domain's declared schema, so an unlisted
    public sequence is no longer a lint refusal; it must still emit an exact
    ownership requirement, and the retired `tenant` schema is still refused."""
    requirements = application_database_sql_target_requirements(
        "GRANT USAGE ON SEQUENCE public.unowned_id_seq TO omninode_runtime;",
        _TOPOLOGY,
    )

    assert any(
        requirement.location == ("public", "unowned_id_seq")
        for requirement in requirements
    )
    assert "'tenant.unowned_id_seq' uses unknown topology schema" in "\n".join(
        lint_application_database_sql(
            "GRANT USAGE ON SEQUENCE tenant.unowned_id_seq TO omninode_runtime;",
            _TOPOLOGY,
        )
    )


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
