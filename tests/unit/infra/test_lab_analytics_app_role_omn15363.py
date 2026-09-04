# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-15363: the lab lane must reach omnidash_analytics as a non-BYPASSRLS role.

Operator ruling 15 (OMN-15379) designates the compose dev/lab lane the FORCE ROW
LEVEL SECURITY proving ground. Postgres exempts a table's OWNER and any role with
SUPERUSER or BYPASSRLS from row-level security unconditionally — FORCE included —
so a lane whose only connecting role is ``postgres`` proves nothing at all: the
policies never evaluate, and "zero RLS errors" is a false clean rather than
evidence.

Two artifacts have to agree for that lane to be able to prove anything, and this
module ratchets both:

* ``docker/migrations/forward/096_grant_role_omnidash_omnidash_analytics.sql``
  provisions the AUTHORIZATION (role attributes + least-privilege grants).
* ``docker/docker-compose.dev-lane.yml`` provisions the IDENTITY (which role the
  lane's analytics consumers actually connect as).

Static-only by design: it fires on hosts without Docker or Postgres, which is
where a silent revert would otherwise go unnoticed until the next lab readback.
Execution proof against a real cluster is the ticket's live readback table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]

MIGRATION = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "096_grant_role_omnidash_omnidash_analytics.sql"
)
ROLLBACK = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "rollback"
    / "rollback_096_grant_role_omnidash_omnidash_analytics.sql"
)
BASE_COMPOSE = REPO_ROOT / "docker" / "docker-compose.infra.yml"
DEV_LANE_OVERLAY = REPO_ROOT / "docker" / "docker-compose.dev-lane.yml"
NON_DEV_OVERLAYS = (
    REPO_ROOT / "docker" / "docker-compose.stability-test.yml",
    REPO_ROOT / "docker" / "docker-compose.prod.yml",
    REPO_ROOT / "docker" / "docker-compose.judge.yml",
)

ANALYTICS_DSN_KEY = "OMNIDASH_ANALYTICS_DB_URL"
APP_ROLE = "role_omnidash"

# The three tables carrying relforcerowsecurity on the lab lane (live readback
# 2026-07-31T03:33Z). Spelled out rather than derived: if the forced set changes,
# that is a decision someone must make here deliberately.
# Not an f-string: ruff's S608 (SQL-injection heuristic) fires on interpolated
# strings that look like queries, and this is an expected-substring literal in a
# static assertion, never a query that is executed.
ROLE_EXISTS_GUARD = (
    "IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'role_omnidash') THEN"
)

FORCED_TABLES = (
    "node_service_registry",
    "projection_delegation_inference_response_text",
    "savings_estimates",
)


def _executable_text() -> str:
    """The migration with every ``--`` comment line removed.

    The file is heavily commented and the comments quote the very statements
    some of these tests assert are ABSENT (``FORCE ROW LEVEL SECURITY``,
    ``PASSWORD``). Asserting absence against the raw text would therefore be
    unfalsifiable in one direction and vacuous in the other; strip the prose
    first and assert against what psql will actually execute.
    """
    return "\n".join(
        line
        for line in MIGRATION.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("--")
    )


def _executable_sql() -> str:
    """``_executable_text`` collapsed to single-spaced one-line form."""
    return " ".join(_executable_text().split())


def _construct_compose_value(loader: yaml.SafeLoader, node: yaml.Node) -> object:
    """Resolve a Docker Compose merge tag to the value it decorates.

    OMN-17562 gave the dev-lane overlay's three runtime services
    ``labels: !override`` (compose APPENDS label sequences, so a plain block
    leaves the base ``autoheal=true`` armed beside the new strict probe).
    ``yaml.safe_load`` raises ``ConstructorError`` on that tag, so every parse of
    this overlay needs the same tag support the stability-lane suite has carried
    since OMN-15217 — the tag is a compose MERGE directive and carries no
    meaning for what these assertions read.
    """
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    assert isinstance(node, yaml.ScalarNode)
    return loader.construct_scalar(node)


class _ComposeSafeLoader(yaml.SafeLoader):
    """Test-local YAML loader with Docker Compose tag support."""


_ComposeSafeLoader.add_constructor("!override", _construct_compose_value)


def _load_compose(path: Path) -> dict[str, Any]:
    doc = yaml.load(path.read_text(encoding="utf-8"), Loader=_ComposeSafeLoader)  # noqa: S506
    assert isinstance(doc, dict)
    return doc


def _services_with_analytics_dsn(path: Path) -> set[str]:
    doc = _load_compose(path)
    services = doc.get("services") or {}
    return {
        name
        for name, cfg in services.items()
        if isinstance(cfg, dict) and ANALYTICS_DSN_KEY in (cfg.get("environment") or {})
    }


def _dev_lane_dsn(service: str) -> str:
    doc = _load_compose(DEV_LANE_OVERLAY)
    value = doc["services"][service]["environment"][ANALYTICS_DSN_KEY]
    assert isinstance(value, str)
    return value


# ---------------------------------------------------------------------------
# Migration 095 — authorization
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_migration_exists_with_a_rollback() -> None:
    assert MIGRATION.is_file(), f"missing forward migration: {MIGRATION}"
    assert ROLLBACK.is_file(), f"missing rollback: {ROLLBACK}"


@pytest.mark.unit
def test_migration_creates_the_role_without_rls_exempting_flags() -> None:
    """NOSUPERUSER + NOBYPASSRLS are the whole point; NOLOGIN is create-time only."""
    sql = _executable_sql()

    assert f"CREATE ROLE {APP_ROLE} WITH NOLOGIN NOSUPERUSER NOBYPASSRLS" in sql
    # Guarded so a role without CREATEROLE (the managed-instance case) can apply
    # the file: Postgres checks create-role privilege before name collision.
    assert ROLE_EXISTS_GUARD in sql
    assert f"ALTER ROLE {APP_ROLE} NOSUPERUSER NOBYPASSRLS" in sql


@pytest.mark.unit
def test_migration_never_grants_login_or_carries_a_credential() -> None:
    """094's invariant: the LOGIN + password attach is deployment-owned.

    Re-asserting NOLOGIN on a pre-existing role would REVOKE a deployment-owned
    attach; embedding a password would put credential material in the repo.
    """
    sql = _executable_sql()

    assert f"ALTER ROLE {APP_ROLE} WITH LOGIN" not in sql
    assert f"ALTER ROLE {APP_ROLE} NOLOGIN" not in sql
    # No password literal anywhere in what psql executes. The create-time NOLOGIN
    # default is the only mention of login state, and it is inside CREATE ROLE.
    assert "PASSWORD '" not in sql.upper()
    assert "PASSWORD $" not in sql.upper()


@pytest.mark.unit
def test_migration_grants_connect_and_schema_usage_but_not_create() -> None:
    """CREATE on the schema would let the app role OWN tables — owners bypass RLS.

    It is equally not REVOKEd: role_omnidash is the DDL principal on cloud RDS
    (OMN-15335), so a blanket revoke here would break the cloud migration path.
    """
    sql = _executable_sql()

    assert f"GRANT CONNECT ON DATABASE omnidash_analytics TO {APP_ROLE}" in sql
    assert f"GRANT USAGE ON SCHEMA public TO {APP_ROLE}" in sql
    assert f"GRANT USAGE, CREATE ON SCHEMA public TO {APP_ROLE}" not in sql
    assert f"REVOKE CREATE ON SCHEMA public FROM {APP_ROLE}" not in sql


@pytest.mark.unit
def test_migration_switches_to_the_analytics_database() -> None:
    """The forward runner applies docker/*.sql against POSTGRES_DB (omnibase_infra).

    Without the directive every table grant below would land on the wrong
    database and the migration would record itself as applied anyway.
    """
    raw = _executable_text()

    assert "\\connect omnidash_analytics" in raw
    connect_at = raw.index("\\connect omnidash_analytics")
    # Cluster-wide statements before the switch, database-scoped ones after it.
    assert raw.index("GRANT CONNECT ON DATABASE") < connect_at
    assert raw.index("GRANT USAGE ON SCHEMA public") > connect_at


@pytest.mark.unit
@pytest.mark.parametrize("table", FORCED_TABLES)
def test_forced_tables_are_granted_by_name_and_least_privilege(table: str) -> None:
    """Named grants, not just the blanket one: these are the tables under test.

    SELECT/INSERT/UPDATE mirrors node migration 0027's writer set. A projection
    writer upserts; it does not reshape the table, so DELETE/TRUNCATE/REFERENCES/
    TRIGGER are deliberately absent from the named grant.
    """
    sql = _executable_sql()

    assert table in sql
    assert "GRANT SELECT, INSERT, UPDATE ON public.%I TO role_omnidash" in sql
    assert f"GRANT ALL ON public.{table}" not in sql
    assert f"ALTER TABLE public.{table} OWNER TO {APP_ROLE}" not in sql


@pytest.mark.unit
def test_forced_tables_are_re_narrowed_after_the_blanket_grant() -> None:
    """The named grant must be true of the RESULTING STATE, not just the statement.

    Step 6 grants DML ``ON ALL TABLES IN SCHEMA public`` so the lane's other ~52
    projections keep working — and that is a SUPERSET of step 5's named
    SELECT/INSERT/UPDATE, silently re-adding DELETE to the three tables step 5
    excluded it from. Proven live on the lab lane 2026-07-31T03:56Z before the
    re-narrow block existed: role_table_grants read DELETE,INSERT,SELECT,UPDATE
    on all three.

    Both the statement AND its position are asserted: a REVOKE placed before the
    blanket GRANT is overwritten by it and the file would read correct while
    doing nothing.
    """
    sql = _executable_sql()
    raw = _executable_text()

    assert (
        "REVOKE DELETE, TRUNCATE, REFERENCES, TRIGGER ON public.%I FROM role_omnidash"
        in sql
    )
    assert raw.index("ON ALL TABLES IN SCHEMA public TO role_omnidash") < raw.index(
        "REVOKE DELETE, TRUNCATE, REFERENCES, TRIGGER"
    )


@pytest.mark.unit
def test_migration_changes_no_rls_or_force_state() -> None:
    """This is an authorization change. The fenced FORCE rollout is not its business."""
    sql = _executable_sql().upper()

    # Asserted as "issues no DDL against any table" rather than as a search for
    # the literal "ROW LEVEL SECURITY": that phrase appears inside a RAISE
    # message, so a search for it would fail on a prose reflow while catching
    # nothing real. There is no ALTER TABLE in this file at all.
    assert "ALTER TABLE" not in sql
    assert "CREATE POLICY" not in sql
    assert "DROP POLICY" not in sql
    assert "ALTER POLICY" not in sql
    assert "OWNER TO" not in sql


@pytest.mark.unit
def test_migration_asserts_the_ownership_and_flag_post_conditions() -> None:
    """Grants are not the isolation control — ownership and the two flags are.

    Severities differ on purpose (the OMN-15351 split): the flags are set by this
    same file, so a wrong value is FATAL; table ownership is decided at each
    lane's provisioning seam (role_omnidash owns part of the RDS schema by
    design), so it is a WARNING that names every table it found.
    """
    sql = _executable_sql()

    assert "pg_get_userbyid(c.relowner) = 'role_omnidash'" in sql
    assert f"RAISE WARNING '{APP_ROLE} OWNS RLS-covered table(s)" in sql
    assert f"RAISE EXCEPTION '{APP_ROLE} OWNS RLS-covered table(s)" not in sql
    assert f"RAISE EXCEPTION '{APP_ROLE} still carries rolsuper" in sql


# ---------------------------------------------------------------------------
# docker-compose.dev-lane.yml — identity
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_every_base_analytics_consumer_is_repointed_on_the_lab_lane() -> None:
    """Derived from the base file, so a NEW consumer cannot silently escape.

    A partial cutover proves nothing: one service left on the superuser DSN can
    open a BYPASSRLS session against the same tables the moment its dispatch path
    changes, and the lane is back to a false clean.
    """
    base_consumers = _services_with_analytics_dsn(BASE_COMPOSE)
    assert base_consumers, "base compose declares no analytics consumers — parse bug"

    overridden = _services_with_analytics_dsn(DEV_LANE_OVERLAY)
    missing = base_consumers - overridden
    assert not missing, (
        f"services still resolve the base (superuser) analytics DSN on the lab "
        f"lane: {sorted(missing)}. Add them to docker-compose.dev-lane.yml."
    )


@pytest.mark.unit
def test_lab_lane_dsn_names_the_app_role_and_never_the_superuser() -> None:
    for service in sorted(_services_with_analytics_dsn(DEV_LANE_OVERLAY)):
        dsn = _dev_lane_dsn(service)
        assert dsn.startswith(f"postgresql://{APP_ROLE}:"), (
            f"{service}: lab-lane analytics DSN must connect as {APP_ROLE}, got {dsn!r}"
        )
        assert "postgresql://postgres:" not in dsn
        assert "POSTGRES_USER" not in dsn
        assert "POSTGRES_PASSWORD" not in dsn
        assert dsn.endswith("@postgres:5432/omnidash_analytics")


@pytest.mark.unit
def test_lab_lane_password_is_fail_closed_not_defaulted() -> None:
    """``:-`` here would render a DSN with an empty password and fail at connect
    time on the proving lane — or, if someone "fixed" it by defaulting to the
    superuser credentials, restore the exact bypass this ticket removes."""
    dsn = _dev_lane_dsn("projection-api")

    assert "${ROLE_OMNIDASH_PASSWORD:?" in dsn
    assert "${ROLE_OMNIDASH_PASSWORD:-" not in dsn
    assert "${ROLE_OMNIDASH_PASSWORD}" not in dsn


@pytest.mark.unit
def test_no_other_lane_is_repointed() -> None:
    """prod, stability-test and judge keep their base DSN.

    The lab lane is the only authorized FORCE proving ground; repointing another
    lane's connection identity is a live runtime change to a lane this ticket has
    no grant for.
    """
    for overlay in NON_DEV_OVERLAYS:
        text = overlay.read_text(encoding="utf-8")
        assert APP_ROLE not in text, (
            f"{overlay.name} mentions {APP_ROLE}; only the dev/lab overlay may "
            "repoint the analytics connection identity"
        )
        # judge DOES declare its own OMNIDASH_ANALYTICS_DB_URL (it rebuilds the
        # whole runtime env block rather than inheriting the base anchor), so the
        # invariant is not "never mentions the key" — it is "whatever it declares
        # still resolves the POSTGRES_USER identity, not the app role".
        for line in text.splitlines():
            if line.strip().startswith(f"{ANALYTICS_DSN_KEY}:"):
                assert "${POSTGRES_USER" in line, (
                    f"{overlay.name} declares {ANALYTICS_DSN_KEY} without the "
                    f"POSTGRES_USER identity: {line.strip()!r}"
                )


@pytest.mark.unit
def test_base_compose_still_carries_the_superuser_default() -> None:
    """The RED control for the two tests above.

    If someone moves the app-role DSN into the base file, the lab-lane assertions
    would keep passing while stability-test, judge and prod silently inherit a
    connection identity change nobody approved. Pin the base to what it is.
    """
    base = yaml.safe_load(BASE_COMPOSE.read_text(encoding="utf-8"))
    dsn = base["services"]["projection-api"]["environment"][ANALYTICS_DSN_KEY]

    assert "${POSTGRES_USER:-postgres}" in dsn
    assert APP_ROLE not in dsn
