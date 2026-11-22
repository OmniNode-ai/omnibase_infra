# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26471 -- the least-grant ``tenant_identity_writer`` login on ``omninode_cloud``, as data.

The grants are the one list ``EXPECTED_GRANTS`` below. The seam
(``docker/migrations/seams/tenant_identity_writer_grants.sql``) states them twice, as
``GRANT`` statements and as the literal strings its readback compares the effective
privileges with; both are parsed here and each must equal the list. The deployment
topology model holds no column-scoped privilege, so this list is the declaration, written
once, in this file.

Every positive control edits a COPY of the seam text and expects the comparison to fire,
so a parse that matches nothing (and would therefore pass any list) cannot go unseen.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
SEAM = ROOT / "docker" / "migrations" / "seams" / "tenant_identity_writer_grants.sql"
GATEWAY_SEAM = ROOT / "docker" / "migrations" / "seams" / "gateway_runtime_grants.sql"
RUNNER = ROOT / "scripts" / "run-forward-migrations.sh"
COMPOSE_INFRA = ROOT / "docker" / "docker-compose.infra.yml"
COMPOSE_JUDGE = ROOT / "docker" / "docker-compose.judge.yml"
COMPOSE_DEV_LANE = ROOT / "docker" / "docker-compose.dev-lane.yml"
DEV_MANIFEST = ROOT / "docker" / "dev-lane-required-env.manifest.txt"
DEV_ENV_TEMPLATE = (
    ROOT / "scripts" / "deploy-agent" / "deploy" / "dev-instance.env.template"
)

ROLE = "tenant_identity_writer"
PASSWORD_VAR = "TENANT_IDENTITY_WRITER_PASSWORD"
DSN_VAR = "TENANT_IDENTITY_DATABASE_DSN"

# One grant per tuple: (kind, object, privilege, column or None).
EXPECTED_GRANTS: frozenset[tuple[str, str, str, str | None]] = frozenset(
    {
        ("database", "omninode_cloud", "CONNECT", None),
        ("schema", "public", "USAGE", None),
        ("table", "tenants", "SELECT", None),
        ("table", "tenants", "INSERT", None),
        *(
            ("column", "tenants", "UPDATE", c)
            for c in (
                "provisioning_state",
                "kc_patch_attempt_count",
                "kc_patch_last_error",
                "principal_id",
                "name",
            )
        ),
        ("table", "plan_tiers", "SELECT", None),
        ("table", "tenant_plans", "SELECT", None),
        ("table", "tenant_plans", "INSERT", None),
        ("table", "tenant_api_keys", "SELECT", None),
        ("table", "tenant_api_keys", "INSERT", None),
        ("column", "tenant_api_keys", "UPDATE", "is_active"),
        ("column", "tenant_api_keys", "UPDATE", "revoked_at"),
        ("table", "tenant_event_outbox", "SELECT", None),
        ("table", "tenant_event_outbox", "INSERT", None),
        *(
            ("column", "tenant_event_outbox", "UPDATE", c)
            for c in ("published_at", "attempts", "last_error")
        ),
        ("sequence", "tenant_event_outbox_id_seq", "USAGE", None),
    }
)

_GRANT = re.compile(
    r"^\s*GRANT\s+(?P<privs>.+?)\s+ON\s+(?P<target>.+?)\s+TO\s+\"tenant_identity_writer\"\s*;\s*$",
    re.MULTILINE,
)


def _seam_text() -> str:
    return SEAM.read_text(encoding="utf-8")


def parse_grants(text: str) -> set[tuple[str, str, str, str | None]]:
    """Every (kind, object, privilege, column) the seam's GRANT statements issue."""
    found: set[tuple[str, str, str, str | None]] = set()
    for m in _GRANT.finditer(text):
        privs, target = m.group("privs"), m.group("target")
        if target.startswith("DATABASE "):
            found.update(
                ("database", target.split()[1], p.strip(), None)
                for p in privs.split(",")
            )
        elif target.startswith("SCHEMA "):
            found.update(
                ("schema", target.split()[1], p.strip(), None) for p in privs.split(",")
            )
        elif target.startswith("SEQUENCE "):
            name = target.split()[1].removeprefix("public.")
            found.update(("sequence", name, p.strip(), None) for p in privs.split(","))
        else:
            name = target.split()[0].removeprefix("public.")
            column_form = re.fullmatch(r"(\w+)\s*\((.+)\)", privs)
            if column_form:
                found.update(
                    ("column", name, column_form.group(1), c.strip())
                    for c in column_form.group(2).split(",")
                )
            else:
                found.update(("table", name, p.strip(), None) for p in privs.split(","))
    return found


def derived_readback_strings() -> dict[str, str]:
    """The four literals the readback must compare with, derived from EXPECTED_GRANTS."""
    table_level = sorted((o, p) for k, o, p, _ in EXPECTED_GRANTS if k == "table")
    any_column = sorted(
        set(table_level)
        | {(o, "UPDATE") for k, o, _, _ in EXPECTED_GRANTS if k == "column"}
    )
    return {
        "table_set": ",".join(f"{o}:{p}" for o, p in table_level),
        "any_column_set": ",".join(f"{o}:{p}" for o, p in any_column),
        "update_columns": ",".join(
            sorted(f"{o}.{c}" for k, o, _, c in EXPECTED_GRANTS if k == "column")
        ),
        "sequence_set": ",".join(
            f"{o}:{p}" for k, o, p, _ in sorted(EXPECTED_GRANTS) if k == "sequence"
        ),
    }


def readback_strings(text: str) -> dict[str, str]:
    """The literals the seam's readback actually compares the four effective sets with."""
    found: dict[str, str] = {}
    for var in ("table_set", "any_column_set", "update_columns", "sequence_set"):
        m = re.search(rf"{var} IS DISTINCT FROM '([^']*)'", text)
        assert m, f"the readback does not compare {var} with a literal"
        found[var] = m.group(1)
    return found


def test_the_seam_exists_outside_the_ordered_corpus() -> None:
    assert SEAM.is_file()
    assert SEAM.parent.name == "seams"
    assert not list(
        (ROOT / "docker" / "migrations" / "forward").glob("*tenant_identity_writer*")
    )


def test_grant_statements_equal_the_declared_list_exactly() -> None:
    assert parse_grants(_seam_text()) == set(EXPECTED_GRANTS)


def test_the_declared_list_has_no_delete_truncate_or_ddl_privilege() -> None:
    privileges = {p for _, _, p, _ in EXPECTED_GRANTS}
    assert privileges <= {"CONNECT", "USAGE", "SELECT", "INSERT", "UPDATE"}
    assert not re.search(
        r"WITH GRANT OPTION",
        "\n".join(m.group(0) for m in _GRANT.finditer(_seam_text())),
    )


def test_the_readback_literals_equal_the_declared_list() -> None:
    assert readback_strings(_seam_text()) == derived_readback_strings()


def test_the_readback_raises_on_any_difference_and_on_delegation() -> None:
    text = _seam_text()
    block = text[text.index("$tenant_identity_readback$") :]
    assert "RAISE EXCEPTION" in block
    for var in ("table_set", "any_column_set", "update_columns", "sequence_set"):
        assert f"{var} IS DISTINCT FROM" in block
    assert "OR delegated <> ''" in block
    for fn in (
        "has_table_privilege",
        "has_any_column_privilege",
        "has_sequence_privilege",
        "has_schema_privilege",
        "has_database_privilege",
    ):
        assert re.search(rf"{fn}\([^)]*WITH GRANT OPTION'\)", block), fn
    for fn in (
        "has_column_privilege",
        "has_database_privilege",
        "has_schema_privilege",
    ):
        assert f"{fn}(" in block, fn
    assert re.search(r"NOT has_database_privilege\([^)]*'CONNECT'\)", block)
    assert re.search(r"has_schema_privilege\([^)]*'CREATE'\)", block)


def test_positive_control_a_dropped_grant_is_seen_by_the_parse() -> None:
    text = _seam_text()
    mutated = "\n".join(
        line
        for line in text.splitlines()
        if "GRANT SELECT ON public.plan_tiers" not in line
    )
    assert parse_grants(mutated) != set(EXPECTED_GRANTS)
    assert set(EXPECTED_GRANTS) - parse_grants(mutated) == {
        ("table", "plan_tiers", "SELECT", None)
    }


def test_positive_control_an_extra_grant_is_seen_by_the_parse() -> None:
    mutated = _seam_text().replace(
        "        COMMIT;",
        '        GRANT DELETE ON public.tenants TO "tenant_identity_writer";\n        COMMIT;',
        1,
    )
    assert parse_grants(mutated) - set(EXPECTED_GRANTS) == {
        ("table", "tenants", "DELETE", None)
    }


def test_positive_control_a_widened_readback_literal_is_seen() -> None:
    text = _seam_text()
    widened = text.replace(
        "tenants:INSERT,tenants:SELECT'",
        "tenants:DELETE,tenants:INSERT,tenants:SELECT'",
        1,
    )
    assert widened != text
    assert readback_strings(widened) != derived_readback_strings()


def test_the_seam_resets_before_it_grants_and_skips_by_name() -> None:
    text = _seam_text()
    assert text.index("REVOKE ALL ON ALL TABLES IN SCHEMA public") < text.index(
        "GRANT USAGE ON SCHEMA public"
    )
    assert "REVOKE ALL ON ALL SEQUENCES IN SCHEMA public" in text
    for skip in (
        "skip  tenant_identity_writer (TENANT_IDENTITY_WRITER_PASSWORD not set)",
        "skip  tenant_identity_writer (database omninode_cloud absent on this lane)",
        "skip  tenant_identity_writer grants (a declared table, column or sequence does not exist in omninode_cloud yet",
    ):
        assert skip in text, skip
    assert (
        "ok    tenant_identity_writer holds exactly the declared grants on omninode_cloud"
        in text
    )
    assert "ok    tenant_identity_writer LOGIN credential asserted" in text


def test_the_credential_is_minted_without_ever_being_printed_or_widened() -> None:
    text = _seam_text()
    assert "\\getenv writer_pw TENANT_IDENTITY_WRITER_PASSWORD" in text
    assert text.count(":'writer_pw'") == 4  # two format() calls, the hex and set tests
    echoes = [ln for ln in text.splitlines() if ln.lstrip().startswith("\\echo")]
    assert echoes and not any("writer_pw" in ln for ln in echoes)
    for attribute in (
        "LOGIN",
        "NOSUPERUSER",
        "NOBYPASSRLS",
        "NOCREATEDB",
        "NOCREATEROLE",
        "NOREPLICATION",
    ):
        assert attribute in text.split("CREATE ROLE", 1)[1].split("\\gexec", 1)[0]
    assert "rolsuper OR rolbypassrls" in text
    assert "is not hexadecimal" in text


def test_the_runner_applies_the_seam_in_the_gateway_grants_psql_run() -> None:
    lines = RUNNER.read_text(encoding="utf-8").splitlines()
    calls = [
        ln
        for ln in lines
        if "tenant_identity_writer_grants.sql" in ln and not ln.lstrip().startswith("#")
    ]
    assert len(calls) == 1
    assert (
        '-f "$gateway_grants_file" -f /migrations/seams/tenant_identity_writer_grants.sql'
        in calls[0]
    )
    assert "ON_ERROR_STOP=1" in calls[0]
    assert GATEWAY_SEAM.is_file()


def test_the_dev_lane_migration_service_receives_the_writer_password_fail_closed() -> (
    None
):
    dev = COMPOSE_DEV_LANE.read_text(encoding="utf-8")
    start = dev.index("\n  forward-migration:\n")
    block = dev[start : dev.index("\n    labels:", start)]
    assert f"{PASSWORD_VAR}: ${{{PASSWORD_VAR}:?" in block
    for base in (COMPOSE_INFRA, COMPOSE_JUDGE):
        assert PASSWORD_VAR not in base.read_text(encoding="utf-8")


def test_the_dev_lane_renders_the_dsn_fail_closed_and_no_base_file_names_it() -> None:
    dev = COMPOSE_DEV_LANE.read_text(encoding="utf-8")
    m = re.search(rf"^  {DSN_VAR}: \"(?P<dsn>[^\"]+)\"", dev, re.MULTILINE)
    assert m
    dsn = m.group("dsn")
    assert dsn.startswith(f"postgresql://{ROLE}:${{{PASSWORD_VAR}:?")
    assert dsn.endswith("@postgres:5432/omninode_cloud")
    assert "*dev_lane_tenant_identity_env" in dev
    for base in (COMPOSE_INFRA, COMPOSE_JUDGE):
        assert DSN_VAR not in base.read_text(encoding="utf-8")


def test_the_required_env_manifest_and_the_instance_template_name_the_password() -> (
    None
):
    manifest = DEV_MANIFEST.read_text(encoding="utf-8").splitlines()
    assert PASSWORD_VAR in manifest
    template = DEV_ENV_TEMPLATE.read_text(encoding="utf-8").splitlines()
    assert f"{PASSWORD_VAR}=" in template
