# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The identity schema seam on a fresh omninode_cloud (OMN-26121, T14 carrying T6's minimum).

Static contract tests over the seam files, the runner's lane `case` and the compose
indicator. The live proof (an empty scratch Postgres, a pre-expiry warm copy, a
drifted copy) is a lab run recorded in the pull request, not a skipped test.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
SEAMS = ROOT / "docker" / "migrations" / "seams"
CREATE = SEAMS / "tenant_identity_schema.sql"
READBACK = SEAMS / "tenant_identity_schema_readback.sql"
EXPIRY = SEAMS / "tenant_api_keys_expiry.sql"
RUNNER = ROOT / "scripts" / "run-forward-migrations.sh"
DOCKER = ROOT / "docker"
INDICATOR = "ONEX_MIGRATION_GATEWAY_LANE"
STABILITY = "stability" + "-test"
STABILITY_VALUE = STABILITY.replace("-", "_")
COMPOSE_STABILITY = f"docker-compose.{STABILITY}.yml"
ENTRY = SEAMS / "tenant_identity_gateway_lane.sql"
GRANTS = SEAMS / "gateway_runtime_grants.sql"

TABLES = (
    "tenants",
    "plan_tiers",
    "plan_entitlements",
    "tenant_plans",
    "tenant_api_keys",
    "tenant_event_outbox",
    "gateway_workflows",
    "gateway_rate_limit_events",
)


def _sql(path: Path) -> str:
    """The seam text with `--` comment lines removed."""
    return "\n".join(
        line
        for line in path.read_text().splitlines()
        if not line.lstrip().startswith("--")
    )


def test_seam_files_exist_with_spdx_headers() -> None:
    for path in (CREATE, READBACK, EXPIRY, ENTRY):
        assert path.is_file(), f"{path.relative_to(ROOT)} is missing"
        head = path.read_text().splitlines()[:2]
        assert head[0].startswith("-- SPDX-FileCopyrightText:"), path.name
        assert head[1] == "-- SPDX-License-Identifier: MIT", path.name


@pytest.mark.parametrize("table", TABLES)
def test_create_step_creates_each_table_only_when_absent(table: str) -> None:
    text = _sql(CREATE)
    assert re.search(rf"CREATE TABLE public\.{table}\b", text), table
    assert f"to_regclass('public.{table}') IS NULL" in text, (
        f"{table}: creation must be gated on the table being absent so a warm volume is never altered"
    )


def test_create_step_never_alters_drops_or_deletes() -> None:
    # The one ALTER is the second, identical UNIQUE constraint the warm volume carries, added
    # to the tenants table inside the block that has just created it (CREATE TABLE merges it).
    added = "ALTER TABLE public.tenants ADD CONSTRAINT tenants_tenant_slug_unique UNIQUE (tenant_slug);"
    text = _sql(CREATE)
    assert text.count(added) == 1
    block_start = text.index("\\if :new_tenants")
    assert block_start < text.index(added) < text.index("\\endif", block_start)
    remaining = text.replace(added, "").upper()
    for token in (
        "ALTER TABLE",
        "DROP ",
        "TRUNCATE",
        "DELETE FROM",
        "UPDATE ",
        "GRANT ",
        "CREATE ROLE",
    ):
        assert token not in remaining, f"the create step must not contain {token!r}"


def test_create_step_leaves_expires_at_to_the_expiry_seam() -> None:
    assert "expires_at" not in _sql(CREATE)


def test_create_step_gates_on_database_and_connects_to_it() -> None:
    text = _sql(CREATE)
    assert "pg_database WHERE datname = 'omninode_cloud'" in text
    assert "\\connect omninode_cloud" in text
    assert (
        "skip  tenant identity schema (database omninode_cloud absent on this lane)"
        in text
    )


def test_create_step_seeds_the_default_plan_only_into_an_empty_table() -> None:
    text = _sql(CREATE)
    assert "'discovery'" in text
    assert "NOT EXISTS (SELECT 1 FROM public.plan_tiers)" in text


def test_create_step_outbox_id_draws_from_its_own_sequence() -> None:
    assert re.search(r"id\s+bigserial\s+PRIMARY KEY", _sql(CREATE))


def test_create_step_carries_the_chk_constraints_the_gateway_relies_on() -> None:
    text = _sql(CREATE)
    for name in (
        "chk_key_hash_valid",
        "chk_name_valid",
        "chk_revocation_consistency",
        "gateway_workflows_status_check",
        "uq_tenant_api_keys_tenant_id_name_active",
        "idx_tenant_plans_one_active_per_tenant",
    ):
        assert name in text, name


def test_expiry_seam_adds_the_column_and_partial_index_behind_a_gate() -> None:
    text = _sql(EXPIRY)
    assert "pg_database WHERE datname = 'omninode_cloud'" in text
    assert "to_regclass('public.tenant_api_keys') IS NOT NULL" in text
    assert (
        "ALTER TABLE public.tenant_api_keys ADD COLUMN IF NOT EXISTS expires_at timestamptz NULL"
        in text
    )
    assert re.search(
        r"CREATE INDEX IF NOT EXISTS idx_tenant_api_keys_expires_at\s+ON public\.tenant_api_keys \(expires_at\) WHERE is_active",
        text,
    )
    assert (
        "skip  tenant_api_keys expiry (database omninode_cloud absent on this lane)"
        in text
    )
    assert (
        "skip  tenant_api_keys expiry (table public.tenant_api_keys absent in omninode_cloud on this lane)"
        in text
    )
    assert "ok    tenant_api_keys.expires_at present" in text


def test_readback_declares_every_table_including_expires_at_and_raises() -> None:
    text = _sql(READBACK)
    for table in TABLES:
        assert f"{table}|" in text, f"readback declares no row for {table}"
    assert "tenant_api_keys|column|expires_at|timestamp with time zone|YES|" in text
    assert "idx_tenant_api_keys_expires_at" in text
    assert "pg_get_constraintdef" in text
    assert "pg_indexes" in text or "pg_get_indexdef" in text
    assert "RAISE EXCEPTION" in text
    assert "ok    tenant identity schema matches the declared one" in text
    assert "ALTER " not in text.upper()


def test_readback_skips_by_name_when_the_database_is_absent() -> None:
    assert (
        "skip  tenant identity schema readback (database omninode_cloud absent on this lane)"
        in _sql(READBACK)
    )


def test_runner_documents_the_indicator_beside_the_lane_one() -> None:
    text = RUNNER.read_text()
    header = text[: text.index("set -e")]
    assert INDICATOR in header
    assert header.index("ONEX_MIGRATION_LANE") < header.index(INDICATOR)


def test_runner_gains_no_code_line_for_the_indicator() -> None:
    # The runner is a baselined file that only shrinks: the lane gate is the entry seam's.
    code = [
        line
        for line in RUNNER.read_text().splitlines()
        if INDICATOR in line and not line.lstrip().startswith("#")
    ]
    assert code == []


def test_entry_seam_runs_create_then_expiry_then_readback_for_the_two_gateway_lanes() -> (
    None
):
    text = _sql(ENTRY)
    assert f"IN ('dev', '{STABILITY_VALUE}')" in text
    assert "\\getenv gateway_lane ONEX_MIGRATION_GATEWAY_LANE" in text
    order = [
        text.index("\\ir tenant_identity_schema.sql"),
        text.index("\\ir tenant_api_keys_expiry.sql"),
        text.index("\\ir tenant_identity_schema_readback.sql"),
    ]
    assert order == sorted(order)
    assert text.index("\\connect :identity_start_db") > order[-1], (
        "the session must return to the database it started on"
    )


def test_entry_seam_fails_closed_when_the_indicator_is_unset_or_unknown() -> None:
    text = _sql(ENTRY)
    assert "\\set gateway_lane ''" in text
    assert (
        "skip  tenant identity schema (ONEX_MIGRATION_GATEWAY_LANE unset: this lane does not serve the gateway)"
        in text
    )
    assert "unknown ONEX_MIGRATION_GATEWAY_LANE" in text


def test_grants_seam_includes_the_entry_before_its_first_statement() -> None:
    text = _sql(GRANTS)
    assert "\\ir tenant_identity_gateway_lane.sql" in text
    assert text.index("\\ir tenant_identity_gateway_lane.sql") < text.index(
        "role_present"
    )


def test_runner_still_applies_the_grants_file_inside_the_slot_guarded_branch() -> None:
    text = RUNNER.read_text()
    guard = text.index('slot_skip_role_seam "login-only role grants (OMN-18060)"')
    call = text.index(
        "gateway_grants_file=/migrations/seams/gateway_runtime_grants.sql"
    )
    assert guard < call, (
        "the call must stay in the non-slot branch, so no slot runs the seams"
    )


def _forward_migration_block(path: Path) -> str:
    """The text of the compose file's `forward-migration:` service (a two-space-indented key)."""
    text = path.read_text()
    match = re.search(
        r"^  forward-migration:\n((?:    .*\n|\s*\n)*)", text, re.MULTILINE
    )
    assert match, f"{path.name} has no forward-migration service"
    return match.group(1)


@pytest.mark.parametrize(
    ("compose", "value"),
    [
        ("docker-compose.dev-lane.yml", "dev"),
        (COMPOSE_STABILITY, STABILITY_VALUE),
    ],
)
def test_the_two_gateway_lanes_carry_the_indicator(compose: str, value: str) -> None:
    block = _forward_migration_block(DOCKER / compose)
    assert re.search(rf"^\s+{INDICATOR}: {value}\s*$", block, re.MULTILINE), compose


def test_no_other_compose_file_and_not_the_catalog_carries_the_indicator() -> None:
    carriers = {
        p.name for p in DOCKER.glob("docker-compose*.yml") if INDICATOR in p.read_text()
    }
    assert carriers == {
        "docker-compose.dev-lane.yml",
        COMPOSE_STABILITY,
    }
    for path in (DOCKER / "catalog").rglob("*.yaml"):
        assert INDICATOR not in path.read_text(), path
