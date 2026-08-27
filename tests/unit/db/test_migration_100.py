# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit checks for migration 100 gateway_link_health + liveness-threshold drift.

OMN-15570 (G3) hardcodes the ``gateway_link_health_status`` view's health
thresholds (``max_silence_window_seconds`` / ``lag_threshold_messages`` /
``lag_threshold_seconds``) as SQL literals rather than reading them from the
contract at migration/config load time -- there is no existing repo pattern
for materializing a value into a static, already-applied SQL view at
migration time (unlike the runtime config loader's
``_materialize_contract_*`` functions, which run at process start against a
YAML file, a view body is baked in once at ``CREATE OR REPLACE VIEW`` time).
This is the 4th independent copy of these three numbers (contract.yaml,
``ModelGatewayForwarderConfig`` defaults, deploy YAML, and now this view) --
tracked as gateway config triplication under OMN-15762.

Per that ticket's disclosed scope, this test is the fail-closed guard: it
pins the SQL view's literals to the contract's declared values and fails if
they diverge, so an operator changing the contract's liveness block without
updating the migration finds out here instead of via a silent, wrong
``HEALTHY``/``UNHEALTHY`` verdict in production.
"""

import hashlib
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]

# OMN-16759: the relation moved off the flat loop. The flat loop connects only to
# omnibase_infra, which has no omninode_internal schema and whose migration role
# cannot create one -- so the flat file's CREATE SCHEMA failed with "permission
# denied for database omnibase_infra" and blocked every staging deploy. The DDL
# now lives on the node loop, which connects to the application database
# (omnidash_analytics) where that schema exists. These constants follow it; the
# superseded flat file is asserted separately below.
FORWARD = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "nodes"
    / "node_gateway_link_health_write_effect"
    / "0001_create_gateway_link_health.sql"
)
ROLLBACK = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "rollback"
    / "rollback_node_gateway_link_health_0001.sql"
)
SUPERSEDED_FLAT = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "100_create_gateway_link_health.sql"
)
CONTRACT = (
    REPO_ROOT
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_bus_forwarder_effect"
    / "contract.yaml"
)


def _contract_liveness() -> dict[str, int]:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    liveness = contract["config"]["gateway_forwarder"]["liveness"]
    return {
        "max_silence_window_seconds": int(liveness["max_silence_window_seconds"]),
        "lag_threshold_messages": int(liveness["lag_threshold_messages"]),
        "lag_threshold_seconds": int(liveness["lag_threshold_seconds"]),
    }


def test_100_forward_creates_table_and_view() -> None:
    sql = FORWARD.read_text(encoding="utf-8").lower()

    assert "create table if not exists omninode_internal.gateway_link_health" in sql
    assert "create or replace view omninode_internal.gateway_link_health_status" in sql
    assert "primary key (tenant_id)" in sql


def test_100_rollback_drops_view_and_table() -> None:
    sql = ROLLBACK.read_text(encoding="utf-8").lower()

    assert "drop view if exists omninode_internal.gateway_link_health_status" in sql
    assert "drop table if exists omninode_internal.gateway_link_health" in sql


def test_100_view_liveness_thresholds_match_contract_omn_15762() -> None:
    """Fail-closed drift guard (OMN-15762): the migration's hardcoded
    thresholds must match node_bus_forwarder_effect/contract.yaml's
    liveness block. This is the 4th copy of these numbers (contract,
    ModelGatewayForwarderConfig defaults, deploy YAML, this view) -- see
    module docstring. A mismatch here means the contract changed and the
    view was not updated to match, which would silently mis-classify
    gateway health.
    """
    sql = FORWARD.read_text(encoding="utf-8")
    contract = _contract_liveness()

    silence_match = re.search(
        r"NOW\(\)\s*-\s*last_seen_at\s*>\s*INTERVAL\s*'(\d+)\s*seconds'",
        sql,
    )
    assert silence_match is not None, (
        "expected an `INTERVAL '<n> seconds'` silence-window comparison in "
        "the gateway_link_health migration -- update this regex if the "
        "view's SQL shape changed"
    )
    assert int(silence_match.group(1)) == contract["max_silence_window_seconds"], (
        "the gateway_link_health migration's silence-window literal has "
        "drifted from node_bus_forwarder_effect/contract.yaml's "
        "gateway_forwarder.liveness.max_silence_window_seconds (OMN-15762 "
        "4th-copy class) -- update the migration's hardcoded literal (a new "
        "migration, since this one is declared in the canonical node ledger) "
        "to match the "
        "contract, or update the contract to match an intentional migration "
        "change"
    )

    lag_messages_match = re.search(
        r"lag_messages\s+IS\s+NOT\s+NULL\s+AND\s+lag_messages\s*>\s*(\d+)",
        sql,
    )
    assert lag_messages_match is not None
    assert int(lag_messages_match.group(1)) == contract["lag_threshold_messages"], (
        "the gateway_link_health migration's lag_threshold_messages "
        "literal has drifted from the contract (OMN-15762 4th-copy class)"
    )

    lag_seconds_match = re.search(
        r"lag_seconds\s+IS\s+NOT\s+NULL\s+AND\s+lag_seconds\s*>\s*(\d+)",
        sql,
    )
    assert lag_seconds_match is not None
    assert int(lag_seconds_match.group(1)) == contract["lag_threshold_seconds"], (
        "the gateway_link_health migration's lag_threshold_seconds literal "
        "has drifted from the contract (OMN-15762 4th-copy class)"
    )


def test_100_view_ranks_self_reported_degraded_above_lag_and_below_stale() -> None:
    """Pin the CASE arm ORDER, not just its presence (OMN-15742/G2).

    The live-apply proof in
    tests/integration/migrations/test_100_gateway_link_health_verdicts.py is
    what actually executes these arms. This is the cheap always-runs guard for
    the case where that suite is skipped because Postgres tooling is absent:
    it fails if someone reorders the arms, since CASE stops at the first TRUE
    and the order IS the precedence.
    """
    sql = FORWARD.read_text(encoding="utf-8")
    # Anchor on the CASE expression itself. Splitting on the CREATE statement
    # is not enough: the header comment carries an example query containing
    # 'HEALTHY', which would be found ahead of the real arms.
    case_block = sql.split("    CASE\n", 1)[1].split("END AS health_status", 1)[0]

    stale = case_block.index("'UNHEALTHY'")
    self_reported = case_block.index("'DEGRADED_SELF_REPORTED'")
    lag = case_block.index("'DEGRADED_LAG'")
    healthy = case_block.index("'HEALTHY'")

    assert stale < self_reported < lag < healthy


def test_100_table_and_view_carry_the_self_reported_status_columns() -> None:
    sql = FORWARD.read_text(encoding="utf-8").lower()

    assert "reported_status text not null" in sql
    assert "consecutive_failures integer not null" in sql
    # Anything that is not 'active' is degraded -- an equality test against a
    # closed set would score an unrecognised future status HEALTHY by omission.
    assert "reported_status <> 'active'" in sql


# ---------------------------------------------------------------------------
# OMN-16759: the flat file is superseded, and the node-owned file asserts
# ---------------------------------------------------------------------------
def test_node_migration_asserts_the_schema_instead_of_creating_it() -> None:
    """RED against the bytes that blocked every staging deploy.

    ``CREATE SCHEMA IF NOT EXISTS omninode_internal`` needs CREATE on the
    DATABASE. Read live from the managed instance before this fix was written:
    ``has_database_privilege(role_omnibase_infra, omnibase_infra, CREATE)`` is
    false, and so is ``has_database_privilege(role_omnidash,
    omnidash_analytics, CREATE)`` -- NEITHER migration role can create a schema
    on that lane. `IF NOT EXISTS` does not help, because Postgres checks the
    privilege before it checks existence.

    The class-level gate over the whole corpus lives in
    ``tests/unit/db/test_migration_no_database_level_privilege_omn16759.py``.
    This pins the positive half for the file that carries the relation: it must
    ASSERT its schema through the OMN-16249 ``pg_catalog.pg_namespace`` probe.
    """
    sql = FORWARD.read_text(encoding="utf-8")
    statements = [
        line.strip()
        for line in sql.splitlines()
        if not line.strip().startswith("--") and "CREATE SCHEMA" in line.upper()
    ]

    assert not statements, (
        f"the gateway_link_health migration issues {statements} -- the exact "
        "statement OMN-16249 removed from 0005 and OMN-16759 removed from the "
        "flat 100. Assert the schema; never create it."
    )
    assert "pg_catalog.pg_namespace" in sql
    assert "nspname = 'omninode_internal'" in sql, (
        "the precondition must assert the schema this file's objects target, "
        "otherwise it proves nothing about them"
    )


def test_the_flat_100_creates_nothing_and_names_its_replacement() -> None:
    """The flat loop reaches only omnibase_infra, which has no
    omninode_internal schema (read live: schema count 0) and whose role cannot
    make one. 100 must therefore create nothing at all -- and must say where the
    relation went, so the next reader is not left hunting.
    """
    sql = SUPERSEDED_FLAT.read_text(encoding="utf-8")
    executable = [
        line.strip()
        for line in sql.splitlines()
        if line.strip() and not line.strip().startswith("--")
    ]

    assert not any(
        line.upper().startswith(("CREATE ", "ALTER ", "DROP ", "INSERT ", "GRANT "))
        for line in executable
    ), f"the superseded flat 100 still carries DDL/DML: {executable}"
    assert "nodes/node_gateway_link_health_write_effect/" in sql.replace(
        "\n--     ", ""
    ).replace("\n--   ", ""), "the superseded file must name its replacement path"


def test_the_superseded_flat_100_is_still_present() -> None:
    """Applied migration history is preserved, never deleted (OMN-15695).

    The compose lanes applied 100's original bytes -- their runner connects as
    the postgres superuser, so the CREATE SCHEMA succeeded there. Their ledger
    rows key on this filename, so the file stays.
    """
    assert SUPERSEDED_FLAT.is_file()


def test_the_node_migration_is_declared_in_the_canonical_ledger() -> None:
    """An undeclared node migration is one bootstrap.sql cannot resolve."""
    manifest = (
        REPO_ROOT
        / "docker"
        / "migrations"
        / "forward"
        / "_ledger"
        / "application-migrations.tsv"
    ).read_text(encoding="utf-8")
    artifact = (
        "nodes/node_gateway_link_health_write_effect/"
        "0001_create_gateway_link_health.sql"
    )

    rows = [line for line in manifest.splitlines() if line.startswith(f"{artifact}\t")]

    assert len(rows) == 1, f"expected exactly one declaration for {artifact}"
    fields = rows[0].split("\t")
    assert fields[3] == "omninode_internal", (
        "the declared domain must match the schema the SQL actually targets"
    )
    declared_checksum = fields[5]
    actual = hashlib.sha256(FORWARD.read_bytes()).hexdigest()
    assert declared_checksum == actual, (
        "declared checksum does not match the file on disk -- the forward "
        "runner FATALs with 'conflicting migration checksum' on this"
    )
