# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Who owns ``llm_call_metrics`` -- and why 031 cannot simply be retired.

OMN-15857 raised ``llm_call_metrics`` as a **double declaration**: the table is
created by a legacy flat migration
(``forward/031_create_llm_call_metrics_and_cost_aggregates.sql``) *and* by a
node-grammar migration
(``forward/nodes/node_projection_llm_cost/0001_create_llm_call_metrics.sql``).
Doctrine says nodes own their migrations, so the node declaration is canonical
and the legacy one should be retired in the manifest.

**It cannot be, and the reason is structural rather than a matter of taste.**
The two files do not compete for one object -- they create same-named tables in
two different databases, and ``run-forward-migrations.sh`` sends each to exactly
one of them:

* flat ``docker/NNN_*.sql`` migrations run against ``PGDB`` (``POSTGRES_DB``,
  compose: ``omnibase_infra``), the service database;
* node ``nodes/<node>/*.sql`` migrations run against ``NODE_PGDB``
  (``NODE_POSTGRES_DB``, compose: ``omnidash_analytics``), the application
  database, which is also the only database ``bootstrap.sql`` converges.

So each database already has exactly ONE declaring owner for its copy, which is
what "one owner for cold bring-up" actually requires. Retiring 031 would not
remove a second owner from a database; it would delete the *only* owner the
service database has.

Four further consequences are pinned below, each of which the manifest has no
way to express. They are asserted rather than described so that a future change
that retires 031 fails here, in CI, with the reasoning attached -- instead of
failing at deploy time on a lane.

Ticket: OMN-15857 (ownership ruling). Related: OMN-12970 (why the node
migration was added), OMN-15561 (the service ledger records a constant literal
checksum and compares nothing), OMN-15857 CodeRabbit thread on the
NODE_POSTGRES_DB fallback (the distinct-database condition is now asserted, not
assumed) and migration 087, which cleans up the decoy tables produced the last
time the two variables resolved to one database.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = [pytest.mark.unit]


def _load_compose_services(path: Path) -> dict[str, Any]:
    """Parse a compose file's ``services`` block, tolerating ``!override``.

    ``yaml.safe_load`` has no constructor for compose's ``!override`` tag (the
    judge overlay uses it), so it is stripped first -- the same approach
    ``tests/unit/infra/test_compose_lane_internal_dsn_omn16843.py`` takes.
    """
    text = path.read_text(encoding="utf-8").replace("!override", "")
    doc = yaml.safe_load(text) or {}
    services = doc.get("services") or {}
    assert isinstance(services, dict), path
    return services


REPO_ROOT = Path(__file__).resolve().parents[2]
FORWARD = REPO_ROOT / "docker" / "migrations" / "forward"
LEDGER = FORWARD / "_ledger"
RUNNER = REPO_ROOT / "scripts" / "run-forward-migrations.sh"
APPEND_ONLY = REPO_ROOT / "scripts" / "validation" / "check_migration_append_only.py"

LEGACY_031 = FORWARD / "031_create_llm_call_metrics_and_cost_aggregates.sql"
NODE_0001 = (
    FORWARD / "nodes" / "node_projection_llm_cost" / "0001_create_llm_call_metrics.sql"
)

# Flat migrations that ALTER the table 031 creates, in the service database.
DEPENDENT_FLAT_MIGRATIONS = (
    "071_add_llm_call_metrics_idempotency_unique.sql",
    "072_add_llm_call_metrics_attribution.sql",
    "073_add_llm_call_metrics_gpu_fields.sql",
    "077_migrate_usage_source_vocab.sql",
)


def test_both_declarations_are_still_present() -> None:
    """The premise. If either file goes, the rest of this file is stale."""
    assert LEGACY_031.is_file()
    assert NODE_0001.is_file()


def test_the_two_declarations_target_different_databases() -> None:
    """The runner routes flat and node migrations to two distinct databases.

    This is the whole ruling in one assertion: there is no database in which
    both files run, so there is no database with two owners of this table.
    """
    runner = RUNNER.read_text(encoding="utf-8")
    assert re.search(r'^NODE_PGDB="\$\{NODE_POSTGRES_DB:-\$\{PGDB\}\}"', runner, re.M)
    # bootstrap.sql -- the code that raised the lane-blocking exception -- is
    # only ever pointed at the node database.
    assert re.search(r'^\s*prepare_canonical_ledger "\$NODE_PGDB"', runner, re.M)
    assert 'prepare_canonical_ledger "$PGDB"' not in runner

    # The routing above collapses when NODE_POSTGRES_DB is unset, because
    # NODE_PGDB then FALLS BACK to PGDB and node migrations land in the service
    # database after all. That is not hypothetical: migration
    # 087_drop_stale_delegation_events_decoy.sql exists solely to clean up the
    # decoy tables a real deployment created while the two variables resolved to
    # the same database. So the separation is only real while every compose lane
    # that runs this migrator pins them apart -- assert that, not just the
    # runner's default expression.
    #
    # Asserted per SERVICE, not per file. A file-wide set of the two variables
    # cannot distinguish "every migrator pins NODE_POSTGRES_DB" from "one of
    # several migrators pins it and the rest silently fall back" -- and it is
    # the un-pinned one that recreates the 087 decoy.
    checked: list[tuple[str, str]] = []
    for compose_name in ("docker-compose.infra.yml", "docker-compose.judge.yml"):
        services = _load_compose_services(REPO_ROOT / "docker" / compose_name)
        migrators = {
            name: cfg
            for name, cfg in services.items()
            if RUNNER.name in f"{cfg.get('command', '')}{cfg.get('entrypoint', '')}"
        }
        assert migrators, f"{compose_name} declares no {RUNNER.name} service"
        for name, cfg in migrators.items():
            env = cfg.get("environment") or {}
            assert isinstance(env, dict), (compose_name, name, "list-form environment")
            node_db = env.get("NODE_POSTGRES_DB")
            service_db = env.get("POSTGRES_DB")
            assert node_db == "omnidash_analytics", (compose_name, name, node_db)
            assert service_db and service_db != node_db, (
                compose_name,
                name,
                service_db,
                node_db,
            )
            checked.append((compose_name, name))
    assert checked


def test_the_application_manifest_cannot_name_a_flat_migration() -> None:
    """The manifest has no vocabulary for retiring 031.

    ``_ledger/application-migrations.tsv`` declares node artifacts only -- not
    one row addresses a flat ``docker/NNN`` migration -- so "retire the legacy
    declaration in the migration manifest" has no expressible form here.
    """
    rows = [
        line.split("\t")
        for line in (LEDGER / "application-migrations.tsv")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert rows
    assert all(row[0].startswith("nodes/") for row in rows)
    assert not any(row[0] == LEGACY_031.name for row in rows)


def test_the_supersession_manifest_cannot_name_a_flat_migration_either() -> None:
    """The other candidate manifest is likewise node-scoped by construction.

    ``check_migration_append_only.py`` rejects any supersession path that is not
    ``nodes/<node>/<ordinal>_<name>.sql``, so a row retiring a flat migration
    could not be parsed even if someone wrote one.
    """
    guard = APPEND_ONLY.read_text(encoding="utf-8")
    assert "supersession path must be nodes/<node>/<ordinal>_<name>.sql" in guard
    rows = [
        line.split("\t")
        for line in (LEDGER / "migration-supersessions.tsv")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert all(row[0].startswith("nodes/") for row in rows)


def test_031_owns_a_table_the_node_migration_deliberately_declines() -> None:
    """031 also creates ``llm_cost_aggregates``; the node file says it will not.

    Retiring 031 would drop that table's only declaration. The node migration's
    own header states the exclusion, so this is a documented boundary rather
    than an oversight to tidy up.
    """
    node_sql = NODE_0001.read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS llm_cost_aggregates" not in node_sql
    assert "llm_cost_aggregates is intentionally NOT created" in node_sql
    assert "CREATE TABLE IF NOT EXISTS llm_cost_aggregates" in LEGACY_031.read_text(
        encoding="utf-8"
    )


def test_later_flat_migrations_depend_on_the_table_031_creates() -> None:
    """Four later flat migrations ALTER the 031 table in the service database.

    None of them is a node migration, so none of them would follow the node
    declaration into ``omnidash_analytics``. Retiring 031 orphans all four
    against a table nothing creates any more.
    """
    for name in DEPENDENT_FLAT_MIGRATIONS:
        sql = (FORWARD / name).read_text(encoding="utf-8")
        assert "llm_call_metrics" in sql, name
        assert not (FORWARD / "nodes").joinpath(name).exists(), name


def test_the_node_migration_records_why_it_duplicates_031() -> None:
    """The duplication is a deliberate, documented cross-database mirror.

    OMN-12970 added the node migration because the projection API binds to
    ``omnidash_analytics`` and the table only ever existed in
    ``omnibase_infra``. The header is the ownership contract; this pins it so a
    future reader does not re-litigate the duplication as an accident.
    """
    header = NODE_0001.read_text(encoding="utf-8")[:4000]
    assert "OMN-12970" in header
    assert "omnidash_analytics" in header
    assert "031_create_llm_call_metrics_and_cost_aggregates.sql" in header
