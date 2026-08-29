# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The sentinel checksum that blocked every stability-lane refresh (OMN-15857).

``_ledger/bootstrap.sql`` accepts a node row out of ``public.schema_migrations``
only when its checksum is a 64-hex ``content_sha256`` matching the manifest, or
the literal ``applied-by-runner``. The ``.201`` stability-test lane carries
``hotfix-applied-by-codex`` on
``node:node_projection_llm_cost:0001_create_llm_call_metrics.sql``, which is
neither, so the import loop raised and aborted the transaction -- taking the
whole lane's migration convergence with it. Live on 2026-08-28::

    psql:/migrations/forward/_ledger/bootstrap.sql:709: ERROR:
      conflicting migration checksum for version
      node:node_projection_llm_cost:0001_create_llm_call_metrics.sql

The fix is NOT to widen the accepted spelling. The sentinel means "nobody proved
the hand-applied SQL matches the checked-in file", and a blanket tolerance would
launder that open question into a clean row -- reopening exactly the
out-of-band-mutation-reads-as-verified gap the sentinel records. The fix is a
per-version declaration in ``_ledger/verified-checksum-adoptions.tsv`` backed by
a mechanical equivalence proof from
``scripts/migrations/verify_migration_checksum_adoption.py``.

The three states this file pins, in the order the ticket demands them:

``test_sentinel_without_a_declaration_is_still_an_atomic_red``
    RED, and it stays red forever. This is the behaviour that broke the lane; if
    a future change makes a bare sentinel converge, that is the regression.
``test_verified_adoption_converges_the_sentinel_row``
    GREEN only through the declaration, and the adopted row lands with the
    manifest content hash plus a provenance string naming the ticket and the
    receipt that proved it -- the sentinel is preserved in that provenance, not
    erased.
``test_declaration_pinned_to_stale_manifest_content_is_a_red``
``test_declaration_for_a_different_sentinel_is_a_red``
    The declaration is not a permanent hall pass. Rewriting the migration file
    after the proof, or a row carrying a sentinel the declaration does not name,
    both fail closed.

The third leg the ticket asks for -- still failing when the *applied schema*
genuinely diverges -- cannot live here: bootstrap.sql compares checksums and
never sees a table. It is proven in
``test_migration_checksum_verifier_omn15857.py``, which is where the schema
comparison actually happens.

Ticket: OMN-15857
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration.migrations.test_application_migration_ledger_omn15413 import (
    MANIFEST,
    Pg16Cluster,
    _run_bootstrap,
    _seed_migration_id_ledger,
)

pytestmark = [pytest.mark.integration, pytest.mark.postgres, pytest.mark.serial]

SENTINEL = "hotfix-applied-by-codex"
VERSION = "node:node_projection_llm_cost:0001_create_llm_call_metrics.sql"
TICKET = "OMN-15857"
# sha256 of the committed receipt; shape-checked by the manifest validator.
RECEIPT_SHA = "db7d2d12fd664fcd006787a8f95b6e1265570416d30d6b897bedb3a2759d81b5"


def _manifest_checksum(version: str) -> str:
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if fields[4] == version:
            return fields[5]
    raise AssertionError(f"{version} is not declared in {MANIFEST}")


def _write_adoption(
    tmp_path: Path,
    *,
    version: str = VERSION,
    source_checksum: str = SENTINEL,
    manifest_checksum: str | None = None,
) -> Path:
    path = tmp_path / "verified-checksum-adoptions.tsv"
    path.write_text(
        "\t".join(
            (
                version,
                source_checksum,
                manifest_checksum or _manifest_checksum(version),
                TICKET,
                RECEIPT_SHA,
                "2026-08-28",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return path


SIBLING = "node:node_projection_cost_summary:0001_create_llm_cost_aggregates.sql"


def _seed_sentinel_ledger(pg16: Pg16Cluster, database: str) -> None:
    """Reproduce the lane's four-column node ledger with one sentinel row.

    The sibling ``node_projection_cost_summary`` row rides along as the contrast
    case: it carries the runner literal, converges cleanly on the lane today,
    and must keep converging cleanly after the adoption branch lands.
    """
    _seed_migration_id_ledger(
        pg16,
        database,
        [(VERSION, SENTINEL, "node"), (SIBLING, "applied-by-runner", "node")],
    )


def _empty_adoptions(tmp_path: Path) -> Path:
    path = tmp_path / "no-adoptions.tsv"
    path.write_text("", encoding="utf-8")
    return path


def test_sentinel_without_a_declaration_is_still_an_atomic_red(
    pg16: Pg16Cluster, tmp_path: Path
) -> None:
    """The lane-breaking failure, pinned. No declaration, no convergence."""
    database = "omn15857_no_declaration"
    pg16.create_database(database)
    _seed_sentinel_ledger(pg16, database)

    completed = _run_bootstrap(pg16, database, adoptions=_empty_adoptions(tmp_path))

    assert completed.returncode != 0
    assert "conflicting migration checksum for version" in completed.stderr
    assert VERSION in completed.stderr
    # Atomic: the aborted transaction leaves no canonical ledger behind.
    assert (
        pg16.sql(database, "SELECT to_regclass('platform_catalog.schema_migrations')")
        == ""
    )


def test_verified_adoption_converges_the_sentinel_row(
    pg16: Pg16Cluster, tmp_path: Path
) -> None:
    """With the proof declared, the row adopts the manifest content hash."""
    database = "omn15857_adopted"
    pg16.create_database(database)
    _seed_sentinel_ledger(pg16, database)

    completed = _run_bootstrap(pg16, database, adoptions=_write_adoption(tmp_path))
    assert completed.returncode == 0, completed.stderr

    checksum, checksum_kind, provenance = pg16.sql(
        database,
        "SELECT checksum, checksum_kind, provenance "  # noqa: S608
        "FROM platform_catalog.schema_migrations "
        f"WHERE version = '{VERSION}'",
    ).split("|")

    assert checksum == _manifest_checksum(VERSION)
    # content_sha256, not a weaker kind: the equivalence proof means this row
    # really does assert the file bytes, so the OMN-16705 canonical-history
    # guard must keep policing it.
    assert checksum_kind == "content_sha256"
    # The sentinel survives in the provenance. Adoption records the question and
    # its answer; it does not erase the evidence that a question existed.
    assert provenance.startswith("verified-adoption:")
    assert f"raw-checksum={SENTINEL}" in provenance
    assert f"ticket={TICKET}" in provenance
    assert f"receipt={RECEIPT_SHA}" in provenance

    # The sibling runner-literal row is untouched by the new branch.
    sibling = pg16.sql(
        database,
        "SELECT provenance FROM platform_catalog.schema_migrations "  # noqa: S608
        f"WHERE version = '{SIBLING}'",
    )
    assert sibling.startswith("adopted:")
    assert "ticket=" not in sibling


def test_bootstrap_is_idempotent_under_a_verified_adoption(
    pg16: Pg16Cluster, tmp_path: Path
) -> None:
    """Second convergence must not trip the double-declaration guard.

    The adopted row's provenance differs from the plain-adoption spelling, and
    the re-import path compares provenance exactly -- so a mismatched second
    render would raise ``double migration declaration``. Deploys re-run this.
    """
    database = "omn15857_twice"
    pg16.create_database(database)
    _seed_sentinel_ledger(pg16, database)
    adoptions = _write_adoption(tmp_path)

    first = _run_bootstrap(pg16, database, adoptions=adoptions)
    assert first.returncode == 0, first.stderr
    second = _run_bootstrap(pg16, database, adoptions=adoptions)
    assert second.returncode == 0, second.stderr

    assert (
        pg16.sql(
            database,
            "SELECT count(*) FROM platform_catalog.schema_migrations "  # noqa: S608
            f"WHERE version = '{VERSION}'",
        )
        == "1"
    )


def test_declaration_pinned_to_stale_manifest_content_is_a_red(
    pg16: Pg16Cluster, tmp_path: Path
) -> None:
    """Rewriting the migration after the proof re-opens the question."""
    database = "omn15857_stale_pin"
    pg16.create_database(database)
    _seed_sentinel_ledger(pg16, database)

    stale = _write_adoption(tmp_path, manifest_checksum="0" * 64)
    completed = _run_bootstrap(pg16, database, adoptions=stale)

    assert completed.returncode != 0
    assert "conflicting migration checksum for version" in completed.stderr
    assert VERSION in completed.stderr


def test_declaration_for_a_different_sentinel_is_a_red(
    pg16: Pg16Cluster, tmp_path: Path
) -> None:
    """A declaration covers one hand-edit, not every hand-edit of that version."""
    database = "omn15857_wrong_sentinel"
    pg16.create_database(database)
    _seed_sentinel_ledger(pg16, database)

    wrong = _write_adoption(tmp_path, source_checksum="applied-manually-omn-11760")
    completed = _run_bootstrap(pg16, database, adoptions=wrong)

    assert completed.returncode != 0
    assert "conflicting migration checksum for version" in completed.stderr


def test_adoption_provenance_does_not_leak_to_the_next_row(
    pg16: Pg16Cluster, tmp_path: Path
) -> None:
    """plpgsql RECORDs persist across loop iterations; this proves the reset.

    ``adoption_row`` is a RECORD reused by every iteration of the import loop.
    Without the explicit clear at the top of the loop, a version sorting AFTER
    the adopted one would inherit its ticket and receipt into a provenance
    string that names a proof which never covered it. ``cost_summary`` sorts
    after ``llm_cost``, so this ordering is the one that would catch it.
    """
    database = "omn15857_no_leak"
    pg16.create_database(database)
    _seed_sentinel_ledger(pg16, database)

    completed = _run_bootstrap(pg16, database, adoptions=_write_adoption(tmp_path))
    assert completed.returncode == 0, completed.stderr

    leaked = pg16.sql(
        database,
        "SELECT count(*) FROM platform_catalog.schema_migrations "  # noqa: S608
        f"WHERE version <> '{VERSION}' AND provenance LIKE '%%{TICKET}%%'",
    )
    assert leaked == "0"
