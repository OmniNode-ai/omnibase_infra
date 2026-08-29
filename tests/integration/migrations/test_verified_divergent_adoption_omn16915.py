# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The divergent-bytes checksum that blocked the lane after OMN-15857 (OMN-16915).

OMN-15857 cleared seven *sentinel* rows -- hand-written non-hash markers like
``hotfix-applied-by-codex`` -- out of ``public.schema_migrations``. The very next
sanctioned stability-lane repair got past them and died one block further down::

    psql:.../_ledger/bootstrap.sql:876: ERROR:
      conflicting migration checksum for version
      node:node_contract_registry:0000_create_contract_registry.sql

This is a different failure wearing the same error string. The row lives in
``public.omnimarket_schema_migrations``, and its checksum is not a sentinel at
all -- it is a real, well-formed sha256
(``686b135950659fa6ab11161a439fac60ae273ff3cf410fe6b209f505bb8983e7``) that
simply is not the manifest's (``bff49882...``). Nothing was hand-applied. The
lane applied a genuine *earlier revision* of the checked-in file -- the artifact
bytes as of ``5b904d881`` (2026-07-21), the last state before OMN-15376 (#2537)
rewrote six node migrations in place on 2026-07-29 -- and was never
re-converged.

OMN-15857's machinery cannot reach it, on two independent axes: its discovery
query reads ``public.schema_migrations`` only, and filters
``checksum !~ '^[0-9a-f]{64}$'`` -- a well-formed hash is excluded by
construction. And the ``$omnimarket_import$`` block had no adoption consultation
at all; it raised unconditionally.

The temptation here is worse than in OMN-15857, because the history is *so*
clean: six rows, one commit, a diff that is visibly nothing but guarded
``ADD COLUMN IF NOT EXISTS`` / ``SET NOT NULL`` / ``ADD CONSTRAINT`` statements
designed to be no-ops on the fresh-create path. It would be easy to call that
proof. It is not. "These bytes are an older revision of this file" is evidence
about *provenance*; the ledger's claim is about *schema*. Only executing both
revisions answers it, which is what
``scripts/migrations/verify_migration_checksum_adoption.py`` does -- and this
file pins the SQL half: bootstrap.sql must refuse the row until that proof is
declared, and must keep refusing when the declaration goes stale.

Ticket: OMN-16915
"""

# ruff: noqa: S608 -- every interpolated value is a checked-in manifest
# constant or a literal defined in this file; there is no untrusted input here.

from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration.migrations.test_application_migration_ledger_omn15413 import (
    MANIFEST,
    Pg16Cluster,
    _run_bootstrap,
)

pytestmark = [pytest.mark.integration, pytest.mark.postgres, pytest.mark.serial]

NODE = "node_contract_registry"
FILENAME = "0000_create_contract_registry.sql"
VERSION = f"node:{NODE}:{FILENAME}"
# The revision the .201 stability lane actually applied: the artifact as of
# 5b904d881 (2026-07-21), superseded by OMN-15376 #2537 on 2026-07-29.
DIVERGENT = "686b135950659fa6ab11161a439fac60ae273ff3cf410fe6b209f505bb8983e7"
TICKET = "OMN-16915"
RECEIPT_SHA = "f" * 64

# A second omnimarket row that agrees with the manifest. It rides along in every
# case as the contrast: it converges today, and must keep converging unchanged
# after the divergent branch lands.
CLEAN_NODE = "node_projection_registration"
CLEAN_FILENAME = "0000_create_node_service_registry.sql"
CLEAN_VERSION = f"node:{CLEAN_NODE}:{CLEAN_FILENAME}"


def _manifest_checksum(version: str) -> str:
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if fields and fields[4] == version:
            return fields[5]
    raise AssertionError(f"{version} is not declared in {MANIFEST}")


def _seed_omnimarket_ledger(pg16: Pg16Cluster, database: str) -> None:
    """Reproduce the lane's omnimarket ledger: one divergent row, one clean."""
    pg16.command(
        database,
        "-f",
        "-",
        input_text=f"""
CREATE TABLE public.omnimarket_schema_migrations (
  id SERIAL PRIMARY KEY,
  node_name TEXT NOT NULL,
  version TEXT NOT NULL,
  filename TEXT NOT NULL,
  checksum TEXT NOT NULL,
  applied_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (node_name, version)
);
INSERT INTO public.omnimarket_schema_migrations
  (node_name, version, filename, checksum, applied_at)
VALUES
  ('{NODE}', '{FILENAME}', '{FILENAME}', '{DIVERGENT}',
   TIMESTAMPTZ '2026-05-23 13:49:15.138389+00'),
  ('{CLEAN_NODE}', '{CLEAN_FILENAME}', '{CLEAN_FILENAME}',
   '{_manifest_checksum(CLEAN_VERSION)}',
   TIMESTAMPTZ '2026-05-23 13:49:15.138389+00');
""",
        check=True,
    )


def _write_divergent_adoption(
    tmp_path: Path,
    *,
    version: str = VERSION,
    source_checksum: str = DIVERGENT,
    manifest_checksum: str | None = None,
    name: str = "verified-divergent-adoptions.tsv",
) -> Path:
    path = tmp_path / name
    path.write_text(
        "\t".join(
            (
                version,
                source_checksum,
                manifest_checksum or _manifest_checksum(version),
                TICKET,
                RECEIPT_SHA,
                "2026-08-29",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _empty(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_text("", encoding="utf-8")
    return path


def test_divergent_checksum_without_a_declaration_is_still_an_atomic_red(
    pg16: Pg16Cluster, tmp_path: Path
) -> None:
    """The lane-breaking failure, pinned. It must stay red forever.

    If a future change makes a bare divergent checksum converge on its own, that
    is the regression this test exists to catch.
    """
    database = "omn16915_no_declaration"
    pg16.create_database(database)
    _seed_omnimarket_ledger(pg16, database)

    completed = _run_bootstrap(
        pg16,
        database,
        divergent_adoptions=_empty(tmp_path, "none.tsv"),
    )

    assert completed.returncode != 0
    assert "conflicting migration checksum for version" in completed.stderr
    assert VERSION in completed.stderr
    # Atomic: the aborted transaction leaves no canonical ledger behind, so the
    # clean sibling does not land either.
    assert (
        pg16.sql(database, "SELECT to_regclass('platform_catalog.schema_migrations')")
        == ""
    )


def test_verified_divergent_adoption_converges_the_row(
    pg16: Pg16Cluster, tmp_path: Path
) -> None:
    """With the proof declared, the row adopts the MANIFEST content hash.

    Not the divergent one. The proof says the live schema is what the current
    checked-in bytes produce, so the manifest hash is the honest claim to record
    -- and the divergent hash survives verbatim in the provenance rather than
    being erased.
    """
    database = "omn16915_adopted"
    pg16.create_database(database)
    _seed_omnimarket_ledger(pg16, database)

    completed = _run_bootstrap(
        pg16, database, divergent_adoptions=_write_divergent_adoption(tmp_path)
    )
    assert completed.returncode == 0, completed.stderr

    checksum, checksum_kind, provenance = pg16.sql(
        database,
        "SELECT checksum, checksum_kind, provenance "
        "FROM platform_catalog.schema_migrations "
        f"WHERE version = '{VERSION}'",
    ).split("|")

    assert checksum == _manifest_checksum(VERSION)
    assert checksum != DIVERGENT
    # content_sha256, not a weaker kind: the equivalence proof means this row
    # really does assert the file bytes, so the OMN-16705 canonical-history
    # guard must keep policing it.
    assert checksum_kind == "content_sha256"

    assert provenance.startswith("verified-divergent-adoption:")
    assert f"raw-checksum={DIVERGENT}" in provenance
    assert f"ticket={TICKET}" in provenance
    assert f"receipt={RECEIPT_SHA}" in provenance

    # The clean sibling keeps the plain legacy spelling -- the new branch is not
    # in its path at all.
    sibling = pg16.sql(
        database,
        "SELECT provenance FROM platform_catalog.schema_migrations "
        f"WHERE version = '{CLEAN_VERSION}'",
    )
    assert sibling.startswith("legacy:")
    assert "ticket=" not in sibling


def test_bootstrap_is_idempotent_under_a_verified_divergent_adoption(
    pg16: Pg16Cluster, tmp_path: Path
) -> None:
    """Second convergence must not trip the double-declaration guard.

    The adopted row's provenance differs from the plain legacy spelling, and the
    re-import path compares provenance exactly -- a mismatched second render
    would raise ``double migration declaration``. Every deploy re-runs this.
    """
    database = "omn16915_twice"
    pg16.create_database(database)
    _seed_omnimarket_ledger(pg16, database)
    adoptions = _write_divergent_adoption(tmp_path)

    first = _run_bootstrap(pg16, database, divergent_adoptions=adoptions)
    assert first.returncode == 0, first.stderr
    second = _run_bootstrap(pg16, database, divergent_adoptions=adoptions)
    assert second.returncode == 0, second.stderr

    assert (
        pg16.sql(
            database,
            "SELECT count(*) FROM platform_catalog.schema_migrations "
            f"WHERE version = '{VERSION}'",
        )
        == "1"
    )


def test_declaration_pinned_to_stale_manifest_content_is_a_red(
    pg16: Pg16Cluster, tmp_path: Path
) -> None:
    """Rewriting the migration after the proof re-opens the question.

    This is the leg that makes the declaration a fact with a shelf life rather
    than a permanent hall pass: the proof ran against one manifest checksum, and
    if the file has moved since, nothing has been proven about the new bytes.
    """
    database = "omn16915_stale_pin"
    pg16.create_database(database)
    _seed_omnimarket_ledger(pg16, database)

    stale = _write_divergent_adoption(tmp_path, manifest_checksum="0" * 64)
    completed = _run_bootstrap(pg16, database, divergent_adoptions=stale)

    assert completed.returncode != 0
    assert "conflicting migration checksum for version" in completed.stderr
    assert VERSION in completed.stderr


def test_declaration_for_a_different_revision_is_a_red(
    pg16: Pg16Cluster, tmp_path: Path
) -> None:
    """A declaration covers ONE prior revision, not any prior revision.

    A lane carrying some third revision of the file is a different question with
    a different answer, and this declaration says nothing about it.
    """
    database = "omn16915_wrong_revision"
    pg16.create_database(database)
    _seed_omnimarket_ledger(pg16, database)

    wrong = _write_divergent_adoption(tmp_path, source_checksum="a" * 64)
    completed = _run_bootstrap(pg16, database, divergent_adoptions=wrong)

    assert completed.returncode != 0
    assert "conflicting migration checksum for version" in completed.stderr


def test_malformed_checksum_is_never_adoptable_through_this_path(
    pg16: Pg16Cluster, tmp_path: Path
) -> None:
    """A non-hash in omnimarket's ledger stays a hard red, declaration or not.

    Only a well-formed content hash can be argued about. A sentinel here is not
    a divergent-bytes question and must not be laundered into one by declaring
    it -- the malformed-checksum raise fires before the consultation.
    """
    database = "omn16915_malformed"
    pg16.create_database(database)
    pg16.command(
        database,
        "-f",
        "-",
        input_text=f"""
CREATE TABLE public.omnimarket_schema_migrations (
  id SERIAL PRIMARY KEY,
  node_name TEXT NOT NULL,
  version TEXT NOT NULL,
  filename TEXT NOT NULL,
  checksum TEXT NOT NULL,
  applied_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (node_name, version)
);
INSERT INTO public.omnimarket_schema_migrations
  (node_name, version, filename, checksum, applied_at)
VALUES
  ('{NODE}', '{FILENAME}', '{FILENAME}', 'hotfix-applied-by-codex',
   TIMESTAMPTZ '2026-05-23 13:49:15.138389+00');
""",
        check=True,
    )

    declared = _write_divergent_adoption(
        tmp_path, source_checksum="hotfix-applied-by-codex"
    )
    completed = _run_bootstrap(pg16, database, divergent_adoptions=declared)

    assert completed.returncode != 0
    assert "conflicting migration checksum for version" in completed.stderr


def test_sentinel_declaration_cannot_adopt_a_divergent_row(
    pg16: Pg16Cluster, tmp_path: Path
) -> None:
    """The two declaration files are not interchangeable.

    OMN-15857's ``verified-checksum-adoptions.tsv`` answers "the hand-applied SQL
    behind this sentinel produced the checked-in schema". It says nothing about a
    stale-revision row in omnimarket's ledger. Writing the same version into the
    sentinel file must not open the divergent path -- that is the whole reason
    OMN-16915 got a relation of its own instead of a seventh column.
    """
    database = "omn16915_wrong_file"
    pg16.create_database(database)
    _seed_omnimarket_ledger(pg16, database)

    # Correct content, wrong relation.
    misfiled = _write_divergent_adoption(tmp_path, name="misfiled.tsv")
    completed = _run_bootstrap(
        pg16,
        database,
        adoptions=misfiled,
        divergent_adoptions=_empty(tmp_path, "none.tsv"),
    )

    assert completed.returncode != 0
    assert "conflicting migration checksum for version" in completed.stderr
    assert VERSION in completed.stderr


def test_divergent_adoption_provenance_does_not_leak_to_the_next_row(
    pg16: Pg16Cluster, tmp_path: Path
) -> None:
    """plpgsql RECORDs persist across loop iterations; this proves the reset.

    ``divergent_row`` is reused by every iteration of the omnimarket import loop.
    Without the explicit clear at the top of the loop, a version sorting AFTER
    the adopted one would inherit its ticket and receipt into a provenance string
    naming a proof that never covered it. ``node_projection_registration`` sorts
    after ``node_contract_registry``, so this ordering is the one that catches it.
    """
    database = "omn16915_no_leak"
    pg16.create_database(database)
    _seed_omnimarket_ledger(pg16, database)

    completed = _run_bootstrap(
        pg16, database, divergent_adoptions=_write_divergent_adoption(tmp_path)
    )
    assert completed.returncode == 0, completed.stderr

    leaked = pg16.sql(
        database,
        "SELECT count(*) FROM platform_catalog.schema_migrations "
        f"WHERE version <> '{VERSION}' AND provenance LIKE '%%{TICKET}%%'",
    )
    assert leaked == "0"


# ---------------------------------------------------------------------------
# the lane itself
# ---------------------------------------------------------------------------

# The .201 stability-test lane's public.omnimarket_schema_migrations, read live
# on 2026-08-29. Six rows diverge from the manifest; four agree. bootstrap.sql
# raises on the FIRST divergence in (node_name, version) order, so clearing one
# row at a time would have cost six more failed heal cycles to discover the rest.
LANE_ROWS: tuple[tuple[str, str, str], ...] = (
    (
        "node_contract_registry",
        "0000_create_contract_registry.sql",
        "686b135950659fa6ab11161a439fac60ae273ff3cf410fe6b209f505bb8983e7",
    ),
    (
        "node_evidence_dashboard_reducer",
        "0001_create_evidence_dashboard_projection_tables.sql",
        "a83717d2077fe6a4ca737adda2f7d3ab37544f14826284ec3839b810925bf051",
    ),
    (
        "node_nightly_loop_controller",
        "001_create_nightly_loop_tables.sql",
        "55fd0b90a3dd033df7c57351d0739e9218fe58ba290c732c78e78d7df49eaa60",
    ),
    (
        "node_pr_review_bot",
        "001_create_review_bot_bypass_log.sql",
        "63e2646a7f8767fad9ec969b224982e00b83aacb665107ab5b103964d5616e00",
    ),
    (
        "node_projection_delegation",
        "0008_generation_events.sql",
        "e12015762c7b2c598ed03f38a4e6d2b3cf7cbd6d6f8c30e05d21758a420580b6",
    ),
    (
        "node_projection_delegation",
        "0009_delegate_skill_projection_metrics.sql",
        "ba2f5e8428084d2f7e3386372864aed7b8d47f31d4048e12878f8799060fb03d",
    ),
    (
        "node_projection_dep_health",
        "001_create_dep_health_findings.sql",
        "5bfcf6862b235ea79bab65dd8681db9217f851511c2dfa8283191e2c7094217d",
    ),
    (
        "node_projection_registration",
        "0000_create_node_service_registry.sql",
        "5c320392190e6b9a1e06ab7177b7560181ca00282e107aebfccc452441e4d478",
    ),
    (
        "node_projection_registration",
        "0001_add_heartbeat_columns.sql",
        "2cda779401d041ae1383937df3f4c614df74d09c80c78f24f7993685ba7a6aed",
    ),
    (
        "node_projection_session_outcome",
        "0021_session_outcomes.sql",
        "68adbc4faff969f8f1be689899ccae591331bd858a14bef30c62e4d73600b66b",
    ),
)


def _seed_the_real_lane(pg16: Pg16Cluster, database: str) -> None:
    values = ",\n  ".join(
        f"('{node}', '{filename}', '{filename}', '{checksum}', "
        "TIMESTAMPTZ '2026-05-23 13:49:15.138389+00')"
        for node, filename, checksum in LANE_ROWS
    )
    pg16.command(
        database,
        "-f",
        "-",
        input_text=f"""
CREATE TABLE public.omnimarket_schema_migrations (
  id SERIAL PRIMARY KEY,
  node_name TEXT NOT NULL,
  version TEXT NOT NULL,
  filename TEXT NOT NULL,
  checksum TEXT NOT NULL,
  applied_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (node_name, version)
);
INSERT INTO public.omnimarket_schema_migrations
  (node_name, version, filename, checksum, applied_at)
VALUES
  {values};
""",
        check=True,
    )


def test_the_real_lane_ledger_converges_under_the_committed_declarations(
    pg16: Pg16Cluster,
) -> None:
    """The heal-resume proof: the actual lane rows, the actual committed TSV.

    Every other test in this file drives bootstrap.sql with a declaration file
    written for that test. This one uses the file that will ship, against the
    exact ten rows the .201 stability lane carries, and asserts the whole import
    converges -- which is the thing OMN-16803's next heal attempt needs to be
    true. If a declaration is missing, stale, or covers the wrong revision, this
    goes red here instead of on the lane.
    """
    database = "omn16915_real_lane"
    pg16.create_database(database)
    _seed_the_real_lane(pg16, database)

    completed = _run_bootstrap(pg16, database)
    assert completed.returncode == 0, completed.stderr

    landed = pg16.sql(
        database,
        "SELECT count(*) FROM platform_catalog.schema_migrations "
        "WHERE provenance LIKE 'legacy:%' OR provenance LIKE 'verified-divergent-adoption:%'",
    )
    assert landed == str(len(LANE_ROWS))

    adopted = pg16.sql(
        database,
        "SELECT count(*) FROM platform_catalog.schema_migrations "
        "WHERE provenance LIKE 'verified-divergent-adoption:%'",
    )
    assert adopted == "6", (
        "six of the ten lane rows diverge from the manifest; if this number "
        "moves, the lane or the manifest changed and the proofs need re-running"
    )

    # Every adopted row records the manifest hash, never the divergent one.
    for _, _, checksum in LANE_ROWS:
        stored = pg16.sql(
            database,
            "SELECT count(*) FROM platform_catalog.schema_migrations "
            f"WHERE checksum = '{checksum}' "
            "AND provenance LIKE 'verified-divergent-adoption:%'",
        )
        assert stored == "0", (
            f"a divergent checksum {checksum[:12]}... was written into the "
            "canonical ledger as though it were the declared content"
        )


def test_the_real_lane_ledger_is_still_red_without_the_declarations(
    pg16: Pg16Cluster, tmp_path: Path
) -> None:
    """The same ten rows, no declarations: exactly the production failure.

    This pins that the convergence above is bought by the committed proofs and
    nothing else -- and reproduces the error string, version, and bootstrap.sql
    line the 2026-08-28 heal run died on.
    """
    database = "omn16915_real_lane_red"
    pg16.create_database(database)
    _seed_the_real_lane(pg16, database)

    completed = _run_bootstrap(
        pg16, database, divergent_adoptions=_empty(tmp_path, "none.tsv")
    )

    assert completed.returncode != 0
    assert "conflicting migration checksum for version" in completed.stderr
    # node_contract_registry sorts first, so it is the row the lane reported.
    assert VERSION in completed.stderr
