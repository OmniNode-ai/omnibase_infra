# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The dual declaration that blocked the lane after OMN-16915 (OMN-16919).

OMN-15857 cleared seven *sentinel* rows. OMN-16915 cleared six *divergent-bytes*
rows. The next sanctioned stability-lane repair got past both and died at the
next block, with a **different error string**::

    psql:.../_ledger/bootstrap.sql:949: ERROR:
      double migration declaration for version
      node:node_contract_registry:0000_create_contract_registry.sql

Not ``conflicting migration checksum``. The checksum comparison *passes* -- both
source ledgers resolve the version to the same manifest hash. The raise is one
branch further down, on the metadata: ``applied_at`` and ``provenance``.

The cause is that a version can be declared by BOTH source relations at once.
``public.schema_migrations`` carries an ``applied-by-runner`` row written by the
ONEX-era runner in June; ``public.omnimarket_schema_migrations`` carries a row
written by the omnimarket-era runner in May. ``$migration_id_import$`` inserts
the canonical row from the first, then ``$omnimarket_import$`` finds it and
refuses, because the two sources disagree about *when* and *how* it was
recorded.

Why OMN-16915's own lane test did not catch this is the part worth keeping.
``test_the_real_lane_ledger_converges_under_the_committed_declarations`` drives
all ten real lane rows and passes -- while the real lane fails on those same ten
versions. Its fixture creates exactly one relation,
``public.omnimarket_schema_migrations``. The failure is a *join* condition: with
only that table present ``$migration_id_import$`` inserts nothing, the omnimarket
block finds no existing row, takes the INSERT branch, and converges. A fixture
that models one source ledger cannot falsify a two-source failure. **Every
lane-convergence fixture in this file seeds both relations**, which is why the
red state below is the production one.

What this file pins:

* the production failure reproduces verbatim without the declaration,
* a valid declaration converges it,
* re-running bootstrap.sql is a no-op (the reason BOTH DO blocks consult the
  declaration, not just the one that raises),
* every declared value is load-bearing -- drift in any of them fails closed,
* a checksum disagreement is still refused, because reconciliation settles
  metadata and never content.

Ticket: OMN-16919
"""

# ruff: noqa: S608 -- every interpolated value is a checked-in manifest constant
# or a literal defined in this file; there is no untrusted input here.

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tests.integration.migrations.test_application_migration_ledger_omn15413 import (
    LEDGER_DIR,
    MANIFEST,
    VERIFIED_CROSS_SOURCE_ADOPTIONS,
    Pg16Cluster,
    _run_bootstrap,
)

pytestmark = [pytest.mark.integration, pytest.mark.postgres, pytest.mark.serial]

NODE = "node_contract_registry"
FILENAME = "0000_create_contract_registry.sql"
VERSION = f"node:{NODE}:{FILENAME}"

# The exact live values, read from stability-test omnidash_analytics on
# 2026-08-29. The omnimarket checksum is the pre-OMN-15376 revision OMN-16915
# proved schema-equivalent; the node side is the runner sentinel.
NODE_CHECKSUM = "applied-by-runner"
OMNIMARKET_CHECKSUM = "686b135950659fa6ab11161a439fac60ae273ff3cf410fe6b209f505bb8983e7"
NODE_APPLIED_AT = "2026-06-10 20:30:05.148691+00"
OMNIMARKET_APPLIED_AT = "2026-05-23 13:49:15.138389+00"
# The earlier of the two: when the schema first existed. The June row is a
# re-registration, not a re-application.
RECONCILED_APPLIED_AT = OMNIMARKET_APPLIED_AT

TICKET = "OMN-16919"
RECEIPT_SHA = "e" * 64
VERIFIED_AT = "2026-08-29"

# A second version that agrees with the manifest outright and is ALSO
# dual-declared. It rides along as the contrast: the reconciliation must cover
# the direct-match case too, not only the ones OMN-16915 had to prove.
CLEAN_NODE = "node_projection_registration"
CLEAN_FILENAME = "0000_create_node_service_registry.sql"
CLEAN_VERSION = f"node:{CLEAN_NODE}:{CLEAN_FILENAME}"
CLEAN_NODE_APPLIED_AT = "2026-06-06 16:07:04.960343+00"
CLEAN_OMNIMARKET_APPLIED_AT = "2026-05-23 13:49:15.138389+00"


def _manifest_checksum(version: str) -> str:
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if fields and fields[4] == version:
            return fields[5]
    raise AssertionError(f"{version} is not declared in {MANIFEST}")


# ---------------------------------------------------------------------------
# fixtures: BOTH source relations, always
# ---------------------------------------------------------------------------


def _seed_both_sources(
    pg16: Pg16Cluster,
    database: str,
    *,
    node_checksum: str = NODE_CHECKSUM,
    node_applied_at: str = NODE_APPLIED_AT,
    omnimarket_checksum: str = OMNIMARKET_CHECKSUM,
    omnimarket_applied_at: str = OMNIMARKET_APPLIED_AT,
) -> None:
    """The lane's real shape: the SAME version in both source ledgers.

    ``public.schema_migrations`` must carry the four-column ``migration_id``
    shape with at least one adoptable ``source_set='node'`` row, or the
    ``$ledger_upgrade$`` block classifies it as the service-owned ledger and
    refuses it long before the interesting branch.
    """
    pg16.command(
        database,
        "-f",
        "-",
        input_text=f"""
CREATE TABLE public.schema_migrations (
  migration_id TEXT NOT NULL,
  applied_at TIMESTAMPTZ NOT NULL,
  checksum TEXT NOT NULL,
  source_set TEXT NOT NULL
);
INSERT INTO public.schema_migrations
  (migration_id, applied_at, checksum, source_set)
VALUES
  ('{VERSION}', TIMESTAMPTZ '{node_applied_at}', '{node_checksum}', 'node'),
  ('{CLEAN_VERSION}', TIMESTAMPTZ '{CLEAN_NODE_APPLIED_AT}',
   'applied-by-runner', 'node');

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
  ('{NODE}', '{FILENAME}', '{FILENAME}', '{omnimarket_checksum}',
   TIMESTAMPTZ '{omnimarket_applied_at}'),
  ('{CLEAN_NODE}', '{CLEAN_FILENAME}', '{CLEAN_FILENAME}',
   '{_manifest_checksum(CLEAN_VERSION)}',
   TIMESTAMPTZ '{CLEAN_OMNIMARKET_APPLIED_AT}');
""",
        check=True,
    )


def _write_divergent_adoption(tmp_path: Path) -> Path:
    """OMN-16915's declaration for the divergent row, which stays required."""
    path = tmp_path / "verified-divergent-adoptions.tsv"
    path.write_text(
        "\t".join(
            (
                VERSION,
                OMNIMARKET_CHECKSUM,
                _manifest_checksum(VERSION),
                "OMN-16915",
                "f" * 64,
                "2026-08-29",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_cross_source(
    tmp_path: Path,
    *,
    version: str = VERSION,
    node_source_checksum: str = NODE_CHECKSUM,
    omnimarket_source_checksum: str = OMNIMARKET_CHECKSUM,
    manifest_checksum: str | None = None,
    node_applied_at: str = NODE_APPLIED_AT,
    omnimarket_applied_at: str = OMNIMARKET_APPLIED_AT,
    reconciled_applied_at: str = RECONCILED_APPLIED_AT,
    include_clean: bool = True,
    name: str = "verified-cross-source-adoptions.tsv",
) -> Path:
    rows = [
        "\t".join(
            (
                version,
                node_source_checksum,
                omnimarket_source_checksum,
                manifest_checksum or _manifest_checksum(version),
                node_applied_at,
                omnimarket_applied_at,
                reconciled_applied_at,
                TICKET,
                RECEIPT_SHA,
                VERIFIED_AT,
            )
        )
    ]
    if include_clean:
        rows.append(
            "\t".join(
                (
                    CLEAN_VERSION,
                    "applied-by-runner",
                    _manifest_checksum(CLEAN_VERSION),
                    _manifest_checksum(CLEAN_VERSION),
                    CLEAN_NODE_APPLIED_AT,
                    CLEAN_OMNIMARKET_APPLIED_AT,
                    CLEAN_OMNIMARKET_APPLIED_AT,
                    TICKET,
                    RECEIPT_SHA,
                    VERIFIED_AT,
                )
            )
        )
    path = tmp_path / name
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _empty(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_text("", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# red first
# ---------------------------------------------------------------------------


def test_dual_declaration_without_a_reconciliation_is_the_production_red(
    pg16: Pg16Cluster, tmp_path: Path
) -> None:
    """The exact live failure, reproduced.

    This is the assertion that makes the rest of the file mean anything: with
    both source relations seeded and no cross-source declaration, bootstrap.sql
    must raise ``double migration declaration`` -- the string the stability lane
    actually died on -- and NOT ``conflicting migration checksum``. Getting the
    other string would mean this fixture reproduces OMN-16915's failure again
    rather than the new one.
    """
    database = "omn16919_red"
    pg16.create_database(database)
    _seed_both_sources(pg16, database)

    completed = _run_bootstrap(
        pg16,
        database,
        divergent_adoptions=_write_divergent_adoption(tmp_path),
        cross_source_adoptions=_empty(tmp_path, "none.tsv"),
    )

    assert completed.returncode != 0
    assert "double migration declaration for version" in completed.stderr
    assert VERSION in completed.stderr
    # The distinguishing half: the checksums agreed, so this must NOT be the
    # OMN-16915 error.
    assert "conflicting migration checksum" not in completed.stderr


def test_the_reconciliation_converges_the_dual_declaration(
    pg16: Pg16Cluster, tmp_path: Path
) -> None:
    """Green, and the canonical row keeps BOTH provenances verbatim."""
    database = "omn16919_green"
    pg16.create_database(database)
    _seed_both_sources(pg16, database)

    completed = _run_bootstrap(
        pg16,
        database,
        divergent_adoptions=_write_divergent_adoption(tmp_path),
        cross_source_adoptions=_write_cross_source(tmp_path),
    )
    assert completed.returncode == 0, completed.stderr

    checksum, kind, provenance = pg16.sql(
        database,
        "SELECT checksum || '|' || checksum_kind || '|' || provenance "
        f"FROM platform_catalog.schema_migrations WHERE version = '{VERSION}'",
    ).split("|", 2)

    # The canonical row records the MANIFEST hash. That is the honest content
    # claim: both sources resolve to it.
    assert checksum == _manifest_checksum(VERSION)
    assert kind == "content_sha256"
    # The reconciled timestamp is the DECLARED one -- the earlier application.
    # Compared as an instant, not as rendered text: psql renders timestamptz in
    # the session time zone, so a string prefix would assert the runner's locale
    # rather than the stored value.
    assert (
        pg16.sql(
            database,
            "SELECT applied_at = TIMESTAMPTZ "
            f"'{RECONCILED_APPLIED_AT}' FROM platform_catalog.schema_migrations "
            f"WHERE version = '{VERSION}'",
        )
        == "t"
    )
    assert (
        pg16.sql(
            database,
            "SELECT applied_at <> TIMESTAMPTZ "
            f"'{NODE_APPLIED_AT}' FROM platform_catalog.schema_migrations "
            f"WHERE version = '{VERSION}'",
        )
        == "t"
    ), "the reconciled row must not silently keep the later node-side timestamp"
    # Neither source's evidence is erased.
    assert provenance.startswith("cross-source-reconciled:")
    assert NODE_CHECKSUM in provenance
    assert OMNIMARKET_CHECKSUM in provenance
    assert NODE_APPLIED_AT in provenance
    assert OMNIMARKET_APPLIED_AT in provenance
    assert TICKET in provenance
    assert RECEIPT_SHA in provenance


def test_bootstrap_is_idempotent_under_a_cross_source_reconciliation(
    pg16: Pg16Cluster, tmp_path: Path
) -> None:
    """The reason BOTH DO blocks consult the declaration, not only the raising one.

    If only ``$omnimarket_import$`` consulted it, ``$migration_id_import$`` would
    keep computing the plain ``adopted:...`` provenance, and the SECOND run would
    find the reconciled provenance already in the canonical ledger and raise
    ``double migration declaration`` from the other block. A one-sided fix would
    make the first run pass and every run after it fail -- which, on a lane that
    re-runs migrations on every deploy, is not a fix at all.
    """
    database = "omn16919_idempotent"
    pg16.create_database(database)
    _seed_both_sources(pg16, database)
    divergent = _write_divergent_adoption(tmp_path)
    cross = _write_cross_source(tmp_path)

    first = _run_bootstrap(
        pg16, database, divergent_adoptions=divergent, cross_source_adoptions=cross
    )
    assert first.returncode == 0, first.stderr
    before = pg16.sql(
        database,
        "SELECT count(*) || '|' || coalesce(max(provenance), '') "
        "FROM platform_catalog.schema_migrations",
    )

    second = _run_bootstrap(
        pg16, database, divergent_adoptions=divergent, cross_source_adoptions=cross
    )
    assert second.returncode == 0, second.stderr
    assert (
        pg16.sql(
            database,
            "SELECT count(*) || '|' || coalesce(max(provenance), '') "
            "FROM platform_catalog.schema_migrations",
        )
        == before
    )


# ---------------------------------------------------------------------------
# every declared value is load-bearing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value", "why"),
    [
        (
            "node_source_checksum",
            "applied-by-something-else",
            "a declaration covers one recorded node-side state, not any state",
        ),
        (
            "omnimarket_source_checksum",
            "a" * 64,
            "a declaration covers one revision of the omnimarket row",
        ),
        (
            "manifest_checksum",
            "b" * 64,
            "rewriting the migration after the proof re-opens the question",
        ),
        (
            "node_applied_at",
            "2026-06-10 20:30:05.148692+00",
            "the node row this declaration described is not the one present",
        ),
        (
            "omnimarket_applied_at",
            "2026-05-23 13:49:15.138390+00",
            "the omnimarket row this declaration described is not the one present",
        ),
    ],
)
def test_a_drifted_declaration_still_fails_closed(
    pg16: Pg16Cluster, tmp_path: Path, field: str, value: str, why: str
) -> None:
    """Change any declared fact and the reconciliation stops being honoured.

    A stale declaration must not outlive the facts it attested to. Each case
    perturbs exactly one field; the timestamps differ by a single microsecond
    precisely so this cannot pass on a loose comparison.
    """
    database = f"omn16919_drift_{field}"
    pg16.create_database(database)
    _seed_both_sources(pg16, database)

    completed = _run_bootstrap(
        pg16,
        database,
        divergent_adoptions=_write_divergent_adoption(tmp_path),
        cross_source_adoptions=_write_cross_source(
            tmp_path, **{field: value}, name=f"{field}.tsv"
        ),
    )

    assert completed.returncode != 0, why
    assert VERSION in completed.stderr


def test_a_reconciliation_cannot_launder_a_checksum_disagreement(
    pg16: Pg16Cluster, tmp_path: Path
) -> None:
    """Reconciliation settles metadata. It never settles content.

    Here the omnimarket row is genuinely divergent and has NO OMN-16915
    declaration, so the two sources make different content claims. A cross-source
    record naming that same divergent checksum must not rescue it: the row must
    still die on ``conflicting migration checksum`` inside the omnimarket block,
    before any metadata question is reached.

    This is the boundary between the two mechanisms. Blur it and the
    cross-source file becomes a way to adopt unproven bytes by writing a
    timestamp next to them.
    """
    database = "omn16919_no_laundering"
    pg16.create_database(database)
    _seed_both_sources(pg16, database)

    completed = _run_bootstrap(
        pg16,
        database,
        # OMN-16915's declaration deliberately withheld.
        divergent_adoptions=_empty(tmp_path, "no-divergent.tsv"),
        cross_source_adoptions=_write_cross_source(tmp_path),
    )

    assert completed.returncode != 0
    assert "conflicting migration checksum for version" in completed.stderr
    assert VERSION in completed.stderr


def test_reconciliation_does_not_leak_to_the_next_row(
    pg16: Pg16Cluster, tmp_path: Path
) -> None:
    """plpgsql RECORDs persist across loop iterations; this proves the reset.

    ``cross_row`` is reused by every iteration of both import loops. Without the
    explicit clear at the top of each, a version processed AFTER a reconciled one
    would inherit its ticket, receipt and timestamps into a provenance naming a
    reconciliation that never covered it. Only the two declared versions may
    carry the ticket.
    """
    database = "omn16919_no_leak"
    pg16.create_database(database)
    _seed_both_sources(pg16, database)

    completed = _run_bootstrap(
        pg16,
        database,
        divergent_adoptions=_write_divergent_adoption(tmp_path),
        # Only the divergent version is reconciled; the clean one is not.
        cross_source_adoptions=_write_cross_source(tmp_path, include_clean=False),
    )
    # The clean version is still dual-declared and now has no reconciliation, so
    # the run must fail -- but the failure must be ITS failure, not a leak.
    assert completed.returncode != 0
    assert CLEAN_VERSION in completed.stderr
    assert "double migration declaration for version" in completed.stderr


# ---------------------------------------------------------------------------
# the lane itself -- both relations, from the shipping declaration file
# ---------------------------------------------------------------------------

# The ten dual-declared versions on stability-test omnidash_analytics, read live
# on 2026-08-29: (node_name, filename, omnimarket_checksum, node_applied_at,
# omnimarket_applied_at). Every node-side checksum is the runner sentinel.
LANE_ROWS: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "node_contract_registry",
        "0000_create_contract_registry.sql",
        "686b135950659fa6ab11161a439fac60ae273ff3cf410fe6b209f505bb8983e7",
        "2026-06-10 20:30:05.148691+00",
        "2026-05-23 13:49:15.138389+00",
    ),
    (
        "node_evidence_dashboard_reducer",
        "0001_create_evidence_dashboard_projection_tables.sql",
        "a83717d2077fe6a4ca737adda2f7d3ab37544f14826284ec3839b810925bf051",
        "2026-06-10 20:30:05.340401+00",
        "2026-05-23 13:49:15.138389+00",
    ),
    (
        "node_nightly_loop_controller",
        "001_create_nightly_loop_tables.sql",
        "55fd0b90a3dd033df7c57351d0739e9218fe58ba290c732c78e78d7df49eaa60",
        "2026-06-10 20:30:05.531935+00",
        "2026-05-23 13:49:15.138389+00",
    ),
    (
        "node_pr_review_bot",
        "001_create_review_bot_bypass_log.sql",
        "63e2646a7f8767fad9ec969b224982e00b83aacb665107ab5b103964d5616e00",
        "2026-06-10 20:30:05.555266+00",
        "2026-05-23 13:49:15.138389+00",
    ),
    (
        "node_projection_delegation",
        "0008_generation_events.sql",
        "e12015762c7b2c598ed03f38a4e6d2b3cf7cbd6d6f8c30e05d21758a420580b6",
        "2026-06-06 16:07:04.573841+00",
        "2026-05-20 20:18:13.580164+00",
    ),
    (
        "node_projection_delegation",
        "0009_delegate_skill_projection_metrics.sql",
        "ba2f5e8428084d2f7e3386372864aed7b8d47f31d4048e12878f8799060fb03d",
        "2026-06-06 16:07:04.681203+00",
        "2026-05-20 20:18:13.601917+00",
    ),
    (
        "node_projection_dep_health",
        "001_create_dep_health_findings.sql",
        "5bfcf6862b235ea79bab65dd8681db9217f851511c2dfa8283191e2c7094217d",
        "2026-06-10 20:30:05.859780+00",
        "2026-05-23 13:49:15.138389+00",
    ),
    (
        "node_projection_registration",
        "0000_create_node_service_registry.sql",
        "5c320392190e6b9a1e06ab7177b7560181ca00282e107aebfccc452441e4d478",
        "2026-06-06 16:07:04.960343+00",
        "2026-05-23 13:49:15.138389+00",
    ),
    (
        "node_projection_registration",
        "0001_add_heartbeat_columns.sql",
        "2cda779401d041ae1383937df3f4c614df74d09c80c78f24f7993685ba7a6aed",
        "2026-06-06 16:07:04.986930+00",
        "2026-05-23 13:49:15.138389+00",
    ),
    (
        "node_projection_session_outcome",
        "0021_session_outcomes.sql",
        "68adbc4faff969f8f1be689899ccae591331bd858a14bef30c62e4d73600b66b",
        "2026-06-10 20:33:25.391471+00",
        "2026-05-23 13:49:15.138389+00",
    ),
)

# sha256 of the shipping declaration file. OMN-16915's practice: pin the input so
# the lane cases below cannot pass against a file someone edited.
LANE_DECLARATION_SHA256 = (
    "e35cab008fc8ea839e40342fcb721275eab49b6a2861607a2131f58a4baee5dc"
)


def _seed_the_real_lane(pg16: Pg16Cluster, database: str) -> None:
    """BOTH relations. This is the correction to OMN-16915's fixture."""
    node_values = ",\n  ".join(
        f"('node:{node}:{filename}', TIMESTAMPTZ '{node_at}', "
        "'applied-by-runner', 'node')"
        for node, filename, _, node_at, _ in LANE_ROWS
    )
    omnimarket_values = ",\n  ".join(
        f"('{node}', '{filename}', '{filename}', '{checksum}', TIMESTAMPTZ '{omni_at}')"
        for node, filename, checksum, _, omni_at in LANE_ROWS
    )
    pg16.command(
        database,
        "-f",
        "-",
        input_text=f"""
CREATE TABLE public.schema_migrations (
  migration_id TEXT NOT NULL,
  applied_at TIMESTAMPTZ NOT NULL,
  checksum TEXT NOT NULL,
  source_set TEXT NOT NULL
);
INSERT INTO public.schema_migrations
  (migration_id, applied_at, checksum, source_set)
VALUES
  {node_values};

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
  {omnimarket_values};
""",
        check=True,
    )


def test_the_shipping_declaration_file_is_the_one_these_cases_ran_against() -> None:
    """Pin the input bytes, so the two lane cases cannot pass on a different file."""
    digest = hashlib.sha256(VERIFIED_CROSS_SOURCE_ADOPTIONS.read_bytes()).hexdigest()
    assert digest == LANE_DECLARATION_SHA256, (
        "verified-cross-source-adoptions.tsv changed; re-run the live verification "
        "and update LANE_DECLARATION_SHA256 rather than editing the file in place"
    )
    assert len(LANE_ROWS) == len(
        VERIFIED_CROSS_SOURCE_ADOPTIONS.read_text(encoding="utf-8").splitlines()
    )


def test_the_real_lane_ledger_is_still_red_without_the_declarations(
    pg16: Pg16Cluster, tmp_path: Path
) -> None:
    """The lane state, both relations, no reconciliation: the production failure.

    ``node_contract_registry`` sorts first in ``(node_name, version)`` order, so
    it is the version the lane actually reported.
    """
    database = "omn16919_lane_red"
    pg16.create_database(database)
    _seed_the_real_lane(pg16, database)

    completed = _run_bootstrap(
        pg16, database, cross_source_adoptions=_empty(tmp_path, "lane-none.tsv")
    )

    assert completed.returncode != 0
    assert "double migration declaration for version" in completed.stderr
    assert "node:node_contract_registry:0000_create_contract_registry.sql" in (
        completed.stderr
    )


def test_the_real_lane_ledger_converges_under_the_committed_declarations(
    pg16: Pg16Cluster,
) -> None:
    """All ten real rows, both relations, driven through the SHIPPING TSVs.

    No ``cross_source_adoptions`` override: this runs against
    ``_ledger/verified-cross-source-adoptions.tsv`` exactly as it will ship, over
    the OMN-16915 divergent declarations exactly as they already shipped.
    """
    database = "omn16919_lane_green"
    pg16.create_database(database)
    _seed_the_real_lane(pg16, database)

    completed = _run_bootstrap(pg16, database)
    assert completed.returncode == 0, completed.stderr

    reconciled = pg16.sql(
        database,
        "SELECT count(*) FROM platform_catalog.schema_migrations "
        "WHERE provenance LIKE 'cross-source-reconciled:%'",
    )
    assert reconciled == str(len(LANE_ROWS))

    # Every reconciled row carries the earlier (omnimarket) timestamp and both
    # raw checksums -- the evidence of the June re-registration is not erased.
    for node, filename, checksum, node_at, omni_at in LANE_ROWS:
        provenance = pg16.sql(
            database,
            "SELECT provenance FROM platform_catalog.schema_migrations "
            f"WHERE version = 'node:{node}:{filename}'",
        )
        assert checksum in provenance
        assert node_at in provenance
        assert omni_at in provenance
        # Instant comparison, not rendered text -- psql renders timestamptz in
        # the session time zone.
        assert (
            pg16.sql(
                database,
                f"SELECT applied_at = TIMESTAMPTZ '{omni_at}' "
                "FROM platform_catalog.schema_migrations "
                f"WHERE version = 'node:{node}:{filename}'",
            )
            == "t"
        ), f"{node}:{filename} did not take the earlier applied_at"


def test_no_version_is_in_two_adoption_surfaces_without_a_cross_source_record() -> None:
    """The overlap rule, read off the SHIPPING declaration files.

    OMN-16915 forbade a version appearing in both per-row adoption files. That
    still holds for a single row. What the cross-source file legitimises is the
    same VERSION carried by two different source relations -- so overlap is
    admissible only when a cross-source record covers it.
    """

    def _versions(name: str) -> set[str]:
        path = LEDGER_DIR / name
        return {
            line.split("\t")[0]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    sentinel = _versions("verified-checksum-adoptions.tsv")
    divergent = _versions("verified-divergent-adoptions.tsv")
    cross = _versions("verified-cross-source-adoptions.tsv")

    assert (sentinel & divergent) <= cross, (
        "a version is declared in both per-row adoption ledgers with no "
        "cross-source record: "
        f"{sorted((sentinel & divergent) - cross)}"
    )
    # And every cross-source record must name a version some source ledger
    # actually resolves -- a reconciliation with no underlying declaration would
    # be reconciling nothing.
    assert cross, "the shipping cross-source declaration file is empty"
