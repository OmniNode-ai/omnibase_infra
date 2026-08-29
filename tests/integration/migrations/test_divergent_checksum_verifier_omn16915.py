# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The equivalence proof behind a verified DIVERGENT adoption (OMN-16915).

``_ledger/verified-divergent-adoptions.tsv`` lets ``bootstrap.sql`` adopt the
manifest checksum for an omnimarket ledger row whose stored checksum is a real,
well-formed sha256 that simply is not the manifest's. bootstrap.sql cannot check
the proof behind that declaration -- it compares checksums and never looks at a
table -- so the load-bearing assertions live here.

Two things must hold, and the second is the one that matters:

1. The verifier must SEE these rows at all. OMN-15857's
   ``discover_non_canonical_rows`` cannot: it reads ``public.schema_migrations``
   only, and selects only checksums that are NOT 64-hex. A divergent-bytes row
   fails both filters, which is exactly why the stability lane hit an
   unconsultable raise at bootstrap.sql:876 immediately after the OMN-15857 rows
   were cleared.
2. The verdict must still be earned by execution. The divergent case is
   *seductive* in a way the sentinel case was not: the six rows on the .201
   stability lane have a spotless provenance story -- one commit (OMN-15376
   #2537, 2026-07-29), a diff containing nothing but guarded
   ``ADD COLUMN IF NOT EXISTS`` / ``SET NOT NULL`` / ``ADD CONSTRAINT``
   statements written to be no-ops on the fresh-create path. It would be very
   easy to accept that story as proof. It is not proof. "These bytes are an
   older revision of this file" is a claim about provenance; the ledger's claim
   is about *schema*. Only running both revisions answers it.

So ``divergent_verified`` must behave exactly like ``equivalent`` did: earned by
a clean surface diff, and refused the moment the live schema actually differs. A
verifier that returned ``divergent_verified`` on the strength of the git history
would pass a naive test and adopt a genuinely drifted lane.

Ticket: OMN-16915
"""

# ruff: noqa: S608 -- every interpolated value is a checked-in manifest
# constant or a literal defined in this file; there is no untrusted input here.

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

from tests.integration.migrations.test_application_migration_ledger_omn15413 import (
    Pg16Cluster,
)

pytestmark = [pytest.mark.integration, pytest.mark.postgres, pytest.mark.serial]

REPO_ROOT = Path(__file__).resolve().parents[3]
VERIFIER_PATH = (
    REPO_ROOT / "scripts" / "migrations" / "verify_migration_checksum_adoption.py"
)

NODE = "node_contract_registry"
FILENAME = "0000_create_contract_registry.sql"
VERSION = f"node:{NODE}:{FILENAME}"
# The revision the .201 stability lane actually applied -- the artifact bytes as
# of 5b904d881 (2026-07-21), superseded by OMN-15376 #2537 on 2026-07-29.
DIVERGENT = "686b135950659fa6ab11161a439fac60ae273ff3cf410fe6b209f505bb8983e7"
# The commit whose tree carries exactly those bytes: the last state before
# OMN-15376 (#2537) rewrote six node migrations in place on 2026-07-29.
PRIOR_REVISION = "5b904d881ba51a697e5b3d50b28460abbb2fd5aa"


def _load_verifier() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "verify_migration_checksum_adoption_omn16915", VERIFIER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


verifier = _load_verifier()


def _counter() -> Iterator[int]:
    value = 0
    while True:
        yield value
        value += 1


_COUNTER = _counter()


@pytest.fixture
def audited(pg16: Pg16Cluster) -> tuple[str, verifier.PsqlClient]:
    """A database standing in for a live lane, plus a client pointed at it."""
    database = f"omn16915_audited_{next(_COUNTER)}"
    pg16.create_database(database)
    client = verifier.PsqlClient(
        argv=(
            str(pg16.bin_dir / "psql"),
            "-h",
            "127.0.0.1",
            "-p",
            str(pg16.port),
            "-U",
            "postgres",
        ),
        label="audited",
    )
    return database, client


def _target_migration() -> Path:
    replay = verifier.resolve_replay_set(VERSION)
    assert replay is not None, "the node migration under test must be checked in"
    target, _ = replay
    return target


def _apply_faithfully(pg16: Pg16Cluster, database: str) -> None:
    """Put the database in the state a correct apply of the CURRENT file leaves."""
    pg16.command(
        database, "-f", "-", input_text=_target_migration().read_text(encoding="utf-8")
    )


def _seed_omnimarket_ledger(
    pg16: Pg16Cluster, database: str, *, checksum: str = DIVERGENT
) -> None:
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
  ('{NODE}', '{FILENAME}', '{FILENAME}', '{checksum}',
   TIMESTAMPTZ '2026-05-23 13:49:15.138389+00');
""",
        check=True,
    )


def _verify(database: str, client: verifier.PsqlClient) -> verifier.RowVerdict:
    bin_dir = verifier._postgres_bin_dir()
    assert bin_dir is not None, "a local Postgres is required to replay migrations"
    with tempfile.TemporaryDirectory(prefix="omn16915-scratch-") as tmp:
        scratch = verifier.ScratchServer(bin_dir, Path(tmp))
        scratch.start()
        try:
            return verifier.verify_row(
                version=VERSION,
                source_checksum=DIVERGENT,
                source_set="omnimarket",
                database=database,
                success_verdict=verifier.VERDICT_DIVERGENT_VERIFIED,
                live=client,
                scratch=scratch,
                manifest=verifier.load_manifest(),
                legacy=verifier.load_legacy_declarations(),
            )
        finally:
            scratch.stop()


def _manifest_checksum() -> str:
    return verifier.load_manifest()[VERSION]["checksum"]


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def test_the_sentinel_discovery_query_cannot_see_this_row(
    pg16: Pg16Cluster, audited: tuple[str, verifier.PsqlClient]
) -> None:
    """Pins WHY OMN-15857's machinery was structurally inapplicable.

    If a future refactor makes ``discover_non_canonical_rows`` start returning
    divergent-bytes rows, the two admission paths have been merged by accident
    and the separate declaration files stop protecting anything.
    """
    database, client = audited
    _seed_omnimarket_ledger(pg16, database)
    # The sentinel discovery needs its own source table to exist at all.
    pg16.command(
        database,
        "-c",
        "CREATE TABLE public.schema_migrations ("
        "migration_id TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
        "checksum TEXT NOT NULL, source_set TEXT NOT NULL)",
        check=True,
    )

    assert verifier.discover_non_canonical_rows(client, database) == []


def test_divergent_discovery_finds_the_row_the_lane_died_on(
    pg16: Pg16Cluster, audited: tuple[str, verifier.PsqlClient]
) -> None:
    database, client = audited
    _seed_omnimarket_ledger(pg16, database)

    rows = verifier.discover_divergent_omnimarket_rows(
        client, database, verifier.load_manifest()
    )

    assert rows == [(VERSION, DIVERGENT, "omnimarket")]


def test_a_row_agreeing_with_the_manifest_is_not_discovered(
    pg16: Pg16Cluster, audited: tuple[str, verifier.PsqlClient]
) -> None:
    """Only a genuine disagreement is this tool's business.

    bootstrap.sql already accepts an agreeing row directly; surfacing it here
    would invite a declaration asserting a proof for a question nobody asked.
    """
    database, client = audited
    _seed_omnimarket_ledger(pg16, database, checksum=_manifest_checksum())

    assert (
        verifier.discover_divergent_omnimarket_rows(
            client, database, verifier.load_manifest()
        )
        == []
    )


def test_an_absent_omnimarket_ledger_is_not_an_error(
    audited: tuple[str, verifier.PsqlClient],
) -> None:
    """Most databases have no omnimarket ledger; that is not a finding."""
    database, client = audited

    assert (
        verifier.discover_divergent_omnimarket_rows(
            client, database, verifier.load_manifest()
        )
        == []
    )


# ---------------------------------------------------------------------------
# the proof itself
# ---------------------------------------------------------------------------


def test_a_faithfully_applied_schema_is_divergent_verified(
    pg16: Pg16Cluster, audited: tuple[str, verifier.PsqlClient]
) -> None:
    """The adoptable verdict, and it is NOT spelled ``equivalent``.

    The distinct spelling is what keeps a divergent-bytes proof out of
    OMN-15857's sentinel ledger, where bootstrap.sql's ``$migration_id_import$``
    block would consult it for a question it never answered.
    """
    database, client = audited
    _apply_faithfully(pg16, database)

    verdict = _verify(database, client)

    assert verdict.verdict == verifier.VERDICT_DIVERGENT_VERIFIED
    assert verdict.verdict != verifier.VERDICT_EQUIVALENT
    assert verdict.divergences == []
    assert verdict.declared_objects, "an empty surface cannot prove anything"
    assert verdict.manifest_checksum == _manifest_checksum()
    assert verdict.source_checksum == DIVERGENT
    # Only this verdict may be written to the divergent ledger.
    assert verifier.VERDICT_DIVERGENT_VERIFIED in verifier.DIVERGENT_ADOPTABLE_VERDICTS
    assert verifier.VERDICT_DIVERGENT_VERIFIED not in verifier.ADOPTABLE_VERDICTS, (
        "a divergent proof must never satisfy the sentinel adoption path"
    )


def test_the_prior_revision_actually_produces_this_schema(
    pg16: Pg16Cluster, audited: tuple[str, verifier.PsqlClient]
) -> None:
    """The claim the whole ticket rests on, executed rather than asserted.

    The .201 lane applied the PRE-OMN-15376 revision of this file. The proof
    above replays the CURRENT revision and diffs it against the lane. That is
    only a valid adoption if the older revision really does leave the same
    schema behind -- which is a fact about two SQL files, not about a diff that
    looks harmless. So run the old bytes and demand the current file's surface.

    If OMN-15376's guarded reconciliation had changed any observable shape, this
    is the test that would have caught it, and the correct outcome would have
    been STOP-and-report rather than adoption.
    """
    database, client = audited
    artifact = Path("docker/migrations/forward/nodes") / NODE / FILENAME

    # Fetch the historical revision byte-for-byte out of git.
    revision = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", f"{PRIOR_REVISION}:{artifact}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if revision.returncode != 0:
        pytest.skip(
            f"commit {PRIOR_REVISION[:12]} is not present in this clone "
            "(shallow checkout); the historical bytes cannot be replayed"
        )
    old_sql = revision.stdout

    # Without this the test could silently apply the CURRENT bytes and pass for
    # the wrong reason.
    assert hashlib.sha256(old_sql.encode()).hexdigest() == DIVERGENT, (
        "this test is only meaningful if it applies the exact bytes the lane's "
        "checksum names"
    )
    pg16.command(database, "-f", "-", input_text=old_sql, check=True)

    verdict = _verify(database, client)

    assert verdict.verdict == verifier.VERDICT_DIVERGENT_VERIFIED, (
        f"the revision the lane applied does NOT produce the current file's "
        f"schema: {verdict.divergences}"
    )


def test_a_drifted_schema_is_refused_not_adopted(
    pg16: Pg16Cluster, audited: tuple[str, verifier.PsqlClient]
) -> None:
    """A clean provenance story does not rescue a genuinely drifted lane.

    This is the case the ticket calls STOP-and-report: real divergence is a
    decision, not an adoption.
    """
    database, client = audited
    _apply_faithfully(pg16, database)
    pg16.command(
        database, "-c", "ALTER TABLE contract_registry DROP COLUMN target_profile"
    )

    verdict = _verify(database, client)

    assert verdict.verdict == verifier.VERDICT_DIVERGENT
    assert verdict.verdict not in verifier.DIVERGENT_ADOPTABLE_VERDICTS
    assert verdict.divergences


def test_a_table_that_was_never_created_is_refused(
    pg16: Pg16Cluster, audited: tuple[str, verifier.PsqlClient]
) -> None:
    """An empty lane must not read as 'nothing differs'."""
    database, client = audited

    verdict = _verify(database, client)

    assert verdict.verdict == verifier.VERDICT_DIVERGENT
    assert verdict.verdict not in verifier.DIVERGENT_ADOPTABLE_VERDICTS


def test_a_widened_column_type_is_refused(
    pg16: Pg16Cluster, audited: tuple[str, verifier.PsqlClient]
) -> None:
    """Type drift is invisible to a column-presence check; pin it explicitly."""
    database, client = audited
    _apply_faithfully(pg16, database)
    pg16.command(
        database,
        "-c",
        "ALTER TABLE contract_registry ALTER COLUMN node_version TYPE TEXT "
        "USING node_version::text",
    )

    verdict = _verify(database, client)

    assert verdict.verdict == verifier.VERDICT_DIVERGENT
    assert verdict.divergences


# ---------------------------------------------------------------------------
# the committed declarations
# ---------------------------------------------------------------------------


def test_every_committed_divergent_adoption_matches_its_committed_receipt() -> None:
    """Each declaration must resolve to an in-tree proof that says so.

    Without this the receipt hash is decoration: a row could cite any 64 hex
    characters. The check binds every declaration to a receipt that exists in
    the tree, covers that exact version, reached the only adoptable verdict, and
    was run against the same two checksums the row declares.
    """
    adoptions = verifier.load_divergent_adoptions()
    assert adoptions, "the lane-blocking rows must be declared"

    receipts_dir = verifier.LEDGER_DIR / "receipts"
    for version, declared in sorted(adoptions.items()):
        matching = [
            path
            for path in sorted(receipts_dir.glob("*.json"))
            if verifier.file_sha256(path) == declared["receipt_sha256"]
        ]
        assert matching, (
            f"no receipt in {receipts_dir} hashes to "
            f"{declared['receipt_sha256']}; {version} cites a proof that is not "
            "in the tree"
        )

        receipt = json.loads(matching[0].read_text(encoding="utf-8"))
        entry = next(item for item in receipt["verdicts"] if item["version"] == version)
        assert entry["verdict"] == verifier.VERDICT_DIVERGENT_VERIFIED
        assert entry["source_checksum"] == declared["source_checksum"]
        assert entry["manifest_checksum"] == declared["manifest_checksum"]
        assert entry["declared_objects"], "an empty surface cannot prove anything"
        assert entry["divergences"] == []


def test_committed_declarations_pin_the_current_manifest_content() -> None:
    """A declaration proven against different bytes than the manifest now holds
    is stale, and must fail here rather than at deploy time on the lane."""
    manifest = verifier.load_manifest()
    for version, declared in sorted(verifier.load_divergent_adoptions().items()):
        assert version in manifest, f"{version} has no manifest declaration"
        assert declared["manifest_checksum"] == manifest[version]["checksum"], (
            f"{version} was proven against {declared['manifest_checksum']} but "
            f"the manifest now declares {manifest[version]['checksum']}"
        )
        # The whole premise of this ledger: the applied hash really does differ.
        assert declared["source_checksum"] != declared["manifest_checksum"]


def test_no_version_is_declared_in_both_adoption_ledgers() -> None:
    """A version has one applied checksum; it cannot be both an unproven
    hand-apply (OMN-15857) and a stale revision (OMN-16915)."""
    overlap = set(verifier.load_adoptions()) & set(verifier.load_divergent_adoptions())
    assert overlap == set(), f"declared in both adoption ledgers: {sorted(overlap)}"
