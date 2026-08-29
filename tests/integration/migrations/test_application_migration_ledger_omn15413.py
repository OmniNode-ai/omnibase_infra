# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""PostgreSQL 16 proof for OMN-15413 ledger selection and import semantics.

OMN-15695 extends this module with the adopt/convert proof for the
pre-OMN-15413 ``public.schema_migrations(migration_id, applied_at, checksum,
source_set)`` node ledger written by the historical runner into the application
database.  Operator ruling 2026-08-04: adopt that exact shape non-destructively;
every other unrecognized shape stays fail-closed.
"""

# ruff: noqa: S608 -- all interpolated SQL values are checked-in manifest data.

from __future__ import annotations

import hashlib
import os
import shutil
import socket
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.postgres, pytest.mark.serial]

REPO_ROOT = Path(__file__).resolve().parents[3]
LEDGER_DIR = REPO_ROOT / "docker" / "migrations" / "forward" / "_ledger"
MANIFEST = LEDGER_DIR / "application-migrations.tsv"
LEGACY_NODE_DECLARATIONS = LEDGER_DIR / "legacy-node-migrations.tsv"
VERIFIED_ADOPTIONS = LEDGER_DIR / "verified-checksum-adoptions.tsv"
BOOTSTRAP = LEDGER_DIR / "bootstrap.sql"
RUNNER = REPO_ROOT / "scripts" / "run-forward-migrations.sh"


def _postgres_bin_dir() -> Path | None:
    initdb = shutil.which("initdb")
    candidates = [Path(initdb).parent] if initdb else []
    candidates.extend(
        sorted(Path("/opt/homebrew/opt").glob("postgresql@*/bin"), reverse=True)
    )
    candidates.extend(sorted(Path("/usr/lib/postgresql").glob("*/bin"), reverse=True))
    for candidate in candidates:
        required = ("initdb", "pg_ctl", "psql")
        if not all((candidate / binary).is_file() for binary in required):
            continue
        version = subprocess.run(
            [str(candidate / "psql"), "--version"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        if " 16." in version:
            return candidate
    return None


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass(frozen=True)
class Pg16Cluster:
    bin_dir: Path
    port: int

    @property
    def psql(self) -> str:
        return str(self.bin_dir / "psql")

    def command(
        self,
        database: str,
        *arguments: str,
        input_text: str | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                self.psql,
                "-X",
                "-q",
                "-h",
                "127.0.0.1",
                "-p",
                str(self.port),
                "-U",
                "postgres",
                "-d",
                database,
                "-v",
                "ON_ERROR_STOP=1",
                *arguments,
            ],
            input=input_text,
            capture_output=True,
            text=True,
            timeout=30,
            check=check,
            env={**os.environ, "PGPASSWORD": ""},
        )

    def sql(self, database: str, statement: str) -> str:
        return self.command(database, "-At", "-c", statement).stdout.strip()

    def create_database(self, database: str) -> None:
        self.command("postgres", "-c", f"CREATE DATABASE {database}")


@pytest.fixture(scope="module")
def pg16(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Pg16Cluster]:
    bin_dir = _postgres_bin_dir()
    if bin_dir is None:
        pytest.skip("PostgreSQL 16 initdb/pg_ctl/psql are unavailable")

    cluster_dir = tmp_path_factory.mktemp("omn15413-pg16")
    data_dir = cluster_dir / "data"
    port = _free_port()
    init = subprocess.run(
        [
            str(bin_dir / "initdb"),
            "-D",
            str(data_dir),
            "-U",
            "postgres",
            "--auth=trust",
            "--no-sync",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if init.returncode != 0:
        pytest.fail(f"PostgreSQL 16 initdb failed: {init.stderr}")

    start = subprocess.run(
        [
            str(bin_dir / "pg_ctl"),
            "-D",
            str(data_dir),
            "-o",
            f"-F -h 127.0.0.1 -p {port}",
            "-l",
            str(cluster_dir / "postgres.log"),
            "-w",
            "start",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if start.returncode != 0:
        postgres_log = cluster_dir / "postgres.log"
        log_text = (
            postgres_log.read_text(encoding="utf-8", errors="replace")
            if postgres_log.is_file()
            else ""
        )
        pytest.skip(
            "PostgreSQL 16 binaries are present but an ephemeral cluster could "
            f"not start; pg_ctl stderr={start.stderr!r}; postgres.log={log_text!r}"
        )

    try:
        yield Pg16Cluster(bin_dir=bin_dir, port=port)
    finally:
        subprocess.run(
            [
                str(bin_dir / "pg_ctl"),
                "-D",
                str(data_dir),
                "-m",
                "immediate",
                "-w",
                "stop",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )


def _declarations() -> list[list[str]]:
    return [line.split("\t") for line in MANIFEST.read_text().splitlines()]


def _run_bootstrap(
    pg16: Pg16Cluster, database: str, *, adoptions: Path | None = None
) -> subprocess.CompletedProcess[str]:
    create_manifest = """
CREATE TEMP TABLE onex_application_migration_manifest (
  artifact_path TEXT NOT NULL,
  migration_stream TEXT NOT NULL,
  owner TEXT NOT NULL,
  domain TEXT NOT NULL,
  version TEXT NOT NULL,
  checksum TEXT NOT NULL,
  PRIMARY KEY (artifact_path),
  UNIQUE (migration_stream, domain, version)
)
"""
    return pg16.command(
        database,
        "-c",
        create_manifest,
        "-c",
        """
CREATE TEMP TABLE onex_legacy_node_migration_declarations (
  migration_stream TEXT NOT NULL,
  owner TEXT NOT NULL,
  domain TEXT NOT NULL,
  version TEXT NOT NULL PRIMARY KEY,
  source_checksum TEXT NOT NULL,
  ticket TEXT NOT NULL
)
""",
        "-c",
        """
CREATE TEMP TABLE onex_verified_checksum_adoptions (
  version TEXT NOT NULL PRIMARY KEY,
  source_checksum TEXT NOT NULL,
  manifest_checksum TEXT NOT NULL,
  ticket TEXT NOT NULL,
  receipt_sha256 TEXT NOT NULL,
  verified_at TEXT NOT NULL
)
""",
        "-c",
        (
            "\\copy onex_application_migration_manifest "
            f"FROM '{MANIFEST}' WITH (FORMAT text, DELIMITER E'\\t')"
        ),
        "-c",
        (
            "\\copy onex_legacy_node_migration_declarations "
            f"FROM '{LEGACY_NODE_DECLARATIONS}' WITH (FORMAT text, DELIMITER E'\\t')"
        ),
        "-c",
        (
            "\\copy onex_verified_checksum_adoptions "
            f"FROM '{adoptions or VERIFIED_ADOPTIONS}' WITH (FORMAT text, DELIMITER E'\\t')"
        ),
        "-f",
        str(BOOTSTRAP),
        check=False,
    )


def _seed_sources(
    pg16: Pg16Cluster,
    database: str,
    *,
    node_version: str,
    node_checksum: str,
    omnimarket: list[str] | None = None,
) -> None:
    omnimarket_sql = ""
    if omnimarket is not None:
        artifact, _, _, _, version, checksum = omnimarket
        node_name = artifact.split("/")[1]
        filename = artifact.split("/")[2]
        assert version == f"node:{node_name}:{filename}"
        omnimarket_sql = f"""
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
  ('{node_name}', '{filename}', '{filename}', '{checksum}',
   TIMESTAMPTZ '2026-01-03 00:00:00+00');
"""
    pg16.command(
        database,
        "-f",
        "-",
        input_text=f"""
CREATE TABLE public.schema_migrations (
  filename TEXT PRIMARY KEY,
  applied_at TIMESTAMPTZ NOT NULL
);
INSERT INTO public.schema_migrations VALUES
  ('0001_filename_only.sql', TIMESTAMPTZ '2026-01-01 00:00:00+00');
CREATE TABLE public.node_schema_migrations (
  version TEXT PRIMARY KEY,
  applied_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  checksum TEXT NOT NULL
);
INSERT INTO public.node_schema_migrations VALUES
  ('{node_version}', TIMESTAMPTZ '2026-01-02 00:00:00+00', '{node_checksum}');
{omnimarket_sql}
""",
    )


def test_selects_one_checksum_ledger_and_imports_sources_twice(
    pg16: Pg16Cluster,
) -> None:
    database = "omn15413_green"
    pg16.create_database(database)
    first, second = _declarations()[:2]
    _seed_sources(
        pg16,
        database,
        node_version=first[4],
        node_checksum=first[5],
        omnimarket=second,
    )
    node_oid = pg16.sql(
        database, "SELECT 'public.node_schema_migrations'::regclass::oid"
    )
    filename_oid = pg16.sql(
        database, "SELECT 'public.schema_migrations'::regclass::oid"
    )
    omnimarket_oid = pg16.sql(
        database, "SELECT 'public.omnimarket_schema_migrations'::regclass::oid"
    )

    signatures: list[str] = []
    for _ in range(2):
        run = _run_bootstrap(pg16, database)
        assert run.returncode == 0, run.stderr
        signatures.append(
            pg16.sql(
                database,
                """
SELECT string_agg(
  migration_stream || '|' || owner || '|' || domain || '|' || version ||
  '|' || checksum || '|' || checksum_kind || '|' || applied_at::text ||
  '|' || provenance,
  E'\n' ORDER BY migration_stream, domain, version
)
FROM platform_catalog.schema_migrations
""",
            )
        )

    assert signatures[0] == signatures[1]
    assert len(signatures[0].splitlines()) == 3
    assert (
        "legacy:filename-only|legacy:filename-only|legacy_unclassified"
        in (signatures[0])
    )
    assert signatures[0].count("|content_sha256|") == 2
    assert signatures[0].count("|legacy_attestation|") == 1
    assert (
        pg16.sql(database, "SELECT 'platform_catalog.schema_migrations'::regclass::oid")
        == node_oid
    )
    assert (
        pg16.sql(database, "SELECT 'public.schema_migrations'::regclass::oid")
        == filename_oid
    )
    assert (
        pg16.sql(
            database, "SELECT 'public.omnimarket_schema_migrations'::regclass::oid"
        )
        == omnimarket_oid
    )
    assert (
        pg16.sql(
            database, "SELECT to_regclass('public.node_schema_migrations') IS NULL"
        )
        == "t"
    )
    assert (
        pg16.sql(
            database,
            "SELECT count(*) FROM public.schema_migrations "
            "WHERE filename = '0001_filename_only.sql'",
        )
        == "1"
    )


@pytest.mark.parametrize(
    ("case_name", "version_factory", "checksum_factory", "signature"),
    [
        (
            "checksum_conflict",
            lambda row: row[4],
            lambda row: "0" * 64,
            "conflicting migration checksum",
        ),
        (
            "unknown_stream",
            lambda row: "node:unknown:0001.sql",
            lambda row: row[5],
            "unknown migration stream/domain",
        ),
    ],
)
def test_checksum_conflict_and_unknown_stream_are_atomic_reds(
    pg16: Pg16Cluster,
    case_name: str,
    version_factory: Callable[[list[str]], str],
    checksum_factory: Callable[[list[str]], str],
    signature: str,
) -> None:
    database = f"omn15413_{case_name}"
    pg16.create_database(database)
    first = _declarations()[0]
    version = version_factory(first)
    checksum = checksum_factory(first)
    _seed_sources(pg16, database, node_version=version, node_checksum=checksum)

    run = _run_bootstrap(pg16, database)

    assert run.returncode != 0
    assert signature in run.stderr
    assert (
        pg16.sql(
            database, "SELECT to_regclass('public.node_schema_migrations') IS NOT NULL"
        )
        == "t"
    )
    assert (
        pg16.sql(
            database,
            "SELECT to_regclass('platform_catalog.schema_migrations') IS NULL",
        )
        == "t"
    )


def test_node_and_omnimarket_double_declaration_is_atomic_red(
    pg16: Pg16Cluster,
) -> None:
    database = "omn15413_double_declaration"
    pg16.create_database(database)
    first = _declarations()[0]
    _seed_sources(
        pg16,
        database,
        node_version=first[4],
        node_checksum=first[5],
        omnimarket=first,
    )

    run = _run_bootstrap(pg16, database)

    assert run.returncode != 0
    assert "double migration declaration" in run.stderr
    assert (
        pg16.sql(
            database, "SELECT to_regclass('public.node_schema_migrations') IS NOT NULL"
        )
        == "t"
    )
    assert (
        pg16.sql(
            database,
            "SELECT to_regclass('platform_catalog.schema_migrations') IS NULL",
        )
        == "t"
    )


def _synthetic_migration_tree(tmp_path: Path) -> tuple[Path, str, str]:
    migrations_dir = tmp_path / "forward"
    ledger_dir = migrations_dir / "_ledger"
    node_dir = migrations_dir / "nodes" / "node_example"
    ledger_dir.mkdir(parents=True)
    node_dir.mkdir(parents=True)
    shutil.copy2(BOOTSTRAP, ledger_dir / "bootstrap.sql")

    (migrations_dir / "000_db_metadata.sql").write_text(
        """
CREATE TABLE IF NOT EXISTS public.db_metadata (
  id BOOLEAN PRIMARY KEY,
  migrations_complete BOOLEAN NOT NULL DEFAULT FALSE,
  runner_completed_at TIMESTAMPTZ,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
INSERT INTO public.db_metadata (id) VALUES (TRUE) ON CONFLICT (id) DO NOTHING;
""",
        encoding="utf-8",
    )
    node_file = node_dir / "0001_create_example.sql"
    node_file.write_text(
        "CREATE TABLE IF NOT EXISTS public.omn15413_example (id INTEGER PRIMARY KEY);\n",
        encoding="utf-8",
    )
    checksum = hashlib.sha256(node_file.read_bytes()).hexdigest()
    version = "node:node_example:0001_create_example.sql"
    (ledger_dir / "application-migrations.tsv").write_text(
        "\t".join(
            (
                "nodes/node_example/0001_create_example.sql",
                "node:node_example",
                "node:node_example",
                "omninode_internal",
                version,
                checksum,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (ledger_dir / "application-migration-blocks.tsv").write_text("", encoding="utf-8")
    (ledger_dir / "legacy-node-migrations.tsv").write_text("", encoding="utf-8")
    (ledger_dir / "verified-checksum-adoptions.tsv").write_text("", encoding="utf-8")
    (ledger_dir / "cloud-migration-aliases.tsv").write_text(
        "20260101_cloud\t20260101_cloud.sql\n", encoding="utf-8"
    )
    # OMN-15349: the runner unconditionally requires the single-sourced
    # operator fence manifest under MIGRATIONS_DIR; none of this synthetic
    # tree's ids need to be fenced, so an empty-baseline manifest suffices.
    (migrations_dir / "fenced-node-migrations.yaml").write_text(
        "fenced_node_migrations: []\n", encoding="utf-8"
    )
    # OMN-15336 item 4 added a second unconditionally-required manifest
    # alongside the fence, and this synthetic tree was never updated to stage
    # it -- so the runner exits 1 at "FATAL: FORCE-RLS grandfather manifest not
    # found" before reaching anything these tests assert about. Same reasoning
    # as the fence above: none of the synthetic ids are grandfathered, so an
    # empty baseline is the correct content, not a workaround.
    (migrations_dir / "grandfathered-force-rls-migrations.yaml").write_text(
        "grandfathered_force_rls_migrations: []\n", encoding="utf-8"
    )
    return migrations_dir, version, checksum


def _run_forward_runner(
    pg16: Pg16Cluster,
    migrations_dir: Path,
    *,
    service_database: str,
    application_database: str,
    cloud_database: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["sh", str(RUNNER)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env={
            **os.environ,
            "PATH": (
                f"{pg16.bin_dir}:/opt/homebrew/bin:/usr/local/bin:"
                "/usr/bin:/bin:/usr/sbin:/sbin"
            ),
            "POSTGRES_HOST": "127.0.0.1",
            "POSTGRES_PORT": str(pg16.port),
            "POSTGRES_USER": "postgres",
            "POSTGRES_PASSWORD": "",
            "POSTGRES_DB": service_database,
            "NODE_POSTGRES_DB": application_database,
            "OMNINODE_CLOUD_HISTORY_DB": cloud_database,
            "MIGRATIONS_DIR": str(migrations_dir),
            "PG_WAIT_RETRIES": "5",
            "MIGRATION_LOCK_WAIT_SECONDS": "10",
        },
    )


def test_real_runner_fresh_install_twice_on_postgresql_16(
    pg16: Pg16Cluster, tmp_path: Path
) -> None:
    migrations_dir, version, checksum = _synthetic_migration_tree(tmp_path)
    service_database = "omn15413_runner_fresh_service"
    application_database = "omn15413_runner_fresh_app"
    cloud_database = "omn15413_runner_fresh_cloud"
    for database in (service_database, application_database, cloud_database):
        pg16.create_database(database)

    runs = [
        _run_forward_runner(
            pg16,
            migrations_dir,
            service_database=service_database,
            application_database=application_database,
            cloud_database=cloud_database,
        )
        for _ in range(2)
    ]

    for run in runs:
        assert run.returncode == 0, f"{run.stdout}\n{run.stderr}"
        assert "Sentinel set. Migration gate will report HEALTHY." in run.stdout
    assert (
        "Complete: 0 infra applied, 1 infra skipped; 0 node applied, 1 node skipped"
        in (runs[1].stdout)
    )
    assert (
        pg16.sql(
            application_database,
            "SELECT checksum FROM platform_catalog.schema_migrations "
            f"WHERE version = '{version}'",
        )
        == checksum
    )
    assert (
        pg16.sql(
            application_database,
            "SELECT count(*) FROM platform_catalog.schema_migrations",
        )
        == "1"
    )


def test_real_runner_imports_sanitized_legacy_history_twice_on_postgresql_16(
    pg16: Pg16Cluster, tmp_path: Path
) -> None:
    migrations_dir, version, checksum = _synthetic_migration_tree(tmp_path)
    service_database = "omn15413_runner_legacy_service"
    application_database = "omn15413_runner_legacy_app"
    cloud_database = "omn15413_runner_legacy_cloud"
    for database in (service_database, application_database, cloud_database):
        pg16.create_database(database)

    pg16.command(
        application_database,
        "-f",
        "-",
        input_text=f"""
CREATE TABLE public.schema_migrations (
  filename TEXT PRIMARY KEY,
  applied_at TIMESTAMPTZ NOT NULL
);
INSERT INTO public.schema_migrations VALUES
  ('0001_legacy_dashboard.sql', TIMESTAMPTZ '2026-01-01 00:00:00+00');
CREATE TABLE public.node_schema_migrations (
  version TEXT PRIMARY KEY,
  applied_at TIMESTAMPTZ NOT NULL,
  checksum TEXT NOT NULL
);
INSERT INTO public.node_schema_migrations VALUES
  ('{version}', TIMESTAMPTZ '2026-01-02 00:00:00+00', '{checksum}');
""",
    )
    pg16.command(
        cloud_database,
        "-f",
        "-",
        input_text="""
CREATE TABLE public.schema_migrations (
  version TEXT PRIMARY KEY,
  applied_at TIMESTAMPTZ NOT NULL,
  checksum TEXT
);
INSERT INTO public.schema_migrations VALUES
  ('20260101_cloud.sql', TIMESTAMPTZ '2026-01-03 00:00:00+00', NULL);
CREATE TABLE public.migrations_log (
  id SERIAL PRIMARY KEY,
  migration_name TEXT NOT NULL,
  direction TEXT NOT NULL,
  executed_at TIMESTAMPTZ NOT NULL,
  notes TEXT,
  UNIQUE (migration_name, direction)
);
INSERT INTO public.migrations_log
  (migration_name, direction, executed_at, notes)
VALUES
  ('20260101_cloud', 'forward', TIMESTAMPTZ '2026-01-03 00:00:00+00',
   'synthetic fixture');
""",
    )
    source_signature = pg16.sql(
        cloud_database,
        "SELECT version || '|' || coalesce(checksum, '<NULL>') || '|' || "
        "applied_at::text FROM public.schema_migrations",
    )

    signatures: list[str] = []
    for _ in range(2):
        run = _run_forward_runner(
            pg16,
            migrations_dir,
            service_database=service_database,
            application_database=application_database,
            cloud_database=cloud_database,
        )
        assert run.returncode == 0, f"{run.stdout}\n{run.stderr}"
        signatures.append(
            pg16.sql(
                application_database,
                "SELECT string_agg(migration_stream || '|' || owner || '|' || "
                "domain || '|' || version || '|' || checksum || '|' || "
                "checksum_kind || '|' || applied_at::text || '|' || provenance, "
                "E'\\n' ORDER BY migration_stream, domain, version) "
                "FROM platform_catalog.schema_migrations",
            )
        )

    assert signatures[0] == signatures[1]
    assert len(signatures[0].splitlines()) == 3
    assert "omninode-cloud|service:onex_api|legacy_unclassified" in signatures[0]
    assert ";migrations_log:20260101_cloud" in signatures[0]
    assert (
        pg16.sql(
            cloud_database,
            "SELECT version || '|' || coalesce(checksum, '<NULL>') || '|' || "
            "applied_at::text FROM public.schema_migrations",
        )
        == source_signature
    )


def test_cloud_log_only_alias_is_a_real_runner_red(
    pg16: Pg16Cluster, tmp_path: Path
) -> None:
    migrations_dir, _, _ = _synthetic_migration_tree(tmp_path)
    service_database = "omn15413_log_only_service"
    application_database = "omn15413_log_only_app"
    cloud_database = "omn15413_log_only_cloud"
    for database in (service_database, application_database, cloud_database):
        pg16.create_database(database)
    pg16.command(
        cloud_database,
        "-f",
        "-",
        input_text="""
CREATE TABLE public.schema_migrations (
  version TEXT NOT NULL,
  applied_at TIMESTAMPTZ NOT NULL,
  checksum TEXT
);
CREATE TABLE public.migrations_log (
  migration_name TEXT NOT NULL,
  direction TEXT NOT NULL,
  executed_at TIMESTAMPTZ NOT NULL
);
INSERT INTO public.migrations_log VALUES
  ('20260101_cloud', 'forward', TIMESTAMPTZ '2026-01-03 00:00:00+00');
""",
    )

    run = _run_forward_runner(
        pg16,
        migrations_dir,
        service_database=service_database,
        application_database=application_database,
        cloud_database=cloud_database,
    )

    assert run.returncode != 0
    assert "log-only alias cannot be imported as applied" in run.stderr
    assert pg16.sql(cloud_database, "SELECT count(*) FROM public.migrations_log") == "1"


def test_duplicate_cloud_versions_are_a_real_runner_red(
    pg16: Pg16Cluster, tmp_path: Path
) -> None:
    migrations_dir, _, _ = _synthetic_migration_tree(tmp_path)
    service_database = "omn15413_duplicate_service"
    application_database = "omn15413_duplicate_app"
    cloud_database = "omn15413_duplicate_cloud"
    for database in (service_database, application_database, cloud_database):
        pg16.create_database(database)
    pg16.command(
        cloud_database,
        "-f",
        "-",
        input_text="""
CREATE TABLE public.schema_migrations (
  version TEXT NOT NULL,
  applied_at TIMESTAMPTZ NOT NULL,
  checksum TEXT
);
INSERT INTO public.schema_migrations VALUES
  ('duplicate.sql', TIMESTAMPTZ '2026-01-03 00:00:00+00', NULL),
  ('duplicate.sql', TIMESTAMPTZ '2026-01-04 00:00:00+00', NULL);
""",
    )

    run = _run_forward_runner(
        pg16,
        migrations_dir,
        service_database=service_database,
        application_database=application_database,
        cloud_database=cloud_database,
    )

    assert run.returncode != 0
    assert "duplicate migration version in import" in run.stderr
    assert (
        pg16.sql(cloud_database, "SELECT count(*) FROM public.schema_migrations") == "2"
    )


# ---------------------------------------------------------------------------
# OMN-15695: adopt/convert the pre-OMN-15413 migration_id node ledger.
#
# The live dev-lane application database (omnidash_analytics) carries 80 rows
# written by the historical runner as
#   public.schema_migrations(migration_id, applied_at, checksum, source_set)
# with checksum='applied-by-runner' and source_set='node'.  That relation is
# the predecessor NODE ledger of the application database, not the service
# ledger, so the fail-closed migration_id arm was a false negative for it.
# ---------------------------------------------------------------------------

LIVE_APPLIED_AT = "2026-07-31 06:44:27.696803+00"

LEDGER_SIGNATURE_SQL = """
SELECT string_agg(
  migration_stream || '|' || owner || '|' || domain || '|' || version ||
  '|' || checksum || '|' || checksum_kind || '|' || applied_at::text ||
  '|' || provenance,
  E'\n' ORDER BY migration_stream, domain, version
)
FROM platform_catalog.schema_migrations
"""


def _seed_migration_id_ledger(
    pg16: Pg16Cluster,
    database: str,
    rows: list[tuple[str, str, str]],
    *,
    applied_at: str = LIVE_APPLIED_AT,
) -> None:
    """Create the historical runner ledger shape verbatim and seed ``rows``.

    ``rows`` are ``(migration_id, checksum, source_set)`` triples.
    """
    values = ",\n  ".join(
        f"('{migration_id}', TIMESTAMPTZ '{applied_at}', '{checksum}', '{source_set}')"
        for migration_id, checksum, source_set in rows
    )
    pg16.command(
        database,
        "-f",
        "-",
        input_text=f"""
CREATE TABLE public.schema_migrations (
  migration_id TEXT PRIMARY KEY,
  applied_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  checksum     TEXT NOT NULL,
  source_set   TEXT NOT NULL
);
INSERT INTO public.schema_migrations
  (migration_id, applied_at, checksum, source_set)
VALUES
  {values};
""",
    )


def _live_shaped_rows(count: int = 80) -> list[tuple[str, str, str]]:
    """Reproduce the live dev-lane ledger: N declared node ids, no byte evidence."""
    return [(row[4], "applied-by-runner", "node") for row in _declarations()[:count]]


def _canonical_rows(pg16: Pg16Cluster, database: str) -> list[list[str]]:
    dumped = pg16.sql(
        database,
        "SELECT version || E'\\t' || migration_stream || E'\\t' || owner || "
        "E'\\t' || domain || E'\\t' || checksum || E'\\t' || checksum_kind || "
        "E'\\t' || provenance FROM platform_catalog.schema_migrations "
        "ORDER BY version",
    )
    return [line.split("\t") for line in dumped.splitlines()]


def test_migration_id_node_ledger_is_adopted_twice(pg16: Pg16Cluster) -> None:
    database = "omn15695_adopt"
    pg16.create_database(database)
    declarations = _declarations()[:80]
    assert len(declarations) == 80
    _seed_migration_id_ledger(pg16, database, _live_shaped_rows())
    source_oid = pg16.sql(database, "SELECT 'public.schema_migrations'::regclass::oid")

    signatures: list[str] = []
    for _ in range(2):
        run = _run_bootstrap(pg16, database)
        assert run.returncode == 0, f"{run.stdout}\n{run.stderr}"
        signatures.append(pg16.sql(database, LEDGER_SIGNATURE_SQL))

    assert signatures[0] == signatures[1]
    assert len(signatures[0].splitlines()) == 80
    assert (
        pg16.sql(database, "SELECT count(*) FROM platform_catalog.schema_migrations")
        == "80"
    )
    assert (
        pg16.sql(
            database,
            "SELECT count(*) FROM platform_catalog.schema_migrations "
            "WHERE checksum_kind = 'content_sha256'",
        )
        == "80"
    )
    # applied_at is preserved verbatim: this is the ruling's history clause.
    assert (
        pg16.sql(
            database,
            "SELECT count(*) FROM platform_catalog.schema_migrations "
            f"WHERE applied_at = TIMESTAMPTZ '{LIVE_APPLIED_AT}'",
        )
        == "80"
    )

    expected = {row[4]: (row[1], row[2], row[3], row[5]) for row in declarations}
    observed = _canonical_rows(pg16, database)
    assert len(observed) == 80
    for version, stream, owner, domain, checksum, kind, provenance in observed:
        assert expected[version] == (stream, owner, domain, checksum)
        assert kind == "content_sha256"
        assert provenance == (
            f"adopted:{database}:public.schema_migrations:migration_id:"
            f"{version}:raw-checksum=applied-by-runner"
        )

    # The source relation is never renamed, updated, or deleted.
    assert (
        pg16.sql(database, "SELECT 'public.schema_migrations'::regclass::oid")
        == source_oid
    )
    assert pg16.sql(database, "SELECT count(*) FROM public.schema_migrations") == "80"


def test_pr_review_bot_bypass_log_adopts_cleanly_omn15717(pg16: Pg16Cluster) -> None:
    """Reproduces the exact OMN-15717 live failure and proves the fix.

    Before the OMN-15717 declaration was added, a database carrying the
    legacy runner's row for
    ``node:node_pr_review_bot:001_create_review_bot_bypass_log.sql`` failed
    bootstrap.sql with "unknown migration stream/domain: adopted node
    version ... has no checked-in declaration" (the live
    refresh_stability_lane.sh forensic log, bootstrap.sql:673). This test
    seeds that exact legacy row and asserts bootstrap.sql now adopts it
    without error, is idempotent, and classifies it omninode_internal (R-q:
    bookkeeping/audit-log state, not tenant workload data).
    """
    database = "omn15717_pr_review_bot_adopt"
    pg16.create_database(database)
    version = "node:node_pr_review_bot:001_create_review_bot_bypass_log.sql"
    _seed_migration_id_ledger(pg16, database, [(version, "applied-by-runner", "node")])

    for _ in range(2):
        run = _run_bootstrap(pg16, database)
        assert run.returncode == 0, f"{run.stdout}\n{run.stderr}"

    row = pg16.sql(
        database,
        "SELECT migration_stream || E'\\t' || owner || E'\\t' || domain || E'\\t' || "
        "checksum_kind || E'\\t' || provenance FROM platform_catalog.schema_migrations "
        f"WHERE version = '{version}'",
    )
    stream, owner, domain, checksum_kind, provenance = row.split("\t")
    assert stream == "node:node_pr_review_bot"
    assert owner == "node:node_pr_review_bot"
    assert domain == "omninode_internal"
    assert checksum_kind == "content_sha256"
    assert provenance == (
        f"adopted:{database}:public.schema_migrations:migration_id:"
        f"{version}:raw-checksum=applied-by-runner"
    )
    # Legacy source row is preserved verbatim, never rewritten or deleted.
    assert pg16.sql(database, "SELECT count(*) FROM public.schema_migrations") == "1"


def test_historical_projection_delegation_rows_adopt_cleanly_omn15717(
    pg16: Pg16Cluster,
) -> None:
    """The complete stability predecessor-ledger corpus is idempotently adopted.

    These historical identities have no current vendored SQL artifact, so they
    can only enter the canonical ledger through the checked-in historical-node
    declaration table.  Their ``hotfix-applied-by-codex`` values are source
    records, not file hashes; bootstrap records a deterministic legacy
    attestation and therefore cannot let either row satisfy an active-file
    migration probe.
    """
    database = "omn15717_legacy_projection_delegation"
    pg16.create_database(database)
    versions = (
        "node:node_projection_delegation:0014_create_live_event_projection_view.sql",
        "node:node_projection_delegation:0015_create_generation_dashboard_views.sql",
    )
    _seed_migration_id_ledger(
        pg16,
        database,
        [(version, "hotfix-applied-by-codex", "node") for version in versions],
    )

    signatures: list[str] = []
    for _ in range(2):
        run = _run_bootstrap(pg16, database)
        assert run.returncode == 0, f"{run.stdout}\n{run.stderr}"
        signatures.append(pg16.sql(database, LEDGER_SIGNATURE_SQL))

    assert signatures[0] == signatures[1]
    rows = _canonical_rows(pg16, database)
    assert len(rows) == 2
    for version, stream, owner, domain, checksum, kind, provenance in rows:
        assert version in versions
        assert stream == "node:node_projection_delegation"
        assert owner == stream
        assert domain == "omninode_internal"
        assert len(checksum) == 64
        assert kind == "legacy_attestation"
        assert provenance == (
            f"legacy-adopted:{database}:public.schema_migrations:migration_id:"
            f"{version}:raw-checksum=hotfix-applied-by-codex:ticket=OMN-15717"
        )
    assert pg16.sql(database, "SELECT count(*) FROM public.schema_migrations") == "2"


def test_historical_projection_delegation_source_checksum_mismatch_is_atomic_red(
    pg16: Pg16Cluster,
) -> None:
    database = "omn15717_legacy_projection_mismatch"
    pg16.create_database(database)
    _seed_migration_id_ledger(
        pg16,
        database,
        [
            (
                "node:node_projection_delegation:0014_create_live_event_projection_view.sql",
                "applied-by-runner",
                "node",
            )
        ],
    )

    run = _run_bootstrap(pg16, database)

    assert run.returncode != 0
    assert "conflicting migration checksum" in run.stderr
    assert (
        pg16.sql(
            database, "SELECT to_regclass('platform_catalog.schema_migrations') IS NULL"
        )
        == "t"
    )


def test_adopted_ledger_makes_the_real_runner_skip(
    pg16: Pg16Cluster, tmp_path: Path
) -> None:
    migrations_dir, version, checksum = _synthetic_migration_tree(tmp_path)
    service_database = "omn15695_runner_adopt_service"
    application_database = "omn15695_runner_adopt_app"
    cloud_database = "omn15695_runner_adopt_cloud"
    for database in (service_database, application_database, cloud_database):
        pg16.create_database(database)
    _seed_migration_id_ledger(
        pg16, application_database, [(version, "applied-by-runner", "node")]
    )

    runs = [
        _run_forward_runner(
            pg16,
            migrations_dir,
            service_database=service_database,
            application_database=application_database,
            cloud_database=cloud_database,
        )
        for _ in range(2)
    ]

    for run in runs:
        assert run.returncode == 0, f"{run.stdout}\n{run.stderr}"
        # The FIRST run is the load-bearing no-re-application proof.
        assert "0 node applied, 1 node skipped" in run.stdout
        assert f"skip  {version} (already applied)" in run.stdout
    assert (
        pg16.sql(
            application_database,
            "SELECT checksum FROM platform_catalog.schema_migrations "
            f"WHERE version = '{version}'",
        )
        == checksum
    )


def test_service_owned_migration_id_ledger_still_fails_closed(
    pg16: Pg16Cluster,
) -> None:
    database = "omn15695_service_only"
    pg16.create_database(database)
    _seed_migration_id_ledger(
        pg16,
        database,
        [
            ("docker/000_db_metadata.sql", "applied-by-runner", "docker"),
            ("docker/001_initial.sql", "skip-manifest", "docker"),
        ],
    )

    run = _run_bootstrap(pg16, database)

    assert run.returncode != 0
    assert (
        "unknown migration stream: service-owned migration_id ledger cannot be "
        "selected for the application database" in run.stderr
    )
    assert (
        pg16.sql(
            database, "SELECT to_regclass('platform_catalog.schema_migrations') IS NULL"
        )
        == "t"
    )
    assert pg16.sql(database, "SELECT count(*) FROM public.schema_migrations") == "2"


def test_mixed_service_and_node_rows_partition(pg16: Pg16Cluster) -> None:
    database = "omn15695_mixed"
    pg16.create_database(database)
    first = _declarations()[0]
    _seed_migration_id_ledger(
        pg16,
        database,
        [
            ("docker/000_db_metadata.sql", "applied-by-runner", "docker"),
            (first[4], "applied-by-runner", "node"),
        ],
    )

    run = _run_bootstrap(pg16, database)

    assert run.returncode == 0, f"{run.stdout}\n{run.stderr}"
    assert (
        pg16.sql(database, "SELECT version FROM platform_catalog.schema_migrations")
        == first[4]
    )
    assert pg16.sql(database, "SELECT count(*) FROM public.schema_migrations") == "2"
    assert (
        pg16.sql(
            database,
            "SELECT count(*) FROM platform_catalog.schema_migrations "
            "WHERE version LIKE 'docker/%'",
        )
        == "0"
    )


@pytest.mark.parametrize(
    ("case_name", "migration_id_factory", "source_set"),
    [
        ("unknown_source_set", lambda row: row[4], "cloud"),
        ("bare_filename_identity", lambda row: "0001_bare.sql", "node"),
    ],
)
def test_unrecognized_migration_id_rows_are_atomic_red(
    pg16: Pg16Cluster,
    case_name: str,
    migration_id_factory: Callable[[list[str]], str],
    source_set: str,
) -> None:
    database = f"omn15695_{case_name}"
    pg16.create_database(database)
    first = _declarations()[0]
    _seed_migration_id_ledger(
        pg16,
        database,
        [(migration_id_factory(first), "applied-by-runner", source_set)],
    )

    run = _run_bootstrap(pg16, database)

    assert run.returncode != 0
    assert "unrecognized migration_id rows" in run.stderr
    assert (
        pg16.sql(
            database, "SELECT to_regclass('platform_catalog.schema_migrations') IS NULL"
        )
        == "t"
    )
    assert pg16.sql(database, "SELECT count(*) FROM public.schema_migrations") == "1"


def test_undeclared_node_version_is_atomic_red(pg16: Pg16Cluster) -> None:
    database = "omn15695_undeclared"
    pg16.create_database(database)
    _seed_migration_id_ledger(
        pg16, database, [("node:unknown:0001.sql", "applied-by-runner", "node")]
    )

    run = _run_bootstrap(pg16, database)

    assert run.returncode != 0
    assert "unknown migration stream/domain" in run.stderr
    assert (
        pg16.sql(
            database, "SELECT to_regclass('platform_catalog.schema_migrations') IS NULL"
        )
        == "t"
    )
    assert pg16.sql(database, "SELECT count(*) FROM public.schema_migrations") == "1"


@pytest.mark.parametrize(
    ("case_name", "checksum"),
    [("hex_conflict", "0" * 64), ("non_sentinel", "migrated-by-hand")],
)
def test_conflicting_adoption_checksums_are_atomic_reds(
    pg16: Pg16Cluster, case_name: str, checksum: str
) -> None:
    database = f"omn15695_{case_name}"
    pg16.create_database(database)
    first = _declarations()[0]
    _seed_migration_id_ledger(pg16, database, [(first[4], checksum, "node")])

    run = _run_bootstrap(pg16, database)

    assert run.returncode != 0
    assert "conflicting migration checksum" in run.stderr
    assert (
        pg16.sql(
            database, "SELECT to_regclass('platform_catalog.schema_migrations') IS NULL"
        )
        == "t"
    )
    assert pg16.sql(database, "SELECT count(*) FROM public.schema_migrations") == "1"


def test_adoption_is_idempotent_beside_a_populated_canonical_ledger(
    pg16: Pg16Cluster,
) -> None:
    """Regression for the double-declaration guard and the filename-import guard.

    The source relation is deliberately preserved, so a second bootstrap sees a
    checksum-capable ``public.schema_migrations`` beside a populated canonical
    ledger.  Neither the ledger-selection guard nor the filename import may
    treat that as a double declaration.
    """
    database = "omn15695_idempotent"
    pg16.create_database(database)
    _seed_migration_id_ledger(pg16, database, _live_shaped_rows(5))

    first_run = _run_bootstrap(pg16, database)
    assert first_run.returncode == 0, f"{first_run.stdout}\n{first_run.stderr}"

    second_run = _run_bootstrap(pg16, database)

    assert second_run.returncode == 0, f"{second_run.stdout}\n{second_run.stderr}"
    assert "double migration declaration" not in second_run.stderr
    assert "remains beside the canonical ledger" not in second_run.stderr
    assert (
        pg16.sql(database, "SELECT count(*) FROM platform_catalog.schema_migrations")
        == "5"
    )


def test_tampered_adopted_row_is_a_double_declaration_red(
    pg16: Pg16Cluster,
) -> None:
    database = "omn15695_tampered"
    pg16.create_database(database)
    first = _declarations()[0]
    _seed_migration_id_ledger(pg16, database, [(first[4], "applied-by-runner", "node")])
    first_run = _run_bootstrap(pg16, database)
    assert first_run.returncode == 0, f"{first_run.stdout}\n{first_run.stderr}"
    pg16.command(
        database,
        "-c",
        "UPDATE platform_catalog.schema_migrations "
        "SET provenance = provenance || ':tampered'",
    )

    run = _run_bootstrap(pg16, database)

    assert run.returncode != 0
    assert "double migration declaration for version" in run.stderr


def test_unknown_public_ledger_shape_still_fails_closed(pg16: Pg16Cluster) -> None:
    """A shape that is neither filename, version, migration_id, nor node stays RED."""
    database = "omn15695_unknown_shape"
    pg16.create_database(database)
    pg16.command(
        database,
        "-f",
        "-",
        input_text="""
CREATE TABLE public.schema_migrations (
  migration_id TEXT PRIMARY KEY,
  applied_at   TIMESTAMPTZ NOT NULL,
  checksum     TEXT NOT NULL,
  source_set   TEXT NOT NULL,
  extra_column TEXT NOT NULL
);
""",
    )

    run = _run_bootstrap(pg16, database)

    assert run.returncode != 0
    assert "unknown migration ledger shape" in run.stderr
    assert (
        pg16.sql(
            database, "SELECT to_regclass('platform_catalog.schema_migrations') IS NULL"
        )
        == "t"
    )
