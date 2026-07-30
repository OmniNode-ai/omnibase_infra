# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""PostgreSQL 16 proof for OMN-15413 ledger selection and import semantics."""

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
        pytest.fail(f"PostgreSQL 16 startup failed: {start.stderr}")

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
    pg16: Pg16Cluster, database: str
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
        (
            "\\copy onex_application_migration_manifest "
            f"FROM '{MANIFEST}' WITH (FORMAT text, DELIMITER E'\\t')"
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
    (ledger_dir / "cloud-migration-aliases.tsv").write_text(
        "20260101_cloud\t20260101_cloud.sql\n", encoding="utf-8"
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
