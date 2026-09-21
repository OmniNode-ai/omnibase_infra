# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18953 — the cloud migration runner must not need the maintenance database.

WHAT BROKE
----------
OMN-18892 (`833d692e`) added a seam that revokes PUBLIC's ``CONNECT`` on every
database this repository creates, and it reaches the maintenance database
``postgres``. That revocation is CORRECT and stays. What was wrong was a dependency
on the privilege it removed.

``docker/migrations/cloud/run-cloud-migrations.sh`` enumerated the server's databases
to evaluate the manifest's ``requires_database`` condition through a helper that
connected ``-d postgres``. Running as ``role_omninode``, which holds no ``CONNECT``
there, it died on its first server-level call::

    psql: error: ... FATAL:  permission denied for database "postgres"
    DETAIL:  User does not have CONNECT privilege.

Under ``set -e`` that killed the runner (exit 2, 330 ms), and the deploy agent's
fail-closed migration preflight then refused the rebuild. Six consecutive jobs were
refused from 19:14Z and the .201 dev lane recreated no runtime container for five
hours, which blocked every lab proof behind it.

WHY THE FIX IS A DELETION AND NOT A GRANT
------------------------------------------
``pg_database`` is a SHARED catalog: the same rows are visible from every database in
the cluster, so no maintenance-database connection was ever needed. The runner already
holds a proven connection to its own database. Granting ``CONNECT`` back would also not
survive: the merged seam re-asserts the revocation on every bring-up, so the live ACL
and the committed code agree and a hand ``GRANT`` would be undone at the next rebuild.

WHAT THIS MODULE PINS
---------------------
That the runner names no maintenance database, that it still enumerates the catalog
(the feature was moved, not dropped), and that no other file under ``docker/migrations``
reintroduces the same assumption. Every scan carries a positive control, because a scan
pointed at the wrong place and a scan that legitimately finds nothing are the same green
test (CLAUDE.md operating rule 16).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MIGRATIONS_DIR = REPO_ROOT / "docker" / "migrations"
CLOUD_RUNNER = MIGRATIONS_DIR / "cloud" / "run-cloud-migrations.sh"

# A literal maintenance-database target on a psql invocation. Variable targets such as
# -d "$DB_NAME" are deliberately NOT matched: what is being refused is hardcoding the
# one database a non-superuser migration role is least likely to reach.
_LITERAL_MAINTENANCE_DB = re.compile(r"""-d\s+["']?postgres["']?(?:\s|$)""")
_PSQL_INVOCATION = re.compile(r"\bpsql\b")
_COMMENT_LINE = re.compile(r"^\s*#")

pytestmark = pytest.mark.unit


def _code_lines(path: Path) -> list[str]:
    return [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if not _COMMENT_LINE.match(line)
    ]


def _shell_scripts_under_migrations() -> list[Path]:
    return sorted(MIGRATIONS_DIR.rglob("*.sh"))


class TestTheScanIsPointedSomewhereReal:
    """Positive controls. The zeros below are only evidence if these pass."""

    def test_the_cloud_runner_exists_and_invokes_psql(self) -> None:
        assert CLOUD_RUNNER.is_file(), f"{CLOUD_RUNNER} is missing"
        hits = [
            line for line in _code_lines(CLOUD_RUNNER) if _PSQL_INVOCATION.search(line)
        ]
        assert hits, (
            "no psql invocation found in the cloud runner; the maintenance-database "
            "assertions below would pass vacuously"
        )

    def test_the_migration_shell_scan_finds_scripts_that_use_psql(self) -> None:
        scripts = _shell_scripts_under_migrations()
        assert scripts, f"no shell scripts found under {MIGRATIONS_DIR}"
        with_psql = [
            path
            for path in scripts
            if any(_PSQL_INVOCATION.search(line) for line in _code_lines(path))
        ]
        assert with_psql, (
            "the scan found shell scripts but none invoking psql, so it is not reading "
            "what it thinks it is"
        )

    def test_the_maintenance_db_pattern_actually_matches(self) -> None:
        # Without this, the prohibition below would pass against a broken regex.
        assert _LITERAL_MAINTENANCE_DB.search(
            'psql -U "$DB_USER" -d postgres -v ON_ERROR_STOP=1'
        )
        assert _LITERAL_MAINTENANCE_DB.search("psql -d 'postgres'")
        # A variable target is not the thing being refused.
        assert not _LITERAL_MAINTENANCE_DB.search('psql -d "$DB_NAME" -tAc "SELECT 1"')
        assert not _LITERAL_MAINTENANCE_DB.search('psql -d "$PGADMINDB"')


class TestTheCloudRunnerNeedsNoMaintenanceDatabase:
    """AC-1 and AC-3: the dependency is removed, the revocation is not reversed."""

    def test_the_runner_names_no_literal_maintenance_database(self) -> None:
        offenders = [
            line
            for line in _code_lines(CLOUD_RUNNER)
            if _LITERAL_MAINTENANCE_DB.search(line)
        ]
        assert not offenders, (
            f"{CLOUD_RUNNER.relative_to(REPO_ROOT)} still connects to the maintenance "
            f"database: {offenders}. It runs as the migration role, which holds no "
            "CONNECT on `postgres` since OMN-18892, so this refuses every dev-lane "
            "rebuild. pg_database is a shared catalog and is readable from the "
            "runner's own database; the fix is to stop connecting there, never to "
            "grant CONNECT back, because the seam re-revokes it on every bring-up"
        )

    def test_the_runner_defines_no_server_scoped_psql_helper(self) -> None:
        text = CLOUD_RUNNER.read_text(encoding="utf-8")
        assert "psql_server" not in text, (
            "psql_server() is back. It existed only to reach the maintenance database; "
            "a helper kept for a caller that no longer needs it is how the assumption "
            "returns"
        )

    def test_the_catalog_enumeration_still_happens_through_the_database_helper(
        self,
    ) -> None:
        # The feature was MOVED, not dropped. Without this, deleting the enumeration
        # outright would satisfy the two assertions above and silently change which
        # migrations the manifest's requires_database condition selects.
        lines = _code_lines(CLOUD_RUNNER)
        enumerating = [line for line in lines if "FROM pg_database" in line]
        assert len(enumerating) == 1, (
            f"expected exactly one pg_database enumeration, found {len(enumerating)}: "
            f"{enumerating}"
        )
        assert "psql_db" in enumerating[0], (
            "the enumeration no longer runs through psql_db, the helper bound to the "
            f"runner's own database: {enumerating[0]!r}"
        )


class TestNoOtherMigrationScriptReintroducesIt:
    """AC-2: the audit, derived by scanning rather than from a list kept here."""

    def test_no_shell_script_under_migrations_names_the_maintenance_database(
        self,
    ) -> None:
        offenders = {
            str(path.relative_to(REPO_ROOT)): [
                line
                for line in _code_lines(path)
                if _LITERAL_MAINTENANCE_DB.search(line)
            ]
            for path in _shell_scripts_under_migrations()
        }
        offenders = {path: lines for path, lines in offenders.items() if lines}
        assert not offenders, (
            f"migration shell scripts name the maintenance database: {offenders}. "
            "Every one of these runs as a migration role rather than the superuser"
        )
