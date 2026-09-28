# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A pre-PR verify slot's forward migration grants to the SLOT's principals.

Measured on the fourth slot boot (2026-09-24, omnibase_infra#3944 at
116e920b7). With every database name confined to the slot, forward 099 still
ran ``GRANT CONNECT ON DATABASE omnidash_analytics_prepr1 TO omninode_runtime``:
the dev lane's principal got the slot's database and the slot's own
``omninode_runtime_prepr1`` got nothing, so the runtime died on ``permission
denied for database omnidash_analytics_prepr1``. The same run issued
``ALTER ROLE`` on the dev lane's ``role_omnidash`` and
``tenant_projection_writer`` from 096 and 103.

These tests run the runner's own helper under ``sh`` rather than matching its
text (OMN-19404).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "scripts" / "run-forward-migrations.sh"
PROVISIONER = REPO_ROOT / "scripts" / "provision_db_slot.sh"
FORWARD = REPO_ROOT / "docker" / "migrations" / "forward"
SLOT = "prepr1"
CONNECT_LINE = re.compile(r"^\s*\\(connect|c)(\s|$)")
TOKEN = re.compile(r"[A-Za-z0-9_]+")

pytestmark = pytest.mark.unit


def _helpers() -> str:
    text = RUNNER.read_text(encoding="utf-8")
    start = text.index("# Asserts a name is inside the slot's fence")
    end = text.index("# ---- END slot \\connect rewrite (OMN-18893) ----")
    return text[start:end]


def _managed_roles() -> list[str]:
    match = re.search(r'^SLOT_MANAGED_ROLES="([^"]*)"', _helpers(), re.M)
    assert match, "the runner declares no SLOT_MANAGED_ROLES"
    return match.group(1).split()


def _rewrite(
    path: Path, *, slot_active: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["sh", "-c", _helpers() + '\nslot_migration_file "$1"', "sh", str(path)],
        capture_output=True,
        text=True,
        check=False,
        env={
            "PATH": "/usr/bin:/bin",
            "SLOT_ACTIVE": "1" if slot_active else "0",
            "ONEX_DB_SLOT": SLOT,
        },
    )


def _rewritten_lines(path: Path) -> list[str]:
    proc = _rewrite(path)
    assert proc.returncode == 0, proc.stderr
    copy = Path(proc.stdout.strip())
    if copy.resolve() == path.resolve():
        return path.read_text(encoding="utf-8").splitlines()
    try:
        return copy.read_text(encoding="utf-8").splitlines()
    finally:
        if FORWARD not in copy.parents and copy.parent != path.parent:
            copy.unlink(missing_ok=True)


def _names_a_managed_role(line: str) -> bool:
    roles = set(_managed_roles())
    return not line.lstrip().startswith("--") and any(
        tok in roles for tok in TOKEN.findall(line)
    )


def _role_files() -> list[Path]:
    return sorted(
        p
        for p in FORWARD.rglob("*.sql")
        if any(
            _names_a_managed_role(ln)
            for ln in p.read_text(encoding="utf-8").splitlines()
        )
    )


def test_managed_roles_are_exactly_the_principals_the_provisioner_mints() -> None:
    """A role the provisioner mints and the runner does not rewrite gets no grant."""
    proc = subprocess.run(
        ["bash", str(PROVISIONER), "--print-scope"],
        capture_output=True,
        text=True,
        check=False,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
            "ONEX_DB_SLOT": SLOT,
        },
    )
    assert proc.returncode == 0, proc.stderr
    minted = sorted(
        line.removeprefix("role=").removesuffix(f"_{SLOT}")
        for line in proc.stdout.splitlines()
        if line.startswith("role=")
    )
    assert minted, proc.stdout
    assert sorted(_managed_roles()) == minted


def test_the_corpus_grants_to_managed_roles() -> None:
    """Positive control: with no role-naming file left, the pins below prove nothing."""
    names = {p.name for p in _role_files()}
    assert "099_create_omninode_internal_live_events.sql" in names
    assert "103_create_tenant_projection_writer_role.sql" in names


@pytest.mark.parametrize("migration", _role_files(), ids=lambda p: p.name)
def test_every_managed_role_in_the_corpus_is_confined_to_the_slot(
    migration: Path,
) -> None:
    roles = set(_managed_roles())
    original = migration.read_text(encoding="utf-8").splitlines()
    rewritten = _rewritten_lines(migration)
    assert len(original) == len(rewritten)
    for before, after in zip(original, rewritten, strict=True):
        if before.lstrip().startswith("--") or CONNECT_LINE.match(before):
            continue
        # Only suffixes were added: nothing else on the line moved.
        assert after.replace(f"_{SLOT}", "") == before, after
        left = [tok for tok in TOKEN.findall(after) if tok in roles]
        assert left == [], f"dev principal left in a slot statement: {after}"


def test_099_grants_the_slot_database_to_the_slot_runtime_principal() -> None:
    lines = _rewritten_lines(FORWARD / "099_create_omninode_internal_live_events.sql")
    assert (
        "    EXECUTE 'GRANT CONNECT ON DATABASE omnidash_analytics_prepr1 "
        "TO omninode_runtime_prepr1';"
    ) in lines
    assert (
        "  IF EXISTS (SELECT 1 FROM pg_database WHERE datname = "
        "'omnidash_analytics_prepr1') THEN"
    ) in lines


def test_103_asserts_connect_for_the_slot_principal_on_the_slot_database() -> None:
    lines = _rewritten_lines(FORWARD / "103_create_tenant_projection_writer_role.sql")
    assert (
        "  SELECT has_database_privilege('tenant_projection_writer_prepr1', "
        "'omnidash_analytics_prepr1', 'CONNECT')"
    ) in lines


def test_only_whole_tokens_and_only_database_assertions_are_rewritten(
    tmp_path: Path,
) -> None:
    migration = tmp_path / "0001_probe.sql"
    migration.write_text(
        """\
-- GRANT SELECT ON t TO omninode_runtime;
GRANT SELECT ON t TO omninode_runtime;
CREATE POLICY p ON t TO "tenant_projection_writer" USING (true);
SELECT omninode_runtime_password, role_omnidash2, xrole_omnidash;
SELECT OMNINODE_RUNTIME;
UPDATE db_metadata SET owner_service = 'omnibase_infra';
SELECT 1 FROM pg_database WHERE datname='omnidash_analytics';
GRANT SELECT ON t TO app_dashboard;
""",
        encoding="utf-8",
    )
    assert _rewritten_lines(migration) == [
        "-- GRANT SELECT ON t TO omninode_runtime;",
        "GRANT SELECT ON t TO omninode_runtime_prepr1;",
        'CREATE POLICY p ON t TO "tenant_projection_writer_prepr1" USING (true);',
        "SELECT omninode_runtime_password, role_omnidash2, xrole_omnidash;",
        "SELECT OMNINODE_RUNTIME;",
        "UPDATE db_metadata SET owner_service = 'omnibase_infra';",
        "SELECT 1 FROM pg_database WHERE datname='omnidash_analytics_prepr1';",
        "GRANT SELECT ON t TO app_dashboard;",
    ]


def test_a_file_naming_only_a_role_is_rewritten_and_outside_a_slot_is_not(
    tmp_path: Path,
) -> None:
    migration = tmp_path / "0002_probe.sql"
    migration.write_text(
        "GRANT USAGE ON SCHEMA public TO role_omnidash;\n", encoding="utf-8"
    )
    assert _rewritten_lines(migration) == [
        "GRANT USAGE ON SCHEMA public TO role_omnidash_prepr1;"
    ]
    proc = _rewrite(migration, slot_active=False)
    assert proc.returncode == 0, proc.stderr
    assert Path(proc.stdout.strip()) == migration
