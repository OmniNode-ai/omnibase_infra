# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A pre-PR verify slot's forward migration must not follow a ``\\connect`` out.

Measured on the first slot boot (2026-09-24, omnibase_infra#3944). The runner
announced ``targeting omnibase_infra_prepr1 / omnidash_analytics_prepr1`` and
then applied the post-``\\connect`` bodies of forward 083, 096, 097, 098 and 099
to the DEV lane's own ``omnidash_analytics`` as the superuser: a ``\\connect``
inside a file overrides the ``-d`` the runner passed, and suffixing PGDB and
NODE_PGDB never reached it.

These tests run the runner's own helper under ``sh`` rather than matching its
text, and the corpus test carries a positive control so a corpus with no
``\\connect`` left in it cannot make the pin pass by having nothing to check.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "scripts" / "run-forward-migrations.sh"
FORWARD = REPO_ROOT / "docker" / "migrations" / "forward"
CONNECT_LINE = re.compile(r"^\s*\\(connect|c)(\s|$)")
ON_DATABASE = re.compile(r"ON DATABASE ([a-z_][a-z0-9_]*)")

pytestmark = pytest.mark.unit


def _helpers() -> str:
    """The fence and rewrite functions, cut from the runner between their markers."""
    text = RUNNER.read_text(encoding="utf-8")
    start = text.index("# Asserts a name is inside the slot's fence")
    end = text.index("# ---- END slot \\connect rewrite (OMN-18893) ----")
    return text[start:end]


def _rewrite(path: Path, *, slot_active: bool) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["sh", "-c", _helpers() + '\nslot_migration_file "$1"', "sh", str(path)],
        capture_output=True,
        text=True,
        check=False,
        env={
            "PATH": "/usr/bin:/bin",
            "SLOT_ACTIVE": "1" if slot_active else "0",
            "ONEX_DB_SLOT": "prepr1",
        },
    )


def _names_a_database(line: str) -> bool:
    if CONNECT_LINE.match(line):
        return True
    return not line.lstrip().startswith("--") and bool(ON_DATABASE.search(line))


def _connect_files() -> list[Path]:
    return sorted(
        p
        for p in FORWARD.rglob("*.sql")
        if any(
            _names_a_database(ln) for ln in p.read_text(encoding="utf-8").splitlines()
        )
    )


def test_the_corpus_still_switches_database_inside_a_file() -> None:
    """Positive control: without a live ``\\connect`` the pins below prove nothing."""
    names = {p.name for p in _connect_files()}
    assert "083_create_log_entries.sql" in names, sorted(names)
    # 103 names the dev database only in a GRANT, with no connect line; the
    # second boot failed on exactly that form in 097.
    assert "103_create_tenant_projection_writer_role.sql" in names, sorted(names)


@pytest.mark.parametrize("migration", _connect_files(), ids=lambda p: p.name)
def test_every_connect_in_the_corpus_is_confined_to_the_slot(
    migration: Path, tmp_path: Path
) -> None:
    proc = _rewrite(migration, slot_active=True)
    assert proc.returncode == 0, proc.stderr
    copy = Path(proc.stdout.strip())
    try:
        assert copy != migration, "under a slot the file was applied unrewritten"
        original = migration.read_text(encoding="utf-8").splitlines()
        rewritten = copy.read_text(encoding="utf-8").splitlines()
        assert len(original) == len(rewritten)
        for before, after in zip(original, rewritten, strict=True):
            if CONNECT_LINE.match(before):
                assert after.split()[1] == before.split()[1] + "_prepr1", after
            elif _names_a_database(before):
                expected = ON_DATABASE.sub(r"ON DATABASE \1_prepr1", before)
                assert after == expected, after
            else:
                assert after == before, "a line naming no database changed"
    finally:
        copy.unlink(missing_ok=True)


def test_outside_a_slot_the_file_is_applied_as_it_stands() -> None:
    migration = FORWARD / "083_create_log_entries.sql"
    proc = _rewrite(migration, slot_active=False)
    assert proc.returncode == 0, proc.stderr
    assert Path(proc.stdout.strip()) == migration


@pytest.mark.parametrize(
    "line",
    [
        "\\connect omnidash_analytics other_user",
        '\\connect :"DB"',
        "\\c",
        "grant connect on database omnidash_analytics to app_dashboard;",
        'GRANT CONNECT ON DATABASE "omnidash_analytics" TO app_dashboard;',
    ],
)
def test_a_connect_that_cannot_be_rewritten_is_refused(
    line: str, tmp_path: Path
) -> None:
    """A form this cannot confine is a form that reaches outside the fence."""
    migration = tmp_path / "0001_probe.sql"
    migration.write_text(f"SELECT 1;\n{line}\nSELECT 2;\n", encoding="utf-8")
    proc = _rewrite(migration, slot_active=True)
    assert proc.returncode == 4, (proc.returncode, proc.stdout, proc.stderr)
    assert "slot_fence_refusal" in proc.stderr
    assert proc.stdout.strip() == ""


def test_both_apply_sites_go_through_the_rewrite() -> None:
    """The flat and the node corpus each apply files; neither may bypass it."""
    code = [
        ln
        for ln in RUNNER.read_text(encoding="utf-8").splitlines()
        if not ln.lstrip().startswith("#")
    ]
    applies = [ln for ln in code if 'ON_ERROR_STOP=1 -f "$' in ln]
    assert len(applies) == 2, applies
    assert all('-f "$apply_file"' in ln for ln in applies), applies
    rewrite = 'apply_file="$(slot_migration_file "$migration_file")"'
    assert sum(rewrite in ln for ln in code) == 2
