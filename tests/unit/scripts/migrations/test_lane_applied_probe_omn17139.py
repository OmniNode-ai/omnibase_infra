# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17139 — the applied-ness question must be asked of the table that gates.

``check_migration_append_only.py`` refuses an in-place edit of a declared node
migration unless the author writes a supersession row. Writing that row commits
the author to one factual claim: this migration is not already applied anywhere.

On 2026-08-30 that claim was made, in ``migration-supersessions.tsv``, on the
strength of this probe::

    SELECT to_regclass('public.onex_application_migration_manifest');   -- NULL

and it was false. ``onex_application_migration_manifest`` is a TEMP table
``run-forward-migrations.sh`` builds from the checked-in TSV for the life of one
bootstrap session; it is absent on every lane at every moment an author could
look, so a probe against it returns "clean" unconditionally. Meanwhile the
canonical ledger the runner actually gates on -- ``platform_catalog.schema_migrations``
-- held a row recorded by that runner two hours earlier, and the rewrite bricked
the dev lane.

``test_manifest_relation_probe_reads_clean_on_an_applied_lane`` is the RED
control: it drives the ORIGINAL query against a lane fixture that unambiguously
HAS the migration applied, and asserts it still answers "clean". Without it the
GREEN proof would merely show that some query returns a row, not that the query
it replaced could not.

No database is required: ``--psql-exec`` is the injection seam the shipped tool
already exposes, so these proofs drive the real ``main()`` against a scripted
psql that answers exactly as the .201 dev lane answered.
"""

from __future__ import annotations

import ast
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[4]
PROBE = REPO_ROOT / "scripts" / "migrations" / "check_migration_applied_on_lane.py"

# The live row, verbatim from the .201 dev lane on 2026-08-30 (OMN-17139).
VERSION = "node:node_projection_work_events:0001_create_work_events.sql"
RECORDED = "cba8013e54d5b8b663a50858cb88911b39a503f9896a9d5a138fe51eec8b6664"
PROVENANCE = "file:nodes/node_projection_work_events/0001_create_work_events.sql"
APPLIED_AT = "2026-08-30 04:59:57.430276+00"

# A lane exactly as .201 answers: the canonical ledger exists and carries the
# row; the manifest relation does not exist, because it never does.
FAKE_PSQL = f"""#!/bin/sh
sql=$(cat)
case "$sql" in
  *onex_application_migration_manifest*)
    # to_regclass() of a relation that is only ever a per-session TEMP table.
    printf '\\n'
    ;;
  *"to_regclass('platform_catalog.schema_migrations')"*)
    printf 'platform_catalog.schema_migrations\\n'
    ;;
  *"FROM platform_catalog.schema_migrations"*)
    printf '{RECORDED}\\037content_sha256\\037{PROVENANCE}\\037{APPLIED_AT}\\n'
    ;;
  *)
    printf '\\n'
    ;;
esac
"""


@pytest.fixture
def fake_psql(tmp_path: Path) -> str:
    path = tmp_path / "fake-psql"
    path.write_text(FAKE_PSQL, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(path)


def _run_probe(fake_psql: str, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(PROBE),
            "--version",
            VERSION,
            "--database",
            "omnidash_analytics",
            "--psql-exec",
            f'["{fake_psql}"]',
            *extra,
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ},
        timeout=120,
    )


def test_manifest_relation_probe_reads_clean_on_an_applied_lane(
    fake_psql: str,
) -> None:
    """RED control: the query that was actually run cannot see the applied row.

    Same lane, same moment, same fixture the GREEN proof below uses -- and this
    one answers "no prior checksum recorded". That is the whole defect: not a
    weak answer, a fabricated one.
    """
    result = subprocess.run(
        [fake_psql],
        input="SELECT to_regclass('public.onex_application_migration_manifest');",
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "", (
        "the manifest-relation probe must come back empty on a lane that HAS the "
        "migration applied -- if this ever returns a row, the control is no "
        f"longer reproducing the 2026-08-30 conditions: {result.stdout!r}"
    )


def test_probe_reports_applied_from_the_canonical_ledger(fake_psql: str) -> None:
    """GREEN: the shipped probe asks the table the runner gates on, and finds it."""
    result = _run_probe(fake_psql)

    assert result.returncode == 1, (
        "an applied migration must exit non-zero: a probe that exits 0 here reads "
        f"as permission to rewrite the file.\n{result.stdout}\n{result.stderr}"
    )
    assert "APPLIED" in result.stdout, result.stdout
    assert RECORDED in result.stdout, result.stdout
    assert PROVENANCE in result.stdout, result.stdout
    assert APPLIED_AT in result.stdout, result.stdout


def test_probe_names_only_the_canonical_ledger(fake_psql: str) -> None:
    """The tool must not reintroduce the relation that answers clean everywhere."""
    source = PROBE.read_text(encoding="utf-8")
    # The module docstring quotes the wrong query verbatim, on purpose -- that is
    # the record of what went wrong. Only executable source is scanned, so the
    # explanation cannot fail the check it exists to explain.
    module = ast.parse(source)
    lines = source.splitlines()
    first = module.body[0] if module.body else None
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
        and first.end_lineno is not None
    ):
        lines = lines[first.end_lineno :]
    executable_lines = [
        line
        for line in lines
        if "onex_application_migration_manifest" in line
        and not line.lstrip().startswith("#")
    ]
    assert not executable_lines, (
        "the probe must never query onex_application_migration_manifest; it is a "
        f"per-session TEMP table and reads clean on every lane: {executable_lines}"
    )
    assert 'CANONICAL_LEDGER = "platform_catalog.schema_migrations"' in source


def test_absent_ledger_is_indeterminate_not_clean(tmp_path: Path) -> None:
    """A database with no canonical ledger must not read as a negative answer."""
    path = tmp_path / "no-ledger-psql"
    path.write_text("#!/bin/sh\ncat >/dev/null\nprintf '\\n'\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    result = _run_probe(str(path))
    assert result.returncode == 2, (
        "NO_LEDGER must exit 2, distinct from CLEAN's 0: an unanswered question "
        f"is not a negative answer.\n{result.stdout}\n{result.stderr}"
    )
    assert "NO_LEDGER" in result.stdout, result.stdout
