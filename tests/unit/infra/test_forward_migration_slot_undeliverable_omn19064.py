# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A pre-PR verify slot skips the flat migrations the corpus declares undeliverable.

Measured on the live slot bring-up of 2026-10-18 (OMN-19064): after the directive
database was provisioned, the forward migration stopped on
``083_create_log_entries.sql`` with ``permission denied for database
"omnidash_analytics_prepr1"``. The file ``\\connect``s to a second database and
the slot's migration principal is NOSUPERUSER with no CONNECT there. The same
corpus declares 083, 096, 097, 098 and 099 ``undeliverable`` in
``cross-database-flat-migrations.yaml``, so no managed lane applies them either.

The tests run the runner's own helper under ``sh`` and carry a positive control
so a corpus with no declared file cannot make the pins pass by having nothing in it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "scripts" / "run-forward-migrations.sh"
FORWARD = REPO_ROOT / "docker" / "migrations" / "forward"
LEDGER = FORWARD / "cross-database-flat-migrations.yaml"

pytestmark = pytest.mark.unit


def _helper() -> str:
    text = RUNNER.read_text(encoding="utf-8")
    start = text.index(
        "# ---- BEGIN slot undeliverable flat migrations (OMN-19064) ----"
    )
    end = text.index("# ---- END slot undeliverable flat migrations (OMN-19064) ----")
    return text[start:end]


def _undeliverable(filename: str, migrations_dir: Path) -> int:
    return subprocess.run(
        [
            "sh",
            "-c",
            _helper() + '\nslot_flat_migration_undeliverable "$1"',
            "sh",
            filename,
        ],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin", "MIGRATIONS_DIR": str(migrations_dir)},
    ).returncode


def _declared(disposition: str) -> list[str]:
    entries = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))["entries"]
    return sorted(e["file"] for e in entries if e["disposition"] == disposition)


def test_the_corpus_still_declares_undeliverable_files() -> None:
    """Positive control: an empty ledger would make the pins below vacuous."""
    assert "083_create_log_entries.sql" in _declared("undeliverable")


@pytest.mark.parametrize("filename", _declared("undeliverable"))
def test_every_declared_undeliverable_file_is_skipped(filename: str) -> None:
    assert _undeliverable(filename, FORWARD) == 0


def test_an_ordinary_flat_file_is_still_applied() -> None:
    assert _undeliverable("042_create_keycloak_db.sql", FORWARD) == 1


def test_a_corpus_without_the_ledger_skips_nothing(tmp_path: Path) -> None:
    assert _undeliverable("083_create_log_entries.sql", tmp_path) == 1


def test_the_flat_loop_consults_the_helper_only_under_a_slot() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert (
        'if [ "$SLOT_ACTIVE" -eq 1 ] && slot_flat_migration_undeliverable "$filename"; then'
        in text
    )
