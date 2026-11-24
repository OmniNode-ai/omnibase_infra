# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Only the live read of the ledger file in the disk-gc units is counted (OMN-26330).

The ledger file census counts a ``.service`` comment that spells the ledger
file's name as a reader, because the census does not strip comments from that
suffix. Two prose comments in ``deploy/disk-gc`` did, which held two census
hits that read nothing. They now say "the rolling work ledger".

The one executable read left in those units is the ``--ledger`` argument of
``onex-worktree-prune.service``, handed to omniclaude's
``worktree_auto_prune.py``. It cannot move to the bus from this repository: the
script opens the file itself, so the unit stays on the census row until that
script asks the reader node.

This module pins both facts: no disk-gc unit spells the ledger file's name
outside that one ExecStart line, and the ExecStart line still does (the
positive control that shows the scan can see a hit).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

UNIT_DIR = Path(__file__).resolve().parents[2] / "deploy" / "disk-gc"
# Spelled in two parts so this module is not itself a census hit.
LEDGER_FILE_NAME = "ROLLING_WORK" + "_LEDGER"


def _lines_naming_the_ledger(unit: Path) -> list[str]:
    return [
        line
        for line in unit.read_text(encoding="utf-8").splitlines()
        if LEDGER_FILE_NAME in line
    ]


def test_the_disk_gc_unit_comments_do_not_spell_the_ledger_file_name() -> None:
    offenders = {
        unit.name: hits
        for unit in sorted(UNIT_DIR.glob("*.service"))
        if (
            hits := [
                line
                for line in _lines_naming_the_ledger(unit)
                if line.lstrip().startswith("#")
            ]
        )
    }
    assert offenders == {}, (
        "a comment in a disk-gc unit spells the ledger file's name, which the "
        f"census counts as a reader: {offenders}"
    )


def test_the_prune_unit_exec_start_is_the_one_live_ledger_path() -> None:
    """Positive control: the scan finds the ExecStart line, so its zeros are real."""
    hits = _lines_naming_the_ledger(UNIT_DIR / "onex-worktree-prune.service")
    assert len(hits) == 1
    assert hits[0].startswith("ExecStart=")
    assert "--ledger" in hits[0]
