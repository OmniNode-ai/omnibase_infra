# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17403: the roll writes paths INTO committed markdown, so they must be portable.

The defect, measured on ``OmniNode-ai/omni_home`` ``origin/main`` at
``7663ea047c``. The 2026-09-16 roll ran from a worktree, and
``_pointer_block`` wrote ``str(archive_path)`` -- the absolute path of the
machine it ran on -- into two lines of the ledger it was rolling::

    <!-- ledger-roll: {"archive": "/Users/.../omni_worktrees/OMN-17403/omni_home/docs/tracking/archive/..."} -->
    > Older rows live in `/Users/.../omni_worktrees/OMN-17403/omni_home/docs/tracking/archive/...`

Both lines are committed content. The path names a worktree that is deleted
when its ticket closes, on one machine, so the pointer a reader is meant to
follow is dead on arrival for every other reader -- and it is a plain
``omni_home`` CLAUDE.md rule 6 violation (no ``/Users/`` or ``/Volumes/`` in
committed source).

The same defect is in the ARCHIVE file's own header, which records
``"source": str(ledger)``.

The contract asserted here: every path the roll writes into a markdown file it
also commits is relative -- to the repository root when the file is inside one,
and to the ledger's own directory otherwise. The roll RECEIPT is deliberately
exempt: it is operational stdout read by the trigger on the machine that ran
it, never committed, and an absolute path is the useful form there.

Red-first: with ``_portable_path`` absent (or reverted to ``str(...)``) every
test in ``TestPointerBlockIsPortable`` and ``TestArchiveHeaderIsPortable``
fails, and ``test_receipt_still_carries_the_absolute_path`` passes -- that last
one is the positive control proving the tests are reading a real roll and not
an empty one.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "ledger_lock.py"

SECTION_HEADING = "## section (append-only)"

# The two machine-root prefixes omni_home CLAUDE.md rule 6 names by hand.
FORBIDDEN_PREFIXES = ("/Users/", "/Volumes/")


def _ledger_text(rows: int) -> str:
    body = "".join(
        f"2026-09-{(i % 28) + 1:02d}T00:00:0{i % 10}Z | ROW | lane=l{i} | body {i}\n"
        for i in range(rows)
    )
    return "# Ledger\n\nintro\n\n" + SECTION_HEADING + "\n\n" + body


def _run_roll(
    ledger: Path, archive_dir: Path, keep: int
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(ledger),
            "--roll-section",
            "--section-heading",
            SECTION_HEADING,
            "--on-cap",
            "roll",
            "--archive-dir",
            str(archive_dir),
            "--roll-keep-entries",
            str(keep),
            "--max-section-rows",
            "10",
            "--timeout",
            "30",
        ],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def rolled_repo(tmp_path: Path) -> dict[str, object]:
    """A throwaway git repo whose ledger has actually been rolled.

    A real ``.git`` entry is what makes "relative to the repository root" a
    question with an answer, and the roll must be driven -- not simulated --
    so what is asserted is the shipped writer's own bytes.
    """
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    ledger_dir = repo / "docs" / "tracking"
    ledger_dir.mkdir(parents=True)
    ledger = ledger_dir / "LEDGER.md"
    ledger.write_text(_ledger_text(30), encoding="utf-8")
    archive_dir = ledger_dir / "archive"

    proc = _run_roll(ledger, archive_dir, keep=5)
    assert proc.returncode == 0, proc.stderr

    receipt_line = next(
        line
        for line in proc.stdout.splitlines()
        if line.startswith("ledger_lock: ROLL ")
    )
    receipt = json.loads(receipt_line[len("ledger_lock: ROLL ") :])
    assert receipt["entries_rolled"] == 25, receipt

    archive_path = Path(str(receipt["archive"]))
    return {
        "repo": repo,
        "ledger": ledger,
        "live_text": ledger.read_text(encoding="utf-8"),
        "archive_text": archive_path.read_text(encoding="utf-8"),
        "receipt": receipt,
    }


class TestPointerBlockIsPortable:
    def test_no_machine_absolute_path_anywhere_in_the_live_ledger(
        self, rolled_repo: dict[str, object]
    ) -> None:
        text = str(rolled_repo["live_text"])
        root = str(rolled_repo["repo"])
        offenders = [
            line
            for line in text.splitlines()
            if root in line or any(prefix in line for prefix in FORBIDDEN_PREFIXES)
        ]
        assert offenders == [], offenders

    def test_marker_archive_field_is_repo_relative(
        self, rolled_repo: dict[str, object]
    ) -> None:
        text = str(rolled_repo["live_text"])
        marker = next(
            line for line in text.splitlines() if line.startswith("<!-- ledger-roll:")
        )
        payload = json.loads(marker[len("<!-- ledger-roll:") : -len("-->")].strip())
        assert payload[
            "archive"
        ] == "docs/tracking/archive/LEDGER_2026-09-16-split.md".replace(
            "2026-09-16", payload["rolled_at"][:10]
        )
        assert not Path(payload["archive"]).is_absolute()

    def test_prose_pointer_is_the_same_relative_path(
        self, rolled_repo: dict[str, object]
    ) -> None:
        text = str(rolled_repo["live_text"])
        marker = next(
            line for line in text.splitlines() if line.startswith("<!-- ledger-roll:")
        )
        payload = json.loads(marker[len("<!-- ledger-roll:") : -len("-->")].strip())
        prose = next(
            line
            for line in text.splitlines()
            if line.startswith("> Older rows live in")
        )
        assert f"`{payload['archive']}`" in prose


class TestArchiveHeaderIsPortable:
    def test_no_machine_absolute_path_anywhere_in_the_archive(
        self, rolled_repo: dict[str, object]
    ) -> None:
        text = str(rolled_repo["archive_text"])
        root = str(rolled_repo["repo"])
        offenders = [
            line
            for line in text.splitlines()
            if root in line or any(prefix in line for prefix in FORBIDDEN_PREFIXES)
        ]
        assert offenders == [], offenders

    def test_archive_header_source_is_repo_relative(
        self, rolled_repo: dict[str, object]
    ) -> None:
        text = str(rolled_repo["archive_text"])
        header = next(
            line
            for line in text.splitlines()
            if line.startswith("<!-- ledger-roll-archive ")
        )
        payload = json.loads(
            header[len("<!-- ledger-roll-archive ") : -len("-->")].strip()
        )
        assert payload["source"] == "docs/tracking/LEDGER.md"


class TestOutsideARepository:
    """A ledger with no repository root above it still must not leak the machine root."""

    def test_falls_back_to_a_path_relative_to_the_ledger(self, tmp_path: Path) -> None:
        ledger_dir = tmp_path / "loose"
        ledger_dir.mkdir()
        ledger = ledger_dir / "LEDGER.md"
        ledger.write_text(_ledger_text(30), encoding="utf-8")

        proc = _run_roll(ledger, ledger_dir / "archive", keep=5)
        assert proc.returncode == 0, proc.stderr

        live = ledger.read_text(encoding="utf-8")
        assert str(ledger_dir) not in live, live[:400]
        assert not any(prefix in live for prefix in FORBIDDEN_PREFIXES), live[:400]
        marker = next(
            line for line in live.splitlines() if line.startswith("<!-- ledger-roll:")
        )
        payload = json.loads(marker[len("<!-- ledger-roll:") : -len("-->")].strip())
        assert payload["archive"].startswith("archive/")


class TestReceiptIsDeliberatelyAbsolute:
    """Positive control, and the stated exemption.

    The receipt is stdout consumed by the roll trigger on the machine that ran
    it. It is never committed, so an absolute path is the useful form -- and
    its presence proves these tests read a roll that really happened rather
    than an empty file that trivially contains no ``/Users/``.
    """

    def test_receipt_still_carries_the_absolute_path(
        self, rolled_repo: dict[str, object]
    ) -> None:
        receipt = rolled_repo["receipt"]
        assert isinstance(receipt, dict)
        assert Path(str(receipt["archive"])).is_absolute()
        assert Path(str(receipt["ledger"])).is_absolute()
