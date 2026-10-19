# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A pre-PR slot's failed migration names its own error (OMN-19064).

The slot bring-up writes prepr_verify_lane.sh's whole output to a build log in
the run's work directory, the teardown removes that directory, and the build
phase printed only the entrypoint's own ``[prepr-verify-lane]`` lines. A live
slot 2 run on h201 (2026-10-19) ended "the intelligence migration failed" with
no psql error left anywhere to read. The prove script now prints the migration
one-shot's own last lines, the ones before the entrypoint's refusal, as
``migration-output`` lines of the build phase.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PROVE_SH = REPO_ROOT / "scripts" / "runtime_build" / "prepr_pool_prove.sh"

pytestmark = pytest.mark.unit

BUILD_LOG = """\
[prepr-verify-lane] building the runtime image from the snapshot ...
#42 exporting layers done
[prepr-verify-lane] running the forward migration against the slot's databases ...
[forward-migration] applied 111_widen_runtime_error_triage_active_unique_to_escalated.sql
[prepr-verify-lane] running the intelligence migration against the slot's intelligence database ...
[intelligence-migration] pre-PR verify slot 'prepr2' active: targeting omniintelligence_prepr2
[intelligence-migration]   apply 0007_create_pattern_store...

psql:/migrations/intelligence/0007_create_pattern_store.sql:3: ERROR:  permission denied for schema public
[prepr-verify-lane] REFUSED/FAILED: the intelligence migration failed against the slot's fresh
  omniintelligence_prepr2. As with the forward migration, on a branch under
  test this is a FINDING. Record it as the slot's verdict.
"""


def _functions() -> str:
    text = PROVE_SH.read_text(encoding="utf-8")
    begin = "# --- slot-migration-output begin ---"
    end = "# --- slot-migration-output end ---"
    assert begin in text and end in text, "slot-migration-output"
    return text[text.index(begin) : text.index(end)]


def _run(log: Path, lines: int | None = None) -> list[str]:
    call = f'slot_migration_output "{log}"'
    if lines is not None:
        call += f" {lines}"
    proc = subprocess.run(
        ["bash", "-c", _functions() + "\n" + call],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.splitlines()


def test_prints_the_migration_error_before_the_refusal(tmp_path: Path) -> None:
    log = tmp_path / "build.log"
    log.write_text(BUILD_LOG, encoding="utf-8")
    out = _run(log)
    assert (
        "migration-output psql:/migrations/intelligence/0007_create_pattern_store.sql:3:"
        " ERROR:  permission denied for schema public"
    ) in out
    assert out[-1].endswith("permission denied for schema public")


def test_leaves_out_the_entrypoint_lines_and_its_refusal_text(tmp_path: Path) -> None:
    log = tmp_path / "build.log"
    log.write_text(BUILD_LOG, encoding="utf-8")
    out = _run(log)
    assert all(line.startswith("migration-output ") for line in out)
    assert not any("[prepr-verify-lane]" in line for line in out)
    assert not any("FINDING" in line for line in out)
    assert "migration-output " not in out


def test_keeps_only_the_last_lines(tmp_path: Path) -> None:
    log = tmp_path / "build.log"
    log.write_text(BUILD_LOG, encoding="utf-8")
    out = _run(log, 2)
    assert out == [
        "migration-output [intelligence-migration]   apply 0007_create_pattern_store...",
        "migration-output psql:/migrations/intelligence/0007_create_pattern_store.sql:3:"
        " ERROR:  permission denied for schema public",
    ]


def test_a_missing_log_prints_one_line_and_does_not_fail(tmp_path: Path) -> None:
    assert _run(tmp_path / "absent.log") == ["migration-output none: no build log"]


def test_the_slot_branch_prints_it_on_a_migration_failure() -> None:
    text = PROVE_SH.read_text(encoding="utf-8")
    branch = text[text.index("9) if grep -q 'migration failed'") :]
    branch = branch[: branch.index(";;")]
    assert 'slot_migration_output "$W/build.log"' in branch
