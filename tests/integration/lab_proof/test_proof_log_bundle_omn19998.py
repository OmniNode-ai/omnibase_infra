# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Proof logs survive a real shell read-back and keep the failing step (OMN-19998).

The bundle's read-back command runs in a real shell against a real work
directory; the archive it prints is parsed, kept on disk and read for the failing
build step. The ledger streak alarm is checked against real ledger rows.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from omnibase_infra.models.lab_proof.model_proof_ledger import ModelProofLedger
from omnibase_infra.models.lab_proof.model_proof_log_bundle import (
    ARCHIVE_KEY,
    ModelProofLogBundle,
)

pytestmark = [pytest.mark.integration]

BUILD_LOG = "\n".join(
    [
        "#1 [stage 1/2] RUN true",
        "#1 DONE 0.1s",
        "#2 [stage 2/2] RUN make local-env",
        '#2 ERROR: process "/bin/sh -c make local-env" did not complete successfully',
        "trailing line",
    ]
)


@pytest.mark.skipif(
    not all(shutil.which(tool) for tool in ("tar", "base64", "tail", "sh")),
    reason="needs tar, base64, tail and sh",
)
def test_logs_read_in_a_real_shell_survive_removal_of_the_work_directory(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    (work / "build.log").write_text(BUILD_LOG, encoding="utf-8")
    command = ModelProofLogBundle.fetch_command(str(work))
    done = subprocess.run(
        ["sh", "-c", command], capture_output=True, text=True, check=True
    )
    shutil.rmtree(work)

    kept_dir = tmp_path / "receipts" / "logs"
    proof_text, build_output = ModelProofLogBundle.keep(
        {"build": "build rc=2", ARCHIVE_KEY: done.stdout}, kept_dir
    )

    assert (kept_dir / "build.log").read_text(encoding="utf-8") == BUILD_LOG
    assert f"proof-logs kept={kept_dir}" in proof_text
    assert "build-failing-step step #2" in build_output
    assert "make local-env" in build_output


def test_two_inconclusive_proofs_in_a_row_write_one_alarm_row() -> None:
    ledger = ModelProofLedger(
        lines=(
            "2026-11-15T19:00:00Z | RELEASE | lane=a | re=h1 | "
            "LAB PROOF INCONCLUSIVE: repo_x#1 on host",
        )
    )
    rows: list[str] = []
    notes = ledger.alarm_on_streak(
        "INCONCLUSIVE", ["repo_x"], "h2", "2026-11-15T20:00:00Z", rows.append
    )

    assert len(rows) == 1
    assert "proofs=h1,h2" in rows[0]
    assert "alarm written" in notes
