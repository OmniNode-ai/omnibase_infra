# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Incident replay: the zero-byte lab-load artifact a green run uploaded (OMN-18247).

THE INCIDENT
    ``dev-lane-liveness.yml``'s ``lab-load-probe`` job uploaded an artifact named
    ``lab-load`` containing a ``lab-load.json`` of **zero bytes**, on a workflow
    run whose conclusion was ``success``. That is not a reconstruction: the
    fixture is the byte-for-byte zip GitHub serves for artifact 10298194249 of
    run 34694995666, and its central directory records the member's length as 0.

    Nothing went red. The producing step's redirect (``python3 ... >
    lab-load.json``) creates the file before the interpreter runs, so a crashing
    producer leaves an empty file rather than no file. ``if-no-files-found``
    cannot see that at ANY of its three settings, because the path matches.
    ``warn`` was the setting in use, so even absence would have been a note.

THE REPLAY
    The real guard, ``scripts/ci/assert_evidence_artifact.py``, is driven over
    the extracted bytes and must REJECT them. Direction is ``false_green``: the
    input was genuinely bad and the run said success.

THE DISCRIMINATOR IS LOAD-BEARING
    A guard that rejected everything would satisfy the case above and be
    worthless -- and it would be worse than worthless here, because it would fail
    the lane probe on every run and be removed within a day. So the same guard is
    driven over a second real capture, run 34711967693 at 18:41Z, whose
    ``lab-load.json`` carries an actual measurement, and is required to accept.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD = REPO_ROOT / "scripts" / "ci" / "assert_evidence_artifact.py"
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "omn18247"

# The run whose lab-load.json was zero bytes while the run concluded success.
INCIDENT_CAPTURE = FIXTURES / "lab-load-run34694995666.zip.captured"
# The same probe after OMN-18031's repair, measuring the fleet for real.
CONTROL_CAPTURE = FIXTURES / "lab-load-run34711967693.zip.captured"


def _extract(capture: Path, into: Path) -> Path:
    """Unpack the captured artifact zip and return the record it carries."""
    with zipfile.ZipFile(capture) as archive:
        names = archive.namelist()
        assert names == ["lab-load.json"], (
            f"{capture.name} holds {names}; the capture is supposed to be the "
            "artifact GitHub served for this run, which carried exactly one member."
        )
        for member in archive.infolist():
            target = (into / member.filename).resolve()
            assert target == (into / "lab-load.json").resolve(), (
                f"{capture.name} contains unsafe or unexpected member path "
                f"{member.filename!r}"
            )
            archive.extract(member, into)
    return into / "lab-load.json"


def _run_guard(workdir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(GUARD),
            "--artifact",
            "lab-load",
            "--require",
            "lab-load.json",
        ],
        cwd=workdir,
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_captured_incident_artifact_really_is_zero_bytes(tmp_path: Path) -> None:
    """Without this, the fixture is only a quotation.

    The claim under replay is that a real run uploaded an empty record. Prove it
    from the captured bytes before asserting anything about the guard.
    """
    record = _extract(INCIDENT_CAPTURE, tmp_path)
    assert record.is_file()
    assert record.stat().st_size == 0, (
        "the incident capture no longer carries a zero-byte lab-load.json, so it "
        "is no longer the artifact that failed"
    )


def test_the_real_guard_rejects_the_zero_byte_record_a_green_run_uploaded(
    tmp_path: Path,
) -> None:
    """R5, false_green: the guard must say NO to the input the run said yes to."""
    _extract(INCIDENT_CAPTURE, tmp_path)
    result = _run_guard(tmp_path)
    assert result.returncode == 1, (
        "the guard accepted the zero-byte record from run 34694995666. That run "
        "concluded success with this exact artifact attached, which is the whole "
        "incident.\n" + result.stdout + result.stderr
    )
    assert "EMPTY" in result.stdout
    assert "lab-load" in result.stdout


def test_the_same_guard_accepts_a_real_measurement(tmp_path: Path) -> None:
    """The discriminator: a guard that rejects everything is not a guard.

    Driven over run 34711967693's capture, a genuine fleet measurement, and
    required to pass. Without this, a guard stuck at "reject" would replay the
    incident perfectly and break the lane probe on every subsequent run.
    """
    record = _extract(CONTROL_CAPTURE, tmp_path)
    assert record.stat().st_size > 0
    result = _run_guard(tmp_path)
    assert result.returncode == 0, (
        "the guard rejected a real measured lab-load record.\n"
        + result.stdout
        + result.stderr
    )
    assert "present and non-empty" in result.stdout
