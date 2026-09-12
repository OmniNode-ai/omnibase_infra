# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Incident replay: the probe died, the run stayed green (OMN-18253).

THE INCIDENT, in the runner's own words
    ``tests/fixtures/omn18253/lab-load-probe-job103556840056.log.gz.captured``
    is the verbatim log of the ``lab-load-probe`` job in run 34694995666. It records
    a ``Traceback``, ``ModuleNotFoundError: No module named 'yaml'``, and
    ``Process completed with exit code 1``.

    The job's conclusion is ``failure``. **The run's conclusion is ``success``**,
    because ``continue-on-error: true`` sat on the job. The artifact that run
    uploaded was a zero-byte ``lab-load.json``, because ``> lab-load.json``
    creates the file before the interpreter runs.

    So the probe crashed, said so in its own log, and nothing anywhere above it
    changed colour. That is the false green, and it held for the job's whole life.

THE REPLAY
    The real module, ``scripts/ci/probe_lab_load.py``, is driven over an import
    failure of exactly that shape and must REJECT it: exit non-zero AND leave a
    non-empty record naming the cause. Both halves matter, and neither alone
    closes the incident. Exiting non-zero without a record reproduces the
    zero-byte artifact; writing a record without exiting non-zero reproduces the
    green step.

THE DISCRIMINATOR IS LOAD-BEARING
    A module that always exited non-zero would replay this perfectly and turn
    every scheduled run red, which is how the suppression would come back. The
    discriminator drives the same module over a SATURATED lab -- the state the
    monitor exists to report -- and requires exit 0.
"""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD = REPO_ROOT / "scripts" / "ci" / "probe_lab_load.py"
CAPTURE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "omn18253"
    / "lab-load-probe-job103556840056.log.gz.captured"
)

# The import failure the capture records, reproduced as a module the probe
# loader will fail on for the same reason: the name does not resolve.
UNIMPORTABLE = "import yaml_that_does_not_exist  # noqa\n"
SATURATED = (
    "def probe_lab_saturation_from_fleet(token, group, api):\n"
    '    return {"ok": True, "age_seconds": 0,\n'
    '            "hosts": [{"label": "org:omnibase-ci", "ratio": 1.0,'
    ' "free_mem_mib": 121697}]}\n'
)


def _drive(
    tmp_path: Path, probe_src: str
) -> tuple[subprocess.CompletedProcess[str], Path]:
    (tmp_path / "stub_probe.py").write_text(probe_src, encoding="utf-8")
    out = tmp_path / "lab-load.json"
    result = subprocess.run(
        [sys.executable, str(GUARD), "--out", str(out), "--probe-module", "stub_probe"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(tmp_path)},
    )
    return result, out


def test_the_capture_records_a_crashed_probe_under_a_green_run() -> None:
    """Without this the fixture is only a quotation.

    Prove the two facts the replay rests on -- the probe died, and it died with
    the import error -- from the captured bytes themselves.
    """
    assert CAPTURE.is_file(), f"{CAPTURE} is missing"
    # Stored gzipped, and the compression is deterministic (`gzip -n`, no
    # mtime) so the committed sha256 pins the exact log bytes. The raw log
    # carries the runner's ANSI escape sequences, and a committed file holding
    # them makes `gh pr diff` refuse the whole diff, which takes the hostile
    # reviewer down with it. Compressing keeps the bytes byte-for-byte rather
    # than stripping them, which would have made the capture no longer the log
    # that failed.
    log = gzip.decompress(CAPTURE.read_bytes()).decode("utf-8", errors="replace")
    assert "ModuleNotFoundError: No module named 'yaml'" in log
    assert "Process completed with exit code 1" in log


def test_the_real_module_rejects_the_import_failure_that_ran_green(
    tmp_path: Path,
) -> None:
    """R5, false_green: non-zero exit AND an honest record, not one or the other."""
    result, out = _drive(tmp_path, UNIMPORTABLE)
    assert result.returncode != 0, (
        "the module accepted the import failure that run 34694995666 reported as "
        "a crashed step under a green run.\n" + result.stdout + result.stderr
    )
    assert out.is_file(), "the record is absent, which is the zero-byte artifact again"
    text = out.read_text(encoding="utf-8")
    assert text.strip(), "the record is blank, which is the zero-byte artifact again"
    record = json.loads(text)
    assert record["ok"] is False
    assert record["error"].startswith("probe_unimportable:ModuleNotFoundError")


def test_the_same_module_accepts_a_fully_saturated_lab(tmp_path: Path) -> None:
    """The discriminator, and it is not hypothetical.

    A live read of the org runner registry at 2026-09-12T22:20:35Z returned
    ``ratio: 1.0`` -- every runner in the group busy -- and the repaired probe
    exited 0 on it. Those are the numbers used here. A module that failed on
    that reading would paint the scheduler red through every busy hour and the
    suppression this ticket removed would be restored within a week.
    """
    result, out = _drive(tmp_path, SATURATED)
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["ok"] is True
    assert record["hosts"][0]["ratio"] == 1.0
