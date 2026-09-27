# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Incident replay for the lab probe-window check (OMN-19412).

THE INCIDENT. Before ``config/lab_probe_windows.yaml`` the only list of probe
windows was one ledger row (omni_home ``docs/tracking/ROLLING_WORK_LEDGER.md``,
2026-09-23T20:23:28Z, lane merge-drain-7f). It named C16 and C15 and set four
runtime-merge blackouts from them, 20:45Z-21:45Z, 22:55Z-23:45Z,
00:55Z-01:45Z and 02:45Z-03:45Z. It did not name C11, which reads the same dev
lane at 02:17Z, 08:17Z, 14:17Z and 20:17Z. A runtime merge at 02:00Z was
therefore permitted by the row's rules and would have redeployed the lane under
C11's run. (Whether one did is not claimed here; the list's gap is the defect.)

THE ARTIFACT is the C11 workflow exactly as it stood on omnibase_infra ``dev``
at 186be1a4d, fetched with ``git show`` and committed unmodified. The guard is
driven over those bytes with the list the ledger row carried (C15 and C16 only)
and must refuse, naming C11's workflow as an unlisted probe. The discriminator
drives it over the same bytes with the committed window file's C11 entry and
requires it to accept, so a guard that refused everything cannot pass.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, time, timedelta
from pathlib import Path

import pytest

from scripts.ci import check_lab_probe_windows as plw

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO_ROOT / "tests/fixtures/omn19412/chain-canary-c11-negative-paths.yml.captured"
)
FIXTURE_SHA256 = "adb37c7c26e5c96f02ee4054f3a0d04ba1c3941865a0cd56e9d376445f98c12a"
C11_PATH = ".github/workflows/chain-canary-c11-negative-paths.yml"

# The row's blackouts, UTC, as the row states them.
ROW_BLACKOUTS = (
    (time(20, 45), time(21, 45)),
    (time(22, 55), time(23, 45)),
    (time(0, 55), time(1, 45)),
    (time(2, 45), time(3, 45)),
)
ROW_NAMED_PROBES = ("C15", "C16")


def _root_with_the_captured_bytes(tmp_path: Path) -> Path:
    root = tmp_path / "omnibase_infra"
    target = root / C11_PATH
    target.parent.mkdir(parents=True)
    target.write_bytes(FIXTURE.read_bytes())
    return root


def _committed(ids: tuple[str, ...]) -> list[plw.ProbeWindow]:
    windows = plw.load_windows(REPO_ROOT / "config/lab_probe_windows.yaml")
    return [w for w in windows if w.id in ids]


@pytest.mark.unit
def test_the_fixture_is_the_captured_bytes() -> None:
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256


@pytest.mark.unit
def test_the_row_left_c11_runs_outside_every_blackout() -> None:
    """Not vacuous: the captured workflow really does run on the dev lane in a
    gap the row left open."""
    workflow = plw.read_workflow(FIXTURE)
    assert plw.job_name_lanes(workflow) == {"dev"}
    (cron,) = plw.workflow_crons(workflow)
    day = datetime(2026, 9, 24, tzinfo=UTC)
    probe = plw.ProbeWindow(
        "C11", "c11", "omnibase_infra", C11_PATH, (cron,), "dev", "job_name", 10
    )
    starts = probe.occurrences(day, day + timedelta(days=1))
    assert starts
    for start in starts:
        assert not any(lo <= start.time() <= hi for lo, hi in ROW_BLACKOUTS), start


@pytest.mark.unit
def test_the_real_guard_refuses_the_row_list_on_the_captured_c11(
    tmp_path: Path,
) -> None:
    root = _root_with_the_captured_bytes(tmp_path)
    row_list = _committed(ROW_NAMED_PROBES)
    assert {w.id for w in row_list} == set(ROW_NAMED_PROBES)
    # Only the unlisted direction is under test here: the row's own two probes
    # are judged against their own workflows elsewhere, so drop their
    # "workflow not found" errors against this one-file root.
    errors = [
        e
        for e in plw.check(row_list, {"omnibase_infra": root})
        if e.startswith("unlisted")
    ]
    assert len(errors) == 1, errors
    assert f"omnibase_infra:{C11_PATH}" in errors[0]
    assert "17 2,8,14,20 * * *" in errors[0]


@pytest.mark.unit
def test_the_same_guard_accepts_the_same_bytes_with_the_committed_entry(
    tmp_path: Path,
) -> None:
    root = _root_with_the_captured_bytes(tmp_path)
    assert plw.check(_committed(("C11",)), {"omnibase_infra": root}) == []
