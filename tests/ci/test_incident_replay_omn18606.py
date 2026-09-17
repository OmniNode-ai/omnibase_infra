# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Incident replay — the census that could not refresh itself (OMN-18606, OMN-15547).

THE INCIDENT. `deploy/lane-census/census-snapshot.json` carried
`emitted_at 2026-09-10T10:06:57Z`. Seven days later, at 2026-09-17T10:07Z, the
`Lane Census Staleness` gate went red on every `omnibase_infra` PR touching the
census paths. Nothing on the lab host had failed: the hourly
`onex-disk-gc.service` reported `SUCCESS` every hour, and it reported `SUCCESS`
*because the census pass it was supposed to run wrote nothing and there was
nothing to fail*. That is the false_green: an hourly job whose green is
indistinguishable from an hourly job that is not doing the thing.

Both artifacts below are captured bytes, not reconstructions.

  1. `stale-census-snapshot-2e4eec05.json.captured` — the actual file that went
     stale, read out of the git object store at the commit that last touched it
     before OMN-18606 (`2e4eec05`, OMN-18320).

  2. `installed-dropin-no-snapshot.conf.captured` — the drop-in as it was
     ACTUALLY INSTALLED on the lab host, `scp`'d off `.201` while it was still
     the pre-fix copy. Its single `ExecStart` names the collector with no output
     destination of any kind. This is the configuration whose hourly `SUCCESS`
     meant nothing.

WHAT WOULD HAVE CAUGHT IT. Nothing did, which is the point — there was no
decision anywhere that looked at the committed census and said "this needs to
move". `scripts/lane_census_refresh_decision.py` is that decision, and the first
test drives it with the real stale bytes at the real moment the gate went red.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO / "scripts"
_FIXTURES = _REPO / "tests" / "fixtures" / "omn18606"
_STALE_CENSUS = _FIXTURES / "stale-census-snapshot-2e4eec05.json.captured"
_INSTALLED_DROPIN = _FIXTURES / "installed-dropin-no-snapshot.conf.captured"
_CENSUS_SH = _SCRIPTS / "lane-census-check.sh"
_REPO_DROPIN = (
    _REPO / "deploy" / "lane-census" / "onex-disk-gc.service.d" / "20-lane-census.conf"
)

sys.path.insert(0, str(_SCRIPTS))

from lane_census_refresh_decision import (
    REASON_AGING,
    decide_refresh,
)

# The instant the staleness gate first went red on this census: seven days and
# ten minutes after the captured artifact's own emitted_at.
_WENT_RED_AT = datetime(2026, 9, 17, 10, 7, 0, tzinfo=UTC)


def _exec_start_argv(dropin_text: str) -> list[str]:
    """The argv a systemd `ExecStart=` line in this drop-in actually runs."""
    lines = [line for line in dropin_text.splitlines() if line.startswith("ExecStart=")]
    assert len(lines) == 1, f"expected exactly one ExecStart, got {lines}"
    # Strip the `ExecStart=` key and systemd's leading `-` (fail-soft marker).
    return lines[0].removeprefix("ExecStart=").lstrip("-").split()


def test_the_real_decision_refuses_to_leave_the_stale_census_alone() -> None:
    """The captured stale census, at the moment it went red, must say REFRESH.

    This is the false_green pin. On the pre-OMN-18606 tree nothing evaluated the
    committed census at all, so this artifact sat in the repository for seven
    days and then broke every PR that touched its paths. Driven here with the
    real bytes and the real timestamp.
    """
    committed = json.loads(_STALE_CENSUS.read_text(encoding="utf-8"))
    assert committed["emitted_at"] == "2026-09-10T10:06:57.003308+00:00", (
        "the captured artifact is not the one that went stale"
    )

    candidate = dict(committed)
    candidate["emitted_at"] = _WENT_RED_AT.isoformat()

    decision = decide_refresh(committed, candidate, now=_WENT_RED_AT)

    assert decision.refresh is True, (
        "the real stale census must be refreshed; leaving it is what produced "
        "six hand-opened heal PRs and a seventh during OMN-18606"
    )
    assert decision.reason == REASON_AGING
    assert "7d old" in decision.detail


def test_the_same_decision_stays_quiet_three_days_earlier() -> None:
    """The same artifact, inside the refresh window, is a no-op.

    Without this the test above would also pass for a decision that says REFRESH
    unconditionally — which would open four no-op PRs a day and get the leg
    removed. Same real bytes, earlier clock.
    """
    committed = json.loads(_STALE_CENSUS.read_text(encoding="utf-8"))
    emitted = datetime.fromisoformat(committed["emitted_at"])
    day_one = emitted + timedelta(days=1)

    candidate = dict(committed)
    candidate["emitted_at"] = day_one.isoformat()

    decision = decide_refresh(committed, candidate, now=day_one)

    assert decision.refresh is False, (
        "a one-day-old census describing an unchanged fleet is not worth a PR"
    )


def test_the_dropin_actually_installed_on_the_host_could_not_write_a_census() -> None:
    """The captured drop-in names the collector with NO output destination.

    This is why the hourly unit's `SUCCESS` was worthless: the pass ran, the
    reconcile happened, a drift event went to the bus, and the file the
    staleness gate reads was never touched. Asserted against the bytes that
    were on `.201`, not against a description of them.
    """
    argv = _exec_start_argv(_INSTALLED_DROPIN.read_text(encoding="utf-8"))

    assert any("lane-census-check.sh" in part for part in argv), (
        f"captured drop-in does not invoke the collector at all: {argv}"
    )
    assert not any(part.startswith("--snapshot") for part in argv), (
        "the captured artifact is supposed to be the PRE-fix drop-in; if it "
        f"carries --snapshot the capture is wrong: {argv}"
    )
    # No redirect either — a shell redirect would have to appear as an argument
    # here, because systemd ExecStart is execve, not a shell.
    assert not any(part.startswith(">") for part in argv), (
        f"systemd does not run a shell, so a redirect here never worked: {argv}"
    )


def test_the_repo_dropin_now_carries_an_output_destination() -> None:
    """The fixed configuration, for contrast with the captured one."""
    argv = _exec_start_argv(_REPO_DROPIN.read_text(encoding="utf-8"))
    assert "--snapshot" in argv, f"the shipped drop-in still writes no census: {argv}"
    assert argv[argv.index("--snapshot") + 1].endswith(".json"), (
        f"--snapshot must name a destination file: {argv}"
    )


def test_the_real_collector_accepts_the_fixed_dropins_argv() -> None:
    """Drive the REAL script with the fixed drop-in's flags and prove it parses.

    The pre-fix collector answered `Unknown argument: --snapshot` with exit 2 —
    so the one argv that could have made the hourly pass write a census was
    rejected by the script itself. Exit 2 is the argument-parse refusal; 3
    (docker absent in the test environment) and 4 (inventory unobservable) are
    both fine here, because this asserts the argument contract and not the
    collection.
    """
    argv = _exec_start_argv(_REPO_DROPIN.read_text(encoding="utf-8"))
    flags = list(argv[argv.index("--snapshot") :])
    flags[flags.index("--snapshot") + 1] = "/dev/null"

    result = subprocess.run(
        ["bash", str(_CENSUS_SH), *flags],
        capture_output=True,
        text=True,
        check=False,
        cwd=_REPO,
    )

    assert result.returncode != 2, (
        "the collector rejected the fixed drop-in's own argv, which is the "
        f"pre-fix failure exactly: {result.stderr.strip()}"
    )
    assert "Unknown argument" not in result.stderr, result.stderr
