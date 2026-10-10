# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Every cron entry this repo ships for the lab host records each run's completion (OMN-20805).

A ``CRON`` start record proves a start and nothing else. The automation
monitoring plan (T22) needs every scheduled process to leave a per-run record of
its end and exit, so a run that hangs or fails is distinguishable from a run
that never started. A systemd timer gives that for free: the journal records the
unit's start, its end and its exit status for every run.

What this module pins:

* the maintenance installer ships no ``/etc/cron.d`` unit; each former root
  schedule is a oneshot service with a timer that fires on the same calendar;
* the services let the script's exit status reach systemd (no ``-`` prefix on
  ExecStart, no ``SuccessExitStatus`` widening), because an exit swallowed
  there is an exit the journal cannot record;
* the host sync manifest installs every unit and the sync enables the timers
  and retires the legacy cron files only after the timers are active;
* the runner monitor installer writes user timers, never a crontab line;
* the scanner that enforces this refuses a planted cron source (positive
  control), so a green result is not an empty scan.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MAINTENANCE = REPO_ROOT / "deploy" / "maintenance"
UNIT_DIR = MAINTENANCE / "systemd"
SYNC_SCRIPT = MAINTENANCE / "omninode-host-maintenance-sync.sh"
DEPLOY_RUNNERS = REPO_ROOT / "scripts" / "deploy-runners.sh"
PUSH_LANES_DEPLOY = REPO_ROOT / "scripts" / "push_lanes" / "deploy-push-lanes.sh"
PUSH_LANE_UNITS = REPO_ROOT / "deploy" / "push-lanes"

# unit stem -> (calendar, ExecStart). These are the six root schedules the lab
# host ran from /etc/cron.d before OMN-20805; the calendar is the cron slot
# translated one to one, so the minute separation the cron tests pinned
# (:00/:15/:30/:45 alert, :19 reconcile, :37 sync, :49 converge) still holds.
EXPECTED_ROOT_UNITS: dict[str, tuple[str, str]] = {
    "omninode-host-maintenance-sync": (
        "*-*-* *:37:00",
        "/data/maintenance/bin/omninode-host-maintenance-sync.sh --converge --slack",
    ),
    "omninode-runner-tree-converge": (
        "*-*-* *:49:00",
        "/data/maintenance/bin/omninode-runner-tree-converge.sh --converge",
    ),
    "omninode-workspace-reconcile": (
        "*-*-* *:19:00",
        "/data/maintenance/bin/omninode-workspace-reconcile.sh",
    ),
    "omninode-system-slack-report-alert": (
        "*-*-* *:00/15:00",
        "/data/maintenance/bin/omninode-system-slack-report.sh --mode alert",
    ),
    "omninode-system-slack-report-digest": (
        "*-*-* 08:05:00",
        "/data/maintenance/bin/omninode-system-slack-report.sh --mode digest",
    ),
    "omninode-inference-log-retention": (
        "*-*-* 03:47:00",
        "/usr/bin/find /data/inference/logs /data/inference/opt-vllm/logs"
        " -mindepth 1 -type f -mtime +30 -print -delete",
    ),
}


def cron_sources(root: Path) -> list[str]:
    """Every cron source under ``root`` that the lab host would run.

    A cron source is a file under ``deploy/**/cron.d/`` or a host-maintenance
    manifest row installing into ``/etc/cron.d``. Returns one description per
    source, empty when the tree ships none.
    """
    found: list[str] = []
    for unit in sorted((root / "deploy").glob("**/cron.d/*")):
        if unit.is_file():
            found.append(str(unit.relative_to(root)))
    sync = root / "deploy" / "maintenance" / "omninode-host-maintenance-sync.sh"
    if sync.is_file():
        in_manifest = False
        for line in sync.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("MANIFEST=("):
                in_manifest = True
                continue
            if in_manifest and stripped == ")":
                in_manifest = False
            if in_manifest and stripped.startswith('"') and "|/etc/cron.d/" in stripped:
                found.append(f"{sync.name}: {stripped}")
    return found


def _unit(name: str) -> str:
    path = UNIT_DIR / name
    assert path.is_file(), f"{path.relative_to(REPO_ROOT)} is not shipped"
    return path.read_text(encoding="utf-8")


def _directive(text: str, key: str) -> list[str]:
    return re.findall(rf"^{key}=(.*)$", text, flags=re.MULTILINE)


def test_cron_entries_have_completion_evidence_scanner_finds_a_planted_cron_unit(
    tmp_path: Path,
) -> None:
    """Positive control: the scan that returns nothing for this repo can return something."""
    planted = tmp_path / "deploy" / "maintenance" / "cron.d"
    planted.mkdir(parents=True)
    (planted / "omninode-planted").write_text(
        "7 * * * * root /data/maintenance/bin/planted.sh >> /tmp/planted.log 2>&1\n",
        encoding="utf-8",
    )
    assert cron_sources(tmp_path) == ["deploy/maintenance/cron.d/omninode-planted"]


def test_cron_entries_have_completion_evidence_scanner_finds_a_planted_manifest_row(
    tmp_path: Path,
) -> None:
    sync = tmp_path / "deploy" / "maintenance"
    sync.mkdir(parents=True)
    (sync / "omninode-host-maintenance-sync.sh").write_text(
        'MANIFEST=(\n  "deploy/maintenance/cron.d/x|/etc/cron.d/x|0644"\n)\n',
        encoding="utf-8",
    )
    assert len(cron_sources(tmp_path)) == 1


def test_cron_entries_have_completion_evidence_maintenance_ships_no_cron_unit() -> None:
    assert cron_sources(REPO_ROOT) == [], (
        "a cron unit leaves a start record and no completion record; schedule it "
        "with a systemd timer under deploy/maintenance/systemd instead"
    )


@pytest.mark.parametrize("stem", sorted(EXPECTED_ROOT_UNITS))
def test_cron_entries_have_completion_evidence_root_schedule_is_a_timer(
    stem: str,
) -> None:
    calendar, command = EXPECTED_ROOT_UNITS[stem]
    timer = _unit(f"{stem}.timer")
    service = _unit(f"{stem}.service")

    assert _directive(timer, "OnCalendar") == [calendar]
    assert _directive(timer, "Unit") in ([], [f"{stem}.service"])
    assert "WantedBy=timers.target" in timer

    assert _directive(service, "Type") == ["oneshot"]
    assert _directive(service, "ExecStart")[0] == command, (
        "ExecStart must be the script itself with no '-' prefix: an exit the unit "
        "swallows is an exit the journal cannot record"
    )
    assert _directive(service, "SuccessExitStatus") == []
    assert "HOME=/root" in _directive(service, "Environment"), (
        "cron gave root jobs HOME=/root; the scripts run under set -u and a system "
        "service has no HOME unless it sets one (observed on the lab host: "
        "'HOME: unbound variable' on the first timer run)"
    )
    assert _directive(service, "User") == [], "the cron lines ran as root"


def test_cron_entries_have_completion_evidence_no_unexpected_unit_ships() -> None:
    shipped = {p.stem for p in UNIT_DIR.glob("*.timer")} | {
        p.stem for p in UNIT_DIR.glob("*.service")
    }
    assert shipped == set(EXPECTED_ROOT_UNITS)


def test_cron_entries_have_completion_evidence_root_calendars_do_not_collide() -> None:
    """The minute separation the cron tests pinned, restated for the timers."""
    minutes: dict[str, set[int]] = {}
    for stem, (calendar, _) in EXPECTED_ROOT_UNITS.items():
        hourly = re.fullmatch(r"\*-\*-\* \*:(\d+)(?:/(\d+))?:00", calendar)
        if hourly is None:
            continue  # a daily slot cannot share an hourly minute with a different hour
        start, step = int(hourly.group(1)), hourly.group(2)
        minutes[stem] = set(range(start, 60, int(step))) if step else {start}
    claimed: dict[int, str] = {}
    for stem, mins in minutes.items():
        for minute in mins:
            assert minute not in claimed, (
                f"{stem} and {claimed[minute]} both fire at minute {minute}"
            )
            claimed[minute] = stem


def test_cron_entries_have_completion_evidence_manifest_installs_every_unit() -> None:
    text = SYNC_SCRIPT.read_text(encoding="utf-8")
    for stem in EXPECTED_ROOT_UNITS:
        for suffix in ("service", "timer"):
            row = (
                f'"deploy/maintenance/systemd/{stem}.{suffix}|'
                f'/etc/systemd/system/{stem}.{suffix}|0644"'
            )
            assert row in text, f"manifest does not install {stem}.{suffix}"


def test_cron_entries_have_completion_evidence_sync_retires_cron_after_timers() -> None:
    text = SYNC_SCRIPT.read_text(encoding="utf-8")
    assert "enable --now" in text
    assert "daemon-reload" in text
    assert "RETIRED_CRON_FILES" in text
    enable = text.index('"$SYSTEMCTL" enable --now')
    retire = text.index('mv "$legacy"')
    assert enable < retire, (
        "cron files must be retired only after the timers are active"
    )


def test_cron_entries_have_completion_evidence_runner_monitor_installs_timers() -> None:
    text = DEPLOY_RUNNERS.read_text(encoding="utf-8")
    start = text.index("install_monitor_timers() {")
    body = text[start : text.index("\n}\n", start)]
    # The only crontab write left is the filter that retires the legacy lines.
    assert len(re.findall(r"crontab -(?!l)", body)) == 1
    assert "grep -Ev 'runner-monitor|runner-repair-check' | crontab -" in body
    assert "OnCalendar=${calendar}" in body
    assert "Type=oneshot" in body
    assert "systemctl --user enable --now" in body
    enable = body.index("systemctl --user enable --now")
    retire = body.index("grep -Ev 'runner-monitor|runner-repair-check'")
    assert enable < retire, "legacy cron lines go only after the timers are active"


def test_cron_entries_have_completion_evidence_push_lane_detector_installs_timer() -> (
    None
):
    text = PUSH_LANES_DEPLOY.read_text(encoding="utf-8")
    assert "onex-foreign-prepush-detect.timer" in text
    assert text.index("systemctl --user enable --now") < text.index(
        "grep -v 'omn16968-foreign-prepush-detect'"
    )
    service = (PUSH_LANE_UNITS / "onex-foreign-prepush-detect.service").read_text(
        encoding="utf-8"
    )
    timer = (PUSH_LANE_UNITS / "onex-foreign-prepush-detect.timer").read_text(
        encoding="utf-8"
    )
    assert _directive(service, "Type") == ["oneshot"]
    assert _directive(service, "ExecStart")[0].startswith(
        "/usr/bin/python3 %h/push-lanes/detect_foreign_prepush.py"
    )
    assert _directive(timer, "OnCalendar") == ["*:0/3"]
