# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The `.201` scheduler wiring for the workspace reconciler (OMN-17311).

A reconciler nothing runs is a script, not a mechanism (CLAUDE.md rule 5). These
tests pin the two properties that decide whether it actually runs, and keep
running:

1. **It is in the maintenance MANIFEST.** An artifact merged into this repo but
   never installed on the host — with nothing alarming — is the OMN-15525
   condition. It has already bitten twice (the system Slack reporter, then the
   gateway forwarder). The MANIFEST is the mechanism that turns "someone should
   copy this file" into a scheduled check that reddens.

2. **Its cron slot does not collide.** Three root jobs touching the same clones
   and the same Slack workspace need to not run at the same minute. The
   separation is asserted here rather than left in a comment, because a comment
   does not survive the next edit.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MAINTENANCE = _REPO_ROOT / "deploy" / "maintenance"
_CRON_DIR = _MAINTENANCE / "cron.d"
_SYNC_SCRIPT = _MAINTENANCE / "omninode-host-maintenance-sync.sh"
_UNIT = _CRON_DIR / "omninode-workspace-reconcile"
_WRAPPER = _MAINTENANCE / "omninode-workspace-reconcile.sh"

_CRON_LINE = re.compile(
    r"^(?P<minute>\S+)\s+(?P<hour>\S+)\s+\S+\s+\S+\s+\S+\s+(?P<user>\S+)\s+(?P<command>.+)$"
)


def _cron_entries(unit: Path) -> list[re.Match[str]]:
    entries = []
    for line in unit.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" in line.split()[0]:
            continue
        matched = _CRON_LINE.match(line)
        assert matched, f"unparseable cron line in {unit.name}: {line}"
        entries.append(matched)
    return entries


def _expand_minutes(field: str) -> set[int]:
    if field.startswith("*/"):
        step = int(field[2:])
        return set(range(0, 60, step))
    if field == "*":
        return set(range(60))
    return {int(part) for part in field.split(",")}


# --------------------------------------------------------------------------- #
# AC2 -- governed by the manifest, so drift reddens
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("repo_path", "host_path"),
    [
        (
            "deploy/maintenance/omninode-workspace-reconcile.sh",
            "/data/maintenance/bin/omninode-workspace-reconcile.sh",
        ),
        (
            "deploy/maintenance/cron.d/omninode-workspace-reconcile",
            "/etc/cron.d/omninode-workspace-reconcile",
        ),
    ],
)
def test_host_artifacts_are_in_the_sync_manifest(
    repo_path: str, host_path: str
) -> None:
    manifest = _SYNC_SCRIPT.read_text(encoding="utf-8")
    assert f'"{repo_path}|{host_path}|' in manifest, (
        f"{repo_path} is not in omninode-host-maintenance-sync.sh's MANIFEST. "
        "An artifact absent from it is installed by hand, drifts silently, and "
        "nothing alarms — the exact OMN-15525 condition the manifest exists for."
    )


def test_both_host_artifacts_exist_in_the_repo() -> None:
    assert _UNIT.is_file()
    assert _WRAPPER.is_file()


# --------------------------------------------------------------------------- #
# AC3 -- no schedule collision
# --------------------------------------------------------------------------- #
def test_reconcile_slot_does_not_collide_with_any_other_root_cron_job() -> None:
    ours = _cron_entries(_UNIT)
    assert len(ours) == 1, (
        "one reconcile schedule, so there is one thing to reason about"
    )
    our_minutes = _expand_minutes(ours[0].group("minute"))

    for unit in sorted(_CRON_DIR.iterdir()):
        if unit == _UNIT or not unit.is_file():
            continue
        for entry in _cron_entries(unit):
            other = _expand_minutes(entry.group("minute"))
            overlap = our_minutes & other
            assert not overlap, (
                f"the reconcile slot collides with {unit.name} at minute(s) "
                f"{sorted(overlap)}. Three root jobs contending on the same "
                "clones and the same Slack rate limit is avoidable by choosing "
                "a different minute; pick one and update the unit's comment."
            )


def test_the_unit_runs_the_governed_wrapper_and_not_an_inline_recipe() -> None:
    """The cron line must not carry logic of its own.

    A cron line is the least reviewable, least testable place in the system to
    put behaviour, and it is invisible to every gate in this repo.
    """
    command = _cron_entries(_UNIT)[0].group("command")
    assert command.startswith("/data/maintenance/bin/omninode-workspace-reconcile.sh")
    for shell_ism in ("&&", "||", ";", "$("):
        assert shell_ism not in command.split(">>")[0], (
            f"the cron command contains {shell_ism!r}; put it in the wrapper, "
            "which is version-controlled, manifest-governed and testable"
        )


# --------------------------------------------------------------------------- #
# The wrapper stays a scheduler adapter
# --------------------------------------------------------------------------- #
def test_wrapper_delegates_to_the_one_reconciler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One reconciler, every machine. The wrapper must not reimplement it.

    If this file ever grows its own fetch/sync, the Mac and `.201` stop being
    reconciled by the same code and the two hosts start drifting in ways only a
    third incident would reveal.
    """
    source = _WRAPPER.read_text(encoding="utf-8")
    assert "scripts/reconcile-host.sh" in source
    for repair in (
        "uv sync",
        "git pull",
        "git reset --hard",
        "install-node-skill-package",
    ):
        assert repair not in source.split("# Env")[0].replace("#", ""), (
            f"the scheduler adapter performs its own {repair!r}; repair belongs "
            "in reconcile-host.sh so every host runs identical logic"
        )


def test_wrapper_fails_closed_when_the_reconciler_is_absent(tmp_path: Path) -> None:
    """A missing reconciler must be loud, not a silent no-op cron tick."""
    import subprocess

    proc = subprocess.run(
        ["bash", str(_WRAPPER)],
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "OMNI_HOME": str(tmp_path),
            "OMNINODE_ALERT_ENV_FILE": str(tmp_path / "absent.env"),
        },
        timeout=60,
        check=False,
    )
    assert proc.returncode == 3
    assert "no reconciler at" in proc.stderr


def test_wrapper_never_echoes_the_alert_credentials(tmp_path: Path) -> None:
    """A secret printed into a root cron log outlives the run.

    The wrapper sources an env file that carries a Slack bot token, so this is a
    real exposure path rather than a hypothetical one.
    """
    import subprocess

    env_file = tmp_path / "alert.env"
    env_file.write_text(
        "SLACK_BOT_TOKEN=xoxb-canary-must-not-appear\n", encoding="utf-8"
    )

    proc = subprocess.run(
        ["bash", "-x", str(_WRAPPER)],
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "OMNI_HOME": str(tmp_path),
            "OMNINODE_ALERT_ENV_FILE": str(env_file),
        },
        timeout=60,
        check=False,
    )
    assert "xoxb-canary-must-not-appear" not in proc.stdout
    assert "xoxb-canary-must-not-appear" not in proc.stderr
