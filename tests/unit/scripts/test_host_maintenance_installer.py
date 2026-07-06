# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Static checks for the host maintenance installer (OMN-14030)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
INSTALLER = REPO_ROOT / "deploy" / "disk-gc" / "install-host-maintenance.sh"
DEPLOY_RUNNERS = REPO_ROOT / "scripts" / "deploy-runners.sh"


def test_host_maintenance_installer_exists_and_is_valid_bash() -> None:
    assert INSTALLER.exists()
    result = subprocess.run(
        ["bash", "-n", str(INSTALLER)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_host_maintenance_installer_covers_both_timer_layers() -> None:
    body = INSTALLER.read_text(encoding="utf-8")
    assert "install-disk-gc.sh" in body
    assert "install-worktree-reaper.sh" in body
    assert "onex-disk-gc.timer" in body
    assert "onex-worktree-reaper.timer" in body
    assert "--retire-legacy-cron" in body
    assert "/etc/cron.d/docker-prune" in body
    assert "/etc/cron.d/omninode-docker-anonymous-volume-prune" in body


def test_host_maintenance_status_has_json_mode() -> None:
    body = INSTALLER.read_text(encoding="utf-8")
    assert "--status --json" in body
    assert "json.dumps" in body
    assert "systemctl" in body
    assert "NextElapseUSecRealtime" in body
    assert "legacy_cron" in body


def test_deploy_runners_no_longer_installs_legacy_docker_prune_cron() -> None:
    body = DEPLOY_RUNNERS.read_text(encoding="utf-8")
    assert "/etc/cron.d/docker-prune" not in body
    assert "sudo tee" not in body
    assert "install-host-maintenance.sh" in body


def test_disk_gc_service_runs_repo_owned_volume_gc() -> None:
    service = REPO_ROOT / "deploy" / "disk-gc" / "onex-disk-gc.service"
    body = service.read_text(encoding="utf-8")
    assert "scripts/docker-volume-gc.sh --execute" in body
    assert "unused ephemeral" in body


def test_worktree_reaper_service_soft_fails_projection_outages() -> None:
    service = REPO_ROOT / "deploy" / "disk-gc" / "onex-worktree-reaper.service"
    body = service.read_text(encoding="utf-8")
    assert "ExecStart=-/usr/bin/env python3" in body
    assert "transient projection API outages" in body


def test_docker_volume_gc_only_targets_disposable_volume_classes() -> None:
    script = REPO_ROOT / "scripts" / "docker-volume-gc.sh"
    body = script.read_text(encoding="utf-8")
    assert "unused-anonymous-volume" in body
    assert "unused-ephemeral-boot-redpanda-volume" in body
    assert "unused-legacy-redpanda-volume" in body
    assert "omnibase-infra-boot-" in body
    assert "omnibase-infra-prod-redpanda-data" not in body
    assert "docker ps -aq" in body
    assert "docker inspect --format" in body
    assert ".Mounts" in body
