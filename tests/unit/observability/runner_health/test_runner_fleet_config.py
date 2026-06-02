# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for authoritative runner fleet configuration."""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from omnibase_infra.observability.runner_health.model_runner_fleet_config import (
    load_runner_fleet_config,
)

REPO_ROOT = Path(__file__).parents[4]


def test_runner_fleet_config_loads_from_repo_config() -> None:
    config = load_runner_fleet_config(REPO_ROOT / "config" / "runner_fleet.yaml")

    assert config.github_org == "OmniNode-ai"
    assert config.runner_group == "omnibase-ci"
    assert config.runner_name_prefix == "omninode-runner"
    # OMN-12582: reconciled to the live .201 fleet of 48 always-on steady-state
    # runners (no burst tier), so burst_count == expected_count.
    assert config.expected_count == 48
    assert config.burst_count == 48


def test_runner_compose_matches_configured_count() -> None:
    config = load_runner_fleet_config(REPO_ROOT / "config" / "runner_fleet.yaml")
    compose = yaml.safe_load(
        (REPO_ROOT / "docker" / "docker-compose.runners.yml").read_text(
            encoding="utf-8"
        )
    )

    services = compose["services"]
    steady_runner_services = [
        name
        for name, definition in services.items()
        if re.fullmatch(rf"{config.runner_name_prefix}-\d+", name)
        and "profiles" not in definition
    ]
    all_runner_services = [
        name
        for name in services
        if re.fullmatch(rf"{config.runner_name_prefix}-\d+", name)
    ]

    assert len(steady_runner_services) == config.expected_count
    assert len(all_runner_services) == config.burst_count


def test_runner_compose_resource_limits_match_live_capacity() -> None:
    compose = yaml.safe_load(
        (REPO_ROOT / "docker" / "docker-compose.runners.yml").read_text(
            encoding="utf-8"
        )
    )

    base = compose["x-runner-base"]
    assert base["mem_limit"] == "6g"
    assert base["memswap_limit"] == "12g"
    assert base["cpus"] == "2.0"
    assert base["pids_limit"] == 4096


def test_runner_scripts_do_not_embed_legacy_count() -> None:
    deploy_script = (REPO_ROOT / "scripts" / "deploy-runners.sh").read_text(
        encoding="utf-8"
    )
    monitor_script = (REPO_ROOT / "docker" / "runners" / "runner-monitor.sh").read_text(
        encoding="utf-8"
    )

    assert "RUNNER_COUNT=10" not in deploy_script
    assert "EXPECTED_RUNNERS=10" not in monitor_script


def test_runner_healthcheck_probes_github_egress() -> None:
    """OMN-12433: the runner healthcheck must verify github.com egress.

    A pgrep-only healthcheck passes while a runner has silently lost its
    connection to GitHub (egress fault), letting dead runners stay "healthy"
    in Docker and wedge the merge queue. The healthcheck script must prove both
    the listener is alive AND github.com is reachable.
    """
    script = (REPO_ROOT / "docker" / "runners" / "healthcheck.sh").read_text(
        encoding="utf-8"
    )
    assert "pgrep -f Runner.Listener" in script
    assert "--max-time" in script
    curl_commands = [
        shlex.split(line.removeprefix("if ! ").removesuffix("; then").strip())
        for line in script.splitlines()
        if line.startswith("if ! curl ")
    ]
    assert len(curl_commands) == 1
    endpoint = urlsplit(curl_commands[0][-1])
    assert (endpoint.scheme, endpoint.netloc, endpoint.path) == (
        "https",
        "github.com",
        "/",
    )


def test_runner_compose_healthcheck_uses_egress_script() -> None:
    """OMN-12433: every runner service must run the egress healthcheck script,
    not the old pgrep-only test, and mount the script into the container."""
    compose = yaml.safe_load(
        (REPO_ROOT / "docker" / "docker-compose.runners.yml").read_text(
            encoding="utf-8"
        )
    )

    base_test = compose["x-runner-base"]["healthcheck"]["test"]
    assert base_test == ["CMD-SHELL", "/usr/local/bin/healthcheck.sh"]

    hc_mount = "./runners/healthcheck.sh:/usr/local/bin/healthcheck.sh:ro"
    for name, definition in compose["services"].items():
        if not re.fullmatch(r"omninode-runner-\d+", name):
            continue
        # Per-service volumes override the anchor (YAML lists don't deep-merge),
        # so each runner must mount the healthcheck script explicitly.
        assert hc_mount in definition["volumes"], f"{name} missing healthcheck mount"
        # No runner may regress to the bare pgrep-only healthcheck.
        resolved_test = definition.get("healthcheck", {}).get("test", base_test)
        assert resolved_test == ["CMD-SHELL", "/usr/local/bin/healthcheck.sh"]


def test_runner_compose_reconciled_to_live_48_fleet() -> None:
    """OMN-12582: the repo compose must match the proven live .201 fleet of 48
    always-on steady-state runners, so `deploy-runners.sh` cannot orphan-remove
    live runners 21-48 (which would drop the org CI fleet 48->20 and trigger an
    outage). All 48 runners are steady (no burst profiles) and each mounts the
    OMN-12433 egress healthcheck script.
    """
    compose = yaml.safe_load(
        (REPO_ROOT / "docker" / "docker-compose.runners.yml").read_text(
            encoding="utf-8"
        )
    )

    runner_services = {
        name: definition
        for name, definition in compose["services"].items()
        if re.fullmatch(r"omninode-runner-\d+", name)
    }
    assert len(runner_services) == 48, "expected exactly 48 runner services"
    # Contiguous runner-1 .. runner-48, no gaps.
    indices = sorted(int(name.rsplit("-", 1)[1]) for name in runner_services)
    assert indices == list(range(1, 49))

    hc_mount = "./runners/healthcheck.sh:/usr/local/bin/healthcheck.sh:ro"
    for name, definition in runner_services.items():
        # All 48 are steady-state: no burst profile gating any runner.
        assert "profiles" not in definition, f"{name} unexpectedly profile-gated"
        assert hc_mount in definition["volumes"], f"{name} missing healthcheck mount"
        assert definition["volumes"][-1] == (
            f"runner-{name.rsplit('-', 1)[1]}-creds:/home/runner/.runner-creds"
        )

    # A backing named volume exists for each of the 48 runners.
    volume_names = {
        name for name in compose["volumes"] if re.fullmatch(r"runner-\d+-creds", name)
    }
    assert len(volume_names) == 48


def test_deploy_ships_healthcheck_script_to_host() -> None:
    """OMN-12582: the compose bind-mounts ./runners/healthcheck.sh, so the deploy
    rsync MUST ship that file to the host. Without it the bind mount resolves to
    an empty path on the host and every runner's healthcheck breaks. This guards
    the latent gap where OMN-12433 added the mount but deploy never synced the
    artifact.
    """
    deploy_script = (REPO_ROOT / "scripts" / "deploy-runners.sh").read_text(
        encoding="utf-8"
    )
    # Declared in the SYNC_PATHS manifest (drives the dry-run log).
    assert '"docker/runners/healthcheck.sh"' in deploy_script
    # And in the real rsync invocation that ships into docker/runners/.
    assert '"${REPO_ROOT}/docker/runners/healthcheck.sh" \\' in deploy_script
