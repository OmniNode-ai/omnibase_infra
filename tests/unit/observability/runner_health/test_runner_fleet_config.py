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
    assert config.expected_count == 14
    assert config.burst_count == 20


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
