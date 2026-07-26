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
    assert config.runner_host == "omninode-pc.tail75df5e.ts.net"
    assert config.runner_group == "omnibase-ci"
    assert config.runner_name_prefix == "omninode-runner"
    # OMN-14029: phase-A scale-up to 64 always-on steady-state
    # runners (no burst tier), so burst_count == expected_count.
    assert config.expected_count == 64
    assert config.burst_count == 64


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


def test_runner_compose_has_fleet_uv_concurrency_cap() -> None:
    """OMN-14027 C3: the x-runner-env anchor pins the uv download/build/install
    concurrency ceiling and the 600s HTTP timeout as FLEET defaults so the raw
    ``uv sync`` paths that bypass the hardened setup-python-uv composite
    (OMN-14193) inherit the stampede cap too.

    The value is the deliberate pre-cache fail-safe of 1: the composite already
    pins ``${...:-1}`` = 1, so a fleet default of 1 keeps that proven-safe path
    UNCHANGED while capping the currently-uncapped raw-uv paths down from uv's
    built-in default. A fleet default of 2 would loosen the composite path.
    """
    compose = yaml.safe_load(
        (REPO_ROOT / "docker" / "docker-compose.runners.yml").read_text(
            encoding="utf-8"
        )
    )
    env = compose["x-runner-base"]["environment"]
    assert env["UV_CONCURRENT_DOWNLOADS"] == "1"
    assert env["UV_CONCURRENT_BUILDS"] == "1"
    assert env["UV_CONCURRENT_INSTALLS"] == "1"
    assert env["UV_HTTP_TIMEOUT"] == "600"


def test_runner_compose_pypi_index_wiring_stays_inert() -> None:
    """OMN-14027 C1: the fleet-wide PyPI cache index wiring must stay INERT
    (commented out) until the soak-gated rollout. A merged, active
    ``UV_DEFAULT_INDEX`` would point all 64 runners at a cache host that is not
    yet stood up. This guards against accidentally activating the egress cache
    from the design/canary PR.
    """
    raw = (REPO_ROOT / "docker" / "docker-compose.runners.yml").read_text(
        encoding="utf-8"
    )
    compose = yaml.safe_load(raw)
    env = compose["x-runner-base"]["environment"]
    # Not an active env key...
    assert "UV_DEFAULT_INDEX" not in env
    assert "PIP_INDEX_URL" not in env
    # ...but the shovel-ready wiring exists as an inert comment.
    assert "# UV_DEFAULT_INDEX:" in raw


def test_runner_fleet_config_pypi_cache_is_recorded_but_inert() -> None:
    """OMN-14027 C1: the PyPI pull-through cache endpoint is recorded as fleet
    source-of-truth but stays inert (active=False) until the soak-gated rollout
    wires the runner env. Proves the shovel-ready record parses under the
    strict (extra='forbid') fleet-config model and does not activate the cache.
    """
    config = load_runner_fleet_config(REPO_ROOT / "config" / "runner_fleet.yaml")

    assert config.pypi_cache is not None
    assert config.pypi_cache.active is False
    assert config.pypi_cache.host == "omninode-pc.tail75df5e.ts.net"
    assert config.pypi_cache.port == 3141
    assert config.pypi_cache.simple_index_url.endswith("/root/pypi/+simple/")
    assert config.pypi_cache.fallback_index_url == "https://pypi.org/simple/"


def test_runner_fleet_config_pypi_cache_is_optional(tmp_path: Path) -> None:
    """A fleet config predating the egress-cache work (no pypi_cache block) must
    still validate — the field is optional and defaults to None."""
    minimal = tmp_path / "runner_fleet.yaml"
    minimal.write_text(
        "version: '1.0'\n"
        "github_org: OmniNode-ai\n"
        "runner_host: example.ts.net\n"
        "runner_group: omnibase-ci\n"
        "runner_name_prefix: omninode-runner\n"
        "expected_count: 64\n",
        encoding="utf-8",
    )

    config = load_runner_fleet_config(minimal)

    assert config.pypi_cache is None


def test_runner_scripts_do_not_embed_legacy_count() -> None:
    deploy_script = (REPO_ROOT / "scripts" / "deploy-runners.sh").read_text(
        encoding="utf-8"
    )
    monitor_script = (REPO_ROOT / "docker" / "runners" / "runner-monitor.sh").read_text(
        encoding="utf-8"
    )

    assert "RUNNER_COUNT=10" not in deploy_script
    assert "EXPECTED_RUNNERS=10" not in monitor_script


def test_deploy_runner_monitor_cron_uses_bash_for_source() -> None:
    """Runner monitor cron must not rely on /bin/sh accepting ``source``.

    Cron runs commands with /bin/sh unless SHELL is overridden. On Ubuntu that is
    dash, so a line like ``set -a && source .monitor-env`` exits before loading
    Slack/GitHub credentials and no alert is sent. The deploy script must install
    a cron line that explicitly uses bash and captures setup failures in the log.
    """
    deploy_script = (REPO_ROOT / "scripts" / "deploy-runners.sh").read_text(
        encoding="utf-8"
    )

    assert "/bin/bash -lc" in deploy_script
    assert "source ${monitor_env}" in deploy_script
    assert ">> /tmp/runner-monitor.log 2>&1" in deploy_script
    assert 'local cron_line="*/3 * * * * set -a && source' not in deploy_script


def test_deploy_runner_repair_cron_runs_every_ten_minutes() -> None:
    """Runner repair must be a bounded timer, not an ad hoc operator command."""
    deploy_script = (REPO_ROOT / "scripts" / "deploy-runners.sh").read_text(
        encoding="utf-8"
    )

    assert "*/10 * * * *" in deploy_script
    assert "runner-repair-check" in deploy_script
    assert "MONITOR_AUTO_BOUNCE=1" in deploy_script
    assert "OFFLINE_IDLE_RECREATE_AGE_SECONDS=600" in deploy_script
    assert ">> /tmp/runner-repair.log 2>&1" in deploy_script
    assert "grep -Ev 'runner-monitor|runner-repair-check'" in deploy_script


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
    # OMN-13915: the pgrep pattern is RUNNER_HOME-anchored so wrapper
    # processes (or another runner's listener) can never satisfy it.
    assert 'pgrep -f "${listener_pattern}"' in script
    assert "bin/Runner\\.Listener" in script
    assert "--max-time" in script
    assert "--connect-timeout" in script
    assert "-fsS" in script
    # OMN-13915: the egress curl is gated behind RUNNER_HEALTH_EGRESS_CHECK
    # (default on) and therefore indented — strip before matching.
    curl_commands = [
        shlex.split(line.strip().removeprefix("if ! ").removesuffix("; then").strip())
        for line in script.splitlines()
        if line.strip().startswith("if ! curl ")
    ]
    assert len(curl_commands) == 1
    assert any(arg.startswith("-") and "I" in arg for arg in curl_commands[0])
    endpoint = urlsplit(curl_commands[0][-1])
    assert (endpoint.scheme, endpoint.netloc, endpoint.path) == (
        "https",
        "github.com",
        "/",
    )


def test_runner_entrypoint_disables_self_update_and_relaunches_clean_exit() -> None:
    """Runner self-update can make ``run.sh`` exit 0 and leave no listener.

    The entrypoint must disable self-update on registration and treat clean
    runner exits as relaunchable so a container does not stay Up without
    ``Runner.Listener``.
    """
    script = (REPO_ROOT / "docker" / "runners" / "entrypoint.sh").read_text(
        encoding="utf-8"
    )
    assert "--disableupdate" in script
    assert "Relaunching listener after short backoff" in script
    assert "continue" in script


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


def test_runner_compose_reconciled_to_phase_a_64_fleet() -> None:
    """OMN-14029: the repo compose must match the Phase-A .201 fleet of 64
    always-on steady-state runners, so `deploy-runners.sh` cannot orphan-remove
    live runners beyond 48 (which would shrink the org CI fleet and trigger an
    outage). All 64 runners are steady (no burst profiles) and each mounts the
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
    assert len(runner_services) == 64, "expected exactly 64 runner services"
    # Contiguous runner-1 .. runner-64, no gaps.
    indices = sorted(int(name.rsplit("-", 1)[1]) for name in runner_services)
    assert indices == list(range(1, 65))

    hc_mount = "./runners/healthcheck.sh:/usr/local/bin/healthcheck.sh:ro"
    for name, definition in runner_services.items():
        # All 64 are steady-state: no burst profile gating any runner.
        assert "profiles" not in definition, f"{name} unexpectedly profile-gated"
        assert hc_mount in definition["volumes"], f"{name} missing healthcheck mount"
        assert definition["volumes"][-1] == (
            f"runner-{name.rsplit('-', 1)[1]}-creds:/home/runner/.runner-creds"
        )

    # A backing named volume exists for each of the 64 runners.
    volume_names = {
        name for name in compose["volumes"] if re.fullmatch(r"runner-\d+-creds", name)
    }
    assert len(volume_names) == 64


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
