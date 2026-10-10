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
    # OMN-18411: capped 88 -> 60 after a CI burst drove one-minute load to
    # 100.9 on the 32-core host -- the registered runner count is the
    # worst-case concurrent-job ceiling. OMN-19077 proposed capping 60 -> 40
    # after 58 concurrent jobs drove IO pressure to 52.8 and crashed dockerd;
    # operator ruling 2026-09-24 (ROLLING_WORK_LEDGER.md:3794) kept the fleet
    # at 60 and fixed the incident with the aggregate omnirunners.slice
    # cgroup instead. All 60 are always-on steady-state (no burst tier), so
    # burst_count == expected_count. OMN-19895 (operator consent
    # ROLLING_WORK_LEDGER.md:13156) SPREAD the 60 across two hosts: these
    # scalars are the primary host's 44, and .202's omnipc2-ci-runner pool
    # carries the other 16, so the action fleet's declared total stays 60.
    assert config.expected_count == 44
    assert config.burst_count == 44
    assert config.declared_total("action") == 60


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
    ``UV_DEFAULT_INDEX`` would point all 88 runners at a cache host that is not
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


def test_runner_fleet_config_pypi_cache_is_recorded_but_fleet_inert() -> None:
    """OMN-14027 C1: the PyPI pull-through cache endpoint is recorded as fleet
    source-of-truth and stays FLEET-inert (active=False) until the soak-gated
    step-5 rollout wires the fleet env. Proves the record parses under the strict
    (extra='forbid') fleet-config model and does not activate the cache fleet-wide.

    ``active`` means FLEET-active. A non-empty ``canary_runners`` with
    ``active=False`` is the normal mid-rollout state and is asserted separately
    below — do not read a live canary as fleet activation.
    """
    config = load_runner_fleet_config(REPO_ROOT / "config" / "runner_fleet.yaml")

    assert config.pypi_cache is not None
    assert config.pypi_cache.active is False
    assert config.pypi_cache.host == "omninode-pc.tail75df5e.ts.net"
    assert config.pypi_cache.port == 3141
    assert config.pypi_cache.simple_index_url.endswith("/root/pypi/+simple/")
    assert config.pypi_cache.fallback_index_url == "https://pypi.org/simple/"


def test_pypi_canary_membership_matches_override_file() -> None:
    """OMN-14027 C1: ``pypi_cache.canary_runners`` must name exactly the services
    wired in docker/docker-compose.pypi-canary.yml.

    Without this gate the two drift silently and the recorded canary membership
    stops describing what is actually wired — the same class of failure as the
    canary that reverted unnoticed, just in the config plane instead of the
    runtime plane.
    """
    config = load_runner_fleet_config(REPO_ROOT / "config" / "runner_fleet.yaml")
    assert config.pypi_cache is not None

    override_path = REPO_ROOT / "docker" / "docker-compose.pypi-canary.yml"
    override = yaml.safe_load(override_path.read_text(encoding="utf-8"))
    wired = set(override["services"])

    assert set(config.pypi_cache.canary_runners) == wired, (
        "config/runner_fleet.yaml pypi_cache.canary_runners "
        f"({sorted(config.pypi_cache.canary_runners)}) does not match the services "
        f"wired in {override_path.name} ({sorted(wired)})"
    )

    # Every canary member must be a real steady-state runner service.
    compose = yaml.safe_load(
        (REPO_ROOT / "docker" / "docker-compose.runners.yml").read_text(
            encoding="utf-8"
        )
    )
    assert wired <= set(compose["services"]), (
        "canary override names services absent from the fleet compose file: "
        f"{sorted(wired - set(compose['services']))}"
    )

    # Each wired runner must actually carry the cache index env, pointed at the
    # configured endpoint, with PyPI retained as the fallback so a cache
    # miss/outage degrades rather than failing the job closed.
    for name in sorted(wired):
        env = override["services"][name]["environment"]
        assert env["UV_DEFAULT_INDEX"] == config.pypi_cache.simple_index_url
        assert env["PIP_INDEX_URL"] == config.pypi_cache.simple_index_url
        assert env["PIP_EXTRA_INDEX_URL"] == config.pypi_cache.fallback_index_url
        assert env["UV_INDEX_STRATEGY"] == "unsafe-best-match"


def test_pypi_canary_override_is_layered_by_auto_repair() -> None:
    """OMN-14027 C1: the canary override must be listed in
    docker/compose-overrides.list, and runner-monitor.sh must consume that list.

    The auto-bounce cron recreates runner containers unattended. If it recreates
    from docker-compose.runners.yml alone, canary wiring is stripped with no log
    line and no alert, and the soak keeps reporting on runners that silently
    reverted to direct egress. This gate keeps that path wired.
    """
    overrides_list = REPO_ROOT / "docker" / "compose-overrides.list"
    entries = {
        line.split("#", 1)[0].strip()
        for line in overrides_list.read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip()
    }
    assert "docker-compose.pypi-canary.yml" in entries

    monitor = (REPO_ROOT / "docker" / "runners" / "runner-monitor.sh").read_text(
        encoding="utf-8"
    )
    assert "COMPOSE_OVERRIDES_LIST" in monitor
    # The force-recreate call sites must use the layered args, not COMPOSE_FILE
    # alone — that regression is exactly what strips the canary.
    assert 'docker compose "${COMPOSE_FILE_ARGS[@]}" up -d --force-recreate' in monitor
    assert 'docker compose -f "${COMPOSE_FILE}" up -d --force-recreate' not in monitor


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


def test_runner_fleet_config_git_mirror_is_recorded() -> None:
    """OMN-16053 (OMN-14027 C2): the host-local git-mirror component is recorded
    as fleet source-of-truth and parses under the strict (extra='forbid')
    fleet-config model. active=True reflects deployed reality: the daemon,
    refresh timer, and fail-open pre-seed are live on the runner host.
    """
    config = load_runner_fleet_config(REPO_ROOT / "config" / "runner_fleet.yaml")

    assert config.git_mirror is not None
    assert config.git_mirror.active is True
    # Docker bridge gateway ONLY — git:// is unauthenticated, so the mirrors of
    # private repos must never be bound to a LAN/Tailscale address.
    assert config.git_mirror.bind_address == "172.18.0.1"
    assert config.git_mirror.port == 9418
    assert config.git_mirror.serialized is True
    assert config.git_mirror.refresh_interval_seconds == 120
    assert "onex_change_control" in config.git_mirror.repos
    assert config.git_mirror.kill_switch_env == "OMNI_GIT_MIRROR_DISABLE"


def test_runner_fleet_config_git_mirror_covers_every_ci_repo() -> None:
    """OMN-16056: the C2 git mirror's original 5-repo set (OMN-16053) left
    omniweb, omnimemory, omnibase_compat, and knowledge-base doing full remote
    fetches -- exposed to the same GnuTLS/early-EOF checkout-failure class the
    mirror exists to remove (one of the 21 pre-mirror OMN-16030 failures was
    exactly a sibling fetch of one of these: omnimarket's `Clone
    omnibase_compat` step). This asserts the fix's full coverage, not just
    membership of one repo.

    OMN-20597: knowledge-base-internal was the largest gap of all. Over
    2026-10-04T18:09Z..10-05T11:05Z its unmirrored depth-1 checkout spent
    2,048 of ~2,075 self-hosted checkout minutes (p50 158 s, max 1831 s),
    while the mirrored repos checked out in 0-4 s.
    """
    config = load_runner_fleet_config(REPO_ROOT / "config" / "runner_fleet.yaml")

    assert config.git_mirror is not None
    assert set(config.git_mirror.repos) == {
        "onex_change_control",
        "omnibase_infra",
        "omninode_infra",
        "omnibase_core",
        "omnimarket",
        "omniclaude",
        "omniweb",
        "omnimemory",
        "omnibase_compat",
        "knowledge_base",
        "knowledge-base-internal",
    }


def _refresh_script_mirror_repos(script_text: str) -> list[str]:
    """The MIRROR_REPOS bash array as the refresh script executes it."""
    match = re.search(r"^MIRROR_REPOS=\(\n(.*?)^\)$", script_text, re.M | re.S)
    assert match is not None, "MIRROR_REPOS=( ... ) array not found"
    repos: list[str] = []
    for line in match.group(1).splitlines():
        entry = line.split("#", 1)[0].strip()
        if entry:
            repos.append(entry)
    return repos


def test_runner_fleet_config_git_mirror_repos_match_refresh_script() -> None:
    """OMN-20597: the yaml list is the fleet record and the script's
    MIRROR_REPOS is what the host actually mirrors. Nothing compared them, so
    omninode_infra was added to the script (OMN-19895) and never to the record.
    A repo in one and not the other is either mirrored without a record or
    recorded without a mirror; both read as coverage that is not there.
    """
    config = load_runner_fleet_config(REPO_ROOT / "config" / "runner_fleet.yaml")
    script = REPO_ROOT / "docker" / "runners" / "git-mirror-refresh.sh"

    assert config.git_mirror is not None
    assert list(config.git_mirror.repos) == _refresh_script_mirror_repos(
        script.read_text(encoding="utf-8")
    )


def test_refresh_script_mirror_repos_parser_sees_a_divergence() -> None:
    """Positive control for the parity test: the parser must return the array
    entries and skip comments, or the parity assertion proves nothing."""
    text = (
        "set -euo pipefail\n"
        "MIRROR_REPOS=(\n"
        "    alpha\n"
        "    # a comment line\n"
        "    beta  # trailing comment\n"
        ")\n"
        'for repo in "${MIRROR_REPOS[@]}"; do :; done\n'
    )
    assert _refresh_script_mirror_repos(text) == ["alpha", "beta"]
    assert _refresh_script_mirror_repos(text) != ["alpha", "beta", "gamma"]


def test_runner_fleet_config_tool_cache_durability_is_recorded() -> None:
    """OMN-16053 (OMN-14027 C2): RUNNER_TOOL_CACHE lives in the container
    filesystem (durable=False), so fleet recreates must be bracketed by the
    recorded seed script and a recreate procedure that resolves somewhere real.

    ``seed_script`` is executable code and stays a repo file. ``recreate_procedure``
    is prose, and OMN-16607 moved this repo's prose into the knowledge bases, so it
    is now a ``knowledge-base:``/``knowledge-base-internal:`` reference. This repo
    cannot open the far side, so what it asserts instead is that the value is a
    well-formed reference into a named knowledge base rather than a bare string --
    which is what keeps a typo or an emptied field from passing.
    """
    config = load_runner_fleet_config(REPO_ROOT / "config" / "runner_fleet.yaml")

    assert config.tool_cache is not None
    assert config.tool_cache.durable is False
    assert (REPO_ROOT / config.tool_cache.seed_script).is_file()

    procedure = config.tool_cache.recreate_procedure
    kb, _, kb_path = procedure.partition(":")
    assert config.git_mirror is not None
    knowledge_bases = {r for r in config.git_mirror.repos if r.startswith("knowledge")}
    assert kb in knowledge_bases, procedure
    assert kb_path.endswith(".md"), procedure
    assert not kb_path.startswith("/"), procedure
    assert len(Path(kb_path).parts) >= 2, procedure


def test_runner_fleet_config_dns_cache_is_recorded_but_inert() -> None:
    """OMN-15736: the local DNS-cache endpoint is recorded as fleet
    source-of-truth but stays inert (active=False) until the operator-gated
    rollout repoints canary runners' `dns:` directive. Proves the
    shovel-ready record parses under the strict (extra='forbid') fleet-config
    model and does not activate the cache.
    """
    config = load_runner_fleet_config(REPO_ROOT / "config" / "runner_fleet.yaml")

    assert config.dns_cache is not None
    assert config.dns_cache.active is False
    assert config.dns_cache.host == "192.168.86.201"
    assert config.dns_cache.port == 53
    assert config.dns_cache.upstream_forwarders == (
        "192.168.86.1",
        "1.1.1.1",
        "8.8.8.8",
    )


def test_runner_fleet_config_git_mirror_and_tool_cache_are_optional(
    tmp_path: Path,
) -> None:
    """A fleet config predating the git-transport egress and DNS-cache work
    must still validate — the fields are optional and default to None."""
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

    assert config.git_mirror is None
    assert config.tool_cache is None
    assert config.dns_cache is None


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
    # OMN-18819 moved the logs off /tmp into the fleet state dir; OMN-20805
    # moved the schedule from a crontab line to a user timer whose service
    # appends to the same file.
    assert 'local log_dir="${RUNNER_HOST_DIR}/.onex_state/runner-fleet-logs"' in (
        deploy_script
    )
    assert '"omninode-runner-monitor|*:0/3||runner-monitor.log"' in deploy_script
    assert "StandardOutput=append:${log_dir}/${logfile}" in deploy_script
    assert "StandardError=append:${log_dir}/${logfile}" in deploy_script
    assert "/tmp/runner-monitor.log" not in deploy_script  # noqa: S108
    assert 'local monitor_cron_line="*/3 * * * *' not in deploy_script


def test_deploy_runner_repair_cron_runs_every_ten_minutes() -> None:
    """Runner repair must be a bounded timer, not an ad hoc operator command."""
    deploy_script = (REPO_ROOT / "scripts" / "deploy-runners.sh").read_text(
        encoding="utf-8"
    )

    assert (
        '"omninode-runner-repair|*:0/10|MONITOR_AUTO_BOUNCE=1 '
        'OFFLINE_IDLE_RECREATE_AGE_SECONDS=600|runner-repair.log"'
    ) in deploy_script
    assert "runner-repair-check" in deploy_script  # the legacy line it retires
    # OMN-18819 moved the logs off /tmp into the fleet state dir; OMN-20805
    # moved the schedule to a user timer whose service appends to the same file.
    assert "StandardOutput=append:${log_dir}/${logfile}" in deploy_script
    assert "/tmp/runner-repair.log" not in deploy_script  # noqa: S108
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


def test_runner_compose_reconciled_to_the_spread_60_fleet() -> None:
    """OMN-18411, reaffirmed by OMN-19077's operator ruling 2026-09-24
    (ROLLING_WORK_LEDGER.md:3794): the action fleet is 60 always-on
    steady-state runners. OMN-19895 (operator consent
    ROLLING_WORK_LEDGER.md:13156) spread them across two hosts so no required
    check relies on one machine: 44 in the primary compose file and 16 in
    .202's docker-compose.runners-omnipc2-ci-runner.yml. Each file must match
    its own declared count exactly, so `deploy-runners.sh` can neither
    orphan-remove nor resurrect a runner, and every runner in both mounts the
    OMN-12433 egress healthcheck script and is steady (no burst profile).

    Capped down from 88 (OMN-15978) after a CI burst drove one-minute load to
    100.9 on the 32-core `.201` host: idle runners cost nothing, so the
    registered runner count is what sets the worst-case concurrent-job
    ceiling on a fixed-core box.
    """
    hc_mount = "./runners/healthcheck.sh:/usr/local/bin/healthcheck.sh:ro"
    total = 0
    for filename, prefix, volume_prefix, count in (
        ("docker-compose.runners.yml", "omninode-runner", "runner", 44),
        (
            "docker-compose.runners-omnipc2-ci-runner.yml",
            "omnipc2-ci-runner",
            "omnipc2-ci-runner",
            16,
        ),
    ):
        compose = yaml.safe_load(
            (REPO_ROOT / "docker" / filename).read_text(encoding="utf-8")
        )
        runner_services = {
            name: definition
            for name, definition in compose["services"].items()
            if re.fullmatch(rf"{prefix}-\d+", name)
        }
        assert len(runner_services) == count, f"{filename}: expected {count}"
        indices = sorted(int(name.rsplit("-", 1)[1]) for name in runner_services)
        assert indices == list(range(1, count + 1)), f"{filename}: gaps {indices}"
        for name, definition in runner_services.items():
            assert "profiles" not in definition, f"{name} unexpectedly profile-gated"
            assert hc_mount in definition["volumes"], (
                f"{name} missing healthcheck mount"
            )
            assert definition["volumes"][-1] == (
                f"{volume_prefix}-{name.rsplit('-', 1)[1]}-creds:/home/runner/.runner-creds"
            )
        volume_names = {
            name
            for name in compose["volumes"]
            if re.fullmatch(rf"{volume_prefix}-\d+-creds", name)
        }
        assert len(volume_names) == count
        total += count
    assert total == 60


def test_every_general_pool_runner_runs_in_the_aggregate_slice() -> None:
    """OMN-19077: the general pool shares ONE cgroup whose aggregate limits
    live in docker/runners/systemd/omnirunners.slice. A runner outside it is a
    sibling of the lab lanes in system.slice again, and a CI burst is
    reclaimed from the lanes. The non-pool runners (deploy, verify,
    customer-plane, prod-deploy) are deliberately not in it.
    """
    compose = yaml.safe_load(
        (REPO_ROOT / "docker" / "docker-compose.runners.yml").read_text(
            encoding="utf-8"
        )
    )
    slice_unit = REPO_ROOT / "docker" / "runners" / "systemd" / "omnirunners.slice"
    assert slice_unit.is_file()
    pool = {
        name: definition
        for name, definition in compose["services"].items()
        if re.fullmatch(r"omninode-runner-\d+", name)
    }
    assert pool
    for name, definition in pool.items():
        assert definition.get("cgroup_parent") == slice_unit.name, name
    for name, definition in compose["services"].items():
        if name not in pool:
            assert "cgroup_parent" not in definition, name


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
