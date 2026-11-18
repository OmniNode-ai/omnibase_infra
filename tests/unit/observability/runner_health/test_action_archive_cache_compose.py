# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# OMN-24299 -- runner action-archive cache: compose declarations + deploy-script wiring.
#
# The runner reads ACTIONS_RUNNER_ACTION_ARCHIVE_CACHE and, when a job step uses
# `owner/repo@<sha>`, copies <cache>/<owner>_<repo>/<sha>.tar.gz instead of
# downloading from GitHub (fail-open when absent). These tests pin the compose
# declarations (env + read-only volume on every runner service of each
# pool that declares a cache root) and the deploy-runners.sh wiring (sync_action_archive_cache defined and
# called from secondary_deploy and relabel_roll, DRY_RUN, rsync flags).
#
# The host-side root comes from each pool's host row `action_archive_host_root`
# in config/runner_fleet.yaml, and the pools checked are DISCOVERED from the rows
# that declare one, so a pool added to the declaration without its compose
# wiring fails here instead of being skipped.

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]

# Positive control: an unrelated runners compose file that must NOT carry the
# action-archive cache env or volume, so the check can actually fail.
UNRELATED_COMPOSE = "docker/docker-compose.runners-hp053-ci-mq-runner.yml"

DEPLOY_SCRIPT = "scripts/deploy-runners.sh"

ENV_VAR = "ACTIONS_RUNNER_ACTION_ARCHIVE_CACHE"
CONTAINER_PATH = "/opt/onex-action-archives"


def _declared_pools(fleet_config: dict) -> dict[str, str]:
    """compose file -> action pool prefix, for every host row that declares a cache root."""
    pools: dict[str, str] = {}
    for row in fleet_config["hosts"]:
        if "action_archive_host_root" not in row:
            continue
        candidates = [
            (row["runner_name_prefix"], row.get("classes") or []),
            *(
                (p["runner_name_prefix"], p.get("classes") or [])
                for p in row.get("pools") or []
            ),
        ]
        for prefix, classes in candidates:
            if any(c in ("action", "action-multiarch") for c in classes):
                pools[f"docker/docker-compose.runners-{prefix}.yml"] = prefix
    return pools


@pytest.fixture(scope="module")
def fleet_config() -> dict:
    path = REPO_ROOT / "config" / "runner_fleet.yaml"
    with path.open() as fh:
        return yaml.safe_load(fh)


def _host_rows(fleet_config: dict) -> dict:
    rows = fleet_config.get("hosts")
    assert isinstance(rows, list), (
        "config/runner_fleet.yaml must declare a `hosts:` list"
    )
    by_name = {}
    for row in rows:
        name = row.get("host")
        if name is not None:
            by_name[name] = row
    return by_name


def _host_root_for_prefix(prefix: str, fleet_config: dict) -> str:
    """Return the action_archive_host_root of the host row that owns `prefix`.

    The key must be present: a missing key is a real failure, not a skip.
    """
    rows = _host_rows(fleet_config)
    for name, row in rows.items():
        pools = row.get("pools") or []
        for pool in pools:
            if pool.get("runner_name_prefix") == prefix:
                assert "action_archive_host_root" in row, (
                    f"host {name!r} (pool {prefix!r}) must declare action_archive_host_root "
                    f"in config/runner_fleet.yaml (OMN-24299)"
                )
                return row["action_archive_host_root"]
        if row.get("runner_name_prefix") == prefix:
            assert "action_archive_host_root" in row, (
                f"host {name!r} (prefix {prefix!r}) must declare action_archive_host_root "
                f"in config/runner_fleet.yaml (OMN-24299)"
            )
            return row["action_archive_host_root"]
    pytest.fail(
        f"no host row in config/runner_fleet.yaml declares runner_name_prefix {prefix!r}"
    )


def _services(compose: dict) -> dict:
    services = compose.get("services")
    assert isinstance(services, dict) and services, (
        "compose file must declare `services:`"
    )
    return services


def _env_map(service: dict) -> dict:
    env = service.get("environment")
    if env is None:
        return {}
    if isinstance(env, dict):
        return env
    if isinstance(env, list):
        out = {}
        for item in env:
            if isinstance(item, str) and "=" in item:
                k, v = item.split("=", 1)
                out[k] = v
        return out
    return {}


def _volume_strings(service: dict) -> list:
    vols = service.get("volumes")
    if not vols:
        return []
    out = []
    for item in vols:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            src = item.get("source")
            tgt = item.get("target")
            if src is not None and tgt is not None:
                mode = ":ro" if item.get("read_only") else ""
                out.append(f"{src}:{tgt}{mode}")
    return out


def _cache_volume_host_side(volumes: list) -> str | None:
    for vol in volumes:
        if vol.endswith(f":{CONTAINER_PATH}:ro"):
            return vol[: -(len(CONTAINER_PATH) + 4)]
    return None


def _assert_pool_compose(relpath: str, prefix: str, fleet_config: dict) -> None:
    path = REPO_ROOT / relpath
    with path.open() as fh:
        compose = yaml.safe_load(fh)
    services = _services(compose)
    expected_root = _host_root_for_prefix(prefix, fleet_config)
    checked = 0
    for name, service in services.items():
        if not name.startswith(prefix):
            continue
        checked += 1
        env = _env_map(service)
        assert env.get(ENV_VAR) == CONTAINER_PATH, (
            f"{relpath} service {name!r} must set {ENV_VAR} == {CONTAINER_PATH!r} (got {env.get(ENV_VAR)!r})"
        )
        host_side = _cache_volume_host_side(_volume_strings(service))
        assert host_side is not None, (
            f"{relpath} service {name!r} must mount a volume ending :{CONTAINER_PATH}:ro"
        )
        assert host_side == expected_root, (
            f"{relpath} service {name!r} cache volume host side {host_side!r} != "
            f"{expected_root!r} (action_archive_host_root of pool {prefix!r})"
        )
    assert checked > 0, (
        f"{relpath} declares no service whose name starts with {prefix!r}"
    )


def test_ci_pool_compose_declarations(fleet_config: dict) -> None:
    pools = _declared_pools(fleet_config)
    assert len(pools) == 1, (
        f"expected the one wired CI pool to declare a cache root, got {sorted(pools)}"
    )
    for relpath, prefix in pools.items():
        _assert_pool_compose(relpath, prefix, fleet_config)


def test_unrelated_compose_has_no_cache(fleet_config: dict) -> None:
    path = REPO_ROOT / UNRELATED_COMPOSE
    with path.open() as fh:
        compose = yaml.safe_load(fh)
    services = _services(compose)
    for name, service in services.items():
        assert _env_map(service).get(ENV_VAR) is None, (
            f"{UNRELATED_COMPOSE} service {name!r} must NOT set {ENV_VAR}"
        )
        assert _cache_volume_host_side(_volume_strings(service)) is None, (
            f"{UNRELATED_COMPOSE} service {name!r} must NOT mount the action-archive cache volume"
        )


def test_deploy_script_hands_the_fleet_declaration_and_clone_root_to_the_prepare_effect() -> (
    None
):
    """The cache fill and sync live in the node (HandlerDeployRunners), not in the
    script: the one dispatch both stand-up paths already make carries the two
    paths the effect needs, and an unset OMNI_HOME is dropped, not sent empty."""
    text = (REPO_ROOT / DEPLOY_SCRIPT).read_text()
    match = re.search(r"^ensure_runner_tmp_dirs\(\) \{\n(.*?)\n\}", text, re.M | re.S)
    assert match, "deploy-runners.sh must define ensure_runner_tmp_dirs()"
    dispatch = match.group(1)
    assert '--arg fleet_config "${RUNNER_FLEET_CONFIG}"' in dispatch
    assert '--arg clone_root "${OMNI_HOME:-}"' in dispatch
    assert 'select(.value != "")' in dispatch
    assert "node_deploy_runners" in dispatch
    assert "sync_action_archive_cache" not in text
    for caller in ("secondary_deploy", "relabel_roll"):
        body = text[text.index(f"{caller}() {{") :]
        assert "ensure_runner_tmp_dirs" in body.split("\n}\n", 1)[0], caller
