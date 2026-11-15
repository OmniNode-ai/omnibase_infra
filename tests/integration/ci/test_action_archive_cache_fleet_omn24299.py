# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-24299: the shipped fleet file, typed model and runner compose files agree.

Loads the real ``config/runner_fleet.yaml`` through the typed fleet model and
reads the real compose file of every pool whose host row declares an action
archive root, so a pool that declares a cache without the runner env and
read-only mount fails here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.models.model_action_archive_cache_config import (
    ModelActionArchiveCacheConfig,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_VAR = "ACTIONS_RUNNER_ACTION_ARCHIVE_CACHE"
UNRELATED = "docker/docker-compose.runners-hp053-ci-mq-runner.yml"


def _fleet() -> dict:
    return yaml.safe_load((REPO_ROOT / "config" / "runner_fleet.yaml").read_text())


def _compose(rel: str) -> dict:
    return yaml.safe_load((REPO_ROOT / rel).read_text())


def _cache_env_and_mount_services(compose: dict, container_path: str) -> list[str]:
    ok = []
    for name, svc in compose["services"].items():
        mounts = [str(v) for v in svc.get("volumes", [])]
        env = svc.get("environment", {})
        if env.get(ENV_VAR) == container_path and any(
            m.endswith(f":{container_path}:ro") for m in mounts
        ):
            ok.append(name)
    return ok


def test_declared_cache_is_typed_and_mounted_on_every_h053_action_runner() -> None:
    fleet = _fleet()
    cache = ModelActionArchiveCacheConfig.model_validate(fleet["action_archive_cache"])
    assert cache.active
    rows = [r for r in fleet["hosts"] if "action_archive_host_root" in r]
    assert rows, "no host row declares action_archive_host_root"
    for row in rows:
        compose = _compose(
            f"docker/docker-compose.runners-{row['runner_name_prefix']}.yml"
        )
        mounted = _cache_env_and_mount_services(compose, cache.container_path)
        assert len(mounted) == len(compose["services"]) == row["expected_count"]


def test_a_pool_without_a_declared_root_carries_no_cache() -> None:
    compose = _compose(UNRELATED)
    assert _cache_env_and_mount_services(compose, "/opt/onex-action-archives") == []
