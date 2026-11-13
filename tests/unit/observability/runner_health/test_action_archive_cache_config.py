# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-24299: the action-archive cache component of the runner fleet config.

The typed config is the fleet's source of truth for the runner's
``ACTIONS_RUNNER_ACTION_ARCHIVE_CACHE`` behaviour: where each runner container
reads the cache and the deploy-host staging root the fill writes. These tests pin the repo
config's recorded values and the models' refusals, in the shape of the
git-mirror record tests in ``test_runner_fleet_config.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from omnibase_infra.models.model_action_archive_cache_config import (
    ModelActionArchiveCacheConfig,
)
from omnibase_infra.observability.runner_health.model_git_mirror_config import (
    ModelGitMirrorConfig,
)
from omnibase_infra.observability.runner_health.model_runner_fleet_config import (
    load_runner_fleet_config,
)
from omnibase_infra.observability.runner_health.model_runner_fleet_host import (
    ModelRunnerFleetHost,
)

REPO_ROOT = Path(__file__).parents[4]


def _cache_kwargs(**overrides: object) -> dict[str, object]:
    """A valid ``ModelActionArchiveCacheConfig`` field set, with overrides."""
    kwargs: dict[str, object] = {
        "active": True,
        "staging_root": "~/.omnibase/runners/action-archive-cache",
        "container_path": "/opt/onex-action-archives",
        "local_owner": "OmniNode-ai",
        "remote_url_template": "https://github.com/{repo}.git",
        "fetch_timeout_seconds": 120,
        "refresh_interval_seconds": 3600,
    }
    kwargs.update(overrides)
    return kwargs


def test_runner_fleet_config_action_archive_cache_is_recorded_and_active() -> None:
    """OMN-24299: the cache component is recorded as fleet source-of-truth and
    parses under the strict (extra='forbid') fleet-config model, with the
    rollout's values."""
    config = load_runner_fleet_config(REPO_ROOT / "config" / "runner_fleet.yaml")

    assert config.action_archive_cache is not None
    assert config.action_archive_cache.active is True
    assert config.action_archive_cache.staging_root == (
        "~/.omnibase/runners/action-archive-cache"
    )
    assert config.action_archive_cache.container_path == "/opt/onex-action-archives"
    assert config.action_archive_cache.local_owner == "OmniNode-ai"
    assert config.action_archive_cache.remote_url_template == (
        "https://github.com/{repo}.git"
    )
    assert config.action_archive_cache.fetch_timeout_seconds == 120
    assert config.action_archive_cache.refresh_interval_seconds == 3600


def test_a_relative_container_path_is_refused() -> None:
    """The container path is what every runner container resolves the cache
    through; a relative one would resolve per working directory."""
    with pytest.raises(ValidationError, match="absolute path"):
        ModelActionArchiveCacheConfig(
            **_cache_kwargs(container_path="opt/onex-action-archives")
        )


def test_an_empty_source_list_is_refused() -> None:
    """The workflows scanned are those of git_mirror.repos; an empty list there
    is refused by the fleet config, so the cache can never be filled from no clone."""
    with pytest.raises(ValidationError, match="at least 1"):
        ModelGitMirrorConfig(
            active=False,
            bind_address="127.0.0.1",
            docker_network="n",
            port=9418,
            mirror_root="/m",
            refresh_interval_seconds=60,
            serialized=True,
            repos=(),
            kill_switch_env="K",
        )


def test_a_remote_url_template_without_the_repo_placeholder_is_refused() -> None:
    """The fill substitutes ``{repo}`` once per third-party ref; a template
    without it would fetch the same URL for every repo."""
    with pytest.raises(ValidationError, match=r"must contain the text \{repo\}"):
        ModelActionArchiveCacheConfig(
            **_cache_kwargs(remote_url_template="https://github.com/example.git")
        )


def test_the_ci_hosts_declare_an_absolute_action_archive_host_root() -> None:
    """OMN-24299: the wired CI host declares where the cache lives on the host
    (its compose file mounts it read-only at container_path), the primary host
    declares none, and a relative root is refused."""
    config = load_runner_fleet_config(REPO_ROOT / "config" / "runner_fleet.yaml")
    roots = sorted(
        host.action_archive_host_root
        for host in config.hosts
        if host.action_archive_host_root is not None
    )
    assert roots == ["/data/scratch/action-archives"]
    assert config.primary_host().action_archive_host_root is None

    with pytest.raises(ValidationError, match="should match pattern"):
        ModelRunnerFleetHost(
            host="h099",
            arch="amd64",
            runner_name_prefix="h099-ci-runner",
            expected_count=1,
            classes=["action"],
            action_archive_host_root="scratch/cache/action-archives",
        )


def test_a_config_without_the_block_still_loads(tmp_path: Path) -> None:
    """A fleet config predating OMN-24299 (no ``action_archive_cache`` block)
    must still validate — the field is optional and defaults to None."""
    text = (REPO_ROOT / "config" / "runner_fleet.yaml").read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    start = next(
        i
        for i, line in enumerate(lines)
        if line.rstrip("\n") == "action_archive_cache:"
    )
    end = start + 1
    while end < len(lines) and (
        lines[end].startswith((" ", "\t")) or lines[end].strip() == ""
    ):
        end += 1
    stripped = tmp_path / "runner_fleet.yaml"
    stripped.write_text("".join(lines[:start] + lines[end:]), encoding="utf-8")

    config = load_runner_fleet_config(stripped)

    assert config.action_archive_cache is None
