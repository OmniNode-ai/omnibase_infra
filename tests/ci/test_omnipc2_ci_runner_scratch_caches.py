# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Every omnipc2-ci-runner service keeps its tool caches on /scratch, not in its layer.

A runner's writable layer grew to 10 to 17 GB of tool caches and filled the .202
root disk. Each service carries its own bind mount per tool, at the tool's default
path, because a service's own `volumes:` list replaces the anchor's (YAML merge
keys do not extend lists).
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest
import yaml

COMPOSE = (
    Path(__file__).resolve().parents[2]
    / "docker"
    / "docker-compose.runners-omnipc2-ci-runner.yml"
)

CACHE_TARGETS = {
    "pip": "/home/runner/.cache/pip",
    "pre-commit": "/home/runner/.cache/pre-commit",
    "pnpm": "/home/runner/.cache/pnpm",
    "ms-playwright": "/home/runner/.cache/ms-playwright",
    "ci-envs": "/home/runner/.cache/omni/ci-envs",
    "npm": "/home/runner/.npm",
}


@pytest.fixture(scope="module")
def services() -> dict[str, dict[str, object]]:
    loaded: dict[str, dict[str, dict[str, object]]] = yaml.safe_load(
        COMPOSE.read_text()
    )
    return loaded["services"]


@pytest.mark.unit
def test_sixteen_runners_declared(services: dict[str, dict[str, object]]) -> None:
    assert sorted(services) == sorted(f"omnipc2-ci-runner-{n}" for n in range(1, 17))


@pytest.mark.unit
@pytest.mark.parametrize("n", range(1, 17))
def test_each_runner_binds_every_cache_to_its_own_scratch_dir(
    services: dict[str, dict[str, object]], n: int
) -> None:
    volumes = services[f"omnipc2-ci-runner-{n}"]["volumes"]
    assert isinstance(volumes, list)
    for tool, target in CACHE_TARGETS.items():
        expected = f"/scratch/cache/runners/{n}/{tool}:{target}"
        assert expected in volumes, expected


@pytest.mark.unit
def test_creds_volume_is_kept(services: dict[str, dict[str, object]]) -> None:
    for n in range(1, 17):
        volumes = services[f"omnipc2-ci-runner-{n}"]["volumes"]
        assert isinstance(volumes, list)
        assert f"omnipc2-ci-runner-{n}-creds:/home/runner/.runner-creds" in volumes


# OMN-20210: the job workspace and the TMPDIR that pytest and uv builds use.
JOB_DIR_TARGETS = {
    "work": "/home/runner/actions-runner/_work",
    "tmp": "/home/runner/tmp",
}


@pytest.mark.unit
@pytest.mark.parametrize("n", range(1, 17))
def test_each_runner_keeps_its_job_workspace_and_tmpdir_on_scratch(
    services: dict[str, dict[str, object]], n: int
) -> None:
    """Job checkouts, venvs and temp files leave the boot NVMe."""
    volumes = services[f"omnipc2-ci-runner-{n}"]["volumes"]
    assert isinstance(volumes, list)
    for name, target in JOB_DIR_TARGETS.items():
        expected = f"/scratch/runners/{n}/{name}:{target}"
        assert expected in volumes, expected
    environment = services[f"omnipc2-ci-runner-{n}"]["environment"]
    assert isinstance(environment, dict)
    assert environment["TMPDIR"] == JOB_DIR_TARGETS["tmp"]


@pytest.mark.unit
def test_no_bind_mount_hides_the_image_tmp(
    services: dict[str, dict[str, object]],
) -> None:
    """The image bakes omni-ci-bin and omni-ci-metadata under the root temp dir."""
    for n in range(1, 17):
        volumes = services[f"omnipc2-ci-runner-{n}"]["volumes"]
        assert isinstance(volumes, list)
        targets = [PurePosixPath(str(v).split(":")[1]) for v in volumes]
        assert PurePosixPath("/") / "tmp" not in targets


# OMN-19998: the uv cache sits inside the work mount so uv can hardlink.
UV_CACHE_DIR = "/home/runner/actions-runner/_work/.uv-cache"


@pytest.mark.unit
def test_uv_cache_shares_the_work_mount_so_installs_hardlink(
    services: dict[str, dict[str, object]],
) -> None:
    """A hardlink cannot cross a mount; copy mode took a warm sync from 32 s to 262 s."""
    assert len(services) == 16
    for service in services.values():
        volumes = service["volumes"]
        environment = service["environment"]
        assert isinstance(volumes, list)
        assert isinstance(environment, dict)
        assert environment["UV_CACHE_DIR"] == UV_CACHE_DIR
        assert "UV_LINK_MODE" not in environment
        targets = [PurePosixPath(str(v).split(":")[1]) for v in volumes]
        mount_of_cache = [
            t for t in targets if PurePosixPath(UV_CACHE_DIR).is_relative_to(t)
        ]
        assert mount_of_cache == [PurePosixPath(JOB_DIR_TARGETS["work"])]
