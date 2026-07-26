# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Rollout gate: the runner image must ship the Docker Buildx plugin (OMN-15141).

Background. `docker/Dockerfile.runtime` uses BuildKit-only `RUN --mount=type=cache,...`
syntax (apt cache + uv cache mounts). The release-train deploy job builds that
Dockerfile via `docker compose ... build` inside the `omninode-deploy-runner`
container. The runner image (``docker/runners/Dockerfile``) installed
``docker-ce-cli`` + ``docker-compose-plugin`` (OMN-14966) but NOT the
``docker-buildx-plugin``, so `docker compose build` had no BuildKit backend to
route through and fell back to the legacy builder, which cannot execute
``--mount`` syntax:

    the --mount option requires BuildKit. Refer to
    https://docs.docker.com/go/buildkit/ to learn how to build images with
    BuildKit enabled
    [deploy] ERROR: Image build failed.

Deploy run 30178195370 (2026-07-25) hit exactly this at the stability
release-train deploy hop's iteration N+4 — the first hop to reach the actual
image-build step, immediately after the OMN-14966 (`docker-compose-plugin`)
and OMN-15137 (sibling-clone provisioning) fixes cleared the earlier blockers.
The build failure triggered the health-gate, which correctly rolled the lane
back to HEALTHY -- no runtime regression, but the deploy hop could not
progress.

This test is the static gate that the buildx plugin package stays in the
image. It is a package-level assertion on the Dockerfile (the same
enforcement style as ``test_runner_image_compose_plugin.py`` /
``test_runner_image_node24_floor.py``); the image build-smoke and pre-commit
run it before the runner image can ship. Adding the package is
identity-neutral — the Dockerfile is not part of the
``runner-image.lock.json`` bound identity.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_DOCKERFILE = REPO_ROOT / "docker" / "runners" / "Dockerfile"


def _dockerfile_source() -> str:
    return RUNNER_DOCKERFILE.read_text(encoding="utf-8")


def _run_blocks(source: str) -> list[str]:
    """Return each top-level RUN block (through its trailing continuations)."""
    return re.findall(
        r"(?ms)^RUN .*?(?=^RUN |^COPY |^USER |^WORKDIR |^ENTRYPOINT |\Z)", source
    )


def test_dockerfile_installs_docker_buildx_plugin() -> None:
    """`docker buildx` must be installed so `--mount=type=cache` builds work.

    Without ``docker-buildx-plugin``, ``docker compose build`` / ``docker
    build`` have no BuildKit backend inside the runner and any Dockerfile using
    ``RUN --mount=type=cache,...`` (e.g. ``docker/Dockerfile.runtime``) fails
    with "the --mount option requires BuildKit" — the OMN-15141 stability
    deploy-hop blocker (iteration N+4, deploy run 30178195370).
    """
    source = _dockerfile_source()
    assert "docker-buildx-plugin" in source, (
        "runner Dockerfile must install docker-buildx-plugin so BuildKit-only "
        "syntax (RUN --mount=type=cache) resolves inside the runner; without "
        "it any docker/Dockerfile.runtime build dies with 'the --mount option "
        "requires BuildKit' (OMN-15141)"
    )


def test_buildx_plugin_installed_from_the_docker_apt_repo() -> None:
    """The plugin must be apt-installed in the same RUN that adds the Docker repo.

    ``docker-buildx-plugin`` is published by the Docker apt repository that the
    Dockerfile already configures for ``docker-ce-cli`` / ``docker-compose-plugin``.
    Installing it in that same RUN block guarantees the repo + key are present
    and avoids a second ``apt-get update``.
    """
    blocks = _run_blocks(_dockerfile_source())
    docker_repo_block = next(
        (b for b in blocks if "download.docker.com/linux/ubuntu" in b), None
    )
    assert docker_repo_block is not None, (
        "expected a RUN block that configures the Docker apt repo"
    )
    assert (
        "docker-ce-cli" in docker_repo_block
        and "docker-compose-plugin" in docker_repo_block
        and "docker-buildx-plugin" in docker_repo_block
    ), (
        "docker-buildx-plugin must be apt-installed alongside docker-ce-cli and "
        "docker-compose-plugin in the Docker-apt-repo RUN block (OMN-15141)"
    )
