# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Rollout gate: the runner image must ship the Docker Compose v2 plugin (OMN-14966).

Background. The release-train deploy job runs `refresh_stability_lane.sh` /
`refresh_dev_lane.sh` inside the `omninode-deploy-runner` container. Both the
deploy targeted-recreate and the rollback targeted-recreate call
``docker compose ... up``. The runner image (``docker/runners/Dockerfile``)
installed ``docker-ce-cli`` (bare Docker CLI) but NOT the
``docker-compose-plugin``, so ``docker compose`` resolved to nothing:

    docker: unknown command: docker compose   (exit 125)

Deploy run 29977968728 (2026-07-23) hit exactly this — the rollback recreate
died 125 — making BOTH recreate paths structurally impossible from inside the
runner, independent of the OMN-14958 env/probe defects.

This test is the static gate that the plugin package stays in the image. It is
a package-level assertion on the Dockerfile (the same enforcement style as
``test_runner_image_node24_floor.py``); the image build-smoke and pre-commit
run it before the runner image can ship. Adding the package is identity-neutral
— the Dockerfile is not part of the ``runner-image.lock.json`` bound identity.
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


def test_dockerfile_installs_docker_compose_plugin() -> None:
    """`docker compose` v2 must be installed so compose-based recreate works.

    Without ``docker-compose-plugin`` every ``docker compose`` invocation inside
    the runner fails with ``docker: unknown command: docker compose`` (exit 125)
    — the OMN-14966 deploy/rollback recreate blocker.
    """
    source = _dockerfile_source()
    assert "docker-compose-plugin" in source, (
        "runner Dockerfile must install docker-compose-plugin so `docker compose` "
        "v2 resolves inside the runner; without it the release-train deploy AND "
        "rollback targeted-recreate paths die with exit 125 (OMN-14966)"
    )


def test_compose_plugin_installed_from_the_docker_apt_repo() -> None:
    """The plugin must be apt-installed in the same RUN that adds the Docker repo.

    ``docker-compose-plugin`` is published by the Docker apt repository that the
    Dockerfile already configures for ``docker-ce-cli``. Installing it in that
    same RUN block guarantees the repo + key are present and avoids a second
    ``apt-get update``.
    """
    blocks = _run_blocks(_dockerfile_source())
    docker_repo_block = next(
        (b for b in blocks if "download.docker.com/linux/ubuntu" in b), None
    )
    assert docker_repo_block is not None, (
        "expected a RUN block that configures the Docker apt repo"
    )
    assert "docker-ce-cli docker-compose-plugin" in docker_repo_block or (
        "docker-ce-cli" in docker_repo_block
        and "docker-compose-plugin" in docker_repo_block
    ), (
        "docker-compose-plugin must be apt-installed alongside docker-ce-cli in "
        "the Docker-apt-repo RUN block (OMN-14966)"
    )
