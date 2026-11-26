# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Run the launch integration branches on the stability lane (OMN-26715).

Operator RULING 2026-11-26T21:08:59Z made the stability lane the integration
test runtime, so the release-tag-only rule of ``refresh_stability_lane.sh``
(OMN-21611) no longer binds this source. The order is fixed:

1. read the head of ``integration/launch-<date>`` of every repository (an absent
   branch refuses, nothing is built);
2. record the image every service runs now as the rollback target, on disk,
   before anything changes;
3. cut a ``deploy-<UTC stamp>-<sha12>`` tag on each head (RULING
   2026-11-18T15:28:49Z, separation plan S10: the lab deploys from a private
   integration branch and a node-made deploy-* tag is admitted on a declared lab
   path). The tag lives in the host's own clone and is never pushed;
4. build the image from the tags, in the ``stability-integration`` first-party
   audit context that admits a deploy-* tag and nothing else (a build failure
   leaves the lane untouched);
5. recreate the services on it, keeping every volume;
6. read /health until 200 or the window ends; at the window's end the recorded
   images are restored and the result reads ``rolled_back``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import tomllib
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_stability_integration_refresh_request import (
    ModelStabilityIntegrationRefreshRequest,
)
from omnibase_infra.models.model_stability_integration_refresh_result import (
    ModelStabilityIntegrationRefreshResult,
)

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.protocols.protocol_stability_integration_host import (
        ProtocolStabilityIntegrationHost,
    )


_AUDIT_CONTEXT = "stability-integration"
_STAGE_SCRIPT = "scripts/runtime_build/stage_workspace.sh"
_RECREATE_SH = (
    'set -euo pipefail; set -a; source "$1/docker/runtime-policy.env"; '
    '[ -r "$2" ] && source "$2"; set +a; shift 2; exec docker compose "$@"'
)


def _run(argv: Sequence[str], *, cwd: Path | None = None) -> str:
    done = subprocess.run(
        list(argv), cwd=cwd, capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        tail = "\n".join((done.stdout + done.stderr).splitlines()[-12:])
        raise RuntimeError(
            f"{argv[0]} {argv[1] if len(argv) > 1 else ''} exit {done.returncode}: {tail}"
        )
    return done.stdout.strip()


def runtime_version_of(tree: Path) -> str:
    """The ``[project] version`` of the tree's pyproject.toml; a workspace build refuses the placeholder."""
    try:
        data = tomllib.loads((tree / "pyproject.toml").read_text(encoding="utf-8"))
        version = str(data["project"]["version"]).strip()
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeError(f"Could not resolve RUNTIME_VERSION from {tree}") from exc
    if not version:
        raise RuntimeError(f"Empty RUNTIME_VERSION in {tree}")
    return version


class DockerStabilityIntegrationHost:
    """The real host: canonical clones under ``registry_root``, Docker on this machine."""

    def __init__(
        self,
        *,
        registry_root: Path,
        infra_checkout: Path,
        work_dir: Path,
        operator_env_file: Path,
        health_timeout_seconds: float,
    ) -> None:
        self._operator_env = operator_env_file
        self._health_timeout = health_timeout_seconds
        self._registry_root = registry_root
        self._infra = infra_checkout
        self._work = work_dir

    def resolve_head(self, repository: str, branch: str) -> str:
        done = subprocess.run(
            [
                "git",
                "-C",
                str(self._registry_root / repository),
                "rev-parse",
                "--verify",
                f"origin/{branch}^{{commit}}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return done.stdout.strip() if done.returncode == 0 else ""

    def cut_deploy_tags(
        self, heads: Sequence[tuple[str, str]]
    ) -> tuple[tuple[str, str], ...]:
        stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        tags = tuple((repo, f"deploy-{stamp}-{sha[:12]}") for repo, sha in heads)
        for (repo, sha), (_, tag) in zip(heads, tags, strict=True):
            _run(
                ["git", "-C", str(self._registry_root / repo), "tag", tag, sha],
            )
        return tags

    def build_image(
        self, heads: Sequence[tuple[str, str]], tags: Sequence[tuple[str, str]]
    ) -> str:
        by_repo = dict(heads)
        tag_of = dict(tags)
        self._work.mkdir(parents=True, exist_ok=True)
        env = [
            f"OMNI_HOME={self._registry_root}",
            f"DEPLOY_REF={by_repo['omnibase_infra']}",
            "DEPLOY_SIBLING_REFS="
            + " ".join(
                f"{repo}={tag}" for repo, tag in tags if repo != "omnibase_infra"
            ),
            f"DEPLOY_SOURCE_REFS_OUT={self._work / 'refs.json'}",
            f"DEPLOY_SOURCE_WORKTREE_ROOT={self._work / 'trees'}",
        ]
        tree = self._work / "infra-tree"
        shutil.rmtree(tree, ignore_errors=True)
        _run(
            [
                "git",
                "clone",
                "-q",
                "--no-checkout",
                str(self._registry_root / "omnibase_infra"),
                str(tree),
            ]
        )
        _run(
            [
                "git",
                "-C",
                str(tree),
                "checkout",
                "-q",
                "--detach",
                tag_of["omnibase_infra"],
            ]
        )
        _run(["env", *env, "bash", _STAGE_SCRIPT], cwd=tree)
        infra_tag = f"onex-integration-infra:{by_repo['omnibase_infra'][:12]}"
        _run(
            [
                "docker",
                "build",
                "-f",
                "docker/Dockerfile.runtime",
                "-t",
                infra_tag,
                "--build-arg",
                f"VCS_REF={by_repo['omnibase_infra']}",
                "--build-arg",
                f"GIT_SHA={by_repo['omnibase_infra']}",
                "--build-arg",
                f"BUILD_DATE={datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}",
                "--build-arg",
                f"RUNTIME_VERSION={runtime_version_of(tree)}",
                "--build-arg",
                f"RUNTIME_SOURCE_HASH={by_repo['omnibase_infra']}",
                "--build-arg",
                "BUILD_SOURCE=workspace",
                "--build-arg",
                "EXPECTED_BUILD_SOURCE=workspace",
                "--build-arg",
                "OMNI_HOME=/workspace",
                "--build-arg",
                "PROMOTION_CLASS=stability-candidate",
                "--build-arg",
                "NON_MAIN_LINEAGE=true",
                "--build-arg",
                f"FIRST_PARTY_BUILD_CONTEXT={_AUDIT_CONTEXT}",
                ".",
            ],
            cwd=tree,
        )
        market = tree / "workspace" / "sibling-repos" / "omnimarket"
        final_tag = f"onex-integration-runtime:{by_repo['omnibase_infra'][:12]}-{by_repo['omnimarket'][:12]}"
        _run(
            [
                "docker",
                "buildx",
                "build",
                "--load",
                "--platform",
                "linux/amd64",
                "--file",
                str(market / "Dockerfile.runtime"),
                "--build-arg",
                f"INFRA_RUNTIME_IMAGE={infra_tag}",
                "--build-arg",
                f"OMNIMARKET_VCS_REF={by_repo['omnimarket']}",
                "--tag",
                final_tag,
                str(market),
            ]
        )
        return _run(["docker", "image", "inspect", "--format", "{{.Id}}", final_tag])

    def service_image_id(self, compose_project: str, service: str) -> str:
        done = subprocess.run(
            [
                "docker",
                "ps",
                "-q",
                "--filter",
                f"label=com.docker.compose.project={compose_project}",
                "--filter",
                f"label=com.docker.compose.service={service}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        ids = done.stdout.split()
        if len(ids) != 1:
            return ""
        return _run(["docker", "inspect", "--format", "{{.Image}}", ids[0]])

    def recreate(
        self, compose_project: str, services: Sequence[tuple[str, str]]
    ) -> None:
        for service, image_id in services:
            _run(["docker", "tag", image_id, f"{compose_project}-{service}:latest"])
        first = services[0][0]
        container = _run(
            [
                "docker",
                "ps",
                "-q",
                "--filter",
                f"label=com.docker.compose.project={compose_project}",
                "--filter",
                f"label=com.docker.compose.service={first}",
            ]
        )
        labels = json.loads(
            _run(
                ["docker", "inspect", "--format", "{{json .Config.Labels}}", container]
            )
        )
        files = labels["com.docker.compose.project.config_files"].split(",")
        file_args = [arg for path in files for arg in ("-f", path)]
        _run(
            [
                "bash",
                "-c",
                _RECREATE_SH,
                "recreate",
                str(self._infra),
                str(self._operator_env),
                "-p",
                compose_project,
                *file_args,
                "up",
                "-d",
                "--no-deps",
                "--no-build",
                "--force-recreate",
                *[service for service, _ in services],
            ],
            cwd=Path(labels["com.docker.compose.project.working_dir"]),
        )

    def health_status(self, url: str) -> int:
        try:
            return httpx.get(url, timeout=self._health_timeout).status_code
        except httpx.HTTPError:
            return 0


class HandlerStabilityIntegrationRefresh:
    """Build the integration heads, run them on stability, roll back on a red /health."""

    poll_seconds: float = 5.0

    def __init__(self, container: ModelONEXContainer) -> None:
        self._container = container
        self._host: ProtocolStabilityIntegrationHost = container.get_service(
            DockerStabilityIntegrationHost
        )
        self.clock = time.monotonic
        self.sleep = time.sleep

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def handle(
        self, request: ModelStabilityIntegrationRefreshRequest
    ) -> ModelStabilityIntegrationRefreshResult:
        branch = f"integration/launch-{request.integration_date}"
        heads = tuple(
            (repo, self._host.resolve_head(repo, branch))
            for repo in request.repositories
        )
        missing = [repo for repo, sha in heads if not sha]
        if missing:
            return ModelStabilityIntegrationRefreshResult(
                outcome="refused",
                heads=heads,
                detail=f"no {branch} in {', '.join(missing)}",
            )

        target = tuple(
            (svc, self._host.service_image_id(request.compose_project, svc))
            for svc in request.services
        )
        unreadable = [svc for svc, image in target if not image]
        if unreadable:
            return ModelStabilityIntegrationRefreshResult(
                outcome="refused",
                heads=heads,
                detail=f"no rollback target for {', '.join(unreadable)}",
            )
        self._record(request, heads, target)

        try:
            tags = self._host.cut_deploy_tags(heads)
            image_id = self._host.build_image(heads, tags)
        except RuntimeError as exc:
            return ModelStabilityIntegrationRefreshResult(
                outcome="build_failed",
                heads=heads,
                rollback_target=target,
                detail=str(exc),
            )

        self._host.recreate(
            request.compose_project, tuple((svc, image_id) for svc in request.services)
        )
        started = self.clock()
        while True:
            if self._host.health_status(request.health_url) == 200:
                return ModelStabilityIntegrationRefreshResult(
                    outcome="refreshed",
                    heads=heads,
                    deploy_tags=tags,
                    built_image_id=image_id,
                    rollback_target=target,
                    seconds_to_healthy=self.clock() - started,
                )
            if self.clock() - started >= request.health_window_seconds:
                break
            self.sleep(self.poll_seconds)

        self._host.recreate(request.compose_project, target)
        return ModelStabilityIntegrationRefreshResult(
            outcome="rolled_back",
            heads=heads,
            deploy_tags=tags,
            built_image_id=image_id,
            rollback_target=target,
            detail=f"/health not 200 within {request.health_window_seconds}s",
        )

    @staticmethod
    def _record(
        request: ModelStabilityIntegrationRefreshRequest,
        heads: tuple[tuple[str, str], ...],
        target: tuple[tuple[str, str], ...],
    ) -> None:
        directory = Path(request.record_dir)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        (directory / f"rollback-{stamp}.json").write_text(
            json.dumps(
                {
                    "compose_project": request.compose_project,
                    "heads": dict(heads),
                    "rollback_target": dict(target),
                },
                indent=2,
            )
        )
