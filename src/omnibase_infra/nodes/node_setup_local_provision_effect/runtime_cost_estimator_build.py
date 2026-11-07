# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Host I/O and composition root for the provisioning effect's estimator build."""

from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from subprocess import TimeoutExpired
from types import ModuleType

import yaml

from omnibase_core.container import ModelONEXContainer
from omnibase_core.utils.util_run_coro_sync import run_coro_sync
from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.handlers.handler_cost_estimator_build import (
    REVISION_LABEL,
    SOURCE_LABEL,
    HandlerCostEstimatorBuild,
    validate_source,
)
from omnibase_infra.handlers.handler_cost_estimator_build import (
    dev_lane_build_settings as parse_build_settings,
)
from omnibase_infra.models.model_cost_estimator_build_config import (
    ModelCostEstimatorBuildConfig,
    ModelCostEstimatorWebFollow,
)
from omnibase_infra.models.model_cost_estimator_build_request import (
    ModelCostEstimatorBuildRequest,
)
from omnibase_infra.protocols.protocol_cost_estimator_build_io import (
    ProtocolCostEstimatorBuildIO,
)

logger = logging.getLogger(__name__)


class CostEstimatorBuildIO:
    """Only the host effect spawns processes or reads the filesystem."""

    def execute_build_command(
        self, argv: list[str], timeout: int, *, allow_failure: bool = False
    ) -> str | None:
        result = subprocess.run(
            argv,
            timeout=timeout,
            capture_output=True,
            text=True,
            check=False,
            env=scrub_git_location_env(),
        )
        if result.returncode:
            if allow_failure:
                return None
            raise RuntimeError(
                f"estimator {argv[0]} {argv[1]} exited {result.returncode}"
            )
        return result.stdout.strip()

    def build_image(self, argv: list[str], timeout: int, identity_path: str) -> str:
        self.execute_build_command(argv, timeout)
        return Path(identity_path).read_text(encoding="utf-8").strip()

    def workspace(self) -> AbstractContextManager[str]:
        return tempfile.TemporaryDirectory(prefix="estimator-build-")


def _declaration(execution: ModuleType) -> dict[str, object]:
    module_file = execution.__file__
    if not module_file:
        raise RuntimeError("estimator build requires the deploy agent source location")
    contract = (
        Path(module_file).resolve().parents[3]
        / "src/omnibase_infra/nodes/node_setup_local_provision_effect/contract.yaml"
    )
    declaration: dict[str, object] = yaml.safe_load(
        contract.read_text(encoding="utf-8")
    )["config"]["cost_estimator_build"]
    return declaration


def dev_lane_build_settings(
    execution: ModuleType, lane: str
) -> tuple[ModelCostEstimatorBuildConfig, dict[str, str]] | None:
    if (
        lane != execution.EnumRuntimeLane.DEV
        or "cost-estimator-web" in execution.lane_config_for(lane).disabled_services
    ):
        return None
    return parse_build_settings(execution, lane, _declaration(execution))


class CostEstimatorWebFollow(ModelCostEstimatorWebFollow):
    """The declared web source, with the two reads its idle follower decides on."""

    def running_ref(self) -> str | None:
        """The web commit the running estimator image was built from, or ``None``.

        ``None`` for an absent container, labels that are not JSON, an image built
        from another source repository, or a revision that is not a full commit
        sha: an unread revision proves nothing and the decision refuses.
        """
        output = _read(
            [
                "docker",
                "inspect",
                "--format",
                "{{json .Config.Labels}}",
                self.container,
            ],
            self.read_timeout_seconds,
        )
        try:
            labels = json.loads(output or "null")
        except ValueError:
            return None
        if (
            not isinstance(labels, dict)
            or labels.get(self.source_label) != self.repository
        ):
            return None
        return _sha(labels.get(self.revision_label))

    def head_ref(self) -> str | None:
        """The head of the declared branch on the declared url, or ``None``."""
        output = _read(
            [
                "git",
                "ls-remote",
                "--",
                self.repository_url,
                f"refs/heads/{self.branch}",
            ],
            self.read_timeout_seconds,
        )
        return _sha(next(iter((output or "").split()), None))


def _read(argv: list[str], timeout: int) -> str | None:
    try:
        result = subprocess.run(
            argv,
            timeout=timeout,
            capture_output=True,
            text=True,
            check=False,
            env=scrub_git_location_env(),
        )
    except (OSError, TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def _sha(value: object) -> str | None:
    if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value):
        return value
    return None


def dev_lane_web_follow(
    execution: ModuleType, lane: str
) -> CostEstimatorWebFollow | None:
    """The web source the idle follower tracks, from the declaration (OMN-22808).

    ``None`` when the lane is not dev, the estimator service is disabled, the
    declaration has no ``follow`` block, the lane fragment names no source yet
    (the build itself refuses that, naming the missing settings, before it
    touches anything), or the declaration cannot be read: the follower then
    follows nothing and says why, and the agent's other followed repositories
    are unaffected. Nothing here reads the gate secret.
    """
    if (
        lane != execution.EnumRuntimeLane.DEV
        or "cost-estimator-web" in execution.lane_config_for(lane).disabled_services
    ):
        return None
    try:
        declaration = _declaration(execution)
        follow = declaration.get("follow")
        build = declaration["build"]
        if not isinstance(follow, dict) or not isinstance(build, dict):
            return None
        environment = execution._compose_env()
        values = {
            field: environment.get(str(build[f"{field}_env"]), "").strip()
            for field in ("repository", "repository_url")
        }
        if not all(values.values()):
            return None
        validate_source(values["repository"], values["repository_url"])
        return CostEstimatorWebFollow(
            **values,
            branch=str(build["branch"]),
            container=str(follow["container"]),
            source_label=SOURCE_LABEL,
            revision_label=REVISION_LABEL,
            read_timeout_seconds=int(build["read_timeout_seconds"]),
        )
    except (OSError, KeyError, ValueError, TypeError, RuntimeError, yaml.YAMLError):
        logger.warning("estimator web follow declaration unreadable", exc_info=True)
        return None


def build_dev_lane_services(
    execution: ModuleType,
    build_images: Callable[..., None],
    git_sha: str,
    on_phase_update: Callable[..., None],
    *,
    build_source: str,
    lane: str,
    git_ref: str,
    read_settings: Callable[
        [], tuple[ModelCostEstimatorBuildConfig, dict[str, str]] | None
    ]
    | None = None,
) -> dict[str, str]:
    """Validate first, then build dev images before the existing recreate."""
    if lane != execution.EnumRuntimeLane.DEV:
        return {}
    settings = (
        read_settings()
        if read_settings is not None
        else dev_lane_build_settings(execution, lane)
    )
    if execution.DEV_LANE_BUILD_SERVICES:
        build_images(
            execution.Scope.RUNTIME,
            git_sha,
            on_phase_update,
            build_source=build_source,
            runtime_lane=lane,
            git_ref=git_ref,
            compose_files=execution.lane_config_for(lane).compose_files,
            services=execution.DEV_LANE_BUILD_SERVICES,
            stage_workspace=False,
        )
    if settings is None:
        return {}
    recipe, environment = settings
    container = ModelONEXContainer()
    try:
        io_key: type[object] = ProtocolCostEstimatorBuildIO
        run_coro_sync(
            container.service_registry.register_instance(
                io_key,
                CostEstimatorBuildIO(),
            )
        )
        run_coro_sync(
            container.service_registry.register_instance(
                ModelCostEstimatorBuildConfig,
                recipe,
            )
        )
        run_coro_sync(
            container.service_registry.register_instance(
                HandlerCostEstimatorBuild,
                HandlerCostEstimatorBuild(container),
            )
        )
        handler = container.get_service(HandlerCostEstimatorBuild)
        result = handler.handle(
            ModelCostEstimatorBuildRequest(
                lane="dev",
                enabled=True,
                image=environment.get("COST_ESTIMATOR_WEB_IMAGE"),
            )
        )
        if result.image is None:
            raise RuntimeError("estimator build returned no image identity")
        return {
            "COST_ESTIMATOR_WEB_IMAGE": result.image,
            "FEATURE_ESTIMATOR_GATE_ENABLED": environment.get(
                "FEATURE_ESTIMATOR_GATE_ENABLED", "true"
            ),
            "ESTIMATOR_GATE_PASSWORD": environment.get("ESTIMATOR_GATE_PASSWORD", ""),
        }
    finally:
        container.close()
