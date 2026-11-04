# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Build the foreign-source page before the dev agent's existing recreate.

The source checkout is disposable: neither a canonical clone nor another lane's
working tree is checked out or reset. Git uses the host's existing credential
helper. No credentials are added to a command, image label or build argument.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import yaml

from omnibase_core.container import ModelONEXContainer
from omnibase_core.utils.util_run_coro_sync import run_coro_sync
from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_cost_estimator_build_config import (
    ModelCostEstimatorBuildConfig,
)
from omnibase_infra.models.model_cost_estimator_build_request import (
    ModelCostEstimatorBuildRequest,
)
from omnibase_infra.models.model_cost_estimator_build_result import (
    ModelCostEstimatorBuildResult,
)

logger = logging.getLogger(__name__)
SOURCE_LABEL = "ai.omninode.image.source-repo"
REVISION_LABEL = "org.opencontainers.image.revision"


def _checked(argv: list[str], timeout: int) -> str:
    result = subprocess.run(
        argv,
        timeout=timeout,
        capture_output=True,
        text=True,
        check=False,
        env=scrub_git_location_env(),
    )
    if result.returncode:
        # Tool output can contain credentials; the failed operation is enough.
        raise RuntimeError(f"estimator {argv[0]} {argv[1]} exited {result.returncode}")
    return result.stdout.strip()


def _matches(image: str, repository: str, revision: str, timeout: int) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{json .Config.Labels}}", image],
        timeout=timeout,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        return False
    try:
        labels = json.loads(result.stdout)
    except (ValueError, TypeError):
        return False
    return (
        isinstance(labels, dict)
        and labels.get(SOURCE_LABEL) == repository
        and labels.get(REVISION_LABEL) == revision
    )


class HandlerCostEstimatorBuild:
    """Stateless host effect called only under the deploy agent's lane lock."""

    def __init__(self, container: ModelONEXContainer) -> None:
        self._recipe = container.get_service(ModelCostEstimatorBuildConfig)

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def handle(
        self, request: ModelCostEstimatorBuildRequest
    ) -> ModelCostEstimatorBuildResult:
        if request.lane != "dev" or not request.enabled:
            return ModelCostEstimatorBuildResult(status="not-applicable")
        recipe = self._recipe
        image = request.image or recipe.image
        with tempfile.TemporaryDirectory(prefix="estimator-build-") as temporary:
            source = str(Path(temporary) / "source")
            _checked(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--single-branch",
                    "--branch",
                    recipe.branch,
                    recipe.repository_url,
                    source,
                ],
                recipe.clone_timeout_seconds,
            )
            revision = _checked(
                ["git", "-C", source, "rev-parse", "HEAD"], recipe.read_timeout_seconds
            )
            if not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise RuntimeError("estimator source revision is not a full commit SHA")
            if _matches(
                image, recipe.repository, revision, recipe.read_timeout_seconds
            ):
                return ModelCostEstimatorBuildResult(
                    status="current", image=image, revision=revision
                )
            _checked(
                [
                    "docker",
                    "build",
                    "--label",
                    f"{SOURCE_LABEL}={recipe.repository}",
                    "--label",
                    f"{REVISION_LABEL}={revision}",
                    "--tag",
                    image,
                    source,
                ],
                recipe.build_timeout_seconds,
            )
            if not _matches(
                image, recipe.repository, revision, recipe.read_timeout_seconds
            ):
                raise RuntimeError(
                    "estimator image revision readback differs from the source"
                )
        logger.info("estimator image %s built at web revision %s", image, revision)
        return ModelCostEstimatorBuildResult(
            status="built", image=image, revision=revision
        )


def dev_lane_build_settings(
    execution: ModuleType, lane: str
) -> tuple[ModelCostEstimatorBuildConfig, dict[str, str]] | None:
    """Validate declared settings before any image build or container mutation."""
    if lane != execution.EnumRuntimeLane.DEV:
        return None
    if "cost-estimator-web" in execution.lane_config_for(lane).disabled_services:
        return None
    module_file = execution.__file__
    if not module_file:
        raise RuntimeError("estimator build requires the deploy agent source location")
    contract = (
        Path(module_file).resolve().parents[3]
        / "src/omnibase_infra/nodes/node_setup_local_provision_effect/contract.yaml"
    )
    declaration = yaml.safe_load(contract.read_text(encoding="utf-8"))["config"][
        "cost_estimator_build"
    ]
    build = declaration["build"]
    source_configuration = declaration["source_configuration"]
    gate = declaration["gate"]
    environment = execution._compose_env()
    missing: list[str] = []
    for field in ("repository", "repository_url"):
        name = build.pop(f"{field}_env")
        value = environment.get(name, "").strip()
        if not value:
            missing.append(name)
        build[field] = value
    enabled = environment.get(gate["enabled_env"], str(gate["default_enabled"]))
    if enabled.lower() not in ("true", "false"):
        raise RuntimeError(f"estimator {gate['enabled_env']} must be true or false")
    if (
        enabled.lower() == "true"
        and not environment.get(gate["password_env"], "").strip()
    ):
        missing.append(gate["password_env"])
    if missing:
        raise RuntimeError(
            "estimator missing required settings: "
            + ", ".join(missing)
            + "; source settings belong in the deployment overlay "
            + source_configuration["fragment_schema"]
            + " fragment at "
            + source_configuration["fragment_directory"]
            + "; gate secrets belong in the operator secret store"
        )
    return ModelCostEstimatorBuildConfig.model_validate(build), environment


def build_dev_lane_services(
    execution: ModuleType,
    build_images: Callable[..., None],
    git_sha: str,
    on_phase_update: Callable[..., None],
    *,
    build_source: str,
    lane: str,
    git_ref: str,
) -> None:
    """Validate first, then build dev images before the existing recreate."""
    if lane != execution.EnumRuntimeLane.DEV:
        return
    settings = dev_lane_build_settings(execution, lane)
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
        return
    recipe, environment = settings
    container = ModelONEXContainer()
    try:
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
        handler.handle(
            ModelCostEstimatorBuildRequest(
                lane="dev",
                enabled=True,
                image=environment.get("COST_ESTIMATOR_WEB_IMAGE"),
            )
        )
    finally:
        container.close()
