# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Host I/O and composition root for the provisioning effect's estimator build."""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from types import ModuleType
from typing import cast

import yaml

from omnibase_core.container import ModelONEXContainer
from omnibase_core.utils.util_run_coro_sync import run_coro_sync
from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.handlers.handler_cost_estimator_build import (
    HandlerCostEstimatorBuild,
)
from omnibase_infra.handlers.handler_cost_estimator_build import (
    dev_lane_build_settings as parse_build_settings,
)
from omnibase_infra.models.model_cost_estimator_build_config import (
    ModelCostEstimatorBuildConfig,
)
from omnibase_infra.models.model_cost_estimator_build_request import (
    ModelCostEstimatorBuildRequest,
)
from omnibase_infra.protocols.protocol_cost_estimator_build_io import (
    ProtocolCostEstimatorBuildIO,
)


class AdapterCostEstimatorBuildIO:
    """Only the host effect spawns processes or reads the filesystem."""

    def run(
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
        self.run(argv, timeout)
        return Path(identity_path).read_text(encoding="utf-8").strip()

    def workspace(self) -> AbstractContextManager[str]:
        return tempfile.TemporaryDirectory(prefix="estimator-build-")


def dev_lane_build_settings(
    execution: ModuleType, lane: str
) -> tuple[ModelCostEstimatorBuildConfig, dict[str, str]] | None:
    if (
        lane != execution.EnumRuntimeLane.DEV
        or "cost-estimator-web" in execution.lane_config_for(lane).disabled_services
    ):
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
    return parse_build_settings(execution, lane, declaration)


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
        run_coro_sync(
            container.service_registry.register_instance(
                cast(
                    "type[ProtocolCostEstimatorBuildIO]", ProtocolCostEstimatorBuildIO
                ),
                AdapterCostEstimatorBuildIO(),
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
