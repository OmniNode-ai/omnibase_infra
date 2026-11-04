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
from collections.abc import Mapping
from types import ModuleType
from typing import cast

from pydantic import ValidationError

from omnibase_core.container import ModelONEXContainer
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
from omnibase_infra.protocols.protocol_cost_estimator_build_io import (
    ProtocolCostEstimatorBuildIO,
)

logger = logging.getLogger(__name__)
SOURCE_LABEL = "ai.omninode.image.source-repo"
REVISION_LABEL = "org.opencontainers.image.revision"


def validated_image(image: str) -> str:
    image = image.strip()
    if not re.fullmatch(r"[a-z0-9][a-z0-9._/:-]*(?:@sha256:[a-f0-9]{64})?", image):
        raise RuntimeError("estimator invalid COST_ESTIMATOR_WEB_IMAGE")
    return image


def validate_source(repository: str, url: str) -> None:
    from urllib.parse import urlsplit

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", repository):
        raise RuntimeError("estimator invalid COST_ESTIMATOR_WEB_SOURCE_REPOSITORY")
    parsed = urlsplit(url)
    # file URLs support host-local sources and the real-Git delivery fixture.
    # Git's ext transport and option-like values are never accepted.
    if (
        parsed.scheme not in {"https", "ssh", "file"}
        or not parsed.path
        or any(c.isspace() for c in url)
    ):
        raise RuntimeError("estimator invalid COST_ESTIMATOR_WEB_SOURCE_URL")
    if (
        parsed.password
        or (parsed.scheme == "https" and parsed.username)
        or (parsed.scheme != "file" and not parsed.hostname)
    ):
        raise RuntimeError("estimator source URL must use the host credential helper")


class HandlerCostEstimatorBuild:
    """Stateless host effect called only under the deploy agent's lane lock."""

    def __init__(self, container: ModelONEXContainer) -> None:
        self._recipe = container.get_service(ModelCostEstimatorBuildConfig)
        self._io = container.get_service(
            cast("type[ProtocolCostEstimatorBuildIO]", ProtocolCostEstimatorBuildIO)
        )

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def _checked(self, argv: list[str], timeout: int) -> str:
        result = self._io.execute_build_command(argv, timeout)
        if result is None:
            raise RuntimeError(f"estimator {argv[0]} {argv[1]} failed")
        return result

    def _matching_id(
        self, image: str, repository: str, revision: str, timeout: int
    ) -> str | None:
        output = self._io.execute_build_command(
            ["docker", "image", "inspect", "--format", "{{json .}}", "--", image],
            timeout,
            allow_failure=True,
        )
        try:
            metadata = json.loads(output or "null")
            if not isinstance(metadata, dict):
                return None
            config = metadata.get("Config")
            if not isinstance(config, dict):
                return None
            labels = config.get("Labels")
            identity = metadata.get("Id", "")
        except (ValueError, TypeError):
            return None
        if (
            isinstance(labels, dict)
            and labels.get(SOURCE_LABEL) == repository
            and labels.get(REVISION_LABEL) == revision
            and isinstance(identity, str)
            and re.fullmatch(r"sha256:[a-f0-9]{64}", identity)
        ):
            return str(identity)
        return None

    def handle(
        self, request: ModelCostEstimatorBuildRequest
    ) -> ModelCostEstimatorBuildResult:
        if request.lane != "dev" or not request.enabled:
            return ModelCostEstimatorBuildResult(status="not-applicable")
        recipe = self._recipe
        validate_source(recipe.repository, recipe.repository_url)
        image = validated_image((request.image or "").strip() or recipe.image)
        with self._io.workspace() as temporary:
            source = temporary + "/source"
            self._checked(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--single-branch",
                    "--branch",
                    recipe.branch,
                    "--",
                    recipe.repository_url,
                    source,
                ],
                recipe.clone_timeout_seconds,
            )
            revision = self._checked(
                ["git", "-C", source, "rev-parse", "HEAD"], recipe.read_timeout_seconds
            )
            if not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise RuntimeError("estimator source revision is not a full commit SHA")
            if current := self._matching_id(
                image, recipe.repository, revision, recipe.read_timeout_seconds
            ):
                return ModelCostEstimatorBuildResult(
                    status="current", image=current, revision=revision
                )
            identity = self._io.build_image(
                [
                    "docker",
                    "build",
                    "--label",
                    f"{SOURCE_LABEL}={recipe.repository}",
                    "--label",
                    f"{REVISION_LABEL}={revision}",
                    "--iidfile",
                    temporary + "/image-id",
                    "--tag",
                    image,
                    source,
                ],
                recipe.build_timeout_seconds,
                temporary + "/image-id",
            )
            if (
                not re.fullmatch(r"sha256:[a-f0-9]{64}", identity)
                or self._matching_id(
                    identity, recipe.repository, revision, recipe.read_timeout_seconds
                )
                != identity
            ):
                raise RuntimeError(
                    "estimator image revision readback differs from the source"
                )
        logger.info("estimator image %s built at web revision %s", image, revision)
        return ModelCostEstimatorBuildResult(
            status="built", image=identity, revision=revision
        )


def dev_lane_build_settings(
    execution: ModuleType, lane: str, declaration: Mapping[str, object]
) -> tuple[ModelCostEstimatorBuildConfig, dict[str, str]] | None:
    """Validate declared settings before any image build or container mutation."""
    if lane != execution.EnumRuntimeLane.DEV:
        return None
    if "cost-estimator-web" in execution.lane_config_for(lane).disabled_services:
        return None
    build = dict(cast("Mapping[str, object]", declaration["build"]))
    source_configuration = cast(
        "Mapping[str, str]", declaration["source_configuration"]
    )
    gate = cast("Mapping[str, object]", declaration["gate"])
    enabled_env = gate["enabled_env"]
    password_env = gate["password_env"]
    if not isinstance(enabled_env, str) or not isinstance(password_env, str):
        raise RuntimeError("estimator gate environment names must be strings")
    environment = execution._compose_env()
    missing: list[str] = []
    for field in ("repository", "repository_url"):
        name = build.pop(f"{field}_env", None)
        if not isinstance(name, str) or not name:
            raise RuntimeError(f"estimator contract requires {field}_env")
        value = environment.get(name, "").strip()
        if not value:
            missing.append(name)
        build[field] = value
    enabled = environment.get(enabled_env, str(gate["default_enabled"]))
    if enabled.lower() not in ("true", "false"):
        raise RuntimeError(f"estimator {enabled_env} must be true or false")
    if enabled.lower() == "true" and not environment.get(password_env, "").strip():
        missing.append(password_env)
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
    try:
        recipe = ModelCostEstimatorBuildConfig.model_validate(build)
    except ValidationError as error:
        fields = ", ".join(str(item["loc"][0]) for item in error.errors())
        raise RuntimeError(
            f"estimator invalid build contract fields: {fields}"
        ) from None
    validate_source(recipe.repository, recipe.repository_url)
    environment = dict(environment)
    environment["COST_ESTIMATOR_WEB_IMAGE"] = validated_image(
        environment.get("COST_ESTIMATOR_WEB_IMAGE", "").strip() or recipe.image
    )
    return recipe, environment
