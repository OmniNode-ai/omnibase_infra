# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Build the onex-api image the dev lane's pin delivery advances to (OMN-23521).

The deploy agent's settle delivers the onex-api pin on every passing dev roll.
The image it pins used to be built as one step of the retired cluster apply, so
an instance could not get a fresh onex-api without that apply. This handler is
that one build, on its own: one ``docker build`` on the host daemon, from the
archived source tree, tagged ``<revision8>-<stamp>`` and labelled with the
revision and source repository the delivery restricts its pin to.

FAILS CLOSED, with the reason said. Inputs that belong to another job, a missing
runtime image or a source revision that is not a full commit are refused before
any build runs; a build that exits non-zero or cannot start is ``BUILD_FAILED``
with its output tail. The result never names an image it did not build, so the
delivery never pins another lineage.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_onex_api_image_build_request import (
    ModelOnexApiImageBuildRequest,
)
from omnibase_infra.models.model_onex_api_image_build_result import (
    ModelOnexApiImageBuildResult,
)

#: ``(argv, timeout=..., cwd=..., check=False) -> CompletedProcess``.
RunCommand = Callable[..., "subprocess.CompletedProcess[str]"]


class HandlerOnexApiImageBuild:
    """Build the onex-api image from the archived source tree."""

    def __init__(self, *, run: RunCommand) -> None:
        self._run = run

    @property
    def handler_type(self) -> EnumHandlerType:
        """Infrastructure handler over the host's Docker daemon."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Side-effecting: builds and tags a docker image."""
        return EnumHandlerTypeCategory.EFFECT

    def handle(
        self, request: ModelOnexApiImageBuildRequest
    ) -> ModelOnexApiImageBuildResult:
        # The archived tree's commit is known before any refusal below, and the
        # delivery restricts its pin to it; only a non-commit is reported as none.
        known = request.source_revision if len(request.source_revision) == 40 else None
        if request.capture_sha != request.sha:
            return ModelOnexApiImageBuildResult(
                built=False,
                image="",
                source_revision=known,
                reason=(
                    f"REFUSED_CAPTURE_SHA_MISMATCH: "
                    f"{request.capture_sha[:12]} != {request.sha[:12]}"
                ),
            )
        if request.capture_error is not None or not request.runtime_image:
            return ModelOnexApiImageBuildResult(
                built=False,
                image="",
                source_revision=known,
                reason=(
                    f"REFUSED_NO_RUNTIME_IMAGE: "
                    f"{request.capture_error or 'no runtime image was captured'}"
                ),
            )
        if len(request.source_revision) != 40:
            return ModelOnexApiImageBuildResult(
                built=False,
                image="",
                source_revision=None,
                reason=(
                    f"REFUSED_NO_SOURCE_REVISION: "
                    f"{request.source_revision!r} is not a 40-char commit"
                ),
            )
        target = f"{request.image_name}:{request.source_revision[:8]}-{request.stamp}"
        argv = [
            "docker",
            "build",
            "-f",
            request.dockerfile,
            "-t",
            target,
            "--label",
            f"{request.revision_label}={request.source_revision}",
            "--label",
            f"{request.source_label}={request.source_repo}",
            "--build-arg",
            f"{request.runtime_image_build_arg}={request.runtime_image}",
            request.context,
        ]
        try:
            completed = self._run(
                argv,
                timeout=request.timeout_seconds,
                cwd=request.source_tree,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return self._failed(request, f"{type(exc).__name__}: {exc}")
        if completed.returncode != 0:
            tail = str(completed.stderr or completed.stdout or "").strip()[-240:]
            return self._failed(
                request, f"docker build exited {completed.returncode}: {tail}"
            )
        return ModelOnexApiImageBuildResult(
            built=True,
            image=target,
            source_revision=request.source_revision,
            reason=f"BUILT: {target}",
        )

    @staticmethod
    def _failed(
        request: ModelOnexApiImageBuildRequest, detail: str
    ) -> ModelOnexApiImageBuildResult:
        """A build that ran and failed: the lineage is known, the image is not."""
        return ModelOnexApiImageBuildResult(
            built=False,
            image="",
            source_revision=request.source_revision,
            reason=f"BUILD_FAILED: {detail[:240]}",
        )


def build_image_build_handler(run: RunCommand) -> HandlerOnexApiImageBuild:
    """Wire the handler to the caller's command runner (the agent has no container)."""
    return HandlerOnexApiImageBuild(run=run)


def build_onex_api_image(
    run: RunCommand, request: ModelOnexApiImageBuildRequest
) -> ModelOnexApiImageBuildResult:
    """The deploy agent's one call, made on its settle before the pin delivery."""
    return build_image_build_handler(run).handle(request)
