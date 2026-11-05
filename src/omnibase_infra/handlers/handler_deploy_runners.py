# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Converge TMPDIR binds using Compose and SSH, without recreating a runner."""

from __future__ import annotations

import asyncio
import logging
import re
import shlex
from pathlib import Path

import yaml

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_deploy_runners import (
    ModelDeployRunnersRequest,
    ModelDeployRunnersResult,
    ModelRunnerTmpCompose,
    ModelRunnerTmpPolicy,
)

_LOG = logging.getLogger(__name__)
_POLICY = Path(__file__).parents[1] / "nodes/node_deploy_runners/overlays/ci.yaml"


class HandlerDeployRunners:
    """Own the SSH effects and return one typed success or named refusal."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def _remote(self, host: str, command: str, timeout: int) -> str:
        process = await asyncio.create_subprocess_exec(
            "ssh",
            "--",
            host,
            command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout)
        except TimeoutError:
            process.kill()
            await process.wait()
            raise
        if process.returncode:
            raise OSError(f"SSH exited {process.returncode}")
        return stdout.decode()

    async def handle(
        self, request: ModelDeployRunnersRequest
    ) -> ModelDeployRunnersResult:
        if not request.secondary or request.dry_run:
            return ModelDeployRunnersResult(success=True)
        try:
            policy = ModelRunnerTmpPolicy.model_validate(
                yaml.safe_load(_POLICY.read_text())
            )
            # Read only the registration variable names; no credential value
            # leaves the remote Compose stdout -> typed model boundary.
            token_vars = sorted(
                set(
                    re.findall(
                        r"RUNNER_TOKEN:\s*\$\{([A-Z0-9_]+)",
                        Path(request.compose_file).read_text(),
                    )
                )
            )
            exports = " ".join(f"export {name}=preflight;" for name in token_vars)
            compose = shlex.join(
                [
                    "docker",
                    "compose",
                    "-p",
                    request.prefix,
                    "-f",
                    str(
                        Path(request.host_dir)
                        / "docker"
                        / Path(request.compose_file).name
                    ),
                    "config",
                    "--format",
                    "json",
                ]
            )
            preamble = (
                'export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"; '
                'if [ -f "$HOME/.omnibase/runners/docker-cli/config.json" ]; then '
                'export DOCKER_CONFIG="$HOME/.omnibase/runners/docker-cli"; fi; '
            )
            rendered = await self._remote(
                request.host,
                f"set -eu; {preamble}{exports} {compose}",
                policy.timeout_seconds,
            )
            services = ModelRunnerTmpCompose.model_validate_json(rendered).services
            directories: set[str] = set()
            for number in range(1, request.count + 1):
                service = services[f"{request.prefix}-{number}"]
                tmpdir = service.environment.get("TMPDIR")
                for volume in service.volumes:
                    if volume.type == "bind" and volume.target == tmpdir:
                        if (
                            not Path(volume.source).is_absolute()
                            or "\n" in volume.source
                        ):
                            raise ValueError("invalid TMPDIR source")
                        directories.add(volume.source)
        except (OSError, ValueError, KeyError, TimeoutError, yaml.YAMLError):
            return ModelDeployRunnersResult(
                success=False,
                error_message=f"RUNNER-TMP-DISCOVERY host={request.host}: cannot resolve TMPDIR binds.",
            )

        prepared: list[str] = []
        for directory in sorted(directories):
            quoted = shlex.quote(directory)
            install = shlex.join(
                [
                    "sudo",
                    "-n",
                    "install",
                    "-d",
                    "-o",
                    str(policy.uid),
                    "-m",
                    policy.mode,
                    "--",
                    directory,
                ]
            )
            try:
                await self._remote(
                    request.host,
                    f"set -eu; test ! -L {quoted}; {install}",
                    policy.timeout_seconds,
                )
            except (OSError, TimeoutError):
                return ModelDeployRunnersResult(
                    success=False,
                    prepared=tuple(prepared),
                    error_message=f"RUNNER-TMP-PREPARE host={request.host} path={directory}: cannot prepare directory.",
                )
            try:
                actual = await self._remote(
                    request.host,
                    f"LC_ALL=C stat -c '%F:%u:%a' -- {quoted}",
                    policy.timeout_seconds,
                )
                if actual.strip() != f"directory:{policy.uid}:{policy.mode}":
                    raise ValueError("directory readback differs")
            except (OSError, ValueError, TimeoutError):
                return ModelDeployRunnersResult(
                    success=False,
                    prepared=tuple(prepared),
                    error_message=f"RUNNER-TMP-VERIFY host={request.host} path={directory}: owner or mode readback failed.",
                )
            prepared.append(directory)
            _LOG.info(
                "RUNNER-TMP-READY host=%s path=%s uid=%s mode=%s",
                request.host,
                directory,
                policy.uid,
                policy.mode,
            )
        return ModelDeployRunnersResult(success=True, prepared=tuple(prepared))
