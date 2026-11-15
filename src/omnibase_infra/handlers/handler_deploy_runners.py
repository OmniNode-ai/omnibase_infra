# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Converge TMPDIR binds using Compose and SSH, without recreating a runner."""

from __future__ import annotations

import asyncio
import logging
import re
import shlex
import sys
from pathlib import Path

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.enums.enum_compose_mount_type import EnumComposeMountType
from omnibase_infra.models.model_action_archive_cache_fill_result import (
    ModelActionArchiveCacheFillResult,
)
from omnibase_infra.models.model_deploy_runners_request import ModelDeployRunnersRequest
from omnibase_infra.models.model_deploy_runners_result import ModelDeployRunnersResult
from omnibase_infra.models.model_runner_tmp_compose import ModelRunnerTmpCompose
from omnibase_infra.models.model_runner_tmp_policy import ModelRunnerTmpPolicy
from omnibase_infra.observability.runner_health.model_runner_fleet_config import (
    load_runner_fleet_config,
)

_LOG = logging.getLogger(__name__)
# The node's runtime entry for the cache fill, run as a file so this handler names no
# node module (the node-boundary gate keeps handlers from importing their node).
_FILL_ENTRY = (
    Path(__file__).resolve().parents[1]
    / "nodes"
    / "node_deploy_runners"
    / "runtime_action_archive_cache_fill.py"
)


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

    async def _rsync(self, source: str, host: str, target: str, timeout: int) -> None:
        process = await asyncio.create_subprocess_exec(
            "rsync",
            "-a",
            "--ignore-existing",
            "--exclude=.sources",
            f"{source}/",
            f"{host}:{target}/",
            stdin=asyncio.subprocess.DEVNULL,
        )
        try:
            await asyncio.wait_for(process.wait(), timeout)
        except TimeoutError:
            process.kill()
            await process.wait()
            raise
        if process.returncode:
            raise OSError(f"rsync exited {process.returncode}")

    async def _fill(
        self, fleet_config: str, clone_root: str, timeout: int
    ) -> ModelActionArchiveCacheFillResult:
        """Run the fill through its node runtime entry and read its one result line."""
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(_FILL_ENTRY),
            "--config",
            fleet_config,
            "--clone-root",
            clone_root,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout)
        except TimeoutError:
            process.kill()
            await process.wait()
            raise
        prefix = "[action-archive-cache-fill] "
        for line in stdout.decode().splitlines():
            if line.startswith(prefix):
                return ModelActionArchiveCacheFillResult.model_validate_json(
                    line[len(prefix) :]
                )
        raise ValueError("the fill printed no result line")

    async def _sync_action_archive_cache(
        self, request: ModelDeployRunnersRequest, timeout: int
    ) -> tuple[str, ...]:
        """OMN-24299: fill the cache on this host and copy it to the runner host.

        Fail-open by construction: a runner that finds no archive downloads from
        GitHub as it always did, so every problem is a named warning, never a
        refusal. rsync ``--ignore-existing`` because an archive is keyed by its
        sha and immutable; ``--exclude=.sources`` because the fill's fetch mirrors
        are deploy-host working state.
        """
        if not request.fleet_config or not request.clone_root:
            return ()
        try:
            fleet = load_runner_fleet_config(Path(request.fleet_config))
        except (OSError, ValueError):
            return ("ACTION-ARCHIVE-CONFIG: the fleet declaration cannot be read.",)
        row = next((h for h in fleet.hosts if h.host == request.host), None)
        cache = fleet.action_archive_cache
        if (
            row is None
            or row.action_archive_host_root is None
            or not (cache and cache.active)
        ):
            return ()
        try:
            filled = await self._fill(
                request.fleet_config, request.clone_root, cache.refresh_interval_seconds
            )
        except (OSError, TimeoutError, ValueError):
            return ("ACTION-ARCHIVE-FILL the fill did not finish.",)
        warnings = [
            f"ACTION-ARCHIVE-FILL {f.reason_code} {f.ref}" for f in filled.failed
        ]
        if not filled.staging_root:
            return tuple(warnings)
        root = row.action_archive_host_root
        try:
            # Owned by the deploy user (so rsync can write it) and read-only to the
            # runner containers, which mount it ro; sudo as the TMPDIR directories do.
            await self._remote(
                request.host,
                f'set -eu; sudo -n install -d -o "$(id -un)" -m 0755 -- {shlex.quote(root)}',
                timeout,
            )
            await self._rsync(filled.staging_root, request.host, root, timeout)
        except (OSError, TimeoutError):
            warnings.append(
                f"ACTION-ARCHIVE-SYNC host={request.host}: cannot copy the cache to {root}."
            )
        return tuple(warnings)

    async def handle(
        self, request: ModelDeployRunnersRequest
    ) -> ModelDeployRunnersResult:
        if not request.secondary or request.dry_run:
            return ModelDeployRunnersResult(success=True)
        try:
            policy = ModelRunnerTmpPolicy()
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
                    if (
                        volume.type is EnumComposeMountType.BIND
                        and volume.target == tmpdir
                    ):
                        if (
                            not Path(volume.source).is_absolute()
                            or "\n" in volume.source
                        ):
                            raise ValueError("invalid TMPDIR source")
                        directories.add(volume.source)
        except (OSError, ValueError, KeyError, TimeoutError):
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
        warnings = await self._sync_action_archive_cache(
            request, policy.timeout_seconds
        )
        return ModelDeployRunnersResult(
            success=True, prepared=tuple(prepared), warnings=warnings
        )
