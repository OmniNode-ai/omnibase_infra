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
from omnibase_infra.models.model_runner_lab_host_table_policy import (
    ModelRunnerLabHostTablePolicy,
)
from omnibase_infra.models.model_runner_tmp_compose import ModelRunnerTmpCompose
from omnibase_infra.models.model_runner_tmp_policy import ModelRunnerTmpPolicy
from omnibase_infra.observability.runner_health.model_runner_fleet_config import (
    load_runner_fleet_config,
)

_LOG = logging.getLogger(__name__)
_PAUSE_DIR_TARGET_SUFFIX = "/.onex-disk-admission-pause"
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

    async def _write_remote(
        self, host: str, path: str, data: str, timeout: int
    ) -> None:
        """Overwrite ``path`` in place (inode, owner and mode kept) with ``data`` sent on stdin."""
        process = await asyncio.create_subprocess_exec(
            "ssh",
            "--",
            host,
            f"set -eu; test -f {shlex.quote(path)}; cat > {shlex.quote(path)}",
            stdin=asyncio.subprocess.PIPE,
        )
        try:
            await asyncio.wait_for(process.communicate(data.encode()), timeout)
        except TimeoutError:
            process.kill()
            await process.wait()
            raise
        if process.returncode:
            raise OSError(f"SSH exited {process.returncode}")

    @staticmethod
    def _home_path(value: str) -> str:
        """A shell word for ``value``: absolute as is, otherwise under the remote user's home."""
        if value.startswith("/"):
            return shlex.quote(value)
        return f'"$HOME"/{shlex.quote(value)}'

    @staticmethod
    def with_env_value(content: str, key: str, value: str) -> str:
        """``content`` with exactly one ``key=value`` line and every other byte untouched.

        The first existing ``key=`` line is replaced in place, later ones are dropped, and a missing key is
        appended on its own line.
        """
        lines = re.findall(r"[^\n]*\n|[^\n]+", content)  # split on LF alone
        out: list[str] = []
        placed = False
        for line in lines:
            if line.startswith(f"{key}="):
                if not placed:
                    ending = "\n" if line.endswith("\n") else ""
                    out.append(f"{key}={value}{ending}")
                    placed = True
                continue
            out.append(line)
        if not placed:
            if out and not out[-1].endswith("\n"):
                out[-1] += "\n"
            out.append(f"{key}={value}\n")
        return "".join(out)

    async def _converge_lab_host_table(
        self, request: ModelDeployRunnersRequest, timeout: int
    ) -> str | None:
        """OMN-26098: copy the overlay projection where the runners can read it and point ``docker/.env`` at it.

        Returns a named refusal, or ``None`` when the copy is verified and the host's env file names it. The
        projection is never created or replaced by an empty file: a missing or empty projection refuses the deploy.
        """
        table = ModelRunnerLabHostTablePolicy()
        source = self._home_path(
            request.lab_host_table_projection or table.projection_home_relative
        )
        target = self._home_path(table.copy_home_relative)
        uid = table.reader_uid
        script = "; ".join(
            [
                "set -eu",
                f"src={source}",
                f"dst={target}",
                'if [ ! -s "$src" ]; then printf "MISSING:%s\\n" "$src"; exit 0; fi',
                'test ! -L "$dst"',
                'grp="$(id -g)"',
                f'sudo -n install -d -o {uid} -g "$grp" -m {table.copy_dir_mode} -- "$(dirname "$dst")"',
                f'if [ ! -e "$dst" ]; then sudo -n install -o {uid} -g "$grp" -m {table.copy_mode} /dev/null "$dst"; fi',
                'sudo -n sh -c \'cat -- "$1" > "$2"\' sh "$src" "$dst"',
                f'sudo -n chown {uid}:"$grp" -- "$dst"',
                f'sudo -n chmod {table.copy_mode} -- "$dst"',
                'printf "COPIED:%s\\n" "$dst"',
            ]
        )
        try:
            answer = (await self._remote(request.host, script, timeout)).strip()
        except (OSError, TimeoutError):
            return f"RUNNER-LAB-HOST-TABLE-COPY host={request.host}: cannot copy the projection."
        kind, _, path = answer.partition(":")
        if kind == "MISSING" and path:
            return (
                f"RUNNER-LAB-HOST-TABLE-PROJECTION host={request.host} path={path}: "
                "the overlay projection is missing or empty."
            )
        if kind != "COPIED" or not path.startswith("/"):
            return f"RUNNER-LAB-HOST-TABLE-COPY host={request.host}: unreadable copy answer."
        try:
            actual = await self._remote(
                request.host,
                f"LC_ALL=C stat -c '%F:%u:%a' -- {shlex.quote(path)}",
                timeout,
            )
        except (OSError, TimeoutError):
            actual = ""
        if actual.strip() != f"regular file:{uid}:{table.copy_mode}":
            return f"RUNNER-LAB-HOST-TABLE-VERIFY host={request.host} path={path}: owner or mode readback failed."
        env_path = str(Path(request.host_dir) / "docker" / ".env")
        try:
            current = await self._remote(
                request.host, f"cat -- {shlex.quote(env_path)}", timeout
            )
            updated = self.with_env_value(current, table.env_key, path)
            if updated != current:
                await self._write_remote(request.host, env_path, updated, timeout)
        except (OSError, TimeoutError, ValueError):
            return f"RUNNER-LAB-HOST-TABLE-ENV host={request.host} path={env_path}: cannot update the key."
        _LOG.info(
            "RUNNER-LAB-HOST-TABLE-READY host=%s path=%s uid=%s mode=%s",
            request.host,
            path,
            uid,
            table.copy_mode,
        )
        return None

    @staticmethod
    def _job_dir_root(request: ModelDeployRunnersRequest) -> str | None:
        """OMN-26093: the host root the pool's job directories are declared under, if any.

        Unreadable or absent declarations read as no root, so a pool with no declared root
        keeps preparing the TMPDIR and pause directories alone, as before.
        """
        if not request.fleet_config:
            return None
        try:
            fleet = load_runner_fleet_config(Path(request.fleet_config))
        except (OSError, ValueError):
            return None
        row = next((h for h in fleet.hosts if h.host == request.host), None)
        if row is None or row.runner_name_prefix != request.prefix:
            return None
        return row.job_dir_host_root

    async def handle(
        self, request: ModelDeployRunnersRequest
    ) -> ModelDeployRunnersResult:
        if not request.secondary or request.dry_run:
            return ModelDeployRunnersResult(success=True)
        try:
            policy = ModelRunnerTmpPolicy()
            compose_text = Path(request.compose_file).read_text()
            # Read only the registration variable names; no credential value
            # leaves the remote Compose stdout -> typed model boundary.
            token_vars = sorted(
                set(re.findall(r"RUNNER_TOKEN:\s*\$\{([A-Z0-9_]+)", compose_text))
            )
            # OMN-26098: a compose that mounts the lab host table needs the copy and the env key in place before
            # Compose renders it, here and at the container create that follows.
            if ModelRunnerLabHostTablePolicy().env_key in compose_text:
                refusal = await self._converge_lab_host_table(
                    request, policy.timeout_seconds
                )
                if refusal is not None:
                    return ModelDeployRunnersResult(
                        success=False, error_message=refusal
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
            job_root = self._job_dir_root(request)
            directories: dict[str, str] = {}
            for number in range(1, request.count + 1):
                service = services[f"{request.prefix}-{number}"]
                tmpdir = service.environment.get("TMPDIR")
                for volume in service.volumes:
                    # OMN-25674: the disk-admission pause directory is a bind source too;
                    # Docker creates it root-owned and the hook writes its marker as runner.
                    # OMN-26093: every other bind whose source is under the pool's declared
                    # job-directory root (workspace, caches) is created runner-owned too, so a
                    # runner's first job does not meet a root-owned cache directory.
                    under_job_root = (
                        job_root is not None
                        and Path(volume.source).is_absolute()
                        and Path(volume.source).is_relative_to(job_root)
                    )
                    if volume.type is EnumComposeMountType.BIND and (
                        volume.target == tmpdir
                        or volume.target.endswith(_PAUSE_DIR_TARGET_SUFFIX)
                        or under_job_root
                    ):
                        if (
                            not Path(volume.source).is_absolute()
                            or "\n" in volume.source
                        ):
                            raise ValueError("invalid TMPDIR source")
                        if volume.target == tmpdir:
                            directories[volume.source] = policy.mode
                        elif volume.target.endswith(_PAUSE_DIR_TARGET_SUFFIX):
                            directories[volume.source] = policy.pause_dir_mode
                        else:
                            directories[volume.source] = policy.job_dir_mode
        except (OSError, ValueError, KeyError, TimeoutError):
            return ModelDeployRunnersResult(
                success=False,
                error_message=f"RUNNER-TMP-DISCOVERY host={request.host}: cannot resolve TMPDIR binds.",
            )

        prepared: list[str] = []
        for directory, mode in sorted(directories.items()):
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
                    mode,
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
                if actual.strip() != f"directory:{policy.uid}:{mode}":
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
                mode,
            )
        warnings = await self._sync_action_archive_cache(
            request, policy.timeout_seconds
        )
        return ModelDeployRunnersResult(
            success=True, prepared=tuple(prepared), warnings=warnings
        )
