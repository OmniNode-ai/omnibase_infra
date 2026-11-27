# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Installer effects for retiring a whole declared runner pool on one registry host (OMN-26105)."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shlex
from typing import cast

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_runner_pool_retire_request import (
    ModelRunnerPoolRetireRequest,
)
from omnibase_infra.models.model_runner_pool_retire_result import (
    ModelRunnerPoolRetireResult,
)
from omnibase_infra.observability.runner_health.model_runner_fleet_config import (
    load_runner_fleet_config,
)

_LOG = logging.getLogger(__name__)
_COMMAND_LIMIT_SECONDS = 30
_STOP_LIMIT_SECONDS = 120
_POLL_INTERVAL_SECONDS = 15
_RETRY_PASSES = 3
_OFFLINE_POLLS = 20
_BUSY = 42
_UNREADABLE = 43
# Only a CI pool is retired here. A deploy, verify or customer-plane pool shares hosts and name stems with CI pools
# and serves other work; retiring one needs its own declared verb.
_RETIRABLE_CLASSES = frozenset({"action", "action-multiarch", "merge-queue"})
_DOCKER_ENV = (
    'export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"; '
    'if test -f "$HOME/.omnibase/runners/docker-cli/config.json"; then '
    'export DOCKER_CONFIG="$HOME/.omnibase/runners/docker-cli"; fi; '
)
_LIST_FORMAT = (
    "{{.ID}} {{.Names}} {{.State}} "
    '{{.Label "com.docker.compose.project"}} {{.Label "com.docker.compose.service"}}'
)
# 0 idle, 42 a job worker is running, anything else unreadable. A container that is not running cannot be mid-job.
_IDLE_PROBE = """
running=$(docker inspect --type container --format '{{{{.State.Running}}}}' {container}) || exit 1
if [ "$running" = false ]; then echo idle; exit 0; fi
processes=$(docker top {container}) || exit 1
case "$processes" in *Runner.Worker*) echo busy; exit 0 ;; esac
last=$(docker logs --tail 200 {container} 2>&1 | grep -E 'Running job:|completed with result' | tail -1 || true)
case "$last" in *'Running job:'*) echo busy ;; *) echo idle ;; esac
"""


class RunnerRetireError(Exception):
    """An installer refusal safe to show without subprocess output."""


class RunnerRetireUncertainError(RunnerRetireError):
    """The remote stop may still be running."""


class HandlerRunnerPoolRetire:
    """Select the pool by its exact prefix, then drain and remove one verified idle runner at a time."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self, request: ModelRunnerPoolRetireRequest
    ) -> ModelRunnerPoolRetireResult:
        try:
            org = self._declared_pool(request)
            members = await self._members(request)
            if request.only:
                members = [m for m in members if m[1] == request.only]
                if not members:
                    raise RunnerRetireError(
                        f"--only {request.only} is not a runner of pool {request.prefix} on {request.host}."
                    )
            planned = tuple(name for _, name in members[: request.limit or None])
            if request.dry_run:
                return await self._plan(request, planned, members)
            return await self._retire(request, org, planned, members)
        except RunnerRetireError as exc:
            return ModelRunnerPoolRetireResult(success=False, error_message=str(exc))
        except TimeoutError:
            return ModelRunnerPoolRetireResult(
                success=False,
                error_message=f"Installer command timed out on {request.host}.",
            )
        except (OSError, ValueError):
            return ModelRunnerPoolRetireResult(
                success=False, error_message="Could not read the fleet declaration."
            )

    @staticmethod
    def _declared_pool(request: ModelRunnerPoolRetireRequest) -> str:
        """The organization of the fleet, once ``prefix`` is a declared CI pool of ``host``; refuses before any read."""
        fleet = load_runner_fleet_config(request.fleet_config)
        host = next((row for row in fleet.hosts if row.host == request.host), None)
        pools = (
            {}
            if host is None
            else {
                host.runner_name_prefix: host.classes,
                **{pool.runner_name_prefix: pool.classes for pool in host.pools},
            }
        )
        classes = pools.get(request.prefix, ())
        if not classes or not set(classes) <= _RETIRABLE_CLASSES:
            raise RunnerRetireError(
                "Pool retire requires a declared CI pool (action, action-multiarch or merge-queue) of the named host."
            )
        return fleet.github_org

    @staticmethod
    async def _command(
        *argv: str, stdin: bytes | None = None, timeout: int = _COMMAND_LIMIT_SECONDS
    ) -> tuple[int, str]:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE
            if stdin is not None
            else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _ = await asyncio.wait_for(
                process.communicate(stdin), timeout=timeout
            )
        except (TimeoutError, asyncio.CancelledError):
            if process.returncode is None:
                process.kill()
            await process.wait()
            raise
        return cast("int", process.returncode), stdout.decode(errors="replace").strip()

    async def _remote(
        self,
        host: str,
        script: str,
        *,
        stdin: bytes | None = None,
        timeout: int = _COMMAND_LIMIT_SECONDS,
    ) -> tuple[int, str]:
        return await self._command(
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "--",
            host,
            _DOCKER_ENV + "exec bash -c " + shlex.quote(script),
            stdin=stdin,
            timeout=timeout,
        )

    @staticmethod
    async def _pause() -> None:
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)

    async def _members(
        self, request: ModelRunnerPoolRetireRequest
    ) -> list[tuple[str, str]]:
        """``(container id, name)`` of every ``<prefix>-<digits>`` container of the pool, in numeric order.

        The docker name filter only narrows the read; the selection is the exact match below, and a container whose
        Compose project is not the pool's is left out whatever it is called.
        """
        rc, output = await self._remote(
            request.host,
            f"docker ps -a --no-trunc --filter {shlex.quote(f'name=^/{request.prefix}-[0-9]+$')} "
            f"--format {shlex.quote(_LIST_FORMAT)}",
        )
        if rc:
            raise RunnerRetireError(
                f"Could not list containers on {request.host} (exit {rc})."
            )
        runner = re.compile(rf"{re.escape(request.prefix)}-([0-9]+)")
        found: list[tuple[int, str, str]] = []
        for line in output.splitlines():
            fields = line.split()
            if len(fields) != 5:
                continue
            container_id, name, _state, project, service = fields
            matched = runner.fullmatch(name)
            if matched and project == request.prefix and service == name:
                found.append((int(matched.group(1)), container_id, name))
        return [(container_id, name) for _, container_id, name in sorted(found)]

    async def _plan(
        self,
        request: ModelRunnerPoolRetireRequest,
        planned: tuple[str, ...],
        members: list[tuple[str, str]],
    ) -> ModelRunnerPoolRetireResult:
        busy: list[str] = []
        for container_id, name in members:
            if name in planned and not await self._container_idle(
                request, container_id
            ):
                busy.append(name)
        _LOG.info(
            "[DRY RUN] %s on %s: would retire %s one at a time; busy now, would be left alone: %s",
            request.prefix,
            request.host,
            ", ".join(planned) or "nothing",
            ", ".join(busy) or "none",
        )
        return ModelRunnerPoolRetireResult(
            success=True, planned=planned, busy=tuple(busy)
        )

    async def _container_idle(
        self, request: ModelRunnerPoolRetireRequest, container_id: str
    ) -> bool:
        rc, verdict = await self._remote(
            request.host,
            _IDLE_PROBE.format(container=shlex.quote(container_id)),
        )
        return rc == 0 and verdict == "idle"

    async def _registry(
        self, org: str, name: str
    ) -> tuple[int, str, bool | None] | None:
        rc, response = await self._command(
            "gh",
            "api",
            "--paginate",
            "--slurp",
            "-H",
            "Cache-Control: no-cache",
            f"/orgs/{org}/actions/runners?per_page=100",
        )
        if rc:
            raise RunnerRetireError(
                f"Could not read the organization runner registry (gh exit {rc})."
            )
        try:
            rows = [
                row
                for page in json.loads(response)
                for row in page["runners"]
                if row["name"] == name
            ]
            if len(rows) > 1:
                raise RunnerRetireError(f"{name}: more than one registration.")
            if rows:
                busy = rows[0].get("busy")
                return (
                    int(rows[0]["id"]),
                    str(rows[0].get("status")),
                    busy if isinstance(busy, bool) else None,
                )
        except (KeyError, TypeError, ValueError) as exc:
            raise RunnerRetireError(
                "Could not decode the organization runner registry."
            ) from exc
        return None

    async def _idle(
        self,
        request: ModelRunnerPoolRetireRequest,
        org: str,
        container_id: str,
        name: str,
    ) -> bool:
        """Idle on two signals: the container's own evidence and the registry. Unreadable is not idle."""
        if not await self._container_idle(request, container_id):
            return False
        registration = await self._registry(org, name)
        return registration is None or registration[1:] in {
            ("online", False),
            ("offline", False),
            ("offline", None),
        }

    async def _stop(
        self,
        request: ModelRunnerPoolRetireRequest,
        container_id: str,
        name: str,
    ) -> bool:
        """Stop new jobs: the worker check and the stop run back-to-back on the host, on the inspected id."""
        script = f"""
set -euo pipefail
current=$(docker inspect --type container --format '{{{{.Id}}}}' {shlex.quote(name)}) || exit {_UNREADABLE}
test "$current" = {shlex.quote(container_id)} || exit {_BUSY}
running=$(docker inspect --format '{{{{.State.Running}}}}' {shlex.quote(container_id)}) || exit {_UNREADABLE}
if [ "$running" = true ]; then
    processes=$(docker top {shlex.quote(container_id)}) || exit {_BUSY}
    case "$processes" in *Runner.Worker*) exit {_BUSY} ;; esac
    docker stop -t 60 {shlex.quote(container_id)}
fi
"""
        try:
            rc, _ = await self._remote(
                request.host, script, timeout=_STOP_LIMIT_SECONDS
            )
        except (OSError, TimeoutError) as exc:
            raise RunnerRetireUncertainError(
                f"HALT: lost stop readback for {name}."
            ) from exc
        if rc in {_BUSY, _UNREADABLE}:
            return False
        if rc == 255:
            raise RunnerRetireUncertainError(
                f"HALT: SSH lost stop readback for {name}."
            )
        if rc:
            raise RunnerRetireError(f"HALT: stop failed for {name} (exit {rc}).")
        return True

    async def _wait_offline(self, org: str, name: str) -> int | None:
        """The registration id once GitHub reports the stopped runner offline; ``None`` when it has none."""
        for _ in range(_OFFLINE_POLLS):
            registration = await self._registry(org, name)
            if registration is None:
                return None
            if registration[1] == "offline":
                return registration[0]
            await self._pause()
        raise RunnerRetireError(
            f"HALT: {name} is stopped but GitHub still reports it online; its registration and container are left in place."
        )

    async def _deregister(self, org: str, name: str, runner_id: int) -> None:
        rc, _ = await self._command(
            "gh", "api", "-X", "DELETE", f"/orgs/{org}/actions/runners/{runner_id}"
        )
        if rc:
            raise RunnerRetireError(
                f"HALT: could not delete the registration of {name} (gh exit {rc}); its container is stopped and left in place."
            )

    async def _remove(
        self, request: ModelRunnerPoolRetireRequest, container_id: str, name: str
    ) -> None:
        # One container, by id, never `-v`: the named credential volume is not deleted by this verb.
        rc, _ = await self._remote(
            request.host,
            f"set -eu; test \"$(docker inspect --format '{{{{.State.Running}}}}' {shlex.quote(container_id)})\" = false; "
            f"docker rm {shlex.quote(container_id)}",
        )
        if rc:
            raise RunnerRetireError(f"HALT: could not remove {name} (exit {rc}).")

    async def _retire(
        self,
        request: ModelRunnerPoolRetireRequest,
        org: str,
        planned: tuple[str, ...],
        members: list[tuple[str, str]],
    ) -> ModelRunnerPoolRetireResult:
        ids = {name: container_id for container_id, name in members}
        retired: list[str] = []
        deregistered: list[str] = []
        pending = list(planned)
        for attempt in range(_RETRY_PASSES + 1):
            if attempt:
                await self._pause()
            retry: list[str] = []
            for name in pending:
                try:
                    if not await self._idle(request, org, ids[name], name):
                        retry.append(name)
                        continue
                    if not await self._stop(request, ids[name], name):
                        retry.append(name)
                        continue
                    runner_id = await self._wait_offline(org, name)
                    if runner_id is not None:
                        await self._deregister(org, name, runner_id)
                        deregistered.append(name)
                    await self._remove(request, ids[name], name)
                except RunnerRetireUncertainError:
                    raise
                except (RunnerRetireError, OSError, TimeoutError) as exc:
                    reason = (
                        str(exc)
                        if isinstance(exc, RunnerRetireError)
                        else type(exc).__name__
                    )
                    return ModelRunnerPoolRetireResult(
                        success=False,
                        planned=planned,
                        retired=tuple(retired),
                        deregistered=tuple(deregistered),
                        error_message=f"{name}: {reason}",
                    )
                retired.append(name)
                _LOG.info("%s: drained, deregistered and removed", name)
            pending = retry
            if not pending:
                return ModelRunnerPoolRetireResult(
                    success=True,
                    planned=planned,
                    retired=tuple(retired),
                    deregistered=tuple(deregistered),
                )
            _LOG.info("Busy or unreadable, left untouched: %s", ", ".join(pending))
        return ModelRunnerPoolRetireResult(
            success=False,
            planned=planned,
            retired=tuple(retired),
            deregistered=tuple(deregistered),
            busy=tuple(pending),
            error_message=f"Still busy or unreadable after retries: {', '.join(pending)}",
        )
