# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Restore the declared cache mounts of idle runner containers to the runner user (OMN-26113).

A runner's cache mounts (``~/.npm`` and ``~/.cache/<name>``) are bind mounts of host directories. When one turns
root-owned (Docker created the source, a root step wrote into it, a restore from backup) every job, which runs as the
runner user, fails its first npm or pre-commit call with EACCES. A pre-job hook runs as the runner user and cannot
fix it, so this check does, through ``docker exec -u 0``.

Safety, each proven by a test:

* it runs a ``chown`` only in a container whose process list is read and holds no ``Runner.Worker`` (exact name match,
  never a pattern that matches the reader's own command line); an unreadable list counts as busy;
* it reads the process list again immediately before the ``chown``;
* it changes ownership only of the container's declared bind mounts that are cache targets, never ``/home/runner``,
  the credential directories, the docker socket or the work directory;
* a container with nothing to restore is not written to, and nothing is ever restarted, recreated or stopped.

The command line entry is ``nodes/node_deploy_runners/runtime_runner_cache_ownership_restore.py``.
"""

from __future__ import annotations

import json
import re
import subprocess

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_runner_cache_ownership_restore_request import (
    ModelRunnerCacheOwnershipRestoreRequest,
)
from omnibase_infra.models.model_runner_cache_ownership_restore_result import (
    ModelRunnerCacheOwnershipRestoreResult,
)
from omnibase_infra.models.model_runner_cache_restore_finding import (
    ModelRunnerCacheRestoreFinding,
)
from omnibase_infra.models.model_runner_tmp_policy import ModelRunnerTmpPolicy

_WORKER = "Runner.Worker"
_LISTENER = "Runner.Listener"


class DockerCacheCli:
    """The docker calls the check needs, through the CLI without a shell. Tests replace it with a recording."""

    def __init__(self, timeout_seconds: int, chown_timeout_seconds: int) -> None:
        self._timeout = timeout_seconds
        self._chown_timeout = chown_timeout_seconds

    def _run(
        self, *args: str, timeout: int | None = None
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["docker", *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout or self._timeout,
        )

    def container_names(self) -> list[str]:
        done = self._run("ps", "--format", "{{.Names}}")
        if done.returncode:
            raise OSError(f"docker ps exited {done.returncode}")
        return sorted(done.stdout.split())

    def bind_destinations(self, name: str) -> list[str]:
        """The container's declared bind mount destinations, from ``docker inspect``."""
        done = self._run("inspect", name)
        if done.returncode:
            raise OSError(f"docker inspect {name} exited {done.returncode}")
        mounts = json.loads(done.stdout)[0].get("Mounts") or []
        return [
            str(m["Destination"])
            for m in mounts
            if m.get("Type") == "bind" and m.get("Destination")
        ]

    def process_names(self, name: str) -> list[str] | None:
        """The command names in the container, or None when the list cannot be read."""
        done = self._run("exec", name, "ps", "-eo", "comm")
        names = done.stdout.split()
        return names if done.returncode == 0 and names else None

    def unowned_entry(self, name: str, uid: int, gid: int, paths: list[str]) -> str:
        """The first entry under ``paths`` the runner user does not own (empty when none)."""
        done = self._run(
            "exec", "-u", "0", name, "find", *paths,
            "(", "!", "-uid", str(uid), "-o", "!", "-gid", str(gid), ")",
            "-print", "-quit",
        )  # fmt: skip
        if done.returncode:
            raise OSError(f"find in {name} exited {done.returncode}")
        return done.stdout.strip()

    def chown(self, name: str, uid: int, gid: int, paths: list[str]) -> None:
        done = self._run(
            "exec", "-u", "0", name, "chown", "-hR", f"{uid}:{gid}", *paths,
            timeout=self._chown_timeout,
        )  # fmt: skip
        if done.returncode:
            raise OSError(f"chown in {name} exited {done.returncode}")


def is_idle(docker: DockerCacheCli, name: str) -> bool:
    """True only when the process list is readable, shows the runner's listener and holds no ``Runner.Worker``.

    A list that cannot be read, or that shows no ``Runner.Listener`` (a container that is not a running runner), is
    not idle: the check writes nothing there.
    """
    processes = docker.process_names(name)
    if processes is None or _LISTENER not in processes:
        return False
    return _WORKER not in processes


class HandlerRunnerCacheOwnershipRestore:
    """Restores ownership of declared cache mounts, idle containers only; reports, never restarts."""

    # No constructor: the local node runtime builds every handler a contract binds from its class alone, so the
    # docker calls are made per request through ``make_docker``, which a caller may replace in a subclass.

    def make_docker(
        self, request: ModelRunnerCacheOwnershipRestoreRequest
    ) -> DockerCacheCli:
        return DockerCacheCli(
            request.docker_timeout_seconds, request.chown_timeout_seconds
        )

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def handle(
        self, request: ModelRunnerCacheOwnershipRestoreRequest
    ) -> ModelRunnerCacheOwnershipRestoreResult:
        docker = self.make_docker(request)
        policy = ModelRunnerTmpPolicy()
        try:
            names = [
                n
                for n in docker.container_names()
                if re.search(request.container_match, n)
            ]
        except (OSError, subprocess.SubprocessError) as exc:
            return ModelRunnerCacheOwnershipRestoreResult(
                outcome="error", error_message=f"{type(exc).__name__}: {exc}"
            )
        if not names:
            return ModelRunnerCacheOwnershipRestoreResult(
                outcome="error",
                error_message=f"no containers match {request.container_match}",
            )
        already: list[str] = []
        busy: list[str] = []
        no_mounts: list[str] = []
        findings: list[ModelRunnerCacheRestoreFinding] = []
        for name in names:
            try:
                paths = sorted(
                    d
                    for d in docker.bind_destinations(name)
                    if policy.is_cache_target(d)
                )
                if not paths:
                    no_mounts.append(name)
                    continue
                if not is_idle(docker, name):
                    busy.append(name)
                    continue
                if not docker.unowned_entry(name, request.uid, request.gid, paths):
                    already.append(name)
                    continue
                if not is_idle(docker, name):  # a job may have started since the read
                    busy.append(name)
                    continue
                docker.chown(name, request.uid, request.gid, paths)
                left = docker.unowned_entry(name, request.uid, request.gid, paths)
            except (
                OSError,
                ValueError,
                KeyError,
                IndexError,
                subprocess.SubprocessError,
            ) as exc:
                findings.append(
                    ModelRunnerCacheRestoreFinding(
                        container=name,
                        paths=(),
                        outcome="failed",
                        detail=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            findings.append(
                ModelRunnerCacheRestoreFinding(
                    container=name,
                    paths=tuple(paths),
                    outcome="failed" if left else "restored",
                    detail=f"still not owned: {left}" if left else "",
                )
            )
        return ModelRunnerCacheOwnershipRestoreResult(
            outcome=(
                "failed" if any(f.outcome == "failed" for f in findings) else "ok"
            ),
            checked=len(names),
            already_owned=tuple(already),
            busy=tuple(busy),
            no_cache_mounts=tuple(no_mounts),
            findings=tuple(findings),
        )
