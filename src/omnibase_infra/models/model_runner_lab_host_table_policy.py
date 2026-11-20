# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Where the runner installer reads the lab host table from, where the containers read it, and how it is converged."""

from __future__ import annotations

import logging
import re
import shlex
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.model_runner_lab_host_table_result import (
    ModelRunnerLabHostTableResult,
)

_LOG = logging.getLogger(__name__)

# ``(host, shell command, timeout seconds) -> stdout``; raises OSError or TimeoutError on a failed command.
RunOnHost = Callable[[str, str, int], Awaitable[str]]
# ``(host, shell word naming the file, new content, timeout seconds)``; overwrites the file in place.
WriteOnHost = Callable[[str, str, str, int], Awaitable[None]]


class ModelRunnerLabHostTablePolicy(BaseModel):
    """OMN-26098: the declared source and the installer-owned copy of the lab host table.

    The declared source is the overlay projection on the runner host (owner-only, mode 0600, owned by the deploy user).
    A runner container cannot read that file, so each deploy copies it to ``copy_home_relative`` and writes the
    copy's absolute path to ``env_key`` in the host's ``docker/.env``; every runner compose mounts that path read-only
    at ``/etc/onex/lab_run_hosts.yaml``.

    Both paths are relative to the home directory of the SSH user on the runner host, resolved on that host. The
    deploy request's ``lab_host_table_projection`` (``RUNNER_LAB_HOST_TABLE_PROJECTION`` in the deploying
    environment) replaces the projection path.

    ``reader_uid`` is the uid the Runner.Worker runs as: ``docker/runners/Dockerfile`` creates ``runner`` as uid 1001,
    the entrypoint starts as root and drops to it with gosu. The copy is owned by that uid and by the deploy user's
    group, mode 0440: the runner and the deploy user read it, no other host user does, and nobody writes it but root.

    The two installers that end in ``docker compose up`` of such a pool (the deploy effect and the secondary pool image
    roll) each hand their own SSH seam to :meth:`converge`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    env_key: str = "RUNNER_LAB_HOST_TABLE"
    projection_home_relative: str = (
        ".local/state/onex/overlay-store/lab_hosts/overlay.yaml"
    )
    env_file_home_relative: str = ".omnibase/runners/docker/.env"
    copy_home_relative: str = ".omnibase/runners/lab-host-table/lab_run_hosts.yaml"
    reader_uid: int = Field(default=1001, ge=0)
    copy_mode: str = "440"
    copy_dir_mode: str = "750"
    timeout_seconds: int = Field(default=60, gt=0)

    @staticmethod
    def home_path(value: str) -> str:
        """A shell word for ``value``: absolute as is, otherwise under the remote user's home."""
        if value.startswith("/"):
            return shlex.quote(value)
        return f'"$HOME"/{shlex.quote(value)}'

    def with_env_value(self, content: str, value: str) -> str:
        """``content`` with exactly one ``env_key=value`` line and every other byte untouched.

        The first existing line is replaced in place, later ones are dropped, and a missing key is appended on its own
        line. Lines are split on LF alone.
        """
        out: list[str] = []
        placed = False
        for line in re.findall(r"[^\n]*\n|[^\n]+", content):
            if line.startswith(f"{self.env_key}="):
                if not placed:
                    ending = "\n" if line.endswith("\n") else ""
                    out.append(f"{self.env_key}={value}{ending}")
                    placed = True
                continue
            out.append(line)
        if not placed:
            if out and not out[-1].endswith("\n"):
                out[-1] += "\n"
            out.append(f"{self.env_key}={value}\n")
        return "".join(out)

    def copy_script(self, projection: str) -> str:
        """The remote script that copies the projection for the runner user and prints ``COPIED:<path>``.

        A missing or empty projection prints ``MISSING:<path>`` and creates nothing.
        """
        uid = self.reader_uid
        return "; ".join(
            [
                "set -eu",
                f"src={self.home_path(projection or self.projection_home_relative)}",
                f"dst={self.home_path(self.copy_home_relative)}",
                'if [ ! -s "$src" ]; then printf "MISSING:%s\\n" "$src"; exit 0; fi',
                'test ! -L "$dst"',
                'grp="$(id -g)"',
                f'sudo -n install -d -o {uid} -g "$grp" -m {self.copy_dir_mode} -- "$(dirname "$dst")"',
                f'if [ ! -e "$dst" ]; then sudo -n install -o {uid} -g "$grp" -m {self.copy_mode} /dev/null "$dst"; fi',
                'sudo -n sh -c \'cat -- "$1" > "$2"\' sh "$src" "$dst"',
                f'sudo -n chown {uid}:"$grp" -- "$dst"',
                f'sudo -n chmod {self.copy_mode} -- "$dst"',
                'printf "COPIED:%s\\n" "$dst"',
            ]
        )

    async def converge(
        self,
        *,
        host: str,
        env_file: str,
        projection: str,
        run: RunOnHost,
        write: WriteOnHost,
    ) -> ModelRunnerLabHostTableResult:
        """Copy the projection for the runner user, then make ``env_file`` name the copy.

        ``env_file`` and ``projection`` are absolute or relative to the SSH user's home on ``host``. A missing or
        empty projection refuses, naming the path; nothing is created and no stale value is kept.
        """
        timeout = self.timeout_seconds
        uid = self.reader_uid
        try:
            answer = (await run(host, self.copy_script(projection), timeout)).strip()
        except (OSError, TimeoutError):
            return self._refusal(f"COPY host={host}: cannot copy the projection.")
        kind, _, path = answer.partition(":")
        if kind == "MISSING" and path:
            return self._refusal(
                f"PROJECTION host={host} path={path}: the overlay projection is missing or empty."
            )
        if kind != "COPIED" or not path.startswith("/"):
            return self._refusal(f"COPY host={host}: unreadable copy answer.")
        try:
            actual = await run(
                host, f"LC_ALL=C stat -c '%F:%u:%a' -- {shlex.quote(path)}", timeout
            )
        except (OSError, TimeoutError):
            actual = ""
        if actual.strip() != f"regular file:{uid}:{self.copy_mode}":
            return self._refusal(
                f"VERIFY host={host} path={path}: owner or mode readback failed."
            )
        env_word = self.home_path(env_file)
        try:
            current = await run(host, f"cat -- {env_word}", timeout)
            updated = self.with_env_value(current, path)
            if updated != current:
                await write(host, env_word, updated, timeout)
        except (OSError, TimeoutError, ValueError):
            return self._refusal(
                f"ENV host={host} path={env_file}: cannot update the key."
            )
        _LOG.info(
            "RUNNER-LAB-HOST-TABLE-READY host=%s path=%s uid=%s mode=%s",
            host,
            path,
            uid,
            self.copy_mode,
        )
        return ModelRunnerLabHostTableResult(success=True, path=path)

    @staticmethod
    def _refusal(detail: str) -> ModelRunnerLabHostTableResult:
        return ModelRunnerLabHostTableResult(
            success=False, error_message=f"RUNNER-LAB-HOST-TABLE-{detail}"
        )
