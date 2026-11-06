# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Compare the hook each runner container EXECUTES with the host copy (OMN-22478).

The retired host-only checker hashed ``/usr/local/bin/<script>`` in every
container. Since the runners mount the hook directory and run
``/opt/onex-runner-hooks/<script>``, that path is an unused copy baked into the
image, so it reported drift for every runner that was in fact current. This
handler reads the path each container really runs -- the job-started hook from
``ACTIONS_RUNNER_HOOK_JOB_STARTED``, the entrypoint and the healthcheck from the
container's own entrypoint, command and healthcheck -- and hashes that path.

A container that declares no hook path for a script is reported in
``not_checked``, never as a pass. A missing host file or an unreadable container
list is ``outcome="error"``, never ``ok``.

The command line entry is ``nodes/node_deploy_runners/runtime_hook_mount_drift.py``.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import re
import subprocess
import sys
from pathlib import Path

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_runner_hook_mount_drift_request import (
    ModelRunnerHookMountDriftRequest,
)
from omnibase_infra.models.model_runner_hook_mount_drift_result import (
    ModelRunnerHookMountDriftResult,
)
from omnibase_infra.models.model_runner_hook_mount_finding import (
    ModelRunnerHookMountFinding,
)

_JOB_STARTED_ENV = "ACTIONS_RUNNER_HOOK_JOB_STARTED"
_JOB_STARTED_SCRIPT = "runner-job-started.sh"


class DockerCli:
    """The docker reads the check needs, through the CLI without a shell.

    Built by the handler's ``make_docker``; tests replace it with a recording.
    """

    def __init__(self, timeout_seconds: int) -> None:
        self._timeout = timeout_seconds

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["docker", *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=self._timeout,
        )

    def container_names(self) -> list[str]:
        done = self._run("ps", "--format", "{{.Names}}")
        if done.returncode:
            raise OSError(f"docker ps exited {done.returncode}")
        return sorted(done.stdout.split())

    def inspect(self, name: str) -> dict[str, object]:
        done = self._run("inspect", name)
        if done.returncode:
            raise OSError(f"docker inspect {name} exited {done.returncode}")
        return json.loads(done.stdout)[0]

    def sha256(self, name: str, path: str) -> str | None:
        done = self._run("exec", name, "sha256sum", path)
        words = done.stdout.split()
        return words[0] if done.returncode == 0 and words else None


def executed_paths(
    inspected: dict[str, object], scripts: tuple[str, ...]
) -> dict[str, str | None]:
    """The in-container path each script runs from, or None where none is declared."""
    config = inspected.get("Config")
    config = config if isinstance(config, dict) else {}
    env = dict(e.split("=", 1) for e in (config.get("Env") or []) if "=" in e)
    startup = " ".join([*(config.get("Entrypoint") or []), *(config.get("Cmd") or [])])
    health = config.get("Healthcheck")
    health_text = " ".join((health or {}).get("Test") or []) if health else ""
    paths: dict[str, str | None] = {}
    for script in scripts:
        if script == _JOB_STARTED_SCRIPT:
            declared = env.get(_JOB_STARTED_ENV)
            paths[script] = declared if declared and declared.endswith(script) else None
            continue
        source = health_text if script.startswith("health") else startup
        found = re.findall(rf"(/[^\s\"';&|]*/{re.escape(script)})", source)
        paths[script] = found[-1] if found else None
    return paths


class HandlerRunnerHookMountDrift:
    """Read-only: reports, never remounts and never restarts a container."""

    # No constructor: the local node runtime builds every handler a contract binds
    # from its class alone, and a deploy-runners run that routes another operation
    # of this node builds this one too. The docker reads are therefore made per
    # request through ``make_docker``, which a caller may replace in a subclass.

    def make_docker(self, request: ModelRunnerHookMountDriftRequest) -> DockerCli:
        return DockerCli(request.docker_timeout_seconds)

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def handle(
        self, request: ModelRunnerHookMountDriftRequest
    ) -> ModelRunnerHookMountDriftResult:
        docker = self.make_docker(request)
        host: dict[str, str] = {}
        for script in request.scripts:
            file = Path(request.host_hook_dir) / script
            if not file.is_file():
                return ModelRunnerHookMountDriftResult(
                    outcome="error", error_message=f"host script missing: {script}"
                )
            host[script] = hashlib.sha256(file.read_bytes()).hexdigest()
        try:
            names = [
                n
                for n in docker.container_names()
                if re.search(request.container_match, n)
            ]
            if not names:
                return ModelRunnerHookMountDriftResult(
                    outcome="error",
                    error_message=f"no containers match {request.container_match}",
                )
            findings: list[ModelRunnerHookMountFinding] = []
            not_checked: list[str] = []
            checked = 0
            for name in names:
                paths = executed_paths(docker.inspect(name), request.scripts)
                for script, path in paths.items():
                    if path is None:
                        not_checked.append(f"{name} {script}")
                        continue
                    checked += 1
                    got = docker.sha256(name, path)
                    if got == host[script]:
                        continue
                    findings.append(
                        ModelRunnerHookMountFinding(
                            container=name,
                            script=script,
                            executed_path=path,
                            reason="unreadable" if got is None else "differs",
                            container_sha256=got,
                            host_sha256=host[script],
                        )
                    )
        except (OSError, ValueError, KeyError, IndexError) as exc:
            return ModelRunnerHookMountDriftResult(
                outcome="error", error_message=f"{type(exc).__name__}: {exc}"
            )
        return ModelRunnerHookMountDriftResult(
            outcome="drift" if findings else "ok",
            checked=checked,
            not_checked=tuple(not_checked),
            findings=tuple(findings),
        )


def transition_notice(
    result: ModelRunnerHookMountDriftResult, state_file: Path
) -> str | None:
    """The alert text, only when the finding count changed since the last run."""
    previous = state_file.read_text().strip() if state_file.is_file() else "0"
    state_file.write_text(f"{len(result.findings)}\n")
    if not result.findings or previous == str(len(result.findings)):
        return None
    lines = "\n".join(
        f"{f.container} {f.script} {f.executed_path} "
        f"executed={(f.container_sha256 or 'unreadable')[:16]} host={f.host_sha256[:16]}"
        for f in result.findings[:20]
    )
    return (
        f"*[HOOK MOUNT DRIFT]* {len(result.findings)}/{result.checked} executed runner "
        "hooks differ from the host copy. Runners are executing old code.\n"
        f"```\n{lines}\n```"
    )


def slack_notify(text: str, env_file: Path, timeout_seconds: int) -> None:
    """Post to the alert channel the retired script posted to.

    The bot token and channel come from the monitor's env file (read as a file,
    never printed); silent when the file or either key is absent.
    """
    settings: dict[str, str] = {}
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            key, _, value = line.partition("=")
            settings[key.strip()] = value.strip().strip("\"'")
    token = settings.get("SLACK_BOT_TOKEN")
    channel = settings.get("SLACK_CHANNEL_ID")
    if not token or not channel:
        return
    body = json.dumps(
        {
            "channel": channel,
            "attachments": [
                {"color": "danger", "text": text, "footer": "hook-mount-drift"}
            ],
        }
    ).encode()
    connection = http.client.HTTPSConnection("slack.com", timeout=timeout_seconds)
    try:
        connection.request(
            "POST",
            "/api/chat.postMessage",
            body,
            {"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        connection.getresponse().read()
    finally:
        connection.close()
