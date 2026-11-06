# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Node runtime for ``runner.hook-mount.check``: the cron entry on the runner host (OMN-22478).

Run as ``python -m omnibase_infra.nodes.node_deploy_runners.runtime_hook_mount_drift``.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import sys
from collections.abc import Callable
from pathlib import Path

from omnibase_infra.handlers.handler_runner_hook_mount_drift import (
    DockerCli,
    HandlerRunnerHookMountDrift,
    slack_notify,
    transition_notice,
)
from omnibase_infra.models.model_runner_hook_mount_drift_request import (
    ModelRunnerHookMountDriftRequest,
)


def _handler_with_docker(docker: DockerCli) -> HandlerRunnerHookMountDrift:
    """The handler reading through a docker the caller supplies (tests)."""

    class HandlerWithDocker(HandlerRunnerHookMountDrift):
        def make_docker(self, request: ModelRunnerHookMountDriftRequest) -> DockerCli:
            return docker

    return HandlerWithDocker()


def main(
    argv: list[str] | None = None,
    *,
    docker: DockerCli | None = None,
    notify: Callable[[str], None] | None = None,
) -> int:
    """Exit 0 no drift, 1 drift, 2 the check could not read (matches the retired script)."""
    parser = argparse.ArgumentParser(
        description="Compare executed runner hooks with the host copies."
    )
    parser.add_argument("--hook-dir", required=True)
    parser.add_argument("--container-match", required=True)
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--docker-timeout", type=int, required=True)
    parser.add_argument("--alert-env-file", type=Path, required=True)
    parser.add_argument("--notify-timeout", type=int, required=True)
    args = parser.parse_args(argv)
    handler = (
        HandlerRunnerHookMountDrift()
        if docker is None
        else _handler_with_docker(docker)
    )
    result = handler.handle(
        ModelRunnerHookMountDriftRequest(
            host_hook_dir=args.hook_dir,
            container_match=args.container_match,
            docker_timeout_seconds=args.docker_timeout,
        )
    )
    sys.stdout.write(f"[hook-mount-drift] {result.model_dump_json()}\n")
    if result.outcome == "error":
        return 2
    notice = transition_notice(result, Path(args.state_file))
    if notice:
        (
            notify
            or functools.partial(
                slack_notify,
                env_file=args.alert_env_file,
                timeout_seconds=args.notify_timeout,
            )
        )(notice)
    return 1 if result.findings else 0


if __name__ == "__main__":
    sys.exit(main())
