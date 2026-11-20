# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Node runtime for ``runner.cache-ownership.restore``: the periodic entry on the runner host (OMN-26113).

Run as ``python -m omnibase_infra.nodes.node_deploy_runners.runtime_runner_cache_ownership_restore``.
"""

from __future__ import annotations

import argparse
import sys

from omnibase_infra.handlers.handler_runner_cache_ownership_restore import (
    DockerCacheCli,
    HandlerRunnerCacheOwnershipRestore,
)
from omnibase_infra.models.model_runner_cache_ownership_restore_request import (
    ModelRunnerCacheOwnershipRestoreRequest,
)


def _handler_with_docker(
    docker: DockerCacheCli,
) -> HandlerRunnerCacheOwnershipRestore:
    """The handler working through a docker the caller supplies (tests)."""

    class InjectedDockerRestore(HandlerRunnerCacheOwnershipRestore):
        def make_docker(
            self, request: ModelRunnerCacheOwnershipRestoreRequest
        ) -> DockerCacheCli:
            return docker

    return InjectedDockerRestore()


def main(argv: list[str] | None = None, *, docker: DockerCacheCli | None = None) -> int:
    """Exit 0 ok (busy runners are left for the next pass), 1 a restore did not hold, 2 the list was unread."""
    parser = argparse.ArgumentParser(
        description="Restore runner cache mount ownership in idle runner containers."
    )
    parser.add_argument("--container-match", required=True)
    parser.add_argument("--docker-timeout", type=int, required=True)
    parser.add_argument("--chown-timeout", type=int, required=True)
    args = parser.parse_args(argv)
    handler = (
        HandlerRunnerCacheOwnershipRestore()
        if docker is None
        else _handler_with_docker(docker)
    )
    result = handler.handle(
        ModelRunnerCacheOwnershipRestoreRequest(
            container_match=args.container_match,
            docker_timeout_seconds=args.docker_timeout,
            chown_timeout_seconds=args.chown_timeout,
        )
    )
    sys.stdout.write(f"[runner-cache-ownership-restore] {result.model_dump_json()}\n")
    return {"ok": 0, "failed": 1, "error": 2}[result.outcome]


if __name__ == "__main__":
    sys.exit(main())
