# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Composition root of the stability integration refresh (OMN-26715).

    refresh --request FILE --registry-root DIR --infra-checkout DIR --work-dir DIR
            --operator-env-file FILE --lane-declaration-dir DIR --health-timeout S

``FILE`` holds a ``ModelStabilityIntegrationRefreshRequest`` as JSON. The lane
declaration folder is the ``config/lab`` folder that holds ``ci_bus_lanes.yaml``;
the per-lane bifrost folder is its ``lane-overlays`` subfolder, the layout the deploy agent
resolves. Both reach the host by injection. The registered host is what
``HandlerStabilityIntegrationRefresh`` resolves from the container.

The result is printed as JSON; the exit code is 0 for ``refreshed`` and 1 for
anything else.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_infra.handlers.handler_stability_integration_refresh import (
    DockerStabilityIntegrationHost,
    HandlerStabilityIntegrationRefresh,
)
from omnibase_infra.models.model_stability_integration_refresh_request import (
    ModelStabilityIntegrationRefreshRequest,
)


async def bootstrap_container(
    host: DockerStabilityIntegrationHost,
) -> ModelONEXContainer:
    """Bind the host, then the handler that resolves it, at the composition root."""
    container = ModelONEXContainer()
    await container.service_registry.register_instance(
        DockerStabilityIntegrationHost, host
    )
    await container.service_registry.register_instance(
        HandlerStabilityIntegrationRefresh,
        HandlerStabilityIntegrationRefresh(container),
    )
    return container


def main(argv: list[str] | None = None) -> int:
    """Parse one refresh command, run it and print its result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["refresh"])
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--infra-checkout", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--operator-env-file", type=Path, required=True)
    parser.add_argument("--lane-declaration-dir", type=Path, required=True)
    parser.add_argument("--health-timeout", type=float, required=True)
    args = parser.parse_args(argv)
    request = ModelStabilityIntegrationRefreshRequest.model_validate_json(
        args.request.read_text(encoding="utf-8")
    )
    host = DockerStabilityIntegrationHost(
        registry_root=args.registry_root,
        infra_checkout=args.infra_checkout,
        work_dir=args.work_dir,
        operator_env_file=args.operator_env_file,
        health_timeout_seconds=args.health_timeout,
        lane_declaration_dir=args.lane_declaration_dir,
        lane_overlay_dir=args.lane_declaration_dir / "lane-overlays",
    )
    container = asyncio.run(bootstrap_container(host))
    result = container.get_service(HandlerStabilityIntegrationRefresh).handle(request)
    sys.stdout.write(result.model_dump_json(indent=2) + "\n")
    return 0 if result.outcome == "refreshed" else 1


if __name__ == "__main__":
    sys.exit(main())
