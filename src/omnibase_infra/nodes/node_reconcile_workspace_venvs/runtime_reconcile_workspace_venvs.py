# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Bootstrap runtime for the host reconciler's contract-bound hook sync.

This operation runs before the dispatcher is available, after its gate venv
has been reconciled. Host ownership remains the shell adapter's responsibility.
The shell invokes this module explicitly; it is not a broker-discovered entry
point, since deployed consumers cannot establish the host ownership precondition.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import yaml

from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_infra.models.model_hook_venv_reconcile_policy import (
    ModelHookVenvReconcilePolicy,
)
from omnibase_infra.models.model_hook_venv_reconcile_request import (
    ModelHookVenvReconcileRequest,
)
from omnibase_infra.nodes.node_reconcile_workspace_venvs.handler_reconcile_workspace_venvs import (
    HandlerHookVenvReconcileHost,
    HandlerReconcileWorkspaceVenvs,
)
from omnibase_infra.protocols.protocol_hook_venv_reconcile_host import (
    HOOK_VENV_HOST_SERVICE,
)


async def _bootstrap(contract_path: Path) -> ModelONEXContainer:
    """Bind contract configuration and the effect at the composition root."""
    contract = yaml.safe_load(contract_path.read_text())
    policy = ModelHookVenvReconcilePolicy.model_validate(contract["reconcile_policy"])
    container = ModelONEXContainer()
    await container.service_registry.register_instance(
        ModelHookVenvReconcilePolicy, policy
    )
    await container.service_registry.register_instance(
        HOOK_VENV_HOST_SERVICE, HandlerHookVenvReconcileHost(container)
    )
    await container.service_registry.register_instance(
        HandlerReconcileWorkspaceVenvs, HandlerReconcileWorkspaceVenvs(container)
    )
    return container


def main() -> int:
    """Adapt the pre-dispatch shell reconciler to the contract-bound handler."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--uv", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    args = parser.parse_args()
    container = asyncio.run(_bootstrap(args.contract))
    result = container.get_service(HandlerReconcileWorkspaceVenvs).handle(
        ModelHookVenvReconcileRequest(
            workspace_root=args.workspace,
            project=args.project,
            uv_bin=args.uv,
        )
    )
    if result.output:
        sys.stderr.write(result.output)
    stream = sys.stderr if result.status == "refused" else sys.stdout
    stream.write("[reconcile] " + result.detail + "\n")
    return 2 if result.status == "refused" else 0


if __name__ == "__main__":
    raise SystemExit(main())
