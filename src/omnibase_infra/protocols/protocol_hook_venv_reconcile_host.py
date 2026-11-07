# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Host effect boundary used by the hook reconciliation handler."""

from pathlib import Path
from typing import Protocol, runtime_checkable

from omnibase_infra.models.model_hook_venv_reconcile_request import (
    ModelHookVenvReconcileRequest,
)

HookVenvRevision = tuple[str, str]


@runtime_checkable
class ProtocolHookVenvReconcileHost(Protocol):
    """Read installer state and execute bounded package and ancestry operations."""

    def installed(self, project: Path) -> dict[str, HookVenvRevision]:
        """Read installed Git URL/commit pairs."""
        ...

    def carries(
        self, root: Path, before: HookVenvRevision, after: HookVenvRevision
    ) -> bool:
        """Determine whether the proposed commit preserves the installed one."""
        ...

    def uv(
        self, request: ModelHookVenvReconcileRequest, flag: str | None = None
    ) -> tuple[int, str]:
        """Run exact frozen sync, returning its exit status and visible output."""
        ...


# The container uses interface classes as keys; it does not instantiate them.
HOOK_VENV_HOST_SERVICE: type = ProtocolHookVenvReconcileHost
