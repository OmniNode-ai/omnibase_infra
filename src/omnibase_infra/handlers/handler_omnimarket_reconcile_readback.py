# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Read both sides of the drift comparison after an out-of-process repair."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_omnimarket_reconcile_readback_request import (
    ModelOmnimarketReconcileReadbackRequest,
)
from omnibase_infra.models.model_omnimarket_reconcile_readback_result import (
    ModelOmnimarketReconcileReadbackResult,
)

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer


def read_omnimarket_reconcile_state(
    workspace_root: str | Path | None,
) -> ModelOmnimarketReconcileReadbackResult:
    """Invalidate package caches, then read the install and clone as they stand."""
    # Import at invocation time: the CLI guard also calls this handler's reader.
    from omnibase_infra.cli.omnimarket_drift_guard import (
        canonical_local_omnimarket_commit,
        installed_omnimarket_commit,
    )

    importlib.invalidate_caches()
    result = ModelOmnimarketReconcileReadbackResult(
        installed=installed_omnimarket_commit() or "",
        canonical=canonical_local_omnimarket_commit(
            str(workspace_root) if workspace_root else None
        )
        or "",
    )
    if result.matches:
        logging.getLogger(__name__).info(
            "omnimarket drift reconciled in-flight to %s; continuing.",
            result.canonical[:12],
        )
    return result


class HandlerOmnimarketReconcileReadback:
    """Observe package metadata and local Git state without repairing either."""

    def __init__(self, container: ModelONEXContainer) -> None:
        self._container = container

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self, request: ModelOmnimarketReconcileReadbackRequest
    ) -> ModelOmnimarketReconcileReadbackResult:
        return read_omnimarket_reconcile_state(request.workspace_root or None)
