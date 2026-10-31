# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Binding site for HandlerWorkspaceConfigRootResolve (OMN-21683).

The delegate lane selector and the runtime-config materializer resolve the
workspace config root through this handler. This module is the one place that
builds it.
"""

from __future__ import annotations

from omnibase_infra.handlers.handler_workspace_config_root_resolve import (
    HandlerWorkspaceConfigRootResolve,
)

__all__: list[str] = ["bind_root_resolve"]


def bind_root_resolve() -> HandlerWorkspaceConfigRootResolve:
    """Build the workspace config root resolver."""
    return HandlerWorkspaceConfigRootResolve()
