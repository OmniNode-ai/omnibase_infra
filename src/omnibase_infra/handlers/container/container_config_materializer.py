# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Binding site for HandlerWorkspaceRuntimeConfigMaterializer (OMN-21683).

The delegate CLI materializes the workspace runtime config through this handler.
This module is the one place that builds it.
"""

from __future__ import annotations

from omnibase_infra.handlers.handler_workspace_runtime_config_materializer import (
    HandlerWorkspaceRuntimeConfigMaterializer,
)

__all__: list[str] = ["bind_workspace_runtime_config_materializer"]


def bind_workspace_runtime_config_materializer() -> (
    HandlerWorkspaceRuntimeConfigMaterializer
):
    """Build the workspace runtime config materializer."""
    return HandlerWorkspaceRuntimeConfigMaterializer()
