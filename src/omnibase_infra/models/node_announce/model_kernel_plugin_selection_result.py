# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The kernel plugins to register, in the overlay's order (OMN-21591)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from omnibase_infra.models.node_announce.model_selected_kernel_plugin import (
    ModelSelectedKernelPlugin,
)


class ModelKernelPluginSelectionResult(BaseModel):
    """Plugins both enabled by the overlay and admitted as ``kernel_plugin`` entries."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    selected: tuple[ModelSelectedKernelPlugin, ...] = ()
