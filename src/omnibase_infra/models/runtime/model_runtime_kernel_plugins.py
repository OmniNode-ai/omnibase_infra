# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The kernel plugins a lane enables, read from its overlay (OMN-21591).

``ONEX_RUNTIME_KERNEL_PLUGINS`` is the comma-separated value of the lane
overlay's ``runtime.kernel_plugins``: the names of the announced entries of kind
``kernel_plugin`` this lane's kernel registers. The code holds no plugin name and
no default list. Unset or blank names nothing, so no announced plugin is
registered; a name the catalog does not hold as an admitted ``kernel_plugin`` is a
typed boot error raised before the container exists.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

ENV_RUNTIME_KERNEL_PLUGINS = "ONEX_RUNTIME_KERNEL_PLUGINS"


class ModelRuntimeKernelPlugins(BaseModel):
    """The plugin names a lane's overlay enables, in declared order, deduplicated."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    names: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Kernel plugin names the lane enables, in declared order. Registration "
            "follows this order, so a lane lists a light plugin before a heavy one."
        ),
    )

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None
    ) -> ModelRuntimeKernelPlugins:
        """Read the enabled plugin names from the environment."""
        source: Mapping[str, str] = os.environ if environ is None else environ
        raw = source.get(ENV_RUNTIME_KERNEL_PLUGINS) or ""
        return cls(
            names=tuple(
                dict.fromkeys(part.strip() for part in raw.split(",") if part.strip())
            )
        )


__all__ = ["ENV_RUNTIME_KERNEL_PLUGINS", "ModelRuntimeKernelPlugins"]
