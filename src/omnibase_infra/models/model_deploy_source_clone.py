# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The deploy agent's own deploy-source clone, as declared and as created (OMN-24562)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


class ModelDeploySourceClone(BaseModel):
    """Request and result of the deploy-source clone effect.

    ``resolve`` reads the declared settings and refuses a deploy source that is a
    canonical clone or the agent's own code checkout (``code_checkout``, the repository
    root of ``DEPLOY_AGENT_DIR``). ``ensure`` also creates the agent's own clone when it is absent.
    ``served_sha`` is the commit the agent was serving when the clone is created (empty
    means the tracked ref's head on origin); ``timeout_seconds`` bounds each git command of a creation and must be declared by the
    caller of ``ensure`` (zero is refused); ``created`` and ``mirror_fallbacks`` are
    results; a request leaves them at zero.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    operation: Literal["resolve", "ensure"]
    clone_dir: str
    canonical_clone: str
    mirror_url: str = ""
    tracking_ref: str
    served_sha: str = ""
    lab_overlay_source: str
    host_root: str = ""
    code_checkout: str = ""
    registry_root: str = ""
    timeout_seconds: float = 0.0
    created: bool = False
    mirror_fallbacks: int = 0

    @property
    def canonical_root(self) -> str:
        """The directory holding the canonical clone and its sibling clones."""
        return str(Path(self.canonical_clone).parent)
