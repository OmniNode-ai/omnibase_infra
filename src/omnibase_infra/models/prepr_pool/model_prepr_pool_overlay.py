# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The pre-merge runtime pool driver's contract overlay (OMN-22153).

``scripts/runtime_build/prepr_runtime_pool.py`` declares none of these values:
its timeouts, the health poll interval, the git host and the two command
prefixes come from ``config/onex/overlays/prepr_runtime_pool/overlay.yaml`` under
the workspace config root. Every field is required and none has a default, so an
overlay that omits one is refused when it is read, not when a run reaches it.
"""

from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

OVERLAY_RELATIVE = (
    Path("config") / "onex" / "overlays" / "prepr_runtime_pool" / "overlay.yaml"
)


class ModelPreprPoolOverlay(BaseModel):
    """One pool overlay file, validated whole."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    git_host: str = Field(min_length=1)
    ssh_connect_timeout_s: int = Field(gt=0)
    probe_timeout_s: float = Field(gt=0)
    lease_call_timeout_s: float = Field(gt=0)
    remote_setup_timeout_s: float = Field(gt=0)
    fact_publish_timeout_s: float = Field(gt=0)
    check_run_dispatch_timeout_s: float = Field(gt=0)
    git_resolve_timeout_s: float = Field(gt=0)
    probe_poll_s: int = Field(gt=0)
    remote_command_prefix: str
    local_command_prefix: str

    @classmethod
    @cache
    def read(cls, config_root: Path) -> "ModelPreprPoolOverlay":
        """The overlay under ``config_root``; a missing file or key is a loud refusal."""
        path = config_root / OVERLAY_RELATIVE
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    def clone_base_url(self) -> str:
        """The clone base of every first-party repository: this git host and the layer map's organisation."""
        layers = yaml.safe_load(
            files("omnibase_core.contracts")
            .joinpath("repo_layers.yaml")
            .read_text(encoding="utf-8")
        )
        return f"https://{self.git_host}/{layers['github_org']}"
