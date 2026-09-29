# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Environment override provenance for delegation configuration."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelDelegateEnvConfigOverride(BaseModel):
    """A delegation configuration path selected by an environment variable."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    config_key: str
    source: str = "contract_overlay_env"
    resolved_path: str
