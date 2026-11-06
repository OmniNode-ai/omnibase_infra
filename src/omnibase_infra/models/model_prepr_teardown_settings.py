# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Installer budgets declared in the slot teardown contract."""

from pydantic import BaseModel, ConfigDict, Field


class ModelPreprTeardownSettings(BaseModel):
    """Bound dependency installation and import proof before replacing units."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    install_timeout_seconds: int = Field(gt=0)
    import_timeout_seconds: int = Field(gt=0)
