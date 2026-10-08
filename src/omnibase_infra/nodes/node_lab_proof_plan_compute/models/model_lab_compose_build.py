# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The build identity fields used by desired-state rendering."""

from pydantic import BaseModel, ConfigDict


class ModelLabComposeBuild(BaseModel):
    """Compose owns remaining build fields; its config hash covers them."""

    model_config = ConfigDict(extra="ignore", frozen=True)
    context: str
    dockerfile: str = "Dockerfile"
