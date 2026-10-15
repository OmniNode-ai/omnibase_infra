# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One item a deployment declares on a surface."""

from pydantic import BaseModel, ConfigDict, Field


class ModelDeclaredStateItem(BaseModel):
    """A declared item and the content digest the surface must hold."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    item_key: str = Field(min_length=1, pattern=r"\S")
    digest: str = Field(min_length=1, pattern=r"\S")


__all__ = ["ModelDeclaredStateItem"]
