# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Lock-filtered extras of one hook environment's declared install set."""

from pydantic import BaseModel, ConfigDict


class ModelHookVenvExtras(BaseModel):
    """Declared extras split by whether the project's lock still lists them."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    kept: tuple[str, ...] = ()
    stray: tuple[str, ...] = ()
