# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelJobLineage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    build_ref: str = ""
