# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One attachment session of a runtime to a held listener (OMN-22379)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelRuntimeListenerSession(BaseModel):
    """A session is PROVISIONAL until the runtime confirms it is accepting."""

    model_config = ConfigDict(extra="forbid")

    pool: str
    instance: str
    confirmed: bool = False
    granted: bool = False


__all__: list[str] = ["ModelRuntimeListenerSession"]
