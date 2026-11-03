# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The process environment the listener holder reads and writes (OMN-22379)."""

from __future__ import annotations

from collections.abc import MutableMapping

from pydantic import BaseModel, ConfigDict, SkipValidation


class ModelRuntimeListenerEnvironment(BaseModel):
    """``values`` is ``os.environ`` in the agent and a plain dict in a test."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # SkipValidation keeps the caller's mapping itself (os.environ), not a copy of it.
    values: SkipValidation[MutableMapping[str, str]]


__all__: list[str] = ["ModelRuntimeListenerEnvironment"]
