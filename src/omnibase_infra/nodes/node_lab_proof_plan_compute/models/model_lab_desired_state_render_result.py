# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Canonical desired-state bytes represented as UTF-8 JSON."""

from pydantic import BaseModel, ConfigDict, Field


class ModelLabDesiredStateRenderResult(BaseModel):
    """The schema validator at the adapter boundary enforces lab-desired-state.v1."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    document_json: str
    desired_state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
