# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One explicitly configured HTTP operation; credentials are environment references."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class ModelApiFixtureHttpStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    target: str = "api"
    method: Literal["GET", "HEAD", "OPTIONS", "POST", "PUT", "PATCH", "DELETE"]
    path: str = Field(pattern=r"^/[^\s]*$")
    query: dict[str, str] = Field(default_factory=dict)
    credential_headers: dict[str, str] = Field(default_factory=dict)
    body: JsonValue = None
    expected_status: tuple[int, ...] = (200, 201, 202, 204)
    cleanup: tuple[str, ...] = ()
    bindings: dict[str, str] = Field(default_factory=dict)
    scope: Literal["throwaway", "isolated_slot"] = "throwaway"
