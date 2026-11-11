# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Paths, never private bodies or credential values, travel on the command bus."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ModelApiFixtureRecordRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: Literal["api_fixture.record"] = "api_fixture.record"
    mode: Literal["record", "validate"] = "record"
    policy_path: str
    fixture_path: str
    overlay_path: str = ""
    # Private owner-only JSON file: credential reference -> value.
    credentials_path: str = ""
    recorded_path: str
