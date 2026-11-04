# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A mounted route identity and its service-owner-supplied request recipe."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from omnibase_infra.models.model_api_fixture_http_step import ModelApiFixtureHttpStep


class ModelApiFixtureRoute(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    step: ModelApiFixtureHttpStep
    webhook: Literal["", "github", "stripe"] = ""
    signing_secret_env: str = ""
