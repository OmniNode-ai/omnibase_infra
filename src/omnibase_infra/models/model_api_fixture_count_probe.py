# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A read-only count over the run's throwaway resource scope."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from omnibase_infra.models.model_api_fixture_http_step import ModelApiFixtureHttpStep


class ModelApiFixtureCountProbe(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    resource: Literal["tenants", "accounts", "keycloak_users"]
    step: ModelApiFixtureHttpStep
    pointer: str = ""
    measure: Literal["integer", "length"] = "integer"
