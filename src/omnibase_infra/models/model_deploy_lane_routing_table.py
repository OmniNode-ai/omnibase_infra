# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The deploy lane routing table, the fields the key check reads (OMN-26485)."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.models.model_deploy_lane_instance import ModelDeployLaneInstance
from omnibase_infra.models.model_deploy_lane_route import ModelDeployLaneRoute


class ModelDeployLaneRoutingTable(BaseModel):
    """``config/deploy_lane_routing.yaml`` as parsed YAML validates into it.

    It restates no validation of the deploy agent's own parser
    (``deploy_agent.routing.parse_routing_table``); it carries the three things the
    key check needs: the default instance, each instance's availability and the routes.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    default_instance: str
    instances: dict[str, ModelDeployLaneInstance]
    routes: tuple[ModelDeployLaneRoute, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _names_are_declared(self) -> Self:
        """A default or route that names an undeclared instance, or an unavailable default, refuses."""
        default = self.instances.get(self.default_instance)
        if default is None:
            raise ValueError(
                f"default_instance {self.default_instance!r} is not declared"
            )
        if not default.available:
            raise ValueError(
                f"default_instance {self.default_instance!r} is declared unavailable; "
                "a command must always have an instance"
            )
        for route in self.routes:
            if route.instance not in self.instances:
                raise ValueError(f"route names undeclared instance {route.instance!r}")
        return self
