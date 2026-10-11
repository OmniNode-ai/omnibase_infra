# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A catalog bundle rendered to a compose file, plus its deployment topology.

Ticket: OMN-19972
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.models.core.model_deployment_topology import ModelDeploymentTopology


class ModelRenderedBundle(BaseModel):
    """What ``render_bundle`` wrote, in the shape the local provision node takes.

    ``topology`` is the port bridge's output: one LOCAL entry for every service
    that publishes a host port, so the provision node can poll it. Services
    with no published port (one-shot migrations, seeds, projection runners)
    are listed in ``services`` and reached by running the whole compose
    project, never by the topology.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    bundles: tuple[str, ...] = Field(
        ..., description="Catalog bundles rendered, in the order given."
    )
    project: str = Field(..., description="Compose project name the render declares.")
    compose_file_path: str = Field(
        ..., description="Absolute path of the compose file written."
    )
    env_files: tuple[str, ...] = Field(
        default=(),
        description=(
            "Env files compose interpolates the render from, lowest precedence "
            "first. Paths only; values are never carried."
        ),
    )
    services: tuple[str, ...] = Field(
        default=(), description="Every compose service in the render."
    )
    topology: ModelDeploymentTopology = Field(
        ..., description="Port-publishing services as a LOCAL deployment topology."
    )


__all__: list[str] = ["ModelRenderedBundle"]
