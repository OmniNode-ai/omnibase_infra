# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Health server host field for services that bind every interface by default (OMN-21623)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from omnibase_infra.models.network_binding.model_network_binding_settings import (
    load_network_binding_settings,
)


class ModelBindAllHealthServerBinding(BaseModel):
    """The ``health_check_host`` field of a container service whose health server binds all interfaces.

    The default is the declared bind-all address so the container's health probe can
    reach the server. A deployment overrides it through the service's
    ``<PREFIX>HEALTH_CHECK_HOST`` environment variable.

    Attributes:
        health_check_host: Host/IP the health check server binds.
    """

    health_check_host: str = Field(
        default=load_network_binding_settings().bind_all_address,
        description=(
            "Host/IP for health check server binding. The default is the bind-all "
            "address so the container's health probe can reach the server; override "
            "it with the service's HEALTH_CHECK_HOST environment variable."
        ),
    )


__all__: list[str] = ["ModelBindAllHealthServerBinding"]
