# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Health server host field for services that bind loopback by default (OMN-21623)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from omnibase_infra.models.network_binding.model_network_binding_settings import (
    load_network_binding_settings,
)


class ModelLoopbackHealthServerBinding(BaseModel):
    """The ``health_check_host`` field of a service whose health server binds loopback.

    The default is the declared loopback address, so an unconfigured service is not
    exposed to the network. A cluster deployment overrides it through the service's
    ``<PREFIX>HEALTH_CHECK_HOST`` environment variable.

    Attributes:
        health_check_host: Host/IP the health check server binds.
    """

    health_check_host: str = Field(
        default=load_network_binding_settings().loopback_address,
        description=(
            "Host/IP for health check server binding. The default is the loopback "
            "address, which restricts access to the local machine. For container or "
            "Kubernetes deployments, override it with the service's "
            "HEALTH_CHECK_HOST environment variable to allow external probe access."
        ),
    )


__all__: list[str] = ["ModelLoopbackHealthServerBinding"]
