# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""aiohttp test utilities.

Provides helpers for extracting runtime information from aiohttp servers
during integration tests.
"""

from __future__ import annotations

from omnibase_infra.services.health_checker import ServiceHealth


def get_aiohttp_bound_port(health_server: ServiceHealth) -> int:
    """Extract the auto-assigned port from a ServiceHealth server.

    aiohttp does not expose the bound port publicly when using port=0.
    Accessing internals is the only way to discover the auto-assigned port.

    # Verified against aiohttp 3.9.0

    Args:
        health_server: A started ServiceHealth instance using port=0.

    Returns:
        The auto-assigned port number.

    Raises:
        RuntimeError: If the server has not been started.
    """
    site = health_server._site
    if site is None:
        msg = "ServiceHealth has no acceptor; call start() first"
        raise RuntimeError(msg)
    return int(site.port)
