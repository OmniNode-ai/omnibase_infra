# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""aiohttp test utilities.

Provides helpers for extracting runtime information from aiohttp servers
during integration tests.
"""

from __future__ import annotations

import aiohttp

from omnibase_infra.services.service_health import ServiceHealth


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
        RuntimeError: If the aiohttp internal attribute chain has changed.

    Warning:
        **Fragile**: This function accesses private aiohttp internals
        (``_site``, ``_server``, ``.sockets``) that may change without
        notice across aiohttp releases. Verified against aiohttp 3.9.0.
        If aiohttp is upgraded and this breaks, inspect the new internal
        layout and update the attribute chain accordingly.
    """
    try:
        # Verified against aiohttp 3.9.0 -- attribute chain:
        #   ServiceHealth._site -> aiohttp.web.TCPSite
        #   TCPSite._server -> asyncio.Server
        #   asyncio.Server.sockets -> list[socket.socket]
        site = health_server._site
        internal_server = site._server  # type: ignore[union-attr]
        sock = next(iter(internal_server.sockets))  # type: ignore[union-attr]
        return sock.getsockname()[1]  # type: ignore[no-any-return]
    except AttributeError as e:
        msg = (
            f"aiohttp internals changed (currently installed: aiohttp "
            f"{aiohttp.__version__}, verified against 3.9.0). "
            f"The private attribute chain (_site._server.sockets) is no longer valid. "
            f"Check the aiohttp changelog for internal API changes and update "
            f"this function accordingly. Original error: {e}"
        )
        raise RuntimeError(msg) from e
