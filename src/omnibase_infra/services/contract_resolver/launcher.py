# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Launcher module for the Contract Resolver Bridge service.

Creates the FastAPI application using a minimal ONEX container so the service
can be started via uvicorn CLI:

    uvicorn omnibase_infra.services.contract_resolver.launcher:app \\
        --host 0.0.0.0 --port 8091

Or via Python:

    python -m omnibase_infra.services.contract_resolver.launcher

Environment variables:
    CORS_ORIGINS          Comma-separated list of allowed origins (required)
    CONTRACT_RESOLVER_PORT  Uvicorn port override (default: 8091)
    CONTRACT_RESOLVER_HOST  Uvicorn host override (default: 0.0.0.0)
    ONEX_LOG_LEVEL        Log level (default: INFO)

Ticket: OMN-2756
"""

from __future__ import annotations

import logging
import os

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.services.contract_resolver.main import create_app

logger = logging.getLogger(__name__)

# Create application using minimal container (no DB/Kafka required for core resolve)
_container = ModelONEXContainer()
app = create_app(
    container=_container,
    cors_origins=None,  # Reads CORS_ORIGINS env var; fails fast if not set
)


def main() -> None:
    """Run the Contract Resolver Bridge with uvicorn.

    Reads configuration from environment variables:
    - ``CONTRACT_RESOLVER_PORT``: Listening port (default: 8091)
    - ``CONTRACT_RESOLVER_HOST``: Binding host (default: 0.0.0.0)
    - ``ONEX_LOG_LEVEL``: Log level (default: INFO)
    """
    import uvicorn

    port = int(os.environ.get("CONTRACT_RESOLVER_PORT", "8091"))
    host = os.environ.get("CONTRACT_RESOLVER_HOST", "0.0.0.0")  # noqa: S104 — Docker service binds all interfaces by default
    log_level = os.environ.get("ONEX_LOG_LEVEL", "INFO").lower()

    logger.info(
        "Starting Contract Resolver Bridge",
        extra={"host": host, "port": port, "log_level": log_level},
    )

    uvicorn.run(
        "omnibase_infra.services.contract_resolver.launcher:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=False,
    )


if __name__ == "__main__":
    main()


__all__ = ["app", "main"]
