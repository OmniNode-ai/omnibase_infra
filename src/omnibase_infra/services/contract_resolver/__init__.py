# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Contract Resolver Bridge FastAPI service.

Exposes NodeContractResolveCompute (OMN-2754) via synchronous HTTP for
dashboard consumption at port :8091.

Routes:
    POST /api/nodes/contract.resolve — Resolve a contract with patches
    GET  /health                      — Liveness check

Ticket: OMN-2756
"""

from omnibase_infra.services.contract_resolver.main import create_app

__all__ = ["create_app"]
